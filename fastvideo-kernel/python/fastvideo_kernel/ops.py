import math
from typing import List, Tuple

import torch

from .block_sparse_attn import block_sparse_attn, block_sparse_attn_from_indices
from .triton_kernels.st_attn_triton import sliding_tile_attention_triton

# Try to load the C++ extension
try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
    sta_fwd = getattr(fastvideo_kernel_ops, "sta_fwd", None)
    sta_bwd = getattr(fastvideo_kernel_ops, "sta_bwd", None)
except ImportError:
    sta_fwd = None
    sta_bwd = None

_SHAPE_MAP = {"30x48x80": 1, "36x48x48": 2, "18x48x80": 3}


def _padded_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, has_text: bool
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad q/k/v to a multiple of 384 along the sequence dim when has_text is set."""
    seq_length = q.shape[2]
    pad_size = 0
    if has_text:
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)
    return q, k, v, pad_size


def _sta_fwd_inference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       window_size: List[Tuple[int, int, int]], text_length: int,
                       has_text: bool, flag: int) -> torch.Tensor:
    """Inference fast path: matches the original head-by-head loop, no autograd state."""
    output = torch.empty_like(q)
    for head_idx, (t, h, w) in enumerate(window_size):
        q_h = q[:, head_idx:head_idx + 1].contiguous()
        k_h = k[:, head_idx:head_idx + 1].contiguous()
        v_h = v[:, head_idx:head_idx + 1].contiguous()
        o_h = torch.empty_like(q_h)
        sta_fwd(q_h, k_h, v_h, o_h, t, h, w, text_length, False, has_text, flag)
        output[:, head_idx:head_idx + 1] = o_h
    if has_text:
        sta_fwd(q.contiguous(), k.contiguous(), v.contiguous(), output,
                3, 3, 3, text_length, True, True, flag)
    return output


def _sta_fwd_train(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   window_size: List[Tuple[int, int, int]], text_length: int,
                   has_text: bool, flag: int
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Training forward: same kernels, but also captures the per-token LSE for backward."""
    batch, num_heads, seq_len, _ = q.shape
    output = torch.empty_like(q)
    l_vec = torch.empty((batch, num_heads, seq_len, 1), dtype=torch.float32, device=q.device)

    for head_idx, (t, h, w) in enumerate(window_size):
        q_h = q[:, head_idx:head_idx + 1].contiguous()
        k_h = k[:, head_idx:head_idx + 1].contiguous()
        v_h = v[:, head_idx:head_idx + 1].contiguous()
        o_h = torch.empty_like(q_h)
        l_h = torch.empty((batch, 1, seq_len, 1), dtype=torch.float32, device=q.device)
        sta_fwd(q_h, k_h, v_h, o_h, t, h, w, text_length, False, has_text, flag, l_h)
        output[:, head_idx:head_idx + 1] = o_h
        l_vec[:, head_idx:head_idx + 1] = l_h

    if has_text:
        l_text = torch.empty_like(l_vec)
        sta_fwd(q.contiguous(), k.contiguous(), v.contiguous(), output,
                3, 3, 3, text_length, True, True, flag, l_text)
        # Text-Q kernel only writes the last 384 token positions (the text+pad region).
        l_vec[:, :, -384:, :] = l_text[:, :, -384:, :]

    return output, l_vec


class _STAFunction(torch.autograd.Function):
    """Autograd binding for sliding_tile_attention.

    Stores (q, k, v, o, lse) per head; backward dispatches the CUDA BWD kernel
    head-by-head. Falls back to head-loop in Python because the FWD itself is
    head-loop-shaped (per-head window size).
    """

    @staticmethod
    def forward(ctx, q, k, v, window_size, text_length, has_text, flag, pad_size, original_seq_len):
        output, l_vec = _sta_fwd_train(q, k, v, window_size, text_length, has_text, flag)
        ctx.save_for_backward(q, k, v, output, l_vec)
        ctx.window_size = tuple(window_size)
        ctx.text_length = text_length
        ctx.has_text = has_text
        ctx.flag = flag
        ctx.pad_size = pad_size
        ctx.original_seq_len = original_seq_len
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, l_vec = ctx.saved_tensors
        window_size = ctx.window_size
        text_length = ctx.text_length
        has_text = ctx.has_text
        flag = ctx.flag

        if grad_output.shape != o.shape:
            # The forward returns the (possibly padded) output; the user typically
            # slices to the original seq_len before computing loss. Re-pad gradient.
            grad_padded = torch.zeros_like(o)
            grad_padded[:, :, :grad_output.shape[2]] = grad_output
            grad_output = grad_padded

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        for head_idx, (t, h, w) in enumerate(window_size):
            q_h = q[:, head_idx:head_idx + 1].contiguous()
            k_h = k[:, head_idx:head_idx + 1].contiguous()
            v_h = v[:, head_idx:head_idx + 1].contiguous()
            o_h = o[:, head_idx:head_idx + 1].contiguous()
            l_h = l_vec[:, head_idx:head_idx + 1].contiguous()
            do_h = grad_output[:, head_idx:head_idx + 1].contiguous()
            dq_h, dk_h, dv_h = sta_bwd(
                q_h, k_h, v_h, o_h, l_h, do_h,
                t, h, w, text_length, has_text, flag,
            )
            dq[:, head_idx:head_idx + 1] = dq_h
            dk[:, head_idx:head_idx + 1] = dk_h
            dv[:, head_idx:head_idx + 1] = dv_h

        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                None, None, None, None, None, None)


def sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: list,
    text_length: int,
    has_text: bool = True,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    """Sliding-tile attention. Routes to the CUDA autograd path when any input requires grad."""
    if sta_fwd is None:
        return sliding_tile_attention_triton(
            q, k, v, window_size, text_length, has_text, seq_shape
        )

    flag = _SHAPE_MAP[seq_shape]
    seq_length = q.shape[2]

    q_p, k_p, v_p, pad_size = _padded_inputs(q, k, v, has_text)

    requires_grad = (q.requires_grad or k.requires_grad or v.requires_grad) and torch.is_grad_enabled()

    if requires_grad:
        if sta_bwd is None:
            raise RuntimeError(
                "sliding_tile_attention received tensors requiring grad but the CUDA "
                "extension was built without sta_bwd. Rebuild the extension or run with "
                "torch.no_grad() for inference."
            )
        output = _STAFunction.apply(q_p, k_p, v_p, window_size, text_length, has_text,
                                    flag, pad_size, seq_length)
    else:
        output = _sta_fwd_inference(q_p, k_p, v_p, window_size, text_length, has_text, flag)

    return output[:, :, :seq_length]


def video_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_variable_block_sizes: torch.Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: torch.Tensor = None,
) -> torch.Tensor:
    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]
    if v.shape[2] != kv_seq_len:
        raise ValueError(
            f"Expected k and v to have the same sequence length, got "
            f"k.shape[2]={kv_seq_len}, v.shape[2]={v.shape[2]}"
        )
    if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != heads or v.shape[1] != heads:
        raise ValueError("Expected q/k/v to have the same batch and head dimensions.")

    if q_seq_len % block_elements != 0 or kv_seq_len % block_elements != 0:
        raise ValueError(
            f"q_seq_len and kv_seq_len must be divisible by block_elements={block_elements}, "
            f"got q_seq_len={q_seq_len}, kv_seq_len={kv_seq_len}"
        )
    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = kv_seq_len // block_elements

    if variable_block_sizes.numel() != kv_num_blocks:
        raise ValueError(
            f"variable_block_sizes must have length kv_num_blocks={kv_num_blocks}, "
            f"got {variable_block_sizes.numel()}"
        )

    if q_variable_block_sizes.numel() != q_num_blocks:
        raise ValueError(
            f"q_variable_block_sizes must have length q_num_blocks={q_num_blocks}, "
            f"got {q_variable_block_sizes.numel()}"
        )

    # Compression branch
    q_c = q.view(batch, heads, q_num_blocks, block_elements, dim)
    k_c = k.view(batch, heads, kv_num_blocks, block_elements, dim)
    v_c = v.view(batch, heads, kv_num_blocks, block_elements, dim)

    q_c = (q_c.float().sum(dim=3) / q_variable_block_sizes.view(1, 1, -1, 1)).to(
        q.dtype)
    k_c = (k_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        k.dtype)
    v_c = (v_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)

    out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements,
                         1).view(batch, heads, q_seq_len, dim)

    # Sparse branch: feed top-k indices directly, skipping the bool-mask round-trip.
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    q2k_idx = topk_idx.to(torch.int32).contiguous()
    q2k_num = torch.full(
        (batch, heads, q_num_blocks),
        topk,
        dtype=torch.int32,
        device=q.device,
    )
    out_s = block_sparse_attn_from_indices(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )[0]

    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s
    return out_c + out_s
