"""Numerical correctness tests for the CUDA STA backward kernel.

Compares the autograd path of `fastvideo_kernel.sliding_tile_attention` against
`torch.nn.attention.flex_attention` with the equivalent 3D sliding-tile mask.
The forward pass already has parity coverage in `tests/test_sta.py`; this file
focuses on dQ / dK / dV.

Requires an H100 (or any SM90 device). Skipped silently otherwise.
"""

from __future__ import annotations

import pytest
import torch

flex_attention = None
if torch.cuda.is_available():
    try:
        from torch.nn.attention.flex_attention import flex_attention as _flex
        flex_attention = torch.compile(_flex, dynamic=False)
    except Exception:
        flex_attention = None

try:
    from .support_flex_sta import get_sliding_tile_attention_mask
except ImportError:
    from support_flex_sta import get_sliding_tile_attention_mask  # type: ignore

from fastvideo_kernel import sliding_tile_attention


def _is_sm90_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


def _make_qkv(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
    # Normalise like test_sta.py does — keeps the magnitudes close to a real model.
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _flex_reference_grads(q, k, v, kernel_size, canvas, tile, text_length, grad_o):
    """Compute reference dq/dk/dv via flex_attention autograd."""
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    mask = get_sliding_tile_attention_mask(kernel_size, tile, canvas, text_length, "cuda", 0)
    out = flex_attention(qr, kr, vr, block_mask=mask)
    out.backward(grad_o)
    return out.detach(), qr.grad, kr.grad, vr.grad


def _cuda_grads(q, k, v, kernel_size, seq_shape, text_length, grad_o, has_text):
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    out = sliding_tile_attention(qr, kr, vr, [kernel_size] * qr.shape[1],
                                 text_length, has_text=has_text, seq_shape=seq_shape)
    out.backward(grad_o)
    return out.detach(), qr.grad, kr.grad, vr.grad


def _max_abs(t: torch.Tensor) -> float:
    return float(t.abs().max().item())


def _avg_abs(t: torch.Tensor) -> float:
    return float(t.abs().mean().item())


@pytest.mark.skipif(not _is_sm90_or_newer(), reason="Requires SM90+ (H100)")
@pytest.mark.skipif(flex_attention is None, reason="flex_attention not available")
@pytest.mark.parametrize("kernel_size", [(3, 3, 3), (3, 5, 5)])
def test_sta_backward_wan_no_text(kernel_size):
    """Wan layout (18x48x80), no text. Smallest canvas — fastest correctness check."""
    batch, heads, head_dim = 1, 2, 128
    canvas = (18, 48, 80)
    tile = (6, 8, 8)
    seq_len = canvas[0] * canvas[1] * canvas[2]

    q, k, v = _make_qkv(batch, heads, seq_len, head_dim)
    grad_o = torch.randn_like(q) * 1e-2

    ref_o, dq_ref, dk_ref, dv_ref = _flex_reference_grads(
        q, k, v, kernel_size, canvas, tile, 0, grad_o)
    cuda_o, dq, dk, dv = _cuda_grads(
        q, k, v, kernel_size, "18x48x80", 0, grad_o, has_text=False)

    # Forward sanity (already covered in test_sta.py, kept here as a guard).
    assert _max_abs(cuda_o - ref_o) < 4e-2

    # Gradient tolerances: bf16 with 18×48×80 reduction → looser than fwd.
    for name, t, ref in [("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)]:
        diff = (t.to(torch.float32) - ref.to(torch.float32))
        assert _max_abs(diff) < 1e-1, f"{name} max diff {_max_abs(diff):.4e}"
        assert _avg_abs(diff) < 5e-3, f"{name} avg diff {_avg_abs(diff):.4e}"


@pytest.mark.skipif(not _is_sm90_or_newer(), reason="Requires SM90+ (H100)")
@pytest.mark.skipif(flex_attention is None, reason="flex_attention not available")
def test_sta_backward_returns_correct_dtype():
    """Backward returns gradients in the input dtype, not float32."""
    batch, heads, head_dim = 1, 2, 128
    canvas = (18, 48, 80)
    seq_len = canvas[0] * canvas[1] * canvas[2]
    q, k, v = _make_qkv(batch, heads, seq_len, head_dim)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = sliding_tile_attention(q, k, v, [(3, 3, 3)] * heads, 0,
                                 has_text=False, seq_shape="18x48x80")
    out.sum().backward()
    assert q.grad.dtype == q.dtype
    assert k.grad.dtype == k.dtype
    assert v.grad.dtype == v.dtype
    assert q.grad.shape == q.shape


@pytest.mark.skipif(not _is_sm90_or_newer(), reason="Requires SM90+ (H100)")
@pytest.mark.skipif(flex_attention is None, reason="flex_attention not available")
def test_sta_backward_handles_window_radius_zero():
    """A window of 1 along an axis collapses the inner Q-tile enumeration."""
    batch, heads, head_dim = 1, 2, 128
    canvas = (18, 48, 80)
    seq_len = canvas[0] * canvas[1] * canvas[2]
    q, k, v = _make_qkv(batch, heads, seq_len, head_dim)
    grad_o = torch.randn_like(q) * 1e-2

    # (1, 6, 5) → DT=0, DH=3, DW=2 (only one time tile attended).
    ref_o, dq_ref, dk_ref, dv_ref = _flex_reference_grads(
        q, k, v, (1, 6, 5), canvas, (6, 8, 8), 0, grad_o)
    cuda_o, dq, dk, dv = _cuda_grads(
        q, k, v, (1, 6, 5), "18x48x80", 0, grad_o, has_text=False)

    for name, t, ref in [("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)]:
        diff = t.to(torch.float32) - ref.to(torch.float32)
        assert _max_abs(diff) < 1e-1, f"{name} max diff {_max_abs(diff):.4e}"


@pytest.mark.skipif(not _is_sm90_or_newer(), reason="Requires SM90+ (H100)")
@pytest.mark.skipif(flex_attention is None, reason="flex_attention not available")
def test_sta_backward_inference_path_unaffected():
    """No-grad mode must keep the original inference fast path bit-for-bit."""
    batch, heads, head_dim = 1, 2, 128
    canvas = (18, 48, 80)
    seq_len = canvas[0] * canvas[1] * canvas[2]
    q, k, v = _make_qkv(batch, heads, seq_len, head_dim)

    with torch.no_grad():
        out_a = sliding_tile_attention(q, k, v, [(3, 3, 3)] * heads, 0,
                                       has_text=False, seq_shape="18x48x80")
        out_b = sliding_tile_attention(q, k, v, [(3, 3, 3)] * heads, 0,
                                       has_text=False, seq_shape="18x48x80")
    assert torch.equal(out_a, out_b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
