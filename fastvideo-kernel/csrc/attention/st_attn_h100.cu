// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>
#include <c10/cuda/CUDAGuard.h>

// #define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
__device__ __forceinline__ int clamp_int(int value, int min, int max) {
    return (value < min) ? min : ((value > max) ? max : value);
}
// #define ABS(x) ((x) < 0 ? -(x) : (x))
__device__ __forceinline__ int abs_int(int value) {
    return (value < 0) ? -value : value;
}


constexpr int CONSUMER_WARPGROUPS = (3); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (4); 
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (2); 
};

template<int D> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N; 
    const int text_L;
    const int hr;
};


template<int D, bool is_causal, bool text_q, bool text_kv, int DT, int DH, int DW, int CT, int CH, int CW>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);
    int img_kv_blocks;
    int kv_blocks   = g.N / (K::kv_height);
    if constexpr (text_kv) {
        img_kv_blocks = kv_blocks - 3;
    } else {
        img_kv_blocks = kv_blocks;
    }
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx;
    if constexpr (text_q) {
        seq_idx = CT * CH * CW * 6.0 + blockIdx.x * CONSUMER_WARPGROUPS;
    } else {
        seq_idx = blockIdx.x * CONSUMER_WARPGROUPS; 
    }
    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0); 
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        if constexpr (text_q){
            for (int j = 0; j < K::stages - 1; j++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
                tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
                tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
                tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
                tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
            }
        } else {
            int qt = seq_idx / 6 / (CH * CW);
            int qh = (seq_idx / 6) % (CH * CW) / CW;
            int qw = (seq_idx / 6) % CW;
            qt = clamp_int(qt, DT, CT-DT-1);
            qh = clamp_int(qh, DH, CH-DH-1);
            qw = clamp_int(qw, DW, CW-DW-1);
            int count = 0;
            int j = 0;
            while (count < K::stages - 1) {
                int kt = j / 3 / (CH * CW);
                int kh = (j / 3) % (CH * CW) / CW;
                int kw = (j / 3) % CW;
                bool mask = (abs_int(qt - kt) <= DT) && (abs_int(qh - kh) <= DH) && (abs_int(qw - kw) <= DW);
                if (mask){
                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
                    tma::expect_bytes(k_smem_arrived[count], sizeof(k_tile));
                    tma::load_async(k_smem[count], g.k, kv_tile_idx, k_smem_arrived[count]);
                    tma::expect_bytes(v_smem_arrived[count], sizeof(v_tile));
                    tma::load_async(v_smem[count], g.v, kv_tile_idx, v_smem_arrived[count]);
                    count += 1;
                }
                j += 1;
            }
        }
    }
    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)); 
            kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        }
        else { kv_iters = kv_blocks-2;}

        if(warpid == NUM_WORKERS-4) {
            if constexpr (text_q){
                for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                    tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                    tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                    tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                    tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                    kittens::wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
                }
            } else {
                int qt = seq_idx / 6 / (CH * CW);
                int qh = (seq_idx / 6) % (CH * CW) / CW;
                int qw = (seq_idx / 6) % CW;
                qt = clamp_int(qt, DT, CT-DT-1);
                qh = clamp_int(qh, DH, CH-DH-1);
                qw = clamp_int(qw, DW, CW-DW-1);
                int k_t_min = clamp_int(qt-DT, 0, CT-1);
                int k_t_max = clamp_int(qt+DT, 0, CT-1);
                int k_h_min = clamp_int(qh-DH, 0, CH-1);
                int k_h_max = clamp_int(qh+DH, 0, CH-1);
                int k_w_min = clamp_int(qw-DW, 0, CW-1);
                int k_w_max = clamp_int(qw+DW, 0, CW-1);
                int count = 0;
                for (int kt = k_t_min; kt <= k_t_max; kt++) {
                    for (int kh = k_h_min; kh <= k_h_max; kh++) {
                        for (int kw = k_w_min; kw <= k_w_max; kw++) {
                            for (int j = 0; j <= 2; j++){
                                if (count >= K::stages - 1) {
                                    int index = ((kt * (CH * CW)) + (kh * CW) + kw) * 3 + j;
                                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, index, 0};
                                    tma::expect_bytes(k_smem_arrived[count%K::stages], sizeof(k_tile));
                                    tma::load_async(k_smem[count%K::stages], g.k, kv_tile_idx, k_smem_arrived[count%K::stages]);
                                    tma::expect_bytes(v_smem_arrived[count%K::stages], sizeof(v_tile));
                                    tma::load_async(v_smem[count%K::stages], g.v, kv_tile_idx, v_smem_arrived[count%K::stages]);
                                    kittens::wait(compute_done[(count - 1)%K::stages], ((count - 1)/K::stages)%2);
                                    count += 1;
                                } else {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                // for text 
                for (int index = img_kv_blocks; index < kv_blocks; index++) {
                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, index, 0};
                    tma::expect_bytes(k_smem_arrived[count%K::stages], sizeof(k_tile));
                    tma::load_async(k_smem[count%K::stages], g.k, kv_tile_idx, k_smem_arrived[count%K::stages]);
                    tma::expect_bytes(v_smem_arrived[count%K::stages], sizeof(v_tile));
                    tma::load_async(v_smem[count%K::stages], g.v, kv_tile_idx, v_smem_arrived[count%K::stages]);
                    kittens::wait(compute_done[(count - 1)%K::stages], ((count - 1)/K::stages)%2);
                    count += 1;
                }
            }


        }
    }
    else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters/8);
        }
        else if constexpr (text_q){ 
            // the last three kv blocks are for text, we process them separately
            kv_iters = img_kv_blocks - 1;
        } else {
            kv_iters = clamp_int(DT*2+1, 1, CT) * clamp_int(DH*2+1, 1, CH) * clamp_int(DW*2+1, 1, CW) * 3 - 1 ; 
        }

        kittens::wait(qsmem_semaphore, 0);
        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {

            kittens::wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            
            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
            
            warpgroup::mma_async_wait();

            row_max(max_vec, att_block, max_vec);
            
            if constexpr (D == 64) { 
                mul(att_block, att_block,    1.44269504089f*0.125f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   { 
                mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            add(att_block, att_block, 0.f);
            copy(att_block_mma, att_block); 
            mul_row(o_reg, o_reg, max_vec_last_scaled); 

            kittens::wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();

            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }
        // the last three kv blocks are for text, we process them separately
        if constexpr(text_kv) {
            for (auto kv_idx = kv_iters + 1; kv_idx <= kv_iters + 3; kv_idx++) {

                kittens::wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
                warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
                
                copy(max_vec_last_scaled, max_vec);
                if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
                else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
                
                warpgroup::mma_async_wait();
                // apply non-pad mask
                int offset = g.text_L - (kv_idx - (kv_iters + 1)) * K::kv_height;
                // printf("k_idx_start: %d, k_idx_end: %d, text_end: %d, offset: %d\n", k_idx_start, k_idx_end, text_end, offset);
                right_fill(att_block, att_block, offset, base_types::constants<float>::neg_infty());


                row_max(max_vec, att_block, max_vec);
                
                if constexpr (D == 64) { 
                    mul(att_block, att_block,    1.44269504089f*0.125f); 
                    mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
                }
                else                   { 
                    mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                    mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
                }

                sub_row(att_block, att_block, max_vec_scaled);
                exp2(att_block, att_block);
                sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                exp2(max_vec_last_scaled,       max_vec_last_scaled);
                mul(norm_vec,            norm_vec,     max_vec_last_scaled);
                row_sum(norm_vec,  att_block, norm_vec);
                add(att_block, att_block, 0.f);
                copy(att_block_mma, att_block); 
                mul_row(o_reg, o_reg, max_vec_last_scaled); 

                kittens::wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

                warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
                warpgroup::mma_async_wait();

                if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
            }
        }

        div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg); 
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec_scaled);

        if constexpr (D == 64) { mul(norm_vec, norm_vec, -8.0f); }
        else                   { mul(norm_vec, norm_vec, -11.313708499f); }
    
        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, (seq_idx) + warpgroupid};
            tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        tma::store_async_wait();
    }
}


// =====================================================================
// Backward pass.
//
// Mirrors the block-sparse BWD recipe (preprocess kernel that materialises
// D = rowsum(O * dO), then a per-KV-tile main kernel that streams the
// Q tiles which attend to the current KV tile and accumulates dQ/dK/dV).
//
// The only STA-specific bit is the inner enumeration of Q tiles that
// fall in the 3D sliding-tile window of the current KV tile, plus the
// usual image-vs-text split when `has_text_kv` / `has_text_q` are set.
// =====================================================================

template<int D> struct bwd_prep_globals {
    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;
};

constexpr int BWD_PREP_NUM_WARPS = 1;

template<int D>
__global__ __launch_bounds__(BWD_PREP_NUM_WARPS*kittens::WARP_THREADS, (D == 64) ? 6 : 3)
void bwd_attend_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid();

    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    og_tile (&og_smem)[BWD_PREP_NUM_WARPS] = al.allocate<og_tile, BWD_PREP_NUM_WARPS>();
    o_tile  (&o_smem) [BWD_PREP_NUM_WARPS] = al.allocate<o_tile , BWD_PREP_NUM_WARPS>();
    d_tile  (&d_smem) [BWD_PREP_NUM_WARPS] = al.allocate<d_tile , BWD_PREP_NUM_WARPS>();

    rt_fl<4*16, D> og_reg, o_reg;
    col_vec<rt_fl<4*16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;
    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * BWD_PREP_NUM_WARPS * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < BWD_PREP_NUM_WARPS; w++) {
            coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y, (blockIdx.x * BWD_PREP_NUM_WARPS) + w, 0};
            tma::load_async(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }

    wait(smem_semaphore, 0);
    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);
    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    store(d_smem[warpid], d_reg);
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < BWD_PREP_NUM_WARPS; w++) {
            coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0, (blockIdx.x * BWD_PREP_NUM_WARPS) + w};
            tma::store_async(g.d, d_smem[w], tile_idx);
        }
    }
    tma::store_async_wait();
}

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int tile_h     = 4*16;
    constexpr static int tile_h_qo  = 4*16;
};

template<int D>
struct bwd_globals {
    using G = bwd_attend_ker_tile_dims<D>;

    using q_tile  =         st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using v_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using og_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;
    using l_gl  = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    q_gl  q;
    k_gl  k;
    v_gl  v;
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;
    l_gl  l;
    d_gl  d;

    const int N;
    const int text_L;
    const int hr;
};

// Load a row-vec held in smem into a register tile by broadcasting along rows.
__device__ static inline void
bwd_stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
    }
}

// In-place subtract the smem row-vec from each row of a register tile.
__device__ static inline void
bwd_stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(laneid()%4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
    }
}

// Q/KV tile size in BWD (64 for both) ⇒ 6 tiles per 384-token spatial cell and 6 text tiles.
constexpr int BWD_Q_TILES_PER_CELL  = 6;
constexpr int BWD_KV_TILES_PER_CELL = 6;
constexpr int BWD_TEXT_Q_TILES      = 6;

template<int D, bool has_text_kv, bool has_text_q,
         int DT, int DH, int DW,
         int CT, int CH, int CW>
__global__ __launch_bounds__(128, (D == 64) ? 3 : 2)
void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using G = bwd_attend_ker_tile_dims<D>;

    using kg_tile   = st_fl<G::tile_h, G::tile_width>;
    using vg_tile   = st_fl<G::tile_h, G::tile_width>;
    using k_tile    = st_bf<G::tile_h, G::tile_width>;
    using v_tile    = st_bf<G::tile_h, G::tile_width>;
    using q_tile    = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile   = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile   = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;

    k_tile  (&k_smem) [1] = al.allocate<k_tile, 1>();
    v_tile  (&v_smem) [1] = al.allocate<v_tile, 1>();
    q_tile  (&q_smem) [1] = al.allocate<q_tile, 1>();
    og_tile (&og_smem)[1] = al.allocate<og_tile, 1>();
    qg_tile (&qg_smem)    = al.allocate<qg_tile>();
    l_tile  (&l_smem) [1] = al.allocate<l_tile, 1>();
    d_tile  (&d_smem) [1] = al.allocate<d_tile, 1>();
    attn_tile (&ds_smem_t)[1] = al.allocate<attn_tile, 1>();

    kg_tile (*kg_smem) = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);
    vg_tile (*vg_smem) = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]);

    const int kv_head_idx = blockIdx.y / g.hr;

    // Image and text region boundaries, in units of KV/Q tiles.
    constexpr int img_kv_blocks = CT * CH * CW * BWD_KV_TILES_PER_CELL;
    constexpr int img_q_blocks  = CT * CH * CW * BWD_Q_TILES_PER_CELL;

    const int kv_block = blockIdx.x;
    const bool kv_is_text = has_text_kv && (kv_block >= img_kv_blocks);

    // Build the list of Q tiles that attend to this KV tile in shared memory.
    // Bound: (2*DT+1)*(2*DH+1)*(2*DW+1)*Q_TILES_PER_CELL + (text ? 6 : 0).
    // Worst-case (Hunyuan, DT=2/DH=3/DW=5) ⇒ 5*7*11*6 + 6 = 2316 ints = ~9KB, fits in smem.
    constexpr int MAX_WINDOW_CELLS = (2*DT+1) * (2*DH+1) * (2*DW+1);
    constexpr int MAX_Q_TILES =
        (MAX_WINDOW_CELLS * BWD_Q_TILES_PER_CELL)
        + (has_text_q ? BWD_TEXT_Q_TILES : 0);

    __shared__ int q_tile_list[MAX_Q_TILES];
    __shared__ int q_tile_count_smem;

    if (threadIdx.x == 0) {
        int count = 0;
        if (kv_is_text) {
            // Text KV is attended to by all image Q tiles.
            for (int q = 0; q < img_q_blocks; q++) q_tile_list[count++] = q;
        } else {
            const int kv_cell = kv_block / BWD_KV_TILES_PER_CELL;
            const int kt = kv_cell / (CH * CW);
            const int kh = (kv_cell % (CH * CW)) / CW;
            const int kw = kv_cell % CW;
            // FWD applies clamp_int(qt, DT, CT-DT-1) before the window check, so any
            // Q in [0, DT-1] (resp. [CT-DT, CT-1]) effectively queries with qt=DT
            // (resp. CT-DT-1). Walk every Q cell whose effective coord lies in the
            // window of (kt, kh, kw).
            for (int qt = 0; qt < CT; qt++) {
                const int qt_eff = clamp_int(qt, DT, CT - DT - 1);
                if (abs_int(qt_eff - kt) > DT) continue;
                for (int qh = 0; qh < CH; qh++) {
                    const int qh_eff = clamp_int(qh, DH, CH - DH - 1);
                    if (abs_int(qh_eff - kh) > DH) continue;
                    for (int qw = 0; qw < CW; qw++) {
                        const int qw_eff = clamp_int(qw, DW, CW - DW - 1);
                        if (abs_int(qw_eff - kw) > DW) continue;
                        const int q_cell = (qt * CH * CW) + (qh * CW) + qw;
                        #pragma unroll
                        for (int j = 0; j < BWD_Q_TILES_PER_CELL; j++) {
                            q_tile_list[count++] = q_cell * BWD_Q_TILES_PER_CELL + j;
                        }
                    }
                }
            }
        }
        if constexpr (has_text_q) {
            for (int j = 0; j < BWD_TEXT_Q_TILES; j++) {
                q_tile_list[count++] = img_q_blocks + j;
            }
        }
        q_tile_count_smem = count;
    }
    __syncthreads();
    const int q_tile_count = q_tile_count_smem;

    if (q_tile_count == 0) return;

    __shared__ kittens::semaphore kv_b, q_b[1], o_b[1], vec_b[1];

    int store_qg_block_index;
    int load_q_block_index;

    if (threadIdx.x == 0) {
        load_q_block_index = q_tile_list[0];

        init_semaphore(kv_b,     0, 1);
        init_semaphore(q_b[0],   0, 1);
        init_semaphore(o_b[0],   0, 1);
        init_semaphore(vec_b[0], 0, 1);

        tma::expect_bytes(kv_b, sizeof(k_smem[0]) + sizeof(v_smem[0]));
        coord<k_tile> tile_idx_kv = {blockIdx.z, kv_head_idx, kv_block, 0};
        tma::load_async(k_smem[0], g.k, tile_idx_kv, kv_b);
        tma::load_async(v_smem[0], g.v, tile_idx_kv, kv_b);

        coord<q_tile> tile_idx_qo = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
        coord<l_tile> vec_idx     = {blockIdx.z, blockIdx.y, 0, load_q_block_index};

        tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
        tma::load_async(og_smem[0], g.og, tile_idx_qo, o_b[0]);

        tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
        tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
        tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);

        tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
        tma::load_async(q_smem[0], g.q, tile_idx_qo, q_b[0]);
    }
    __syncthreads();

    rt_fl<16, G::tile_width> kg_reg, vg_reg;
    rt_fl<16, 64> s_block_t,  p_block_t;
    rt_fl<16, 64> ds_block_t, dp_block_t;
    rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

    zero(kg_reg);
    zero(vg_reg);

    wait(kv_b, 0);

    for (int q_idx = 0; q_idx < q_tile_count; q_idx++) {
        const bool is_last = (q_idx == q_tile_count - 1);

        store_qg_block_index = load_q_block_index;
        if (!is_last) load_q_block_index = q_tile_list[q_idx + 1];

        wait(o_b[0], q_idx % 2);
        warpgroup::mm_ABt(dp_block_t, v_smem[0], og_smem[0]);
        warpgroup::mma_commit_group();

        wait(vec_b[0], q_idx % 2);
        bwd_stream_tile(s_block_t, l_smem, 0);
        wait(q_b[0], q_idx % 2);
        warpgroup::mma_ABt(s_block_t, k_smem[0], q_smem[0]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();

        if constexpr (D == 64) { mul(s_block_t, s_block_t, 1.44269504089f * 0.125f); }
        else                   { mul(s_block_t, s_block_t, 1.44269504089f * 0.08838834764f); }

        // Mask out padded text positions: s_block_t is K-major (rows = K positions),
        // so we use lower_fill on the K-row dimension. Per-warp local-row threshold,
        // matching the block-sparse BWD's masking idiom.
        if constexpr (has_text_kv) {
            if (kv_is_text) {
                const int text_kv_local = kv_block - img_kv_blocks;
                const int fill_start = g.text_L - text_kv_local * G::tile_h
                                       - 16 * kittens::warpid();
                lower_fill(s_block_t, s_block_t, fill_start, base_types::constants<float>::neg_infty());
            }
        }

        exp2(s_block_t, s_block_t);
        copy(p_block_t, s_block_t);
        copy(p_block_t_mma, s_block_t);
        bwd_stream_sub_tile(dp_block_t, d_smem, 0);
        mul(ds_block_t, p_block_t, dp_block_t);

        if constexpr (D == 64) { mul(ds_block_t, ds_block_t, 0.125f); }
        else                   { mul(ds_block_t, ds_block_t, 0.08838834764f); }

        if (!is_last) {
            if (threadIdx.x == 0) {
                coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, load_q_block_index};
                tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
                tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);
            }
        }

        warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[0]);
        warpgroup::mma_commit_group();
        copy(ds_block_t_mma, ds_block_t);
        warpgroup::store(ds_smem_t[0], ds_block_t);
        warpgroup::mma_async_wait();

        if (!is_last) {
            if (threadIdx.x == 0) {
                coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
                tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
                tma::load_async(og_smem[0], g.og, tile_idx, o_b[0]);
            }
        }

        warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[0]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();

        if (!is_last) {
            if (threadIdx.x == 0) {
                coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
                tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
                tma::load_async(q_smem[0], g.q, q_tile_idx, q_b[0]);
            }
        }

        rt_fl<16, G::tile_width> qg_reg;
        __syncthreads();
        warpgroup::mm_AtB(qg_reg, ds_smem_t[0], k_smem[0]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();
        warpgroup::store(qg_smem, qg_reg);
        __syncthreads();

        if (threadIdx.x / 32 == 0) {
            coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, store_qg_block_index, 0};
            tma::store_add_async(g.qg, qg_smem, tile_idx);
            tma::store_async_wait();
        }
    }

    __syncthreads();
    warpgroup::store(kg_smem[0], kg_reg);
    __syncthreads();
    if (threadIdx.x / 32 == 0) {
        coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx, kv_block, 0};
        tma::store_add_async(g.kg, kg_smem[0], tile_idx);
        tma::store_commit_group();
    }

    warpgroup::store(vg_smem[0], vg_reg);
    __syncthreads();
    if (kittens::warpid() % 4 == 0) {
        coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx, kv_block, 0};
        tma::store_add_async(g.vg, vg_smem[0], tile_idx);
        tma::store_commit_group();
    }
    tma::store_async_wait();
}


#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

torch::Tensor
sta_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, int kernel_t_size, int kernel_h_size, int kernel_w_size, int text_length, bool process_text, bool has_text, int kernel_aspect_ratio_flag, std::optional<torch::Tensor> l_out)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2); 
    auto head_dim = q.size(3);  
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seq_len, "V sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");  

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);
    

    
    torch::Tensor l_vec = l_out.has_value()
        ? l_out.value()
        : torch::empty({static_cast<const uint>(batch),
                        static_cast<const uint>(qo_heads),
                        static_cast<const uint>(seq_len),
                        static_cast<const uint>(1)},
                       torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));


    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    //cudadevicesynchronize();
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream(); 


    if (head_dim == 128) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<128>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len),  static_cast<int>(text_length), static_cast<int>(hr)};

        // Shared memory size for the kernel.
        // We use the maximum available shared memory (kittens::MAX_SHARED_MEMORY) 
        // which is approximately 227KB on H100, necessary for the high-performance 
        // TMA-based attention tiles with multiple stages.
        constexpr int mem_size = kittens::MAX_SHARED_MEMORY;
        int threads = NUM_WORKERS * kittens::WARP_THREADS;
        if (has_text) {
            // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
            dim3 grid_image(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4)-2, qo_heads, batch);
            dim3 grid_text(2, qo_heads, batch);
            if (!process_text) {
#define LAUNCH_IMAGE_KER(DT_VAL, DH_VAL, DW_VAL) \
                    cudaFuncSetAttribute( \
                        fwd_attend_ker<128, false, false, true, DT_VAL, DH_VAL, DW_VAL, 5, 6, 10>, \
                        cudaFuncAttributeMaxDynamicSharedMemorySize, \
                        mem_size \
                    ); \
                    fwd_attend_ker<128, false, false, true, DT_VAL, DH_VAL, DW_VAL, 5, 6, 10><<<grid_image, (32*NUM_WORKERS), mem_size, stream>>>(g);

                if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 3)      { LAUNCH_IMAGE_KER(1, 1, 1); }
                else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 5) { LAUNCH_IMAGE_KER(1, 1, 2); }
                else if (kernel_t_size == 5 && kernel_h_size == 3 && kernel_w_size == 3) { LAUNCH_IMAGE_KER(2, 1, 1); }
                else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 5) { LAUNCH_IMAGE_KER(1, 2, 2); }
                else if (kernel_t_size == 5 && kernel_h_size == 6 && kernel_w_size == 1) { LAUNCH_IMAGE_KER(2, 3, 0); }
                else if (kernel_t_size == 5 && kernel_h_size == 3 && kernel_w_size == 5) { LAUNCH_IMAGE_KER(2, 1, 2); }
                else if (kernel_t_size == 5 && kernel_h_size == 5 && kernel_w_size == 5) { LAUNCH_IMAGE_KER(2, 2, 2); }
                else if (kernel_t_size == 5 && kernel_h_size == 5 && kernel_w_size == 7) { LAUNCH_IMAGE_KER(2, 2, 3); }
                else if (kernel_t_size == 5 && kernel_h_size == 6 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(2, 3, 5); }
                else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(1, 3, 5); }
                else if (kernel_t_size == 5 && kernel_h_size == 1 && kernel_w_size == 1) { LAUNCH_IMAGE_KER(2, 0, 0); }
                else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(0, 3, 5); }
                else if (kernel_t_size == 5 && kernel_h_size == 1 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(2, 0, 5); }
                else {
                    TORCH_CHECK(false, "Invalid kernel size: ", kernel_t_size, "x", kernel_h_size, "x", kernel_w_size);
                }
#undef LAUNCH_IMAGE_KER
            } else {
                cudaFuncSetAttribute(
                    fwd_attend_ker<128, false, true, true, 1, 1, 1, 5, 6, 10>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    mem_size
                );
                fwd_attend_ker<128, false, true, true, 1, 1, 1, 5, 6, 10><<<grid_text, (32*NUM_WORKERS), mem_size, stream>>>(g);
            }

        } else {
            dim3 grid_image(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);
            if (kernel_aspect_ratio_flag == 2){
#define LAUNCH_IMAGE_KER(DT_VAL, DH_VAL, DW_VAL) \
                    cudaFuncSetAttribute( \
                        fwd_attend_ker<128, false, false, false, DT_VAL, DH_VAL, DW_VAL, 6, 6, 6>, \
                        cudaFuncAttributeMaxDynamicSharedMemorySize, \
                        mem_size \
                    ); \
                    fwd_attend_ker<128, false, false, false, DT_VAL, DH_VAL, DW_VAL, 6, 6, 6><<<grid_image, (32*NUM_WORKERS), mem_size, stream>>>(g);

                if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 3)      { LAUNCH_IMAGE_KER(1, 1, 1); }
                else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 6) { LAUNCH_IMAGE_KER(1, 1, 3); }
                else if (kernel_t_size == 6 && kernel_h_size == 3 && kernel_w_size == 3) { LAUNCH_IMAGE_KER(3, 1, 1); }
                else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 6) { LAUNCH_IMAGE_KER(1, 3, 3); }
                else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 3) { LAUNCH_IMAGE_KER(1, 3, 1); }
                else if (kernel_t_size == 6 && kernel_h_size == 3 && kernel_w_size == 6) { LAUNCH_IMAGE_KER(3, 1, 3); }
                else if (kernel_t_size == 6 && kernel_h_size == 6 && kernel_w_size == 6) { LAUNCH_IMAGE_KER(3, 3, 3); }
                else if (kernel_t_size == 6 && kernel_h_size == 1 && kernel_w_size == 1) { LAUNCH_IMAGE_KER(3, 0, 0); }
                else if (kernel_t_size == 6 && kernel_h_size == 1 && kernel_w_size == 6) { LAUNCH_IMAGE_KER(3, 0, 3); }
                else if (kernel_t_size == 6 && kernel_h_size == 6 && kernel_w_size == 1) { LAUNCH_IMAGE_KER(3, 3, 0); }
                else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 6) { LAUNCH_IMAGE_KER(0, 3, 3); }
                else if (kernel_t_size == 1 && kernel_h_size == 1 && kernel_w_size == 6) { LAUNCH_IMAGE_KER(0, 0, 3); }
                else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 1) { LAUNCH_IMAGE_KER(0, 3, 0); }
                else {
                    TORCH_CHECK(false, "Invalid kernel size: ", kernel_t_size, "x", kernel_h_size, "x", kernel_w_size);
                }
#undef LAUNCH_IMAGE_KER
            }
            else if (kernel_aspect_ratio_flag == 3) {
#define LAUNCH_IMAGE_KER(DT_VAL, DH_VAL, DW_VAL) \
                    cudaFuncSetAttribute( \
                        fwd_attend_ker<128, false, false, false, DT_VAL, DH_VAL, DW_VAL, 3, 6, 10>, \
                        cudaFuncAttributeMaxDynamicSharedMemorySize, \
                        mem_size \
                    ); \
                    fwd_attend_ker<128, false, false, false, DT_VAL, DH_VAL, DW_VAL, 3, 6, 10><<<grid_image, (32*NUM_WORKERS), mem_size, stream>>>(g);

                if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 3)      { LAUNCH_IMAGE_KER(1, 1, 1); }
                else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 5) { LAUNCH_IMAGE_KER(1, 1, 2); }
                else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 5) { LAUNCH_IMAGE_KER(1, 2, 2); }
                else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 1) { LAUNCH_IMAGE_KER(1, 3, 0); }
                else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 7) { LAUNCH_IMAGE_KER(1, 2, 3); }
                else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 9) { LAUNCH_IMAGE_KER(1, 2, 4); }
                else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(1, 3, 5); }
                else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 3) { LAUNCH_IMAGE_KER(1, 3, 1); }
                else if (kernel_t_size == 3 && kernel_h_size == 1 && kernel_w_size == 1) { LAUNCH_IMAGE_KER(1, 0, 0); }
                else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(0, 3, 5); }
                else if (kernel_t_size == 1 && kernel_h_size == 5 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(0, 2, 5); }
                else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 7) { LAUNCH_IMAGE_KER(0, 3, 3); }
                else if (kernel_t_size == 1 && kernel_h_size == 5 && kernel_w_size == 7) { LAUNCH_IMAGE_KER(0, 2, 3); }
                else if (kernel_t_size == 1 && kernel_h_size == 5 && kernel_w_size == 9) { LAUNCH_IMAGE_KER(0, 2, 4); }
                else if (kernel_t_size == 3 && kernel_h_size == 1 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(1, 0, 5); }
                else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(1, 1, 5); }
                else if (kernel_t_size == 1 && kernel_h_size == 3 && kernel_w_size == 10){ LAUNCH_IMAGE_KER(0, 1, 5); }
                else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 5) { LAUNCH_IMAGE_KER(0, 3, 2); }
                else {
                    TORCH_CHECK(false, "Invalid kernel size: ", kernel_t_size, "x", kernel_h_size, "x", kernel_w_size);
                }
#undef LAUNCH_IMAGE_KER
            }

            else {
                TORCH_CHECK(false, "Unsupported kernel_aspect_ratio_flag: ", kernel_aspect_ratio_flag);
            }

        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        // cudaStreamSynchronize(stream);
    }

    return o;
    //cudadevicesynchronize();
}


// =====================================================================
// Backward host launcher.
//
// Returns (dQ, dK, dV) as float32 tensors. Mirrors block-sparse BWD: a
// preprocess pass to compute D = rowsum(O * dO), then a per-KV-tile main
// pass that enumerates window-matching Q tiles using the same canvas
// dimensions and window radii baked into the FWD template instantiation.
// =====================================================================

// Single source-of-truth dispatch for (kernel_t, kernel_h, kernel_w) ->
// (DT, DH, DW) template parameters. Used by every BWD launcher branch
// to avoid re-listing the FWD's hardcoded windows in two places.
#define STA_BWD_DISPATCH_HUNYUAN_TEXT(KER) \
    if      (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 3)  { KER(1, 1, 1); } \
    else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 5)  { KER(1, 1, 2); } \
    else if (kernel_t_size == 5 && kernel_h_size == 3 && kernel_w_size == 3)  { KER(2, 1, 1); } \
    else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 5)  { KER(1, 2, 2); } \
    else if (kernel_t_size == 5 && kernel_h_size == 6 && kernel_w_size == 1)  { KER(2, 3, 0); } \
    else if (kernel_t_size == 5 && kernel_h_size == 3 && kernel_w_size == 5)  { KER(2, 1, 2); } \
    else if (kernel_t_size == 5 && kernel_h_size == 5 && kernel_w_size == 5)  { KER(2, 2, 2); } \
    else if (kernel_t_size == 5 && kernel_h_size == 5 && kernel_w_size == 7)  { KER(2, 2, 3); } \
    else if (kernel_t_size == 5 && kernel_h_size == 6 && kernel_w_size == 10) { KER(2, 3, 5); } \
    else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 10) { KER(1, 3, 5); } \
    else if (kernel_t_size == 5 && kernel_h_size == 1 && kernel_w_size == 1)  { KER(2, 0, 0); } \
    else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 10) { KER(0, 3, 5); } \
    else if (kernel_t_size == 5 && kernel_h_size == 1 && kernel_w_size == 10) { KER(2, 0, 5); } \
    else { TORCH_CHECK(false, "Invalid kernel size: ", kernel_t_size, "x", kernel_h_size, "x", kernel_w_size); }

#define STA_BWD_DISPATCH_STEPVIDEO(KER) \
    if      (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 3)  { KER(1, 1, 1); } \
    else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 6)  { KER(1, 1, 3); } \
    else if (kernel_t_size == 6 && kernel_h_size == 3 && kernel_w_size == 3)  { KER(3, 1, 1); } \
    else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 6)  { KER(1, 3, 3); } \
    else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 3)  { KER(1, 3, 1); } \
    else if (kernel_t_size == 6 && kernel_h_size == 3 && kernel_w_size == 6)  { KER(3, 1, 3); } \
    else if (kernel_t_size == 6 && kernel_h_size == 6 && kernel_w_size == 6)  { KER(3, 3, 3); } \
    else if (kernel_t_size == 6 && kernel_h_size == 1 && kernel_w_size == 1)  { KER(3, 0, 0); } \
    else if (kernel_t_size == 6 && kernel_h_size == 1 && kernel_w_size == 6)  { KER(3, 0, 3); } \
    else if (kernel_t_size == 6 && kernel_h_size == 6 && kernel_w_size == 1)  { KER(3, 3, 0); } \
    else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 6)  { KER(0, 3, 3); } \
    else if (kernel_t_size == 1 && kernel_h_size == 1 && kernel_w_size == 6)  { KER(0, 0, 3); } \
    else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 1)  { KER(0, 3, 0); } \
    else { TORCH_CHECK(false, "Invalid kernel size: ", kernel_t_size, "x", kernel_h_size, "x", kernel_w_size); }

#define STA_BWD_DISPATCH_WAN(KER) \
    if      (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 3)  { KER(1, 1, 1); } \
    else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 5)  { KER(1, 1, 2); } \
    else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 5)  { KER(1, 2, 2); } \
    else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 1)  { KER(1, 3, 0); } \
    else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 7)  { KER(1, 2, 3); } \
    else if (kernel_t_size == 3 && kernel_h_size == 5 && kernel_w_size == 9)  { KER(1, 2, 4); } \
    else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 10) { KER(1, 3, 5); } \
    else if (kernel_t_size == 3 && kernel_h_size == 6 && kernel_w_size == 3)  { KER(1, 3, 1); } \
    else if (kernel_t_size == 3 && kernel_h_size == 1 && kernel_w_size == 1)  { KER(1, 0, 0); } \
    else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 10) { KER(0, 3, 5); } \
    else if (kernel_t_size == 1 && kernel_h_size == 5 && kernel_w_size == 10) { KER(0, 2, 5); } \
    else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 7)  { KER(0, 3, 3); } \
    else if (kernel_t_size == 1 && kernel_h_size == 5 && kernel_w_size == 7)  { KER(0, 2, 3); } \
    else if (kernel_t_size == 1 && kernel_h_size == 5 && kernel_w_size == 9)  { KER(0, 2, 4); } \
    else if (kernel_t_size == 3 && kernel_h_size == 1 && kernel_w_size == 10) { KER(1, 0, 5); } \
    else if (kernel_t_size == 3 && kernel_h_size == 3 && kernel_w_size == 10) { KER(1, 1, 5); } \
    else if (kernel_t_size == 1 && kernel_h_size == 3 && kernel_w_size == 10) { KER(0, 1, 5); } \
    else if (kernel_t_size == 1 && kernel_h_size == 6 && kernel_w_size == 5)  { KER(0, 3, 2); } \
    else { TORCH_CHECK(false, "Invalid kernel size: ", kernel_t_size, "x", kernel_h_size, "x", kernel_w_size); }

std::vector<torch::Tensor>
sta_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
             torch::Tensor o, torch::Tensor l_vec, torch::Tensor og,
             int kernel_t_size, int kernel_h_size, int kernel_w_size,
             int text_length, bool has_text, int kernel_aspect_ratio_flag)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(og);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(head_dim == 128, "STA backward currently only supports head_dim=128");
    TORCH_CHECK(q.size(0) == batch && k.size(0) == batch && v.size(0) == batch && o.size(0) == batch && og.size(0) == batch,
                "Batch dim must match for all inputs");
    TORCH_CHECK(q.size(2) == seq_len && k.size(2) == seq_len && v.size(2) == seq_len && o.size(2) == seq_len && og.size(2) == seq_len,
                "Sequence length must match for all inputs");
    TORCH_CHECK(q.size(3) == head_dim && k.size(3) == head_dim && v.size(3) == head_dim && o.size(3) == head_dim && og.size(3) == head_dim,
                "Head dim must match for all inputs");
    TORCH_CHECK(qo_heads >= kv_heads && qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads && o.size(1) == qo_heads && og.size(1) == qo_heads,
                "QO heads must match for q/o/og");
    TORCH_CHECK(k.size(1) == kv_heads && v.size(1) == kv_heads, "KV heads must match for k/v");

    auto hr = qo_heads / kv_heads;

    bf16*  d_q  = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16*  d_k  = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16*  d_v  = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16*  d_o  = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_og = reinterpret_cast<bf16*>(og.data_ptr<c10::BFloat16>());
    float* d_l  = l_vec.data_ptr<float>();

    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous);

    torch::Tensor qg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(qo_heads),
                                     static_cast<const uint>(seq_len),
                                     static_cast<const uint>(head_dim)}, opts_fp32);
    torch::Tensor kg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(seq_len),
                                     static_cast<const uint>(head_dim)}, opts_fp32);
    torch::Tensor vg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(seq_len),
                                     static_cast<const uint>(head_dim)}, opts_fp32);
    torch::Tensor d_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(1)}, opts_fp32);

    float* d_qg = qg.data_ptr<float>();
    float* d_kg = kg.data_ptr<float>();
    float* d_vg = vg.data_ptr<float>();
    float* d_d  = d_vec.data_ptr<float>();

    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int mem_size_prep = kittens::MAX_SHARED_MEMORY;
    constexpr int mem_size_main = kittens::MAX_SHARED_MEMORY;

    // Prep kernel: D = rowsum(O * dO).
    {
        using og_tile = st_bf<4*16, 128>;
        using o_tile  = st_bf<4*16, 128>;
        using d_tile  = col_vec<st_fl<4*16, 128>>;
        using og_global = gl<bf16,  -1, -1, -1, -1, og_tile>;
        using o_global  = gl<bf16,  -1, -1, -1, -1, o_tile>;
        using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

        og_global og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        o_global  o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        d_global  d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U, static_cast<unsigned int>(seq_len)};

        bwd_prep_globals<128> prep_g{og_arg, o_arg, d_arg};

        cudaFuncSetAttribute(bwd_attend_prep_ker<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size_prep);

        const int prep_threads = BWD_PREP_NUM_WARPS * kittens::WARP_THREADS;
        dim3 grid_prep(seq_len / (BWD_PREP_NUM_WARPS * kittens::TILE_ROW_DIM<bf16> * 4), qo_heads, batch);
        bwd_attend_prep_ker<128><<<grid_prep, prep_threads, mem_size_prep, stream>>>(prep_g);
    }

    // Main BWD kernel: per-KV-tile, accumulate into FP32 grads.
    {
        using G = bwd_attend_ker_tile_dims<128>;
        using bwd_q_tile  = st_bf<G::tile_h_qo, G::tile_width>;
        using bwd_k_tile  = st_bf<G::tile_h,    G::tile_width>;
        using bwd_v_tile  = st_bf<G::tile_h,    G::tile_width>;
        using bwd_og_tile = st_bf<G::tile_h_qo, G::tile_width>;
        using bwd_qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
        using bwd_kg_tile = st_fl<G::tile_h,    G::tile_width>;
        using bwd_vg_tile = st_fl<G::tile_h,    G::tile_width>;
        using bwd_l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
        using bwd_d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

        using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, bwd_q_tile>;
        using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, bwd_k_tile>;
        using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, bwd_v_tile>;
        using bwd_og_global = gl<bf16,  -1, -1, -1, -1, bwd_og_tile>;
        using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
        using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
        using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;
        using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
        using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;

        bwd_q_global  q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_k_global  k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_v_global  v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_og_global og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_qg_global qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_kg_global kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_vg_global vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_l_global  l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U, static_cast<unsigned int>(seq_len)};
        bwd_d_global  d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U, static_cast<unsigned int>(seq_len)};

        bwd_globals<128> bwd_g{q_arg, k_arg, v_arg, og_arg, qg_arg, kg_arg, vg_arg, l_arg, d_arg,
                               static_cast<int>(seq_len), static_cast<int>(text_length), static_cast<int>(hr)};

        const int kv_tile_h = G::tile_h;
        dim3 grid_bwd(seq_len / kv_tile_h, qo_heads, batch);
        const int threads = 128;

        if (has_text) {
            // Hunyuan: 30x48x80 canvas, 5x6x10 tiles, with text segment.
            #define LAUNCH_BWD_HUNYUAN(DT_VAL, DH_VAL, DW_VAL) \
                cudaFuncSetAttribute( \
                    bwd_attend_ker<128, true, true, DT_VAL, DH_VAL, DW_VAL, 5, 6, 10>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size_main); \
                bwd_attend_ker<128, true, true, DT_VAL, DH_VAL, DW_VAL, 5, 6, 10> \
                    <<<grid_bwd, threads, mem_size_main, stream>>>(bwd_g);

            STA_BWD_DISPATCH_HUNYUAN_TEXT(LAUNCH_BWD_HUNYUAN)
            #undef LAUNCH_BWD_HUNYUAN
        } else if (kernel_aspect_ratio_flag == 2) {
            // Stepvideo: 36x48x48 canvas, 6x6x6 tiles, no text.
            #define LAUNCH_BWD_STEPVIDEO(DT_VAL, DH_VAL, DW_VAL) \
                cudaFuncSetAttribute( \
                    bwd_attend_ker<128, false, false, DT_VAL, DH_VAL, DW_VAL, 6, 6, 6>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size_main); \
                bwd_attend_ker<128, false, false, DT_VAL, DH_VAL, DW_VAL, 6, 6, 6> \
                    <<<grid_bwd, threads, mem_size_main, stream>>>(bwd_g);

            STA_BWD_DISPATCH_STEPVIDEO(LAUNCH_BWD_STEPVIDEO)
            #undef LAUNCH_BWD_STEPVIDEO
        } else if (kernel_aspect_ratio_flag == 3) {
            // Wan: 18x48x80 canvas, 3x6x10 tiles, no text.
            #define LAUNCH_BWD_WAN(DT_VAL, DH_VAL, DW_VAL) \
                cudaFuncSetAttribute( \
                    bwd_attend_ker<128, false, false, DT_VAL, DH_VAL, DW_VAL, 3, 6, 10>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size_main); \
                bwd_attend_ker<128, false, false, DT_VAL, DH_VAL, DW_VAL, 3, 6, 10> \
                    <<<grid_bwd, threads, mem_size_main, stream>>>(bwd_g);

            STA_BWD_DISPATCH_WAN(LAUNCH_BWD_WAN)
            #undef LAUNCH_BWD_WAN
        } else {
            TORCH_CHECK(false, "Unsupported kernel_aspect_ratio_flag: ", kernel_aspect_ratio_flag);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    return {qg, kg, vg};
}

#undef STA_BWD_DISPATCH_HUNYUAN_TEXT
#undef STA_BWD_DISPATCH_STEPVIDEO
#undef STA_BWD_DISPATCH_WAN

