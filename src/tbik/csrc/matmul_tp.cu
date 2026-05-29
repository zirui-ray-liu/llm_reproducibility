// TP-invariant deterministic bf16 matmul for Hopper (sm_90a), via CUTLASS cute + wgmma.
//
// Computes C[M,N] = A[M,K] @ B[K,N] with a tree-reduction numerical contract that makes the
// result bitwise-identical regardless of how K is split across tensor-parallel ranks (and
// regardless of batch size), so it is exactly reproducible under TP all-reduce:
//   * K is split into 256-wide leaves; leaves are grouped into G = 2^a groups of
//     FLB = oddpart(T) leaves each (T = K/256).
//   * within a group the fp32 wgmma accumulator runs continuously (cleared at group start,
//     accumulated by cute::gemm) -> exactly one bf16 round per group.
//   * groups are combined by a fixed balanced binary tree, each combine a bf16 add
//     (__hadd2, round-to-nearest-even of fp32(a)+fp32(b)) -> bitwise == the TP all-reduce.
// Group boundaries align with every valid TP shard boundary (oddpart invariance), and each
// group's fp32 result is computed identically whether the kernel runs on the full K or on a
// shard, so TP-invariance holds exactly (the wgmma internal accumulation order is the same in
// the full run and in every shard, over a never-split group).
//
// Operand majors for C = A@B, all ROW-MAJOR (torch):
//   A row-major [M,K] -> K contiguous -> GMMA::Major::K  + Layout_K_SW128
//   B row-major [K,N] -> N contiguous -> viewed as (N,K) stride (1,N), Major::MN + Layout_MN_SW128
//
// Two backends, selected per shape in matmul_tp():
//   * a monolithic kernel (1 or 2 wgmma warpgroups), and
//   * a warp-specialized cooperative kernel (producer warpgroup feeds consumer warpgroups).
//
// Requires -arch=sm_90a and the CUTLASS/cute include path.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <climits>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
using namespace cute;
using bf16 = __nv_bfloat16;
using CT   = cutlass::bfloat16_t;   // cute operand type (bitwise == __nv_bfloat16)

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#define BK 64                       // 128B-swizzle K atom (= 64 bf16); leaf=256 => 4 BK-tiles/leaf
#ifndef NSTAGES
#define NSTAGES 3
#endif
#define CHUNKS_PER_LEAF (256/BK)    // = 4
#ifndef ML_CAP
#define ML_CAP 6
#endif

template <class TA, class TB, class SLA, class SLB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TA, cosize_v<SLA>> A;
  alignas(128) cute::ArrayEngine<TB, cosize_v<SLB>> B;
};

// Monolithic wgmma kernel. ML = bf16 group-tree stack depth (register-resident).
template <int ML, class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class TiledCopyA,
          class BStride, class BSmemLayout, class TiledCopyB,
          class CStride, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void wgmma_tp_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
        const CT* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
        const CT* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
        CT* C, CStride dC, TiledMma mma,
        int T, int FLB, int LEVEL_K) {
  using TA = CT; using TB = CT;
  const int Ltree = LEVEL_K - 1;
  const int tiles_per_group = FLB * CHUNKS_PER_LEAF;   // BK-tiles spanned by one fp32 group

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BM,BK,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BN,BK,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BM,BN)

  extern __shared__ char smem_raw[];
  using SS = SharedStorage<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_raw);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BM,BK,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  ThrCopy tca = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = tca.partition_S(gA);
  Tensor tAsA = tca.partition_D(as_position_independent_swizzle_tensor(sA));
  ThrCopy tcb = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = tcb.partition_S(gB);
  Tensor tBsB = tcb.partition_D(as_position_independent_swizzle_tensor(sB));

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor acc  = thr_mma.make_fragment_C(tCgC);   // fp32 accumulators, this group

  constexpr int NACC = decltype(size(acc))::value;       // fp32 accs / thread
  constexpr int NP   = NACC / 2;                         // packed bf16x2 pairs
  // register-resident bf16 group-tree stack + per-level carry counters
  __nv_bfloat162 stk[NP][ML];
  int gcount[ML];
  #pragma unroll
  for (int l=0;l<ML;l++) gcount[l]=0;

  auto K_TILE_MAX = size<3>(tAgA);
  auto K_PIPE = size<3>(tAsA);
  int kpr = 0, kpw = K_PIPE-1;
  // prologue: kick off the first K_PIPE-1 tile loads
  CUTE_UNROLL
  for (int s=0; s<K_PIPE-1; ++s) {
    copy(copy_a, tAgA(_,_,_,s), tAsA(_,_,_,s));
    copy(copy_b, tBgB(_,_,_,s), tBsB(_,_,_,s));
    cp_async_fence();
  }

  for (int kt=0; kt<K_TILE_MAX; ++kt) {
    cp_async_wait<NSTAGES-2>();          // tile kt arrived; keep the rest in flight (overlap)
    __syncthreads();

    int gpos = kt % tiles_per_group;
    if (gpos == 0) clear(acc);                       // start of a new fp32 group

    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    cute::gemm(mma, tCrA(_,_,_,kpr), tCrB(_,_,_,kpr), acc);   // accumulate this BK-tile (async)
    warpgroup_commit_batch();
    ++kpr; if (kpr==K_PIPE) kpr=0;

    bool gend = (gpos == tiles_per_group - 1);
    // group end -> drain all wgmma so acc is complete for the tree; otherwise keep <=1 wgmma
    // in flight (overlap). This also guarantees wgmma(kt-1) is done before its smem buffer is
    // recycled by the prefetch below (kpw == tile (kt-1)'s buffer).
    if (gend) warpgroup_wait<0>(); else warpgroup_wait<1>();
    warpgroup_fence_operand(acc);

    // prefetch the tile K_PIPE-1 ahead into the now-free buffer (overlaps with this wgmma)
    int knext = kt + (K_PIPE-1);
    if (knext < K_TILE_MAX) {
      copy(copy_a, tAgA(_,_,_,knext), tAsA(_,_,_,kpw));
      copy(copy_b, tBgB(_,_,_,knext), tBsB(_,_,_,kpw));
    }
    cp_async_fence();
    ++kpw; if (kpw==K_PIPE) kpw=0;

    if (gend) {
      // group complete: round fp32 acc -> bf16 once, push into the balanced binary tree.
      // binary-carry plan over levels (computed once, shared by all pairs; top radix=INF).
      unsigned cmask = 0; int cpark = 0;
      if (Ltree > 0) {
        int level = 0;
        while (true) {
          int radix = (level == Ltree-1) ? INT_MAX : 2;
          int cnt = gcount[level] + 1;
          if (cnt > 1) cmask |= (1u<<level);
          if (cnt == radix) { gcount[level]=0; level++; }
          else { gcount[level]=cnt; cpark=level; break; }
        }
      }
      #pragma unroll
      for (int p=0; p<NP; ++p) {
        __nv_bfloat162 x = __floats2bfloat162_rn(acc(2*p), acc(2*p+1));
        if (Ltree == 0) { stk[p][0] = x; }
        else {
          #pragma unroll
          for (int L=0; L<ML; ++L) if (cmask & (1u<<L)) x = __hadd2(stk[p][L], x);
          #pragma unroll
          for (int L=0; L<ML; ++L) if (L==cpark) stk[p][L] = x;
        }
      }
    }
    __syncthreads();
  }

  // epilogue: tree top (stk[*][ML-1]) holds the final bf16 result. Unpack to C via cute coords.
  #pragma unroll
  for (int p=0; p<NP; ++p) {
    __nv_bfloat162 v = stk[p][ML-1];
    reinterpret_cast<bf16&>(tCgC(2*p))   = __low2bfloat16(v);
    reinterpret_cast<bf16&>(tCgC(2*p+1)) = __high2bfloat16(v);
  }
}

// ===================== warp-specialized cooperative kernel =====================
// A producer warpgroup streams A/B tiles into a PipelineAsync ring via cp.async; the consumer
// warpgroups run wgmma and carry the fp32-group + bf16-tree contract in their group-end epilogue
// (no cluster / TMA). Decoupling load from compute keeps the tensor cores fed; setmaxnreg moves
// registers from the load-only producer to the math+tree consumers so the tree fits without spills.
#ifndef WS_STAGES
#define WS_STAGES 4
#endif
using WSPipeline = cutlass::PipelineAsync<WS_STAGES>;
template <class TA, class TB, class SLA, class SLB>
struct SharedStorageWS {
  alignas(128) cute::ArrayEngine<TA, cosize_v<SLA>> A;
  alignas(128) cute::ArrayEngine<TB, cosize_v<SLB>> B;
  alignas(16) typename WSPipeline::SharedStorage pipeline;
};

template <int ML, class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class ProdCopyA,
          class BStride, class BSmemLayout, class ProdCopyB,
          class CStride, class TiledMma>
__global__ static __launch_bounds__(384)
void wgmma_tp_ws_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
        const CT* A, AStride dA, ASmemLayout sA_layout, ProdCopyA copy_a,
        const CT* B, BStride dB, BSmemLayout sB_layout, ProdCopyB copy_b,
        CT* C, CStride dC, TiledMma mma,
        int T, int FLB, int LEVEL_K) {
  using TA = CT; using TB = CT;
  const int Ltree = LEVEL_K - 1;
  const int tiles_per_group = FLB * CHUNKS_PER_LEAF;
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});
  extern __shared__ char smem_raw[];
  using SS = SharedStorageWS<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_raw);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  // 384 threads = 2 consumer warpgroups (tid 0..255: wgmma + tree, 64 accs/thread) + 1 producer
  // warpgroup (tid 256..383: cp.async). The register split (consumer alloc 232 + producer dealloc 24
  // = 256) keeps 2 blocks/SM resident.
  int tid = threadIdx.x; bool is_prod = (tid >= 256);
  typename WSPipeline::Params params;
  params.role = is_prod ? WSPipeline::ThreadCategory::Producer : WSPipeline::ThreadCategory::Consumer;
  params.producer_arv_count = 128; params.consumer_arv_count = 256; params.initializing_warp = 0;
  WSPipeline pipeline(smem.pipeline, params, cute::true_type{});
  __syncthreads();
  int K_TILE_MAX = size<2>(gA);

  if (is_prod) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    ThrCopy tca = copy_a.get_slice(tid - 256);
    Tensor tAgA = tca.partition_S(gA);
    Tensor tAsA = tca.partition_D(as_position_independent_swizzle_tensor(sA));
    ThrCopy tcb = copy_b.get_slice(tid - 256);
    Tensor tBgB = tcb.partition_S(gB);
    Tensor tBsB = tcb.partition_D(as_position_independent_swizzle_tensor(sB));
    auto wr = cutlass::make_producer_start_state<WSPipeline>();
    CUTLASS_PRAGMA_NO_UNROLL
    for (int kt = 0; kt < K_TILE_MAX; ++kt) {
      pipeline.producer_acquire(wr);
      int s = wr.index();
      copy(copy_a, tAgA(_,_,_,kt), tAsA(_,_,_,s));
      copy(copy_b, tBgB(_,_,_,kt), tBsB(_,_,_,s));
      pipeline.producer_commit(wr, cutlass::arch::cpasync_barrier_arrive);
      ++wr;
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    ThrMMA thr_mma = mma.get_slice(tid);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor acc  = thr_mma.make_fragment_C(tCgC);
    constexpr int NACC = decltype(size(acc))::value;
    constexpr int NP   = NACC / 2;
    __nv_bfloat162 stk[NP][ML];
    int gcount[ML];
    #pragma unroll
    for (int l=0;l<ML;l++) gcount[l]=0;
    auto rd = cutlass::PipelineState<WS_STAGES>();
    for (int kt=0; kt<K_TILE_MAX; ++kt) {
      int gpos = kt % tiles_per_group;
      if (gpos == 0) clear(acc);                     // start of a new fp32 group
      auto tok = pipeline.consumer_try_wait(rd);
      pipeline.consumer_wait(rd, tok);
      int s = rd.index();
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(mma, tCrA(_,_,_,s), tCrB(_,_,_,s), acc);
      warpgroup_commit_batch();
      bool gend = (gpos == tiles_per_group - 1);
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);
      pipeline.consumer_release(rd);                 // wgmma on stage s is done -> free the buffer
      ++rd;
      if (gend) {
        // group complete: round fp32 acc -> bf16 once, push into the balanced binary tree.
        unsigned cmask = 0; int cpark = 0;
        if (Ltree > 0) {
          int level = 0;
          while (true) {
            int radix = (level == Ltree-1) ? INT_MAX : 2;
            int cnt = gcount[level] + 1;
            if (cnt > 1) cmask |= (1u<<level);
            if (cnt == radix) { gcount[level]=0; level++; }
            else { gcount[level]=cnt; cpark=level; break; }
          }
        }
        #pragma unroll
        for (int p=0; p<NP; ++p) {
          __nv_bfloat162 x = __floats2bfloat162_rn(acc(2*p), acc(2*p+1));
          if (Ltree == 0) { stk[p][0] = x; }
          else {
            #pragma unroll
            for (int L=0; L<ML; ++L) if (cmask & (1u<<L)) x = __hadd2(stk[p][L], x);
            #pragma unroll
            for (int L=0; L<ML; ++L) if (L==cpark) stk[p][L] = x;
          }
        }
      }
    }
    #pragma unroll
    for (int p=0; p<NP; ++p) {
      __nv_bfloat162 v = stk[p][ML-1];
      reinterpret_cast<bf16&>(tCgC(2*p))   = __low2bfloat16(v);
      reinterpret_cast<bf16&>(tCgC(2*p+1)) = __high2bfloat16(v);
    }
  }
}

template <int ML>
static void launch_ws(const CT* A, const CT* B, CT* C,
                      int M, int N, int K, int T, int FLB, int LEVEL_K,
                      int num_sms, cudaStream_t stream) {
  auto prob = make_shape(M, N, K);
  auto bM = Int<128>{}; auto bN = Int<BN>{}; auto bK = Int<BK>{}; auto bP = Int<WS_STAGES>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(Int<1>{}, N);
  auto dC = make_stride(N, Int<1>{});
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<CT>{},  make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<CT>{}, make_shape(bN,bK,bP));
  // producer = 1 warpgroup (128 thr, all do cp.async): A K-major 16x8, B MN-major 16x8.
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, CT>{},
        Layout<Shape<_16,_8>,Stride<_8,_1>>{}, Layout<Shape<_1,_8>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, CT>{},
        Layout<Shape<_16,_8>>{}, Layout<Shape<_8,_1>>{});
  // 2 consumer warpgroups stacked along M (128x128 tile) -> 64 accs/thread.
  TiledMMA mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{},
        Layout<Shape<_2,_1,_1>>{});
  int smem_bytes = int(sizeof(SharedStorageWS<CT,CT,decltype(sA),decltype(sB)>));
  auto kernel = &wgmma_tp_ws_kernel<ML, decltype(prob), decltype(cta_tiler),
      decltype(dA), decltype(sA), decltype(copyA),
      decltype(dB), decltype(sB), decltype(copyB),
      decltype(dC), decltype(mma)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  dim3 grid(ceil_div(M,bM), ceil_div(N,bN)), block(384);
  kernel<<<grid, block, smem_bytes, stream>>>(prob, cta_tiler,
      A, dA, sA, copyA, B, dB, sB, copyB, C, dC, mma, T, FLB, LEVEL_K);
}

// ---- host ----
// Monolithic launch, parametric on CTA tile height BMv (= NWG*64). The second warpgroup is stacked
// along M (Layout<Shape<NWG,_1,_1>> over the SM90_64x128x16 atom) so each thread still owns 64 fp32
// accs (no spill). All copy/smem layouts derive from NWG.
//   * BMv=128 (2 warpgroups): wide outputs -> doubled B-tile reuse, fewer CTAs.
//   * BMv=64  (1 warpgroup) : narrow / small-M shapes -> more M-parallelism to fill SMs.
template <int ML, int BMv>
static void launch_wgmma(const CT* A, const CT* B, CT* C,
                         int M, int N, int K, int T, int FLB, int LEVEL_K,
                         int num_sms, cudaStream_t stream) {
  constexpr int NWG = BMv / 64;            // warpgroups stacked along M (1 or 2)
  auto prob = make_shape(M, N, K);
  auto bM = Int<BMv>{}; auto bN = Int<BN>{}; auto bK = Int<BK>{}; auto bP = Int<NSTAGES>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto dA = make_stride(K, Int<1>{});       // A [M,K] row-major (K-major)
  auto dB = make_stride(Int<1>{}, N);       // B [K,N] row-major -> (N,K) MN-major
  auto dC = make_stride(N, Int<1>{});       // C [M,N] row-major
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<CT>{},  make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<CT>{}, make_shape(bN,bK,bP));
  // copy thread layouts scale with NWG (128 thr/warpgroup): A K-major 16*NWG x 8, B MN-major 16 x 8*NWG.
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, CT>{},
        Layout<Shape<Int<16*NWG>,_8>,Stride<_8,_1>>{}, Layout<Shape<_1,_8>>{});   // K-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, CT>{},
        Layout<Shape<_16,Int<8*NWG>>>{}, Layout<Shape<_8,_1>>{});                 // MN-major
  TiledMMA mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{},
        Layout<Shape<Int<NWG>,_1,_1>>{});

  int smem_bytes = int(sizeof(SharedStorage<CT,CT,decltype(sA),decltype(sB)>));
  auto kernel = &wgmma_tp_kernel<ML, decltype(prob), decltype(cta_tiler),
      decltype(dA), decltype(sA), decltype(copyA),
      decltype(dB), decltype(sB), decltype(copyB),
      decltype(dC), decltype(mma)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  dim3 grid(ceil_div(M,bM), ceil_div(N,bN)), block(size(mma));
  kernel<<<grid, block, smem_bytes, stream>>>(prob, cta_tiler,
      A, dA, sA, copyA, B, dB, sB, copyB, C, dC, mma, T, FLB, LEVEL_K);
}

void matmul_tp(at::Tensor A, at::Tensor B, at::Tensor C) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(A.scalar_type()==at::kBFloat16 && B.scalar_type()==at::kBFloat16 && C.scalar_type()==at::kBFloat16, "bf16 required");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "contiguous required");
  int M=A.size(0), K=A.size(1), N=B.size(1);
  TORCH_CHECK(B.size(0)==K, "K mismatch");
  TORCH_CHECK(K%256==0, "K%256");
  TORCH_CHECK(N%BN==0, "N%BN");
  // Shape dispatch (static -> TP-invariance unaffected; all backends share the same contract):
  //   * warp-specialized cooperative kernel for shapes with enough work to fill the SMs at the
  //     128x128 tile (M>=1024 and the output is the common N==2048 hidden size or wide, N>=4096);
  //   * otherwise the monolithic kernel, bM=128 for wide moderate-M outputs (N>=2560, 128<=M<=2048)
  //     and bM=64 for narrow / small / very-tall shapes.
  const bool use_ws = (M >= 1024) && (N == 2048 || N >= 4096);
  const bool mono_bm128 = (N >= 2560 && M >= 128 && M <= 2048);
  const int BMrt = use_ws ? 128 : (mono_bm128 ? 128 : 64);
  // M-edge (e.g. decode M=1/16/32): pad rows up to a multiple of the chosen BM, run the same
  // kernel, slice back. Deterministic per call -> TP-invariance unaffected (the real M rows are
  // identical regardless of padding). Padding overhead is negligible for tiny decode M.
  if (M % BMrt != 0) {
    int Mp = ((M + BMrt - 1) / BMrt) * BMrt;
    auto Apad = at::zeros({Mp, K}, A.options());
    Apad.narrow(0, 0, M).copy_(A);
    auto Cpad = at::empty({Mp, N}, C.options());
    matmul_tp(Apad, B, Cpad);
    C.copy_(Cpad.narrow(0, 0, M));
    return;
  }
  int T=K/256, FLB=T, a=0;
  while (FLB%2==0 && FLB>1) { FLB/=2; a++; }
  int LEVEL_K=a+1;
  TORCH_CHECK(LEVEL_K<=ML_CAP, "LEVEL_K>ML_CAP");
  int dev=A.get_device(), num_sms=0; cudaDeviceGetAttribute(&num_sms,cudaDevAttrMultiProcessorCount,dev);
  auto stream=at::cuda::getCurrentCUDAStream();
  const CT* pA=reinterpret_cast<const CT*>(A.data_ptr());
  const CT* pB=reinterpret_cast<const CT*>(B.data_ptr());
  CT* pC=reinterpret_cast<CT*>(C.data_ptr());
  #define LAUNCH(MLV) do { \
    if (use_ws)         launch_ws<MLV>(pA,pB,pC,M,N,K,T,FLB,LEVEL_K,num_sms,stream); \
    else if (mono_bm128)  launch_wgmma<MLV,128>(pA,pB,pC,M,N,K,T,FLB,LEVEL_K,num_sms,stream); \
    else                launch_wgmma<MLV, 64>(pA,pB,pC,M,N,K,T,FLB,LEVEL_K,num_sms,stream); \
  } while(0)
  switch (LEVEL_K) {
    case 1: case 2: LAUNCH(1); break;
    case 3: LAUNCH(2); break;
    case 4: LAUNCH(3); break;
    case 5: LAUNCH(4); break;
    default: LAUNCH(ML_CAP); break;
  }
  #undef LAUNCH
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("matmul_tp",&matmul_tp,"TP-invariant bf16 matmul (wgmma + fp32-group tree)");}
