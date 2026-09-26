
#ifndef brmoe_CUDA_WITH_ZERO_KERNEL_CUH 
#define brmoe_CUDA_WITH_ZERO_KERNEL_CUH


#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include<algorithm>

constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I2 = Vec<int, 2>;
using I2_2 = Vec<I2,2>;
// Matrix fragments for tensor core instructions; their precise layout is documented here: 
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>; // quantization scales
using FragZ = Vec<half2, 1>; 

// Predicated asynchronous global->shared copy; used for inputs A where we apply predication to handle batchsizes that
// are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n" 
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// 注意: 原来这两个函数带 createpolicy + L2::cache_hint。在 sm_120 上它与 MOE
// 序言的寄存器分配互相触发 illegal instruction (printf 即消失的 Heisenbug,
// sanitizer 定位到这条 asm, job 38989)。策略提示只是微优化, 直接去掉。
__device__ inline void cp_async_stream2(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 8;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   cp.async.ca.shared.global [%0], [%1], %2;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

__device__ inline void cp_async_stream1(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   cp.async.ca.shared.global [%0], [%1], %2;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32 output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases. 
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

__device__ inline FragB dequant_with_zeros(int& q) {
  const int LO = 0x00070007;
  const int HI = 0x00380038;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64006400;
  const int MUL = 0x30003000;
  const int ADD = 0xd800d800;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used only for grouped quantization.
__device__ inline void scale_with_zeros(FragB& frag_b, FragS& frag_s, int i, FragZ& frag_z) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  half2 z = __half2half2(reinterpret_cast<__half*>(&frag_z)[i]);
  frag_b[0] = __hfma2(frag_b[0], s, z);
  frag_b[1] = __hfma2(frag_b[1], s, z);
}


// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible globally.
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}


// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible globally, while releasing the barrier. 
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val)); 
  }
}


template <
  const int threads, // number of threads in a threadblock
  const int thread_m_blocks, // number of 16x16 blocks in the m dimension (batchsize) of the threadblock 
  const int thread_n_blocks, // same for n dimension (output) 
  const int thread_k_blocks, // same for k dimension (reduction)
  const int stages, // number of stages for the async global->shared fetch pipeline
  const int group_blocks = -1, // number of consecutive 16x16 blocks with a separate quantization scale
  const bool MOE = false       // grouped MoE 模式: 一块 = (一个 m-tile, 一个 n-tile), 按块查专家
>
__global__
#ifdef BRMOE_MOE_MIN_BLOCKS
__launch_bounds__(threads, MOE ? BRMOE_MOE_MIN_BLOCKS : 1)
#endif
void brmoeWithZeros(
  const int4* __restrict__ A, // fp16 input matrix of shape mxk (MOE: 已按专家排序的 sorted 空间)
  const I2* __restrict__ B1, // 3bit quantized weight matrix of shape kxn (MOE: [E, ...] 带专家维)
  const int* __restrict__ B2,
        int4* __restrict__ C, // fp16 output buffer of shape mxn (MOE: sorted 空间)
  const int4* __restrict__ s, // fp16 quantization scales of shape (k/groupsize)xn
  const int4* __restrict__ z, 
  int  prob_m, // batch dimension m (MOE: 无意义, 由 num_post_ptr 覆盖)
  int  prob_n, // output dimension n
  int  prob_k, // reduction dimension k
  int* locks, // extra global storage for barrier synchronization (MOE: 不用)
  // ---- MOE 专属参数 (MOE=false 时全为 nullptr/0) ----
  const int* __restrict__ expert_ids = nullptr,  // [m_blocks_max] 每个 m-tile 的专家号
  const int* __restrict__ num_post_ptr = nullptr, // [1] device 上的 num_post (零 host 同步)
  long b1_e_stride = 0,  // 专家间 stride, 单位 = 指针元素 (I2 / int / int4)
  long b2_e_stride = 0,
  long s_e_stride = 0,
  long z_e_stride = 0,
  // ---- MOE split-K (仅 MOE 模式; grid.y = k_splits, blockIdx.y 选 K 段) ----
  // >1 时各段的部分和以 fp32 atomicAdd 累进 C32 (调用方预清零), fp16 的 C 不写。
  // 动机: 小 M 下 grid.x 只有 m_blocks*n_tiles 百来个 block (M=1 时 132 个,
  // 填不满 170 个 SM), 且每块串行走完整个 K (2048/128=16 次流水迭代) ->
  // 延迟受限。拆 K 后关键路径除以 k_splits, 与 GEMV 的 split-K 同理。
  int k_splits = 1,
  float* __restrict__ C32 = nullptr
) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the same size, which might involve multiple 
  // column "slices" (of width 16 * `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM example: 
  //   0 1 3 
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it ensures good utilization of all SMs
  // for many kinds of shape and GPU configurations, while requiring as few slow global cross-threadblock reductions as 
  // possible.
  
  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a better partitioning with less reductions
  //if( threadIdx.x == 0 & blockIdx.x == 0)
  //  printf("get s: %d, get s: %d, get s: %d, get s: %d", ((int*)s)[0], ((int*)s)[1], ((int*)s)[2], ((int*)s)[3]);
  int parallel = 1;
  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters, slice_row, slice_col_par, slice_col, slice_iters, slice_count = 0, slice_idx;
  bool moe_active = true;

  if constexpr (MOE) {
    // ---- grouped MoE 调度 (v1): 不做 stream-K ----
    // grid.x = m_blocks_max * n_tiles (超发); num_post 是 device 值 -> 零 host 同步。
    // 一个 block = (一个 16 行 m-tile, 一个 n-tile), 跑完整 K。slice_count=1 ->
    // 无跨 block 全局归约, locks 不用。
    // 注意: 超发块**不能提前 return** —— 本 kernel 的 extern __shared__ 声明在序言
    // 之后, 提前 return 在 sm_120 上会触发代码生成问题 (illegal instruction,
    // 加 printf 即消失的 Heisenbug, 见 debug_moe_cuda.py / 38984-38986)。
    // 改成"不活跃块全零化": prob_m=0 使 A 谓词全假, slice_iters=0 跳过主循环。
    constexpr int m_rows = 16 * thread_m_blocks;      // 16 (thread_m_blocks=1)
    int pid_m = blockIdx.x / n_tiles;
    int num_post = num_post_ptr[0];
    moe_active = pid_m * m_rows < num_post;
    long expert = moe_active ? (long)expert_ids[pid_m] : 0;
    B1 += expert * b1_e_stride;
    B2 += expert * b2_e_stride;
    s  += expert * s_e_stride;
    z  += expert * z_e_stride;
    A  += (long)pid_m * m_rows * (prob_k / 8);        // A/C 都在 sorted 空间, 按行块平移
    C  += (long)pid_m * m_rows * (prob_n / 8);
    if (C32) C32 += (long)pid_m * m_rows * prob_n;    // split-K 部分和缓冲同步平移
    prob_m = moe_active ? min(m_rows, num_post - pid_m * m_rows) : 0;
    // 调度器赋值走原始的 init_slice() 路径 (下方调用点): 代入
    // iters=k_tiles / slice_row=0 / slice_col_par=pid_n 恰好得到 slice_iters=k_tiles。
    iters = k_tiles;
    slice_row = 0;
    slice_col_par = blockIdx.x % n_tiles;
    slice_col = slice_col_par;
    slice_iters = 0;   // 占位, 由 init_slice() + MOE 修正覆盖
  } else {
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
  //printf("here!%d,  %d, %d, %d, \n",iters, parallel, n_tiles, k_tiles);
  // Ensure that the number of tiles in each stripe is a multiple of the groupsize; this avoids an annoying special case
  // where a stripe starts in the middle of group.
//   if (group_blocks != -1)
//     iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks)); //4,

  slice_row = (iters * blockIdx.x) % k_tiles;
  slice_col_par = (iters * blockIdx.x) / k_tiles;
  slice_col = slice_col_par;
  }

  // We can easily implement parallel problem execution by just remapping indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  // Compute all information about the current slice which is required for synchronization.
  auto init_slice = [&] () {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    //printf("here!%d,  %d, %d, %d, %d \n",iters, slice_iters, slice_row, slice_col_par, k_tiles);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles) 
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  // MOE 也走 init_slice(): 代入序言的值后恰好得到 slice_iters=k_tiles。
  // 但 init_slice 的 slice_idx 数学假设块连续扫 stripe, pid_m>0 会算出负数 ->
  // 钉回 slice_idx=0/slice_count=1 (每块独立写完自己的 tile, 无归约);
  // 不活跃块直接 slice_iters=0 (全零化, 见序言注释)。
  init_slice();
  if constexpr (MOE) {
    if (moe_active) {
      slice_idx = 0;
      slice_count = 1;
    } else {
      slice_iters = 0;
    }
  }
  // MOE split-K: blockIdx.y 选 K 段, 段边界按 k-tile (128) 划分。
  // thread_k_blocks(8) % group_blocks(4/8) == 0 -> 任意 tile 边界都与量化组对齐,
  // s/z 的组索引 (thread_k_blocks*slice_row)/group_blocks 恒为整数。
  // 空段 (kb >= ke) 置 slice_iters=0: 主循环与写出全跳过 (不能提前 return,
  // 见序言的 sm_120 Heisenbug 注释)。
  if constexpr (MOE) {
    if (k_splits > 1 && moe_active) {
      int k_per = ceildiv(k_tiles, k_splits);
      int kb = blockIdx.y * k_per;
      int ke = min(kb + k_per, k_tiles);
      slice_row = kb;
      slice_iters = max(ke - kb, 0);
    }
  }

  int a_gl_stride = prob_k / 8; // stride of the A matrix in global memory
  // We typically use `constexpr` to indicate that this value is a compile-time constant
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8; // stride of an A matrix tile in shared memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8; // delta between subsequent A tiles in global memory
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o); // between subsequent accesses within a tile
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o); // between shared memory writes
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4)); // between shared memory tile reads
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16; // within a shared memory tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks); // overall size of a tile
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta); // number of shared write iterations for a tile

  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8; //32, 16, 8
  constexpr int s_sh_stage = s_sh_stride * (thread_k_blocks/ group_blocks);
  int s_gl_rd_delta = s_gl_stride * thread_k_blocks / group_blocks;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  int s_sh_wr = 2 *thread_n_blocks * (threadIdx.x / 32) + (threadIdx.x % 32);
  //int s_iter = thread_k_blocks / group_blocks;
  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / 32) + s_sh_stride * slice_col + threadIdx.x % 32;
  int s_sh_rd = s_sh_stride * (((threadIdx.x / 32) / (thread_n_blocks / 4))/4) + 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4; //from share to register
  int s_sh_rd_delta;
  if (thread_k_blocks / group_blocks > 1) {
    s_sh_rd_delta = 16;
  }
  else{
    s_sh_rd_delta = 0;
  } 
  // Precompute which thread should not read memory in which iterations; this is needed if there are more threads than
  // required for a certain tilesize or when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // To ensure that writing and reading A tiles to/from shared memory, the latter in fragment format, is fully bank
  // conflict free, we need to use a rather fancy XOR-based layout. The key here is that neither reads nor writes of 
  // the 16-byte `int4` blocks of 8 consecutive threads involve the same shared memory banks. Further, it seems (based
  // on NSight-Compute) that each warp must also write a consecutive memory segment?
  auto transform_a = [&] (int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main loop unrolls, all shared memory 
  // accesses are static, we simply precompute both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
    #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd); 
  }

  // Since B-accesses have non-constant stride they have to be computed at runtime; we break dependicies between
  // subsequent accesses with a tile by maintining multiple pointers (we have enough registers), a tiny optimization.
  const I2* B1_ptr[b_sh_wr_iters];
  const int* B2_ptr[b_sh_wr_iters];
   
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
  {
    B1_ptr[i] = B1 + b_gl_rd_delta_i * i + b_gl_rd;
    B2_ptr[i] = B2 + b_gl_rd_delta_i * i + b_gl_rd;
  }

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines. 
  
  int4* sh_a = sh;
  I2* sh_b1 = reinterpret_cast<I2*>(sh_a + stages * a_sh_stage);
  int* sh_b2 = reinterpret_cast<int*>(sh_b1 + stages * b_sh_stage);
  int4* sh_s = sh_a + stages * a_sh_stage + stages * b_sh_stage;
  int4* sh_z = sh_s + stages * s_sh_stage;

  // Register storage for double buffer of shared memory reads. 
  FragA frag_a[2][thread_m_blocks];
  I2_2 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];
  FragZ frag_z[2][4];

  // Zero accumulators.
  auto zero_accums = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next shared memory pipeline location.
  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
          &sh_a_stage[a_sh_wr_trans[i]],
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
          a_sh_wr_pred[i]
        );
      }
      I2* sh_b1_stage = sh_b1 + b_sh_stage * pipe;
      int* sh_b2_stage = sh_b2 + b_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async_stream2(&sh_b1_stage[b_sh_wr_delta * i + b_sh_wr], B1_ptr[i]);
        cp_async_stream1(&sh_b2_stage[b_sh_wr_delta * i + b_sh_wr], B2_ptr[i]);
        B1_ptr[i] += b_gl_rd_delta_o;
        B2_ptr[i] += b_gl_rd_delta_o;
      }
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
      cp_async4_pred(&sh_s_stage[s_sh_wr], &s[s_gl_rd], (threadIdx.x / 32 < thread_k_blocks / group_blocks) && (threadIdx.x % 32< s_sh_stride));
      int4* sh_z_stage = sh_z + s_sh_stage * pipe;
      cp_async4_pred(&sh_z_stage[s_sh_wr], &z[s_gl_rd], (threadIdx.x / 32 < thread_k_blocks / group_blocks) && (threadIdx.x % 32< s_sh_stride));
      s_gl_rd += s_gl_rd_delta;
    }
    // Insert a fence even when we are winding down the pipeline to ensure that waiting is also correct at this point.
    cp_async_fence();
  };
  
  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&] () {
    // We only have `stages - 2` active fetches since we are double buffering and can only issue the next fetch when
    // it is guaranteed that the previous shared memory load is fully complete (as it may otherwise be overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe into the current register buffer.
  auto fetch_to_registers = [&] (int k, int pipe) {
    int4* sh_s_stage = sh_s + s_sh_stage * pipe;
    // 注意 s_sh_rd_delta 这一项: k-tile 内每个 sub-tile (b_sh_wr_iters 个) 对应一个
    // 独立的量化组 (gs=64 + thread_k=128 时 2 组/tile), 必须按 (k % b_sh_wr_iters)
    // 前进组偏移。with_zeros 版原来漏了这一项 -> 每组 tile 的第 2+ 个组永远读
    // group 0 的 scale/zero (debug_kernel_probe.py 编码探针实测: n=64..127 全部
    // 错用 group 0)。对称版 brmoe_cuda_kernel.cu 一直是对的, 这里对齐它。
    reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd_delta * (k % b_sh_wr_iters) + s_sh_rd];
    int4* sh_z_stage = sh_z + s_sh_stage * pipe;
    reinterpret_cast<int4*>(&frag_z[k % 2])[0] = sh_z_stage[s_sh_rd_delta * (k % b_sh_wr_iters) + s_sh_rd];
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    I2* sh_b1_stage = sh_b1 + b_sh_stage * pipe;
    int* sh_b2_stage = sh_b2 + b_sh_stage * pipe;
    frag_b_quant[k % 2][0] = sh_b1_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd];
    frag_b_quant[k % 2][1][0] = sh_b2_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd];

  };

  // Execute the actual tensor core matmul of a sub-tile. 
  auto matmul = [&] (int k_mod_2) {
    int b_quant, b_quant_shift;
    int b_quant3 = 0;
    FragB frag_b0, frag_b1;
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      b_quant = frag_b_quant[k_mod_2][j/2][j%2];
      b_quant_shift = b_quant >> 6;
      frag_b0 = dequant_with_zeros(b_quant);
      scale_with_zeros(frag_b0, frag_s[k_mod_2][j], 0,frag_z[k_mod_2][j]);
      frag_b1 = dequant_with_zeros(b_quant_shift);
      scale_with_zeros(frag_b1, frag_s[k_mod_2][j], 1,frag_z[k_mod_2][j]);
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k_mod_2][i], frag_b0, frag_c[i][j][0]);
        mma(frag_a[k_mod_2][i], frag_b1, frag_c[i][j][1]);
      }
      b_quant3 |= (b_quant& 0xf000f000) >> 4*(3-j);
    }   
    frag_b0 = dequant_with_zeros(b_quant3);
    b_quant_shift = b_quant3 >> 6;
    scale_with_zeros(frag_b0, frag_s[k_mod_2][3], 0,frag_z[k_mod_2][3]);
    frag_b1 =  dequant_with_zeros(b_quant_shift);
    scale_with_zeros(frag_b1, frag_s[k_mod_2][3], 1,frag_z[k_mod_2][3]);
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      mma(frag_a[k_mod_2][i], frag_b0, frag_c[i][3][0]);
      mma(frag_a[k_mod_2][i], frag_b1, frag_c[i][3][1]);
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the number of warps while keeping the n
  // dimension of a tile reasonable, we have multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&] () {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride; 
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any unnecessary read or write iterations,
      // e.g., for two warps we write only once by warp 1 and read only once by warp 0. 
      
      #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
        #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
            #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
                #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
            sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }

        if (red_idx == 0) {
          #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
            #pragma unroll
            for (int j = 0; j < 4; j++){
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
            //   if (threadIdx.x == 0 && blockIdx.x == 0)
            //    printf("m: %d, i: %d, j: %d c_rd: %f, fragc : %f \n",m_block, i, j, c_rd[j], reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j]);
            }
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we finally have to globally reduce over
  // the results. As the striped partioning minimizes the number of such reductions and our outputs are usually rather
  // small, we perform this reduction serially in L2 cache.
  auto global_reduce = [&] (bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to maximize L2 cache utilization in this step. 
    // To do this, we write out results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
        // Interestingly, doing direct global accesses here really seems to mess up the compiler and lead to slowdowns,
        // hence we also use async-copies even though these fetches are not actually asynchronous.
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(
            &sh[c_sh_wr + c_sh_wr_delta * i],
            &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m
          );
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

      #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] += __half2float(
                reinterpret_cast<__half*>(&c_red)[j]
              );
            }
          }
          if (!last) {
            int4 c;
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<__half*>(&c)[j] = __float2half(
                reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]
              );
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;
          }
        }
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually reshuffle matrix fragments in this step,
  // the reduction above is performed in fragment layout. 
  auto write_result = [&] () {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final global write patterns
    auto write = [&] (int idx, float c0, float c1, FragS& s) {
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));
      ((half2*) sh)[idx] = res;
    };
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);
        }
        
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    
    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {      
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // split-K 写出: fp32 部分和 atomicAdd 进 C32。
  // 布局必须走 write_result 的同一套 smem 重排 —— 注意 global_reduce 的 C 寻址
  // 与 write_result 的**不一样** (stream-K 的中间部分和用 global_reduce 自写自读,
  // 置换自相抵消所以看不出来; debug_splitk 的编码探针实测二者相差一个
  // k = 2*(n//8) + 8*(n%8) 的列置换)。这里直接复用 write_result 的重排,
  // 只是把 smem 里的 half2 换成 float2 (8 fp32/槽), 末端合并写换成 atomicAdd。
  auto write_result_atomic = [&] () {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    auto write = [&] (int idx, float c0, float c1) {
      ((float2*) sh)[idx] = make_float2(c0, c1);
    };
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3]);
        }

        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        // sh 的 int4 槽 c_sh_rd 现在是 4 个 float2 (8 个 fp32)
        const float2* shf = reinterpret_cast<const float2*>(sh) + 4 * c_sh_rd;
        float* dst = C32 + 8L * c_gl_wr;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          atomicAdd(dst + 2 * j,     shf[j].x);
          atomicAdd(dst + 2 * j + 1, shf[j].y);
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // Start global fetch and register load pipelines. 
  auto start_pipes = [&] () {
    #pragma unroll
    for (int i = 0; i < stages - 1; i++)
      fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();
  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to ensure all shared memory accesses are
    // static. Note that both pipelines have even length meaning that the next iteration will always start at index 0.

    #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }      
    a_gl_rd += a_gl_rd_delta_o * stages;
    
    // Process results and, if necessary, proceed to the next column slice. While this pattern may not be the most
    // readable, other ways of writing the loop seemed to noticeably worse performance after compliation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      thread_block_reduce();
      if (slice_count > 1) { // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }

      if (last) // only the last block in a slice actually writes the result
      {
       //printf("write blockIdx.x : %d \n", blockIdx.x);
        if constexpr (MOE) {
          if (C32) write_result_atomic(); else write_result();
        } else {
          write_result();
        }
      }        
      if constexpr (MOE) {
        slice_iters = 0;      // MOE: 单 slice, 写完即退出主循环
      } else {
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      }
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
        #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
        {
          B1_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
          B2_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        }
          
        if (slice_col == 0) {
          #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++){
            B1_ptr[i] -= b_gl_stride;
            B2_ptr[i] -= b_gl_stride;
          }
        }
        s_gl_rd = s_gl_stride * (threadIdx.x / 32) + s_sh_stride * slice_col + threadIdx.x % 32;
        start_pipes();
      }
    }
  }
}


// 8 warps are a good choice since every SM has 4 schedulers and having more than 1 warp per schedule allows some more
// latency hiding. At the same time, we want relatively few warps to have many registers per warp and small tiles.
const int THREADS = 256;
const int STAGES = 4; // 4 pipeline stages fit into shared memory
const int SHARED_MEM = 96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    cudaFuncSetAttribute( \
      brmoeWithZeros<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, \
      SHARED_MEM \
    ); \
    brmoeWithZeros\
    <THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS \
    ><<<blocks, THREADS, SHARED_MEM, stream>>>( \
      A_ptr, B1_ptr, B2_ptr, C_ptr, s_ptr, z_ptr, \
      prob_m, prob_n, prob_k, \
      locks \
    ); \
  }

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int brmoe_cuda_with_zeros(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
        void* z,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      // For small batchizes, better partioning is slightly more important than better compute utilization
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const I2* B1_ptr = (const I2*) B1;
  const int* B2_ptr = (const int*) B2;

  //printf("%d\n", *B2_ptr);

  int4* C_ptr = (int4*) C;
  const int4* s_ptr = (const int4*) s;
  const int4* z_ptr = (const int4*) z;

  int cols = prob_n / thread_n;
  int* locks = (int*) workspace;
  
  int mem_constrain = 16 / thread_k_blocks; //4,8,16
  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += mem_constrain) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > mem_constrain) {
      par = (16 * thread_m_blocks - pad) / (16 * mem_constrain);
      if (par > max_par)
        par = max_par;
      prob_m = 16 * mem_constrain * par;
      i += mem_constrain * (par - 1);
      thread_m_blocks = mem_constrain;
    }
    
    // For compilation speed, we only define the kernel configurations that have seemed useful (in terms of performance)
    // in our testing, however many more are, in principle, possible.
    if (false) {}
    CALL_IF(1,  8,  8,  4)
    CALL_IF(2,  8,  8,  4)
    CALL_IF(1,  4,  16,  4)
    CALL_IF(1, 16,  4,  4)
    CALL_IF(2, 16,  4,  4)
    CALL_IF(3, 16,  4,  4)
    CALL_IF(4, 16,  4,  4)
    CALL_IF(1,  8,  8,  8)
    CALL_IF(2,  8,  8,  8)
    CALL_IF(1,  4, 16,  8)
    CALL_IF(1, 16,  4,  8)
    CALL_IF(2, 16,  4,  8)
    CALL_IF(3, 16,  4,  8)
    CALL_IF(4, 16,  4,  8)
    else
      ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }

  return ret;
}

// ---------------------------------------------------------------------------
// grouped MoE 版启动器: 一块 = (16 行 m-tile, 一个 n-tile, 可选 split-K 段), 无 stream-K。
// A/C 都在 sorted 空间 (gather/scatter 由调用方用 torch 算子做, CUDA Graph 安全)。
// align 必须用 slot=16 (= 一个 m-tile 的行数), 保证一个 m-tile 只属一个专家。
//
// 注意: 必须 thread_m_blocks=1 (16 行 tile)。实测 (debug_moe_cuda.py, 38980)
// 这个 kernel 的 64 行大 tile 配置 (4,16,4,4) 在 sm_120 上 illegal instruction,
// 与 MOE 改动无关 (MOE=false 也炸); 小 tile 配置两卡都正常。16 行粒度对 decode
// 也更合适 —— padding 浪费比 64 行少 4 倍, 与 Triton 路径的 slot=16 一致。
// ---------------------------------------------------------------------------
int brmoe_moe_with_zeros(
  const void* A,            // [m_blocks_max*16, K] fp16, sorted 空间 (pad 行任意)
  const void* B1,           // [E, ...] 每专家 Marlin 布局
  const void* B2,
        void* C,            // [m_blocks_max*16, N] fp16, sorted 空间
  const void* s,            // [E, ...]
  const void* z,
  int prob_n,
  int prob_k,
  const void* expert_ids,   // [m_blocks_max] int32, device
  const void* num_post_ptr, // [1] int32, device
  int m_blocks_max,
  long b1_e_stride,         // 单位: I2
  long b2_e_stride,         // 单位: int
  long s_e_stride,          // 单位: int4
  long z_e_stride,
  int groupsize,
  int k_splits = 1,         // split-K 段数 (grid.y); >1 时 C32 必填且预清零
  void* C32 = nullptr,      // [m_blocks_max*16, N] fp32 部分和
  int thread_n = 128,       // n-tile = 16*tnb (默认 128)
  int thread_k = 128,       // k-tile = 16*tkb (默认 128)
  int stages = 4,           // cp.async 流水级数 (默认 4; 减级省 smem 换占用率)
  int dev = 0,
  cudaStream_t stream = 0
) {
  // (1,8,8) = 原 kernel 在 sm_120 上实测能跑的配置 (debug_moe_cuda.py step 0)。
  // 注意: thread_n_blocks=16 系 ((4,16,4)/(1,16,4)) 在 sm_120 上 illegal
  // instruction (38980/38982), 与 MOE 改动无关 -> 下面的配置表不收 16 系。
  // thread_m_blocks 必须 = 1 (16 行 tile, 见上方注释)。
  const int thread_n_blocks = thread_n / 16, thread_k_blocks = thread_k / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  if (group_blocks != 4 && group_blocks != 8)
    return ERR_KERN_SHAPE;
  // k-tile 必须覆盖至少一个量化组, 否则 s/z 的加载谓词 (tid/32 < tkb/gb) 全假
  if (thread_k_blocks < group_blocks)
    return ERR_KERN_SHAPE;
  if (prob_n % (16 * thread_n_blocks) != 0 || prob_k % (16 * thread_k_blocks) != 0)
    return ERR_PROB_SHAPE;

  if (k_splits < 1 || (k_splits > 1 && C32 == nullptr))
    return ERR_PROB_SHAPE;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  dim3 blocks(m_blocks_max * n_tiles, k_splits);   // y 维 = split-K 段

  // Opt-in experiment: keep the exact shared-memory layout, but reserve only
  // its actual footprint. Previously every stages/tile configuration reserved
  // 96 KiB, so reducing stages could not improve shared-memory occupancy.
  // Units below match a_sh_stage / b_sh_stage / s_sh_stage in brmoeWithZeros.
  const char* smem_mode = std::getenv("BRMOE_MOE_SMEM");
  const bool rightsize = smem_mode && std::strcmp(smem_mode, "rightsize") == 0;
  const int a_stage = (16 * thread_k_blocks / 8) * 16;
  const int b_stage = (32 * thread_n_blocks / 4) * thread_k_blocks;
  const int sz_stage = (16 * thread_n_blocks / 8) * (thread_k_blocks / group_blocks);
  const int pipeline_bytes = stages * (a_stage + b_stage + 2 * sz_stage) * sizeof(int4);
  // Conservative bound for thread_block_reduce and FP32 split-K epilogue,
  // which reuse the same storage after the async pipeline has drained.
  const int scratch_bytes = THREADS * 8 * sizeof(int4);
  const int launch_smem = rightsize ? std::max(pipeline_bytes, scratch_bytes) : SHARED_MEM;
  if (launch_smem > SHARED_MEM) return ERR_KERN_SHAPE;

  auto launch = [&](auto kfn) {
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);
    kfn<<<blocks, THREADS, launch_smem, stream>>>(
      (const int4*) A, (const I2*) B1, (const int*) B2, (int4*) C,
      (const int4*) s, (const int4*) z,
      /*prob_m=*/0, prob_n, prob_k,
      /*locks=*/nullptr,
      (const int*) expert_ids, (const int*) num_post_ptr,
      b1_e_stride, b2_e_stride, s_e_stride, z_e_stride,
      k_splits, (float*) C32
    );
    return 0;
  };

  // 配置表 (tnb, tkb, stages)。sm_120 上每个新配置必须先过 sweep 脚本的
  // 数值冒烟 (该卡有 illegal-instruction 前科, 见上方注释)。
  #define MOE_TRY(TNB, TKB, STG) \
    if (thread_n_blocks == TNB && thread_k_blocks == TKB && stages == STG) { \
      if (group_blocks == 4) \
        return launch(brmoeWithZeros<THREADS, 1, TNB, TKB, STG, 4, true>); \
      else \
        return launch(brmoeWithZeros<THREADS, 1, TNB, TKB, STG, 8, true>); \
    }
  MOE_TRY(8, 8, 4)     // 默认 (原配置)
  MOE_TRY(8, 8, 3)     // 减一级流水: smem 96K->~72K, A100 上 1->2 block/SM
  MOE_TRY(8, 8, 5)     // 加深流水
  MOE_TRY(4, 8, 4)     // 64 列 n-tile: block 数 x2
  MOE_TRY(8, 4, 4)     // 64 宽 k-tile: 每块 K 链更短
  #undef MOE_TRY
  return ERR_KERN_SHAPE;
}

#endif
