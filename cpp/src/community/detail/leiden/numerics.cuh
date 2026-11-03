/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Shared numerics of the Leiden engine (detail::leiden_engine).
//
// Everything that feeds a decision or an output is either an exact integer or an explicitly
// rounded (`_rn`) IEEE binary64 operation, so the results do not depend on thread order, launch
// geometry or the compiler's contraction choices:
//
//   - edge and vertex weights are quantised once to int64 fixed point (quantize_*);
//   - the modularity penalty lambda * k_v * K_c is an exact floor of a 64 x 64 -> 128-bit
//     product (Mult, pen);
//   - every pseudo-random choice is a counter-based hash of (seed, iteration, level, phase,
//     sweep, vertex) (ctx, order_f), never an RNG stream;
//   - the only floating-point reduction (Q) is a fixed-order tree (tree1024_finish).

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cugraph::detail::leiden_engine {

// The engine's fixed-width types. 64-bit values are (unsigned) long long rather than
// (u)int64_t (long on LP64) because the CUDA atomics, shuffles and conversion intrinsics the
// kernels use are declared for (unsigned) long long; the representation is identical.
using i64 = long long;
using u64 = unsigned long long;
using u32 = unsigned int;

constexpr int kWarp          = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kBlock         = 256;
constexpr int kWarpsPerBlock = kBlock / kWarp;
constexpr int kTreeThreads   = 1024;  // TREE1024
constexpr int kMaxReplicas   = 4;
constexpr int kMaxLevels     = 64;
constexpr i64 kMaxVertices   = 1ll << 30;  // R * n < 2^30
constexpr int kNumSubrounds  = 4;

// Level-0 weights. F32: fp32 input quantised on the fly. I64: materialised W0q
// (fp64 input, an fp32 scale outside [-126, 127], or unit weights with explicit
// zeros; uncounted entries hold 0). UNIT: the quantised 1.0 (no weights).
enum class WKind : int { F32 = 0, I64 = 1, UNIT = 2 };

// max(1, rint(w 2^s)) of the exact product, half to even; 0 if w <= 0. The
// fp32 product is exact for -126 <= s <= 127 (smaller results map to 1).
__device__ __forceinline__ i64 quantize_f32(float w, float scale)
{
  if (!(w > 0.0f)) return 0;
  const i64 q = __float2ll_rn(__fmul_rn(w, scale));
  return q > 0 ? q : 1;
}

// Exact for any s; the result is <= 2^58.
__device__ __forceinline__ i64 quantize_scaled(double w, int s)
{
  if (!(w > 0.0)) return 0;
  const i64 q = __double2ll_rn(scalbn(w, s));
  return q > 0 ? q : 1;
}

// Quantised weight of entry j (0: not counted); cs(j) is a streaming load.
template <WKind K>
struct EdgeW;

template <>
struct EdgeW<WKind::F32> {
  const float* w;
  float scale;
  __device__ __forceinline__ i64 operator()(i64 j) const { return quantize_f32(w[j], scale); }
  __device__ __forceinline__ i64 cs(i64 j) const { return quantize_f32(__ldcs(&w[j]), scale); }
};

template <>
struct EdgeW<WKind::I64> {
  const i64* w;
  __device__ __forceinline__ i64 operator()(i64 j) const
  {
    const i64 x = w[j];
    return x > 0 ? x : 0;
  }
  __device__ __forceinline__ i64 cs(i64 j) const
  {
    const i64 x = __ldcs(&w[j]);
    return x > 0 ? x : 0;
  }
};

template <>
struct EdgeW<WKind::UNIT> {
  i64 q;
  __device__ __forceinline__ i64 operator()(i64) const { return q; }
  __device__ __forceinline__ i64 cs(i64) const { return q; }
};

constexpr int kScaleBudgetLog2   = 58;  // 2m_hat <= 2^58 (1 + 2^-27)
constexpr int kF32ScaleMin       = -126;
constexpr int kF32ScaleMax       = 127;
constexpr double kGammaHeadroom  = 16.0;
constexpr double kGammaMax       = 1048576.0;  // gamma <= 2^20
constexpr i64 kMaxCountedEntries = 1ll << 40;

// s0 = e - 1 with (f, e) = frexp(2^58 / (nnz_c wmax)); split exponents keep
// nnz_c wmax = f_w 2^(e_w + e_m) and its inverse from overflowing.
inline int scale_s0(i64 nnz_c, double wmax)
{
  if (!(nnz_c > 0 && wmax > 0.0)) return 0;
  int e_m = 0, e_w = 0, e_q = 0;
  const double f_m = std::frexp(wmax, &e_m);
  const double f_w = std::frexp(static_cast<double>(nnz_c) * f_m, &e_w);
  std::frexp(1.0 / f_w, &e_q);  // 1 / f_w in (1, 2]: e_q in {1, 2}
  return kScaleBudgetLog2 - (e_w + e_m) + (e_q - 1);
}

// h(gamma) = ceil(log2(gamma / 16)) for gamma > 16, else 0; s = s0 - h keeps
// every penalty below 2^63.
inline int scale_headroom(double gamma)
{
  if (!(gamma > kGammaHeadroom)) return 0;
  int e          = 0;
  const double f = std::frexp(gamma / kGammaHeadroom, &e);
  return (f == 0.5) ? e - 1 : e;
}

inline int scale_for_gamma(int s0, double gamma) { return s0 - scale_headroom(gamma); }

// max(1, rint(2^s)); s <= 58 for unit weights.
inline i64 unit_weight(int s) { return s >= 0 ? (1ll << (s < 62 ? s : 62)) : 1; }

// Integer penalty pen(A, B) = floor(B c(A)) with c(A) = lam A.

struct Mult {
  u64 M;  // c(A) = M * 2^-(64 + sh); M in [2^63, 2^64) or 0
  int sh;
};

__host__ __device__ __forceinline__ Mult make_mult(i64 a, double lam)
{
  Mult m{0ull, 0};
#ifdef __CUDA_ARCH__
  const double c = __dmul_rn(lam, __ll2double_rn(a));
#else
  const double c = lam * static_cast<double>(a);
#endif
  if (c == 0.0) return m;
  int e          = 0;
  const double f = frexp(c, &e);
#ifdef __CUDA_ARCH__
  m.M = __double2ull_rn(ldexp(f, 64));
#else
  m.M = static_cast<u64>(ldexp(f, 64));
#endif
  m.sh = -e;
  return m;
}

// floor(B * M / 2^(64 + sh)) from the exact 128-bit product.
__host__ __device__ __forceinline__ i64 pen(i64 b, Mult m)
{
  const u64 ub = static_cast<u64>(b);
#ifdef __CUDA_ARCH__
  const u64 hi = __umul64hi(ub, m.M);
#else
  const u64 hi = static_cast<u64>((static_cast<unsigned __int128>(ub) * m.M) >> 64);
#endif
  if (m.sh >= 0) return m.sh >= 64 ? 0 : static_cast<i64>(hi >> m.sh);
  const u64 lo = ub * m.M;
  const int l  = -m.sh;  // 0 < l < 64 by the headroom bound
  return static_cast<i64>((hi << l) | (lo >> (64 - l)));
}

__host__ __device__ __forceinline__ u64 mix64(u64 x)
{
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return x;
}

__host__ __device__ __forceinline__ u32 fmix32(u32 x)
{
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}

enum : u32 { kTagMove = 1, kTagOrder = 2 };
enum : u32 { kPhaseDown = 0, kPhaseUp = 1, kPhaseTop = 2 };

__host__ __device__ __forceinline__ u64 seed64(u32 seed)
{
  return mix64(static_cast<u64>(seed) + 0x9E3779B97F4A7C15ull);
}

// Field widths: it < 2^16, l < 2^8, phase < 2^8, sweep < 2^24.
__host__ __device__ __forceinline__ u64 ctx(u64 s64, u32 tag, u32 it, u32 l, u32 phase, u32 sweep)
{
  return mix64(s64 ^ ((static_cast<u64>(tag) << 56) | (static_cast<u64>(it) << 40) |
                      (static_cast<u64>(l) << 32) | (static_cast<u64>(phase) << 24) |
                      static_cast<u64>(sweep)));
}

// Bijective order hash of v (ctx_order = ctx(seed64, ORDER, it, l, 0, 0)).
__host__ __device__ __forceinline__ u32 order_f(u64 ctx_order, int v)
{
  return fmix32(static_cast<u32>(v) ^ static_cast<u32>(ctx_order));
}

__host__ __device__ __forceinline__ i64 order_key(u32 f, bool flipped)
{
  return flipped ? (1ll << 33) - static_cast<i64>(f) : static_cast<i64>(f);
}

__device__ __forceinline__ u64 weight_bits(float w) { return static_cast<u64>(__float_as_uint(w)); }
__device__ __forceinline__ u64 weight_bits(double w)
{
  return static_cast<u64>(__double_as_longlong(w));
}
__device__ __forceinline__ u64 weight_bits(i64 w) { return static_cast<u64>(w); }

// Order-free symmetry fingerprint terms of counted entry (v, u, bits).
__device__ __forceinline__ void fingerprint_terms(i64 v, i64 u, u64 bits, u64& fwd, u64& rev)
{
  const u64 hb = mix64(bits);
  fwd += mix64(((static_cast<u64>(v) << 32) | static_cast<u64>(u)) ^ hb);
  rev += mix64(((static_cast<u64>(u) << 32) | static_cast<u64>(v)) ^ hb);
}

// TREE1024, the fixed-order fp64 sum of Q: thread j's ascending strided partial
// ((0 + t_j) + t_{j+1024}) + ..., then a pairwise tree; every thread gets it.
__device__ __forceinline__ double tree1024_finish(double acc, double* smem)
{
  const int j = threadIdx.x;
  smem[j]     = acc;
  __syncthreads();
#pragma unroll 1
  for (int w = kTreeThreads / 2; w >= 1; w >>= 1) {
    if (j < w) smem[j] = __dadd_rn(smem[j], smem[j + w]);
    __syncthreads();
  }
  const double r = smem[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ double volume_term(i64 k, i64 two_m)
{
  const double x = __ddiv_rn(__ll2double_rn(k), __ll2double_rn(two_m));
  return __dmul_rn(x, x);
}

// Degree classes select the kernel mapping only, never a result.

enum : int { kClassLight = 0, kClassMid = 1, kClassBlock = 2 };
constexpr int kNumClasses = 3;

struct ClassThresholds {  // degrees above `mid`: block per vertex
  int light = 32;         // <= 32 (one entry per lane): warp, registers only
  int mid   = 128;        // <= 128 (table load <= 0.5): warp, shared table
};

__host__ __device__ __forceinline__ int degree_class(i64 deg, const ClassThresholds& t)
{
  return deg <= t.light ? kClassLight : deg <= t.mid ? kClassMid : kClassBlock;
}

template <typename T>
__device__ __forceinline__ T warp_sum(T x)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1)
    x += __shfl_xor_sync(kFullMask, x, o);
  return x;
}

template <typename T>
__device__ __forceinline__ T warp_max(T x)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) {
    const T y = __shfl_xor_sync(kFullMask, x, o);
    x         = y > x ? y : x;
  }
  return x;
}

__device__ __forceinline__ u32 warp_or(u32 x)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1)
    x |= __shfl_xor_sync(kFullMask, x, o);
  return x;
}

// Shared-memory hash tables: int keys (-1 = free) probed linearly from
// fmix32(key), int64 values, emptied by their reader after each row. Warp
// table: one pre-aggregated sum per key and 32-entry chunk (plain adds).
__device__ __forceinline__ void warp_table_add(int* keys, i64* vals, u32 mask, int key, i64 sum)
{
  volatile int* vk = keys;
  u32 h            = fmix32(static_cast<u32>(key)) & mask;
  while (true) {
    const int cur = vk[h];
    if (cur == key) {
      vals[h] += sum;
      return;
    }
    if (cur == -1) {
      if (atomicCAS(&keys[h], -1, key) == -1) {
        vals[h] = sum;
        return;
      }
      continue;  // claimed meanwhile: re-read slot h
    }
    h = (h + 1) & mask;
  }
}

// Hash-range pass of a key among npass passes (passes nest under doubling).
__device__ __forceinline__ u64 table_pass(int key, u64 npass)
{
  return (static_cast<u64>(fmix32(static_cast<u32>(key))) * npass) >> 32;
}

// Launch geometry only (never a result). Cached per host thread and device: the engine has no
// handle, and the grid helpers below run before every launch.
struct DeviceInfo {
  int device             = -1;
  int sm_count           = 1;
  int max_threads_per_sm = 2048;
};

inline const DeviceInfo& device_info()
{
  static thread_local DeviceInfo cached;
  int dev = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&dev));
  if (dev != cached.device) {
    DeviceInfo d;
    d.device = dev;
    int v    = 0;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
    d.sm_count = v;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    d.max_threads_per_sm = v;
    cached               = d;
  }
  return cached;
}

// min(resident blocks, ceil(work / items_per_block)); kernels are grid-stride.
inline unsigned grid_for(i64 work, i64 items_per_block, int block = kBlock)
{
  const DeviceInfo& d = device_info();
  const i64 resident =
    static_cast<i64>(d.sm_count) *
    static_cast<i64>(d.max_threads_per_sm / block > 0 ? d.max_threads_per_sm / block : 1);
  i64 g = (work + items_per_block - 1) / items_per_block;
  if (g > resident) g = resident;
  if (g < 1) g = 1;
  return static_cast<unsigned>(g);
}

inline unsigned grid_resident(i64 work, i64 items_per_block, int blocks_per_sm)
{
  const i64 resident =
    static_cast<i64>(device_info().sm_count) * (blocks_per_sm > 0 ? blocks_per_sm : 1);
  i64 g = (work + items_per_block - 1) / items_per_block;
  if (g > resident) g = resident;
  if (g < 1) g = 1;
  return static_cast<unsigned>(g);
}

// Resident blocks per SM of `kernel` (cached); grids are capped at residency.
template <typename Kernel>
inline int blocks_per_sm(Kernel kernel, int block, std::size_t smem = 0)
{
  struct Entry {
    int device;
    const void* f;
    int block;
    std::size_t smem;
    int nb;
  };
  static thread_local std::vector<Entry> cache;
  const int dev = device_info().device;
  const void* f = reinterpret_cast<const void*>(kernel);
  for (const Entry& e : cache)
    if (e.device == dev && e.f == f && e.block == block && e.smem == smem) return e.nb;
  int nb = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, block, smem));
  if (nb < 1) nb = 1;
  cache.push_back(Entry{dev, f, block, smem, nb});
  return nb;
}

template <typename Kernel>
inline unsigned grid_occ(
  Kernel kernel, i64 work, i64 items_per_block, int block = kBlock, std::size_t smem = 0)
{
  return grid_resident(work, items_per_block, blocks_per_sm(kernel, block, smem));
}

inline unsigned grid_rows(i64 rows)
{  // warp per row
  return grid_for(rows, kWarpsPerBlock);
}
inline unsigned grid_items(i64 items) { return grid_for(items, kBlock); }

}  // namespace cugraph::detail::leiden_engine
