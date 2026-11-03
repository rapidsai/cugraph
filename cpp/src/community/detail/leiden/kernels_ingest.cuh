/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Ingest. Templated on the input arrays <IP, IX, WI> (offsets, indices, weights of the input
// CSR); included by the typed raw-CSR entry point only (community/detail/leiden/
// leiden_csr_impl.cuh). I1c (canonicalize) is in canonicalize.cuh.
//
//   I1   ingest_check      error flags, row-order check (parallel edges), nnz_c (counted
//                          entries), wmax and the order-free symmetry fingerprints of the
//                          counted entries; one sync, before the workspace is sized
//   I2   quantize_degrees  int64 offsets (widened), int32 indices (narrowed from int64
//                          input), W0q when fp32 on the fly is not possible, k_hat, 2m_hat
//                          (exact int64) and the degree classes of level 0
//   I4   replicate_union   replica union graph (R > 1: small graphs run R disjoint copies in
//                          the first iteration and keep the best)
//
// Weight semantics: for a weighted graph the value of a stored entry is the edge weight. Without
// weights every entry has the value 1.0f and the weighted path runs unchanged, so an unweighted
// graph is bitwise equal to one with all weights 1. An entry (v, u) is counted iff u != v and its
// value is > 0: self-loops and zero-weight edges are ignored.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/driver.hpp"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace cugraph::detail::leiden_engine {

constexpr u64 kUnitBits = 0x3F800000ull;  // IEEE bits of 1.0f

// ---------------------------------------------------------------------------
// Effective entry values
// ---------------------------------------------------------------------------

// Value of stored entry j: counted-positive test, IEEE bits for the
// fingerprint and the quantised value. The diagonal is excluded by callers.
template <typename WI>
struct EntryValue {
  const WI* data;  // null: no data (every entry has the value 1.0f)
  bool weighted;   // use_weights && data
  __device__ __forceinline__ bool positive(i64 j) const
  {
    if (weighted) return data[j] > WI(0);
    return !data || data[j] != WI(0);
  }
  __device__ __forceinline__ u64 bits(i64 j) const
  {
    return weighted ? weight_bits(data[j]) : kUnitBits;
  }
  // max(1, rint(w 2^s)), exact; 0 if the entry is not positive
  __device__ __forceinline__ i64 quantized(i64 j, int s) const
  {
    if (weighted) return quantize_scaled(static_cast<double>(data[j]), s);
    return positive(j) ? quantize_scaled(1.0, s) : 0;
  }
};

// ---------------------------------------------------------------------------
// I1: ingest_check (warp per row; block reduction, one atomic per block)
// ---------------------------------------------------------------------------

struct ScanPartial {
  u32 flags   = 0;
  u64 wmax    = 0;
  u64 h_fwd   = 0;
  u64 h_rev   = 0;
  i64 counted = 0;
};

__device__ __forceinline__ void flush_scan_partial(ScanPartial p, Control* ctl)
{
  __shared__ u32 s_flags[kWarpsPerBlock];
  __shared__ u64 s_wmax[kWarpsPerBlock], s_hf[kWarpsPerBlock], s_hr[kWarpsPerBlock];
  __shared__ i64 s_cnt[kWarpsPerBlock];
  p.flags        = warp_or(p.flags);
  p.wmax         = warp_max(p.wmax);
  p.h_fwd        = warp_sum(p.h_fwd);
  p.h_rev        = warp_sum(p.h_rev);
  p.counted      = warp_sum(p.counted);
  const int lane = threadIdx.x & (kWarp - 1), w = threadIdx.x / kWarp;
  if (lane == 0) {
    s_flags[w] = p.flags;
    s_wmax[w]  = p.wmax;
    s_hf[w]    = p.h_fwd;
    s_hr[w]    = p.h_rev;
    s_cnt[w]   = p.counted;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ScanPartial b;
    for (int i = 0; i < kWarpsPerBlock; ++i) {
      b.flags |= s_flags[i];
      b.wmax = s_wmax[i] > b.wmax ? s_wmax[i] : b.wmax;
      b.h_fwd += s_hf[i];
      b.h_rev += s_hr[i];
      b.counted += s_cnt[i];
    }
    if (b.flags) atomicOr(&ctl->flags, b.flags);
    if (b.wmax) atomicMax(&ctl->wmax_bits, b.wmax);
    atomicAdd(&ctl->h_fwd, b.h_fwd);
    atomicAdd(&ctl->h_rev, b.h_rev);
    atomicAdd(reinterpret_cast<u64*>(&ctl->n_counted), static_cast<u64>(b.counted));
  }
}

template <typename IP, typename IX, typename WI>
__global__ void ingest_scan_kernel(const IP* __restrict__ indptr,
                                   const IX* __restrict__ indices,
                                   EntryValue<WI> val,
                                   i64 n,
                                   i64 nnz,
                                   Control* ctl)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  ScanPartial p;
  for (i64 v = warp0; v < n; v += nwarps) {
    const i64 b = static_cast<i64>(indptr[v]);
    const i64 e = static_cast<i64>(indptr[v + 1]);
    if (lane == 0) {
      if (e < b || (v == 0 && b != 0) || (v == n - 1 && e != nnz)) p.flags |= kFlagBadIndptr;
      if (e - b >= (1ll << 31)) p.flags |= kFlagRowTooLong;
    }
    if (e < b || b < 0 || e > nnz) continue;  // warp-uniform guard
    for (i64 j = b + lane; j < e; j += kWarp) {
      const i64 u = static_cast<i64>(indices[j]);
      if (u < 0 || u >= n) {
        p.flags |= kFlagBadIndex;
        continue;
      }
      if (j > b && static_cast<i64>(indices[j - 1]) >= u) p.flags |= kFlagNonCanonical;
      if (val.weighted) {
        const WI w = val.data[j];
        if (!isfinite(w))
          p.flags |= kFlagNonFinite;
        else if (w < WI(0))
          p.flags |= kFlagNegative;
      }
      if (u == v) continue;
      if (!val.positive(j)) {
        if (!val.weighted) p.flags |= kFlagUnitZero;
        continue;
      }
      const u64 bits = val.bits(j);
      if (bits > p.wmax) p.wmax = bits;
      fingerprint_terms(v, u, bits, p.h_fwd, p.h_rev);
      ++p.counted;
    }
  }
  flush_scan_partial(p, ctl);
}

inline double value_from_bits(u64 bits, bool fp32)
{
  if (fp32) {
    const u32 b = static_cast<u32>(bits);
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return static_cast<double>(f);
  }
  double d;
  std::memcpy(&d, &bits, sizeof(d));
  return d;
}

inline void check_structure_flags(u32 flags)
{
  CUGRAPH_EXPECTS(!(flags & kFlagBadIndptr),
                  "leiden: offsets must start at 0, be "
                  "non-decreasing and end at the number of edges.");
  CUGRAPH_EXPECTS(!(flags & kFlagRowTooLong),
                  "leiden: vertices with >= 2^31 stored edges "
                  "are not supported.");
  CUGRAPH_EXPECTS(!(flags & kFlagBadIndex), "leiden: vertex index out of range [0, n).");
}

// I1 on stream `s` with `ctl` (>= kControlBytes of device scratch); one sync
// (S1). Structural errors throw; weight errors, parallel edges (non-canonical
// rows) and asymmetry are returned as flags for the caller to report.
template <typename IP, typename IX, typename WI>
IngestInfo run_ingest_check(const IP* indptr,
                            const IX* indices,
                            const WI* data,
                            bool has_data,
                            i64 n,
                            i64 nnz,
                            bool use_weights,
                            Control* ctl,
                            void* pinned,
                            cudaStream_t s)
{
  CUGRAPH_EXPECTS(n >= 0 && n < kMaxVertices,
                  "leiden: the number of vertices must be in [0, 2^30).");
  CUGRAPH_EXPECTS(n > 0 || nnz == 0, "leiden: a graph without vertices has no edges.");
  const EntryValue<WI> val{has_data ? data : nullptr, use_weights && has_data};
  clear_control(ctl, s);
  if (n > 0) {
    ingest_scan_kernel<IP, IX, WI>
      <<<grid_occ(ingest_scan_kernel<IP, IX, WI>, n, kWarpsPerBlock), kBlock, 0, s>>>(
        indptr, indices, val, n, nnz, ctl);
    RAFT_CHECK_CUDA(s);
  }
  const Control h = read_control(ctl, s, pinned);
  IngestInfo info;
  info.flags     = h.flags;
  info.wmax_bits = h.wmax_bits;
  info.n_counted = h.n_counted;
  info.h_fwd     = h.h_fwd;
  info.h_rev     = h.h_rev;
  check_structure_flags(info.flags);
  CUGRAPH_EXPECTS(info.n_counted < kMaxCountedEntries,
                  "leiden: graphs with >= 2^40 counted edges "
                  "are not supported.");
  if (info.h_fwd != info.h_rev) info.flags |= kFlagAsymmetric;
  info.wmax = value_from_bits(info.wmax_bits, !val.weighted || std::is_same_v<WI, float>);
  return info;
}

// ---------------------------------------------------------------------------
// I2: quantize_degrees (warp per row)
// ---------------------------------------------------------------------------

// Sources of level-0 quantised weights; `off` = (u != v).
struct QSrcF32 {  // fp32 on the fly (-126 <= s <= 127)
  const float* w;
  float scale;
  __device__ __forceinline__ i64 operator()(i64 j, bool off) const
  {
    return off ? quantize_f32(w[j], scale) : 0;
  }
};
template <typename WI>
struct QSrcMat {  // materialises W0q (0 for entries that are not counted)
  EntryValue<WI> val;
  int s;
  i64* out;
  __device__ __forceinline__ i64 operator()(i64 j, bool off) const
  {
    const i64 q = off ? val.quantized(j, s) : 0;
    out[j]      = q;
    return q;
  }
};
struct QSrcUnit {
  i64 q;
  __device__ __forceinline__ i64 operator()(i64, bool off) const { return off ? q : 0; }
};

// Also widens the offsets to int64 and narrows int64 indices to the int32
// level-0 copy (idx32 != null).
template <typename IP, typename IX, typename Src>
__global__ void quantize_degrees_kernel(const IP* __restrict__ indptr,
                                        const IX* __restrict__ indices,
                                        Src src,
                                        i64 n,
                                        ClassThresholds th,
                                        i64* __restrict__ indptr64,
                                        int* __restrict__ idx32,
                                        i64* __restrict__ khat,
                                        Control* ctl)
{
  __shared__ i64 s_two_m[kWarpsPerBlock];
  __shared__ i64 s_cls[kWarpsPerBlock][kNumClasses];
  const int lane = threadIdx.x & (kWarp - 1), wib = threadIdx.x / kWarp;
  const i64 warp0      = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + wib;
  const i64 nwarps     = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  i64 two_m            = 0;
  i64 cls[kNumClasses] = {};
  for (i64 v = warp0; v < n; v += nwarps) {
    const i64 b = static_cast<i64>(indptr[v]);
    const i64 e = static_cast<i64>(indptr[v + 1]);
    i64 k       = 0;
    for (i64 j = b + lane; j < e; j += kWarp) {
      const i64 u = static_cast<i64>(indices[j]);
      if (idx32) idx32[j] = static_cast<int>(u);
      k += src(j, u != v);
    }
    k = warp_sum(k);
    if (lane == 0) {
      const int c = degree_class(e - b, th);
      indptr64[v] = b;
      if (v == n - 1) indptr64[n] = e;
      khat[v] = k;
      two_m += k;
      ++cls[c];
    }
  }
  if (lane == 0) {
    s_two_m[wib] = two_m;
    for (int c = 0; c < kNumClasses; ++c)
      s_cls[wib][c] = cls[c];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    i64 t = 0, cc[kNumClasses] = {};
    for (int i = 0; i < kWarpsPerBlock; ++i) {
      t += s_two_m[i];
      for (int c = 0; c < kNumClasses; ++c)
        cc[c] += s_cls[i][c];
    }
    atomicAdd(reinterpret_cast<u64*>(&ctl->two_m_hat), static_cast<u64>(t));
    for (int c = 0; c < kNumClasses; ++c)
      if (cc[c]) atomicAdd(reinterpret_cast<u64*>(&ctl->class_count[c]), static_cast<u64>(cc[c]));
  }
}

// ---------------------------------------------------------------------------
// I4: replicate_union (warp per union row)
// ---------------------------------------------------------------------------

// Non-template kernel in a header included by both typed entry points: static
// (internal linkage), as cuGraph's other header-defined kernels.
__global__ static void replicate_union_kernel(const i64* __restrict__ indptr0,
                                              const int* __restrict__ idx0,
                                              const float* __restrict__ wf0,
                                              const i64* __restrict__ wq0,
                                              const i64* __restrict__ khat0,
                                              i64 n,
                                              int R,
                                              i64* __restrict__ indptr_u,
                                              int* __restrict__ idx_u,
                                              float* __restrict__ wf_u,
                                              i64* __restrict__ wq_u,
                                              i64* __restrict__ khat_u)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const i64 nnz0   = indptr0[n];
  const i64 rows   = static_cast<i64>(R) * n;
  for (i64 x = warp0; x < rows; x += nwarps) {
    const i64 r = x / n, v = x - r * n;
    const i64 b = indptr0[v], e = indptr0[v + 1], shift = r * nnz0;
    if (lane == 0) {
      indptr_u[x] = shift + b;
      if (x == rows - 1) indptr_u[rows] = static_cast<i64>(R) * nnz0;
      khat_u[x] = khat0[v];
    }
    for (i64 j = b + lane; j < e; j += kWarp) {
      idx_u[shift + j] = static_cast<int>(r * n) + idx0[j];
      if (wf_u) wf_u[shift + j] = wf0[j];
      if (wq_u) wq_u[shift + j] = wq0[j];
    }
  }
}

// ---------------------------------------------------------------------------
// I2 (+ I4) for one resolution
// ---------------------------------------------------------------------------

inline void check_gamma(double gamma)
{
  CUGRAPH_EXPECTS(gamma >= 0.0 && gamma <= kGammaMax,
                  "leiden: resolution must be finite and in [0, 2^20].");
}

// I2 (+ I4) for one scale; the input passed I1 (`info`) without any flag the caller rejects.
template <typename IP, typename IX, typename WI>
QuantizeResult run_quantize(const IP* indptr,
                            const IX* indices,
                            const WI* data,
                            bool has_data,
                            i64 n,
                            i64 nnz,
                            bool use_weights,
                            double gamma,
                            const IngestInfo& info,
                            Layout& L,
                            const ClassThresholds& th,
                            cudaStream_t stream)
{
  check_gamma(gamma);
  CUGRAPH_EXPECTS(!(info.flags & (kFlagNonCanonical | kFlagAsymmetric)),
                  "leiden: the graph must be symmetric and "
                  "without parallel edges.");
  CUGRAPH_EXPECTS(!(info.flags & (kFlagNegative | kFlagNonFinite)),
                  "leiden: edge weights must be finite and "
                  "non-negative.");
  CUGRAPH_EXPECTS(info.n_counted >= 0 && info.n_counted < kMaxCountedEntries,
                  "leiden: counted edges out of range.");
  CUGRAPH_EXPECTS(n == L.p.n && nnz == L.p.nnz, "leiden: layout does not match the graph.");
  QuantizeResult res;
  constexpr bool idx_is_32 = std::is_same_v<IX, int>;
  const bool weighted      = use_weights && has_data;
  const EntryValue<WI> val{has_data ? data : nullptr, weighted};
  res.s      = scale_for_gamma(scale_s0(info.n_counted, info.wmax), gamma);
  res.unit_q = weighted ? 0 : unit_weight(res.s);
  res.wkind  = level0_kind(info.flags, weighted, std::is_same_v<WI, float>, res.s);
  CUGRAPH_EXPECTS(
    (res.wkind != WKind::I64 || L.l0.wq != nullptr) && (idx_is_32 || L.l0.indices != nullptr),
    "leiden: internal error: the workspace layout lacks a level-0 array.");
  Control* ctl = L.ctl;
  clear_control(ctl, stream);
  int* idx32  = idx_is_32 ? nullptr : L.l0.indices;
  res.indptr  = L.l0.indptr;
  res.indices = idx_is_32 ? reinterpret_cast<const int*>(indices) : idx32;
  res.wf32    = res.wkind == WKind::F32 ? reinterpret_cast<const float*>(data) : nullptr;
  res.wq      = res.wkind == WKind::I64 ? L.l0.wq : nullptr;

  // I2
  if (n > 0) {
    auto launch = [&](auto src) {
      using Src        = decltype(src);
      const unsigned g = grid_occ(quantize_degrees_kernel<IP, IX, Src>, n, kWarpsPerBlock);
      quantize_degrees_kernel<IP, IX, Src>
        <<<g, kBlock, 0, stream>>>(indptr, indices, src, n, th, L.l0.indptr, idx32, L.l0.khat, ctl);
      RAFT_CHECK_CUDA(stream);
    };
    if (res.wkind == WKind::UNIT) {
      launch(QSrcUnit{res.unit_q});
    } else if (res.wkind == WKind::I64) {
      launch(QSrcMat<WI>{val, res.s, L.l0.wq});
    } else if constexpr (std::is_same_v<WI, float>) {
      launch(QSrcF32{data, std::ldexp(1.0f, res.s)});
    }
  } else {
    RAFT_CUDA_TRY(cudaMemsetAsync(L.l0.indptr, 0, sizeof(i64), stream));
  }

  // I4: replica union (iteration 1 only)
  if (L.p.replicas > 1 && n > 0) {
    replicate_union_kernel<<<grid_rows(L.p.n_union()), kBlock, 0, stream>>>(
      L.l0.indptr,
      res.indices,
      res.wf32,
      res.wq,
      L.l0.khat,
      n,
      L.p.replicas,
      L.uni.indptr,
      L.uni.indices,
      res.wkind == WKind::F32 ? L.uni.wf32 : nullptr,
      res.wkind == WKind::I64 ? L.uni.wq : nullptr,
      L.uni.khat);
    RAFT_CHECK_CUDA(stream);
  }

  const Control h = read_control(ctl, stream, L.pinned);  // sync S2
  res.two_m_hat   = h.two_m_hat;
  for (int c = 0; c < kNumClasses; ++c)
    res.class_count[c] = h.class_count[c];
  return res;
}

}  // namespace cugraph::detail::leiden_engine
