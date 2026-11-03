/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Workspace layout, carved from one device buffer per call (the caller allocates
// workspace_bytes(p) bytes; nothing else is allocated): control, cub, phase (Move / Refine /
// Aggregate / Final buffers, aliased: disjoint lifetimes), persist (partitions, K_S), level0
// (offsets, int32 index copy, W0q, k_hat), union (replicas) and the arena (levels from the
// bottom, the holey contraction output from the top). A level that does not fit ends the call;
// the caller repeats it on a larger arena. The layout never reaches a decision, so every layout
// gives the bitwise identical result.

#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>

#include <algorithm>
#include <cstring>

namespace cugraph::detail::leiden_engine {

constexpr std::size_t kAlign = 256;

inline std::size_t align_up(std::size_t x, std::size_t a = kAlign) { return (x + a - 1) / a * a; }

// Control block, read back through a pinned buffer.
enum : u32 {
  kFlagNegative     = 1u << 0,  // w < 0 (weighted input)
  kFlagNonFinite    = 1u << 1,  // NaN / Inf (weighted input)
  kFlagNonCanonical = 1u << 2,  // u_j <= u_{j-1} within a row (parallel edges)
  kFlagAsymmetric   = 1u << 3,  // h_fwd != h_rev
  kFlagBadIndex     = 1u << 4,  // column index outside [0, n)
  kFlagBadIndptr    = 1u << 5,  // indptr not monotone / wrong ends
  kFlagUnitZero     = 1u << 6,  // unit weights with off-diagonal zeros
  kFlagRowTooLong   = 1u << 7,  // a row with >= 2^31 stored entries
  kFlagBadLabel     = 1u << 8,  // community id out of range
};

constexpr int kMaxSubrounds           = 64;
constexpr int kSweepChunk             = 4;  // sweeps per launch chunk
constexpr std::size_t kMoveStateBytes = 256;

struct alignas(16) Control {
  u32 flags;
  u32 pad0;
  u64 wmax_bits;  // max counted weight, IEEE bits of the input dtype
  u64 h_fwd;      // order-free symmetry fingerprints (wrapping sums)
  u64 h_rev;
  i64 n_counted;  // nnz_c
  i64 two_m_hat;
  i64 class_count[kNumClasses];
  i64 n_components;  // C after a CC split / compaction
  i64 l_hat[kMaxReplicas];
  double sum_t[kMaxReplicas];
  double q[kMaxReplicas];
  i64 n_next;
  // read back at SL2 with n_next; zeroed by the kernel that starts each
  int tree_list_count[3];    // R5: thread / warp / large trees
  int gather_list_count[3];  // A2: warp / block / wide-warp gathers
  i64 nnz_next;
  i64 coarse_class_count[kNumClasses];
  // LOCAL_MOVE results of the caller's next readback (no per-sweep sync)
  i64 move_C;    // C after COMPACT
  int move_err;  // kMoveErr* bits
  int pad1;
  // movers per (sweep of the current chunk, sub-round)
  int sweep_movers[kSweepChunk][kMaxSubrounds];
};
static_assert(sizeof(Control) <= 4096, "Control block must fit in 4 KB");
constexpr std::size_t kControlBytes = 4096;

// Readbacks go through a pinned host buffer of kPinnedBytes when the caller has one (null:
// pageable host memory, slightly slower per sync).
constexpr std::size_t kPinnedBytes = 4096;
static_assert(sizeof(Control) <= kPinnedBytes, "Control must fit kPinnedBytes");

inline Control read_control(const Control* d_ctl, cudaStream_t s, void* pinned = nullptr)
{
  Control h;
  void* dst = pinned ? pinned : static_cast<void*>(&h);
  RAFT_CUDA_TRY(cudaMemcpyAsync(dst, d_ctl, sizeof(Control), cudaMemcpyDeviceToHost, s));
  RAFT_CUDA_TRY(cudaStreamSynchronize(s));
  if (pinned) std::memcpy(&h, pinned, sizeof(Control));
  return h;
}

inline void clear_control(Control* d_ctl, cudaStream_t s)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(d_ctl, 0, sizeof(Control), s));
}

// One code path for sizing (base == nullptr) and carving.
struct Carver {
  char* base      = nullptr;
  std::size_t off = 0;

  template <typename T>
  T* take(std::size_t count)
  {
    off  = align_up(off);
    T* p = base ? reinterpret_cast<T*>(base + off) : nullptr;
    off += count * sizeof(T);
    return p;
  }
  char* bytes(std::size_t n) { return take<char>(n); }
  std::size_t used() const { return align_up(off); }
};

struct LayoutParams {
  i64 n               = 0;  // level-0 vertices (one copy)
  i64 nnz             = 0;  // stored entries of the input
  WKind wkind         = WKind::F32;
  bool idx64_input    = false;  // int64 indices: int32 level-0 copy
  int replicas        = 1;
  double arena_factor = 1.9;
  int subrounds       = kNumSubrounds;

  i64 n_union() const { return static_cast<i64>(replicas) * n; }
  i64 nnz_union() const { return static_cast<i64>(replicas) * nnz; }

  void validate() const
  {
    CUGRAPH_EXPECTS(n >= 0 && nnz >= 0, "leiden: n and nnz must be >= 0.");
    CUGRAPH_EXPECTS(replicas >= 1 && replicas <= kMaxReplicas,
                    "leiden: replicas must be in [1, 4].");
    CUGRAPH_EXPECTS(n_union() < kMaxVertices,
                    "leiden: graph too large (replicas * n must be < 2^30).");
    CUGRAPH_EXPECTS(arena_factor > 0.0, "leiden: arena_factor must be > 0.");
    CUGRAPH_EXPECTS(subrounds >= 1 && subrounds <= 64, "leiden: subrounds must be in [1, 64].");
  }
};

// Some CUB algorithms take `int num_items`.
inline int cub_items(i64 n, const char* what)
{
  CUGRAPH_EXPECTS(n >= 0 && n < (1ll << 31), "leiden: %s: CUB item count must be < 2^31.", what);
  return static_cast<int>(n);
}

// Largest temp storage of the CUB calls and the graph-friendly scans' tiles.
inline std::size_t cub_temp_bytes(i64 n_u)
{
  const i64 n2     = 2 * n_u + 2;
  std::size_t best = 0, b = 0;
  auto take = [&](cudaError_t e) {
    RAFT_CUDA_TRY(e);
    best = std::max(best, b);
    b    = 0;
  };
  int* i32  = nullptr;
  i64* i64p = nullptr;
  u64* u64p = nullptr;
  take(cub::DeviceScan::ExclusiveSum(nullptr, b, i32, i32, cub_items(n2, "scan query")));
  take(cub::DeviceScan::ExclusiveSum(nullptr, b, i64p, i64p, cub_items(n_u + 2, "scan query")));
  if (n_u > 0) {
    const int items = cub_items(n_u, "sort query");
    take(cub::DeviceRadixSort::SortPairs(nullptr, b, u64p, u64p, i32, i32, items, 0, 64));
    take(cub::DeviceSegmentedSort::SortPairs(
      nullptr, b, i64p, i64p, i32, i32, items, items, i64p, i64p));
  }
  best = std::max(best, static_cast<std::size_t>(8 * (n2 / 4096 + 2)));
  return best;
}

// LOCAL_MOVE + COMPACT; community ids lie in [0, 2 n_U).
struct MoveBufs {
  i64* K              = nullptr;  // [2 n_U] community volumes
  int* csize          = nullptr;  // [2 n_U] vertices per community
  int* rank           = nullptr;  // [2 n_U + 2] COMPACT ranks / activity bitmaps
  int* defer_slot     = nullptr;  // [n_U] deferred activation target
  i64* dest           = nullptr;  // [n_U] (stamp << 32) | community
  int* buckets        = nullptr;  // [S * n_U] per sub-round vertex lists
  int* bucket_count   = nullptr;  // [S * kNumClasses]
  int* movers         = nullptr;  // [n_U]
  unsigned char* desc = nullptr;  // MoveState

  void carve(Carver& c, i64 n_u, int S)
  {
    K            = c.take<i64>(2 * n_u);
    csize        = c.take<int>(2 * n_u);
    rank         = c.take<int>(2 * n_u + 2);
    defer_slot   = c.take<int>(n_u);
    dest         = c.take<i64>(n_u);
    buckets      = c.take<int>(static_cast<std::size_t>(S) * n_u);
    bucket_count = c.take<int>(static_cast<std::size_t>(S) * kNumClasses);
    movers       = c.take<int>(n_u);
    desc         = c.take<unsigned char>(kMoveStateBytes);
  }
};

// PLEIDEN_R (R9 moves cmap into the level stack: AGGREGATE aliases these).
struct RefineBufs {
  i64* ext               = nullptr;  // all [n_U] unless noted
  i64* inner             = nullptr;
  unsigned char* U       = nullptr;
  unsigned char* flipped = nullptr;
  int* hn                = nullptr;
  int* root              = nullptr;
  int* tree_count        = nullptr;  // [n_U + 2]
  int* tree_offset       = nullptr;  // [n_U + 2]
  int* members           = nullptr;
  int* r                 = nullptr;
  int* cid               = nullptr;  // [n_U + 2]
  i64* sort_keys_a       = nullptr;  // segmented sort of trees > 32
  i64* sort_keys_b       = nullptr;
  int* sort_vals_a       = nullptr;
  int* sort_vals_b       = nullptr;
  int* tree_list         = nullptr;  // thread class front, warp class back
  int* large_list        = nullptr;  // trees > 32

  void carve(Carver& c, i64 n_u)
  {
    tree_list   = c.take<int>(n_u);
    large_list  = c.take<int>(n_u);
    ext         = c.take<i64>(n_u);
    inner       = c.take<i64>(n_u);
    U           = c.take<unsigned char>(n_u);
    flipped     = c.take<unsigned char>(n_u);
    hn          = c.take<int>(n_u);
    root        = c.take<int>(n_u);
    tree_count  = c.take<int>(n_u + 2);
    tree_offset = c.take<int>(n_u + 2);
    members     = c.take<int>(n_u);
    r           = c.take<int>(n_u);
    cid         = c.take<int>(n_u + 2);
    sort_keys_a = c.take<i64>(n_u);
    sort_keys_b = c.take<i64>(n_u);
    sort_vals_a = c.take<int>(n_u);
    sort_vals_b = c.take<int>(n_u);
  }
};

struct AggregateBufs {
  int* member_count  = nullptr;  // [n_U + 2]
  i64* member_start  = nullptr;  // [n_U + 2]
  int* members       = nullptr;  // [n_U]
  i64* window        = nullptr;  // [n_U + 2] sum of member degrees
  i64* window_offset = nullptr;  // [n_U + 2]
  i64* cdeg          = nullptr;  // [n_U + 2] coarse row lengths
  int* clist         = nullptr;  // [n_U] warp class front, block class back
  int* clist_wide    = nullptr;  // [n_U] wide-warp class
  i64* khat_next     = nullptr;  // [n_U + 2] member sums of k_hat
  i64* indptr_next   = nullptr;  // [n_U + 2] scan of cdeg

  void carve(Carver& c, i64 n_u)
  {
    member_count  = c.take<int>(n_u + 2);
    member_start  = c.take<i64>(n_u + 2);
    members       = c.take<int>(n_u);
    window        = c.take<i64>(n_u + 2);
    window_offset = c.take<i64>(n_u + 2);
    cdeg          = c.take<i64>(n_u + 2);
    clist         = c.take<int>(n_u);
    clist_wide    = c.take<int>(n_u);
    khat_next     = c.take<i64>(n_u + 2);
    indptr_next   = c.take<i64>(n_u + 2);
  }
};

// CC split, Q, labels.
struct FinalBufs {
  int* parent   = nullptr;  // all [n_U] unless noted
  int* flags    = nullptr;  // [n_U + 2]
  int* rank     = nullptr;  // [n_U + 2]
  i64* K        = nullptr;
  double* t     = nullptr;
  int* rep      = nullptr;  // replica mask of each community (R > 1)
  int* size     = nullptr;
  int* minv     = nullptr;
  u64* keys_in  = nullptr;
  u64* keys_out = nullptr;
  int* vals_in  = nullptr;
  int* vals_out = nullptr;
  int* label_of = nullptr;

  void carve(Carver& c, i64 n_u)
  {
    parent   = c.take<int>(n_u);
    flags    = c.take<int>(n_u + 2);
    rank     = c.take<int>(n_u + 2);
    K        = c.take<i64>(n_u);
    t        = c.take<double>(n_u);
    rep      = c.take<int>(n_u);
    size     = c.take<int>(n_u);
    minv     = c.take<int>(n_u);
    keys_in  = c.take<u64>(n_u);
    keys_out = c.take<u64>(n_u);
    vals_in  = c.take<int>(n_u);
    vals_out = c.take<int>(n_u);
    label_of = c.take<int>(n_u);
  }
};

struct PersistBufs {
  int* P0 = nullptr;  // [n_U] partition carried across iterations
  int* Pa = nullptr;  // [n_U] ping-pong partitions of the levels
  int* Pb = nullptr;
  i64* KS = nullptr;  // [n_U] K_S after COMPACT

  void carve(Carver& c, i64 n_u)
  {
    P0 = c.take<int>(n_u);
    Pa = c.take<int>(n_u);
    Pb = c.take<int>(n_u);
    KS = c.take<i64>(n_u);
  }
};

struct Level0Bufs {
  i64* indptr  = nullptr;  // [n + 1]
  int* indices = nullptr;  // [nnz] (null: int32 input read in place)
  i64* wq      = nullptr;  // [nnz] W0q
  i64* khat    = nullptr;  // [n]

  void carve(Carver& c, const LayoutParams& p)
  {
    indptr  = c.take<i64>(p.n + 1);
    indices = p.idx64_input ? c.take<int>(p.nnz) : nullptr;
    wq      = p.wkind == WKind::I64 ? c.take<i64>(p.nnz) : nullptr;
    khat    = c.take<i64>(p.n);
  }
};

// Replica union: vertex r n + v, edges shifted by r n.
struct UnionBufs {
  i64* indptr  = nullptr;  // [R n + 1]
  int* indices = nullptr;  // [R nnz]
  float* wf32  = nullptr;  // [R nnz]
  i64* wq      = nullptr;  // [R nnz]
  i64* khat    = nullptr;  // [R n]

  void carve(Carver& c, const LayoutParams& p)
  {
    if (p.replicas <= 1) return;
    const i64 nu = p.n_union(), eu = p.nnz_union();
    const bool i64w = p.wkind == WKind::I64;
    indptr          = c.take<i64>(nu + 1);
    indices         = c.take<int>(eu);
    wf32            = p.wkind == WKind::F32 ? c.take<float>(eu) : nullptr;
    wq              = i64w ? c.take<i64>(eu) : nullptr;
    khat            = c.take<i64>(nu);
  }
};

// Level stack + edge arena: 8 B per entry (index, weight) times arena_factor *
// nnz_0, 24 B per vertex (offsets, k_hat, cmaps) and alignment slack.
constexpr std::size_t kArenaEntryBytes  = 8;
constexpr std::size_t kArenaVertexBytes = 24;
constexpr std::size_t kArenaSlack       = kMaxLevels * 8 * kAlign;

inline std::size_t arena_vertex_bytes(const LayoutParams& p)
{
  return kArenaVertexBytes * static_cast<std::size_t>(p.n_union()) + kArenaSlack;
}

inline std::size_t arena_bytes(const LayoutParams& p)
{
  const double rows = std::ceil(p.arena_factor * static_cast<double>(kArenaEntryBytes) *
                                static_cast<double>(p.nnz_union()));
  return align_up(static_cast<std::size_t>(rows) + arena_vertex_bytes(p));
}

// The arena factor under which `needed` arena bytes fit (for the rerun).
inline double arena_required_factor(const LayoutParams& p, std::size_t needed)
{
  const double per = static_cast<double>(kArenaEntryBytes) * static_cast<double>(p.nnz_union());
  if (per <= 0.0) return p.arena_factor;
  const double f = (static_cast<double>(needed) - static_cast<double>(arena_vertex_bytes(p))) / per;
  return std::max(f, p.arena_factor);
}

// Host bookkeeping of the arena: bump pointers from the bottom (levels, reset
// per iteration) and the top (holey rows); a carve that does not fit returns
// nullptr and records the demand.
struct LevelArena {
  char* base           = nullptr;
  std::size_t cap      = 0;
  std::size_t bottom   = 0;
  std::size_t top      = 0;
  std::size_t required = 0;

  void reset()
  {
    bottom = 0;
    top    = 0;
  }
  bool fits(std::size_t extra_bottom, std::size_t extra_top) const
  {
    return align_up(bottom) + extra_bottom + align_up(top) + extra_top <= cap;
  }

  template <typename T>
  T* push_bottom(std::size_t count)
  {
    const std::size_t b = align_up(bottom), nb = count * sizeof(T);
    if (b + nb + top > cap) {
      note_overflow(b + nb + top);
      return nullptr;
    }
    bottom = b + nb;
    return reinterpret_cast<T*>(base + b);
  }

  template <typename T>
  T* push_top(std::size_t count)
  {
    const std::size_t nt = align_up(top + count * sizeof(T));
    if (align_up(bottom) + nt > cap) {
      note_overflow(align_up(bottom) + nt);
      return nullptr;
    }
    top = nt;
    return reinterpret_cast<T*>(base + cap - nt);
  }
  void release_top() { top = 0; }

  void note_overflow(std::size_t need) { required = std::max(required, need); }
};

struct Layout {
  LayoutParams p;
  Control* ctl          = nullptr;
  void* pinned          = nullptr;
  void* cub             = nullptr;
  std::size_t cub_bytes = 0;
  MoveBufs move;  // the phase groups start at the same address
  RefineBufs refine;
  AggregateBufs aggregate;
  FinalBufs final_;
  PersistBufs persist;
  Level0Bufs l0;
  UnionBufs uni;
  LevelArena arena;
  std::size_t total = 0;
};

// The prefix of every layout (control, cub, phase) for n_u = R * n vertices.
inline void carve_prefix(Layout& L, Carver& c, i64 n_u, int S)
{
  L.ctl                   = reinterpret_cast<Control*>(c.bytes(kControlBytes));
  L.cub_bytes             = cub_temp_bytes(n_u);
  L.cub                   = c.bytes(L.cub_bytes);
  const std::size_t start = c.used();
  std::size_t phase       = 0;
  auto group              = [&](auto&& fn) {
    Carver g{c.base, start};
    fn(g);
    phase = std::max(phase, g.used() - start);
  };
  group([&](Carver& g) { L.move.carve(g, n_u, S); });
  group([&](Carver& g) { L.refine.carve(g, n_u); });
  group([&](Carver& g) { L.aggregate.carve(g, n_u); });
  group([&](Carver& g) { L.final_.carve(g, n_u); });
  c.off = start + phase;
}

inline Layout make_layout(const LayoutParams& p, void* base)
{
  p.validate();
  Layout L;
  L.p = p;
  Carver c{static_cast<char*>(base), 0};
  carve_prefix(L, c, p.n_union(), p.subrounds);
  L.persist.carve(c, p.n_union());
  L.l0.carve(c, p);
  L.uni.carve(c, p);
  const std::size_t ab = arena_bytes(p);
  L.arena.base         = c.bytes(ab);
  L.arena.cap          = ab;
  L.total              = c.used();
  return L;
}

inline std::size_t workspace_bytes(const LayoutParams& p) { return make_layout(p, nullptr).total; }

// Runs `fn` (device work on `stream`); if it throws, the stream is synchronised first, so no
// queued kernel can touch the workspace after the caller releases it.
template <typename Fn>
auto run_synced(cudaStream_t stream, Fn&& fn) -> decltype(fn())
{
  try {
    return fn();
  } catch (...) {
    cudaStreamSynchronize(stream);
    cudaGetLastError();  // the original error is the one reported
    throw;
  }
}

}  // namespace cugraph::detail::leiden_engine
