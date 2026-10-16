/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 * SPDX-FileCopyrightText: Copyright (c) 2023, Martin Burtscher
 *
 * ECL-SCC signature propagation with directional vertex queues.
 * Original ECL-SCC copyright and BSD-3-Clause terms are retained in
 * ECL-SCC-LICENSE. Graph storage and scheduling use cuGraph primitives.
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/stream_ref>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

/// @cond

namespace cugraph::detail::ecl_scc {
// The sole ECL implementation uses fixed global Manhattan scheduling.
constexpr int edge_tile_size = 1024;
constexpr int block_size     = 256;
constexpr int warp_size      = 32;

template <typename T>
__device__ T load(T* ptr)
{
  return cuda::atomic_ref<T, cuda::thread_scope_device>(*ptr).load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ bool update_max(T* ptr, T value)
{
  cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
  return ref.load(cuda::memory_order_relaxed) < value &&
         ref.fetch_max(value, cuda::memory_order_relaxed) < value;
}

__device__ inline bool edge_enabled(uint32_t const* mask, size_t edge)
{
  return mask == nullptr || ((mask[packed_bool_offset(edge)] & packed_bool_mask(edge)) != 0);
}

// Count raw CSR destinations directly, with counters matching the edge-offset type.
template <typename vertex_t, typename edge_t>
__global__ void histogram_in_degrees(vertex_t const* indices, size_t count, edge_t* degrees)
{
  for (size_t e = size_t{blockIdx.x} * blockDim.x + threadIdx.x; e < count;
       e += size_t{gridDim.x} * blockDim.x) {
    cuda::atomic_ref<edge_t, cuda::thread_scope_device>(degrees[indices[e]])
      .fetch_add(edge_t{1}, cuda::memory_order_relaxed);
  }
}

}  // namespace cugraph::detail::ecl_scc

namespace cugraph::detail::ecl_scheduling {
constexpr int threads = ecl_scc::block_size;

template <typename vertex_t, typename edge_t>
struct queued_degree {
  edge_t const* offsets;
  vertex_t const* queue;
  size_t count;
  __device__ edge_t operator()(size_t i) const
  {
    if (i == count) return edge_t{0};
    auto v = queue ? queue[i] : static_cast<vertex_t>(i);
    return offsets[v + 1] - offsets[v];
  }
};

template <typename edge_t>
struct global_manhattan_workspace {
  rmm::device_uvector<edge_t> prefix;
  rmm::device_uvector<std::byte> scratch;
  global_manhattan_workspace(size_t capacity, cuda::stream_ref stream)
    : prefix(capacity ? capacity + 1 : 0, stream), scratch(0, stream)
  {
  }

  template <typename vertex_t>
  void prepare(raft::handle_t const& handle,
               edge_t const* offsets,
               vertex_t const* queue,
               size_t count)
  {
    auto stream = handle.get_stream();
    auto degrees =
      thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                      queued_degree<vertex_t, edge_t>{offsets, queue, count});
    // Size scratch once for the largest possible queue; later frontiers reuse it.
    if (scratch.size() == 0) {
      size_t bytes{};
      RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, degrees, prefix.data(), prefix.size(), stream.get()));
      scratch.resize(bytes, stream);
    }
    auto bytes = scratch.size();
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      scratch.data(), bytes, degrees, prefix.data(), count + 1, stream.get()));
  }
};

template <typename edge_t>
__device__ size_t edge_owner(edge_t const* prefix, edge_t edge, size_t low, size_t high)
{
  // Highest prefix <= edge handles repeated prefixes from zero-degree rows.
  while (low + 1 < high) {
    auto mid = low + (high - low) / 2;
    if (prefix[mid] <= edge)
      low = mid;
    else
      high = mid;
  }
  return low;
}

// No host edge-count readback and no shared prefix window. Each block's
// two endpoint searches bound the subsequent per-thread owner searches.
template <typename vertex_t, typename edge_t, typename Op>
__global__ void manhattan_global(
  edge_t const* offsets, vertex_t const* queue, size_t count, edge_t const* prefix, Op op)
{
  __shared__ size_t bounds[2];
  auto total = static_cast<size_t>(prefix[count]);
  for (size_t begin = size_t{blockIdx.x} * ecl_scc::edge_tile_size; begin < total;
       begin += size_t{gridDim.x} * ecl_scc::edge_tile_size) {
    auto end = min(total, begin + ecl_scc::edge_tile_size);
    if (threadIdx.x < 2) {
      auto edge           = static_cast<edge_t>(threadIdx.x == 0 ? begin : end - 1);
      bounds[threadIdx.x] = edge_owner(prefix, edge, 0, count);
    }
    __syncthreads();
    auto first = bounds[0], last = bounds[1];
    size_t low = first;
    for (size_t edge = begin + threadIdx.x; edge < end; edge += threads) {
      low    = edge_owner(prefix, static_cast<edge_t>(edge), low, last + 1);
      auto v = queue ? queue[low] : static_cast<vertex_t>(low);
      op(v, static_cast<edge_t>(offsets[v] + edge - prefix[low]));
    }
    __syncthreads();
  }
}

template <typename vertex_t, typename edge_t, typename Op>
void apply(raft::handle_t const& handle,
           edge_t const* offsets,
           vertex_t const* queue,
           size_t count,
           global_manhattan_workspace<edge_t>& workspace,
           Op op)
{
  if (!count) return;
  workspace.prepare(handle, offsets, queue, count);
  auto max_blocks = handle.get_device_properties().multiProcessorCount * 8;
  manhattan_global<<<max_blocks, threads, 0, handle.get_stream().get()>>>(
    offsets, queue, count, workspace.prefix.data(), op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace cugraph::detail::ecl_scheduling

namespace cugraph {
namespace detail {
namespace ecl_scc {

// Compare the full product even for graphs with 64-bit edge counts. Ties select
// the smallest physical vertex ID, independently of the reduction order.
template <typename vertex_t>
struct degree_seed {
  uint64_t high;
  uint64_t low;
  vertex_t vertex;
};

template <typename vertex_t>
struct maximum_degree_seed {
  __host__ __device__ degree_seed<vertex_t> operator()(degree_seed<vertex_t> a,
                                                       degree_seed<vertex_t> b) const
  {
    return b.high > a.high ||
               (b.high == a.high && (b.low > a.low || (b.low == a.low && b.vertex < a.vertex)))
             ? b
             : a;
  }
};

template <typename vertex_t, typename edge_t>
struct degree_product {
  edge_t const* outgoing_offsets;
  edge_t const* incoming_offsets;
  edge_t const* masked_out_degrees;
  __device__ degree_seed<vertex_t> operator()(vertex_t v) const
  {
    auto in  = static_cast<uint64_t>(incoming_offsets[v + 1] - incoming_offsets[v]);
    auto out = static_cast<uint64_t>(
      masked_out_degrees ? masked_out_degrees[v] : outgoing_offsets[v + 1] - outgoing_offsets[v]);
    uint64_t high{};
    if constexpr (sizeof(edge_t) > sizeof(int32_t)) { high = __umul64hi(in, out); }
    return {high, in * out, v};
  }
};

// The per-call degree-product seed stays on the device. Reduction scratch
// survives until the computation finishes.
template <typename vertex_t, typename edge_t>
struct workspace {
  rmm::device_uvector<degree_seed<vertex_t>> seed;
  rmm::device_uvector<std::byte> scratch;
  workspace(cuda::stream_ref stream) : seed(1, stream), scratch(0, stream) {}
};

// Called only by lanes with an update. Epochs deduplicate without clearing an
// O(V) flag array each iteration; warp reservations reduce queue contention.
// Queue consumers run in subsequent kernels on the same stream, so relaxed
// atomics suffice for reservations and monotonically increasing signatures.
template <typename vertex_t>
__device__ void enqueue(
  vertex_t v, vertex_t* queue, vertex_t* count, uint32_t* epochs, uint32_t epoch)
{
  auto append = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(epochs[v]).exchange(
                  epoch, cuda::memory_order_relaxed) != epoch;
  auto active = __activemask();
  auto votes  = __ballot_sync(active, append);
  if (votes == 0) { return; }
  auto lane   = threadIdx.x % warp_size;
  auto leader = __ffs(votes) - 1;
  vertex_t base{};
  if (lane == leader) {
    base = cuda::atomic_ref<vertex_t, cuda::thread_scope_device>(*count).fetch_add(
      static_cast<vertex_t>(__popc(votes)), cuda::memory_order_relaxed);
  }
  base = static_cast<vertex_t>(__shfl_sync(active, static_cast<unsigned long long>(base), leader));
  if (append) { queue[base + __popc(votes & ((uint32_t{1} << lane) - 1))] = v; }
}

// A vertex with zero in-degree or out-degree is necessarily a singleton SCC.
// Compact the remaining vertices before propagation.
template <typename vertex_t, typename edge_t>
__global__ void trim_singletons(edge_t const* offsets,
                                edge_t const* reverse_offsets,
                                edge_t const* masked_out_degrees,
                                size_t n,
                                vertex_t* labels)
{
  for (size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < n;
       i += size_t{gridDim.x} * blockDim.x) {
    auto out_degree = masked_out_degrees ? masked_out_degrees[i] : offsets[i + 1] - offsets[i];
    if (reverse_offsets[i] == reverse_offsets[i + 1] || out_degree == 0) {
      labels[i] = static_cast<vertex_t>(i);
    }
  }
}

template <typename vertex_t>
__global__ void initialize_signatures(vertex_t const* active,
                                      size_t count,
                                      vertex_t* incoming,
                                      vertex_t* outgoing,
                                      vertex_t n,
                                      degree_seed<vertex_t> const* seed,
                                      bool first_round)
{
  if (first_round && blockIdx.x == 0 && threadIdx.x == 0) {
    incoming[n] = n;
    outgoing[n] = n;
  }
  for (size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count;
       i += size_t{gridDim.x} * blockDim.x) {
    auto v      = active[i];
    auto color  = first_round && v == seed->vertex ? n : v;
    incoming[v] = color;
    outgoing[v] = color;
  }
}

template <typename vertex_t>
__device__ void propagate_edge(vertex_t v,
                               vertex_t dst,
                               vertex_t const* labels,
                               vertex_t const* partition_in,
                               vertex_t const* partition_out,
                               vertex_t* signatures,
                               vertex_t* next_queue,
                               vertex_t* next_count,
                               uint32_t* epochs,
                               uint32_t epoch)
{
  if (labels[dst] != invalid_component_id<vertex_t>::value ||
      partition_in[v] != partition_in[dst] || partition_out[v] != partition_out[dst]) {
    return;
  }
  auto signature = load(signatures + v);
  auto shortcut  = signature > v ? load(signatures + signature) : signature;
  if (shortcut > signature && update_max(signatures + v, shortcut)) {
    enqueue(v, next_queue, next_count, epochs, epoch);
  }
  if (update_max(signatures + dst, shortcut)) {
    enqueue(dst, next_queue, next_count, epochs, epoch);
  }
}

template <typename vertex_t>
__global__ void split_components(vertex_t const* active,
                                 size_t count,
                                 vertex_t const* incoming,
                                 vertex_t const* outgoing,
                                 vertex_t* partition_in,
                                 vertex_t* partition_out,
                                 vertex_t* labels,
                                 vertex_t* next_active,
                                 vertex_t* next_count,
                                 vertex_t n,
                                 degree_seed<vertex_t> const* seed)
{
  for (size_t base = size_t{blockIdx.x} * blockDim.x; base < count;
       base += size_t{gridDim.x} * blockDim.x) {
    auto i = base + threadIdx.x;
    vertex_t v{};
    bool keep = false;
    if (i < count) {
      v = active[i];
      if (incoming[v] == outgoing[v]) {
        labels[v] = incoming[v] == n ? seed->vertex : incoming[v];
      } else {
        partition_in[v]  = incoming[v];
        partition_out[v] = outgoing[v];
        keep             = true;
      }
    }
    auto votes  = __ballot_sync(0xffffffffu, keep);
    auto lane   = threadIdx.x % warp_size;
    auto leader = __ffs(votes) - 1;
    vertex_t pos{};
    if (votes && lane == leader) {
      pos = cuda::atomic_ref<vertex_t, cuda::thread_scope_device>(*next_count)
              .fetch_add(static_cast<vertex_t>(__popc(votes)), cuda::memory_order_relaxed);
    }
    if (votes) {
      pos = static_cast<vertex_t>(
        __shfl_sync(0xffffffffu, static_cast<unsigned long long>(pos), leader));
      if (keep) { next_active[pos + __popc(votes & ((uint32_t{1} << lane) - 1))] = v; }
    }
  }
}

}  // namespace ecl_scc
}  // namespace detail

namespace {

template <typename GraphViewType>
std::unique_ptr<
  graph_t<typename GraphViewType::vertex_type, typename GraphViewType::edge_type, false, false>>
build_ecl_reverse_graph(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  typename GraphViewType::edge_type number_of_edges,
  detail::ecl_scheduling::global_manhattan_workspace<typename GraphViewType::edge_type>&
    global_workspace)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using namespace detail::ecl_scc;
  auto stream = handle.get_stream();
  auto policy = handle.get_thrust_policy();
  auto n      = graph_view.number_of_vertices();
  auto edges =
    edge_partition_device_view_t<vertex_t, edge_t, false>(graph_view.local_edge_partition_view());
  auto mask = graph_view.has_edge_mask() ? graph_view.edge_mask_view()->value_firsts()[0] : nullptr;

  // Build the reverse CSR without renumbering, using cuGraph degree and scan
  // primitives. Keeping IDs avoids an inverse renumber map in every queue pass.
  rmm::device_uvector<edge_t> degrees(0, stream);
  if (!graph_view.has_edge_mask()) {
    // Compute the edge histogram directly on CSR indices when there is no mask.
    degrees.resize(n, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(degrees.data(), 0, degrees.size() * sizeof(edge_t), stream.get()));
    auto raw_edges = graph_view.local_edge_partition_view().indices().size();
    if (raw_edges > 0) {
      auto blocks =
        static_cast<int>(std::min<size_t>(handle.get_device_properties().multiProcessorCount * 8,
                                          (raw_edges + block_size - 1) / block_size));
      histogram_in_degrees<<<blocks, block_size, 0, stream.get()>>>(
        edges.indices(), raw_edges, degrees.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  } else {
    degrees = graph_view.compute_in_degrees(handle);
  }

  rmm::device_uvector<edge_t> offsets(static_cast<size_t>(n) + 1, stream);
  raft::copy(offsets.data(), degrees.data(), n, stream);
  offsets.set_element_to_zero_async(n, stream);
  cugraph::exclusive_scan(policy, offsets.begin(), offsets.end(), offsets.begin());

  rmm::device_uvector<vertex_t> indices(number_of_edges, stream);
  rmm::device_uvector<edge_t> next(n, stream);
  raft::copy(next.data(), offsets.data(), n, stream);
  auto scatter = [=, next_data = next.data(), indices_data = indices.data()] __device__(vertex_t v,
                                                                                        edge_t e) {
    if (edge_enabled(mask, e)) {
      auto dst = edges.indices()[e];
      auto pos = cuda::atomic_ref<edge_t, cuda::thread_scope_device>(next_data[dst])
                   .fetch_add(edge_t{1}, cuda::memory_order_relaxed);
      indices_data[pos] = v;
    }
  };
  detail::ecl_scheduling::apply(
    handle, edges.offsets(), static_cast<vertex_t const*>(nullptr), n, global_workspace, scatter);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return std::make_unique<graph_t<vertex_t, edge_t, false, false>>(
    handle,
    std::move(offsets),
    std::move(indices),
    graph_meta_t<vertex_t, edge_t, false>{
      n, graph_properties_t{graph_view.is_symmetric(), graph_view.is_multigraph()}},
    false);
}

template <typename GraphViewType>
void strongly_connected_components_ecl_impl(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type> components,
  bool do_expensive_check)
{
  // ECL-SCC: Alabandi et al., SC 2023, doi:10.1145/3581784.3607071.
  // Within each unresolved partition, propagate maximum signatures in both
  // directions. Equal signatures identify an SCC; otherwise their pair defines
  // the next partition. An SCC cannot span different signature pairs. The first
  // round gives the maximum-degree-product vertex the extra color N; convert
  // that sentinel back to its physical vertex ID when assigning components.
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");
  static_assert(!GraphViewType::is_multi_gpu, "ECL SCC currently supports single-GPU graphs only.");

  using namespace detail::ecl_scc;
  auto stream = handle.get_stream();
  auto policy = handle.get_thrust_policy();
  auto n      = graph_view.number_of_vertices();
  auto labels = components;
  if (n == 0) { return; }

  // Check input arguments (same contract as strongly_connected_components)
  CUGRAPH_EXPECTS(
    !graph_view.is_symmetric(),
    "Invalid input argument>: call weakly_connnected_components instead for symmetric graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  // Graph input graph data
  auto edges =
    edge_partition_device_view_t<vertex_t, edge_t, false>(graph_view.local_edge_partition_view());
  auto mask = graph_view.has_edge_mask() ? graph_view.edge_mask_view()->value_firsts()[0] : nullptr;
  auto m    = graph_view.compute_number_of_edges(handle);
  // A bounded grid reuses each block across edge tiles while limiting launch overhead.
  auto max_blocks = handle.get_device_properties().multiProcessorCount * 8;
  auto blocks_for = [max_blocks](size_t count) {
    return static_cast<int>(std::min<size_t>(max_blocks, (count + block_size - 1) / block_size));
  };

  rmm::device_uvector<vertex_t> counts(2, stream);
  workspace<vertex_t, edge_t> work(stream);
  detail::ecl_scheduling::global_manhattan_workspace<edge_t> global_workspace(n, stream);

  // Construct the inverse graph
  auto reverse_holder = build_ecl_reverse_graph(handle, graph_view, m, global_workspace);
  auto reverse_edges  = edge_partition_device_view_t<vertex_t, edge_t, false>(
    reverse_holder->view().local_edge_partition_view());

  // ECL algorithmic data
  rmm::device_uvector<vertex_t> partition_in(n, stream), partition_out(n, stream);
  rmm::device_uvector<vertex_t> incoming(static_cast<size_t>(n) + 1, stream);
  rmm::device_uvector<vertex_t> outgoing(static_cast<size_t>(n) + 1, stream);
  std::array<rmm::device_uvector<vertex_t>, 2> active{rmm::device_uvector<vertex_t>(n, stream),
                                                      rmm::device_uvector<vertex_t>(n, stream)};
  std::array<rmm::device_uvector<vertex_t>, 2> forward{rmm::device_uvector<vertex_t>(n, stream),
                                                       rmm::device_uvector<vertex_t>(n, stream)};
  std::array<rmm::device_uvector<vertex_t>, 2> backward{rmm::device_uvector<vertex_t>(n, stream),
                                                        rmm::device_uvector<vertex_t>(n, stream)};
  rmm::device_uvector<uint32_t> forward_epoch(n, stream), backward_epoch(n, stream);

  // Host-side helper for processing propagation queues
  auto propagate_queue = [&](edge_partition_device_view_t<vertex_t, edge_t, false> partition,
                             uint32_t const* edge_mask,
                             vertex_t const* queue,
                             size_t count,
                             vertex_t* signatures,
                             vertex_t* output,
                             int direction,
                             uint32_t current_epoch) {
    auto op = [           =,
               labels     = labels.data(),
               pin        = partition_in.data(),
               pout       = partition_out.data(),
               next_count = counts.data() + direction,
               epochs     = direction == 0 ? forward_epoch.data()
                                           : backward_epoch.data()] __device__(vertex_t v, edge_t e) {
      if (edge_enabled(edge_mask, e)) {
        propagate_edge(v,
                       partition.indices()[e],
                       labels,
                       pin,
                       pout,
                       signatures,
                       output,
                       next_count,
                       epochs,
                       current_epoch);
      }
    };
    detail::ecl_scheduling::apply(handle, partition.offsets(), queue, count, global_workspace, op);
  };

  // Initialize vertex states
  cugraph::fill(policy, labels.begin(), labels.end(), invalid_component_id<vertex_t>::value);
  cugraph::fill(policy, partition_in.begin(), partition_in.end(), vertex_t{0});
  cugraph::fill(policy, partition_out.begin(), partition_out.end(), vertex_t{0});
  cugraph::fill(policy, forward_epoch.begin(), forward_epoch.end(), uint32_t{0});
  cugraph::fill(policy, backward_epoch.begin(), backward_epoch.end(), uint32_t{0});

  // Initialize our seed heuristic -> vertex with the largest in*out degree gets max label
  auto masked_out_degrees = graph_view.has_edge_mask() ? graph_view.compute_out_degrees(handle)
                                                       : rmm::device_uvector<edge_t>(0, stream);
  {
    auto candidates = thrust::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      degree_product<vertex_t, edge_t>{
        edges.offsets(),
        reverse_edges.offsets(),
        masked_out_degrees.size() ? masked_out_degrees.data() : nullptr});
    size_t scratch_bytes{};
    auto initial = degree_seed<vertex_t>{0, 0, std::numeric_limits<vertex_t>::max()};
    RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(nullptr,
                                            scratch_bytes,
                                            candidates,
                                            work.seed.data(),
                                            n,
                                            maximum_degree_seed<vertex_t>{},
                                            initial,
                                            stream.get()));
    work.scratch.resize(scratch_bytes, stream);
    RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(work.scratch.data(),
                                            scratch_bytes,
                                            candidates,
                                            work.seed.data(),
                                            n,
                                            maximum_degree_seed<vertex_t>{},
                                            initial,
                                            stream.get()));
  }

  // A single iteration on trimming to remove trivial SCCs
  trim_singletons<<<blocks_for(n), block_size, 0, stream.get()>>>(
    edges.offsets(),
    reverse_edges.offsets(),
    masked_out_degrees.size() ? masked_out_degrees.data() : nullptr,
    n,
    labels.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Compact active vertices to construct our initial active vertex queue
  auto first_vertex = thrust::make_counting_iterator(vertex_t{0});
  auto active_end   = thrust::copy_if(
    policy,
    first_vertex,
    first_vertex + n,
    labels.begin(),
    active[0].begin(),
    [] __device__(vertex_t label) { return label == invalid_component_id<vertex_t>::value; });
  auto initial_active_count = static_cast<size_t>(active_end - active[0].begin());

  // Cleanup
  masked_out_degrees.resize(0, stream);
  masked_out_degrees.shrink_to_fit(stream);

  // Begin ECL's primary loop
  uint32_t epoch{0};
  size_t active_count = initial_active_count;
  int active_buffer{0};
  bool first_round{true};
  while (active_count > 0) {
    // Initialize this iteration's signatures
    initialize_signatures<<<blocks_for(active_count), block_size, 0, stream.get()>>>(
      active[active_buffer].data(),
      active_count,
      incoming.data(),
      outgoing.data(),
      n,
      work.seed.data(),
      first_round);

    first_round                    = false;
    vertex_t const* forward_queue  = active[active_buffer].data();
    vertex_t const* backward_queue = active[active_buffer].data();
    size_t forward_count = active_count, backward_count = active_count;
    int queue_buffer{0};

    // Propagate until convergence
    while (forward_count > 0 || backward_count > 0) {
      if (++epoch == 0) {
        cugraph::fill(policy, forward_epoch.begin(), forward_epoch.end(), uint32_t{0});
        cugraph::fill(policy, backward_epoch.begin(), backward_epoch.end(), uint32_t{0});
        epoch = 1;
      }
      RAFT_CUDA_TRY(
        cudaMemsetAsync(counts.data(), 0, counts.size() * sizeof(vertex_t), stream.get()));
      if (forward_count > 0) {
        propagate_queue(edges,
                        mask,
                        forward_queue,
                        forward_count,
                        incoming.data(),
                        forward[queue_buffer].data(),
                        0,
                        epoch);
      }
      if (backward_count > 0) {
        propagate_queue(reverse_edges,
                        static_cast<uint32_t const*>(nullptr),
                        backward_queue,
                        backward_count,
                        outgoing.data(),
                        backward[queue_buffer].data(),
                        1,
                        epoch);
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      std::array<vertex_t, 2> host_counts{};
      raft::update_host(host_counts.data(), counts.data(), host_counts.size(), stream);
      handle.sync_stream();
      forward_count  = host_counts[0];
      backward_count = host_counts[1];
      forward_queue  = forward[queue_buffer].data();
      backward_queue = backward[queue_buffer].data();
      queue_buffer   = 1 - queue_buffer;
    }

    // Label new SCCs and repartition remaining vertices
    RAFT_CUDA_TRY(cudaMemsetAsync(counts.data(), 0, sizeof(vertex_t), stream.get()));
    split_components<<<blocks_for(active_count), block_size, 0, stream.get()>>>(
      active[active_buffer].data(),
      active_count,
      incoming.data(),
      outgoing.data(),
      partition_in.data(),
      partition_out.data(),
      labels.data(),
      active[1 - active_buffer].data(),
      counts.data(),
      n,
      work.seed.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    vertex_t host_count{};
    raft::update_host(&host_count, counts.data(), 1, stream);
    handle.sync_stream();
    active_count  = host_count;
    active_buffer = 1 - active_buffer;
  }
  return;
}

}  // namespace

}  // namespace cugraph

/// @endcond
