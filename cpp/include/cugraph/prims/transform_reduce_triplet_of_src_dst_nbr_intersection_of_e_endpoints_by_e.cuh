/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/detail/nbr_intersection.cuh>
#include <cugraph/prims/per_v_pair_transform_src_dst_nbr_intersection.cuh>
#include <cugraph/prims/kv_store.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/pool_free_hook.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

// Reads a supporting edge's property value at the given local edge offset, or returns
// cuda::std::nullopt when no edge property is carried (the base pointer is void*).
template <typename EdgeProperty, typename edge_t>
__device__ auto nbr_intersection_get_e_property(EdgeProperty e_property, edge_t offset)
{
  if constexpr (std::is_same_v<EdgeProperty, void*>) {
    return cuda::std::nullopt;
  } else {
    return e_property[offset];
  }
}

// Multi-GPU only: compute the GPU that owns an edge from a (src, dst, value) triplet. The triplet
// is shuffled as a single value buffer, so the GPU-id operator reads the endpoints out of the
// triplet's first two elements.
template <typename vertex_t>
struct compute_gpu_id_from_edge_endpoints_in_triplet_t {
  compute_gpu_id_from_int_edge_endpoints_t<vertex_t> base{};

  template <typename TripletType>
  __device__ int operator()(TripletType triplet) const
  {
    return base(cuda::std::get<0>(triplet), cuda::std::get<1>(triplet));
  }
};

// Called by detail::nbr_intersection once per common neighbor w of a pair (v0, v1), that is, once
// per triangle. The user operator returns two values: one for the base edge (v0, v1) and one for
// each of the two supporting edges (v0, w) and (v1, w). This functor adds them to those three edges.
//
// Where an addition can go depends on whether this rank holds the edge. In single-GPU it holds all
// three, so all three are plain atomic adds into the per-edge accumulator. In multi-GPU it is only
// guaranteed to hold (v0, w); the other two go to side buffers and are settled after the loop. The
// routing note inside the operator says which goes where and why.
//
// Accumulation uses cuda::atomic_ref<T>::fetch_add, so T must be a scalar arithmetic type; a tuple T
// would instead use cugraph::atomic_add (element-wise). See the @tparam T note on the public API.
template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename IntersectionOp,
          typename T,
          typename AccumulatorIterator>
struct accumulate_triplet_op_t {
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu> edge_partition{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input{};
  IntersectionOp intersection_op{};
  AccumulatorIterator accumulator{};
  T identity{};

  // The two side buffers, both null in single-GPU.

  // Base edge (v0, v1): one slot per broadcast pair. Every rank that sees the pair adds its own
  // share, and the shares are summed at the edge's owner after the loop.
  T* per_pair_values{nullptr};
  // Supporting edge (v1, w): one slot per edge of the fetched N(v1) lists, indexed by
  // v1_w_edge_offset. Hits on the same edge sum in place, and the filled slots are turned back into
  // edges and shipped to their owners after the loop.
  T* remote_acc{nullptr};

  template <typename EdgeProperty0, typename EdgeProperty1>
  __device__ void operator()(vertex_t v0,
                             vertex_t v1,
                             vertex_t w,
                             edge_t v0_v1_edge_offset,
                             edge_t v0_w_edge_offset,
                             edge_t v1_w_edge_offset,
                             EdgeProperty0 v0_e_property,
                             EdgeProperty1 v1_e_property,
                             size_t pair_idx) const
  {
    auto major_offset = edge_partition.major_offset_from_major_nocheck(v0);
    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(v1);
    auto src          = GraphViewType::is_storage_transposed ? v1 : v0;
    auto dst          = GraphViewType::is_storage_transposed ? v0 : v1;
    auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;

    // Supporting-edge property values for (v0, w) and (v1, w) (cuda::std::nullopt when no edge
    // property is requested), passed to the user operator alongside the endpoint vertex properties.
    auto result = intersection_op(src,
                                  dst,
                                  edge_partition_src_value_input.get(src_offset),
                                  edge_partition_dst_value_input.get(dst_offset),
                                  w,
                                  nbr_intersection_get_e_property(v0_e_property, v0_w_edge_offset),
                                  nbr_intersection_get_e_property(v1_e_property, v1_w_edge_offset));
    auto edge_value       = cuda::std::get<0>(result);
    auto supporting_value = cuda::std::get<1>(result);

    // The accumulator is indexed by local CSR edge offset, so a contribution can use it only if its
    // edge lands in this partition. (v0, w) always does, because w was found by walking v0's local
    // adjacency and so lies in this minor slice. (v0, v1) may not: after the minor_comm broadcast v1
    // can lie outside the slice, so its value goes to per_pair_values. (v1, w) is keyed on v1, which
    // is generally not a local major, so it goes to remote_acc.

    // Base edge (v0, v1).
    if constexpr (GraphViewType::is_multi_gpu) {
      // Accumulate per broadcast-pair index; the per-rank partials are summed at the owner by the
      // post-loop shuffle + reduce.
      cuda::atomic_ref<T, cuda::thread_scope_device> base_ref(per_pair_values[pair_idx]);
      base_ref.fetch_add(edge_value, cuda::memory_order_relaxed);
    } else {
      cuda::atomic_ref<T, cuda::thread_scope_device> v0_v1_ref(accumulator[v0_v1_edge_offset]);
      v0_v1_ref.fetch_add(edge_value, cuda::memory_order_relaxed);
    }

    // Skip the supporting contribution when it is the additive identity: it is a no-op locally and
    // (in multi-GPU) avoids buffering/shuffling an identity contribution for no effect.
    if (supporting_value != identity) {
      // (v0, w) is a local edge (v0 is a local major and w lies in this rank's minor slice).
      cuda::atomic_ref<T, cuda::thread_scope_device> v0_w_ref(accumulator[v0_w_edge_offset]);
      v0_w_ref.fetch_add(supporting_value, cuda::memory_order_relaxed);
      if constexpr (GraphViewType::is_multi_gpu) {
        // (v1, w) may be owned by another rank. v1_w_edge_offset is its position in the fetched
        // N(v1) lists, so add there and rebuild the edge when the slots are compacted.
        cuda::atomic_ref<T, cuda::thread_scope_device> acc_ref(remote_acc[v1_w_edge_offset]);
        acc_ref.fetch_add(supporting_value, cuda::memory_order_relaxed);
      } else {
        cuda::atomic_ref<T, cuda::thread_scope_device> v1_w_ref(accumulator[v1_w_edge_offset]);
        v1_w_ref.fetch_add(supporting_value, cuda::memory_order_relaxed);
      }
    }
  }
};

// How much device memory one chunk of intersection work may occupy: a fraction of whatever is free
// at the moment the question is asked, so chunks come out as large as the machine allows and there
// are as few of them as possible. Asking at that moment rather than once up front matters, because
// how much is free depends on what the graph and the rest of the call are already holding.
//
// A budget of zero means one chunk, which is what happens when nothing on the machine can report
// free memory (see pool_free_hook.hpp).
inline double triplet_chunk_budget_bytes()
{
  constexpr double usable_fraction = 0.8;

  size_t cuda_free  = 0;
  size_t cuda_total = 0;
  cudaMemGetInfo(&cuda_free, &cuda_total);
  return static_cast<double>(query_pool_free_or(cuda_free)) * usable_fraction;
}

// Where deg(v1) comes from when v1 lives on another rank.
//
// Chunk sizes are decided from the degrees of the second endpoints, and a rank sees second endpoints
// it does not own. Replicating a degree array would cost O(V) on every rank, which is the kind of
// footprint this primitive is trying to shed, so each rank keeps only the out-degrees of the
// vertices in its own vertex-partition range (~V/p) and answers queries about them. Asking for a
// batch of vertices routes each one to its owner and brings the degree back.
template <typename GraphViewType>
struct owned_degrees_t {
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  kv_store_t<vertex_t, edge_t, true /* use_binary_search */> store;
  rmm::device_uvector<vertex_t> vertex_partition_range_lasts;
  int major_comm_size{1};
  int minor_comm_size{1};
};

template <typename GraphViewType>
owned_degrees_t<GraphViewType> build_owned_degrees(raft::handle_t const& handle,
                                                   GraphViewType const& graph_view)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto local_out_degrees = graph_view.compute_out_degrees(handle);
  // The keys are this rank's whole vertex range, so a counting iterator names them and no key array
  // has to be materialized.
  auto key_first = thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first());
  kv_store_t<vertex_t, edge_t, true> store(key_first,
                                           key_first + local_out_degrees.size(),
                                           local_out_degrees.data(),
                                           edge_t{0} /* invalid_value */,
                                           true /* key_sorted */,
                                           handle.get_stream());

  // Routing a query to its owner is a search over where the vertex partitions end, so those ends are
  // kept on the device alongside the store.
  auto h_range_lasts = graph_view.vertex_partition_range_lasts();
  rmm::device_uvector<vertex_t> d_range_lasts(h_range_lasts.size(), handle.get_stream());
  raft::update_device(
    d_range_lasts.data(), h_range_lasts.data(), h_range_lasts.size(), handle.get_stream());

  return owned_degrees_t<GraphViewType>{
    std::move(store),
    std::move(d_range_lasts),
    handle.get_subcomm(cugraph::partition_manager::major_comm_name()).get_size(),
    handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size()};
}

template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> fetch_owned_degrees(
  raft::handle_t const& handle,
  owned_degrees_t<GraphViewType> const& degrees,
  raft::device_span<typename GraphViewType::vertex_type const> keys)
{
  using vertex_t = typename GraphViewType::vertex_type;

  auto const* range_lasts = degrees.vertex_partition_range_lasts.data();
  auto const num_ranges   = static_cast<int>(degrees.vertex_partition_range_lasts.size());
  auto const major_comm_size = degrees.major_comm_size;
  auto const minor_comm_size = degrees.minor_comm_size;

  return collect_values_for_keys(
    handle,
    degrees.store.view(),
    keys.begin(),
    keys.end(),
    cuda::proclaim_return_type<int>(
      [range_lasts, num_ranges, major_comm_size, minor_comm_size] __device__(vertex_t v) {
        auto const vertex_partition_id = static_cast<int>(cuda::std::distance(
          range_lasts,
          thrust::upper_bound(thrust::seq, range_lasts, range_lasts + num_ranges, v)));
        return cugraph::partition_manager::compute_global_comm_rank_from_vertex_partition_id(
          major_comm_size, minor_comm_size, vertex_partition_id);
      }));
}

// Cut one partition's pair list into chunks that each fit the memory budget, and reorder the pairs
// so that a chunk is a contiguous slice of the returned arrays.
//
// What is being bounded is the gather that accumulate_triplets_for_partition performs right after
// the broadcast: for a set of pairs, the fetched second-endpoint lists cost the sum of deg(v1) over
// the distinct v1's in that set, and the dense remote accumulator gets one slot per edge of those
// lists. A chunk is therefore sized by a degree sum and not by a pair count, since a hundred pairs
// pointing at a hub cost far more than a hundred pairs pointing at leaves.
//
// Two things make this more than slicing the array into equal parts:
//
//   - the pairs are broadcast across minor_comm before the gather, so what chunk c actually costs is
//     driven by the union of every row rank's chunk c, not by this rank's slice alone. The row must
//     therefore cut in the same places. It does, because the cut depends only on v1: the row agrees
//     on the sorted list of v1's present in this partition and on their degrees, takes the running
//     degree total along that list, and places v1 in chunk (total before v1) / (degrees per chunk).
//     Same list, same degrees, same answer on every rank of the row.
//   - each chunk performs collectives, so every rank has to run the same number of them. The count
//     each rank works out for itself is raised to the global maximum before anything is cut, and a
//     rank that ends up with empty chunks still calls into them.
//
// The returned boundaries always have num_chunks + 1 entries. When there is only one chunk the pair
// arrays come back empty, meaning the caller's own arrays are already in the right order.
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::vector<size_t>>
plan_partition_chunks(raft::handle_t const& handle,
                      owned_degrees_t<GraphViewType> const& degrees,
                      raft::device_span<typename GraphViewType::vertex_type const> majors,
                      raft::device_span<typename GraphViewType::vertex_type const> minors,
                      double budget_bytes,
                      double bytes_per_degree)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto const num_pairs = majors.size();

  rmm::device_uvector<vertex_t> chunked_majors(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> chunked_minors(size_t{0}, handle.get_stream());

  auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

  // Step 1. The second endpoints this rank contributes to this partition, deduplicated. Duplicates
  // are dropped because a v1 reached by ten pairs is still fetched once and still costs deg(v1).
  rmm::device_uvector<vertex_t> local_v1(num_pairs, handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), minors.begin(), minors.end(), local_v1.begin());
  thrust::sort(handle.get_thrust_policy(), local_v1.begin(), local_v1.end());
  local_v1.resize(
    static_cast<size_t>(cuda::std::distance(
      local_v1.begin(),
      thrust::unique(handle.get_thrust_policy(), local_v1.begin(), local_v1.end()))),
    handle.get_stream());

  // Step 2. Their degrees, from whichever ranks own them.
  auto local_degrees = fetch_owned_degrees<GraphViewType>(
    handle, degrees, raft::device_span<vertex_t const>(local_v1.data(), local_v1.size()));

  // Step 3. The row's shared picture. Every rank of the row sends its (v1, degree) list to every
  // other, and the concatenation is sorted and deduplicated, so all of them end up holding the same
  // list in the same order. This is what makes the cut identical across the row.
  auto v1_counts     = host_scalar_allgather(minor_comm, local_v1.size(), handle.get_stream());
  std::vector<size_t> v1_displacements(v1_counts.size());
  std::exclusive_scan(v1_counts.begin(), v1_counts.end(), v1_displacements.begin(), size_t{0});
  auto const total_v1 = v1_displacements.back() + v1_counts.back();

  rmm::device_uvector<vertex_t> row_v1(total_v1, handle.get_stream());
  rmm::device_uvector<edge_t> row_degrees(total_v1, handle.get_stream());
  cugraph::device_allgatherv(minor_comm,
                             local_v1.data(),
                             row_v1.begin(),
                             raft::host_span<size_t const>(v1_counts.data(), v1_counts.size()),
                             raft::host_span<size_t const>(v1_displacements.data(),
                                                           v1_displacements.size()),
                             handle.get_stream());
  cugraph::device_allgatherv(minor_comm,
                             local_degrees.data(),
                             row_degrees.begin(),
                             raft::host_span<size_t const>(v1_counts.data(), v1_counts.size()),
                             raft::host_span<size_t const>(v1_displacements.data(),
                                                           v1_displacements.size()),
                             handle.get_stream());
  // Sorted on the vertex id, with the degree riding along as the payload. unique_by_key only
  // collapses adjacent equal keys and the same v1 arrives from several ranks, so the copies have to
  // be brought together; the id is also the only ordering the row is guaranteed to agree on, since
  // ids are distinct while degrees tie. Dropping the duplicate degrees loses nothing, as every rank
  // asked the same owner for deg(v1) and they all carry the same number.
  thrust::sort_by_key(
    handle.get_thrust_policy(), row_v1.begin(), row_v1.end(), row_degrees.begin());
  auto const num_row_v1 = static_cast<size_t>(cuda::std::distance(
    row_v1.begin(),
    thrust::unique_by_key(
      handle.get_thrust_policy(), row_v1.begin(), row_v1.end(), row_degrees.begin())
      .first));
  row_v1.resize(num_row_v1, handle.get_stream());
  row_degrees.resize(num_row_v1, handle.get_stream());

  // Step 4. The running degree total along that shared list, plus the total itself. Position of a v1
  // in this running total is what decides its chunk in step 6.
  rmm::device_uvector<edge_t> row_degree_prefix(num_row_v1, handle.get_stream());
  thrust::exclusive_scan(handle.get_thrust_policy(),
                         row_degrees.begin(),
                         row_degrees.end(),
                         row_degree_prefix.begin(),
                         edge_t{0});
  auto const partition_degrees = thrust::reduce(
    handle.get_thrust_policy(), row_degrees.begin(), row_degrees.end(), edge_t{0});

  // Step 5. How many chunks that degree total needs, and then how many everyone will actually run.
  // The local answer can differ from rank to rank, because free memory does, so the largest wins.
  size_t num_chunks_local = 1;
  if (budget_bytes > 0.0 && partition_degrees > 0) {
    auto const degrees_per_chunk = std::max(1.0, budget_bytes / bytes_per_degree);
    num_chunks_local             = std::max<size_t>(
      size_t{1},
      static_cast<size_t>((static_cast<double>(partition_degrees) + degrees_per_chunk - 1.0) /
                          degrees_per_chunk));
  }
  auto num_chunks = host_scalar_allreduce(
    handle.get_comms(), num_chunks_local, raft::comms::op_t::MAX, handle.get_stream());
  if (num_chunks == 0) { num_chunks = 1; }

  std::vector<size_t> bounds(num_chunks + 1, num_pairs);
  bounds[0] = 0;
  if (num_chunks == 1 || num_pairs == 0) {
    return std::make_tuple(std::move(chunked_majors), std::move(chunked_minors), std::move(bounds));
  }

  // Step 6. Give each pair its chunk. Look its v1 up in the shared list, read the running degree
  // total at that position, and divide by the degrees one chunk is allowed to hold.
  auto const degrees_per_chunk = std::max<edge_t>(
    edge_t{1},
    (partition_degrees + static_cast<edge_t>(num_chunks) - 1) / static_cast<edge_t>(num_chunks));
  rmm::device_uvector<size_t> pair_chunk(num_pairs, handle.get_stream());
  {
    auto const* v1_ptr     = row_v1.data();
    auto const* prefix_ptr = row_degree_prefix.data();
    auto const v1_size     = row_v1.size();
    thrust::transform(handle.get_thrust_policy(),
                      minors.begin(),
                      minors.end(),
                      pair_chunk.begin(),
                      cuda::proclaim_return_type<size_t>(
                        [v1_ptr, prefix_ptr, v1_size, degrees_per_chunk, num_chunks] __device__(
                          vertex_t v1) {
                          auto const pos = static_cast<size_t>(cuda::std::distance(
                            v1_ptr,
                            thrust::lower_bound(thrust::seq, v1_ptr, v1_ptr + v1_size, v1)));
                          if (pos >= v1_size) { return num_chunks - 1; }
                          auto const chunk = static_cast<size_t>(prefix_ptr[pos] / degrees_per_chunk);
                          return chunk >= num_chunks ? num_chunks - 1 : chunk;
                        }));
  }

  // Step 7. Reorder the pairs so that a chunk is one contiguous run. This is a counting sort
  // (histogram, then running offsets, then scatter) rather than a thrust::sort_by_key on the chunk
  // id: the radix sort_by_key path computes a 32-bit byte offset for the (major, minor) payload and
  // silently corrupts the permutation once that payload passes 4 GiB, which int64 vertices reach at
  // ~2^28 pairs. Nothing here is narrower than size_t.
  chunked_majors.resize(num_pairs, handle.get_stream());
  chunked_minors.resize(num_pairs, handle.get_stream());
  rmm::device_uvector<size_t> chunk_offsets(num_chunks, handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), chunk_offsets.begin(), chunk_offsets.end(), size_t{0});
  {
    auto* counts           = chunk_offsets.data();
    auto const* chunk_of   = pair_chunk.data();
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(num_pairs),
                     [counts, chunk_of] __device__(size_t p) {
                       cuda::atomic_ref<size_t, cuda::thread_scope_device> count(
                         counts[chunk_of[p]]);
                       count.fetch_add(size_t{1}, cuda::memory_order_relaxed);
                     });
  }
  thrust::exclusive_scan(handle.get_thrust_policy(),
                         chunk_offsets.begin(),
                         chunk_offsets.end(),
                         chunk_offsets.begin(),
                         size_t{0});
  raft::update_host(bounds.data(), chunk_offsets.data(), num_chunks, handle.get_stream());
  handle.sync_stream();
  bounds[num_chunks] = num_pairs;
  {
    // Each chunk fills its run from the front, so it needs its own running write position, starting
    // at where the run begins.
    rmm::device_uvector<size_t> next_write_offsets(num_chunks, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 chunk_offsets.begin(),
                 chunk_offsets.end(),
                 next_write_offsets.begin());
    auto* next_write       = next_write_offsets.data();
    auto const* chunk_of   = pair_chunk.data();
    auto const* src_majors = majors.data();
    auto const* src_minors = minors.data();
    auto* dst_majors       = chunked_majors.data();
    auto* dst_minors       = chunked_minors.data();
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_pairs),
      [next_write, chunk_of, src_majors, src_minors, dst_majors, dst_minors] __device__(size_t p) {
        cuda::atomic_ref<size_t, cuda::thread_scope_device> write_offset(
          next_write[chunk_of[p]]);
        auto const slot  = write_offset.fetch_add(size_t{1}, cuda::memory_order_relaxed);
        dst_majors[slot] = src_majors[p];
        dst_minors[slot] = src_minors[p];
      });
  }

  // Step 8. The scatter races, so a chunk comes out in no particular order, and the intersection
  // needs sorted pairs. Sort each run. These are cheap because a run is bounded by the budget, and
  // they replace the sort the caller would otherwise have done over the whole partition.
  for (size_t c = 0; c < num_chunks; ++c) {
    auto chunk_first = thrust::make_zip_iterator(chunked_majors.begin() + bounds[c],
                                                 chunked_minors.begin() + bounds[c]);
    thrust::sort(
      handle.get_thrust_policy(), chunk_first, chunk_first + (bounds[c + 1] - bounds[c]));
  }

  return std::make_tuple(std::move(chunked_majors), std::move(chunked_minors), std::move(bounds));
}

// Shared per-partition core for the by_e primitive, used by both the all-edges overload and the
// caller-supplied-edge-list overload. Given a batch of this partition's (majors, minors) pairs, it
// broadcasts them across minor_comm (multi-GPU), fetches whatever neighbor lists live on other
// ranks, runs the intersection, lets the functor above drop each contribution into one of the three
// buffers, then turns all three back into (src, dst, value) triplets and appends them to out_*.
// Nothing is resolved to a final per-edge value here: what comes out is a pile of partials, which
// finalize_triplet_reduction settles.
//
// The pairs arrive as spans rather than owned vectors because a partition is handed over in chunks
// (see plan_partition_chunks), each one a slice of the same array.
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename T>
void accumulate_triplets_for_partition(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  size_t i,
  raft::device_span<typename GraphViewType::vertex_type const> majors_input,
  raft::device_span<typename GraphViewType::vertex_type const> minors_input,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  rmm::device_uvector<typename GraphViewType::vertex_type>& out_srcs,
  rmm::device_uvector<typename GraphViewType::vertex_type>& out_dsts,
  dataframe_buffer_type_t<T>& out_values)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = float;  // dummy

  // Edge-property device view for the supporting edges (dummy when no edge property is requested).
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeDstValueInputWrapper::value_iterator,
      typename EdgeDstValueInputWrapper::value_type>>;

  auto edge_mask_view = graph_view.edge_mask_view();

    // retrieve the i-th local edge partition
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    // retrieve the i-th edge mask (optional)
    auto edge_partition_e_mask =
      edge_mask_view
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : std::nullopt;

    edge_partition_src_input_device_view_t edge_partition_src_value_input{};
    edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_src_value_input = edge_partition_src_input_device_view_t(edge_src_value_input);
      edge_partition_dst_value_input =
        edge_partition_dst_input_device_view_t(edge_dst_value_input, i);
    } else {
      edge_partition_src_value_input =
        edge_partition_src_input_device_view_t(edge_src_value_input, i);
      edge_partition_dst_value_input = edge_partition_dst_input_device_view_t(edge_dst_value_input);
    }

    // This partition's supporting-edge property view (for v0's local edges and, in single-GPU, v1's).
    auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, i);
    // Multi-GPU: a vertex's neighbors are split across the minor_comm row, so broadcast this
    // partition's (major, minor) pairs across minor_comm. Each rank then intersects every pair
    // against its own slice and finds a partial; the partials are summed at the owner later.
    // detail::nbr_intersection requires sorted input pairs.
    //
    // This broadcast is the reason the caller had to chunk, so it is worth being precise about what
    // it costs. Take a pair (v0, v1): v0 is a local major, but v1 is a second endpoint that almost
    // certainly lives on another rank, so N(v1) has to be fetched from whoever owns v1. After the
    // allgather below, this rank no longer holds only its own pairs -- it holds every rank's pairs in
    // the row. So the lists it has to fetch cover the distinct v1's of that whole union, and their
    // cost is the sum of deg(v1) over them, roughly minor_comm_size times what this rank alone
    // contributed. dense_remote_acc then gets one slot per edge of those fetched lists, so a single
    // hub v1 anywhere in the row drags its entire adjacency onto every rank in the row.
    //
    // The pair list itself is not the problem: it is linear in the number of pairs. Everything it
    // pulls in behind it is. That degree sum over the union is what plan_partition_chunks caps, and
    // it caps it with one cut agreed across the row rather than a cut each rank picks for itself.
    // Two things rule out the per-rank version.
    //
    // First, a local cut would measure the wrong number. If each rank only ever handled its own
    // pairs, sizing a chunk from its own pairs would be exactly right. But it does not: N(v0) is
    // itself split across the row, so the rank holding the pair can only see part of the answer, and
    // every rank has to intersect every pair against its own slice and hand back a partial. That is
    // what the broadcast is for. So a rank measures the degree sum of the pairs it holds and then
    // allocates for the degree sum of the pairs the whole row holds. Its own share is a fraction of
    // that, with nothing tying the two together: it can cut its slice into pieces that all sit well
    // inside the budget and still run out of memory in a round where a peer contributed one hub.
    //
    // Second, every chunk runs collectives -- the count allgather and the pair allgatherv here, the
    // degree and neighbor gathers below. They are matched by position, so if one rank cuts into three
    // chunks and another into five, the fourth round has nobody to pair with and the row hangs. Hence
    // the chunk count being raised to the global maximum before anything is cut.
    //
    // What is agreed is only where the cuts fall, and they fall on v1 identity rather than pair
    // index. A rank contributes to chunk c exactly the pairs it happens to have pointing there, which
    // may be none, in which case it still calls through that round's collectives empty.
    //
    // The pairs stay where the caller put them until the broadcast has something of its own to
    // return, so majors_view/minors_view start out pointing at the caller's slice and are redirected
    // to the received buffers below.
    auto majors_view = majors_input;
    auto minors_view = minors_input;
    rmm::device_uvector<vertex_t> rx_majors(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> rx_minors(size_t{0}, handle.get_stream());

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      if (minor_comm.get_size() > 1) {
        // Each rank sends a different number of pairs, so the counts are exchanged first to size the
        // receive buffer and place each rank's block in it.
        auto rx_counts = host_scalar_allgather(minor_comm, majors_input.size(), handle.get_stream());
        std::vector<size_t> rx_displacements(rx_counts.size());
        std::exclusive_scan(
          rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
        auto aggregate_size = rx_displacements.back() + rx_counts.back();

        rx_majors.resize(aggregate_size, handle.get_stream());
        rx_minors.resize(aggregate_size, handle.get_stream());
        // A rank cannot finish the intersection for the pairs it owns, because the rest of each
        // source's neighbors sit on its row peers. Sending every rank's pairs to every rank in the
        // row lets each one contribute what its own slice can see.
        cugraph::device_allgatherv(
          minor_comm,
          majors_input.data(),
          rx_majors.begin(),
          raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
          raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
          handle.get_stream());
        cugraph::device_allgatherv(
          minor_comm,
          minors_input.data(),
          rx_minors.begin(),
          raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
          raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
          handle.get_stream());

        auto broadcast_pair_first = thrust::make_zip_iterator(rx_majors.begin(), rx_minors.begin());
        thrust::sort(
          handle.get_thrust_policy(), broadcast_pair_first, broadcast_pair_first + rx_majors.size());

        majors_view = raft::device_span<vertex_t const>(rx_majors.data(), rx_majors.size());
        minors_view = raft::device_span<vertex_t const>(rx_minors.data(), rx_minors.size());
      }
    }

    // The base edges to intersect. In multi-GPU this is the concatenation of every rank's pairs in
    // the row, so its size is known before any intersection runs, unlike the number of triangles
    // those pairs will produce.
    auto vertex_pair_first = thrust::make_zip_iterator(majors_view.data(), minors_view.data());
    auto const num_pairs   = majors_view.size();

    // The three buffers the functor writes into, allocated one after another below. Each is indexed
    // by whatever is convenient for the atomic add, and each is compacted back into (src, dst, value)
    // triplets further down:
    //   - edge_accumulator (-> local_*): one slot per edge of this local partition, indexed by edge
    //     offset. Takes every contribution whose edge is right here: (v0, w) always, plus (v0, v1)
    //     and (v1, w) in single-GPU.
    //   - per_pair_buffer (-> base_*): multi-GPU only. One slot per broadcast pair, for the (v0, v1)
    //     contribution, whose edge may have fallen outside this minor slice when the pairs were
    //     broadcast.
    //   - dense_remote_acc (-> remote_*): multi-GPU only. One slot per edge of the fetched N(v1)
    //     lists, for the (v1, w) contribution, whose edge is generally owned by another rank.
    // None of the three holds a finished value: each rank only ever sees part of a given edge's
    // triangles, so what comes out is a partial, and the post-loop shuffle-to-owner plus
    // reduce_by_key is what adds the partials up.

    // First buffer: one slot per edge of this local partition. In multi-GPU the (v0, w) leg is the
    // only one that lands here, since w was reached through v0's local adjacency; single-GPU puts
    // all three legs here.
    auto edge_accumulator = allocate_dataframe_buffer<T>(
      static_cast<size_t>(edge_partition.number_of_edges()), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(edge_accumulator),
                 get_dataframe_buffer_end(edge_accumulator),
                 init);

    uint32_t const* edge_mask =
      edge_partition_e_mask ? (*edge_partition_e_mask).value_first() : nullptr;

    auto accumulator_first = get_dataframe_buffer_begin(edge_accumulator);

    // Second buffer: one slot per broadcast pair (multi-GPU only).
    auto per_pair_buffer = allocate_dataframe_buffer<T>(
      GraphViewType::is_multi_gpu ? num_pairs : size_t{0}, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(per_pair_buffer),
                 get_dataframe_buffer_end(per_pair_buffer),
                 init);

    // Third buffer, for the (v1, w) contributions (multi-GPU only). This is the one triangle leg
    // multi-GPU has to write out as actual edges: (v1, w) sits on another rank and there is no way
    // to atomically add into it from here, so each contribution has to be named and shipped. The
    // (v0, v1) and (v0, w) legs never leave a slot index. These are the shipped form, filled at
    // compaction time.
    rmm::device_uvector<vertex_t> remote_srcs(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> remote_dsts(size_t{0}, handle.get_stream());
    auto remote_vals = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

    // Before that, the contributions need somewhere to land, and it has to exist before the
    // intersection starts, when we do not yet know how many there will be. Intersecting once just to
    // count them would double the most expensive step, so give every (v1, w) that can possibly come
    // out a slot of its own:
    //
    //   - every (v1, w) is an edge of N(v1), and multi-GPU has to fetch N(v1) from its owner anyway,
    //     so fetch those neighbor lists first. They come back as a CSR, one row per distinct v1, with
    //     unique_majors giving row -> v1;
    //   - dense_remote_acc gets one slot per edge of that CSR. A contribution to (v1, w) is then an
    //     atomic add at that edge's position, and repeated hits pile into the same slot instead of
    //     each taking a new entry;
    //   - afterwards a filled slot is turned back into (v1, w), via unique_majors for the source and
    //     the CSR indices for the destination, and shuffled to the owner.
    //
    // nbr_intersection would fetch the very same lists itself, so we hand it ours rather than let it
    // fetch them a second time.
    rmm::device_uvector<T> dense_remote_acc(size_t{0}, handle.get_stream());
    rmm::device_uvector<edge_t> gathered_offsets(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> gathered_indices(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> gathered_unique_majors(size_t{0}, handle.get_stream());
    using gathered_e_value_t =
      std::conditional_t<std::is_same_v<typename EdgeValueInputWrapper::value_type,
                                        cuda::std::nullopt_t>,
                         std::byte,
                         typename EdgeValueInputWrapper::value_type>;
    std::unique_ptr<kv_store_t<vertex_t, vertex_t, false>> gathered_idx_map{};
    rmm::device_uvector<gathered_e_value_t> gathered_e_values(size_t{0}, handle.get_stream());

    // Fetch N(v1) for the distinct second endpoints, then size the accumulator to what came back.
    // The fifth returned element, unique_majors, was added for this path: it names the v1 behind each
    // gathered row, without which a filled slot could not be turned back into an edge.
    if constexpr (GraphViewType::is_multi_gpu) {
      auto [idx_map, offsets, indices, e_values, unique_majors] =
        detail::nbr_intersection_collect_second_nbrs(
          handle, graph_view, edge_value_input, vertex_pair_first, vertex_pair_first + num_pairs);
      gathered_idx_map       = std::move(idx_map);
      gathered_offsets       = std::move(offsets);
      gathered_indices       = std::move(indices);
      gathered_e_values      = std::move(e_values);
      gathered_unique_majors = std::move(unique_majors);

      dense_remote_acc.resize(gathered_indices.size(), handle.get_stream());
      thrust::fill(
        handle.get_thrust_policy(), dense_remote_acc.begin(), dense_remote_acc.end(), init);
    }

    // Run the intersection. It calls accumulate_triplet_op_t once per triangle it finds, which is
    // where the three buffers get filled.
    detail::nbr_intersection(
      handle,
      graph_view,
      edge_partition,
      edge_value_input,
      edge_partition_e_value_input,
      vertex_pair_first,
      vertex_pair_first + num_pairs,
      std::array<bool, 2>{true, true},
      accumulate_triplet_op_t<GraphViewType,
                              edge_partition_src_input_device_view_t,
                              edge_partition_dst_input_device_view_t,
                              IntersectionOp,
                              T,
                              decltype(accumulator_first)>{
        edge_partition,
        edge_partition_src_value_input,
        edge_partition_dst_value_input,
        intersection_op,
        accumulator_first,
        init,
        get_dataframe_buffer_begin(per_pair_buffer),
        dense_remote_acc.data()},
      edge_mask,
      // The lists fetched above; a non-null map is what tells nbr_intersection not to fetch its own.
      gathered_idx_map.get(),
      raft::device_span<edge_t const>(gathered_offsets.data(), gathered_offsets.size()),
      raft::device_span<vertex_t const>(gathered_indices.data(), gathered_indices.size()),
      raft::device_span<gathered_e_value_t const>(gathered_e_values.data(),
                                                  gathered_e_values.size()));

    // Now empty the three buffers back out into (src, dst, value) triplets. Each one is a matter of
    // working out which edge a slot stands for, and keeping only the slots that moved off init.
    //
    // The local accumulator first. Slot e is edge e of this partition's CSR, so its destination is
    // indices[e] and its source is the major whose offset range contains e (upper_bound on offsets).
    auto num_partition_edges = static_cast<size_t>(edge_partition.number_of_edges());
    auto offsets_ptr         = edge_partition.offsets();
    auto indices_ptr         = edge_partition.indices();
    auto num_majors          = edge_partition.major_range_size();
    auto major_range_first   = edge_partition.major_range_first();

    // Count the slots that received a contribution, which is the size of the compacted output.
    auto num_updated = static_cast<size_t>(thrust::count_if(
      handle.get_thrust_policy(),
      accumulator_first,
      accumulator_first + num_partition_edges,
      cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; })));

    // The edge list for those slots. These edges are local: the (v0, w) leg, plus the other two legs
    // in single-GPU.
    rmm::device_uvector<vertex_t> local_srcs(num_updated, handle.get_stream());
    rmm::device_uvector<vertex_t> local_dsts(num_updated, handle.get_stream());
    auto local_vals = allocate_dataframe_buffer<T>(num_updated, handle.get_stream());

    // No src/dst arrays are built. These iterators derive the endpoints of slot e on demand, and
    // copy_if only dereferences them at the slots its stencil accepts, so the search below runs only
    // for slots that received a contribution.
    auto src_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(edge_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [offsets_ptr, num_majors, major_range_first] __device__(edge_t e) {
          auto it = thrust::upper_bound(thrust::seq, offsets_ptr, offsets_ptr + num_majors + 1, e);
          return static_cast<vertex_t>(major_range_first +
                                       (thrust::distance(offsets_ptr, it) - edge_t{1}));
        }));
    auto dst_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(edge_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [indices_ptr] __device__(edge_t e) { return indices_ptr[e]; }));
    auto input_first  = thrust::make_zip_iterator(src_first, dst_first, accumulator_first);
    auto output_first = thrust::make_zip_iterator(
      local_srcs.begin(), local_dsts.begin(), get_dataframe_buffer_begin(local_vals));

    // Where the copy actually happens: only the slots that moved off init are written out, and the
    // iterators above are evaluated at exactly those slots.
    thrust::copy_if(handle.get_thrust_policy(),
                    input_first,
                    input_first + num_partition_edges,
                    accumulator_first,
                    output_first,
                    cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; }));

    // The remote accumulator next. Each filled slot has to be named before it can be shipped. Slot e
    // sits in some row of the fetched CSR, found by searching gathered_offsets. That row gives the
    // source, gathered_unique_majors[row], and the slot itself gives the destination,
    // gathered_indices[e]. Since every edge had its own slot, no edge appears twice here.
    size_t num_remote = 0;
    if constexpr (GraphViewType::is_multi_gpu) {
      auto num_slots = dense_remote_acc.size();
      num_remote     = static_cast<size_t>(thrust::count_if(
        handle.get_thrust_policy(),
        dense_remote_acc.begin(),
        dense_remote_acc.end(),
        cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; })));
      remote_srcs.resize(num_remote, handle.get_stream());
      remote_dsts.resize(num_remote, handle.get_stream());
      resize_dataframe_buffer(remote_vals, num_remote, handle.get_stream());

      // All from the fetched second-endpoint lists: one row per distinct v1, its offsets, the v1 each
      // row belongs to, and the neighbors themselves.
      auto offs_ptr = gathered_offsets.data();
      auto n_rows   = gathered_offsets.size() > 0 ? gathered_offsets.size() - 1 : size_t{0};
      auto uniq_ptr = gathered_unique_majors.data();
      auto idx_ptr  = gathered_indices.data();
      auto rsrc_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(edge_t{0}),
        cuda::proclaim_return_type<vertex_t>([offs_ptr, n_rows, uniq_ptr] __device__(edge_t e) {
          auto it  = thrust::upper_bound(thrust::seq, offs_ptr, offs_ptr + n_rows + 1, e);
          auto row = thrust::distance(offs_ptr, it) - edge_t{1};
          return uniq_ptr[row];
        }));
      auto rdst_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(edge_t{0}),
        cuda::proclaim_return_type<vertex_t>(
          [idx_ptr] __device__(edge_t e) { return idx_ptr[e]; }));
      auto rin  = thrust::make_zip_iterator(rsrc_first, rdst_first, dense_remote_acc.begin());
      auto rout = thrust::make_zip_iterator(
        remote_srcs.begin(), remote_dsts.begin(), get_dataframe_buffer_begin(remote_vals));
      // Read the accumulator and write out each filled slot as an edge with its contribution. Scan in
      // <= (1 << 27) tiles: thrust::copy_if overflows 32-bit indices
      // (https://github.com/NVIDIA/thrust/issues/1302) once num_slots exceeds ~2^31. count_if above
      // is 64-bit safe and already sized the output, and num_copied carries the write position from
      // one tile to the next.
      size_t num_scanned = 0;
      size_t num_copied  = 0;
      while (num_scanned < num_slots) {
        size_t this_scan = std::min(size_t{1} << 27, num_slots - num_scanned);
        num_copied += static_cast<size_t>(cuda::std::distance(
          rout + num_copied,
          thrust::copy_if(
            handle.get_thrust_policy(),
            rin + num_scanned,
            rin + num_scanned + this_scan,
            dense_remote_acc.begin() + num_scanned,
            rout + num_copied,
            cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; }))));
        num_scanned += this_scan;
      }
    }

    // The base buffer last. Nothing has to be looked up here, since slot p simply belongs to
    // broadcast pair p, whose edge is (majors[p], minors[p]).
    rmm::device_uvector<vertex_t> base_srcs(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> base_dsts(size_t{0}, handle.get_stream());
    auto base_vals  = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());
    size_t num_base = 0;
    if constexpr (GraphViewType::is_multi_gpu) {
      auto per_pair_first = get_dataframe_buffer_begin(per_pair_buffer);
      num_base            = static_cast<size_t>(thrust::count_if(
        handle.get_thrust_policy(),
        per_pair_first,
        per_pair_first + num_pairs,
        cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; })));
      base_srcs.resize(num_base, handle.get_stream());
      base_dsts.resize(num_base, handle.get_stream());
      resize_dataframe_buffer(base_vals, num_base, handle.get_stream());
      auto base_in =
        thrust::make_zip_iterator(majors_view.data(), minors_view.data(), per_pair_first);
      auto base_out = thrust::make_zip_iterator(
        base_srcs.begin(), base_dsts.begin(), get_dataframe_buffer_begin(base_vals));
      thrust::copy_if(
        handle.get_thrust_policy(),
        base_in,
        base_in + num_pairs,
        per_pair_first,
        base_out,
        cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; }));
    }

    // All three sets are now plain (src, dst, value) triplets and nothing distinguishes them
    // anymore, so they go into one output. Appending rather than overwriting is what lets the caller
    // pour more than one batch into the same buffer, which single-GPU does across its three sets and
    // multi-GPU does only within a chunk, since a chunk is settled before the next one starts.
    auto old_size = out_srcs.size();
    out_srcs.resize(old_size + num_updated + num_remote + num_base, handle.get_stream());
    out_dsts.resize(old_size + num_updated + num_remote + num_base, handle.get_stream());
    resize_dataframe_buffer(
      out_values, old_size + num_updated + num_remote + num_base, handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), local_srcs.begin(), local_srcs.end(), out_srcs.begin() + old_size);
    thrust::copy(
      handle.get_thrust_policy(), local_dsts.begin(), local_dsts.end(), out_dsts.begin() + old_size);
    thrust::copy(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(local_vals),
                 get_dataframe_buffer_end(local_vals),
                 get_dataframe_buffer_begin(out_values) + old_size);
    if constexpr (GraphViewType::is_multi_gpu) {
      thrust::copy(handle.get_thrust_policy(),
                   remote_srcs.begin(),
                   remote_srcs.end(),
                   out_srcs.begin() + old_size + num_updated);
      thrust::copy(handle.get_thrust_policy(),
                   remote_dsts.begin(),
                   remote_dsts.end(),
                   out_dsts.begin() + old_size + num_updated);
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(remote_vals),
                   get_dataframe_buffer_end(remote_vals),
                   get_dataframe_buffer_begin(out_values) + old_size + num_updated);
      thrust::copy(handle.get_thrust_policy(),
                   base_srcs.begin(),
                   base_srcs.end(),
                   out_srcs.begin() + old_size + num_updated + num_remote);
      thrust::copy(handle.get_thrust_policy(),
                   base_dsts.begin(),
                   base_dsts.end(),
                   out_dsts.begin() + old_size + num_updated + num_remote);
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(base_vals),
                   get_dataframe_buffer_end(base_vals),
                   get_dataframe_buffer_begin(out_values) + old_size + num_updated + num_remote);
    }
}

// The settling step every partial has been waiting for. agg_* holds a batch of (src, dst, value)
// contributions, with the same edge appearing as many times as it was touched. This collapses them
// to one value per edge. Single-GPU is already done, since one rank saw every triangle. Multi-GPU
// has to send each contribution to the rank that owns the edge first, and what it returns is one
// row per owned edge, sorted by (src, dst).
template <typename GraphViewType, typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
finalize_triplet_reduction(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  rmm::device_uvector<typename GraphViewType::vertex_type> agg_srcs,
  rmm::device_uvector<typename GraphViewType::vertex_type> agg_dsts,
  dataframe_buffer_type_t<T> agg_values)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  rmm::device_uvector<vertex_t> result_srcs(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(size_t{0}, handle.get_stream());
  auto result_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

  if constexpr (!GraphViewType::is_multi_gpu) {
    // One edge partition, one atomic add per contribution, so the triplets are already final.
    result_srcs   = std::move(agg_srcs);
    result_dsts   = std::move(agg_dsts);
    result_values = std::move(agg_values);
  } else {
    // Multi-GPU: send every contribution to the rank that owns its edge, then add up what arrives.
    // This is where a triangle counted on one rank finally meets the same edge's contributions from
    // every other rank.
    //
    // Everything is shuffled the same way, including the local contributions, whose edges this rank
    // already owns and which could in principle stay home and be merged in after the reduce. That
    // would trade a smaller shuffle for a second merge path; the uniform shuffle is kept for
    // simplicity.
    auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
    rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
      h_vertex_partition_range_lasts.size(), handle.get_stream());
    raft::update_device(d_vertex_partition_range_lasts.data(),
                        h_vertex_partition_range_lasts.data(),
                        h_vertex_partition_range_lasts.size(),
                        handle.get_stream());
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    // Reduce locally first. The same edge can appear many times in agg_*, once per triangle leg that
    // landed on it, and there is no point paying to send each of those separately: one row per edge
    // goes over the wire instead. The reduce after the shuffle still does the real merge across
    // ranks.
    {
      auto agg_edge_first = thrust::make_zip_iterator(agg_srcs.begin(), agg_dsts.begin());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          agg_edge_first,
                          agg_edge_first + agg_srcs.size(),
                          get_dataframe_buffer_begin(agg_values));
      rmm::device_uvector<vertex_t> reduced_agg_srcs(agg_srcs.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> reduced_agg_dsts(agg_srcs.size(), handle.get_stream());
      auto reduced_agg_values = allocate_dataframe_buffer<T>(agg_srcs.size(), handle.get_stream());
      auto agg_reduced_end    = thrust::reduce_by_key(
        handle.get_thrust_policy(),
        agg_edge_first,
        agg_edge_first + agg_srcs.size(),
        get_dataframe_buffer_begin(agg_values),
        thrust::make_zip_iterator(reduced_agg_srcs.begin(), reduced_agg_dsts.begin()),
        get_dataframe_buffer_begin(reduced_agg_values));
      auto num_agg_reduced = static_cast<size_t>(
        thrust::distance(thrust::make_zip_iterator(reduced_agg_srcs.begin(), reduced_agg_dsts.begin()),
                         agg_reduced_end.first));
      reduced_agg_srcs.resize(num_agg_reduced, handle.get_stream());
      reduced_agg_dsts.resize(num_agg_reduced, handle.get_stream());
      resize_dataframe_buffer(reduced_agg_values, num_agg_reduced, handle.get_stream());
      agg_srcs   = std::move(reduced_agg_srcs);
      agg_dsts   = std::move(reduced_agg_dsts);
      agg_values = std::move(reduced_agg_values);
    }

    // The shuffle itself. The triplet travels as one value buffer (a zip of src, dst, value), routed
    // by the edge endpoints it carries.
    auto edge_triplet_first = thrust::make_zip_iterator(
      agg_srcs.begin(), agg_dsts.begin(), get_dataframe_buffer_begin(agg_values));
    auto edge_triplet_last = thrust::make_zip_iterator(
      agg_srcs.end(), agg_dsts.end(), get_dataframe_buffer_end(agg_values));
    auto [rx_buffer, rx_counts] = groupby_gpu_id_and_shuffle_values(
      comm,
      edge_triplet_first,
      edge_triplet_last,
      compute_gpu_id_from_edge_endpoints_in_triplet_t<vertex_t>{
        cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
          raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                            d_vertex_partition_range_lasts.size()),
          comm_size,
          major_comm_size,
          minor_comm_size}},
      handle.get_stream());
    static_cast<void>(rx_counts);
    auto& rx_srcs = std::get<0>(rx_buffer);
    auto& rx_dsts = std::get<1>(rx_buffer);
    auto& rx_vals = std::get<2>(rx_buffer);

    // Everything for a given edge is now on one rank, so summing per edge gives the final value.
    auto rx_edge_first = thrust::make_zip_iterator(rx_srcs.begin(), rx_dsts.begin());
    thrust::sort_by_key(handle.get_thrust_policy(),
                        rx_edge_first,
                        rx_edge_first + rx_srcs.size(),
                        get_dataframe_buffer_begin(rx_vals));
    result_srcs.resize(rx_srcs.size(), handle.get_stream());
    result_dsts.resize(rx_srcs.size(), handle.get_stream());
    resize_dataframe_buffer(result_values, rx_srcs.size(), handle.get_stream());
    auto reduced_end = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      rx_edge_first,
      rx_edge_first + rx_srcs.size(),
      get_dataframe_buffer_begin(rx_vals),
      thrust::make_zip_iterator(result_srcs.begin(), result_dsts.begin()),
      get_dataframe_buffer_begin(result_values));
    auto num_reduced = static_cast<size_t>(
      thrust::distance(thrust::make_zip_iterator(result_srcs.begin(), result_dsts.begin()),
                       reduced_end.first));
    result_srcs.resize(num_reduced, handle.get_stream());
    result_dsts.resize(num_reduced, handle.get_stream());
    resize_dataframe_buffer(result_values, num_reduced, handle.get_stream());
  }

  return std::make_tuple(
    std::move(result_srcs), std::move(result_dsts), std::move(result_values));
}

// Fold a settled batch into this rank's running result.
//
// Both sides are sorted by (src, dst) and hold one row per edge: that is how
// finalize_triplet_reduction leaves its output, and this function hands back the same shape, so the
// invariant survives every fold. That is what makes the cheap path available -- two sorted runs are
// merged and the edges they share are collapsed in one linear pass, instead of re-sorting an
// accumulator that grows with every batch. Every row is already owned by this rank, so nothing here
// touches the network.
template <typename GraphViewType, typename T>
void merge_settled_rows(raft::handle_t const& handle,
                        rmm::device_uvector<typename GraphViewType::vertex_type>& result_srcs,
                        rmm::device_uvector<typename GraphViewType::vertex_type>& result_dsts,
                        dataframe_buffer_type_t<T>& result_values,
                        rmm::device_uvector<typename GraphViewType::vertex_type> batch_srcs,
                        rmm::device_uvector<typename GraphViewType::vertex_type> batch_dsts,
                        dataframe_buffer_type_t<T> batch_values)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (batch_srcs.size() == 0) { return; }
  if (result_srcs.size() == 0) {
    result_srcs   = std::move(batch_srcs);
    result_dsts   = std::move(batch_dsts);
    result_values = std::move(batch_values);
    return;
  }

  auto const merged_size = result_srcs.size() + batch_srcs.size();
  rmm::device_uvector<vertex_t> merged_srcs(merged_size, handle.get_stream());
  rmm::device_uvector<vertex_t> merged_dsts(merged_size, handle.get_stream());
  auto merged_values = allocate_dataframe_buffer<T>(merged_size, handle.get_stream());
  auto result_first  = thrust::make_zip_iterator(result_srcs.begin(), result_dsts.begin());
  auto batch_first   = thrust::make_zip_iterator(batch_srcs.begin(), batch_dsts.begin());
  thrust::merge_by_key(handle.get_thrust_policy(),
                       result_first,
                       result_first + result_srcs.size(),
                       batch_first,
                       batch_first + batch_srcs.size(),
                       get_dataframe_buffer_begin(result_values),
                       get_dataframe_buffer_begin(batch_values),
                       thrust::make_zip_iterator(merged_srcs.begin(), merged_dsts.begin()),
                       get_dataframe_buffer_begin(merged_values));

  // An edge the two runs have in common now sits in two adjacent rows, so one pass over the merge
  // adds them together.
  rmm::device_uvector<vertex_t> reduced_srcs(merged_size, handle.get_stream());
  rmm::device_uvector<vertex_t> reduced_dsts(merged_size, handle.get_stream());
  auto reduced_values = allocate_dataframe_buffer<T>(merged_size, handle.get_stream());
  auto merged_first   = thrust::make_zip_iterator(merged_srcs.begin(), merged_dsts.begin());
  auto reduced_end    = thrust::reduce_by_key(
    handle.get_thrust_policy(),
    merged_first,
    merged_first + merged_size,
    get_dataframe_buffer_begin(merged_values),
    thrust::make_zip_iterator(reduced_srcs.begin(), reduced_dsts.begin()),
    get_dataframe_buffer_begin(reduced_values));
  auto const num_reduced = static_cast<size_t>(cuda::std::distance(
    thrust::make_zip_iterator(reduced_srcs.begin(), reduced_dsts.begin()), reduced_end.first));
  reduced_srcs.resize(num_reduced, handle.get_stream());
  reduced_dsts.resize(num_reduced, handle.get_stream());
  resize_dataframe_buffer(reduced_values, num_reduced, handle.get_stream());

  result_srcs   = std::move(reduced_srcs);
  result_dsts   = std::move(reduced_dsts);
  result_values = std::move(reduced_values);
}

// One partition's worth of work, in as many pieces as the memory budget asks for. Both drivers go
// through here, so neither of them has to know about chunking beyond having a degree store to hand
// over.
//
// Multi-GPU does not carry a chunk's raw contributions any further than it has to. As soon as a
// chunk has been intersected, its triplets are sent to the edge owners and reduced, and only the
// resulting owned rows are folded into result_*. So the largest thing alive at any moment is one
// chunk's contributions plus this rank's share of the answer (~E/p), instead of every contribution
// the rank will ever produce. Settling per chunk costs one shuffle per chunk rather than one per
// rank, which is the price of that bound; the chunk count is agreed on globally, so those shuffles
// stay in step.
//
// Single-GPU has nothing to fetch, nothing to bound and nobody to send to, so it goes through in one
// piece and its contributions are settled once by the caller.
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename T>
void accumulate_triplets_for_partition_in_chunks(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  size_t i,
  owned_degrees_t<GraphViewType> const* degrees,
  raft::device_span<typename GraphViewType::vertex_type const> majors,
  raft::device_span<typename GraphViewType::vertex_type const> minors,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  rmm::device_uvector<typename GraphViewType::vertex_type>& result_srcs,
  rmm::device_uvector<typename GraphViewType::vertex_type>& result_dsts,
  dataframe_buffer_type_t<T>& result_values)
{
  using vertex_t = typename GraphViewType::vertex_type;

  // Single-GPU: the whole partition goes over in one piece. There is nothing to fetch, so nothing to
  // budget, and nobody to send results to, so the contributions land in result_* exactly as they come
  // out and the caller settles them at the end. Everything below this branch is multi-GPU only.
  if constexpr (!GraphViewType::is_multi_gpu) {
    accumulate_triplets_for_partition(handle,
                                      graph_view,
                                      i,
                                      majors,
                                      minors,
                                      edge_src_value_input,
                                      edge_dst_value_input,
                                      edge_value_input,
                                      intersection_op,
                                      init,
                                      result_srcs,
                                      result_dsts,
                                      result_values);
  } else {
    // Each unit of degree in a chunk costs one neighbor id plus one accumulator slot. The factor
    // covers the temporaries the gather and the compaction allocate on top of those two arrays.
    constexpr double transient_factor = 2.5;
    auto const bytes_per_degree =
      static_cast<double>(sizeof(vertex_t) + sizeof(T)) * transient_factor;

    // Decide where to cut this partition into chunks. The planner only ever sees partition i's own
    // pair list, so every chunk it produces is a slice of that one partition and the same i stays
    // valid for all of them. In order, it:
    //   - collects the distinct second endpoints in the list and asks their owners for their
    //     degrees, since the cost of a chunk is the sum of those degrees, not the number of pairs;
    //   - shares that (endpoint, degree) list across the row, so every rank cuts in the same places;
    //   - divides the total degree by the budget to get a chunk count, and raises it to the largest
    //     count any rank asked for, so nobody runs a different number of collectives;
    //   - hands each pair a chunk number based on where its second endpoint falls in the running
    //     degree total, then reorders the pairs so that a chunk is one contiguous run;
    //   - sorts each run, because the intersection needs sorted pairs.
    //
    // Nothing is discarded: every pair comes back, just in a different order. What comes back is the
    // reordered pairs and `bounds`, which holds one more entry than there are chunks, chunk c being
    // the half-open range bounds[c] .. bounds[c + 1]. Chunks hold similar total degree, not similar
    // numbers of pairs, so a chunk aimed at hubs is short and one aimed at leaves is long.
    auto [chunked_majors, chunked_minors, bounds] = plan_partition_chunks<GraphViewType>(
      handle, *degrees, majors, minors, triplet_chunk_budget_bytes(), bytes_per_degree);

    // A single chunk leaves the pairs where they were, and the planner says so by returning empty
    // arrays; anything more comes back reordered, and those are the arrays to read.
    auto const* chunk_majors = chunked_majors.size() > 0 ? chunked_majors.data() : majors.data();
    auto const* chunk_minors = chunked_minors.size() > 0 ? chunked_minors.data() : minors.data();

    // One pass per chunk, walking `bounds` two entries at a time.
    for (size_t c = 0; c + 1 < bounds.size(); ++c) {
      auto const first = std::min(bounds[c], majors.size());
      auto const last  = std::min(bounds[c + 1], majors.size());
      auto const size  = last > first ? last - first : size_t{0};

      // Fresh buffers each time round, so a chunk's contributions are released the moment they have
      // been settled. Empty chunks still go through, because the chunk count was agreed on globally
      // and everything below is full of collectives.
      rmm::device_uvector<vertex_t> chunk_srcs(size_t{0}, handle.get_stream());
      rmm::device_uvector<vertex_t> chunk_dsts(size_t{0}, handle.get_stream());
      auto chunk_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());
      accumulate_triplets_for_partition(
        handle,
        graph_view,
        i,
        raft::device_span<vertex_t const>(chunk_majors + first, size),
        raft::device_span<vertex_t const>(chunk_minors + first, size),
        edge_src_value_input,
        edge_dst_value_input,
        edge_value_input,
        intersection_op,
        init,
        chunk_srcs,
        chunk_dsts,
        chunk_values);

      auto [settled_srcs, settled_dsts, settled_values] =
        finalize_triplet_reduction<GraphViewType, T>(
          handle, graph_view, std::move(chunk_srcs), std::move(chunk_dsts), std::move(chunk_values));
      merge_settled_rows<GraphViewType, T>(handle,
                                           result_srcs,
                                           result_dsts,
                                           result_values,
                                           std::move(settled_srcs),
                                           std::move(settled_dsts),
                                           std::move(settled_values));
    }
  }
}

// The driver behind all four public entry points, in the all-edges form: every edge of graph_view is
// a pair whose endpoints get intersected.
//
// A rank's edges do not sit in one CSR but in one or more local edge partitions, so the work is done
// a partition at a time:
//   1. rebuild that partition's edges as an explicit (major, minor) list, because the intersection
//      consumes pairs, not a CSR;
//   2. hand the list over in memory-budgeted chunks, each of which finds its triangles and, in
//      multi-GPU, sends the results to the edge owners on the spot before folding them into result_*;
//   3. single-GPU had nobody to send to, so its raw contributions are settled at the end.
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = float;  // dummy

  if (do_expensive_check) {
    // currently, nothing to do.
  }

  // What every partition writes into. In multi-GPU this is one row per edge this rank owns, kept
  // sorted and reduced as chunks fold into it; in single-GPU it is the raw contributions, settled by
  // step 3. Reduction values are held in allocate_dataframe_buffer<T> so a tuple value type T is
  // supported in the future.
  rmm::device_uvector<vertex_t> result_srcs(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(size_t{0}, handle.get_stream());
  auto result_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

  // Chunking each partition needs the degrees of second endpoints that live elsewhere, so the store
  // that answers those queries is built once here and reused by every partition.
  std::optional<detail::owned_degrees_t<GraphViewType>> owned_degrees{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    owned_degrees = detail::build_owned_degrees(handle, graph_view);
  }

  auto edge_mask_view = graph_view.edge_mask_view();

  // The partition loop sits here rather than inside nbr_intersection (where the materializing
  // overload keeps it) so that nbr_intersection stays thin and operator-agnostic; moving it in would
  // drag this primitive's buffer machinery along with it. The price is that nbr_intersection has no
  // partition index of its own, so we later hand it the built edge_partition and its slice of the
  // edge mask.
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : std::nullopt;

    // Step 1. Decompress the partition's CSR into a pair list. When a mask is active only the live
    // edges are wanted, so the count of set bits is the length rather than the edge count.
    rmm::device_uvector<vertex_t> majors(
      edge_partition_e_mask
        ? detail::count_set_bits(
            handle, (*edge_partition_e_mask).value_first(), edge_partition.number_of_edges())
        : static_cast<size_t>(edge_partition.number_of_edges()),
      handle.get_stream());
    rmm::device_uvector<vertex_t> minors(majors.size(), handle.get_stream());

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    detail::decompress_edge_partition_to_edgelist<vertex_t,
                                                  edge_t,
                                                  weight_t,
                                                  int32_t,
                                                  GraphViewType::is_multi_gpu>(
      handle,
      edge_partition,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      edge_partition_e_mask,
      raft::device_span<vertex_t>(majors.data(), majors.size()),
      raft::device_span<vertex_t>(minors.data(), minors.size()),
      std::nullopt,
      std::nullopt,
      std::nullopt,
      segment_offsets);

    // Step 2. The pairs are already in CSR order, hence sorted, which is what the intersection
    // expects. Chunking may reorder them, but only into runs that are each sorted again after the
    // broadcast.
    accumulate_triplets_for_partition_in_chunks(
      handle,
      graph_view,
      i,
      owned_degrees ? &(*owned_degrees) : nullptr,
      raft::device_span<vertex_t const>(majors.data(), majors.size()),
      raft::device_span<vertex_t const>(minors.data(), minors.size()),
      edge_src_value_input,
      edge_dst_value_input,
      edge_value_input,
      intersection_op,
      init,
      result_srcs,
      result_dsts,
      result_values);
  }

  // Step 3. Multi-GPU settled every chunk as it went, so result_* is already one value per owned
  // edge and there is nothing left to do. Single-GPU still holds raw contributions.
  if constexpr (GraphViewType::is_multi_gpu) {
    return std::make_tuple(
      std::move(result_srcs), std::move(result_dsts), std::move(result_values));
  } else {
    return finalize_triplet_reduction<GraphViewType, T>(
      handle, graph_view, std::move(result_srcs), std::move(result_dsts), std::move(result_values));
  }
}

// The same driver in subset form: the caller supplies the pairs instead of us reading every edge
// (k-truss peeling, for one, passes only the edges still alive this iteration). Steps 2 and 3 are
// untouched. Only step 1 changes, because there is no CSR to decompress; the caller's pairs have to
// be split by which local edge partition owns each one, which takes two passes:
//   1a. number the pairs 0..n-1 and group the numbers by partition, so no pair data moves yet;
//   1b. per partition, gather that group's pairs and sort them, which the all-edges form got for
//       free from CSR order.
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename VertexPairIterator,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPairIterator vertex_pair_first,
  VertexPairIterator vertex_pair_last,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  if (do_expensive_check) {
    // currently, nothing to do.
  }

  // Same output as the all-edges form: owned rows in multi-GPU, raw contributions in single-GPU.
  rmm::device_uvector<vertex_t> result_srcs(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(size_t{0}, handle.get_stream());
  auto result_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

  // Same degree store as the all-edges form, built once and reused by every partition.
  std::optional<detail::owned_degrees_t<GraphViewType>> owned_degrees{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    owned_degrees = detail::build_owned_degrees(handle, graph_view);
  }

  // Step 1a. The caller provides a flat list of pairs for this GPU, which must be processed one
  // local edge partition at a time. They are grouped by partition through their indices
  // `0, 1, ..., num_input_pairs - 1` instead of moving the actual pair data around.
  auto num_input_pairs =
    static_cast<size_t>(cuda::std::distance(vertex_pair_first, vertex_pair_last));

  rmm::device_uvector<size_t> vertex_pair_indices(num_input_pairs, handle.get_stream());
  thrust::sequence(
    handle.get_thrust_policy(), vertex_pair_indices.begin(), vertex_pair_indices.end(), size_t{0});

  // Each local edge partition covers one contiguous range of majors, so collecting where those
  // ranges end is enough to tell which partition a pair belongs to: look up its major. Under
  // transposed storage the major is the destination, otherwise the source.
  std::vector<vertex_t> h_major_range_lasts(graph_view.number_of_local_edge_partitions());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    if constexpr (GraphViewType::is_storage_transposed) {
      h_major_range_lasts[i] = graph_view.local_edge_partition_dst_range_last(i);
    } else {
      h_major_range_lasts[i] = graph_view.local_edge_partition_src_range_last(i);
    }
  }
  
  // Copy the range ends to the device so the grouping key below can search them.
  rmm::device_uvector<vertex_t> d_major_range_lasts(h_major_range_lasts.size(),
                                                    handle.get_stream());
  raft::update_device(d_major_range_lasts.data(),
                      h_major_range_lasts.data(),
                      h_major_range_lasts.size(),
                      handle.get_stream());

  // Count how many pairs belong to each local edge partition. This also reorders
  // `vertex_pair_indices` into contiguous chunks: all indices belonging to partition 0 come first,
  // then partition 1, etc., so that the partition loop below can `thrust::gather` an entire
  // partition's pairs in a single shot.
  auto d_group_sizes = groupby_and_count(
    vertex_pair_indices.begin(),
    vertex_pair_indices.end(),
    detail::compute_local_edge_partition_id_t<VertexPairIterator>{
      vertex_pair_first,
      graph_view.number_of_local_edge_partitions(),
      raft::device_span<vertex_t const>(d_major_range_lasts.data(), d_major_range_lasts.size())},
    static_cast<int>(graph_view.number_of_local_edge_partitions()),
    std::numeric_limits<size_t>::max(),
    handle.get_stream());

  std::vector<size_t> h_group_sizes(d_group_sizes.size());
  raft::update_host(
    h_group_sizes.data(), d_group_sizes.data(), d_group_sizes.size(), handle.get_stream());
  handle.sync_stream();
  std::vector<size_t> h_group_displacements(h_group_sizes.size());
  std::exclusive_scan(
    h_group_sizes.begin(), h_group_sizes.end(), h_group_displacements.begin(), size_t{0});

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto group_first = vertex_pair_indices.begin() + h_group_displacements[i];
    auto group_size  = h_group_sizes[i];

    // Step 1b. Only now do the actual pairs move: gather this partition's pairs out of the caller's
    // iterator using the grouped indices. Then sort them, since grouping by partition says nothing
    // about order within a group, and the intersection expects sorted input. Note: this is a local
    // memory gather (A[i] = B[idx[i]]), not a network gather.
    rmm::device_uvector<vertex_t> majors(group_size, handle.get_stream());
    rmm::device_uvector<vertex_t> minors(group_size, handle.get_stream());
    auto pair_first = thrust::make_zip_iterator(majors.begin(), minors.begin());
    thrust::gather(handle.get_thrust_policy(),
                   group_first,
                   group_first + group_size,
                   vertex_pair_first,
                   pair_first);
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + group_size);

    // Step 2. Hand this partition's pairs to the shared core, in memory-budgeted chunks. Each chunk
    // is broadcast across the row comm, has its remote neighbor lists fetched, is intersected, and
    // contributes to result_*. This is exactly what the all-edges form does for its decompressed
    // pairs.
    accumulate_triplets_for_partition_in_chunks(
      handle,
      graph_view,
      i,
      owned_degrees ? &(*owned_degrees) : nullptr,
      raft::device_span<vertex_t const>(majors.data(), majors.size()),
      raft::device_span<vertex_t const>(minors.data(), minors.size()),
      edge_src_value_input,
      edge_dst_value_input,
      edge_value_input,
      intersection_op,
      init,
      result_srcs,
      result_dsts,
      result_values);
  }

  // Step 3, likewise.
  if constexpr (GraphViewType::is_multi_gpu) {
    return std::make_tuple(
      std::move(result_srcs), std::move(result_dsts), std::move(result_values));
  } else {
    return finalize_triplet_reduction<GraphViewType, T>(
      handle, graph_view, std::move(result_srcs), std::move(result_dsts), std::move(result_values));
  }
}


}  // namespace detail

/**
 * @brief Iterate over each edge and apply a functor to each vertex in the common source neighbor
 * list of the edge endpoints, reduce the functor output values per-edge.
 *
 * Iterate over every edge; intersect source neighbor lists of source vertex & destination vertex;
 * invoke a user-provided functor once per vertex r in the intersection (i.e. once per
 * (edge, intersection vertex) triplet), and reduce the functor output values (cuda::std::tuple of
 * two values having the same type: one for the edge (src, dst), and one for each supporting edge
 * (the (src, r) & (dst, r) edges)) per-edge. The functor is invoked once per vertex r in the
 * intersection, so it can emit a different value for each (edge, r) triplet. We may add a per-edge
 * variant (transform_reduce_src_nbr_intersection_of_e_endpoints_by_e) in the future that invokes
 * the functor once per edge with the full intersection list, for callers whose emitted value does
 * not vary per intersection vertex. The functor output values are reduced per-edge by summation
 * (each @p intersection_op return value is added into @p init); we may add a reduce_op parameter in
 * the future to support other reductions (e.g. minimum or maximum), as in cugraph's
 * per_v_transform_reduce_* primitives. This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for supporting-edge property values (the
 * (src, r) and (dst, r) edges); use cugraph::edge_dummy_property_t::view() if @p intersection_op
 * does not access supporting-edge property values.
 * @tparam IntersectionOp Type of the septenary per (edge, intersection vertex) operator.
 * @tparam T Type of the per-edge reduction value. Currently restricted to a scalar arithmetic type:
 * contributions are accumulated with cuda::atomic_ref<T>::fetch_add and merged with a plain
 * reduce_by_key (cuda::std::plus). Supporting a tuple T (or non-additive reductions) would accumulate
 * with cugraph::atomic_add (which handles a tuple T element-wise) and reduce with
 * property_op<T, cuda::std::plus> or a reduce_op, as transform_reduce_e and
 * transform_reduce_src_dst_nbr_intersection_of_e_endpoints_by_v do.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p intersection_op needs to access source property values) or
 * cugraph::edge_src_dummy_property_t::view() (if @p intersection_op does not access source property
 * values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p intersection_op needs to access destination property
 * values) or cugraph::edge_dst_dummy_property_t::view() (if @p intersection_op does not access
 * destination property values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access supporting-edge property values for the (src, r)
 * and (dst, r) edges. Use cugraph::edge_property_t::view() (if @p intersection_op needs them) or
 * cugraph::edge_dummy_property_t::view() (if it does not).
 * @param intersection_op septenary operator takes edge source, edge destination, property values for
 * the source, property values for the destination, one vertex r in the intersection of edge
 * source & destination vertices' source neighbors, and the property values of the supporting edges
 * (src, r) and (dst, r) (cuda::std::nullopt when @p edge_value_input is a dummy property), and
 * returns a cuda::std::tuple of two values: one value for the edge (src, dst) and one value for each
 * supporting edge (src, r) and (dst, r).
 * @param init Initial value to be added to the reduced @p intersection_op return values for each
 * edge.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of three device vectors (srcs, dsts, values): for each edge with a non-init reduced
 * value, its source vertex, destination vertex, and reduced value.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_src_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  static_assert(GraphViewType::is_storage_transposed);

  return detail::transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
    handle,
    graph_view,
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    intersection_op,
    init,
    do_expensive_check);
}

/**
 * @brief Iterate over each edge and apply a functor to each vertex in the common destination
 * neighbor list of the edge endpoints, reduce the functor output values per-edge.
 *
 * Iterate over every edge; intersect destination neighbor lists of source vertex & destination
 * vertex; invoke a user-provided functor once per vertex r in the intersection (i.e. once per
 * (edge, intersection vertex) triplet), and reduce the functor output values (cuda::std::tuple of
 * two values having the same type: one for the edge (src, dst), and one for each supporting edge
 * (the (src, r) & (dst, r) edges)) per-edge. The functor is invoked once per vertex r in the
 * intersection, so it can emit a different value for each (edge, r) triplet. We may add a per-edge
 * variant (transform_reduce_dst_nbr_intersection_of_e_endpoints_by_e) in the future that invokes
 * the functor once per edge with the full intersection list, for callers whose emitted value does
 * not vary per intersection vertex. The functor output values are reduced per-edge by summation
 * (each @p intersection_op return value is added into @p init); we may add a reduce_op parameter in
 * the future to support other reductions (e.g. minimum or maximum), as in cugraph's
 * per_v_transform_reduce_* primitives. This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for supporting-edge property values (the
 * (src, r) and (dst, r) edges); use cugraph::edge_dummy_property_t::view() if @p intersection_op
 * does not access supporting-edge property values.
 * @tparam IntersectionOp Type of the septenary per (edge, intersection vertex) operator.
 * @tparam T Type of the per-edge reduction value. Currently restricted to a scalar arithmetic type:
 * contributions are accumulated with cuda::atomic_ref<T>::fetch_add and merged with a plain
 * reduce_by_key (cuda::std::plus). Supporting a tuple T (or non-additive reductions) would accumulate
 * with cugraph::atomic_add (which handles a tuple T element-wise) and reduce with
 * property_op<T, cuda::std::plus> or a reduce_op, as transform_reduce_e and
 * transform_reduce_src_dst_nbr_intersection_of_e_endpoints_by_v do.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p intersection_op needs to access source property values) or
 * cugraph::edge_src_dummy_property_t::view() (if @p intersection_op does not access source property
 * values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p intersection_op needs to access destination property
 * values) or cugraph::edge_dst_dummy_property_t::view() (if @p intersection_op does not access
 * destination property values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access supporting-edge property values for the (src, r)
 * and (dst, r) edges. Use cugraph::edge_property_t::view() (if @p intersection_op needs them) or
 * cugraph::edge_dummy_property_t::view() (if it does not).
 * @param intersection_op septenary operator takes edge source, edge destination, property values for
 * the source, property values for the destination, one vertex r in the intersection of edge
 * source & destination vertices' destination neighbors, and the property values of the supporting
 * edges (src, r) and (dst, r) (cuda::std::nullopt when @p edge_value_input is a dummy property), and
 * returns a cuda::std::tuple of two values: one value for the edge (src, dst) and one value for each
 * supporting edge (src, r) and (dst, r).
 * @param init Initial value to be added to the reduced @p intersection_op return values for each
 * edge.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of three device vectors (srcs, dsts, values): for each edge with a non-init reduced
 * value, its source vertex, destination vertex, and reduced value.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_dst_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed);

  return detail::transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
    handle,
    graph_view,
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    intersection_op,
    init,
    do_expensive_check);
}

/**
 * @brief Same as transform_reduce_triplet_of_src_nbr_intersection_of_e_endpoints_by_e, but restricted
 * to a caller-supplied list of edges (vertex pairs) instead of every edge in @p graph_view.
 *
 * For each input (src, dst) pair, intersect the source neighbor lists of src & dst, invoke @p
 * intersection_op once per vertex r in the intersection, and reduce the functor output values
 * per-edge (identical semantics to the all-edges overload). This is the subset form used by, e.g.,
 * k-truss peeling, which processes a shrinking set of edges each iteration.
 *
 * In multi-GPU the input pairs are assumed to be rank-local (each rank passes the edges it owns),
 * matching the per_v_pair_* convention. The per-edge reduced values returned are still routed to the
 * rank that owns each edge.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for supporting-edge property values.
 * @tparam IntersectionOp Type of the septenary per (edge, intersection vertex) operator.
 * @tparam VertexPairIterator Type of the iterator over (src, dst) vertex pairs.
 * @tparam T Type of the per-edge reduction value.
 * @param handle RAFT handle object.
 * @param graph_view Non-owning graph object.
 * @param vertex_pair_first Iterator to the first (src, dst) input pair.
 * @param vertex_pair_last Iterator to the last (exclusive) (src, dst) input pair.
 * @param edge_src_value_input Wrapper used to access source input property values.
 * @param edge_dst_value_input Wrapper used to access destination input property values.
 * @param edge_value_input Wrapper used to access supporting-edge property values.
 * @param intersection_op septenary per (edge, intersection vertex) operator (see the all-edges
 * overload).
 * @param init Initial value to be added to the reduced @p intersection_op return values for each edge.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of three device vectors (srcs, dsts, values): for each input edge with a non-init
 * reduced value, its source vertex, destination vertex, and reduced value.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename VertexPairIterator,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_src_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPairIterator vertex_pair_first,
  VertexPairIterator vertex_pair_last,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  static_assert(GraphViewType::is_storage_transposed);

  return detail::transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
    handle,
    graph_view,
    vertex_pair_first,
    vertex_pair_last,
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    intersection_op,
    init,
    do_expensive_check);
}

/**
 * @brief Same as transform_reduce_triplet_of_dst_nbr_intersection_of_e_endpoints_by_e, but restricted
 * to a caller-supplied list of edges (vertex pairs) instead of every edge in @p graph_view.
 *
 * For each input (src, dst) pair, intersect the destination neighbor lists of src & dst, invoke @p
 * intersection_op once per vertex r in the intersection, and reduce the functor output values
 * per-edge (identical semantics to the all-edges overload). This is the subset form used by, e.g.,
 * k-truss peeling, which processes a shrinking set of edges each iteration.
 *
 * In multi-GPU the input pairs are assumed to be rank-local (each rank passes the edges it owns),
 * matching the per_v_pair_* convention. The per-edge reduced values returned are still routed to the
 * rank that owns each edge.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for supporting-edge property values.
 * @tparam IntersectionOp Type of the septenary per (edge, intersection vertex) operator.
 * @tparam VertexPairIterator Type of the iterator over (src, dst) vertex pairs.
 * @tparam T Type of the per-edge reduction value.
 * @param handle RAFT handle object.
 * @param graph_view Non-owning graph object.
 * @param vertex_pair_first Iterator to the first (src, dst) input pair.
 * @param vertex_pair_last Iterator to the last (exclusive) (src, dst) input pair.
 * @param edge_src_value_input Wrapper used to access source input property values.
 * @param edge_dst_value_input Wrapper used to access destination input property values.
 * @param edge_value_input Wrapper used to access supporting-edge property values.
 * @param intersection_op septenary per (edge, intersection vertex) operator (see the all-edges
 * overload).
 * @param init Initial value to be added to the reduced @p intersection_op return values for each edge.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of three device vectors (srcs, dsts, values): for each input edge with a non-init
 * reduced value, its source vertex, destination vertex, and reduced value.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename IntersectionOp,
          typename VertexPairIterator,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_dst_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPairIterator vertex_pair_first,
  VertexPairIterator vertex_pair_last,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed);

  return detail::transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
    handle,
    graph_view,
    vertex_pair_first,
    vertex_pair_last,
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    intersection_op,
    init,
    do_expensive_check);
}

}  // namespace CUGRAPH_EXPORT cugraph
