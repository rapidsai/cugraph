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
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>
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
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
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

// Shared per-partition core for the by_e primitive, used by both the all-edges overload and the
// caller-supplied-edge-list overload. Given this partition's (majors, minors) pairs, it broadcasts
// them across minor_comm (multi-GPU), fetches whatever neighbor lists live on other ranks, runs the
// intersection, lets the functor above drop each contribution into one of the three buffers, then
// turns all three back into (src, dst, value) triplets and appends them to agg_*. Nothing is
// resolved to a final per-edge value here; that happens once for the whole rank in
// finalize_triplet_reduction.
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
  rmm::device_uvector<typename GraphViewType::vertex_type> majors,
  rmm::device_uvector<typename GraphViewType::vertex_type> minors,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  IntersectionOp intersection_op,
  T init,
  rmm::device_uvector<typename GraphViewType::vertex_type>& agg_srcs,
  rmm::device_uvector<typename GraphViewType::vertex_type>& agg_dsts,
  dataframe_buffer_type_t<T>& agg_values)
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
    // Note: the broadcast pair list is linear in the number of pairs and does not dominate the
    // footprint. What it feeds does: the fetched N(v1) lists are sized by the sum of degrees over
    // the distinct second endpoints in that list, and dense_remote_acc gets one slot per edge of
    // those lists, so one high-degree v1 drags in its whole adjacency. Chunking the work after the
    // broadcast under a degree-sum budget, and aggregating across chunks, would bound that peak.
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      if (minor_comm.get_size() > 1) {
        // Each rank sends a different number of pairs, so the counts are exchanged first to size the
        // receive buffer and place each rank's block in it.
        auto rx_counts = host_scalar_allgather(minor_comm, majors.size(), handle.get_stream());
        std::vector<size_t> rx_displacements(rx_counts.size());
        std::exclusive_scan(
          rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
        auto aggregate_size = rx_displacements.back() + rx_counts.back();

        rmm::device_uvector<vertex_t> rx_majors(aggregate_size, handle.get_stream());
        rmm::device_uvector<vertex_t> rx_minors(aggregate_size, handle.get_stream());
        // A rank cannot finish the intersection for the pairs it owns, because the rest of each
        // source's neighbors sit on its row peers. Sending every rank's pairs to every rank in the
        // row lets each one contribute what its own slice can see.
        cugraph::device_allgatherv(
          minor_comm,
          majors.begin(),
          rx_majors.begin(),
          raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
          raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
          handle.get_stream());
        cugraph::device_allgatherv(
          minor_comm,
          minors.begin(),
          rx_minors.begin(),
          raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
          raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
          handle.get_stream());
        majors = std::move(rx_majors);
        minors = std::move(rx_minors);

        auto broadcast_pair_first = thrust::make_zip_iterator(majors.begin(), minors.begin());
        thrust::sort(
          handle.get_thrust_policy(), broadcast_pair_first, broadcast_pair_first + majors.size());
      }
    }

    // The base edges to intersect. In multi-GPU this is the concatenation of every rank's pairs in
    // the row, so its size is known before any intersection runs, unlike the number of triangles
    // those pairs will produce.
    auto vertex_pair_first = thrust::make_zip_iterator(majors.begin(), minors.begin());

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
      GraphViewType::is_multi_gpu ? majors.size() : size_t{0}, handle.get_stream());
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
          handle, graph_view, edge_value_input, vertex_pair_first, vertex_pair_first + majors.size());
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
      vertex_pair_first + majors.size(),
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
        per_pair_first + majors.size(),
        cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; })));
      base_srcs.resize(num_base, handle.get_stream());
      base_dsts.resize(num_base, handle.get_stream());
      resize_dataframe_buffer(base_vals, num_base, handle.get_stream());
      auto base_in  = thrust::make_zip_iterator(majors.begin(), minors.begin(), per_pair_first);
      auto base_out = thrust::make_zip_iterator(
        base_srcs.begin(), base_dsts.begin(), get_dataframe_buffer_begin(base_vals));
      thrust::copy_if(
        handle.get_thrust_policy(),
        base_in,
        base_in + majors.size(),
        per_pair_first,
        base_out,
        cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; }));
    }

    // All three sets are now plain (src, dst, value) triplets and nothing distinguishes them
    // anymore, so append them to the aggregate and let the caller's next partition do the same.
    // Appending rather than overwriting is what makes several local edge partitions per rank work.
    //
    // FIXME: agg_* accumulates all local edge partitions' contributions (mostly remote_*, ~O(this rank's
    // total intersection)) before the single post-loop shuffle + reduce, so its peak scales with the
    // rank's whole intersection, not one partition's. If that peak becomes a concern, shuffle +
    // reduce_by_key per partition instead, bounding the footprint to ~one partition's contributions
    // at the cost of minor_comm_size separate collectives (and reserving agg_* up front avoids the
    // per-partition resize growth).
    auto old_size = agg_srcs.size();
    agg_srcs.resize(old_size + num_updated + num_remote + num_base, handle.get_stream());
    agg_dsts.resize(old_size + num_updated + num_remote + num_base, handle.get_stream());
    resize_dataframe_buffer(
      agg_values, old_size + num_updated + num_remote + num_base, handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), local_srcs.begin(), local_srcs.end(), agg_srcs.begin() + old_size);
    thrust::copy(
      handle.get_thrust_policy(), local_dsts.begin(), local_dsts.end(), agg_dsts.begin() + old_size);
    thrust::copy(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(local_vals),
                 get_dataframe_buffer_end(local_vals),
                 get_dataframe_buffer_begin(agg_values) + old_size);
    if constexpr (GraphViewType::is_multi_gpu) {
      thrust::copy(handle.get_thrust_policy(),
                   remote_srcs.begin(),
                   remote_srcs.end(),
                   agg_srcs.begin() + old_size + num_updated);
      thrust::copy(handle.get_thrust_policy(),
                   remote_dsts.begin(),
                   remote_dsts.end(),
                   agg_dsts.begin() + old_size + num_updated);
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(remote_vals),
                   get_dataframe_buffer_end(remote_vals),
                   get_dataframe_buffer_begin(agg_values) + old_size + num_updated);
      thrust::copy(handle.get_thrust_policy(),
                   base_srcs.begin(),
                   base_srcs.end(),
                   agg_srcs.begin() + old_size + num_updated + num_remote);
      thrust::copy(handle.get_thrust_policy(),
                   base_dsts.begin(),
                   base_dsts.end(),
                   agg_dsts.begin() + old_size + num_updated + num_remote);
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(base_vals),
                   get_dataframe_buffer_end(base_vals),
                   get_dataframe_buffer_begin(agg_values) + old_size + num_updated + num_remote);
    }
}

// The settling step every partial has been waiting for. agg_* now holds every (src, dst, value)
// contribution this rank produced, across all of its edge partitions and all three buffers, with the
// same edge appearing as many times as it was touched. This collapses them to one value per edge.
// Single-GPU is already done, since one rank saw every triangle. Multi-GPU has to send each
// contribution to the rank that owns the edge first.
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

    // Reduce locally first. The same edge can appear many times in agg_*, from different legs and
    // different edge partitions, and there is no point paying to send each of those separately: one
    // row per edge goes over the wire instead. The reduce after the shuffle still does the real
    // merge across ranks.
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

// The driver behind all four public entry points, in the all-edges form: every edge of graph_view is
// a pair whose endpoints get intersected.
//
// A rank's edges do not sit in one CSR but in one or more local edge partitions, so the work is done
// a partition at a time:
//   1. rebuild that partition's edges as an explicit (major, minor) list, because the intersection
//      consumes pairs, not a CSR;
//   2. hand the list to accumulate_triplets_for_partition, which finds the triangles and appends its
//      (src, dst, value) contributions to agg_*;
//   3. after every partition has gone through, settle agg_* into one value per edge.
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

  rmm::device_uvector<vertex_t> result_srcs(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(size_t{0}, handle.get_stream());
  // reduction values are held in allocate_dataframe_buffer<T> so a tuple value type T is
  // supported in the future.
  auto result_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

  // Where step 2 deposits its triplets and step 3 picks them up. Partitions append here rather than
  // overwrite, which is what lets a rank own several of them.
  rmm::device_uvector<vertex_t> agg_srcs(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> agg_dsts(size_t{0}, handle.get_stream());
  auto agg_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

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
    // expects.
    accumulate_triplets_for_partition(handle,
                                      graph_view,
                                      i,
                                      std::move(majors),
                                      std::move(minors),
                                      edge_src_value_input,
                                      edge_dst_value_input,
                                      edge_value_input,
                                      intersection_op,
                                      init,
                                      agg_srcs,
                                      agg_dsts,
                                      agg_values);
  }

  // Step 3.
  return finalize_triplet_reduction<GraphViewType, T>(
    handle, graph_view, std::move(agg_srcs), std::move(agg_dsts), std::move(agg_values));
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

  // Same aggregate as the all-edges form, filled by step 2 and drained by step 3.
  rmm::device_uvector<vertex_t> agg_srcs(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> agg_dsts(size_t{0}, handle.get_stream());
  auto agg_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

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

    // Step 2. Hand this partition's pairs to the shared core, which will broadcast them (row comm),
    // fetch any needed remote neighbor lists, run the intersections, and append its (src, dst, value)
    // contributions to the agg_* buffers. This is exactly what the all-edges form does for its
    // decompressed pairs.
    accumulate_triplets_for_partition(handle,
                                      graph_view,
                                      i,
                                      std::move(majors),
                                      std::move(minors),
                                      edge_src_value_input,
                                      edge_dst_value_input,
                                      edge_value_input,
                                      intersection_op,
                                      init,
                                      agg_srcs,
                                      agg_dsts,
                                      agg_values);
  }

  // Step 3, likewise.
  return finalize_triplet_reduction<GraphViewType, T>(
    handle, graph_view, std::move(agg_srcs), std::move(agg_dsts), std::move(agg_values));
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
