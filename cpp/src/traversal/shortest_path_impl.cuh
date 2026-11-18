/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v.cuh>
#include <cugraph/prims/count_if_e.cuh>
#include <cugraph/prims/fill_edge_src_dst_property.cuh>
#include <cugraph/prims/fill_v.cuh>
#include <cugraph/prims/gather_v.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/transform_reduce_e.cuh>
#include <cugraph/prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh>
#include <cugraph/prims/transform_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/prims/update_v_frontier.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/vertex_partition_device_view.cuh>
#include <cugraph/vertex_property_device_view.cuh>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <limits>
#include <optional>

namespace cugraph {

namespace {

template <typename vertex_t, typename weight_t>
struct e_op_t {
  __device__ cuda::std::tuple<weight_t, vertex_t> operator()(
    vertex_t src, vertex_t dst, weight_t src_val, cuda::std::nullopt_t, weight_t w) const
  {
    auto new_distance = src_val + w;
    return cuda::std::make_tuple(new_distance, src);
  }
};

template <typename vertex_t, typename weight_t, bool multi_gpu>
struct pred_op_t {
  vertex_partition_device_view_t<vertex_t, multi_gpu> vertex_partition{};
  detail::vertex_property_device_view_t<vertex_t, weight_t const*> distances{};
  weight_t cutoff{};

  __device__ bool operator()(
    vertex_t src, vertex_t dst, weight_t src_val, cuda::std::nullopt_t, weight_t w) const
  {
    auto push         = true;
    auto new_distance = src_val + w;
    auto threshold    = cutoff;
    if (vertex_partition.in_local_vertex_partition_range_nocheck(dst)) {
      auto v_offset     = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(dst);
      auto old_distance = distances.get(v_offset);
      threshold         = old_distance < threshold ? old_distance : threshold;
    }
    if (new_distance >= threshold) { push = false; }
    return push;
  }
};

template <typename weight_t>
__device__ cuda::std::optional<size_t> sssp_next_bucket(bool update,
                                                        weight_t new_dist,
                                                        weight_t near_far_threshold,
                                                        weight_t near_near_far_threshold,
                                                        size_t next_near_near_bucket,
                                                        size_t near_far_bucket,
                                                        size_t far_bucket)
{
  if (!update) { return cuda::std::nullopt; }
  if (new_dist >= near_far_threshold) { return far_bucket; }
  return new_dist < near_near_far_threshold ? next_near_near_bucket : near_far_bucket;
}

template <typename weight_t>
weight_t compute_subpartition_start(weight_t old_near_far_threshold,
                                    size_t subpartition_idx,
                                    size_t num_subpartitions,
                                    weight_t delta)
{
  return old_near_far_threshold + delta * (static_cast<weight_t>(subpartition_idx) /
                                           static_cast<weight_t>(num_subpartitions));
}

template <typename vertex_t, typename weight_t, bool multi_gpu>
std::tuple<size_t, size_t> compute_new_near_near_partition_range(
  raft::handle_t const& handle,
  key_bucket_t<vertex_t, void, multi_gpu, true> const& key_bucket,
  vertex_partition_device_view_t<vertex_t, multi_gpu> const& vertex_partition,
  vertex_property_view_t<vertex_t, weight_t const*, weight_t> const& distances,
  size_t first_subpartition_idx,
  size_t last_subpartition_idx,
  size_t num_subpartitions,
  vertex_t max_near_near_q_size,
  weight_t old_near_far_threshold,
  weight_t delta)
{
  if ((last_subpartition_idx - first_subpartition_idx) == 1) {
    return std::make_tuple(first_subpartition_idx, last_subpartition_idx);
  }

  rmm::device_uvector<vertex_t> d_counts(last_subpartition_idx - first_subpartition_idx,
                                         handle.get_stream());
  cugraph::fill(handle.get_thrust_policy(), d_counts.begin(), d_counts.end(), vertex_t{0});
  std::vector<weight_t> h_thresholds(d_counts.size() - 1);
  for (size_t i = 0; i < h_thresholds.size(); ++i) {
    h_thresholds[i] = compute_subpartition_start(
      old_near_far_threshold, first_subpartition_idx + i + 1, num_subpartitions, delta);
  }
  rmm::device_uvector<weight_t> d_thresholds(h_thresholds.size(), handle.get_stream());
  raft::update_device(
    d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    key_bucket.begin(),
    key_bucket.end(),
    [vertex_partition,
     distances  = detail::vertex_property_device_view_t<vertex_t, weight_t const*>{distances},
     thresholds = raft::device_span<weight_t const>(d_thresholds.data(), d_thresholds.size()),
     counts = raft::device_span<vertex_t>(d_counts.data(), d_counts.size())] __device__(auto v) {
      auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
      auto dist     = distances.get(v_offset);
      auto idx      = cuda::std::distance(
        thresholds.begin(),
        thrust::lower_bound(thrust::seq, thresholds.begin(), thresholds.end(), dist));
      cuda::atomic_ref<vertex_t, cuda::thread_scope_device> count(counts[idx]);
      count.fetch_add(vertex_t{1}, cuda::std::memory_order_relaxed);
    });
  if constexpr (multi_gpu) {
    device_allreduce(handle.get_comms(),
                     d_counts.begin(),
                     d_counts.begin(),
                     d_counts.size(),
                     raft::comms::op_t::SUM,
                     handle.get_stream());
  }
  std::vector<vertex_t> h_counts(d_counts.size());
  raft::update_host(h_counts.data(), d_counts.data(), d_counts.size(), handle.get_stream());
  handle.sync_stream();

  size_t zero_runs{0};
  for (size_t i = 0; i < h_counts.size(); ++i) {
    if (h_counts[i] == 0) {
      ++zero_runs;
    } else {
      break;
    }
  }

  size_t new_first_near_near_subpartition_idx = first_subpartition_idx + zero_runs;
  size_t new_last_near_near_subpartition_idx  = first_subpartition_idx + (zero_runs + 1);
  vertex_t q_size                             = h_counts[zero_runs];
  for (size_t i = (zero_runs + 1); i < h_counts.size(); ++i) {
    if (q_size + h_counts[i] <= max_near_near_q_size) {
      q_size += h_counts[i];
      ++new_last_near_near_subpartition_idx;
    } else {
      break;
    }
  }

  return std::make_tuple(new_first_near_near_subpartition_idx, new_last_near_near_subpartition_idx);
}

}  // namespace

namespace detail {

template <typename GraphViewType, typename weight_t>
std::tuple<vertex_property_t<typename GraphViewType::vertex_type, weight_t>,
           std::optional<vertex_property_t<typename GraphViewType::vertex_type,
                                           typename GraphViewType::vertex_type>>>
sssp(raft::handle_t const& handle,
     GraphViewType const& graph_view,
     edge_property_view_t<typename GraphViewType::edge_type, weight_t const*> edge_weight_view,
     raft::device_span<typename GraphViewType::vertex_type const> frontier,
     weight_t cutoff,
     bool update_predecessors,
     bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = graph_view.number_of_vertices();
  auto const num_edges    = graph_view.compute_number_of_edges(handle);
  if (num_vertices == 0) {
    return std::make_tuple(
      vertex_property_t<vertex_t, weight_t>(handle, graph_view),
      update_predecessors
        ? std::make_optional(vertex_property_t<vertex_t, vertex_t>(handle, graph_view))
        : std::nullopt);
  }

  // Implements the Near-Far Pile method in
  // A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient parallel GPU methods for
  // single-source shortest paths," 2014.
  // More recent updates extended the initial near-far method to a two-level approach which further
  // partitions the near queue to near-near & near-far queues if the near queue size is too large
  // (we wish to maintain the near queue size just large enough to saturate GPU resources).

  // 1. check input arguments

  if constexpr (GraphViewType::is_multi_gpu) {
    if (do_expensive_check) {
      auto aggregate_n_sources = frontier.size();
      aggregate_n_sources      = host_scalar_allreduce(
        handle.get_comms(), aggregate_n_sources, raft::comms::op_t::SUM, handle.get_stream());
      CUGRAPH_EXPECTS(aggregate_n_sources > 0,
                      "Invalid input argument: input should have at least one frontier vertex.");
    }
  } else {
    CUGRAPH_EXPECTS(frontier.size() > 0,
                    "Invalid input argument: input should have at least one frontier vertex.");
  }

  auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
    graph_view.local_vertex_partition_view());

  if (do_expensive_check) {
    bool is_sorted = thrust::is_sorted(
      handle.get_thrust_policy(), frontier.data(), frontier.data() + frontier.size());
    if constexpr (GraphViewType::is_multi_gpu) {
      auto tmp = static_cast<int32_t>(is_sorted);
      tmp =
        host_scalar_allreduce(handle.get_comms(), tmp, raft::comms::op_t::MIN, handle.get_stream());
      is_sorted = static_cast<bool>(tmp);
    }
    CUGRAPH_EXPECTS(
      is_sorted,
      "Invalid input arguments: input frontier should be sorted in the non-descending order.");

    bool no_duplicates =
      static_cast<size_t>(thrust::unique_count(
        handle.get_thrust_policy(), frontier.data(), frontier.data() + frontier.size())) ==
      frontier.size();
    if constexpr (GraphViewType::is_multi_gpu) {
      auto tmp = static_cast<int32_t>(no_duplicates);
      tmp =
        host_scalar_allreduce(handle.get_comms(), tmp, raft::comms::op_t::MIN, handle.get_stream());
      no_duplicates = static_cast<bool>(tmp);
    }
    CUGRAPH_EXPECTS(no_duplicates,
                    "Invalid input arguments: input frontier should not have duplicates.");

    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       frontier.data(),
                       frontier.data() + frontier.size(),
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: frontier has invalid vertex IDs.");

    auto num_negative_edge_weights =
      count_if_e(handle,
                 graph_view,
                 edge_src_dummy_property_t{}.view(),
                 edge_dst_dummy_property_t{}.view(),
                 edge_weight_view,
                 [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w < 0.0; });
    CUGRAPH_EXPECTS(num_negative_edge_weights == 0,
                    "Invalid input argument: input edge weights should have non-negative values.");
  }

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<weight_t>::max();
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  vertex_property_t<vertex_t, weight_t> distances(handle, graph_view);
  auto predecessors =
    update_predecessors
      ? std::make_optional(vertex_property_t<vertex_t, vertex_t>(handle, graph_view))
      : std::optional<vertex_property_t<vertex_t, vertex_t>>{std::nullopt};

  transform_v(
    handle,
    graph_view,
    [sources = raft::device_span<vertex_t const>(frontier.data(), frontier.size()),
     invalid_distance] __device__(vertex_t v) {
      return thrust::binary_search(thrust::seq, sources.begin(), sources.end(), v)
               ? weight_t{0.0}
               : invalid_distance;
    },
    distances.mutable_view());
  if (predecessors) { fill_v(handle, graph_view, predecessors->mutable_view(), invalid_vertex); }

  if (num_edges == 0) { return std::make_tuple(std::move(distances), std::move(predecessors)); }

  // 3. update delta

  weight_t average_vertex_degree =
    static_cast<weight_t>(num_edges) / static_cast<weight_t>(num_vertices);
  weight_t average_edge_weight{0.0};
  average_edge_weight =
    transform_reduce_e(handle,
                       graph_view,
                       edge_src_dummy_property_t{}.view(),
                       edge_dst_dummy_property_t{}.view(),
                       edge_weight_view,
                       cuda::proclaim_return_type<weight_t>(
                         [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w; }),
                       weight_t{0.0});
  average_edge_weight /= static_cast<weight_t>(num_edges);
  auto delta =
    (static_cast<weight_t>(raft::warp_size()) * average_edge_weight) / average_vertex_degree;
  size_t num_subpartitions    = 16;  // tuning parameter
  size_t max_near_near_q_size = static_cast<size_t>(num_vertices);
  if (average_vertex_degree > weight_t{0.0}) {
    size_t aggregate_sm_counts =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount);
    if constexpr (GraphViewType::is_multi_gpu) {
      aggregate_sm_counts = host_scalar_allreduce(
        handle.get_comms(), aggregate_sm_counts, raft::comms::op_t::SUM, handle.get_stream());
    }
    max_near_near_q_size =
      std::min(static_cast<size_t>(
                 static_cast<weight_t>(aggregate_sm_counts * size_t{2048} /* tuning parameter */) /
                 average_vertex_degree),
               max_near_near_q_size);  // near queue size should ideally be just large enough to
                                       // saturate GPU resources (otherwise, it will end up doing
                                       // more unnecessary computations compared to the Dijkstra's
                                       // method without further increasing GPU utilization).
  }

  // 4. initialize SSSP frontier

  constexpr size_t bucket_idx_cur_near_near  = 0;
  constexpr size_t bucket_idx_next_near_near = 1;
  constexpr size_t bucket_idx_near_far       = 2;
  constexpr size_t bucket_idx_far            = 3;
  constexpr size_t num_buckets               = 4;

  vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle,
                                                                                       num_buckets);

  if (frontier.size() > 0) {
    vertex_frontier.bucket(bucket_idx_cur_near_near)
      .insert(frontier.data(), frontier.data() + frontier.size());
  }

  // 5. SSSP iteration

  auto edge_src_distances = GraphViewType::is_multi_gpu
                              ? edge_src_property_t<vertex_t, weight_t>(handle, graph_view)
                              : edge_src_property_t<vertex_t, weight_t>(handle);
  if constexpr (GraphViewType::is_multi_gpu) {
    fill_edge_src_property(
      handle, graph_view, edge_src_distances.mutable_view(), std::numeric_limits<weight_t>::max());
  }

  auto cur_near_far_threshold = delta;
  auto old_near_far_threshold = weight_t{0.0};
  std::optional<size_t> last_near_near_subpartition_idx{
    std::nullopt};  // valid only when in the lower level
  std::optional<size_t> first_near_near_subpartition_idx{
    std::nullopt};  // valid only when in the lower level
  while (true) {
    if constexpr (GraphViewType::is_multi_gpu) {
      rmm::device_uvector<weight_t> gathered_distances(
        vertex_frontier.bucket(bucket_idx_cur_near_near).size(), handle.get_stream());
      gather_v(handle,
               graph_view,
               vertex_frontier.bucket(bucket_idx_cur_near_near).begin(),
               vertex_frontier.bucket(bucket_idx_cur_near_near).end(),
               distances.view(),
               gathered_distances.begin());
      update_edge_src_property(handle,
                               graph_view,
                               vertex_frontier.bucket(bucket_idx_cur_near_near).begin(),
                               vertex_frontier.bucket(bucket_idx_cur_near_near).end(),
                               gathered_distances.begin(),
                               edge_src_distances.mutable_view());
    }

    auto [new_frontier_vertex_buffer, distance_predecessor_buffer] =
      cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
        handle,
        graph_view,
        vertex_frontier.bucket(bucket_idx_cur_near_near),
        GraphViewType::is_multi_gpu ? edge_src_distances.view()
                                    : make_edge_src_property_view(graph_view, distances.view()),
        edge_dst_dummy_property_t{}.view(),
        edge_weight_view,
        e_op_t<vertex_t, weight_t>{},
        reduce_op::minimum<cuda::std::tuple<weight_t, vertex_t>>(),
        pred_op_t<vertex_t, weight_t, GraphViewType::is_multi_gpu>{
          vertex_partition,
          detail::vertex_property_device_view_t<vertex_t, weight_t const*>{distances.view()},
          cutoff});

    auto next_frontier_bucket_indices =
      std::vector<size_t>{bucket_idx_next_near_near, bucket_idx_near_far, bucket_idx_far};
    auto cur_near_near_far_threshold =
      last_near_near_subpartition_idx
        ? compute_subpartition_start(
            old_near_far_threshold, *last_near_near_subpartition_idx, num_subpartitions, delta)
        : cur_near_far_threshold;
    if (predecessors) {
      update_v_frontier(
        handle,
        graph_view,
        std::move(new_frontier_vertex_buffer),
        std::move(distance_predecessor_buffer),
        vertex_frontier,
        raft::host_span<size_t const>(next_frontier_bucket_indices.data(),
                                      next_frontier_bucket_indices.size()),
        distances.view(),
        view_concat(distances.mutable_view(), predecessors->mutable_view()),
        [cur_near_far_threshold, cur_near_near_far_threshold] __device__(
          auto, auto v_dist, auto pushed_val) {
          auto new_dist = cuda::std::get<0>(pushed_val);
          auto update   = (new_dist < v_dist);
          return cuda::std::make_tuple(
            sssp_next_bucket(update,
                             new_dist,
                             cur_near_far_threshold,
                             cur_near_near_far_threshold,
                             bucket_idx_next_near_near,
                             bucket_idx_near_far,
                             bucket_idx_far),
            update ? cuda::std::optional<cuda::std::tuple<weight_t, vertex_t>>{pushed_val}
                   : cuda::std::nullopt);
        });
    } else {
      update_v_frontier(handle,
                        graph_view,
                        std::move(new_frontier_vertex_buffer),
                        std::move(distance_predecessor_buffer),
                        vertex_frontier,
                        raft::host_span<size_t const>(next_frontier_bucket_indices.data(),
                                                      next_frontier_bucket_indices.size()),
                        distances.view(),
                        distances.mutable_view(),
                        [cur_near_far_threshold, cur_near_near_far_threshold] __device__(
                          auto, auto v_dist, auto pushed_val) {
                          auto new_dist = cuda::std::get<0>(pushed_val);
                          auto update   = (new_dist < v_dist);
                          return cuda::std::make_tuple(
                            sssp_next_bucket(update,
                                             new_dist,
                                             cur_near_far_threshold,
                                             cur_near_near_far_threshold,
                                             bucket_idx_next_near_near,
                                             bucket_idx_near_far,
                                             bucket_idx_far),
                            update ? cuda::std::optional<weight_t>{new_dist} : cuda::std::nullopt);
                        });
    }

    vertex_frontier.bucket(bucket_idx_cur_near_near).clear();
    vertex_frontier.bucket(bucket_idx_cur_near_near).shrink_to_fit();
    vertex_frontier.swap_buckets(bucket_idx_cur_near_near, bucket_idx_next_near_near);

    auto near_near_aggregate_size =
      vertex_frontier.bucket(bucket_idx_cur_near_near).aggregate_size();
    auto near_far_aggregate_size = vertex_frontier.bucket(bucket_idx_near_far).aggregate_size();
    auto far_aggregate_size      = vertex_frontier.bucket(bucket_idx_far).aggregate_size();

    bool split_near_near    = false;
    bool try_split_near_far = false;
    bool try_split_far      = false;
    bool empty_queue        = false;
    if (last_near_near_subpartition_idx) {  // in the lower level
      if ((near_near_aggregate_size > max_near_near_q_size) &&
          ((*last_near_near_subpartition_idx - *first_near_near_subpartition_idx) > 1)) {
        split_near_near = true;
      } else if (near_near_aggregate_size > 0) {
        /* nothing to do */
      } else if (near_far_aggregate_size > 0) {
        try_split_near_far = true;
      } else if (far_aggregate_size > 0) {  // move up the level
        try_split_far = true;
      } else {
        empty_queue = true;
      }
    } else {  // in the first level
      assert(near_far_aggregate_size == 0);
      if (near_near_aggregate_size > max_near_near_q_size) {  // move down the level
        split_near_near = true;
      } else if (near_near_aggregate_size > 0) {
        /* nothing to do */
      } else if (far_aggregate_size > 0) {
        try_split_far = true;
      } else {
        empty_queue = true;
      }
    }

    bool split_near_far = false;
    if (try_split_near_far) {  // scan and remove the vertices with the final distances; if the
                               // queue size is still non-zero, split.
      vertex_frontier.bucket(bucket_idx_near_far)
        .resize(cuda::std::distance(
          vertex_frontier.bucket(bucket_idx_near_far).begin(),
          thrust::remove_if(
            handle.get_thrust_policy(),
            vertex_frontier.bucket(bucket_idx_near_far).begin(),
            vertex_frontier.bucket(bucket_idx_near_far).end(),
            cuda::proclaim_return_type<bool>(
              [vertex_partition,
               distances =
                 detail::vertex_property_device_view_t<vertex_t, weight_t const*>{distances.view()},
               cur_near_near_far_threshold =
                 compute_subpartition_start(old_near_far_threshold,
                                            *last_near_near_subpartition_idx,
                                            num_subpartitions,
                                            delta)] __device__(auto v) {
                auto v_offset =
                  vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                return distances.get(v_offset) < cur_near_near_far_threshold;
              }))));
      near_far_aggregate_size = vertex_frontier.bucket(bucket_idx_near_far).aggregate_size();

      if (near_far_aggregate_size > 0) {
        split_near_far = true;
      } else {
        try_split_far = true;
      }
    }

    bool split_far = false;
    if (try_split_far) {  // scan and remove the vertices with the final distances; if the queue
                          // size is still non-zero, split.
      vertex_frontier.bucket(bucket_idx_far)
        .resize(cuda::std::distance(
          vertex_frontier.bucket(bucket_idx_far).begin(),
          thrust::remove_if(
            handle.get_thrust_policy(),
            vertex_frontier.bucket(bucket_idx_far).begin(),
            vertex_frontier.bucket(bucket_idx_far).end(),
            cuda::proclaim_return_type<bool>(
              [vertex_partition,
               distances =
                 detail::vertex_property_device_view_t<vertex_t, weight_t const*>{distances.view()},
               cur_near_far_threshold] __device__(auto v) {
                auto v_offset =
                  vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                return distances.get(v_offset) < cur_near_far_threshold;
              }))));
      far_aggregate_size = vertex_frontier.bucket(bucket_idx_far).aggregate_size();

      if (far_aggregate_size > 0) {
        split_far = true;
      } else {
        empty_queue = true;
      }
    }

    if (split_far) {
      if (last_near_near_subpartition_idx) {  // move up the level
        last_near_near_subpartition_idx  = std::nullopt;
        first_near_near_subpartition_idx = std::nullopt;
      }

      old_near_far_threshold = cur_near_far_threshold;
      cur_near_far_threshold += delta;

      std::vector<size_t> move_to_bucket_indices = {bucket_idx_cur_near_near};
      while (true) {
        vertex_frontier.split_bucket(
          bucket_idx_far,
          raft::host_span<size_t const>(move_to_bucket_indices.data(),
                                        move_to_bucket_indices.size()),
          [vertex_partition,
           distances =
             detail::vertex_property_device_view_t<vertex_t, weight_t const*>{distances.view()},
           cur_near_far_threshold] __device__(auto v) {
            auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            auto dist     = distances.get(v_offset);
            return cuda::std::optional<size_t>{
              dist < cur_near_far_threshold ? bucket_idx_cur_near_near : bucket_idx_far};
          });
        near_near_aggregate_size =
          vertex_frontier.bucket(bucket_idx_cur_near_near).aggregate_size();
        assert(vertex_frontier.bucket(bucket_idx_far).aggregate_size() > 0);
        if (near_near_aggregate_size > 0) {
          if (near_near_aggregate_size >= max_near_near_q_size) { split_near_near = true; }
          break;
        } else {
          old_near_far_threshold = cur_near_far_threshold;
          cur_near_far_threshold += delta;
        }
      }
    }

    if (split_near_far && ((num_subpartitions - *last_near_near_subpartition_idx) == 1)) {
      vertex_frontier.swap_buckets(bucket_idx_cur_near_near, bucket_idx_near_far);
      first_near_near_subpartition_idx = *last_near_near_subpartition_idx;
      last_near_near_subpartition_idx  = num_subpartitions;
    } else if (split_near_near || split_near_far) {
      size_t first_far_subpartition_idx{};
      size_t last_far_subpartition_idx{};
      size_t this_bucket_idx{};
      std::vector<size_t> move_to_bucket_indices{};
      if (split_near_near) {
        this_bucket_idx        = bucket_idx_cur_near_near;
        move_to_bucket_indices = {bucket_idx_near_far};
        if (last_near_near_subpartition_idx) {  // already in the lower level
          first_far_subpartition_idx = *first_near_near_subpartition_idx;
          last_far_subpartition_idx  = *last_near_near_subpartition_idx;
        } else {  // newly entered the lower level
          first_far_subpartition_idx = 0;
          last_far_subpartition_idx  = num_subpartitions;
        }
      } else {
        this_bucket_idx            = bucket_idx_near_far;
        move_to_bucket_indices     = {bucket_idx_cur_near_near};
        first_far_subpartition_idx = *last_near_near_subpartition_idx;
        last_far_subpartition_idx  = num_subpartitions;
      }

      std::tie(first_near_near_subpartition_idx, last_near_near_subpartition_idx) =
        compute_new_near_near_partition_range<vertex_t, weight_t, GraphViewType::is_multi_gpu>(
          handle,
          vertex_frontier.bucket(this_bucket_idx),
          vertex_partition,
          distances.view(),
          first_far_subpartition_idx,
          last_far_subpartition_idx,
          num_subpartitions,
          max_near_near_q_size,
          old_near_far_threshold,
          delta);

      vertex_frontier.split_bucket(
        this_bucket_idx,
        raft::host_span<size_t const>(move_to_bucket_indices.data(), move_to_bucket_indices.size()),
        [vertex_partition,
         distances =
           detail::vertex_property_device_view_t<vertex_t, weight_t const*>{distances.view()},
         cur_near_near_far_threshold = compute_subpartition_start(old_near_far_threshold,
                                                                  *last_near_near_subpartition_idx,
                                                                  num_subpartitions,
                                                                  delta)] __device__(auto v) {
          auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
          auto dist     = distances.get(v_offset);
          return cuda::std::optional<size_t>{
            dist < cur_near_near_far_threshold ? bucket_idx_cur_near_near : bucket_idx_near_far};
        });
    }

    if (empty_queue) { break; }
  }

  return std::make_tuple(std::move(distances), std::move(predecessors));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, weight_t>,
           std::optional<vertex_property_t<vertex_t, vertex_t>>>
shortest_path(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
              edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
              raft::device_span<vertex_t const> frontier,
              weight_t cutoff,
              bool update_predecessors,
              bool do_expensive_check)
{
  return detail::sssp(handle,
                      graph_view,
                      edge_weight_view,
                      frontier,
                      cutoff,
                      update_predecessors,
                      do_expensive_check);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void sssp(raft::handle_t const& handle,
          graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
          edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
          weight_t* distances,
          vertex_t* predecessors,
          vertex_t source_vertex,
          weight_t cutoff,
          bool do_expensive_check)
{
  CUGRAPH_EXPECTS(graph_view.is_valid_vertex(source_vertex),
                  "Invalid input argument: source vertex out-of-range.");

  rmm::device_uvector<vertex_t> frontier_vertices(
    graph_view.in_local_vertex_partition_range_nocheck(source_vertex) ? size_t{1} : size_t{0},
    handle.get_stream());
  if (frontier_vertices.size() > 0) {
    raft::update_device(frontier_vertices.data(), &source_vertex, size_t{1}, handle.get_stream());
  }

  auto [distance_property, predecessor_property] = shortest_path(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()),
    cutoff,
    predecessors != nullptr,
    do_expensive_check);

  copy_v(handle,
         graph_view,
         distance_property.view(),
         vertex_property_view_t<vertex_t, weight_t*, weight_t>{
           distances, graph_view.local_vertex_partition_range_size()});
  if (predecessor_property) {
    copy_v(handle,
           graph_view,
           predecessor_property->view(),
           vertex_property_view_t<vertex_t, vertex_t*, vertex_t>{
             predecessors, graph_view.local_vertex_partition_range_size()});
  }
}

}  // namespace cugraph
