/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/count_if_v.cuh"
#include "prims/detail/prim_functors.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_e.cuh"
#include "prims/extract_transform_if_v_frontier_outgoing_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/transform_reduce_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

//
// The formula for BC(v) is the sum over all (s,t) where s != v != t of
// sigma_st(v) / sigma_st.  Sigma_st(v) is the number of shortest paths
// that pass through vertex v, whereas sigma_st is the total number of shortest
// paths.
namespace {

template <typename vertex_t>
struct brandes_e_op_t {
  template <typename value_t, typename ignore_t>
  __device__ value_t operator()(vertex_t, vertex_t, value_t src_sigma, vertex_t, ignore_t) const
  {
    return src_sigma;
  }
};

template <typename vertex_t>
struct brandes_pred_op_t {
  const vertex_t invalid_distance_{std::numeric_limits<vertex_t>::max()};

  template <typename value_t, typename ignore_t>
  __device__ bool operator()(
    vertex_t, vertex_t, value_t src_sigma, vertex_t dst_distance, ignore_t) const
  {
    return (dst_distance == invalid_distance_);
  }
};

template <typename vertex_t>
struct extract_edge_e_op_t {
  template <typename edge_t, typename weight_t>
  __device__ cuda::std::tuple<vertex_t, vertex_t> operator()(
    vertex_t src,
    vertex_t dst,
    cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
    cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
    cuda::std::nullopt_t) const
  {
    return cuda::std::make_tuple(src, dst);
  }

  template <typename edge_t, typename weight_t>
  __device__ cuda::std::tuple<vertex_t, vertex_t, edge_t> operator()(
    vertex_t src,
    vertex_t dst,
    cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
    cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
    edge_t edge_multi_index) const
  {
    return cuda::std::make_tuple(src, dst, edge_multi_index);
  }
};

template <typename vertex_t>
struct extract_edge_pred_op_t {
  vertex_t d{};

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
                             cuda::std::nullopt_t) const
  {
    return ((cuda::std::get<0>(src_props) == (d - 1)) && (cuda::std::get<0>(dst_props) == d));
  }

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
                             edge_t edge_multi_index) const
  {
    return ((cuda::std::get<0>(src_props) == (d - 1)) && (cuda::std::get<0>(dst_props) == d));
  }
};

}  // namespace

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> brandes_bfs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_frontier_t<vertex_t, void, multi_gpu, true>& vertex_frontier,
  bool do_expensive_check)
{
  //
  // Do BFS with a multi-output.  If we're on hop k and multiple vertices arrive at vertex v,
  // add all predecessors to the predecessor list, don't just arbitrarily pick one.
  //
  // Predecessors could be a CSR if that's helpful for doing the backwards tracing
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
  constexpr size_t bucket_idx_cur{0};
  constexpr size_t bucket_idx_next{1};

  rmm::device_uvector<edge_t> sigmas(graph_view.local_vertex_partition_range_size(),
                                     handle.get_stream());
  rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                          handle.get_stream());
  detail::scalar_fill(handle, distances.data(), distances.size(), invalid_distance);
  detail::scalar_fill(handle, sigmas.data(), sigmas.size(), edge_t{0});

  edge_src_property_t<vertex_t, edge_t> src_sigmas(handle, graph_view);
  edge_dst_property_t<vertex_t, vertex_t> dst_distances(handle, graph_view);

  auto vertex_partition =
    vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());

  if (vertex_frontier.bucket(bucket_idx_cur).size() > 0) {
    thrust::for_each(
      handle.get_thrust_policy(),
      vertex_frontier.bucket(bucket_idx_cur).begin(),
      vertex_frontier.bucket(bucket_idx_cur).end(),
      [d_sigma = sigmas.begin(), d_distance = distances.begin(), vertex_partition] __device__(
        auto v) {
        auto offset        = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
        d_distance[offset] = 0;
        d_sigma[offset]    = 1;
      });
  }

  edge_t hop{0};

  while (true) {
    update_edge_src_property(handle, graph_view, sigmas.begin(), src_sigmas.mutable_view());
    update_edge_dst_property(handle, graph_view, distances.begin(), dst_distances.mutable_view());

    auto [new_frontier, new_sigma] = cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
      handle,
      graph_view,
      vertex_frontier.bucket(bucket_idx_cur),
      src_sigmas.view(),
      dst_distances.view(),
      cugraph::edge_dummy_property_t{}.view(),
      brandes_e_op_t<vertex_t>{},
      reduce_op::plus<vertex_t>(),
      brandes_pred_op_t<vertex_t>{});

    auto next_frontier_bucket_indices = std::vector<size_t>{bucket_idx_next};
    update_v_frontier(handle,
                      graph_view,
                      std::move(new_frontier),
                      std::move(new_sigma),
                      vertex_frontier,
                      raft::host_span<size_t const>(next_frontier_bucket_indices.data(),
                                                    next_frontier_bucket_indices.size()),
                      thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                      thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                      [hop] __device__(auto v, auto old_values, auto v_sigma) {
                        return cuda::std::make_tuple(
                          cuda::std::make_optional(bucket_idx_next),
                          cuda::std::make_optional(cuda::std::make_tuple(hop + 1, v_sigma)));
                      });

    vertex_frontier.bucket(bucket_idx_cur).clear();
    vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
    vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
    if (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() == 0) { break; }

    ++hop;
  }

  return std::make_tuple(std::move(distances), std::move(sigmas));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void accumulate_vertex_results(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<weight_t> centralities,
  rmm::device_uvector<vertex_t>&& distances,
  rmm::device_uvector<edge_t>&& sigmas,
  bool with_endpoints,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  vertex_t diameter = transform_reduce_v(
    handle,
    graph_view,
    distances.begin(),
    [] __device__(auto, auto d) { return (d == invalid_distance) ? vertex_t{0} : d; },
    vertex_t{0},
    reduce_op::maximum<vertex_t>{},
    do_expensive_check);

  if (with_endpoints) {
    vertex_t count = count_if_v(
      handle,
      graph_view,
      distances.begin(),
      [] __device__(auto, auto d) { return (d != invalid_distance); },
      do_expensive_check);

    thrust::transform(handle.get_thrust_policy(),
                      distances.begin(),
                      distances.end(),
                      centralities.begin(),
                      centralities.begin(),
                      [count] __device__(auto d, auto centrality) {
                        if (d == vertex_t{0}) {
                          return centrality + static_cast<weight_t>(count - 1);
                        } else if (d == invalid_distance) {
                          return centrality;
                        } else {
                          return centrality + weight_t{1};
                        }
                      });
  }

  edge_src_property_t<vertex_t, edge_t> src_sigmas(handle, graph_view);
  edge_dst_property_t<vertex_t, vertex_t> dst_distances(handle, graph_view);
  edge_dst_property_t<vertex_t, edge_t> dst_sigmas(handle, graph_view);
  edge_dst_property_t<vertex_t, weight_t> dst_deltas(handle, graph_view);

  // Update all 3 properties initially (deltas start as 0)
  update_edge_src_property(handle, graph_view, sigmas.begin(), src_sigmas.mutable_view());
  update_edge_dst_property(handle,
                           graph_view,
                           thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                           view_concat(dst_distances.mutable_view(), dst_sigmas.mutable_view()));
  fill_edge_dst_property(handle, graph_view, dst_deltas.mutable_view(), weight_t{0.0});

  // Use binary search method to find frontier boundaries more efficiently
  std::vector<vertex_t> h_bounds{};
  rmm::device_uvector<vertex_t> vertices_sorted(distances.size(), handle.get_stream());
  {
    // Create distance_keys for sorting
    rmm::device_uvector<vertex_t> distance_keys(distances.size(), handle.get_stream());

    // Copy distances for sorting (preserve original)
    raft::copy(distance_keys.data(), distances.data(), distances.size(), handle.get_stream());

    // Use thrust::sequence instead of thrust::copy for vertices
    thrust::sequence(handle.get_thrust_policy(),
                     vertices_sorted.begin(),
                     vertices_sorted.end(),
                     graph_view.local_vertex_partition_range_first());

    // Sort vertices by distance using stable_sort_by_key
    thrust::stable_sort_by_key(handle.get_thrust_policy(),
                               distance_keys.begin(),
                               distance_keys.end(),     // keys (copied distances)
                               vertices_sorted.begin()  // values (vertices)
    );

    rmm::device_uvector<vertex_t> d_bounds(diameter + 1, handle.get_stream());

    // Single vectorized thrust call to compute all bounds for distances 0 to diameter
    thrust::lower_bound(
      handle.get_thrust_policy(),
      distance_keys.begin(),
      distance_keys.end(),                          // sorted distances
      thrust::make_counting_iterator<vertex_t>(0),  // search keys: [0, 1, 2, 3, ...]
      thrust::make_counting_iterator<vertex_t>(diameter + 1),
      d_bounds.data());

    // Copy bounds to host for use in delta loop
    h_bounds.resize(d_bounds.size());
    raft::update_host(h_bounds.data(), d_bounds.data(), d_bounds.size(), handle.get_stream());
    handle.sync_stream();
  }

  // Calculate max frontier size using the precomputed bounds
  vertex_t max_frontier_size = 0;
  for (size_t d = 0; d < h_bounds.size() - 1; ++d) {
    vertex_t frontier_count = h_bounds[d + 1] - h_bounds[d];
    max_frontier_size       = std::max(max_frontier_size, frontier_count);
  }

  // Pre-allocate reusable buffers to avoid repeated allocations (optimized for max frontier size)
  rmm::device_uvector<weight_t> reusable_delta_buffer(max_frontier_size, handle.get_stream());

  // Based on Brandes algorithm, we want to follow back pointers in non-increasing
  // distance from S to compute delta
  for (vertex_t d = diameter; d > 1; --d) {
    vertex_t frontier_count = h_bounds[d] - h_bounds[d - 1];
    if constexpr (multi_gpu) {
      frontier_count = host_scalar_allreduce(
        handle.get_comms(), frontier_count, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (frontier_count > 0) {
      // Create key_bucket_t from the frontier vertices directly
      key_bucket_t<vertex_t, void, multi_gpu, true> vertex_list(
        handle,
        raft::device_span<vertex_t const>(vertices_sorted.data() + h_bounds[d - 1],
                                          h_bounds[d] - h_bounds[d - 1]));

      // Compute deltas for frontier vertices
      per_v_transform_reduce_outgoing_e(
        handle,
        graph_view,
        vertex_list,
        src_sigmas.view(),
        view_concat(dst_distances.view(), dst_sigmas.view(), dst_deltas.view()),
        cugraph::edge_dummy_property_t{}.view(),
        [d] __device__(auto, auto, auto src_sigma, auto dst_props, auto) {
          if (cuda::std::get<0>(dst_props) == d) {
            auto sigma_v = src_sigma;
            auto sigma_w = static_cast<weight_t>(cuda::std::get<1>(dst_props));
            auto delta_w = cuda::std::get<2>(dst_props);
            return (sigma_v / sigma_w) * (1 + delta_w);
          } else {
            return weight_t{0};
          }
        },
        weight_t{0},
        reduce_op::plus<weight_t>{},
        reusable_delta_buffer.begin(),
        do_expensive_check);

      // Only update deltas for vertices in vertex_list
      update_edge_dst_property(handle,
                               graph_view,
                               vertex_list.cbegin(),
                               vertex_list.cend(),
                               reusable_delta_buffer.begin(),
                               dst_deltas.mutable_view());

      // Update centralities - both vertices_sorted and centralities use local vertex IDs
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(vertices_sorted.begin() + h_bounds[d - 1],
                                  reusable_delta_buffer.begin()),
        thrust::make_zip_iterator(vertices_sorted.begin() + h_bounds[d],
                                  reusable_delta_buffer.begin()),
        [centralities = centralities.data(),
         v_first      = graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
          auto v        = cuda::std::get<0>(pair);
          auto delta    = cuda::std::get<1>(pair);
          auto v_offset = v - v_first;
          centralities[v_offset] += delta;
        });
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void accumulate_edge_results(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  edge_property_view_t<edge_t, weight_t*> centralities_view,
  rmm::device_uvector<vertex_t>&& distances,
  rmm::device_uvector<edge_t>&& sigmas,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  vertex_t diameter = transform_reduce_v(
    handle,
    graph_view,
    distances.begin(),
    [] __device__(auto, auto d) { return (d == invalid_distance) ? vertex_t{0} : d; },
    vertex_t{0},
    reduce_op::maximum<vertex_t>{},
    do_expensive_check);

  rmm::device_uvector<weight_t> deltas(sigmas.size(), handle.get_stream());
  detail::scalar_fill(handle, deltas.data(), deltas.size(), weight_t{0});

  edge_src_property_t<vertex_t, cuda::std::tuple<vertex_t, edge_t, weight_t>> src_properties(
    handle, graph_view);
  edge_dst_property_t<vertex_t, cuda::std::tuple<vertex_t, edge_t, weight_t>> dst_properties(
    handle, graph_view);

  // Note: deltas are included here even though they start as 0, because the original approach
  // updates all properties at once. Deltas will be overwritten iteratively in the delta loop.
  update_edge_src_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    src_properties.mutable_view());
  update_edge_dst_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    dst_properties.mutable_view());

  //
  //   For now this will do a O(E) pass over all edges over the diameter
  //   of the graph.
  //
  // Based on Brandes algorithm, we want to follow back pointers in non-increasing
  // distance from S to compute delta
  //
  for (vertex_t d = diameter; d > 0; --d) {
    //
    //  Populate edge_list with edges where `cuda::std::get<0>(dst_props) == d`
    //  and `cuda::std::get<0>(dst_props) == (d-1)`
    //
    cugraph::edge_bucket_t<vertex_t, edge_t, true, multi_gpu, true> edge_list(
      handle, graph_view.is_multigraph());

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> indices{std::nullopt};
    if (graph_view.is_multigraph()) {
      edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(handle, graph_view);
      std::tie(srcs, dsts, indices) = extract_transform_if_e(handle,
                                                             graph_view,
                                                             src_properties.view(),
                                                             dst_properties.view(),
                                                             edge_multi_indices.view(),
                                                             extract_edge_e_op_t<vertex_t>{},
                                                             extract_edge_pred_op_t<vertex_t>{d},
                                                             do_expensive_check);

      auto triplet_first = thrust::make_zip_iterator(srcs.begin(), dsts.begin(), indices->begin());
      thrust::sort(handle.get_thrust_policy(), triplet_first, triplet_first + srcs.size());
    } else {
      std::tie(srcs, dsts) = extract_transform_if_e(handle,
                                                    graph_view,
                                                    src_properties.view(),
                                                    dst_properties.view(),
                                                    edge_dummy_property_t{}.view(),
                                                    extract_edge_e_op_t<vertex_t>{},
                                                    extract_edge_pred_op_t<vertex_t>{d},
                                                    do_expensive_check);
      auto pair_first      = thrust::make_zip_iterator(srcs.begin(), dsts.begin());
      thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + srcs.size());
    }
    edge_list.insert(srcs.begin(),
                     srcs.end(),
                     dsts.begin(),
                     indices ? std::make_optional(indices->begin()) : std::nullopt);

    transform_e(
      handle,
      graph_view,
      edge_list,
      src_properties.view(),
      dst_properties.view(),
      centralities_view,
      [d] __device__(auto src, auto dst, auto src_props, auto dst_props, auto edge_centrality) {
        if ((cuda::std::get<0>(dst_props) == d) && (cuda::std::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(cuda::std::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(cuda::std::get<1>(dst_props));
          auto delta_w = cuda::std::get<2>(dst_props);

          return edge_centrality + (sigma_v / sigma_w) * (1 + delta_w);
        } else {
          return edge_centrality;
        }
      },
      centralities_view,
      do_expensive_check);

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      src_properties.view(),
      dst_properties.view(),
      cugraph::edge_dummy_property_t{}.view(),
      [d] __device__(auto, auto, auto src_props, auto dst_props, auto) {
        if ((cuda::std::get<0>(dst_props) == d) && (cuda::std::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(cuda::std::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(cuda::std::get<1>(dst_props));
          auto delta_w = cuda::std::get<2>(dst_props);

          return (sigma_v / sigma_w) * (1 + delta_w);
        } else {
          return weight_t{0};
        }
      },
      weight_t{0},
      reduce_op::plus<weight_t>{},
      deltas.begin(),
      do_expensive_check);

    update_edge_src_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
      src_properties.mutable_view());
    update_edge_dst_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
      dst_properties.mutable_view());
  }
}

template <typename vertex_t, typename edge_t, typename origin_t, bool multi_gpu>
std::tuple<key_bucket_t<vertex_t, origin_t, multi_gpu, true>, std::vector<size_t>>
batch_partition_frontier(raft::handle_t const& handle,
                         graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                         raft::device_span<edge_t const> out_degrees,
                         key_bucket_t<vertex_t, origin_t, multi_gpu, true>&& frontier,
                         size_t num_sources)
{
  static_assert(!multi_gpu, "batch_partition_frontier is only supported on single-GPU");

  std::optional<rmm::device_uvector<size_t>> source_lasts{
    std::nullopt};  // last source IDs (exclusive) for each batch ID
  if (num_sources > 1) {
    rmm::device_uvector<size_t> source_out_edge_counts(num_sources, handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), source_out_edge_counts.begin(), source_out_edge_counts.end(), 0);
    thrust::for_each(
      handle.get_thrust_policy(),
      frontier.begin(),
      frontier.end(),
      [out_degrees,
       source_out_edge_counts = raft::device_span<size_t>(
         source_out_edge_counts.data(), source_out_edge_counts.size())] __device__(auto pair) {
        auto out_degree = out_degrees[cuda::std::get<0>(pair)];
        auto source_idx = cuda::std::get<1>(pair);
        cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(
          source_out_edge_counts[source_idx]);
        counter.fetch_add(static_cast<size_t>(out_degree), cuda::std::memory_order_relaxed);
      });
    auto max_pushes = thrust::reduce(
      handle.get_thrust_policy(), source_out_edge_counts.begin(), source_out_edge_counts.end());
    auto total_global_mem = handle.get_device_properties().totalGlobalMem;
    auto constexpr max_edge_tuple_data_ratio =
      0.25;  // limit max_sources_per_batch so that the edge tuple data should not exceed
             // max_edge_tuple_data_ratio of total_global_mem
    auto max_pushes_per_batch =
      static_cast<size_t>((total_global_mem * max_edge_tuple_data_ratio) /
                          (sizeof(vertex_t) + sizeof(origin_t) + sizeof(edge_t)));
    if (max_pushes > max_pushes_per_batch) {
      size_t num_batches   = (max_pushes + max_pushes_per_batch - 1) / max_pushes_per_batch;
      max_pushes_per_batch = (max_pushes + num_batches - 1) / num_batches;
      rmm::device_uvector<size_t> source_out_edge_count_inclusive_sums(num_sources,
                                                                       handle.get_stream());
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             source_out_edge_counts.begin(),
                             source_out_edge_counts.end(),
                             source_out_edge_count_inclusive_sums.begin());
      source_lasts     = rmm::device_uvector<size_t>(num_batches, handle.get_stream());
      auto count_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{1}), multiplier_t<size_t>{max_pushes_per_batch});
      thrust::lower_bound(handle.get_thrust_policy(),
                          source_out_edge_count_inclusive_sums.begin(),
                          source_out_edge_count_inclusive_sums.end(),
                          count_first,
                          count_first + num_batches,
                          source_lasts->begin());
    }
  }

  std::vector<size_t> batch_offsets{0, frontier.size()};
  if (source_lasts) {
    thrust::sort(
      handle.get_thrust_policy(),
      frontier.begin(),
      frontier.end(),
      cuda::proclaim_return_type<bool>([source_lasts = raft::device_span<size_t const>(
                                          source_lasts->data(),
                                          source_lasts->size())] __device__(auto lhs, auto rhs) {
        auto lhs_batch_idx = cuda::std::distance(
          source_lasts.begin(),
          thrust::upper_bound(
            thrust::seq, source_lasts.begin(), source_lasts.end(), cuda::std::get<1>(lhs)));
        auto rhs_batch_idx = cuda::std::distance(
          source_lasts.begin(),
          thrust::upper_bound(
            thrust::seq, source_lasts.begin(), source_lasts.end(), cuda::std::get<1>(rhs)));
        return cuda::std::make_tuple(
                 lhs_batch_idx, cuda::std::get<0>(lhs), cuda::std::get<1>(lhs)) <
               cuda::std::make_tuple(rhs_batch_idx, cuda::std::get<0>(rhs), cuda::std::get<1>(rhs));
      }));

    rmm::device_uvector<size_t> d_batch_offsets(source_lasts->size() + 1, handle.get_stream());
    auto batch_id_first = thrust::make_transform_iterator(
      frontier.begin(),
      cuda::proclaim_return_type<size_t>(
        [source_lasts = raft::device_span<size_t const>(
           source_lasts->data(), source_lasts->size())] __device__(auto pair) {
          return static_cast<size_t>(cuda::std::distance(
            source_lasts.begin(),
            thrust::upper_bound(
              thrust::seq, source_lasts.begin(), source_lasts.end(), cuda::std::get<1>(pair))));
        }));
    thrust::lower_bound(handle.get_thrust_policy(),
                        batch_id_first,
                        batch_id_first + frontier.size(),
                        thrust::make_counting_iterator<size_t>(0),
                        thrust::make_counting_iterator<size_t>(d_batch_offsets.size()),
                        d_batch_offsets.begin());
    batch_offsets.resize(d_batch_offsets.size());
    raft::update_host(
      batch_offsets.data(), d_batch_offsets.data(), d_batch_offsets.size(), handle.get_stream());
    handle.sync_stream();
  }

  return std::make_tuple(std::move(frontier), batch_offsets);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> multisource_bfs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<edge_t const> out_degrees,
  VertexIterator sources_first,
  VertexIterator sources_last,
  bool do_expensive_check)
{
  static_assert(!multi_gpu, "Multi-GPU is currently not supported for multisource_bfs");

  using origin_t = uint16_t;

  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  // Use 2D arrays to track per-source distances and sigmas
  // Layout: [source_idx * local_vertex_partition_range_size + v_offset]
  auto local_vertex_partition_range_size = graph_view.local_vertex_partition_range_size();
  auto num_sources = static_cast<size_t>(cuda::std::distance(sources_first, sources_last));

  CUGRAPH_EXPECTS(
    num_sources <= std::numeric_limits<origin_t>::max(),
    "Number of sources exceeds maximum value for origin_t (uint16_t), would cause overflow");

  key_bucket_t<vertex_t, origin_t, multi_gpu, true> cur_frontier(handle);
  rmm::device_uvector<edge_t> sigmas_2d(num_sources * local_vertex_partition_range_size,
                                        handle.get_stream());
  rmm::device_uvector<vertex_t> distances_2d(num_sources * local_vertex_partition_range_size,
                                             handle.get_stream());
  detail::scalar_fill(handle, sigmas_2d.data(), sigmas_2d.size(), edge_t{0});
  detail::scalar_fill(handle, distances_2d.data(), distances_2d.size(), invalid_distance);

  {
    auto tagged_source_first =
      thrust::make_zip_iterator(sources_first, thrust::make_counting_iterator(origin_t{0}));

    cur_frontier.insert(tagged_source_first, tagged_source_first + num_sources);

    thrust::for_each(
      handle.get_thrust_policy(),
      tagged_source_first,
      tagged_source_first + num_sources,
      [d_sigma_2d    = sigmas_2d.begin(),
       d_distance_2d = distances_2d.begin(),
       local_vertex_partition_range_size,
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto tagged_source) {
        auto v             = cuda::std::get<0>(tagged_source);
        auto origin        = cuda::std::get<1>(tagged_source);
        auto v_offset      = v - v_first;
        auto idx           = origin * local_vertex_partition_range_size + v_offset;
        d_distance_2d[idx] = 0;
        d_sigma_2d[idx]    = 1;
      });
  }

  edge_t hop{0};

  while (cur_frontier.size() > 0) {
    using bfs_edge_tuple_t = cuda::std::tuple<vertex_t, origin_t, edge_t>;

    // Step 1: partition the frontier to batches

    std::vector<size_t> batch_offsets{};
    std::tie(cur_frontier, batch_offsets) = batch_partition_frontier(
      handle, graph_view, out_degrees, std::move(cur_frontier), num_sources);

    // Step 2: Iterate over the batches (find new frontier from current frontier)

    rmm::device_uvector<vertex_t> new_frontier_vertices(0, handle.get_stream());
    rmm::device_uvector<origin_t> new_frontier_origins(0, handle.get_stream());
    for (size_t batch_idx = 0; batch_idx < (batch_offsets.size() - 1); ++batch_idx) {
      // Step 2-1: Update the frontier for this batch

      auto this_batch_frontier = key_bucket_t<vertex_t, origin_t, multi_gpu, true>(
        handle,
        raft::device_span<vertex_t const>(cur_frontier.vertex_begin() + batch_offsets[batch_idx],
                                          batch_offsets[batch_idx + 1] - batch_offsets[batch_idx]),
        raft::device_span<origin_t const>(cur_frontier.tag_begin() + batch_offsets[batch_idx],
                                          batch_offsets[batch_idx + 1] - batch_offsets[batch_idx]));

      // Step 2-2: Extract ALL edges from frontier (filtered by unvisited vertices)

      auto [this_batch_frontier_vertices, this_batch_frontier_origins, this_batch_sigmas] =
        extract_transform_if_v_frontier_outgoing_e(
          handle,
          graph_view,
          this_batch_frontier,
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<bfs_edge_tuple_t>(
            [d_sigma_2d = sigmas_2d.begin(),
             local_vertex_partition_range_size,
             v_first = graph_view.local_vertex_partition_range_first()] __device__(auto tagged_src,
                                                                                   auto dst,
                                                                                   auto,
                                                                                   auto,
                                                                                   auto) {
              auto src        = cuda::std::get<0>(tagged_src);
              auto origin     = cuda::std::get<1>(tagged_src);
              auto src_offset = src - v_first;
              auto src_idx    = origin * local_vertex_partition_range_size + src_offset;
              auto src_sigma  = static_cast<edge_t>(d_sigma_2d[src_idx]);

              return cuda::std::make_tuple(dst, origin, src_sigma);
            }),
          // PREDICATE: only process edges to unvisited vertices
          cuda::proclaim_return_type<bool>(
            [d_distances_2d = distances_2d.begin(),
             local_vertex_partition_range_size,
             v_first = graph_view.local_vertex_partition_range_first(),
             invalid_distance] __device__(auto tagged_src, auto dst, auto, auto, auto) {
              auto origin     = cuda::std::get<1>(tagged_src);
              auto dst_offset = dst - v_first;
              auto dst_idx    = origin * local_vertex_partition_range_size + dst_offset;
              return d_distances_2d[dst_idx] == invalid_distance;
            }));

      // Step 2-3: Reduce by (destination, origin) - sums sigmas for multiple paths
      // Sort by (destination, origin) pairs

      thrust::sort_by_key(handle.get_thrust_policy(),
                          thrust::make_zip_iterator(this_batch_frontier_vertices.begin(),
                                                    this_batch_frontier_origins.begin()),
                          thrust::make_zip_iterator(this_batch_frontier_vertices.end(),
                                                    this_batch_frontier_origins.end()),
                          this_batch_sigmas.begin());

      // Use in-place reduction to avoid temporaries
      auto reduced_result = thrust::reduce_by_key(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(this_batch_frontier_vertices.begin(),
                                  this_batch_frontier_origins.begin()),
        thrust::make_zip_iterator(this_batch_frontier_vertices.end(),
                                  this_batch_frontier_origins.end()),
        this_batch_sigmas.begin(),
        thrust::make_zip_iterator(
          this_batch_frontier_vertices.begin(),
          this_batch_frontier_origins.begin()),  // Output keys (overwrite input)
        this_batch_sigmas.begin(),               // Output values (overwrite input)
        thrust::equal_to<cuda::std::tuple<vertex_t, origin_t>>{},
        thrust::plus<edge_t>{});
      size_t num_reduced = cuda::std::distance(this_batch_sigmas.begin(), reduced_result.second);

      // Step 2-4: Manual array updates using in-place reduced data
      // Get count from the values output since keys output is a zip iterator

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(this_batch_frontier_vertices.begin(),
                                                 this_batch_frontier_origins.begin(),
                                                 this_batch_sigmas.begin()),
                       thrust::make_zip_iterator(this_batch_frontier_vertices.begin(),
                                                 this_batch_frontier_origins.begin(),
                                                 this_batch_sigmas.begin()) +
                         num_reduced,
                       [d_distances_2d = distances_2d.begin(),
                        d_sigmas_2d    = sigmas_2d.begin(),
                        local_vertex_partition_range_size,
                        v_first = graph_view.local_vertex_partition_range_first(),
                        hop] __device__(auto tuple) {
                         auto v        = cuda::std::get<0>(tuple);
                         auto origin   = cuda::std::get<1>(tuple);
                         auto sigma    = cuda::std::get<2>(tuple);
                         auto v_offset = v - v_first;
                         auto idx      = origin * local_vertex_partition_range_size + v_offset;

                         // Direct assignment - no atomics needed because reduction already handled
                         // duplicates
                         d_distances_2d[idx] = hop + 1;
                         d_sigmas_2d[idx]    = sigma;
                       });

      this_batch_frontier_vertices.resize(num_reduced, handle.get_stream());
      this_batch_frontier_origins.resize(num_reduced, handle.get_stream());
      this_batch_sigmas.resize(0, handle.get_stream());
      this_batch_frontier_vertices.shrink_to_fit(handle.get_stream());
      this_batch_frontier_origins.shrink_to_fit(handle.get_stream());
      this_batch_sigmas.shrink_to_fit(handle.get_stream());

      if (batch_idx == 0) {
        new_frontier_vertices = std::move(this_batch_frontier_vertices);
        new_frontier_origins  = std::move(this_batch_frontier_origins);
      } else {
        rmm::device_uvector<vertex_t> tmp_frontier_vertices(
          new_frontier_vertices.size() + this_batch_frontier_vertices.size(), handle.get_stream());
        rmm::device_uvector<origin_t> tmp_frontier_origins(
          new_frontier_origins.size() + this_batch_frontier_origins.size(), handle.get_stream());
        thrust::merge(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(new_frontier_vertices.begin(), new_frontier_origins.begin()),
          thrust::make_zip_iterator(new_frontier_vertices.end(), new_frontier_origins.end()),
          thrust::make_zip_iterator(this_batch_frontier_vertices.begin(),
                                    this_batch_frontier_origins.begin()),
          thrust::make_zip_iterator(this_batch_frontier_vertices.end(),
                                    this_batch_frontier_origins.end()),
          thrust::make_zip_iterator(tmp_frontier_vertices.begin(), tmp_frontier_origins.begin()));
        new_frontier_vertices = std::move(tmp_frontier_vertices);
        new_frontier_origins  = std::move(tmp_frontier_origins);
      }
    }

    cur_frontier = key_bucket_t<vertex_t, origin_t, multi_gpu, true>(
      handle, std::move(new_frontier_vertices), std::move(new_frontier_origins));
    ++hop;
  }

  return std::make_tuple(std::move(distances_2d), std::move(sigmas_2d));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
void multisource_backward_pass(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<edge_t const> out_degrees,
  raft::device_span<weight_t> centralities,
  rmm::device_uvector<vertex_t>&& distances_2d,
  rmm::device_uvector<edge_t>&& sigmas_2d,
  VertexIterator sources_first,
  VertexIterator sources_last,
  bool include_endpoints,
  bool do_expensive_check)
{
  static_assert(!multi_gpu, "Multi-GPU is currently not supported for multisource_backward_pass");

  using origin_t = uint16_t;

  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  auto local_vertex_partition_range_size =
    static_cast<size_t>(graph_view.local_vertex_partition_range_size());
  auto num_sources = static_cast<size_t>(cuda::std::distance(sources_first, sources_last));

  CUGRAPH_EXPECTS(
    num_sources <= std::numeric_limits<origin_t>::max(),
    "Number of sources exceeds maximum value for origin_t (uint16_t), would cause overflow");

  thrust::fill(handle.get_thrust_policy(), centralities.begin(), centralities.end(), weight_t{0});

  rmm::device_uvector<weight_t> delta_buffer(num_sources * local_vertex_partition_range_size,
                                             handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), delta_buffer.begin(), delta_buffer.end(), weight_t{0});

  // (vertex, source idx) pairs for each distance level [0, global_max_distance + 1)
  rmm::device_uvector<vertex_t> all_vertices(0, handle.get_stream());
  rmm::device_uvector<origin_t> all_sources(0, handle.get_stream());
  std::vector<size_t> h_distance_offsets{};
  rmm::device_uvector<size_t> d_distance_offsets(0, handle.get_stream());
  {
    auto d_first = thrust::make_transform_iterator(
      distances_2d.begin(),
      cuda::proclaim_return_type<vertex_t>([invalid_distance] __device__(vertex_t d) {
        return d == invalid_distance ? vertex_t{0} : d;
      }));
    vertex_t global_max_distance = thrust::reduce(handle.get_thrust_policy(),
                                                  d_first,
                                                  d_first + distances_2d.size(),
                                                  vertex_t{0},
                                                  thrust::maximum<vertex_t>());

    rmm::device_uvector<size_t> d_distance_counts(global_max_distance + 1, handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), d_distance_counts.begin(), d_distance_counts.end(), size_t{0});

    thrust::for_each_n(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      num_sources * local_vertex_partition_range_size,
      [distances_2d = distances_2d.data(),
       distance_counts =
         raft::device_span<size_t>(d_distance_counts.data(), d_distance_counts.size()),
       local_vertex_partition_range_size,
       global_max_distance] __device__(size_t global_idx) {
        size_t source_idx         = global_idx / local_vertex_partition_range_size;
        vertex_t v_offset         = global_idx % local_vertex_partition_range_size;
        const vertex_t* distances = distances_2d + source_idx * local_vertex_partition_range_size;
        vertex_t dist             = distances[v_offset];

        if (dist >= 0 && dist <= global_max_distance) {
          cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(distance_counts[dist]);
          counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
        }
      });

    std::vector<size_t> h_distance_counts(global_max_distance + 1);
    raft::update_host(h_distance_counts.data(),
                      d_distance_counts.data(),
                      d_distance_counts.size(),
                      handle.get_stream());
    handle.sync_stream();

    h_distance_offsets.resize(h_distance_counts.size() + 1);
    h_distance_offsets[0] = 0;
    std::inclusive_scan(
      h_distance_counts.begin(), h_distance_counts.end(), h_distance_offsets.begin() + 1);
    d_distance_offsets.resize(h_distance_offsets.size(), handle.get_stream());
    raft::update_device(d_distance_offsets.data(),
                        h_distance_offsets.data(),
                        h_distance_offsets.size(),
                        handle.get_stream());

    all_vertices.resize(h_distance_offsets.back(), handle.get_stream());
    all_sources.resize(h_distance_offsets.back(), handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), d_distance_counts.begin(), d_distance_counts.end(), size_t{0});

    thrust::for_each_n(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      num_sources * local_vertex_partition_range_size,
      [distances_2d = distances_2d.data(),
       distance_counts =
         raft::device_span<size_t>(d_distance_counts.data(), d_distance_counts.size()),
       all_vertices = raft::device_span<vertex_t>(all_vertices.data(), all_vertices.size()),
       all_sources  = raft::device_span<origin_t>(all_sources.data(), all_sources.size()),
       distance_offsets =
         raft::device_span<size_t>(d_distance_offsets.data(), d_distance_offsets.size()),
       local_vertex_partition_range_size,
       v_first = graph_view.local_vertex_partition_range_first(),
       global_max_distance] __device__(size_t global_idx) {
        size_t source_idx         = global_idx / local_vertex_partition_range_size;
        vertex_t v_offset         = global_idx % local_vertex_partition_range_size;
        const vertex_t* distances = distances_2d + source_idx * local_vertex_partition_range_size;
        vertex_t dist             = distances[v_offset];

        if (dist >= 0 && dist <= global_max_distance) {
          cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(distance_counts[dist]);
          size_t local_offset  = counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
          size_t global_offset = distance_offsets[dist] + local_offset;
          all_vertices[global_offset] = v_first + v_offset;
          all_sources[global_offset]  = source_idx;
        }
      });
  }

  // Segment-sort all_vertices & all_sources in each distance level
  if (h_distance_offsets.back() > 0) {
    auto approx_vertices_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 20); /* tuning parameter */

    auto [h_distance_level_chunk_offsets, h_vertex_chunk_offsets] =
      detail::compute_offset_aligned_element_chunks(
        handle,
        raft::device_span<size_t const>(d_distance_offsets.data(), d_distance_offsets.size()),
        all_vertices.size(),
        approx_vertices_to_sort_per_iteration);

    auto num_chunks = h_distance_level_chunk_offsets.size() - 1;

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

    rmm::device_uvector<vertex_t> sorted_vertices(all_vertices.size(), handle.get_stream());
    rmm::device_uvector<origin_t> sorted_sources(all_sources.size(), handle.get_stream());

    for (size_t chunk_i = 0; chunk_i < num_chunks; ++chunk_i) {
      size_t chunk_vertex_start    = h_vertex_chunk_offsets[chunk_i];
      size_t chunk_vertex_end      = h_vertex_chunk_offsets[chunk_i + 1];
      size_t chunk_distance_start  = h_distance_level_chunk_offsets[chunk_i];
      size_t chunk_distance_end    = h_distance_level_chunk_offsets[chunk_i + 1];
      size_t chunk_size            = chunk_vertex_end - chunk_vertex_start;
      size_t num_segments_in_chunk = chunk_distance_end - chunk_distance_start;

      if (num_segments_in_chunk > 0) {
        auto offset_first = thrust::make_transform_iterator(
          d_distance_offsets.data() + chunk_distance_start,
          cuda::proclaim_return_type<size_t>([chunk_vertex_start] __device__(size_t offset) {
            return offset - chunk_vertex_start;
          }));

        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedSort::SortPairs(nullptr,
                                            temp_storage_bytes,
                                            all_vertices.data() + chunk_vertex_start,
                                            sorted_vertices.data() + chunk_vertex_start,
                                            all_sources.data() + chunk_vertex_start,
                                            sorted_sources.data() + chunk_vertex_start,
                                            chunk_size,
                                            num_segments_in_chunk,
                                            offset_first,
                                            offset_first + 1,
                                            handle.get_stream());

        if (temp_storage_bytes > d_tmp_storage.size()) {
          d_tmp_storage = rmm::device_uvector<std::byte>(temp_storage_bytes, handle.get_stream());
        }

        cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                            temp_storage_bytes,
                                            all_vertices.data() + chunk_vertex_start,
                                            sorted_vertices.data() + chunk_vertex_start,
                                            all_sources.data() + chunk_vertex_start,
                                            sorted_sources.data() + chunk_vertex_start,
                                            chunk_size,
                                            num_segments_in_chunk,
                                            offset_first,
                                            offset_first + 1,
                                            handle.get_stream());
      }
    }

    all_vertices = std::move(sorted_vertices);
    all_sources  = std::move(sorted_sources);
  }

  // Process distance levels using pre-computed buckets starting from the farthest distance level
  // (now with sorted (vertex, source idx) pairs)
  for (auto d = static_cast<vertex_t>(h_distance_offsets.size() - 2); d > 1; --d) {
    // Step 1: Create vertex frontier with all vertices at distance d-1 for all sources
    // Use tagged vertices with (vertex, source_idx) pairs
    using tagged_vertex_t = cuda::std::tuple<vertex_t, size_t>;

    // Get vertices at distance d-1 from consecutive arrays (O(1) lookup)
    size_t start_offset                = h_distance_offsets[d - 1];
    size_t end_offset                  = h_distance_offsets[d];
    size_t total_vertices_at_d_minus_1 = end_offset - start_offset;

    if (total_vertices_at_d_minus_1 > 0) {
      // Step 2: Use extract_transform_if_v_frontier_e to enumerate all qualifying edges
      // This extracts (src, tag, dst) triplets as recommended
      // Create a proper frontier object for the tagged vertices
      key_bucket_t<vertex_t, origin_t, multi_gpu, true> cur_frontier(handle);

      // Insert tagged vertices directly using zip iterator (no temporary needed)
      auto pair_first = thrust::make_zip_iterator(all_vertices.begin() + start_offset,
                                                  all_sources.begin() + start_offset);
      cur_frontier.insert(pair_first, pair_first + total_vertices_at_d_minus_1);

      std::vector<size_t> batch_offsets{};
      std::tie(cur_frontier, batch_offsets) = batch_partition_frontier(
        handle, graph_view, out_degrees, std::move(cur_frontier), num_sources);

      for (size_t batch_idx = 0; batch_idx < (batch_offsets.size() - 1); ++batch_idx) {
        // Step 2-1: Update the frontier for this batch

        auto this_batch_frontier = key_bucket_t<vertex_t, origin_t, multi_gpu, true>(
          handle,
          raft::device_span<vertex_t const>(
            cur_frontier.vertex_begin() + batch_offsets[batch_idx],
            batch_offsets[batch_idx + 1] - batch_offsets[batch_idx]),
          raft::device_span<origin_t const>(
            cur_frontier.tag_begin() + batch_offsets[batch_idx],
            batch_offsets[batch_idx + 1] - batch_offsets[batch_idx]));

        // Step 2-2: Extract ALL edges from frontier (only process edges where dst is at distance d)

        auto [srcs, source_indices, deltas] = extract_transform_if_v_frontier_outgoing_e(
          handle,
          graph_view,
          this_batch_frontier,
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<cuda::std::tuple<vertex_t, origin_t, weight_t>>(
            [d,
             distances_2d = distances_2d.data(),
             sigmas_2d    = sigmas_2d.data(),
             delta_buffer = delta_buffer.data(),
             local_vertex_partition_range_size,
             invalid_distance,
             v_first = graph_view.local_vertex_partition_range_first()] __device__(auto tagged_src,
                                                                                   auto dst,
                                                                                   auto,
                                                                                   auto,
                                                                                   auto) {
              auto src        = cuda::std::get<0>(tagged_src);
              auto source_idx = cuda::std::get<1>(tagged_src);

              // Calculate delta using Brandes formula with accumulated deltas
              const vertex_t* distances =
                distances_2d + source_idx * local_vertex_partition_range_size;
              const edge_t* sigmas = sigmas_2d + source_idx * local_vertex_partition_range_size;
              const weight_t* deltas =
                delta_buffer + source_idx * local_vertex_partition_range_size;

              auto src_offset = src - v_first;
              auto dst_offset = dst - v_first;

              auto sigma_v = static_cast<weight_t>(sigmas[src_offset]);
              auto sigma_w = static_cast<weight_t>(sigmas[dst_offset]);

              // Get accumulated delta for destination vertex
              weight_t delta_w = deltas[dst_offset];
              weight_t delta   = (sigma_v / sigma_w) * (1 + delta_w);

              return cuda::std::make_tuple(src, source_idx, delta);
            }),
          cuda::proclaim_return_type<bool>(
            [d,
             distances_2d = distances_2d.data(),
             local_vertex_partition_range_size,
             v_first = graph_view.local_vertex_partition_range_first()] __device__(auto tagged_src,
                                                                                   auto dst,
                                                                                   auto,
                                                                                   auto,
                                                                                   auto) {
              auto source_idx = cuda::std::get<1>(tagged_src);
              const vertex_t* distances =
                distances_2d + source_idx * local_vertex_partition_range_size;
              auto dst_offset = dst - v_first;
              return distances[dst_offset] == d;
            }));

        // Work directly with the result buffer
        if (srcs.size() > 0) {
          // Step 3: Sort using (src, source_index) as composite key for efficient reduction
          thrust::stable_sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(srcs.begin(), source_indices.begin()),  // Composite key
            thrust::make_zip_iterator(srcs.end(), source_indices.end()),
            deltas.begin());  // Values to sort

          // Step 4: Use reduce_by_key with in-place reduction
          // Reduce by key and get count in one operation - overwrite input buffers
          auto reduced_result = thrust::reduce_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(srcs.begin(), source_indices.begin()),
            thrust::make_zip_iterator(srcs.end(), source_indices.end()),
            deltas.begin(),
            thrust::make_zip_iterator(srcs.begin(),
                                      source_indices.begin()),  // Output keys (overwrite input)
            deltas.begin(),                                     // Output values (overwrite input)
            thrust::equal_to<cuda::std::tuple<vertex_t, origin_t>>{},
            thrust::plus<weight_t>{});
          size_t num_reduced = cuda::std::distance(deltas.begin(), reduced_result.second);

          // Step 5: Update centralities and deltas from the in-place reduced results
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_reduced),
            [srcs           = srcs.data(),
             source_indices = source_indices.data(),
             deltas         = deltas.data(),
             centralities   = centralities.data(),
             delta_buffer   = delta_buffer.data(),
             local_vertex_partition_range_size,
             v_first = graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
              auto src        = srcs[i];
              auto source_idx = source_indices[i];
              auto delta      = deltas[i];

              // Update centrality using atomic for floating point
              auto src_offset = src - v_first;
              cuda::atomic_ref<weight_t, cuda::thread_scope_device> centrality_counter(
                centralities[src_offset]);
              centrality_counter.fetch_add(delta, cuda::std::memory_order_relaxed);

              // Accumulate delta for next iteration using atomic for floating point
              weight_t* source_deltas =
                delta_buffer + source_idx * local_vertex_partition_range_size;
              cuda::atomic_ref<weight_t, cuda::thread_scope_device> delta_counter(
                source_deltas[src_offset]);
              delta_counter.fetch_add(delta, cuda::std::memory_order_relaxed);
            });
        }
      }
    }
  }

  // Handle source and destination vertex contributions if include_endpoints is true
  if (include_endpoints) {
    auto v_first = graph_view.local_vertex_partition_range_first();

    // Create small temporary buffer for source vertex access (needed for 2D array indexing)
    rmm::device_uvector<vertex_t> sources_buffer(num_sources, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), sources_first, sources_last, sources_buffer.begin());

    // Handle source vertex contributions
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_sources),
      [distances_2d = distances_2d.data(),
       sigmas_2d    = sigmas_2d.data(),
       sources      = sources_buffer.data(),
       centralities = centralities.data(),
       local_vertex_partition_range_size,
       v_first] __device__(size_t source_idx) {
        const vertex_t* distances = distances_2d + source_idx * local_vertex_partition_range_size;
        const edge_t* sigmas      = sigmas_2d + source_idx * local_vertex_partition_range_size;
        vertex_t source_vertex    = sources[source_idx];

        // Source vertex contribution: count of reachable vertices (excluding self)
        weight_t source_contribution = 0;
        for (vertex_t v = 0; v < local_vertex_partition_range_size; ++v) {
          if (v != source_vertex && distances[v] != std::numeric_limits<vertex_t>::max()) {
            source_contribution += 1.0;
          }
        }
        // Convert global vertex ID to local offset
        auto source_offset = source_vertex - v_first;
        cuda::atomic_ref<weight_t, cuda::thread_scope_device> centrality_counter(
          centralities[source_offset]);
        centrality_counter.fetch_add(source_contribution, cuda::std::memory_order_relaxed);
      });

    // Handle destination vertex contributions
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_sources),
      [distances_2d = distances_2d.data(),
       sigmas_2d    = sigmas_2d.data(),
       sources      = sources_buffer.data(),
       centralities = centralities.data(),
       local_vertex_partition_range_size,
       v_first] __device__(size_t source_idx) {
        const vertex_t* distances = distances_2d + source_idx * local_vertex_partition_range_size;
        const edge_t* sigmas      = sigmas_2d + source_idx * local_vertex_partition_range_size;
        vertex_t source_vertex    = sources[source_idx];

        // Destination vertex contributions: each reachable vertex contributes to its own centrality
        for (vertex_t v = 0; v < local_vertex_partition_range_size; ++v) {
          if (v != source_vertex && distances[v] != std::numeric_limits<vertex_t>::max()) {
            // Each destination vertex contributes 1 to its own centrality
            auto dest_offset = v - v_first;
            cuda::atomic_ref<weight_t, cuda::thread_scope_device> centrality_counter(
              centralities[dest_offset]);
            centrality_counter.fetch_add(1.0, cuda::std::memory_order_relaxed);
          }
        }
      });
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
rmm::device_uvector<weight_t> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       vertices_begin,
                       vertices_end,
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  rmm::device_uvector<weight_t> centralities(graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());
  detail::scalar_fill(handle, centralities.data(), centralities.size(), weight_t{0});

  size_t num_sources = cuda::std::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_offsets{{0, num_sources}};
  int my_rank = 0;

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_sources, handle.get_stream());

    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_offsets.resize(source_counts.size() + 1);
    source_offsets[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_offsets.begin() + 1);
    my_rank = handle.get_comms().get_rank();
  }

  if constexpr (multi_gpu) {
    // Multi-GPU: Use sequential version
    for (size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
      //
      //  BFS
      //
      constexpr size_t bucket_idx_cur = 0;
      constexpr size_t num_buckets    = 2;

      vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier(handle, num_buckets);

      if ((source_idx >= source_offsets[my_rank]) && (source_idx < source_offsets[my_rank + 1])) {
        vertex_frontier.bucket(bucket_idx_cur)
          .insert(vertices_begin + (source_idx - source_offsets[my_rank]),
                  vertices_begin + (source_idx - source_offsets[my_rank]) + 1);
      }

      auto [distances, sigmas] = detail::brandes_bfs(
        handle, graph_view, edge_weight_view, vertex_frontier, do_expensive_check);
      detail::accumulate_vertex_results(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<weight_t>{centralities.data(), centralities.size()},
        std::move(distances),
        std::move(sigmas),
        include_endpoints,
        do_expensive_check);
    }
  } else {
    auto out_degrees = graph_view.compute_out_degrees(handle);

    // Single-GPU: Use parallel version
    // Process sources in batches to respect origin_t (uint16_t) limit
    size_t max_sources_per_batch =
      std::min(static_cast<size_t>(std::numeric_limits<uint16_t>::max()), num_sources);
    if (max_sources_per_batch > 1) {
      auto total_global_mem = handle.get_device_properties().totalGlobalMem;
      auto constexpr max_multisource_bfs_result_ratio =
        0.25;  // limit max_sources_per_batch so that the return value of multisource_bfs should not
               // exceed max_multisource_bfs_result_ratio of total_global_mem
      auto bfs_result_size = static_cast<size_t>(graph_view.local_vertex_partition_range_size()) *
                             (sizeof(vertex_t) + sizeof(edge_t));
      max_sources_per_batch =
        std::max(std::min(static_cast<size_t>(total_global_mem * max_multisource_bfs_result_ratio) /
                            bfs_result_size,
                          max_sources_per_batch),
                 size_t{1});
    }
    size_t num_batches = (num_sources + max_sources_per_batch - 1) / max_sources_per_batch;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      size_t batch_start = batch_idx * max_sources_per_batch;
      size_t batch_end   = std::min(batch_start + max_sources_per_batch, num_sources);

      auto batch_vertices_begin = vertices_begin + batch_start;
      auto batch_vertices_end   = vertices_begin + batch_end;

      auto [distances_2d, sigmas_2d] = detail::multisource_bfs(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<edge_t const>(out_degrees.data(), out_degrees.size()),
        batch_vertices_begin,
        batch_vertices_end,
        do_expensive_check);

      detail::multisource_backward_pass(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<edge_t const>(out_degrees.data(), out_degrees.size()),
        raft::device_span<weight_t>{centralities.data(), centralities.size()},
        std::move(distances_2d),
        std::move(sigmas_2d),
        batch_vertices_begin,
        batch_vertices_end,
        include_endpoints,
        do_expensive_check);
    }
  }

  std::optional<weight_t> scale_nonsource{std::nullopt};
  std::optional<weight_t> scale_source{std::nullopt};

  weight_t num_vertices = static_cast<weight_t>(graph_view.number_of_vertices());
  if (!include_endpoints) num_vertices = num_vertices - 1;

  if ((static_cast<edge_t>(num_sources) == num_vertices) || include_endpoints) {
    if (normalized) {
      scale_nonsource = static_cast<weight_t>(num_sources * (num_vertices - 1));
    } else if (graph_view.is_symmetric()) {
      scale_nonsource =
        static_cast<weight_t>(num_sources * 2) / static_cast<weight_t>(num_vertices);
    } else {
      scale_nonsource = static_cast<weight_t>(num_sources) / static_cast<weight_t>(num_vertices);
    }

    scale_source = scale_nonsource;
  } else if (normalized) {
    scale_nonsource = static_cast<weight_t>(num_sources) * (num_vertices - 1);
    scale_source    = static_cast<weight_t>(num_sources - 1) * (num_vertices - 1);
  } else {
    scale_nonsource = static_cast<weight_t>(num_sources) / num_vertices;
    scale_source    = static_cast<weight_t>(num_sources - 1) / num_vertices;

    if (graph_view.is_symmetric()) {
      *scale_nonsource *= 2;
      *scale_source *= 2;
    }
  }

  if (scale_nonsource) {
    auto iter = thrust::make_zip_iterator(
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      centralities.begin());

    thrust::transform(
      handle.get_thrust_policy(),
      iter,
      iter + centralities.size(),
      centralities.begin(),
      [nonsource = *scale_nonsource,
       source    = *scale_source,
       vertices_begin,
       vertices_end] __device__(auto t) {
        vertex_t v          = cuda::std::get<0>(t);
        weight_t centrality = cuda::std::get<1>(t);

        return (thrust::find(thrust::seq, vertices_begin, vertices_end, v) == vertices_end)
                 ? centrality / nonsource
                 : centrality / source;
      });
  }

  return centralities;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
edge_property_t<edge_t, weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const do_expensive_check)
{
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       vertices_begin,
                       vertices_end,
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  edge_property_t<edge_t, weight_t> centralities(handle, graph_view);

  if (graph_view.has_edge_mask()) {
    auto unmasked_graph_view = graph_view;
    unmasked_graph_view.clear_edge_mask();
    fill_edge_property(
      handle, unmasked_graph_view, centralities.mutable_view(), weight_t{0}, do_expensive_check);
  } else {
    fill_edge_property(
      handle, graph_view, centralities.mutable_view(), weight_t{0}, do_expensive_check);
  }

  size_t num_sources = cuda::std::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_offsets{{0, num_sources}};
  int my_rank = 0;

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_sources, handle.get_stream());

    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_offsets.resize(source_counts.size() + 1);
    source_offsets[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_offsets.begin() + 1);
    my_rank = handle.get_comms().get_rank();
  }

  //
  // FIXME: This could be more efficient using something akin to the
  // technique in WCC.  Take the entire set of sources, insert them into
  // a tagged frontier (tagging each source with itself).  Then we can
  // expand from multiple sources concurrently. The challenge is managing
  // the memory explosion.
  //
  for (size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    //
    //  BFS
    //
    constexpr size_t bucket_idx_cur = 0;
    constexpr size_t num_buckets    = 2;

    vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier(handle, num_buckets);

    if ((source_idx >= source_offsets[my_rank]) && (source_idx < source_offsets[my_rank + 1])) {
      vertex_frontier.bucket(bucket_idx_cur)
        .insert(vertices_begin + (source_idx - source_offsets[my_rank]),
                vertices_begin + (source_idx - source_offsets[my_rank]) + 1);
    }

    //
    //  Now we need to do modified BFS
    //
    // FIXME:  This has an inefficiency in early iterations, as it doesn't have enough work to
    //         keep the GPUs busy.  But we can't run too many at once or we will run out of
    //         memory. Need to investigate options to improve this performance
    auto [distances, sigmas] =
      brandes_bfs(handle, graph_view, edge_weight_view, vertex_frontier, do_expensive_check);
    accumulate_edge_results(handle,
                            graph_view,
                            edge_weight_view,
                            centralities.mutable_view(),
                            std::move(distances),
                            std::move(sigmas),
                            do_expensive_check);
  }

  std::optional<weight_t> scale_factor{std::nullopt};

  if (normalized) {
    weight_t n   = static_cast<weight_t>(graph_view.number_of_vertices());
    scale_factor = n * (n - 1);
  } else if (graph_view.is_symmetric()) {
    scale_factor = weight_t{2};
  }

  if (scale_factor) {
    if (graph_view.number_of_vertices() > 1) {
      if (static_cast<vertex_t>(num_sources) < graph_view.number_of_vertices()) {
        (*scale_factor) *= static_cast<weight_t>(num_sources) /
                           static_cast<weight_t>(graph_view.number_of_vertices());
      }

      auto firsts         = centralities.view().value_firsts();
      auto counts         = centralities.view().edge_counts();
      auto mutable_firsts = centralities.mutable_view().value_firsts();
      for (size_t k = 0; k < counts.size(); k++) {
        thrust::transform(
          handle.get_thrust_policy(),
          firsts[k],
          firsts[k] + counts[k],
          mutable_firsts[k],
          [sf = *scale_factor] __device__(auto centrality) { return centrality / sf; });
      }
    }
  }

  return centralities;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  if (vertices) {
    return detail::betweenness_centrality(handle,
                                          graph_view,
                                          edge_weight_view,
                                          vertices->begin(),
                                          vertices->end(),
                                          normalized,
                                          include_endpoints,
                                          do_expensive_check);
  } else {
    return detail::betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      include_endpoints,
      do_expensive_check);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
edge_property_t<edge_t, weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> vertices,
  bool const normalized,
  bool const do_expensive_check)
{
  if (vertices) {
    return detail::edge_betweenness_centrality(handle,
                                               graph_view,
                                               edge_weight_view,
                                               vertices->begin(),
                                               vertices->end(),
                                               normalized,
                                               do_expensive_check);
  } else {
    return detail::edge_betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      do_expensive_check);
  }
}

}  // namespace cugraph
