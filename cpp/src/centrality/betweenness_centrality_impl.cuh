/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "prims/count_if_e.cuh"
#include "prims/count_if_v.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_e.cuh"

#include "prims/fill_edge_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/transform_reduce_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"
#include "prims/detail/prim_functors.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

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
  __device__ thrust::tuple<vertex_t, vertex_t> operator()(
    vertex_t src,
    vertex_t dst,
    thrust::tuple<vertex_t, edge_t, weight_t> src_props,
    thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
    cuda::std::nullopt_t) const
  {
    return thrust::make_tuple(src, dst);
  }

  template <typename edge_t, typename weight_t>
  __device__ thrust::tuple<vertex_t, vertex_t, edge_t> operator()(
    vertex_t src,
    vertex_t dst,
    thrust::tuple<vertex_t, edge_t, weight_t> src_props,
    thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
    edge_t edge_multi_index) const
  {
    return thrust::make_tuple(src, dst, edge_multi_index);
  }
};

template <typename vertex_t>
struct extract_edge_pred_op_t {
  vertex_t d{};

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             thrust::tuple<vertex_t, edge_t, weight_t> src_props,
                             thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
                             cuda::std::nullopt_t) const
  {
    return ((thrust::get<0>(src_props) == (d - 1)) && (thrust::get<0>(dst_props) == d));
  }

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             thrust::tuple<vertex_t, edge_t, weight_t> src_props,
                             thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
                             edge_t edge_multi_index) const
  {
    return ((thrust::get<0>(src_props) == (d - 1)) && (thrust::get<0>(dst_props) == d));
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
                        return thrust::make_tuple(
                          cuda::std::make_optional(bucket_idx_next),
                          cuda::std::make_optional(thrust::make_tuple(hop + 1, v_sigma)));
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
  update_edge_src_property(
    handle,
    graph_view,
    sigmas.begin(),
    src_sigmas.mutable_view());
  update_edge_dst_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
    view_concat(dst_distances.mutable_view(), dst_sigmas.mutable_view()));
  fill_edge_dst_property(
    handle,
    graph_view,
    dst_deltas.mutable_view(),
    weight_t{0.0});

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
                    vertices_sorted.begin(), vertices_sorted.end(),
                    graph_view.local_vertex_partition_range_first());
    
    // Sort vertices by distance using stable_sort_by_key
    thrust::stable_sort_by_key(
      handle.get_thrust_policy(),
      distance_keys.begin(), distance_keys.end(),   // keys (copied distances)
      vertices_sorted.begin()                       // values (vertices)
    );

    rmm::device_uvector<vertex_t> d_bounds(diameter + 1, handle.get_stream());
    
    // Single vectorized thrust call to compute all bounds for distances 0 to diameter
    thrust::lower_bound(
      handle.get_thrust_policy(),
      distance_keys.begin(), distance_keys.end(),   // sorted distances
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
    max_frontier_size = std::max(max_frontier_size, frontier_count);
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
      key_bucket_t<vertex_t, void, multi_gpu, true> vertex_list(handle, raft::device_span<vertex_t const>(vertices_sorted.data() + h_bounds[d - 1], h_bounds[d] - h_bounds[d - 1]));

      // Compute deltas for frontier vertices
      per_v_transform_reduce_outgoing_e(
        handle,
        graph_view,
        vertex_list,
        src_sigmas.view(),
        view_concat(dst_distances.view(), dst_sigmas.view(), dst_deltas.view()),
        cugraph::edge_dummy_property_t{}.view(),
        [d] __device__(auto, auto, auto src_sigma, auto dst_props, auto) {
          if (thrust::get<0>(dst_props) == d) {
            auto sigma_v = src_sigma;
            auto sigma_w = static_cast<weight_t>(thrust::get<1>(dst_props));
            auto delta_w = thrust::get<2>(dst_props);
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
      update_edge_dst_property(handle, graph_view, vertex_list.cbegin(), vertex_list.cend(), reusable_delta_buffer.begin(), dst_deltas.mutable_view());
      
      // Update centralities - both vertices_sorted and centralities use local vertex IDs
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(vertices_sorted.begin() + h_bounds[d - 1], reusable_delta_buffer.begin()),
        thrust::make_zip_iterator(vertices_sorted.begin() + h_bounds[d], reusable_delta_buffer.begin()),
        [centralities = centralities.data(), v_first = graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
          auto v = thrust::get<0>(pair);
          auto delta = thrust::get<1>(pair);
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

  edge_src_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> src_properties(
    handle, graph_view);
  edge_dst_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> dst_properties(
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
    //  Populate edge_list with edges where `thrust::get<0>(dst_props) == d`
    //  and `thrust::get<0>(dst_props) == (d-1)`
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
        if ((thrust::get<0>(dst_props) == d) && (thrust::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(thrust::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(thrust::get<1>(dst_props));
          auto delta_w = thrust::get<2>(dst_props);

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
        if ((thrust::get<0>(dst_props) == d) && (thrust::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(thrust::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(thrust::get<1>(dst_props));
          auto delta_w = thrust::get<2>(dst_props);

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

  //
  // FIXME: This could be more efficient using something akin to the
  // technique in WCC.  Take the entire set of sources, insert them into
  // a tagged frontier (tagging each source with itself).  Then we can
  // expand from multiple sources concurrently. The challenge is managing
  // the memory explosion.
  //
  // Use multisource_bfs for concurrent processing of all sources
  //
  // Convert vertex iterators to a device span for multisource_bfs
  rmm::device_uvector<vertex_t> sources_buffer(num_sources, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), vertices_begin, vertices_end, sources_buffer.begin());
  
  // Run concurrent multi-source BFS
  auto [distances_2d, sigmas_2d] = multisource_bfs(handle,
                                                   graph_view,
                                                   edge_weight_view,
                                                   raft::device_span<vertex_t const>{sources_buffer.data(), sources_buffer.size()},
                                                   do_expensive_check);
  
  // Use parallel multisource backward pass for better performance
  multisource_backward_pass(handle,
                            graph_view,
                            edge_weight_view,
                            raft::device_span<weight_t>{centralities.data(), centralities.size()},
                            std::move(distances_2d),
                            std::move(sigmas_2d),
                            raft::device_span<vertex_t const>{sources_buffer.data(), sources_buffer.size()},
                            include_endpoints,
                            do_expensive_check);

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
        vertex_t v          = thrust::get<0>(t);
        weight_t centrality = thrust::get<1>(t);

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
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> multisource_bfs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<vertex_t const> sources,
  bool do_expensive_check)
{
   constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
   constexpr size_t bucket_idx_cur{0};
   constexpr size_t bucket_idx_next{1};

   using origin_t = uint32_t;  // Source index type

   // Use 2D arrays to track per-source distances and sigmas
   // Layout: [source_idx * num_vertices + vertex_idx]
   auto num_vertices = graph_view.local_vertex_partition_range_size();
   auto num_sources = sources.size();
   
   rmm::device_uvector<edge_t> sigmas_2d(num_sources * num_vertices, handle.get_stream());
   rmm::device_uvector<vertex_t> distances_2d(num_sources * num_vertices, handle.get_stream());
   detail::scalar_fill(handle, distances_2d.data(), distances_2d.size(), invalid_distance);
   detail::scalar_fill(handle, sigmas_2d.data(), sigmas_2d.size(), edge_t{0});

   // Create tagged frontier with origin indices
   vertex_frontier_t<vertex_t, origin_t, multi_gpu, true> vertex_frontier(handle, 2);

   auto vertex_partition =
     vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());

   // Initialize sources with their origins
   if (sources.size() > 0) {
     rmm::device_uvector<origin_t> origins(sources.size(), handle.get_stream());
     
     // Create origins
     thrust::sequence(handle.get_thrust_policy(), origins.begin(), origins.end(), origin_t{0});
     
     // Insert tagged sources into frontier
     vertex_frontier.bucket(bucket_idx_cur).insert(
       thrust::make_zip_iterator(sources.begin(), origins.begin()),
       thrust::make_zip_iterator(sources.end(), origins.end()));
     
     // Initialize distances and sigmas for sources
     thrust::for_each(
       handle.get_thrust_policy(),
       thrust::make_zip_iterator(sources.begin(), origins.begin()),
       thrust::make_zip_iterator(sources.end(), origins.end()),
       [d_sigma_2d = sigmas_2d.begin(), d_distance_2d = distances_2d.begin(), vertex_partition, num_vertices] __device__(
         auto tagged_source) {
         auto v = thrust::get<0>(tagged_source);
         auto origin = thrust::get<1>(tagged_source);
         auto offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
         auto idx = origin * num_vertices + offset;
         d_distance_2d[idx] = 0;
         d_sigma_2d[idx] = 1;
       });
   }

   edge_t hop{0};
   
   // Debug: Print initial frontier size to show multiple sources
   if (hop == 0) {
     printf("DEBUG: Starting multi-source BFS with %zu sources, initial frontier size: %zu\n", 
            num_sources, vertex_frontier.bucket(bucket_idx_cur).aggregate_size());
     
     // Debug: Print first few source vertices
     if (sources.size() > 0) {
       // Copy first few sources to host for printing
       std::vector<vertex_t> host_sources(std::min(size_t{3}, sources.size()));
       raft::update_host(host_sources.data(), sources.data(), host_sources.size(), handle.get_stream());
       handle.sync_stream();
       
       printf("DEBUG: First 3 source vertices: ");
       for (size_t i = 0; i < host_sources.size(); ++i) {
         printf("%ld ", host_sources[i]);
       }
       printf("\n");
     }
   }

   while (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() > 0) {
     // Debug: Print frontier size for first few iterations
     if (hop < 3) {
       printf("DEBUG: BFS hop %d, frontier size: %zu\n", hop, vertex_frontier.bucket(bucket_idx_cur).aggregate_size());
     }
     
     // Step 1: Extract ALL edges from frontier (no filtering)
     using bfs_edge_tuple_t = thrust::tuple<vertex_t, origin_t, edge_t>;
     
     auto result = detail::
       extract_transform_if_v_frontier_e<false, bfs_edge_tuple_t, void>(
         handle,
         graph_view,
         vertex_frontier.bucket(bucket_idx_cur),
         edge_src_dummy_property_t{}.view(),
         edge_dst_dummy_property_t{}.view(),
         edge_dummy_property_t{}.view(),
         cuda::proclaim_return_type<bfs_edge_tuple_t>([d_sigma_2d = sigmas_2d.begin(), num_vertices, vertex_partition] 
         __device__(auto tagged_src, auto dst, auto, auto, auto) {
           auto src = thrust::get<0>(tagged_src);
           auto origin = thrust::get<1>(tagged_src);
           auto src_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(src);
           auto src_idx = origin * num_vertices + src_offset;
           auto src_sigma = static_cast<edge_t>(d_sigma_2d[src_idx]);
           
           return thrust::make_tuple(dst, origin, src_sigma);
         }),
         // PREDICATE: only process edges to unvisited vertices
         cuda::proclaim_return_type<bool>([d_distances_2d = distances_2d.begin(), num_vertices, vertex_partition, invalid_distance] 
         __device__(auto tagged_src, auto dst, auto, auto, auto) {
           auto origin = thrust::get<1>(tagged_src);
           auto dst_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(dst);
           auto dst_idx = origin * num_vertices + dst_offset;
           return d_distances_2d[dst_idx] == invalid_distance;
         }));
     
     // Step 2: Convert buffers to device vectors and extract components
     auto new_frontier_tagged_vertex_buffer = std::move(std::get<0>(result));
     
     rmm::device_uvector<bfs_edge_tuple_t> frontier_tuples(
       size_dataframe_buffer(new_frontier_tagged_vertex_buffer), handle.get_stream());
     
     thrust::copy(handle.get_thrust_policy(),
                  get_dataframe_buffer_begin(new_frontier_tagged_vertex_buffer),
                  get_dataframe_buffer_end(new_frontier_tagged_vertex_buffer),
                  frontier_tuples.begin());
     
     // Extract (vertex, origin) pairs and sigmas for sorting and reduction
     rmm::device_uvector<vertex_t> frontier_vertices(frontier_tuples.size(), handle.get_stream());
     rmm::device_uvector<origin_t> frontier_origins(frontier_tuples.size(), handle.get_stream());
     rmm::device_uvector<edge_t> sigmas(frontier_tuples.size(), handle.get_stream());
     
     thrust::transform(handle.get_thrust_policy(),
                       frontier_tuples.begin(),
                       frontier_tuples.end(),
                       thrust::make_zip_iterator(frontier_vertices.begin(), frontier_origins.begin(), sigmas.begin()),
                       [] __device__(auto tuple) {
                         return thrust::make_tuple(thrust::get<0>(tuple), thrust::get<1>(tuple), thrust::get<2>(tuple));
                       });
     
     // Step 3: Reduce by (destination, origin) - sums sigmas for multiple paths
     // Sort by (destination, origin) pairs
     thrust::sort_by_key(handle.get_thrust_policy(),
                         thrust::make_zip_iterator(frontier_vertices.begin(), frontier_origins.begin()),
                         thrust::make_zip_iterator(frontier_vertices.end(), frontier_origins.end()),
                         sigmas.begin());
     
     // Reduce by key to sum sigmas for identical (destination, origin) pairs
     auto num_unique = thrust::count_if(
       handle.get_thrust_policy(),
       thrust::make_counting_iterator(size_t{0}),
       thrust::make_counting_iterator(frontier_vertices.size()),
       [vertices = frontier_vertices.data(), origins = frontier_origins.data()] __device__(size_t i) {
         return (i == 0) || 
                (vertices[i] != vertices[i - 1]) || 
                (origins[i] != origins[i - 1]);
       });
     
     rmm::device_uvector<vertex_t> unique_vertices(num_unique, handle.get_stream());
     rmm::device_uvector<origin_t> unique_origins(num_unique, handle.get_stream());
     rmm::device_uvector<edge_t> unique_sigmas(num_unique, handle.get_stream());
     
     thrust::reduce_by_key(handle.get_thrust_policy(),
                           thrust::make_zip_iterator(frontier_vertices.begin(), frontier_origins.begin()),
                           thrust::make_zip_iterator(frontier_vertices.end(), frontier_origins.end()),
                           sigmas.begin(),
                           thrust::make_zip_iterator(unique_vertices.begin(), unique_origins.begin()),
                           unique_sigmas.begin(),
                           thrust::equal_to<thrust::tuple<vertex_t, origin_t>>{},
                           thrust::plus<edge_t>{});
     
     // Step 4: Manual array updates (all vertices in unique_vertices are unvisited due to predicate)
     thrust::for_each(
       handle.get_thrust_policy(),
       thrust::make_zip_iterator(unique_vertices.begin(), unique_origins.begin(), unique_sigmas.begin()),
       thrust::make_zip_iterator(unique_vertices.end(), unique_origins.end(), unique_sigmas.end()),
       [d_distances_2d = distances_2d.begin(), d_sigmas_2d = sigmas_2d.begin(),
        num_vertices, hop, vertex_partition] __device__(auto tuple) {
         auto v = thrust::get<0>(tuple);
         auto origin = thrust::get<1>(tuple);
         auto sigma = thrust::get<2>(tuple);
         auto offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
         auto idx = origin * num_vertices + offset;
         
         // Direct assignment - no atomics needed because reduction already handled duplicates
         d_distances_2d[idx] = hop + 1;
         d_sigmas_2d[idx] = sigma;
       });
     
     // Step 5: Update frontier for next iteration (all vertices in unique_vertices are newly discovered)
     vertex_frontier.bucket(bucket_idx_cur).clear();
     vertex_frontier.bucket(bucket_idx_next).insert(
       thrust::make_zip_iterator(unique_vertices.begin(), unique_origins.begin()),
       thrust::make_zip_iterator(unique_vertices.end(), unique_origins.end()));
     
     vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
     ++hop;
   }

        // Debug: Print final BFS results for first few sources
     printf("DEBUG: BFS completed in %d hops\n", hop);
     
     // Copy first few distance values to host for printing
     for (size_t s = 0; s < std::min(size_t{3}, num_sources); ++s) {
       std::vector<vertex_t> host_distances(std::min(size_t{5}, static_cast<size_t>(num_vertices)));
       raft::update_host(host_distances.data(), 
                        distances_2d.data() + s * num_vertices, 
                        host_distances.size(), 
                        handle.get_stream());
       handle.sync_stream();
       
       printf("DEBUG: Source %zu distances: ", s);
       for (size_t v = 0; v < host_distances.size(); ++v) {
         printf("%ld ", host_distances[v]);
       }
       printf("...\n");
     }

   return std::make_tuple(std::move(distances_2d), std::move(sigmas_2d));
 }

// Add a device functor for the backward pass

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
struct multisource_backward_pass_functor {
  vertex_t const* distances_2d;
  edge_t const* sigmas_2d;
  weight_t* centralities; // Final centrality array
  weight_t* dependency_buffer; // Buffer for dependency calculations [num_vertices]
  edge_t const* offsets;
  vertex_t const* indices;
  vertex_t const* sources; // Array of source vertices
  std::optional<detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>> edge_mask_view;
  size_t num_vertices;
  size_t num_sources;
  size_t batch_start;  // Add batch offset
  bool with_endpoints;
  bool do_expensive_check;

    __device__ void operator()(size_t source_idx) const {
    constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
    const vertex_t* distances = distances_2d + source_idx * num_vertices;
    const edge_t* sigmas = sigmas_2d + source_idx * num_vertices;
    
    // Get dependency buffer for this thread
    weight_t* dependency = dependency_buffer + source_idx * num_vertices;
    
    // Get the source vertex for this source index
    vertex_t source_vertex = sources[source_idx];
    
    // Debug: Print thread info (only for first few sources to avoid spam)
    if (source_idx < 3) {
      printf("DEBUG: Thread %lu (block %lu, thread %lu) processing source_idx %lu -> source_vertex %ld\n", 
             threadIdx.x + blockIdx.x * blockDim.x, blockIdx.x, threadIdx.x, source_idx, source_vertex);
      printf("DEBUG: Source_idx %lu: array_index=%lu\n", 
             source_idx, source_idx);
      
      // Debug: Print first few sources array values
      printf("DEBUG: Source_idx %lu sources array: ", source_idx);
      for (size_t i = 0; i < std::min(size_t{5}, num_sources); ++i) {
        printf("%ld ", sources[i]);
      }
      printf("\n");
    }
    
    // Debug: Print thread start time for parallelization verification
    if (source_idx < 3) {
      printf("DEBUG: Thread %lu (source %lu) START at time: %lu\n", 
             threadIdx.x + blockIdx.x * blockDim.x, source_idx, clock64());
    }

    // 1. Find maximum distance for this source
    vertex_t max_distance_from_source = 0;
    for (size_t i = 0; i < num_vertices; ++i) {
      if (distances[i] != invalid_distance && distances[i] > max_distance_from_source) {
        max_distance_from_source = distances[i];
      }
      dependency[i] = 0; // initialize
    }
    
        // Debug: Print max distance for first few sources
    if (source_idx < 3) {
      printf("DEBUG: Source %lu (vertex %ld) has max distance %ld\n", source_idx, source_vertex, max_distance_from_source);
      
      // Debug: Print first few distance values for this source
      printf("DEBUG: Source %lu first 5 distances: ", source_idx);
      for (size_t i = 0; i < std::min(size_t{5}, static_cast<size_t>(num_vertices)); ++i) {
        printf("%ld ", distances[i]);
      }
      printf("\n");
      
      // Debug: Check if all distances are invalid
      size_t valid_distances = 0;
      for (size_t i = 0; i < num_vertices; ++i) {
        if (distances[i] != invalid_distance) {
          valid_distances++;
        }
      }
      printf("DEBUG: Source %lu has %lu valid distances out of %lu vertices\n", source_idx, valid_distances, num_vertices);
    }

    // 2. Backward pass: process vertices by distance level
    for (vertex_t d = max_distance_from_source; d > 0; --d) {
      // Process all vertices at distance d
      for (vertex_t v = 0; v < num_vertices; ++v) {
        if (distances[v] == d) {
          weight_t delta = 0;
          
          // For all neighbors w of v
          edge_t row_start = offsets[v];
          edge_t row_end = offsets[v + 1];
          for (edge_t e = row_start; e < row_end; ++e) {
            // Check edge mask if it exists
            if (edge_mask_view && !edge_mask_view->get(e)) continue; // Skip masked edges
            
            vertex_t w = indices[e];
            if (distances[w] == d + 1 && sigmas[w] > 0) {
              delta += (static_cast<weight_t>(sigmas[v]) / static_cast<weight_t>(sigmas[w])) * (1.0 + dependency[w]);
            }
          }
          dependency[v] += delta;
        }
      }
    }
    
    // 3. Handle endpoint contributions for non-source vertices
    for (vertex_t v = 0; v < num_vertices; ++v) {
      if (v != source_vertex && distances[v] != invalid_distance) {
        weight_t contribution = dependency[v];
        if (with_endpoints) {
          contribution += 1.0; // Add endpoint contribution
        }
        atomicAdd(&centralities[v], contribution);
      }
    }
    
    // 4. Handle source vertex contribution for include_endpoints
    if (with_endpoints) {
      weight_t source_contribution = 0;
      for (vertex_t v = 0; v < num_vertices; ++v) {
        if (v != source_vertex && distances[v] != invalid_distance) {
          source_contribution += 1.0; // Count reachable vertices from source
        }
      }
      atomicAdd(&centralities[source_vertex], source_contribution);
    }
    
    // Debug: Print completion for first few sources
    if (source_idx < 3) {
      printf("DEBUG: Source %lu (vertex %ld) processing completed\n", source_idx, source_vertex);
    }
    
    // Debug: Print thread end time for parallelization verification
    if (source_idx < 3) {
      printf("DEBUG: Thread %lu (source %lu) END at time: %lu\n", 
             threadIdx.x + blockIdx.x * blockDim.x, source_idx, clock64());
    }
  }
};

// Parallelized backward pass

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void multisource_backward_pass(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<weight_t> centralities,
  rmm::device_uvector<vertex_t>&& distances_2d,
  rmm::device_uvector<edge_t>&& sigmas_2d,
  raft::device_span<vertex_t const> sources,
  bool include_endpoints,
  bool do_expensive_check)
{
  auto num_vertices = static_cast<size_t>(graph_view.local_vertex_partition_range_size());
  auto num_sources = sources.size();
  
  printf("DEBUG: Starting multisource backward pass with %zu sources, %zu vertices\n", num_sources, num_vertices);
  
  // Get graph data for device access
  auto offsets = graph_view.local_edge_partition_offsets(0);
  auto indices = graph_view.local_edge_partition_indices(0);
  
  // Get edge mask if it exists
  std::optional<detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>> edge_mask_view_opt;
  if (graph_view.has_edge_mask()) {
    auto edge_mask_view = graph_view.edge_mask_view();
    if (edge_mask_view.has_value()) {
      edge_mask_view_opt = detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>(
        edge_mask_view.value(), 0);
    }
  }
  
  // Initialize centrality array to zero
  thrust::fill(handle.get_thrust_policy(), centralities.begin(), centralities.end(), weight_t{0});
  
  // Process all sources at once (temporarily remove batching)
  printf("DEBUG: Processing all %zu sources at once\n", num_sources);
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Allocate dependency buffer for all sources
  size_t dependency_buffer_size = num_sources * num_vertices * sizeof(weight_t);
  rmm::device_uvector<weight_t> dependency_buffer(num_sources * num_vertices, handle.get_stream());
  
  printf("DEBUG: Allocated dependency buffer: %zu bytes (%.2f MB)\n", 
         dependency_buffer_size, dependency_buffer_size / (1024.0 * 1024.0));
  
  printf("DEBUG: Launching parallel for_each with %zu threads\n", num_sources);
  
  // Debug: Check pointer validity
  printf("DEBUG: distances_2d.data() = %p\n", static_cast<const void*>(distances_2d.data()));
  printf("DEBUG: sigmas_2d.data() = %p\n", static_cast<const void*>(sigmas_2d.data()));
  printf("DEBUG: sources.data() = %p\n", static_cast<const void*>(sources.data()));
  
  // Debug: Print first few source vertices
  printf("DEBUG: First 5 source vertices: ");
  // Copy first few sources to host for printing
  std::vector<vertex_t> host_sources(std::min(size_t{5}, sources.size()));
  raft::update_host(host_sources.data(), sources.data(), host_sources.size(), handle.get_stream());
  handle.sync_stream();
  
  for (size_t i = 0; i < host_sources.size(); ++i) {
    printf("%ld ", host_sources[i]);
  }
  printf("\n");
  
  // Launch parallel backward pass for all sources
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_sources),
    multisource_backward_pass_functor<vertex_t, edge_t, weight_t, multi_gpu>{
      distances_2d.data(),
      sigmas_2d.data(),
      centralities.data(),
      dependency_buffer.data(),
      offsets.data(),
      indices.data(),
      static_cast<vertex_t const*>(sources.data()),  // Cast to proper device pointer
      edge_mask_view_opt,
      num_vertices,
      num_sources,
      0,  // batch_start = 0 for all sources
      include_endpoints,
      do_expensive_check
    });
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  
  printf("DEBUG: Multisource backward pass completed in %.3f ms\n", total_duration.count() / 1000.0);
  
  printf("DEBUG: Multisource backward pass completed:\n");
  printf("  - Total time: %.3f ms\n", total_duration.count() / 1000.0);
  printf("  - Total memory allocated: %zu bytes (%.2f MB)\n", 
         dependency_buffer_size, dependency_buffer_size / (1024.0 * 1024.0));
  printf("  - Processed %zu sources\n", num_sources);
}

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