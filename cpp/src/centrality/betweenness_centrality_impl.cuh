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
    weight_t) const
  {
    return thrust::make_tuple(src, dst);
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
                             weight_t) const
  {
    return ((thrust::get<0>(dst_props) == d) && (thrust::get<0>(src_props) == (d - 1)));
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

  edge_src_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> src_sigmas(
    handle, graph_view);
  edge_dst_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, vertex_t> dst_distances(
    handle, graph_view);

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

  rmm::device_uvector<weight_t> deltas(sigmas.size(), handle.get_stream());
  detail::scalar_fill(handle, deltas.data(), deltas.size(), weight_t{0});

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

  edge_src_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                      thrust::tuple<vertex_t, edge_t, weight_t>>
    src_properties(handle, graph_view);
  edge_dst_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                      thrust::tuple<vertex_t, edge_t, weight_t>>
    dst_properties(handle, graph_view);

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

  // FIXME: To do this efficiently, I need a version of
  //   per_v_transform_reduce_outgoing_e that takes a vertex list
  //   so that we can iterate over the frontier stack.
  //
  //   For now this will do a O(E) pass over all edges over the diameter
  //   of the graph.
  //
  // Based on Brandes algorithm, we want to follow back pointers in non-increasing
  // distance from S to compute delta
  //
  for (vertex_t d = diameter; d > 1; --d) {
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

    thrust::transform(handle.get_thrust_policy(),
                      centralities.begin(),
                      centralities.end(),
                      deltas.begin(),
                      centralities.begin(),
                      thrust::plus<weight_t>());
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

  edge_src_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                      thrust::tuple<vertex_t, edge_t, weight_t>>
    src_properties(handle, graph_view);
  edge_dst_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                      thrust::tuple<vertex_t, edge_t, weight_t>>
    dst_properties(handle, graph_view);

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
    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edge_list(handle);

    {
      auto [src, dst] = extract_transform_if_e(handle,
                                               graph_view,
                                               src_properties.view(),
                                               dst_properties.view(),
                                               centralities_view,
                                               extract_edge_e_op_t<vertex_t>{},
                                               extract_edge_pred_op_t<vertex_t>{d},
                                               do_expensive_check);

      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(src.begin(), dst.begin()),
                   thrust::make_zip_iterator(src.end(), dst.end()));

      // Eliminate duplicates in case of a multi-graph
      auto new_edgelist_end = thrust::unique(handle.get_thrust_policy(),
                                             thrust::make_zip_iterator(src.begin(), dst.begin()),
                                             thrust::make_zip_iterator(src.end(), dst.end()));

      src.resize(
        cuda::std::distance(thrust::make_zip_iterator(src.begin(), dst.begin()), new_edgelist_end),
        handle.get_stream());
      dst.resize(src.size(), handle.get_stream());

      edge_list.insert(src.begin(), src.end(), dst.begin());
    }

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
    accumulate_vertex_results(handle,
                              graph_view,
                              edge_weight_view,
                              raft::device_span<weight_t>{centralities.data(), centralities.size()},
                              std::move(distances),
                              std::move(sigmas),
                              include_endpoints,
                              do_expensive_check);
  }

  std::optional<weight_t> scale_factor{std::nullopt};

  if (normalized) {
    if (include_endpoints) {
      if (graph_view.number_of_vertices() >= 2) {
        scale_factor = static_cast<weight_t>(
          std::min(static_cast<vertex_t>(num_sources), graph_view.number_of_vertices()) *
          (graph_view.number_of_vertices() - 1));
      }
    } else if (graph_view.number_of_vertices() > 2) {
      scale_factor = static_cast<weight_t>(
        std::min(static_cast<vertex_t>(num_sources), graph_view.number_of_vertices() - 1) *
        (graph_view.number_of_vertices() - 2));
    }
  } else if (num_sources < static_cast<size_t>(graph_view.number_of_vertices())) {
    if ((graph_view.number_of_vertices() > 1) && (num_sources > 0))
      scale_factor =
        (graph_view.is_symmetric() ? weight_t{2} : weight_t{1}) *
        static_cast<weight_t>(num_sources) /
        (include_endpoints ? static_cast<weight_t>(graph_view.number_of_vertices())
                           : static_cast<weight_t>(graph_view.number_of_vertices() - 1));
  } else if (graph_view.is_symmetric()) {
    scale_factor = weight_t{2};
  }

  if (scale_factor) {
    thrust::transform(handle.get_thrust_policy(),
                      centralities.begin(),
                      centralities.end(),
                      centralities.begin(),
                      [sf = *scale_factor] __device__(auto centrality) { return centrality / sf; });
  }

  return centralities;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>
edge_betweenness_centrality(
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

  edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t> centralities(
    handle, graph_view);

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
edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>
edge_betweenness_centrality(
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
