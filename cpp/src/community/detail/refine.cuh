/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <community/detail/common_methods.hpp>

#include <community/detail/mis.hpp>

#include <detail/graph_utils.cuh>
#include <prims/per_v_transform_reduce_dst_key_aggregated_outgoing_e.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_e.cuh>
#include <prims/transform_reduce_e_by_src_dst_key.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <utilities/collect_comm.cuh>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)

namespace cugraph {
namespace detail {

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct filter_negative_gain_moves {
  __device__ auto operator()(thrust::tuple<vertex_t, vertex_t, weight_t> src_dst_weight) const
  {
    return (thrust::get<2>(src_dst_weight)) > 0.0;
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct reduce_op_t {
  using type                          = thrust::tuple<vertex_t, weight_t>;
  static constexpr bool pure_function = true;  // this can be called from any process
  // inline static type const identity_element =
  //   thrust::make_tuple(std::numeric_limits<weight_t>::lowest(),
  //   invalid_vertex_id<vertex_t>::value);

  __device__ auto operator()(thrust::tuple<vertex_t, weight_t> p0,
                             thrust::tuple<vertex_t, weight_t> p1) const
  {
    auto id0 = thrust::get<0>(p0);
    auto id1 = thrust::get<0>(p1);
    auto wt0 = thrust::get<1>(p0);
    auto wt1 = thrust::get<1>(p1);

    return (wt0 < wt1) ? p1 : ((wt0 > wt1) ? p0 : ((id0 < id1) ? p0 : p1));
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t, typename cluster_value_t>
struct leiden_key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t resolution{};
  __device__ auto operator()(
    vertex_t src,
    vertex_t neighboring_leiden_cluster,
    thrust::tuple<weight_t, weight_t, weight_t, uint8_t, vertex_t, vertex_t> src_info,
    cluster_value_t keyed_data,
    weight_t aggregated_weight_to_neighboring_leiden_cluster) const
  {
    //
    // Data associated with src vertex
    //
    auto src_weighted_deg           = thrust::get<0>(src_info);
    auto src_vertex_cut             = thrust::get<1>(src_info);
    auto src_louvain_cluster_volume = thrust::get<2>(src_info);
    auto src_singleton_flag         = thrust::get<3>(src_info);
    auto src_leiden_cluster         = thrust::get<4>(src_info);
    auto src_louvain_cluster        = thrust::get<5>(src_info);

    //
    // Data associated with target leiden (aka refined) community
    //

    auto dst_refined_cluster_volume   = thrust::get<0>(keyed_data);
    auto dst_refined_cluster_cut      = thrust::get<1>(keyed_data);
    auto leiden_of_refined_community  = thrust::get<2>(keyed_data);
    auto louvain_of_refined_community = thrust::get<3>(keyed_data);

    // neighboring_leiden_cluster and leiden_of_refined_cluster must have same value

    // E(v, S-v) > ||v||*(||S|| -||v||)
    bool is_src_well_connected =
      src_vertex_cut > src_weighted_deg * (src_louvain_cluster_volume - src_weighted_deg);

    // E(Cr, S-Cr) > ||Cr||*(||S|| -||Cr||)
    bool is_refined_cluster_well_connected =
      dst_refined_cluster_cut >
      dst_refined_cluster_volume * (src_louvain_cluster_volume - dst_refined_cluster_volume);

    // E(v, Cr-v) - ||v||* ||Cr-v||/||V(G)||
    // aggregated_weight_to_neighboring_leiden_cluster == E(v, Cr-v)?

    weight_t theta = -1.0;

    if (src_singleton_flag && is_src_well_connected) {
      if (louvain_of_refined_community == src_louvain_cluster &&
          is_refined_cluster_well_connected) {
        theta = aggregated_weight_to_neighboring_leiden_cluster -
                src_weighted_deg * dst_refined_cluster_volume / total_edge_weight;
      }
    }

    return thrust::make_tuple(neighboring_leiden_cluster, theta);
  }
};

//
// Construct Leiden to Louvain mapping
//
template <typename vertex_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
build_leiden_to_louvain_map(raft::handle_t const& handle,
                            rmm::device_uvector<vertex_t>& leiden_assignment,
                            rmm::device_uvector<vertex_t>& louvain_assignment)
{
  rmm::device_uvector<vertex_t> keys_of_leiden_to_louvain_map(leiden_assignment.size(),
                                                              handle.get_stream());
  rmm::device_uvector<vertex_t> values_of_leiden_to_louvain_map(louvain_assignment.size(),
                                                                handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               leiden_assignment.begin(),
               leiden_assignment.end(),
               keys_of_leiden_to_louvain_map.begin());

  thrust::copy(handle.get_thrust_policy(),
               louvain_assignment.begin(),
               louvain_assignment.end(),
               values_of_leiden_to_louvain_map.begin());

  auto louvain_leiden_zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(
    values_of_leiden_to_louvain_map.begin(), keys_of_leiden_to_louvain_map.begin()));

  auto louvain_leiden_zipped_end = thrust::make_zip_iterator(
    thrust::make_tuple(values_of_leiden_to_louvain_map.end(), keys_of_leiden_to_louvain_map.end()));

  thrust::sort(handle.get_thrust_policy(),
               louvain_leiden_zipped_begin,
               louvain_leiden_zipped_end,
               thrust::less<thrust::tuple<vertex_t, vertex_t>>());

  auto last_unique_louvain_leiden_pair =
    thrust::unique(handle.get_thrust_policy(),
                   louvain_leiden_zipped_begin,
                   louvain_leiden_zipped_end,
                   thrust::less<thrust::tuple<vertex_t, vertex_t>>());

  auto nr_unique_louvain_leiden_pairs = static_cast<size_t>(
    thrust::distance(louvain_leiden_zipped_begin, last_unique_louvain_leiden_pair));

  keys_of_leiden_to_louvain_map.resize(nr_unique_louvain_leiden_pairs, handle.get_stream());
  values_of_leiden_to_louvain_map.resize(nr_unique_louvain_leiden_pairs, handle.get_stream());

  return std::make_tuple(std::move(keys_of_leiden_to_louvain_map),
                         std::move(values_of_leiden_to_louvain_map));
}

//
// Collect Louvain cluster ids for Leiden (aka refined) clusters
//

template <typename vertex_t>
rmm::device_uvector<vertex_t> select_a_radom_set_of_vetices(raft::handle_t const& handle,
                                                            vertex_t begin,
                                                            vertex_t end,
                                                            vertex_t count,
                                                            uint64_t seed,
                                                            int repetitions_per_vertex = 0)
{
#if 0
  auto& comm                  = handle.get_comms();
  auto const comm_rank        = comm.get_rank();
#endif
  vertex_t number_of_vertices = end - begin;

  rmm::device_uvector<vertex_t> vertices(
    std::max((repetitions_per_vertex + 1) * number_of_vertices, count), handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    vertices.begin(),
    vertices.end(),
    [begin, number_of_vertices] __device__(auto v) { return begin + (v % number_of_vertices); });
  thrust::default_random_engine g;
  g.seed(seed);
  thrust::shuffle(handle.get_thrust_policy(), vertices.begin(), vertices.end(), g);
  vertices.resize(count, handle.get_stream());
  return vertices;
}

template <typename GraphViewType, typename weight_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::pair<rmm::device_uvector<typename GraphViewType::vertex_type>,
                     rmm::device_uvector<typename GraphViewType::vertex_type>>>
refine_clustering(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
    edge_weight_view,
  weight_t total_edge_weight,
  weight_t resolution,
  rmm::device_uvector<weight_t> const& vertex_weights_v,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& cluster_keys_v,
  rmm::device_uvector<weight_t>&& cluster_weights_v,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& louvain_assignment,
  edge_src_property_t<GraphViewType, weight_t> const& src_vertex_weights_cache,
  edge_src_property_t<GraphViewType, typename GraphViewType::vertex_type> const&
    src_louvain_assignment,
  edge_dst_property_t<GraphViewType, typename GraphViewType::vertex_type> const&
    dst_louvain_assignment,
  bool up_down)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  rmm::device_uvector<weight_t> vertex_cluster_weights_v =
    lookup_primitive_values_for_keys<vertex_t, weight_t, GraphViewType::is_multi_gpu>(
      handle, cluster_keys_v, cluster_weights_v, louvain_assignment);

  edge_src_property_t<GraphViewType, weight_t> src_louvain_cluster_weights(handle);

  if (GraphViewType::is_multi_gpu) {
    src_louvain_cluster_weights = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, vertex_cluster_weights_v.begin(), src_louvain_cluster_weights);
    vertex_cluster_weights_v.resize(0, handle.get_stream());
    vertex_cluster_weights_v.shrink_to_fit(handle.get_stream());
  }

  //
  // For each vertex, compute its weighted degree (||v||)
  // and cut between itself and its Louvain community (E(v, S-v))
  //

  rmm::device_uvector<weight_t> weighted_degree_of_vertices(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  rmm::device_uvector<weight_t> weighted_cut_of_vertices(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  per_v_transform_reduce_outgoing_e(
    handle,
    graph_view,
    GraphViewType::is_multi_gpu
      ? src_louvain_assignment.view()
      : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(louvain_assignment.data()),
    GraphViewType::is_multi_gpu ? dst_louvain_assignment.view()
                                : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                    louvain_assignment.data(), vertex_t{0}),
    *edge_weight_view,
    [] __device__(auto src, auto dst, auto src_cluster, auto dst_cluster, auto wt) {
      weight_t weighted_deg_contribution{wt};
      weight_t weighted_cut_contribution{0};

      if (src == dst)  // self loop
        weighted_cut_contribution = 0;
      else if (src_cluster == dst_cluster)
        weighted_cut_contribution = wt;

      return thrust::make_tuple(weighted_deg_contribution, weighted_cut_contribution);
    },
    thrust::make_tuple(weight_t{0}, weight_t{0}),
    thrust::make_zip_iterator(
      thrust::make_tuple(weighted_degree_of_vertices.begin(), weighted_cut_of_vertices.begin())));

  //
  // Update weighted degree and cut for edge sources
  //

  edge_src_property_t<GraphViewType, thrust::tuple<weight_t, weight_t>>
    src_weighted_deg_and_cut_pair(handle);

  if constexpr (GraphViewType::is_multi_gpu) {
    src_weighted_deg_and_cut_pair =
      edge_src_property_t<GraphViewType, thrust::tuple<weight_t, weight_t>>(handle, graph_view);
    update_edge_src_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(
        thrust::make_tuple(weighted_degree_of_vertices.begin(), weighted_cut_of_vertices.begin())),
      src_weighted_deg_and_cut_pair);

    weighted_degree_of_vertices.resize(0, handle.get_stream());
    weighted_degree_of_vertices.shrink_to_fit(handle.get_stream());
    weighted_cut_of_vertices.resize(0, handle.get_stream());
    weighted_cut_of_vertices.shrink_to_fit(handle.get_stream());
  }

  //
  // Assign Lieden community Id for vertices.
  // Each vertex starts as a singleton community in the leiden partition
  //

  rmm::device_uvector<vertex_t> leiden_assignment = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  detail::sequence_fill(handle.get_stream(),
                        leiden_assignment.begin(),
                        leiden_assignment.size(),
                        graph_view.local_vertex_partition_range_first());

  //
  // Update Leiden assignment to edge sources and destinitions
  //

  edge_src_property_t<GraphViewType, vertex_t> src_leiden_assignment(handle);
  edge_dst_property_t<GraphViewType, vertex_t> dst_leiden_assignment(handle);

  if constexpr (GraphViewType::is_multi_gpu) {
    src_leiden_assignment = edge_src_property_t<GraphViewType, vertex_t>(handle, graph_view);
    update_edge_src_property(handle, graph_view, leiden_assignment.begin(), src_leiden_assignment);

    dst_leiden_assignment = edge_dst_property_t<GraphViewType, vertex_t>(handle, graph_view);
    update_edge_dst_property(handle, graph_view, leiden_assignment.begin(), dst_leiden_assignment);

    // Couldn't we clear leiden_assignment here?
  }

  //
  // TODO: Compute the following
  //

  // E(v, leiden(v)-v) - ||v||* ||leiden(v)-v||/||V(G)||

  //
  // Update Louvain and Leiden for edge destinitions
  //

  edge_dst_property_t<GraphViewType, thrust::tuple<vertex_t, vertex_t>> dst_louvain_leiden_cache(
    handle);

  if constexpr (GraphViewType::is_multi_gpu) {
    dst_louvain_leiden_cache =
      edge_dst_property_t<GraphViewType, thrust::tuple<vertex_t, vertex_t>>(handle, graph_view);

    update_edge_dst_property(handle,
                             graph_view,
                             thrust::make_zip_iterator(thrust::make_tuple(
                               louvain_assignment.cbegin(), leiden_assignment.cbegin())),
                             dst_louvain_leiden_cache);
  }

  //
  // Generate and shuffle Leiden cluster keys to aggregate volumes and cuts
  //

  rmm::device_uvector<vertex_t> leiden_cluster_keys = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());  //#V

  detail::sequence_fill(handle.get_stream(),
                        leiden_cluster_keys.begin(),
                        leiden_cluster_keys.size(),
                        graph_view.local_vertex_partition_range_first());

  if constexpr (GraphViewType::is_multi_gpu) {
    leiden_cluster_keys = shuffle_ext_vertices_by_gpu_id(handle, std::move(leiden_cluster_keys));
  }

  auto src_input_property_values =
    GraphViewType::is_multi_gpu
      ? view_concat(src_louvain_assignment.view(), src_leiden_assignment.view())
      : view_concat(
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(louvain_assignment.data()),
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(leiden_assignment.data()));

  auto dst_input_property_values =
    GraphViewType::is_multi_gpu
      ? dst_louvain_leiden_cache.view()
      : view_concat(detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                      louvain_assignment.data(), vertex_t{0}),
                    detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                      leiden_assignment.data(), vertex_t{0}));

  rmm::device_uvector<vertex_t> edge_reduced_output_keys(0, handle.get_stream());
  rmm::device_uvector<weight_t> refined_community_volumes(0, handle.get_stream());
  rmm::device_uvector<weight_t> refined_community_cuts(0, handle.get_stream());

  //
  // For each refined community, compute its volume
  // (i.e.sum of weighted degree of all vertices inside it, ||C||) and
  // and cut between itself and its Louvain community (E(C, S-C))
  //

  std::forward_as_tuple(edge_reduced_output_keys,
                        std::tie(refined_community_volumes, refined_community_cuts)) =
    cugraph::transform_reduce_e_by_dst_key(
      handle,
      graph_view,
      src_input_property_values,
      dst_input_property_values,
      *edge_weight_view,
      GraphViewType::is_multi_gpu ? dst_leiden_assignment.view()
                                  : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                      leiden_assignment.data(), vertex_t{0}),

      [] __device__(auto src,
                    auto dst,
                    thrust::tuple<vertex_t, vertex_t> src_louvain_leidn,
                    thrust::tuple<vertex_t, vertex_t> dst_louvain_leiden,
                    auto wt) {
        weight_t refined_partition_volume_contribution{0};
        weight_t refined_partition_cut_contribution{0};

        auto src_louvain = thrust::get<0>(src_louvain_leidn);
        auto src_leiden  = thrust::get<1>(src_louvain_leidn);

        auto dst_louvain = thrust::get<0>(dst_louvain_leiden);
        auto dst_leiden  = thrust::get<1>(dst_louvain_leiden);

        if (src == dst) {
          refined_partition_volume_contribution = wt;
        } else if (src_louvain == dst_louvain) {
          if (src_leiden == dst_leiden) {
            refined_partition_volume_contribution = wt;
          } else {
            refined_partition_cut_contribution = wt;
          }
        }
        return thrust::make_tuple(refined_partition_volume_contribution,
                                  refined_partition_cut_contribution);
      },
      thrust::make_tuple(weight_t{0}, weight_t{0}),
      reduce_op::plus<thrust::tuple<weight_t, weight_t>>{});

  //
  // Mask to indicate if a vertex is singleton
  //

  rmm::device_uvector<uint8_t> singleton_mask(leiden_assignment.size(), handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), singleton_mask.begin(), singleton_mask.end(), uint8_t{0});
  edge_src_property_t<GraphViewType, uint8_t> src_singleton_mask(handle);

  if constexpr (GraphViewType::is_multi_gpu) {
    src_singleton_mask = edge_src_property_t<GraphViewType, uint8_t>(handle, graph_view);
    update_edge_src_property(handle, graph_view, singleton_mask.begin(), src_singleton_mask);
    // Couldn't we clear singleton_mask here?
  }

  //
  // Primitives to decide best (at least good) next clusters for vertices
  //

  // ||v||
  // E(v, louvain(v))
  // ||louvain(v)||
  // is_singleton(v)
  // leiden(v)
  // louvain(v)

  auto zipped_src_device_view =
    GraphViewType::is_multi_gpu
      ? view_concat(src_weighted_deg_and_cut_pair.view(),
                    src_louvain_cluster_weights.view(),
                    src_singleton_mask.view(),
                    src_leiden_assignment.view(),
                    src_louvain_assignment.view())
      : view_concat(
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(
            weighted_degree_of_vertices.data()),
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(
            weighted_cut_of_vertices.data()),
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(
            vertex_cluster_weights_v.data()),
          detail::edge_major_property_view_t<vertex_t, uint8_t const*>(singleton_mask.data()),
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(leiden_assignment.data()),
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(louvain_assignment.data()));

  auto [keys_of_leiden_to_louvain_map, values_of_leiden_to_louvain_map] =
    build_leiden_to_louvain_map<vertex_t, GraphViewType::is_multi_gpu>(
      handle, leiden_assignment, louvain_assignment);

  auto louvain_of_leiden_clusters_to_reduce_on =
    lookup_primitive_values_for_keys<vertex_t, vertex_t, GraphViewType::is_multi_gpu>(
      handle,
      keys_of_leiden_to_louvain_map,
      values_of_leiden_to_louvain_map,
      edge_reduced_output_keys);

  // ||Cr||   //f(Cr)
  // E(Cr, louvain(v) - Cr) //f(Cr)
  // leiden(Cr)  // f(Cr)
  // louvain(Cr)  // f(Cr)
  auto values_for_leiden_cluster_keys =
    thrust::make_zip_iterator(thrust::make_tuple(refined_community_volumes.begin(),
                                                 refined_community_cuts.begin(),
                                                 edge_reduced_output_keys.begin(),  // redundant
                                                 louvain_of_leiden_clusters_to_reduce_on.begin()));

  using value_t = thrust::tuple<weight_t, weight_t, vertex_t, vertex_t>;
  kv_store_t<vertex_t, value_t, true> leiden_cluster_key_values_map(
    edge_reduced_output_keys.begin(),
    edge_reduced_output_keys.begin() + edge_reduced_output_keys.size(),
    values_for_leiden_cluster_keys,
    thrust::make_tuple(std::numeric_limits<weight_t>::max(),
                       std::numeric_limits<weight_t>::max(),
                       invalid_vertex_id<vertex_t>::value,
                       invalid_vertex_id<vertex_t>::value),
    false,
    handle.get_stream());

  //
  // Decide best/positive move for each vertex
  //

  auto output_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, weight_t>>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  per_v_transform_reduce_dst_key_aggregated_outgoing_e(
    handle,
    graph_view,
    zipped_src_device_view,
    *edge_weight_view,
    GraphViewType::is_multi_gpu ? dst_leiden_assignment.view()
                                : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                    leiden_assignment.data(), vertex_t{0}),
    leiden_cluster_key_values_map.view(),
    detail::leiden_key_aggregated_edge_op_t<vertex_t, weight_t, value_t>{total_edge_weight,
                                                                         resolution},
    thrust::make_tuple(vertex_t{-1}, weight_t{0}),
    detail::reduce_op_t<vertex_t, weight_t>{},
    cugraph::get_dataframe_buffer_begin(output_buffer));

  //
  // Create graph from (community, target community, modulraity gain) tuple
  //

  size_t num_vertices = graph_view.local_vertex_partition_range_size();

  rmm::device_uvector<vertex_t> all_srcs(num_vertices, handle.get_stream());
  rmm::device_uvector<vertex_t> d_srcs(num_vertices, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dsts(num_vertices, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> d_weights =
    std::make_optional(rmm::device_uvector<weight_t>(num_vertices, handle.get_stream()));

  thrust::sequence(handle.get_thrust_policy(),
                   all_srcs.begin(),
                   all_srcs.end(),
                   graph_view.local_vertex_partition_range_first());

  auto d_src_dst_weight_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(d_srcs.begin(), d_dsts.begin(), (*d_weights).begin()));

  ///

  auto output_first = get_dataframe_buffer_cbegin(output_buffer);
  auto edge_begin   = thrust::make_zip_iterator(
    thrust::make_tuple(all_srcs.begin(),
                       thrust::get<0>(output_first.get_iterator_tuple()),
                       thrust::get<1>(output_first.get_iterator_tuple())));

  auto output_last = cugraph::get_dataframe_buffer_cend(output_buffer);
  auto edge_end =
    thrust::make_zip_iterator(thrust::make_tuple(all_srcs.end(),
                                                 thrust::get<0>(output_last.get_iterator_tuple()),
                                                 thrust::get<1>(output_last.get_iterator_tuple())));

  ///
  thrust::copy_if(handle.get_thrust_policy(),
                  edge_begin,
                  edge_end,
                  d_src_dst_weight_iterator,
                  filter_negative_gain_moves<vertex_t, weight_t>{});

  constexpr bool storage_transposed = false;
  constexpr bool multi_gpu          = GraphViewType::is_multi_gpu;

  cugraph::graph_t<vertex_t, edge_t, storage_transposed, multi_gpu> decision_graph(handle);

  // std::optional<rmm::device_uvector<vertex_t>> d_vertices =
  //   std::make_optional(rmm::device_uvector<vertex_t>(num_vertices, handle.get_stream()));

  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  using DecisionGraphViewType =
    cugraph::graph_view_t<vertex_t, edge_t, false, GraphViewType::is_multi_gpu>;

  std::optional<edge_property_t<DecisionGraphViewType, weight_t>> coarse_edge_weights{std::nullopt};

  std::tie(decision_graph, coarse_edge_weights, std::ignore, renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, storage_transposed, multi_gpu>(
      handle,
      std::nullopt,
      std::move(d_srcs),
      std::move(d_dsts),
      std::move(d_weights),
      std::nullopt,
      cugraph::graph_properties_t{false, false},
      multi_gpu ? true : false);

  auto decision_graph_view = decision_graph.view();

  //
  // Determine a set of moves using MIS of the decision_graph
  //

  auto chosen_vertices = compute_mis(
    handle,
    decision_graph_view,
    coarse_edge_weights ? std::make_optional(coarse_edge_weights->view()) : std::nullopt);

  // TODO

  // Apply Renumber map

  //
  // TODO: Make sure that the moves make sense?
  //

  //
  // Re-read Leiden to Louvain map, but for remaining (after moving) Leiden communities
  //
  rmm::device_uvector<vertex_t> keys_to_read_value_for(leiden_assignment.size(),
                                                       handle.get_stream());  //#C

  thrust::copy(handle.get_thrust_policy(),
               leiden_assignment.begin(),
               leiden_assignment.end(),
               keys_to_read_value_for.begin());

  thrust::sort(
    handle.get_thrust_policy(), keys_to_read_value_for.begin(), keys_to_read_value_for.end());
  auto nr_remaining_leiden_cluster = static_cast<size_t>(thrust::distance(
    keys_to_read_value_for.begin(),
    thrust::unique(
      handle.get_thrust_policy(), keys_to_read_value_for.begin(), keys_to_read_value_for.end())));

  if constexpr (GraphViewType::is_multi_gpu) {
    keys_to_read_value_for =
      shuffle_ext_vertices_and_values_by_gpu_id(handle, std::move(keys_to_read_value_for));

    thrust::sort(
      handle.get_thrust_policy(), keys_to_read_value_for.begin(), keys_to_read_value_for.end());

    nr_remaining_leiden_cluster = static_cast<size_t>(thrust::distance(
      keys_to_read_value_for.begin(),
      thrust::unique(
        handle.get_thrust_policy(), keys_to_read_value_for.begin(), keys_to_read_value_for.end())));
  }

  keys_to_read_value_for.resize(nr_remaining_leiden_cluster, handle.get_stream());

  auto lovain_of_leiden_cluster_keys =
    lookup_primitive_values_for_keys<vertex_t, vertex_t, GraphViewType::is_multi_gpu>(
      handle,
      keys_of_leiden_to_louvain_map,
      values_of_leiden_to_louvain_map,
      keys_to_read_value_for);

  return std::make_tuple(
    std::move(leiden_assignment),
    std::make_pair(std::move(keys_to_read_value_for), std::move(lovain_of_leiden_cluster_keys)));
  rmm::device_uvector<vertex_t> temp1 = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());
  rmm::device_uvector<vertex_t> temp2 = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  return std::make_tuple(std::move(leiden_assignment),
                         std::make_pair(std::move(temp1), std::move(temp2)));
}
}  // namespace detail
}  // namespace cugraph