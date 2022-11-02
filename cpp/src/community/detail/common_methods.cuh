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
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct compare_leiden_louvain_pair {
  __device__ auto operator()(thrust::tuple<vertex_t, vertex_t> a,
                             thrust::tuple<vertex_t, vertex_t> b)
  {
    auto leiden_a  = thrust::get<1>(a);
    auto louvain_a = thrust::get<0>(a);
    auto leiden_b  = thrust::get<1>(b);
    auto louvain_b = thrust::get<0>(b);
    return louvain_a < louvain_b ? true : (louvain_a > louvain_b ? false : leiden_a <= leiden_b);
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct leiden_key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t resolution{};
  __device__ auto operator()(
    vertex_t src,
    vertex_t neighboring_leiden_cluster,
    weight_t aggregated_weight_to_neighboring_leiden_cluster,
    thrust::tuple<weight_t, weight_t, weight_t, uint8_t, vertex_t, vertex_t> src_info,
    thrust::tuple<weight_t, weight_t, vertex_t, vertex_t> keyed_data) const
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

    // neighboring_leiden_cluster, leiden_of_refined_cluster

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

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t resolution{};
  __device__ auto operator()(
    vertex_t src,
    vertex_t neighbor_cluster,
    weight_t new_cluster_sum,
    thrust::tuple<weight_t, vertex_t, weight_t, weight_t, weight_t> src_info,
    weight_t a_new) const
  {
    auto k_k              = thrust::get<0>(src_info);
    auto src_cluster      = thrust::get<1>(src_info);
    auto a_old            = thrust::get<2>(src_info);
    auto old_cluster_sum  = thrust::get<3>(src_info);
    auto cluster_subtract = thrust::get<4>(src_info);

    if (src_cluster == neighbor_cluster) new_cluster_sum -= cluster_subtract;

    weight_t delta_modularity = 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                     resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                       (total_edge_weight * total_edge_weight));

    return thrust::make_tuple(neighbor_cluster, delta_modularity);
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct reduce_op_t {
  using type                          = thrust::tuple<vertex_t, weight_t>;
  static constexpr bool pure_function = true;  // this can be called from any process
  inline static type const identity_element =
    thrust::make_tuple(std::numeric_limits<weight_t>::lowest(), invalid_vertex_id<vertex_t>::value);

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
template <typename vertex_t, typename weight_t>
struct cluster_update_op_t {
  bool up_down{};
  __device__ auto operator()(vertex_t old_cluster, thrust::tuple<vertex_t, weight_t> p) const
  {
    vertex_t new_cluster      = thrust::get<0>(p);
    weight_t delta_modularity = thrust::get<1>(p);

    return (delta_modularity > weight_t{0})
             ? (((new_cluster > old_cluster) != up_down) ? old_cluster : new_cluster)
             : old_cluster;
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct return_edge_weight_t {
  __device__ auto operator()(
    vertex_t, vertex_t, weight_t w, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return w;
  }
};

template <typename graph_view_t>
typename graph_view_t::weight_type compute_modularity(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const& src_clusters_cache,
  edge_dst_property_t<graph_view_t, typename graph_view_t::vertex_type> const& dst_clusters_cache,
  rmm::device_uvector<typename graph_view_t::vertex_type> const& next_clusters,
  rmm::device_uvector<typename graph_view_t::weight_type> const& cluster_weights,
  typename graph_view_t::weight_type total_edge_weight,
  typename graph_view_t::weight_type resolution)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using weight_t = typename graph_view_t::weight_type;

  //
  // Sum(Sigma_tot_c^2), over all clusters c
  //
  weight_t sum_degree_squared = thrust::transform_reduce(
    handle.get_thrust_policy(),
    cluster_weights.begin(),
    cluster_weights.end(),
    [] __device__(weight_t p) { return p * p; },
    weight_t{0},
    thrust::plus<weight_t>());

  if constexpr (graph_view_t::is_multi_gpu) {
    sum_degree_squared = host_scalar_allreduce(
      handle.get_comms(), sum_degree_squared, raft::comms::op_t::SUM, handle.get_stream());
  }

  // Sum(Sigma_in_c), over all clusters c
  weight_t sum_internal = transform_reduce_e(
    handle,
    graph_view,
    graph_view_t::is_multi_gpu
      ? src_clusters_cache.view()
      : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(next_clusters.begin()),
    graph_view_t::is_multi_gpu ? dst_clusters_cache.view()
                               : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                   next_clusters.begin(), vertex_t{0}),
    [] __device__(auto, auto, weight_t wt, auto src_cluster, auto nbr_cluster) {
      if (src_cluster == nbr_cluster) {
        return wt;
      } else {
        return weight_t{0};
      }
    },
    weight_t{0});

  weight_t Q = sum_internal / total_edge_weight -
               (resolution * sum_degree_squared) / (total_edge_weight * total_edge_weight);

  return Q;
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t> labels)
{
  auto [new_graph, numbering_map] = coarsen_graph(handle, graph_view, labels.data(), true);

  auto new_graph_view = new_graph.view();

  rmm::device_uvector<vertex_t> numbering_indices((*numbering_map).size(), handle.get_stream());
  detail::sequence_fill(handle.get_stream(),
                        numbering_indices.data(),
                        numbering_indices.size(),
                        new_graph_view.local_vertex_partition_range_first());

  relabel<vertex_t, multi_gpu>(
    handle,
    std::make_tuple(static_cast<vertex_t const*>((*numbering_map).begin()),
                    static_cast<vertex_t const*>(numbering_indices.begin())),
    new_graph_view.local_vertex_partition_range_size(),
    labels.data(),
    labels.size(),
    false);

  return std::make_tuple(std::move(new_graph), std::move(numbering_map));
}

template <typename graph_view_t>
std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
           std::pair<rmm::device_uvector<typename graph_view_t::vertex_type>,
                     rmm::device_uvector<typename graph_view_t::vertex_type>>>
refine_clustering(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  typename graph_view_t::weight_type total_edge_weight,
  typename graph_view_t::weight_type resolution,
  rmm::device_uvector<typename graph_view_t::weight_type> const& vertex_weights_v,
  rmm::device_uvector<typename graph_view_t::vertex_type>&& cluster_keys_v,
  rmm::device_uvector<typename graph_view_t::weight_type>&& cluster_weights_v,
  rmm::device_uvector<typename graph_view_t::vertex_type>&& next_clusters_v,
  edge_src_property_t<graph_view_t, typename graph_view_t::weight_type> const&
    src_vertex_weights_cache,
  edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const& src_clusters_cache,
  edge_dst_property_t<graph_view_t, typename graph_view_t::vertex_type> const& dst_clusters_cache,
  bool up_down)
{
  // FIXME: put duplicated code in a function
  using vertex_t = typename graph_view_t::vertex_type;
  using weight_t = typename graph_view_t::weight_type;

  //--------------duplicated code
  rmm::device_uvector<weight_t> vertex_cluster_weights_v(0, handle.get_stream());
  edge_src_property_t<graph_view_t, weight_t> src_cluster_weights(handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      handle.get_comms().get_size()};

    vertex_cluster_weights_v =
      cugraph::collect_values_for_keys(handle.get_comms(),
                                       cluster_keys_v.begin(),
                                       cluster_keys_v.end(),
                                       cluster_weights_v.data(),
                                       next_clusters_v.begin(),
                                       next_clusters_v.end(),
                                       vertex_to_gpu_id_op,
                                       invalid_vertex_id<vertex_t>::value,
                                       std::numeric_limits<weight_t>::max(),
                                       handle.get_stream());

    src_cluster_weights = edge_src_property_t<graph_view_t, weight_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, vertex_cluster_weights_v.begin(), src_cluster_weights);
    vertex_cluster_weights_v.resize(0, handle.get_stream());
    vertex_cluster_weights_v.shrink_to_fit(handle.get_stream());
  } else {
    // sort so we can use lower_bound in the transform function
    thrust::sort_by_key(handle.get_thrust_policy(),
                        cluster_keys_v.begin(),
                        cluster_keys_v.end(),
                        cluster_weights_v.begin());

    // for each vertex, look up the vertex weight of the current cluster it is assigned to
    vertex_cluster_weights_v.resize(next_clusters_v.size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      next_clusters_v.begin(),
                      next_clusters_v.end(),
                      vertex_cluster_weights_v.begin(),
                      [d_cluster_weights = cluster_weights_v.data(),
                       d_cluster_keys    = cluster_keys_v.data(),
                       num_clusters      = cluster_keys_v.size()] __device__(vertex_t cluster) {
                        auto pos = thrust::lower_bound(
                          thrust::seq, d_cluster_keys, d_cluster_keys + num_clusters, cluster);
                        return d_cluster_weights[pos - d_cluster_keys];
                      });
  }

  //-------------duplicated code
  //
  // For each vertex, compute its weighted degree
  // and cut between itself and its Louvain community
  //

  rmm::device_uvector<weight_t> weighted_degree_of_vertices(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  rmm::device_uvector<weight_t> weighted_cut_of_vertices(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  per_v_transform_reduce_outgoing_e(
    handle,
    graph_view,
    graph_view_t::is_multi_gpu
      ? src_clusters_cache.view()
      : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(next_clusters_v.data()),
    graph_view_t::is_multi_gpu ? dst_clusters_cache.view()
                               : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                   next_clusters_v.data(), vertex_t{0}),
    [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto dst_cluster) {
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
  // Inform other processors
  //

  edge_src_property_t<graph_view_t, thrust::tuple<weight_t, weight_t>>
    src_weighted_deg_and_cut_pair(handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    src_weighted_deg_and_cut_pair =
      edge_src_property_t<graph_view_t, thrust::tuple<weight_t, weight_t>>(handle, graph_view);
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
  // Compute the following
  //

  // E(v, leiden(v)-v) - ||v||* ||leiden(v)-v||/||V(G)||

  //
  // Each vertex starts as a singleton community in the leiden partition
  //

  rmm::device_uvector<vertex_t> leiden_assignment = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  detail::sequence_fill(handle.get_stream(),
                        leiden_assignment.begin(),
                        leiden_assignment.size(),
                        graph_view.local_vertex_partition_range_first());

  edge_src_property_t<graph_view_t, vertex_t> src_leiden_cluster_cache(handle);
  edge_dst_property_t<graph_view_t, vertex_t> dst_leiden_cluster_cache(handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    src_leiden_cluster_cache = edge_src_property_t<graph_view_t, vertex_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, leiden_assignment.begin(), src_leiden_cluster_cache);

    dst_leiden_cluster_cache = edge_dst_property_t<graph_view_t, vertex_t>(handle, graph_view);

    update_edge_dst_property(
      handle, graph_view, leiden_assignment.begin(), dst_leiden_cluster_cache);

    // Couldn't we clear leiden_assignment here?
  }

  //
  // For each refined community, compute its volume
  // (i.e.sum of weighted degree of all vertices inside it) and
  // and cut between itself and its Louvain community
  //

  rmm::device_uvector<weight_t> refined_community_volumes(0, handle.get_stream());

  rmm::device_uvector<weight_t> refined_community_cuts(0, handle.get_stream());

  auto src_input_property_values =
    graph_view_t::is_multi_gpu
      ? view_concat(src_clusters_cache.view(), src_leiden_cluster_cache.view())
      : view_concat(
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(next_clusters_v.data()),
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(leiden_assignment.data()));

  auto zipped_dst_louvain_leiden = thrust::make_zip_iterator(
    thrust::make_tuple(next_clusters_v.cbegin(), leiden_assignment.cbegin()));

  edge_dst_property_t<graph_view_t, thrust::tuple<vertex_t, vertex_t>> dst_louvain_leiden_cache(
    handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    dst_louvain_leiden_cache =
      edge_dst_property_t<graph_view_t, thrust::tuple<vertex_t, vertex_t>>(handle, graph_view);
    update_edge_dst_property(
      handle, graph_view, zipped_dst_louvain_leiden, dst_louvain_leiden_cache);
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

  if constexpr (graph_view_t::is_multi_gpu) {
    leiden_cluster_keys =
      shuffle_ext_vertices_and_values_by_gpu_id(handle, std::move(leiden_cluster_keys));
  }

  auto dst_input_property_values =
    graph_view_t::is_multi_gpu
      ? dst_louvain_leiden_cache.view()
      : view_concat(detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                      next_clusters_v.data(), vertex_t{0}),
                    detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                      leiden_assignment.data(), vertex_t{0}));

  std::forward_as_tuple(leiden_cluster_keys,
                        std::tie(refined_community_volumes, refined_community_cuts)) =
    cugraph::transform_reduce_e_by_dst_key(
      handle,
      graph_view,
      src_input_property_values,
      dst_input_property_values,
      graph_view_t::is_multi_gpu ? dst_leiden_cluster_cache.view()
                                 : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                     leiden_assignment.data(), vertex_t{0}),
      [] __device__(auto src,
                    auto dst,
                    auto wt,
                    thrust::tuple<vertex_t, vertex_t> src_louvain_leidn,
                    thrust::tuple<vertex_t, vertex_t> dst_louvain_leiden) {
        weight_t refined_partition_volume_contribution{0};
        weight_t refined_partition_cut_contribution{0};

        auto src_louvain = thrust::get<0>(src_louvain_leidn);
        auto src_leiden  = thrust::get<1>(src_louvain_leidn);

        auto dst_louvain = thrust::get<0>(dst_louvain_leiden);
        auto dst_leiden  = thrust::get<1>(dst_louvain_leiden);

        if (src_louvain == dst_louvain) {
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
  // FIXME: When Primitive get updated to take set of active vertices
  //
  rmm::device_uvector<uint8_t> singleton_mask(leiden_assignment.size(), handle.get_stream());

  thrust::fill(
    handle.get_thrust_policy(), singleton_mask.begin(), singleton_mask.end(), uint8_t{0});

  edge_src_property_t<graph_view_t, uint8_t> src_singleton_mask(handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    src_singleton_mask = edge_src_property_t<graph_view_t, uint8_t>(handle, graph_view);
    update_edge_src_property(handle, graph_view, singleton_mask.begin(), src_singleton_mask);
    // Couldn't we clear singleton_mask here?
  }

  //
  // Primitives to decide best (at least good) next clusters for vertices
  //

  auto output_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, weight_t>>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  auto zipped_src_device_view =
    graph_view_t::is_multi_gpu
      ? view_concat(src_weighted_deg_and_cut_pair.view(),
                    src_cluster_weights.view(),
                    src_singleton_mask.view(),
                    src_leiden_cluster_cache.view(),
                    src_clusters_cache.view())
      : view_concat(
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(
            weighted_degree_of_vertices.data()),
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(
            weighted_cut_of_vertices.data()),
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(
            vertex_cluster_weights_v.data()),
          detail::edge_major_property_view_t<vertex_t, uint8_t const*>(singleton_mask.data()),
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(leiden_assignment.data()),
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(next_clusters_v.data()));

  // ||v||
  // E(v, louvain(v))
  // ||louvain(v)||
  // is_singleton(v)
  // leiden(v)
  // louvain(v)

  // ||Cr||   //f(Cr)
  // E(Cr, louvain(v) - Cr) //f(Cr)
  // louvain(Cr)  // f(Cr)

  //
  // Construct Leiden to Louvain mapping
  //
  rmm::device_uvector<vertex_t> keys_of_leiden_to_louvain_map(leiden_assignment.size(),
                                                              handle.get_stream());
  rmm::device_uvector<vertex_t> values_of_leiden_to_louvain_map(next_clusters_v.size(),
                                                                handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               next_clusters_v.begin(),
               next_clusters_v.end(),
               values_of_leiden_to_louvain_map.begin());

  thrust::copy(handle.get_thrust_policy(),
               leiden_assignment.begin(),
               leiden_assignment.end(),
               keys_of_leiden_to_louvain_map.begin());

  auto louvain_leiden_zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(
    values_of_leiden_to_louvain_map.begin(), keys_of_leiden_to_louvain_map.begin()));

  auto louvain_leiden_zipped_end = thrust::make_zip_iterator(
    thrust::make_tuple(values_of_leiden_to_louvain_map.end(), keys_of_leiden_to_louvain_map.end()));

  thrust::sort(handle.get_thrust_policy(),
               louvain_leiden_zipped_begin,
               louvain_leiden_zipped_end,
               thrust::less<thrust::tuple<vertex_t, vertex_t>>());

  auto last_unique_leiden_louvain_pair =
    thrust::unique(handle.get_thrust_policy(),
                   louvain_leiden_zipped_begin,
                   louvain_leiden_zipped_end,
                   thrust::less<thrust::tuple<vertex_t, vertex_t>>());

  auto nr_unique_leiden_louvain_pairs = static_cast<size_t>(
    thrust::distance(louvain_leiden_zipped_begin, last_unique_leiden_louvain_pair));

  keys_of_leiden_to_louvain_map.resize(nr_unique_leiden_louvain_pairs, handle.get_stream());
  values_of_leiden_to_louvain_map.resize(nr_unique_leiden_louvain_pairs, handle.get_stream());

  //
  // Collect Louvain cluster ids for refined (aka Leiden) clusters
  //
  rmm::device_uvector<vertex_t> lovain_of_refined_comms(0, handle.get_stream());

  if constexpr (graph_view_t::is_multi_gpu) {
    std::tie(keys_of_leiden_to_louvain_map, values_of_leiden_to_louvain_map) =
      shuffle_ext_vertices_and_values_by_gpu_id(handle,
                                                std::move(keys_of_leiden_to_louvain_map),
                                                std::move(values_of_leiden_to_louvain_map));

    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      handle.get_comms().get_size()};

    lovain_of_refined_comms =
      cugraph::collect_values_for_keys(handle.get_comms(),
                                       keys_of_leiden_to_louvain_map.begin(),
                                       keys_of_leiden_to_louvain_map.end(),
                                       values_of_leiden_to_louvain_map.data(),
                                       leiden_cluster_keys.begin(),
                                       leiden_cluster_keys.end(),
                                       vertex_to_gpu_id_op,
                                       invalid_vertex_id<vertex_t>::value,
                                       invalid_vertex_id<vertex_t>::value,
                                       handle.get_stream());

  } else {
    // sort so we can use lower_bound in the transform function
    thrust::sort_by_key(handle.get_thrust_policy(),
                        keys_of_leiden_to_louvain_map.begin(),
                        keys_of_leiden_to_louvain_map.end(),
                        values_of_leiden_to_louvain_map.begin());

    lovain_of_refined_comms.resize(leiden_cluster_keys.size(), handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(),
      leiden_cluster_keys.begin(),
      leiden_cluster_keys.end(),
      lovain_of_refined_comms.begin(),
      [d_map_values = values_of_leiden_to_louvain_map.data(),
       d_map_keys   = keys_of_leiden_to_louvain_map.data(),
       d_map_size   = keys_of_leiden_to_louvain_map.size()] __device__(vertex_t leiden_cluster) {
        auto ptr =
          thrust::lower_bound(thrust::seq, d_map_keys, d_map_keys + d_map_size, leiden_cluster);
        return d_map_values[ptr - d_map_keys];
      });
  }

  auto values_for_leiden_cluster_keys =
    thrust::make_zip_iterator(thrust::make_tuple(refined_community_volumes.begin(),
                                                 refined_community_cuts.begin(),
                                                 leiden_assignment.begin(),
                                                 lovain_of_refined_comms.begin()));

  //
  // Decide best/positive move for each vertex
  //
  /*
  per_v_transform_reduce_dst_key_aggregated_outgoing_e(
    handle,
    graph_view,
    zipped_src_device_view,
    graph_view_t::is_multi_gpu ? dst_leiden_cluster_cache.view()
                               : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                   leiden_assignment.data(), vertex_t{0}),
    leiden_cluster_keys.begin(),
    leiden_cluster_keys.end(),
    values_for_leiden_cluster_keys,
    invalid_vertex_id<vertex_t>::value,
    thrust::make_tuple(std::numeric_limits<weight_t>::max(),
                       std::numeric_limits<weight_t>::max(),
                       invalid_vertex_id<vertex_t>::value,
                       invalid_vertex_id<vertex_t>::value),
    detail::leiden_key_aggregated_edge_op_t<vertex_t, weight_t>{total_edge_weight, resolution},
    thrust::make_tuple(vertex_t{-1}, weight_t{0}),
    detail::reduce_op_t<vertex_t, weight_t>{},
    cugraph::get_dataframe_buffer_begin(output_buffer));
*/
  //
  // Move singleton vertex to other singleton/non-singleton Leiden community
  //

  thrust::transform(handle.get_thrust_policy(),
                    leiden_assignment.begin(),
                    leiden_assignment.end(),
                    cugraph::get_dataframe_buffer_begin(output_buffer),
                    leiden_assignment.begin(),
                    detail::cluster_update_op_t<vertex_t, weight_t>{up_down});

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

  if constexpr (graph_view_t::is_multi_gpu) {
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

  // TODO: Put this in a function
  // Collect Louvain cluster ids for refined (aka Leiden) clusters
  //

  if constexpr (graph_view_t::is_multi_gpu) {
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      handle.get_comms().get_size()};

    lovain_of_refined_comms =
      cugraph::collect_values_for_keys(handle.get_comms(),
                                       keys_of_leiden_to_louvain_map.begin(),
                                       keys_of_leiden_to_louvain_map.end(),
                                       values_of_leiden_to_louvain_map.data(),
                                       keys_to_read_value_for.begin(),
                                       keys_to_read_value_for.end(),
                                       vertex_to_gpu_id_op,
                                       invalid_vertex_id<vertex_t>::value,
                                       invalid_vertex_id<vertex_t>::value,
                                       handle.get_stream());

  } else {
    // sort so we can use lower_bound in the transform function
    thrust::sort_by_key(handle.get_thrust_policy(),
                        keys_of_leiden_to_louvain_map.begin(),
                        keys_of_leiden_to_louvain_map.end(),
                        values_of_leiden_to_louvain_map.begin());

    lovain_of_refined_comms.resize(keys_to_read_value_for.size(), handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(),
      keys_to_read_value_for.begin(),
      keys_to_read_value_for.end(),
      lovain_of_refined_comms.begin(),
      [d_map_values = values_of_leiden_to_louvain_map.data(),
       d_map_keys   = keys_of_leiden_to_louvain_map.data(),
       d_map_size   = keys_of_leiden_to_louvain_map.size()] __device__(vertex_t leiden_cluster) {
        auto ptr =
          thrust::lower_bound(thrust::seq, d_map_keys, d_map_keys + d_map_size, leiden_cluster);
        return d_map_values[ptr - d_map_keys];
      });
  }

  return std::make_tuple(
    std::move(leiden_assignment),
    std::make_pair(std::move(keys_to_read_value_for), std::move(lovain_of_refined_comms)));
  rmm::device_uvector<vertex_t> temp1 = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());
  rmm::device_uvector<vertex_t> temp2 = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  return std::make_tuple(std::move(leiden_assignment),
                         std::make_pair(std::move(temp1), std::move(temp2)));
}

template <typename graph_view_t>
rmm::device_uvector<typename graph_view_t::vertex_type> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  typename graph_view_t::weight_type total_edge_weight,
  typename graph_view_t::weight_type resolution,
  rmm::device_uvector<typename graph_view_t::weight_type> const& vertex_weights_v,
  rmm::device_uvector<typename graph_view_t::vertex_type>&& cluster_keys_v,
  rmm::device_uvector<typename graph_view_t::weight_type>&& cluster_weights_v,
  rmm::device_uvector<typename graph_view_t::vertex_type>&& next_clusters_v,
  edge_src_property_t<graph_view_t, typename graph_view_t::weight_type> const&
    src_vertex_weights_cache,
  edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const& src_clusters_cache,
  edge_dst_property_t<graph_view_t, typename graph_view_t::vertex_type> const& dst_clusters_cache,
  bool up_down)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using weight_t = typename graph_view_t::weight_type;

  rmm::device_uvector<weight_t> vertex_cluster_weights_v(0, handle.get_stream());
  edge_src_property_t<graph_view_t, weight_t> src_cluster_weights(handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      handle.get_comms().get_size()};

    vertex_cluster_weights_v =
      cugraph::collect_values_for_keys(handle.get_comms(),
                                       cluster_keys_v.begin(),
                                       cluster_keys_v.end(),
                                       cluster_weights_v.data(),
                                       next_clusters_v.begin(),
                                       next_clusters_v.end(),
                                       vertex_to_gpu_id_op,
                                       invalid_vertex_id<vertex_t>::value,
                                       std::numeric_limits<weight_t>::max(),
                                       handle.get_stream());

    src_cluster_weights = edge_src_property_t<graph_view_t, weight_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, vertex_cluster_weights_v.begin(), src_cluster_weights);
    vertex_cluster_weights_v.resize(0, handle.get_stream());
    vertex_cluster_weights_v.shrink_to_fit(handle.get_stream());
  } else {
    // sort so we can use lower_bound in the transform function
    thrust::sort_by_key(handle.get_thrust_policy(),
                        cluster_keys_v.begin(),
                        cluster_keys_v.end(),
                        cluster_weights_v.begin());

    // for each vertex, look up the vertex weight of the current cluster it is assigned to
    vertex_cluster_weights_v.resize(next_clusters_v.size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      next_clusters_v.begin(),
                      next_clusters_v.end(),
                      vertex_cluster_weights_v.begin(),
                      [d_cluster_weights = cluster_weights_v.data(),
                       d_cluster_keys    = cluster_keys_v.data(),
                       num_clusters      = cluster_keys_v.size()] __device__(vertex_t cluster) {
                        auto pos = thrust::lower_bound(
                          thrust::seq, d_cluster_keys, d_cluster_keys + num_clusters, cluster);
                        return d_cluster_weights[pos - d_cluster_keys];
                      });
  }

  rmm::device_uvector<weight_t> old_cluster_sum_v(graph_view.local_vertex_partition_range_size(),
                                                  handle.get_stream());
  rmm::device_uvector<weight_t> cluster_subtract_v(graph_view.local_vertex_partition_range_size(),
                                                   handle.get_stream());

  per_v_transform_reduce_outgoing_e(
    handle,
    graph_view,
    graph_view_t::is_multi_gpu
      ? src_clusters_cache.view()
      : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(next_clusters_v.data()),
    graph_view_t::is_multi_gpu ? dst_clusters_cache.view()
                               : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                   next_clusters_v.data(), vertex_t{0}),
    [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto nbr_cluster) {
      weight_t sum{0};
      weight_t subtract{0};

      if (src == dst)
        subtract = wt;
      else if (src_cluster == nbr_cluster)
        sum = wt;

      return thrust::make_tuple(sum, subtract);
    },
    thrust::make_tuple(weight_t{0}, weight_t{0}),
    thrust::make_zip_iterator(
      thrust::make_tuple(old_cluster_sum_v.begin(), cluster_subtract_v.begin())));

  edge_src_property_t<graph_view_t, thrust::tuple<weight_t, weight_t>>
    src_old_cluster_sum_subtract_pairs(handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    src_old_cluster_sum_subtract_pairs =
      edge_src_property_t<graph_view_t, thrust::tuple<weight_t, weight_t>>(handle, graph_view);
    update_edge_src_property(handle,
                             graph_view,
                             thrust::make_zip_iterator(thrust::make_tuple(
                               old_cluster_sum_v.begin(), cluster_subtract_v.begin())),
                             src_old_cluster_sum_subtract_pairs);
    old_cluster_sum_v.resize(0, handle.get_stream());
    old_cluster_sum_v.shrink_to_fit(handle.get_stream());
    cluster_subtract_v.resize(0, handle.get_stream());
    cluster_subtract_v.shrink_to_fit(handle.get_stream());
  }

  auto output_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, weight_t>>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  auto cluster_old_sum_subtract_pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(old_cluster_sum_v.cbegin(), cluster_subtract_v.cbegin()));
  auto zipped_src_device_view =
    graph_view_t::is_multi_gpu
      ? view_concat(src_vertex_weights_cache.view(),
                    src_clusters_cache.view(),
                    src_cluster_weights.view(),
                    src_old_cluster_sum_subtract_pairs.view())
      : view_concat(
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(vertex_weights_v.data()),
          detail::edge_major_property_view_t<vertex_t, vertex_t const*>(next_clusters_v.data()),
          detail::edge_major_property_view_t<vertex_t, weight_t const*>(
            vertex_cluster_weights_v.data()),
          detail::edge_major_property_view_t<vertex_t,
                                             decltype(cluster_old_sum_subtract_pair_first)>(
            cluster_old_sum_subtract_pair_first));

  per_v_transform_reduce_dst_key_aggregated_outgoing_e(
    handle,
    graph_view,
    zipped_src_device_view,
    graph_view_t::is_multi_gpu ? dst_clusters_cache.view()
                               : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                   next_clusters_v.data(), vertex_t{0}),
    cluster_keys_v.begin(),
    cluster_keys_v.end(),
    cluster_weights_v.begin(),
    invalid_vertex_id<vertex_t>::value,
    std::numeric_limits<weight_t>::max(),
    detail::key_aggregated_edge_op_t<vertex_t, weight_t>{total_edge_weight, resolution},
    thrust::make_tuple(vertex_t{-1}, weight_t{0}),
    detail::reduce_op_t<vertex_t, weight_t>{},
    cugraph::get_dataframe_buffer_begin(output_buffer));

  thrust::transform(handle.get_thrust_policy(),
                    next_clusters_v.begin(),
                    next_clusters_v.end(),
                    cugraph::get_dataframe_buffer_begin(output_buffer),
                    next_clusters_v.begin(),
                    detail::cluster_update_op_t<vertex_t, weight_t>{up_down});

  return std::move(next_clusters_v);
}

template <typename graph_view_t>
std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
           rmm::device_uvector<typename graph_view_t::weight_type>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  rmm::device_uvector<typename graph_view_t::vertex_type> const& next_clusters_v,
  edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const& src_clusters_cache)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using weight_t = typename graph_view_t::weight_type;

  auto [cluster_keys, cluster_values] = cugraph::transform_reduce_e_by_src_key(
    handle,
    graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    graph_view_t::is_multi_gpu
      ? src_clusters_cache.view()
      : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(next_clusters_v.data()),
    detail::return_edge_weight_t<vertex_t, weight_t>{},
    weight_t{0},
    reduce_op::plus<weight_t>{});

  return std::make_tuple(std::move(cluster_keys), std::move(cluster_values));
}

}  // namespace detail
}  // namespace cugraph
