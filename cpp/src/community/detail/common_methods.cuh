/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "common_methods.hpp"
#include "detail/graph_partition_utils.cuh"
#include "prims/kv_store.cuh"
#include "prims/per_v_transform_reduce_dst_key_aggregated_outgoing_e.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_e.cuh"
#include "prims/transform_reduce_e_by_src_dst_key.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <cuda/functional>
#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)
// FIXME: a temporary workaround for a compiler error, should be deleted once cuco gets patched.
namespace cuco {
template <>
struct is_bitwise_comparable<cuco::pair<int32_t, float>> : std::true_type {};
}  // namespace cuco

namespace cugraph {
namespace detail {

// FIXME: a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t resolution{};
  __device__ auto operator()(
    vertex_t src,
    vertex_t neighbor_cluster,
    thrust::tuple<weight_t, vertex_t, weight_t, weight_t, weight_t> src_info,
    weight_t a_new,
    weight_t new_cluster_sum) const
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

// FIXME: a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
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

// FIXME: a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct count_updown_moves_op_t {
  bool up_down{};
  __device__ auto operator()(thrust::tuple<vertex_t, thrust::tuple<vertex_t, weight_t>> p) const
  {
    vertex_t old_cluster       = thrust::get<0>(p);
    auto new_cluster_gain_pair = thrust::get<1>(p);
    vertex_t new_cluster       = thrust::get<0>(new_cluster_gain_pair);
    weight_t delta_modularity  = thrust::get<1>(new_cluster_gain_pair);

    auto result_assignment =
      (delta_modularity > weight_t{0})
        ? (((new_cluster > old_cluster) != up_down) ? old_cluster : new_cluster)
        : old_cluster;

    return (delta_modularity > weight_t{0})
             ? (((new_cluster > old_cluster) != up_down) ? false : true)
             : false;
  }
};
// FIXME: a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
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

// FIXME: a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct return_edge_weight_t {
  __device__ auto operator()(
    vertex_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, weight_t w) const
  {
    return w;
  }
};

// FIXME: a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct return_one_t {
  __device__ auto operator()(
    vertex_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) const
  {
    return 1.0;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
weight_t compute_modularity(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  edge_src_property_t<vertex_t, vertex_t> const& src_clusters_cache,
  edge_dst_property_t<vertex_t, vertex_t> const& dst_clusters_cache,
  rmm::device_uvector<vertex_t> const& next_clusters,
  rmm::device_uvector<weight_t> const& cluster_weights,
  weight_t total_edge_weight,
  weight_t resolution)
{
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted.");

  //
  // Sum(Sigma_tot_c^2), over all clusters c
  //
  auto squared_first = thrust::make_transform_iterator(
    cluster_weights.begin(),
    cuda::proclaim_return_type<weight_t>([] __device__(weight_t p) { return p * p; }));
  weight_t sum_degree_squared = thrust::reduce(
    handle.get_thrust_policy(), squared_first, squared_first + cluster_weights.size());

  if constexpr (multi_gpu) {
    sum_degree_squared = host_scalar_allreduce(
      handle.get_comms(), sum_degree_squared, raft::comms::op_t::SUM, handle.get_stream());
  }

  // Sum(Sigma_in_c), over all clusters c
  weight_t sum_internal = transform_reduce_e(
    handle,
    graph_view,
    multi_gpu ? src_clusters_cache.view()
              : make_edge_src_property_view<vertex_t, vertex_t>(
                  graph_view, next_clusters.begin(), next_clusters.size()),
    multi_gpu ? dst_clusters_cache.view()
              : make_edge_dst_property_view<vertex_t, vertex_t>(
                  graph_view, next_clusters.begin(), next_clusters.size()),
    *edge_weight_view,
    cuda::proclaim_return_type<weight_t>(
      [] __device__(auto, auto, auto src_cluster, auto nbr_cluster, weight_t wt) {
        if (src_cluster == nbr_cluster) {
          return wt;
        } else {
          return weight_t{0};
        }
      }),
    weight_t{0});

  weight_t Q = sum_internal / total_edge_weight -
               (resolution * sum_degree_squared) / (total_edge_weight * total_edge_weight);

  return Q;
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>,
           std::optional<edge_property_t<edge_t, weight_t>>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weights_view,
                  raft::device_span<vertex_t> labels)
{
  auto [new_graph, new_edge_weights, numbering_map] =
    coarsen_graph(handle, graph_view, edge_weights_view, labels.data(), true);

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

  return std::make_tuple(std::move(new_graph), std::move(new_edge_weights));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t total_edge_weight,
  weight_t resolution,
  rmm::device_uvector<weight_t> const& vertex_weights_v,
  rmm::device_uvector<vertex_t>&& cluster_keys_v,
  rmm::device_uvector<weight_t>&& cluster_weights_v,
  rmm::device_uvector<vertex_t>&& next_clusters_v,
  edge_src_property_t<vertex_t, weight_t> const& src_vertex_weights_cache,
  edge_src_property_t<vertex_t, vertex_t> const& src_clusters_cache,
  edge_dst_property_t<vertex_t, vertex_t> const& dst_clusters_cache,
  bool up_down)
{
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted.");

  rmm::device_uvector<weight_t> vertex_cluster_weights_v(0, handle.get_stream());
  std::optional<edge_src_property_t<vertex_t, weight_t>> src_cluster_weights{std::nullopt};

  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      comm_size, major_comm_size, minor_comm_size};

    kv_store_t<vertex_t, weight_t, false> cluster_key_weight_map(
      cluster_keys_v.begin(),
      cluster_keys_v.end(),
      cluster_weights_v.data(),
      invalid_vertex_id<vertex_t>::value,
      std::numeric_limits<weight_t>::max(),
      handle.get_stream());
    vertex_cluster_weights_v = cugraph::collect_values_for_keys(comm,
                                                                cluster_key_weight_map.view(),
                                                                next_clusters_v.begin(),
                                                                next_clusters_v.end(),
                                                                vertex_to_gpu_id_op,
                                                                handle.get_stream());

    src_cluster_weights = edge_src_property_t<vertex_t, weight_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, vertex_cluster_weights_v.begin(), src_cluster_weights->mutable_view());
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
    multi_gpu ? src_clusters_cache.view()
              : make_edge_src_property_view<vertex_t, vertex_t>(
                  graph_view, next_clusters_v.begin(), next_clusters_v.size()),
    multi_gpu ? dst_clusters_cache.view()
              : make_edge_dst_property_view<vertex_t, vertex_t>(
                  graph_view, next_clusters_v.begin(), next_clusters_v.size()),
    *edge_weight_view,
    [] __device__(auto src, auto dst, auto src_cluster, auto nbr_cluster, weight_t wt) {
      weight_t sum{0};
      weight_t subtract{0};

      if (src == dst)
        subtract = wt;
      else if (src_cluster == nbr_cluster)
        sum = wt;

      return thrust::make_tuple(sum, subtract);
    },
    thrust::make_tuple(weight_t{0}, weight_t{0}),
    reduce_op::plus<thrust::tuple<weight_t, weight_t>>{},
    thrust::make_zip_iterator(
      thrust::make_tuple(old_cluster_sum_v.begin(), cluster_subtract_v.begin())));

  std::optional<edge_src_property_t<vertex_t, thrust::tuple<weight_t, weight_t>>>
    src_old_cluster_sum_subtract_pairs{std::nullopt};

  if constexpr (multi_gpu) {
    src_old_cluster_sum_subtract_pairs =
      edge_src_property_t<vertex_t, thrust::tuple<weight_t, weight_t>>(handle, graph_view);
    update_edge_src_property(handle,
                             graph_view,
                             thrust::make_zip_iterator(thrust::make_tuple(
                               old_cluster_sum_v.begin(), cluster_subtract_v.begin())),
                             src_old_cluster_sum_subtract_pairs->mutable_view());
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
    multi_gpu ? view_concat(src_vertex_weights_cache.view(),
                            src_clusters_cache.view(),
                            src_cluster_weights->view(),
                            src_old_cluster_sum_subtract_pairs->view())
              : view_concat(
                  make_edge_src_property_view<vertex_t, weight_t>(
                    graph_view, vertex_weights_v.begin(), vertex_weights_v.size()),
                  make_edge_src_property_view<vertex_t, vertex_t>(
                    graph_view, next_clusters_v.begin(), next_clusters_v.size()),
                  make_edge_src_property_view<vertex_t, weight_t>(
                    graph_view, vertex_cluster_weights_v.begin(), vertex_cluster_weights_v.size()),
                  make_edge_src_property_view<vertex_t, thrust::tuple<weight_t, weight_t>>(
                    graph_view, cluster_old_sum_subtract_pair_first, old_cluster_sum_v.size()));

  kv_store_t<vertex_t, weight_t, false> cluster_key_weight_map(
    cluster_keys_v.begin(),
    cluster_keys_v.begin() + cluster_keys_v.size(),
    cluster_weights_v.begin(),
    invalid_vertex_id<vertex_t>::value,
    std::numeric_limits<weight_t>::max(),
    handle.get_stream());
  per_v_transform_reduce_dst_key_aggregated_outgoing_e(
    handle,
    graph_view,
    zipped_src_device_view,
    *edge_weight_view,
    multi_gpu ? dst_clusters_cache.view()
              : make_edge_dst_property_view<vertex_t, vertex_t>(
                  graph_view, next_clusters_v.begin(), next_clusters_v.size()),
    cluster_key_weight_map.view(),
    detail::key_aggregated_edge_op_t<vertex_t, weight_t>{total_edge_weight, resolution},
    thrust::make_tuple(vertex_t{-1}, weight_t{0}),
    detail::reduce_op_t<vertex_t, weight_t>{},
    cugraph::get_dataframe_buffer_begin(output_buffer));

  int nr_moves = thrust::count_if(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(thrust::make_tuple(
      next_clusters_v.begin(), cugraph::get_dataframe_buffer_begin(output_buffer))),
    thrust::make_zip_iterator(
      thrust::make_tuple(next_clusters_v.end(), cugraph::get_dataframe_buffer_end(output_buffer))),
    detail::count_updown_moves_op_t<vertex_t, weight_t>{up_down});

  if (multi_gpu) {
    nr_moves = host_scalar_allreduce(
      handle.get_comms(), nr_moves, raft::comms::op_t::SUM, handle.get_stream());
  }

  if (nr_moves == 0) { up_down = !up_down; }

  thrust::transform(handle.get_thrust_policy(),
                    next_clusters_v.begin(),
                    next_clusters_v.end(),
                    cugraph::get_dataframe_buffer_begin(output_buffer),
                    next_clusters_v.begin(),
                    detail::cluster_update_op_t<vertex_t, weight_t>{up_down});

  return std::move(next_clusters_v);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  rmm::device_uvector<vertex_t> const& next_clusters_v,
  edge_src_property_t<vertex_t, vertex_t> const& src_clusters_cache)
{
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted.");

  auto [cluster_keys, cluster_values] = cugraph::transform_reduce_e_by_src_key(
    handle,
    graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    *edge_weight_view,
    multi_gpu ? src_clusters_cache.view()
              : make_edge_src_property_view<vertex_t, vertex_t>(
                  graph_view, next_clusters_v.begin(), next_clusters_v.size()),
    detail::return_edge_weight_t<vertex_t, weight_t>{},
    weight_t{0},
    reduce_op::plus<weight_t>{});

  return std::make_tuple(std::move(cluster_keys), std::move(cluster_values));
}

}  // namespace detail
}  // namespace cugraph
