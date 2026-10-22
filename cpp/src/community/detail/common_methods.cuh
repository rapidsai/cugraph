/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common_methods.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/count_if_v.cuh>
#include <cugraph/prims/kv_store.cuh>
#include <cugraph/prims/per_v_transform_reduce_dst_key_aggregated_outgoing_e.cuh>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/transform_reduce_e.cuh>
#include <cugraph/prims/transform_reduce_e_by_src_dst_key.cuh>
#include <cugraph/prims/transform_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>
#include <cugraph/utilities/thrust_wrappers/unique.hpp>
#include <cugraph/vertex_property.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <limits>
#include <type_traits>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)
// FIXME: a temporary workaround for a compiler error, should be deleted once cuco gets patched.
namespace cuco {
template <>
struct is_bitwise_comparable<cuco::pair<int32_t, float>> : std::true_type {};
}  // namespace cuco

namespace cugraph {
namespace detail {

template <typename weight_t>
constexpr weight_t louvain_delta_modularity_noise_floor()
{
  if constexpr (std::is_same_v<weight_t, float>) {
    return weight_t{1e-12};
  } else {
    return weight_t{1e-15};
  }
}

template <typename weight_t, typename vertex_t>
weight_t compute_louvain_min_vertex_move_gain(weight_t threshold, vertex_t number_of_vertices)
{
  auto const vertex_count     = std::max(number_of_vertices, vertex_t{1});
  auto const scaled_threshold = threshold / static_cast<weight_t>(vertex_count);
  return std::max(scaled_threshold, louvain_delta_modularity_noise_floor<weight_t>());
}

template <typename vertex_t, typename weight_t>
struct key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t resolution{};
  __device__ auto operator()(
    vertex_t src,
    vertex_t neighbor_cluster,
    cuda::std::tuple<weight_t, vertex_t, weight_t, weight_t, weight_t> src_info,
    weight_t a_new,
    weight_t new_cluster_sum) const
  {
    auto k_k              = cuda::std::get<0>(src_info);
    auto src_cluster      = cuda::std::get<1>(src_info);
    auto a_old            = cuda::std::get<2>(src_info);
    auto old_cluster_sum  = cuda::std::get<3>(src_info);
    auto cluster_subtract = cuda::std::get<4>(src_info);

    if (src_cluster == neighbor_cluster) new_cluster_sum -= cluster_subtract;

    weight_t delta_modularity = 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                     resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                       (total_edge_weight * total_edge_weight));

    return cuda::std::make_tuple(neighbor_cluster, delta_modularity);
  }
};

template <typename vertex_t, typename weight_t>
struct reduce_op_t {
  using type                                = cuda::std::tuple<vertex_t, weight_t>;
  static constexpr bool pure_function       = true;  // this can be called from any process
  inline static type const identity_element = cuda::std::make_tuple(
    std::numeric_limits<weight_t>::lowest(), invalid_vertex_id<vertex_t>::value);

  __device__ auto operator()(cuda::std::tuple<vertex_t, weight_t> p0,
                             cuda::std::tuple<vertex_t, weight_t> p1) const
  {
    auto id0 = cuda::std::get<0>(p0);
    auto id1 = cuda::std::get<0>(p1);
    auto wt0 = cuda::std::get<1>(p0);
    auto wt1 = cuda::std::get<1>(p1);

    return (wt0 < wt1) ? p1 : ((wt0 > wt1) ? p0 : ((id0 < id1) ? p0 : p1));
  }
};

template <typename vertex_t, typename weight_t>
struct count_updown_moves_op_t {
  bool up_down{};
  weight_t min_gain{};
  __device__ bool operator()(vertex_t, cuda::std::tuple<vertex_t, vertex_t, weight_t> p) const
  {
    vertex_t old_cluster      = cuda::std::get<0>(p);
    vertex_t new_cluster      = cuda::std::get<1>(p);
    weight_t delta_modularity = cuda::std::get<2>(p);

    return (delta_modularity > min_gain) ? (((new_cluster > old_cluster) != up_down) ? false : true)
                                         : false;
  }
};

template <typename vertex_t, typename weight_t>
struct cluster_update_op_t {
  bool up_down{};
  weight_t min_gain{};
  __device__ vertex_t operator()(cuda::std::tuple<vertex_t, vertex_t, weight_t> p) const
  {
    vertex_t old_cluster      = cuda::std::get<0>(p);
    vertex_t new_cluster      = cuda::std::get<1>(p);
    weight_t delta_modularity = cuda::std::get<2>(p);

    return (delta_modularity > min_gain)
             ? (((new_cluster > old_cluster) != up_down) ? old_cluster : new_cluster)
             : old_cluster;
  }
};

template <typename vertex_t, typename weight_t>
struct return_edge_weight_t {
  __device__ auto operator()(
    vertex_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, weight_t w) const
  {
    return w;
  }
};

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
  vertex_property_view_t<vertex_t, vertex_t const*> next_clusters,
  rmm::device_uvector<weight_t> const& cluster_weights,
  weight_t total_edge_weight,
  weight_t resolution)
{
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted.");

  //
  // Sum(Sigma_tot_c^2), over all clusters c
  //
  auto squared_first = cuda::make_transform_iterator(
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
    multi_gpu ? src_clusters_cache.view() : make_edge_src_property_view(graph_view, next_clusters),
    multi_gpu ? dst_clusters_cache.view() : make_edge_dst_property_view(graph_view, next_clusters),
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
           std::optional<edge_property_t<edge_t, weight_t>>,
           vertex_property_t<vertex_t, vertex_t>>
graph_contraction(raft::handle_t const& handle,
                  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weights_view,
                  vertex_property_t<vertex_t, vertex_t>&& labels)
{
  auto [new_graph, new_edge_weights, numbering_map] =
    coarsen_graph(handle, graph_view, edge_weights_view, labels.view(), true);

  auto new_graph_view = new_graph.view();

  labels = vertex_property_t<vertex_t, vertex_t>(
    renumber_ext_vertices(handle, new_graph_view, labels.release_buffer(), numbering_map->view()),
    graph_view.local_vertex_partition_range_size());

  return std::make_tuple(std::move(new_graph), std::move(new_edge_weights), std::move(labels));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
vertex_property_t<vertex_t, vertex_t> update_clustering_by_delta_modularity(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t total_edge_weight,
  weight_t resolution,
  vertex_property_view_t<vertex_t, weight_t const*> vertex_weights_v,
  rmm::device_uvector<vertex_t>&& cluster_keys_v,
  rmm::device_uvector<weight_t>&& cluster_weights_v,
  vertex_property_t<vertex_t, vertex_t>&& next_clusters_v,
  edge_src_property_t<vertex_t, weight_t> const& src_vertex_weights_cache,
  edge_src_property_t<vertex_t, vertex_t> const& src_clusters_cache,
  edge_dst_property_t<vertex_t, vertex_t> const& dst_clusters_cache,
  bool up_down,
  weight_t threshold)
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

    {
      kv_store_t<vertex_t, weight_t, false> cluster_key_weight_map(
        cluster_keys_v.begin(),
        cluster_keys_v.end(),
        cluster_weights_v.data(),
        invalid_vertex_id<vertex_t>::value,
        std::numeric_limits<weight_t>::max(),
        handle.get_stream());
      auto next_clusters_view  = next_clusters_v.view();
      vertex_cluster_weights_v = cugraph::collect_values_for_keys(
        handle,
        cluster_key_weight_map.view(),
        next_clusters_view.value_first(),
        next_clusters_view.value_first() + next_clusters_view.range_size(),
        vertex_to_gpu_id_op);
    }

    src_cluster_weights = edge_src_property_t<vertex_t, weight_t>(handle, graph_view);
    update_edge_src_property(handle,
                             graph_view,
                             vertex_property_view_t{vertex_cluster_weights_v.begin(),
                                                    graph_view.local_vertex_partition_range_size()},
                             src_cluster_weights->mutable_view());
    vertex_cluster_weights_v.resize(0, handle.get_stream());
    vertex_cluster_weights_v.shrink_to_fit(handle.get_stream());
  } else {
    // sort so we can use lower_bound in the transform function
    thrust::sort_by_key(handle.get_thrust_policy(),
                        cluster_keys_v.begin(),
                        cluster_keys_v.end(),
                        cluster_weights_v.begin());

    // for each vertex, look up the vertex weight of the current cluster it is assigned to
    vertex_cluster_weights_v.resize(graph_view.local_vertex_partition_range_size(),
                                    handle.get_stream());
    transform_v(handle,
                graph_view,
                next_clusters_v.view(),
                cuda::proclaim_return_type<weight_t>(
                  [d_cluster_weights = cluster_weights_v.data(),
                   d_cluster_keys    = cluster_keys_v.data(),
                   num_clusters      = cluster_keys_v.size()] __device__(vertex_t cluster) {
                    auto pos = thrust::lower_bound(
                      thrust::seq, d_cluster_keys, d_cluster_keys + num_clusters, cluster);
                    return d_cluster_weights[pos - d_cluster_keys];
                  }),
                vertex_property_view_t{vertex_cluster_weights_v.begin(),
                                       graph_view.local_vertex_partition_range_size()});
  }

  rmm::device_uvector<weight_t> old_cluster_sum_v(graph_view.local_vertex_partition_range_size(),
                                                  handle.get_stream());
  rmm::device_uvector<weight_t> cluster_subtract_v(graph_view.local_vertex_partition_range_size(),
                                                   handle.get_stream());

  per_v_transform_reduce_outgoing_e(
    handle,
    graph_view,
    multi_gpu ? src_clusters_cache.view()
              : make_edge_src_property_view(graph_view, next_clusters_v.view()),
    multi_gpu ? dst_clusters_cache.view()
              : make_edge_dst_property_view(graph_view, next_clusters_v.view()),
    *edge_weight_view,
    [] __device__(auto src, auto dst, auto src_cluster, auto nbr_cluster, weight_t wt) {
      weight_t sum{0};
      weight_t subtract{0};

      if (src == dst)
        subtract = wt;
      else if (src_cluster == nbr_cluster)
        sum = wt;

      return cuda::std::make_tuple(sum, subtract);
    },
    cuda::std::make_tuple(weight_t{0}, weight_t{0}),
    reduce_op::plus<cuda::std::tuple<weight_t, weight_t>>{},
    vertex_property_view_t{
      thrust::make_zip_iterator(old_cluster_sum_v.begin(), cluster_subtract_v.begin()),
      graph_view.local_vertex_partition_range_size()});

  std::optional<edge_src_property_t<vertex_t, cuda::std::tuple<weight_t, weight_t>>>
    src_old_cluster_sum_subtract_pairs{std::nullopt};

  if constexpr (multi_gpu) {
    src_old_cluster_sum_subtract_pairs =
      edge_src_property_t<vertex_t, cuda::std::tuple<weight_t, weight_t>>(handle, graph_view);
    update_edge_src_property(
      handle,
      graph_view,
      vertex_property_view_t{
        thrust::make_zip_iterator(old_cluster_sum_v.begin(), cluster_subtract_v.begin()),
        graph_view.local_vertex_partition_range_size()},
      src_old_cluster_sum_subtract_pairs->mutable_view());
    old_cluster_sum_v.resize(0, handle.get_stream());
    old_cluster_sum_v.shrink_to_fit(handle.get_stream());
    cluster_subtract_v.resize(0, handle.get_stream());
    cluster_subtract_v.shrink_to_fit(handle.get_stream());
  }

  vertex_property_t<vertex_t, cuda::std::tuple<vertex_t, weight_t>> output(handle, graph_view);

  auto cluster_old_sum_subtract_pair_first =
    thrust::make_zip_iterator(old_cluster_sum_v.cbegin(), cluster_subtract_v.cbegin());
  auto zipped_src_device_view =
    multi_gpu ? view_concat(src_vertex_weights_cache.view(),
                            src_clusters_cache.view(),
                            src_cluster_weights->view(),
                            src_old_cluster_sum_subtract_pairs->view())
              : view_concat(
                  make_edge_src_property_view(graph_view, vertex_weights_v),
                  make_edge_src_property_view(graph_view, next_clusters_v.view()),
                  make_edge_src_property_view(
                    graph_view,
                    vertex_property_view_t{vertex_cluster_weights_v.begin(),
                                           static_cast<vertex_t>(vertex_cluster_weights_v.size())}),
                  make_edge_src_property_view(
                    graph_view,
                    vertex_property_view_t{cluster_old_sum_subtract_pair_first,
                                           static_cast<vertex_t>(old_cluster_sum_v.size())}));

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
              : make_edge_dst_property_view(graph_view, next_clusters_v.view()),
    cluster_key_weight_map.view(),
    detail::key_aggregated_edge_op_t<vertex_t, weight_t>{total_edge_weight, resolution},
    cuda::std::make_tuple(vertex_t{-1}, weight_t{0}),
    detail::reduce_op_t<vertex_t, weight_t>{},
    output.mutable_view());

  auto const min_vertex_move_gain =
    compute_louvain_min_vertex_move_gain(threshold, graph_view.number_of_vertices());

  auto nr_moves =
    count_if_v(handle,
               graph_view,
               view_concat(next_clusters_v.view(), output.view()),
               detail::count_updown_moves_op_t<vertex_t, weight_t>{up_down, min_vertex_move_gain});

  if (nr_moves == 0) { up_down = !up_down; }

  transform_v(handle,
              graph_view,
              view_concat(next_clusters_v.view(), output.view()),
              detail::cluster_update_op_t<vertex_t, weight_t>{up_down, min_vertex_move_gain},
              next_clusters_v.mutable_view());

  return std::move(next_clusters_v);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_property_view_t<vertex_t, vertex_t const*> next_clusters_v,
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
              : make_edge_src_property_view(graph_view, next_clusters_v),
    detail::return_edge_weight_t<vertex_t, weight_t>{},
    weight_t{0},
    reduce_op::plus<weight_t>{});

  return std::make_tuple(std::move(cluster_keys), std::move(cluster_values));
}

template <typename vertex_t, bool multi_gpu>
vertex_t remove_duplicates(raft::handle_t const& handle, rmm::device_uvector<vertex_t>& input_array)
{
  cugraph::sort(handle.get_thrust_policy(), input_array.begin(), input_array.end());

  auto nr_unique_elements = static_cast<vertex_t>(cuda::std::distance(
    input_array.begin(),
    cugraph::unique(handle.get_thrust_policy(), input_array.begin(), input_array.end())));

  input_array.resize(nr_unique_elements, handle.get_stream());

  if constexpr (multi_gpu) {
    std::tie(input_array, std::ignore) = shuffle_ext_vertices(
      handle, std::move(input_array), std::vector<cugraph::arithmetic_device_uvector_t>{});

    cugraph::sort(handle.get_thrust_policy(), input_array.begin(), input_array.end());

    nr_unique_elements = static_cast<vertex_t>(cuda::std::distance(
      input_array.begin(),
      cugraph::unique(handle.get_thrust_policy(), input_array.begin(), input_array.end())));

    input_array.resize(nr_unique_elements, handle.get_stream());

    nr_unique_elements = host_scalar_allreduce(
      handle.get_comms(), nr_unique_elements, raft::comms::op_t::SUM, handle.get_stream());
  }
  return nr_unique_elements;
}

}  // namespace detail
}  // namespace cugraph
