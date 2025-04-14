/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include "maximal_independent_moves.hpp"
#include "prims/per_v_transform_reduce_dst_key_aggregated_outgoing_e.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_e.cuh"
#include "prims/transform_reduce_e_by_src_dst_key.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>

#include <raft/random/rng_device.cuh>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <optional>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)
// FIXME: a temporary workaround for a compiler error, should be deleted once cuco gets patched.
namespace cuco {
template <>
struct is_bitwise_comparable<cuco::pair<int32_t, float>> : std::true_type {};
}  // namespace cuco

namespace cugraph {
namespace detail {

// FIXME: check if this is still the case
//  a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t, typename cluster_value_t>
struct leiden_key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t resolution{};  // resolution parameter
  weight_t theta{};       // scaling factor
  raft::random::DeviceState<raft::random::PCGenerator>& device_state;
  __device__ auto operator()(
    vertex_t src,
    vertex_t neighboring_leiden_cluster,
    thrust::tuple<weight_t, weight_t, weight_t, uint8_t, vertex_t, vertex_t> src_info,
    cluster_value_t keyed_data,
    weight_t aggregated_weight_to_neighboring_leiden_cluster) const
  {
    // Data associated with src vertex
    auto src_weighted_deg          = thrust::get<0>(src_info);
    auto src_vertex_cut_to_louvain = thrust::get<1>(src_info);
    auto louvain_cluster_volume    = thrust::get<2>(src_info);
    auto is_src_active             = thrust::get<3>(src_info);
    auto src_leiden_cluster        = thrust::get<4>(src_info);
    auto src_louvain_cluster       = thrust::get<5>(src_info);

    // Data associated with target leiden (aka refined) cluster

    auto dst_leiden_volume             = thrust::get<0>(keyed_data);
    auto dst_leiden_cut_to_louvain     = thrust::get<1>(keyed_data);
    auto dst_leiden_cluster_id         = thrust::get<2>(keyed_data);
    auto louvain_of_dst_leiden_cluster = thrust::get<3>(keyed_data);

    // E(Cr, S-Cr) > ||Cr||*(||S|| -||Cr||)
    bool is_dst_leiden_cluster_well_connected =
      dst_leiden_cut_to_louvain > resolution * dst_leiden_volume *
                                    (louvain_cluster_volume - dst_leiden_volume) /
                                    total_edge_weight;

    // E(v, Cr-v) - ||v||* ||Cr-v||/||V(G)||
    // aggregated_weight_to_neighboring_leiden_cluster == E(v, Cr-v)?

    weight_t mod_gain = -1.0;
    if (is_src_active > 0) {
      if ((louvain_of_dst_leiden_cluster == src_louvain_cluster) &&
          (dst_leiden_cluster_id != src_leiden_cluster) && is_dst_leiden_cluster_well_connected) {
        mod_gain = aggregated_weight_to_neighboring_leiden_cluster -
                   resolution * src_weighted_deg * dst_leiden_volume / total_edge_weight;
// FIXME: Disable random moves in refinement phase for now.
#if 0
        weight_t random_number{0.0};
        if (mod_gain > 0.0) {
          auto flat_id = uint64_t{threadIdx.x + blockIdx.x * blockDim.x};
          raft::random::PCGenerator gen(device_state, flat_id);
          raft::random::UniformDistParams<weight_t> int_params{};
          int_params.start = weight_t{0.0};
          int_params.end   = weight_t{1.0};
          raft::random::custom_next(gen, &random_number, int_params, 0, 0);
        }

        mod_gain = mod_gain > 0.0
                     ? __expf(static_cast<float>((2.0 * mod_gain) / (theta * total_edge_weight))) *
                         random_number
                     : -1.0;
#endif
        mod_gain = mod_gain > 0.0 ? mod_gain : -1.0;
      }
    }

    return thrust::make_tuple(mod_gain, neighboring_leiden_cluster);
  }
};

template <typename GraphViewType, typename weight_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::pair<rmm::device_uvector<typename GraphViewType::vertex_type>,
                     rmm::device_uvector<typename GraphViewType::vertex_type>>>
refine_clustering(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  GraphViewType const& graph_view,
  std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
    edge_weight_view,
  weight_t total_edge_weight,
  weight_t resolution,
  weight_t theta,
  rmm::device_uvector<weight_t> const& weighted_degree_of_vertices,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& louvain_cluster_keys,
  rmm::device_uvector<weight_t>&& louvain_cluster_weights,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& louvain_assignment_of_vertices,
  edge_src_property_t<GraphViewType, weight_t> const& src_vertex_weights_cache,
  edge_src_property_t<GraphViewType, typename GraphViewType::vertex_type> const&
    src_louvain_assignment_cache,
  edge_dst_property_t<GraphViewType, typename GraphViewType::vertex_type> const&
    dst_louvain_assignment_cache)
{
  const weight_t POSITIVE_GAIN = 1e-6;
  using vertex_t               = typename GraphViewType::vertex_type;
  using edge_t                 = typename GraphViewType::edge_type;

  kv_store_t<vertex_t, weight_t, false> cluster_key_weight_map(louvain_cluster_keys.begin(),
                                                               louvain_cluster_keys.end(),
                                                               louvain_cluster_weights.data(),
                                                               invalid_vertex_id<vertex_t>::value,
                                                               std::numeric_limits<weight_t>::max(),
                                                               handle.get_stream());
  louvain_cluster_keys.resize(0, handle.get_stream());
  louvain_cluster_keys.shrink_to_fit(handle.get_stream());

  louvain_cluster_weights.resize(0, handle.get_stream());
  louvain_cluster_weights.shrink_to_fit(handle.get_stream());

  rmm::device_uvector<weight_t> vertex_louvain_cluster_weights(0, handle.get_stream());
  if (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      comm_size, major_comm_size, minor_comm_size};

    vertex_louvain_cluster_weights =
      cugraph::collect_values_for_keys(comm,
                                       cluster_key_weight_map.view(),
                                       louvain_assignment_of_vertices.begin(),
                                       louvain_assignment_of_vertices.end(),
                                       vertex_to_gpu_id_op,
                                       handle.get_stream());

  } else {
    vertex_louvain_cluster_weights.resize(louvain_assignment_of_vertices.size(),
                                          handle.get_stream());

    cluster_key_weight_map.view().find(louvain_assignment_of_vertices.begin(),
                                       louvain_assignment_of_vertices.end(),
                                       vertex_louvain_cluster_weights.begin(),
                                       handle.get_stream());
  }
  //
  // For each vertex, compute its weighted degree (||v||)
  // and cut between itself and its Louvain community (E(v, S-v))
  //

  rmm::device_uvector<weight_t> weighted_cut_of_vertices_to_louvain(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  per_v_transform_reduce_outgoing_e(
    handle,
    graph_view,
    GraphViewType::is_multi_gpu ? src_louvain_assignment_cache.view()
                                : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
                                    louvain_assignment_of_vertices.data()),
    GraphViewType::is_multi_gpu ? dst_louvain_assignment_cache.view()
                                : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                    louvain_assignment_of_vertices.data(), vertex_t{0}),
    *edge_weight_view,
    cuda::proclaim_return_type<weight_t>(
      [] __device__(auto src, auto dst, auto src_cluster, auto dst_cluster, auto wt) {
        weight_t weighted_cut_contribution{0};

        if (src == dst)  // self loop
          weighted_cut_contribution = 0;
        else if (src_cluster == dst_cluster)
          weighted_cut_contribution = wt;

        return weighted_cut_contribution;
      }),
    weight_t{0},
    cugraph::reduce_op::plus<weight_t>{},
    weighted_cut_of_vertices_to_louvain.begin());

  // FIXME: Consider using bit mask logic here.  Would reduce memory by 8x
  rmm::device_uvector<uint8_t> singleton_and_connected_flags(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  auto wcut_deg_and_cluster_vol_triple_begin =
    thrust::make_zip_iterator(thrust::make_tuple(weighted_cut_of_vertices_to_louvain.begin(),
                                                 weighted_degree_of_vertices.begin(),
                                                 vertex_louvain_cluster_weights.begin()));
  auto wcut_deg_and_cluster_vol_triple_end =
    thrust::make_zip_iterator(thrust::make_tuple(weighted_cut_of_vertices_to_louvain.end(),
                                                 weighted_degree_of_vertices.end(),
                                                 vertex_louvain_cluster_weights.end()));

  thrust::transform(handle.get_thrust_policy(),
                    wcut_deg_and_cluster_vol_triple_begin,
                    wcut_deg_and_cluster_vol_triple_end,
                    singleton_and_connected_flags.begin(),
                    cuda::proclaim_return_type<uint8_t>([resolution, total_edge_weight] __device__(
                                                          auto wcut_wdeg_and_louvain_volume) {
                      auto wcut           = thrust::get<0>(wcut_wdeg_and_louvain_volume);
                      auto wdeg           = thrust::get<1>(wcut_wdeg_and_louvain_volume);
                      auto louvain_volume = thrust::get<2>(wcut_wdeg_and_louvain_volume);
                      return static_cast<uint8_t>(
                        wcut > (resolution * wdeg * (louvain_volume - wdeg) / total_edge_weight));
                    }));

  edge_src_property_t<GraphViewType, weight_t> src_louvain_cluster_weight_cache(handle);
  edge_src_property_t<GraphViewType, weight_t> src_cut_to_louvain_cache(handle);

  if (GraphViewType::is_multi_gpu) {
    // Update cluster weight, weighted degree and cut for edge sources
    src_louvain_cluster_weight_cache =
      edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
    update_edge_src_property(handle,
                             graph_view,
                             vertex_louvain_cluster_weights.begin(),
                             src_louvain_cluster_weight_cache.mutable_view());

    src_cut_to_louvain_cache = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
    update_edge_src_property(handle,
                             graph_view,
                             weighted_cut_of_vertices_to_louvain.begin(),
                             src_cut_to_louvain_cache.mutable_view());

    vertex_louvain_cluster_weights.resize(0, handle.get_stream());
    vertex_louvain_cluster_weights.shrink_to_fit(handle.get_stream());

    weighted_cut_of_vertices_to_louvain.resize(0, handle.get_stream());
    weighted_cut_of_vertices_to_louvain.shrink_to_fit(handle.get_stream());
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

  edge_src_property_t<GraphViewType, vertex_t> src_leiden_assignment_cache(handle);
  edge_dst_property_t<GraphViewType, vertex_t> dst_leiden_assignment_cache(handle);
  edge_src_property_t<GraphViewType, uint8_t> src_singleton_and_connected_flag_cache(handle);

  // FIXME:  Why is kvstore used here?  Can't this be accomplished by
  //  a direct lookup in louvain_assignment_of_vertices using
  //     leiden - graph_view.local_vertex_partition_range_first() as the
  //     index?
  // Changing this would save memory and time
  kv_store_t<vertex_t, vertex_t, false> leiden_to_louvain_map(
    leiden_assignment.begin(),
    leiden_assignment.end(),
    louvain_assignment_of_vertices.begin(),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value,
    handle.get_stream());

  while (true) {
    vertex_t nr_remaining_active_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       singleton_and_connected_flags.begin(),
                       singleton_and_connected_flags.end(),
                       [] __device__(auto flag) { return flag > 0; });

    if (GraphViewType::is_multi_gpu) {
      nr_remaining_active_vertices = host_scalar_allreduce(handle.get_comms(),
                                                           nr_remaining_active_vertices,
                                                           raft::comms::op_t::SUM,
                                                           handle.get_stream());
    }

    if (nr_remaining_active_vertices == 0) { break; }

    // Update Leiden assignment to edge sources and destinitions
    // and singleton mask to edge sources

    if constexpr (GraphViewType::is_multi_gpu) {
      src_leiden_assignment_cache =
        edge_src_property_t<GraphViewType, vertex_t>(handle, graph_view);
      dst_leiden_assignment_cache =
        edge_dst_property_t<GraphViewType, vertex_t>(handle, graph_view);
      src_singleton_and_connected_flag_cache =
        edge_src_property_t<GraphViewType, uint8_t>(handle, graph_view);

      update_edge_src_property(
        handle, graph_view, leiden_assignment.begin(), src_leiden_assignment_cache.mutable_view());

      update_edge_dst_property(
        handle, graph_view, leiden_assignment.begin(), dst_leiden_assignment_cache.mutable_view());

      update_edge_src_property(handle,
                               graph_view,
                               singleton_and_connected_flags.begin(),
                               src_singleton_and_connected_flag_cache.mutable_view());
    }

    auto src_input_property_values =
      GraphViewType::is_multi_gpu
        ? view_concat(src_louvain_assignment_cache.view(), src_leiden_assignment_cache.view())
        : view_concat(detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
                        louvain_assignment_of_vertices.data()),
                      detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
                        leiden_assignment.data()));

    auto dst_input_property_values =
      GraphViewType::is_multi_gpu
        ? view_concat(dst_louvain_assignment_cache.view(), dst_leiden_assignment_cache.view())
        : view_concat(detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                        louvain_assignment_of_vertices.data(), vertex_t{0}),
                      detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                        leiden_assignment.data(), vertex_t{0}));

    rmm::device_uvector<vertex_t> leiden_keys_used_in_edge_reduction(0, handle.get_stream());
    rmm::device_uvector<weight_t> refined_community_volumes(0, handle.get_stream());
    rmm::device_uvector<weight_t> refined_community_cuts(0, handle.get_stream());

    //
    // For each refined community, compute its volume
    // (i.e.sum of weighted degree of all vertices inside it, ||Cr||) and
    // and cut between itself and its Louvain community (E(Cr, S-Cr))
    //
    // FIXME: Can we update ||Cr|| and E(Cr, S-Cr) instead of recomputing?

    std::forward_as_tuple(leiden_keys_used_in_edge_reduction,
                          std::tie(refined_community_volumes, refined_community_cuts)) =
      cugraph::transform_reduce_e_by_dst_key(
        handle,
        graph_view,
        src_input_property_values,
        dst_input_property_values,
        *edge_weight_view,
        GraphViewType::is_multi_gpu ? dst_leiden_assignment_cache.view()
                                    : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                        leiden_assignment.data(), vertex_t{0}),

        [] __device__(auto src,
                      auto dst,
                      thrust::tuple<vertex_t, vertex_t> src_louvain_leidn,
                      thrust::tuple<vertex_t, vertex_t> dst_louvain_leiden,
                      auto wt) {
          weight_t refined_partition_volume_contribution{wt};
          weight_t refined_partition_cut_contribution{0};

          auto src_louvain = thrust::get<0>(src_louvain_leidn);
          auto src_leiden  = thrust::get<1>(src_louvain_leidn);

          auto dst_louvain = thrust::get<0>(dst_louvain_leiden);
          auto dst_leiden  = thrust::get<1>(dst_louvain_leiden);

          if (src_louvain == dst_louvain) {
            if (src_leiden != dst_leiden) { refined_partition_cut_contribution = wt; }
          }
          return thrust::make_tuple(refined_partition_volume_contribution,
                                    refined_partition_cut_contribution);
        },
        thrust::make_tuple(weight_t{0}, weight_t{0}),
        reduce_op::plus<thrust::tuple<weight_t, weight_t>>{});

    //
    // Primitives to decide best (at least good) next clusters for vertices
    //

    // ||v||
    // E(v, louvain(v))
    // ||louvain(v)||
    // is_singleton_and_connected(v)
    // leiden(v)
    // louvain(v)

    auto zipped_src_device_view =
      GraphViewType::is_multi_gpu
        ? view_concat(src_vertex_weights_cache.view(),
                      src_cut_to_louvain_cache.view(),
                      src_louvain_cluster_weight_cache.view(),
                      src_singleton_and_connected_flag_cache.view(),
                      src_leiden_assignment_cache.view(),
                      src_louvain_assignment_cache.view())
        : view_concat(
            detail::edge_major_property_view_t<vertex_t, weight_t const*>(
              weighted_degree_of_vertices.data()),
            detail::edge_major_property_view_t<vertex_t, weight_t const*>(
              weighted_cut_of_vertices_to_louvain.data()),
            detail::edge_major_property_view_t<vertex_t, weight_t const*>(
              vertex_louvain_cluster_weights.data()),
            detail::edge_major_property_view_t<vertex_t, uint8_t const*>(
              singleton_and_connected_flags.data()),
            detail::edge_major_property_view_t<vertex_t, vertex_t const*>(leiden_assignment.data()),
            detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
              louvain_assignment_of_vertices.data()));

    rmm::device_uvector<vertex_t> louvain_of_leiden_keys_used_in_edge_reduction(
      0, handle.get_stream());

    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
      rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
        vertex_partition_range_lasts.size(), handle.get_stream());

      raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream());

      cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> vertex_to_gpu_id_op{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size};

      // cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
      //   comm_size, major_comm_size, minor_comm_size};

      louvain_of_leiden_keys_used_in_edge_reduction =
        cugraph::collect_values_for_keys(comm,
                                         leiden_to_louvain_map.view(),
                                         leiden_keys_used_in_edge_reduction.begin(),
                                         leiden_keys_used_in_edge_reduction.end(),
                                         vertex_to_gpu_id_op,
                                         handle.get_stream());
    } else {
      louvain_of_leiden_keys_used_in_edge_reduction.resize(
        leiden_keys_used_in_edge_reduction.size(), handle.get_stream());

      leiden_to_louvain_map.view().find(leiden_keys_used_in_edge_reduction.begin(),
                                        leiden_keys_used_in_edge_reduction.end(),
                                        louvain_of_leiden_keys_used_in_edge_reduction.begin(),
                                        handle.get_stream());
    }

    // ||Cr|| //f(Cr)
    // E(Cr, louvain(v) - Cr) //f(Cr)
    // leiden(Cr) // f(Cr)
    // louvain(Cr) // f(Cr)
    auto values_for_leiden_cluster_keys = thrust::make_zip_iterator(
      thrust::make_tuple(refined_community_volumes.begin(),
                         refined_community_cuts.begin(),
                         leiden_keys_used_in_edge_reduction.begin(),
                         louvain_of_leiden_keys_used_in_edge_reduction.begin()));

    using value_t = thrust::tuple<weight_t, weight_t, vertex_t, vertex_t>;
    kv_store_t<vertex_t, value_t, true> leiden_cluster_key_values_map(
      leiden_keys_used_in_edge_reduction.begin(),
      leiden_keys_used_in_edge_reduction.begin() + leiden_keys_used_in_edge_reduction.size(),
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

    raft::random::DeviceState<raft::random::PCGenerator> device_state(rng_state);
    auto gain_and_dst_output_pairs = allocate_dataframe_buffer<thrust::tuple<weight_t, vertex_t>>(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_dst_key_aggregated_outgoing_e(
      handle,
      graph_view,
      zipped_src_device_view,
      *edge_weight_view,
      GraphViewType::is_multi_gpu ? dst_leiden_assignment_cache.view()
                                  : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                      leiden_assignment.data(), vertex_t{0}),
      leiden_cluster_key_values_map.view(),
      detail::leiden_key_aggregated_edge_op_t<vertex_t, weight_t, value_t>{
        total_edge_weight, resolution, theta, device_state},
      thrust::make_tuple(weight_t{0}, vertex_t{-1}),
      reduce_op::maximum<thrust::tuple<weight_t, vertex_t>>(),
      cugraph::get_dataframe_buffer_begin(gain_and_dst_output_pairs));

    src_leiden_assignment_cache.clear(handle);
    dst_leiden_assignment_cache.clear(handle);
    src_singleton_and_connected_flag_cache.clear(handle);

    louvain_of_leiden_keys_used_in_edge_reduction.resize(0, handle.get_stream());
    louvain_of_leiden_keys_used_in_edge_reduction.shrink_to_fit(handle.get_stream());
    leiden_keys_used_in_edge_reduction.resize(0, handle.get_stream());
    leiden_keys_used_in_edge_reduction.shrink_to_fit(handle.get_stream());
    refined_community_volumes.resize(0, handle.get_stream());
    refined_community_volumes.shrink_to_fit(handle.get_stream());
    refined_community_cuts.resize(0, handle.get_stream());
    refined_community_cuts.shrink_to_fit(handle.get_stream());

    //
    // Create edgelist from (source, target community, modulraity gain) tuple
    //

    vertex_t num_vertices   = graph_view.local_vertex_partition_range_size();
    auto gain_and_dst_first = cugraph::get_dataframe_buffer_cbegin(gain_and_dst_output_pairs);
    auto gain_and_dst_last  = cugraph::get_dataframe_buffer_cend(gain_and_dst_output_pairs);

    auto vertex_begin =
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first());
    auto vertex_end =
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last());

    //
    // Filter out moves with -ve gains
    //

    vertex_t nr_valid_tuples = thrust::count_if(handle.get_thrust_policy(),
                                                gain_and_dst_first,
                                                gain_and_dst_last,
                                                [] __device__(auto gain_dst_pair) {
                                                  vertex_t dst  = thrust::get<1>(gain_dst_pair);
                                                  weight_t gain = thrust::get<0>(gain_dst_pair);
                                                  return (gain > POSITIVE_GAIN) && (dst >= 0);
                                                });

    vertex_t total_nr_valid_tuples = nr_valid_tuples;
    if (GraphViewType::is_multi_gpu) {
      total_nr_valid_tuples = host_scalar_allreduce(
        handle.get_comms(), total_nr_valid_tuples, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (total_nr_valid_tuples == 0) {
      cugraph::resize_dataframe_buffer(gain_and_dst_output_pairs, 0, handle.get_stream());
      cugraph::shrink_to_fit_dataframe_buffer(gain_and_dst_output_pairs, handle.get_stream());
      break;
    }

    rmm::device_uvector<vertex_t> d_srcs(nr_valid_tuples, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dsts(nr_valid_tuples, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> d_weights =
      std::make_optional(rmm::device_uvector<weight_t>(nr_valid_tuples, handle.get_stream()));

    auto d_src_dst_gain_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(d_srcs.begin(), d_dsts.begin(), (*d_weights).begin()));

    // edge (src, dst, gain)
    auto edge_begin = thrust::make_zip_iterator(
      thrust::make_tuple(vertex_begin,
                         thrust::get<1>(gain_and_dst_first.get_iterator_tuple()),
                         thrust::get<0>(gain_and_dst_first.get_iterator_tuple())));
    auto edge_end = thrust::make_zip_iterator(
      thrust::make_tuple(vertex_end,
                         thrust::get<1>(gain_and_dst_last.get_iterator_tuple()),
                         thrust::get<0>(gain_and_dst_last.get_iterator_tuple())));

    thrust::copy_if(handle.get_thrust_policy(),
                    edge_begin,
                    edge_end,
                    d_src_dst_gain_iterator,
                    [] __device__(thrust::tuple<vertex_t, vertex_t, weight_t> src_dst_gain) {
                      vertex_t src  = thrust::get<0>(src_dst_gain);
                      vertex_t dst  = thrust::get<1>(src_dst_gain);
                      weight_t gain = thrust::get<2>(src_dst_gain);

                      return (gain > POSITIVE_GAIN) && (dst >= 0);
                    });

    //
    // Create decision graph from edgelist
    //
    constexpr bool store_transposed = false;
    constexpr bool multi_gpu        = GraphViewType::is_multi_gpu;
    using DecisionGraphViewType     = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> decision_graph(handle);

    std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
    std::optional<edge_property_t<DecisionGraphViewType, weight_t>> coarse_edge_weights{
      std::nullopt};

    if constexpr (multi_gpu) {
      std::tie(d_srcs,
               d_dsts,
               d_weights,
               std::ignore,
               std::ignore,
               std::ignore,
               std::ignore,
               std::ignore) =
        cugraph::shuffle_ext_edges<vertex_t, vertex_t, weight_t, int32_t, int32_t>(
          handle,
          std::move(d_srcs),
          std::move(d_dsts),
          std::move(d_weights),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          GraphViewType::is_storage_transposed);
    }

    std::tie(decision_graph, coarse_edge_weights, std::ignore, std::ignore, renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
        handle,
        std::nullopt,
        std::move(d_srcs),
        std::move(d_dsts),
        std::move(d_weights),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{false, false},
        true,
        false);

    auto decision_graph_view = decision_graph.view();

    //
    // Determine a set of moves using MIS of the decision_graph
    //

    auto vertices_in_mis = maximal_independent_moves<vertex_t, edge_t, multi_gpu>(
      handle, decision_graph_view, rng_state);

    rmm::device_uvector<vertex_t> numbering_indices((*renumber_map).size(), handle.get_stream());
    detail::sequence_fill(handle.get_stream(),
                          numbering_indices.data(),
                          numbering_indices.size(),
                          decision_graph_view.local_vertex_partition_range_first());

    //
    // Apply Renumber map to get original vertex ids
    //
    relabel<vertex_t, multi_gpu>(
      handle,
      std::make_tuple(static_cast<vertex_t const*>(numbering_indices.begin()),
                      static_cast<vertex_t const*>((*renumber_map).begin())),
      decision_graph_view.local_vertex_partition_range_size(),
      vertices_in_mis.data(),
      vertices_in_mis.size(),
      false);

    numbering_indices.resize(0, handle.get_stream());
    numbering_indices.shrink_to_fit(handle.get_stream());

    (*renumber_map).resize(0, handle.get_stream());
    (*renumber_map).shrink_to_fit(handle.get_stream());

    if (GraphViewType::is_multi_gpu) {
      vertices_in_mis = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(vertices_in_mis), graph_view.vertex_partition_range_lasts());
    }

    //
    // Mark the chosen vertices as non-singleton and update their leiden cluster to dst
    //

    thrust::for_each(
      handle.get_thrust_policy(),
      vertices_in_mis.begin(),
      vertices_in_mis.end(),
      [dst_first                     = thrust::get<1>(gain_and_dst_first.get_iterator_tuple()),
       leiden_assignment             = leiden_assignment.data(),
       singleton_and_connected_flags = singleton_and_connected_flags.data(),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
        auto v_offset                           = v - v_first;
        auto dst                                = *(dst_first + v_offset);
        singleton_and_connected_flags[v_offset] = false;
        leiden_assignment[v_offset]             = dst;
      });

    //
    // Find the set of dest vertices
    //
    rmm::device_uvector<vertex_t> dst_vertices(vertices_in_mis.size(), handle.get_stream());

    thrust::transform(
      handle.get_thrust_policy(),
      vertices_in_mis.begin(),
      vertices_in_mis.end(),
      dst_vertices.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [dst_first = thrust::get<1>(gain_and_dst_first.get_iterator_tuple()),
         v_first   = graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
          auto dst = *(dst_first + v - v_first);
          return dst;
        }));

    cugraph::resize_dataframe_buffer(gain_and_dst_output_pairs, 0, handle.get_stream());
    cugraph::shrink_to_fit_dataframe_buffer(gain_and_dst_output_pairs, handle.get_stream());

    vertices_in_mis.resize(0, handle.get_stream());
    vertices_in_mis.shrink_to_fit(handle.get_stream());

    thrust::sort(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end());

    dst_vertices.resize(
      static_cast<size_t>(cuda::std::distance(
        dst_vertices.begin(),
        thrust::unique(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end()))),
      handle.get_stream());

    // Shuffle dst vertices to owner GPU, according to vetex partitioning
    if constexpr (GraphViewType::is_multi_gpu) {
      dst_vertices = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(dst_vertices), graph_view.vertex_partition_range_lasts());

      thrust::sort(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end());

      dst_vertices.resize(
        static_cast<size_t>(cuda::std::distance(
          dst_vertices.begin(),
          thrust::unique(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end()))),
        handle.get_stream());
    }

    //
    // Makr all the dest vertices as non-sigleton
    //
    thrust::for_each(
      handle.get_thrust_policy(),
      dst_vertices.begin(),
      dst_vertices.end(),
      [singleton_and_connected_flags = singleton_and_connected_flags.data(),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
        singleton_and_connected_flags[v - v_first] = false;
      });

    dst_vertices.resize(0, handle.get_stream());
    dst_vertices.shrink_to_fit(handle.get_stream());
  }

  src_louvain_cluster_weight_cache.clear(handle);
  src_cut_to_louvain_cache.clear(handle);

  singleton_and_connected_flags.resize(0, handle.get_stream());
  singleton_and_connected_flags.shrink_to_fit(handle.get_stream());
  vertex_louvain_cluster_weights.resize(0, handle.get_stream());
  vertex_louvain_cluster_weights.shrink_to_fit(handle.get_stream());
  weighted_cut_of_vertices_to_louvain.resize(0, handle.get_stream());
  weighted_cut_of_vertices_to_louvain.shrink_to_fit(handle.get_stream());

  //
  // Re-read Leiden to Louvain map, but for remaining (after moving) Leiden communities
  //
  rmm::device_uvector<vertex_t> leiden_keys_to_read_louvain(leiden_assignment.size(),
                                                            handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               leiden_assignment.begin(),
               leiden_assignment.end(),
               leiden_keys_to_read_louvain.begin());

  thrust::sort(handle.get_thrust_policy(),
               leiden_keys_to_read_louvain.begin(),
               leiden_keys_to_read_louvain.end());

  auto nr_unique_leiden_clusters =
    static_cast<size_t>(cuda::std::distance(leiden_keys_to_read_louvain.begin(),
                                            thrust::unique(handle.get_thrust_policy(),
                                                           leiden_keys_to_read_louvain.begin(),
                                                           leiden_keys_to_read_louvain.end())));

  leiden_keys_to_read_louvain.resize(nr_unique_leiden_clusters, handle.get_stream());

  if constexpr (GraphViewType::is_multi_gpu) {
    leiden_keys_to_read_louvain =
      cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(leiden_keys_to_read_louvain), graph_view.vertex_partition_range_lasts());

    thrust::sort(handle.get_thrust_policy(),
                 leiden_keys_to_read_louvain.begin(),
                 leiden_keys_to_read_louvain.end());

    nr_unique_leiden_clusters =
      static_cast<size_t>(cuda::std::distance(leiden_keys_to_read_louvain.begin(),
                                              thrust::unique(handle.get_thrust_policy(),
                                                             leiden_keys_to_read_louvain.begin(),
                                                             leiden_keys_to_read_louvain.end())));
    leiden_keys_to_read_louvain.resize(nr_unique_leiden_clusters, handle.get_stream());
  }

  rmm::device_uvector<vertex_t> lovain_of_leiden_cluster_keys(0, handle.get_stream());

  if (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
    rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
      vertex_partition_range_lasts.size(), handle.get_stream());

    raft::update_device(d_vertex_partition_range_lasts.data(),
                        vertex_partition_range_lasts.data(),
                        vertex_partition_range_lasts.size(),
                        handle.get_stream());

    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> vertex_to_gpu_id_op{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size};

    // cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
    //   comm_size, major_comm_size, minor_comm_size};

    lovain_of_leiden_cluster_keys =
      cugraph::collect_values_for_keys(comm,
                                       leiden_to_louvain_map.view(),
                                       leiden_keys_to_read_louvain.begin(),
                                       leiden_keys_to_read_louvain.end(),
                                       vertex_to_gpu_id_op,
                                       handle.get_stream());

  } else {
    lovain_of_leiden_cluster_keys.resize(leiden_keys_to_read_louvain.size(), handle.get_stream());

    leiden_to_louvain_map.view().find(leiden_keys_to_read_louvain.begin(),
                                      leiden_keys_to_read_louvain.end(),
                                      lovain_of_leiden_cluster_keys.begin(),
                                      handle.get_stream());
  }
  return std::make_tuple(std::move(leiden_assignment),
                         std::make_pair(std::move(leiden_keys_to_read_louvain),
                                        std::move(lovain_of_leiden_cluster_keys)));
}
}  // namespace detail
}  // namespace cugraph
