/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <detail/graph_partition_utils.cuh>
#include <prims/per_v_transform_reduce_dst_key_aggregated_outgoing_e.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_e.cuh>
#include <prims/transform_reduce_e_by_src_dst_key.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <utilities/collect_comm.cuh>

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

#include <algorithm>
#include <cmath>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)

namespace cugraph {
namespace detail {

// FIXME: check if this is still the case
//  a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t, typename cluster_value_t>
struct leiden_key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t gamma{};
  bool debug{};
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
      dst_leiden_cut_to_louvain >
      gamma * dst_leiden_volume * (louvain_cluster_volume - dst_leiden_volume);

    // E(v, Cr-v) - ||v||* ||Cr-v||/||V(G)||
    // aggregated_weight_to_neighboring_leiden_cluster == E(v, Cr-v)?

    weight_t theta = -1.0;
    // if ((is_src_active > 0) && is_src_well_connected) {
    if (is_src_active > 0) {
      if ((louvain_of_dst_leiden_cluster == src_louvain_cluster) &&
          is_dst_leiden_cluster_well_connected) {
        theta = aggregated_weight_to_neighboring_leiden_cluster -
                gamma * src_weighted_deg * dst_leiden_volume / total_edge_weight;
      
      int src_id = static_cast<int>(src);
      int dl_cid = static_cast<int>(neighboring_leiden_cluster);
      int cut_to_dl = static_cast<int>(aggregated_weight_to_neighboring_leiden_cluster);
      int s_wdeg = static_cast<int>(src_weighted_deg);
      int dl_vol = static_cast<int>(dst_leiden_volume);
      int tew = static_cast<int>(total_edge_weight);
      float ftheta = static_cast<float>(theta);

      if(neighboring_leiden_cluster!= dst_leiden_cluster_id){
        printf ("\n BUG \n");
      }
      if(debug) printf("\ntotal_weight = %f  total_weight = %d \n", total_edge_weight, tew);
      if(debug) printf("\ndst_leiden = %d  vol(dst_leiden)=%d  \n", dl_cid,  dl_vol);
      if(debug) printf("\n return dst_leiden = %d  theta=%f  \n", dl_cid, ftheta);

      if(debug) printf("\nsrc = %d  dst_leiden = %d cut(src, dst_leiden) = %d wdeg(src)=%d vol(dst_leiden)=%d total_weight=%d \n",
       src_id, dl_cid, cut_to_dl, s_wdeg, dl_vol, tew);
      }
    }

    return thrust::make_tuple(theta, neighboring_leiden_cluster);
  }
};

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
  rmm::device_uvector<weight_t> const& weighted_degree_of_vertices,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& louvain_cluster_keys,
  rmm::device_uvector<weight_t>&& louvain_cluster_weights,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& louvain_assignment_of_vertices,
  edge_src_property_t<GraphViewType, weight_t> const& src_vertex_weights_cache,
  edge_src_property_t<GraphViewType, typename GraphViewType::vertex_type> const&
    src_louvain_assignment_cache,
  edge_dst_property_t<GraphViewType, typename GraphViewType::vertex_type> const&
    dst_louvain_assignment_cache,
  bool up_down)
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

  bool debug = false;//graph_view.number_of_vertices() < 50;

  if(debug) std::cout << (debug? " True": "False") << std::endl;
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
      cugraph::collect_values_for_keys(handle,
                                       cluster_key_weight_map.view(),
                                       louvain_assignment_of_vertices.begin(),
                                       louvain_assignment_of_vertices.end(),
                                       vertex_to_gpu_id_op);
  #if 1
    auto const comm_rank = comm.get_rank();
    for (int i = 0; i < comm_size; ++i) {
      handle.get_comms().barrier();
      if (comm_rank == i) {
        if (comm_rank == 0) {
          if(debug) std::cout << "---------------------------------------------" << std::endl;
        }
        if(debug) std::cout << "Rank: " << i << std::endl;
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) std::cout << (debug? " True": "False") << std::endl;
        if(debug) raft::print_device_vector("louvain_assignment_of_vertices",
                                  louvain_assignment_of_vertices.data(),
                                  louvain_assignment_of_vertices.size(),
                                  std::cout);
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("vertex_louvain_cluster_weights",
                                  vertex_louvain_cluster_weights.data(),
                                  vertex_louvain_cluster_weights.size(),
                                  std::cout);
      }
      handle.get_comms().barrier();
    }
  #endif

  } else {
    vertex_louvain_cluster_weights.resize(louvain_assignment_of_vertices.size(),
                                          handle.get_stream());

    cluster_key_weight_map.view().find(louvain_assignment_of_vertices.begin(),
                                       louvain_assignment_of_vertices.end(),
                                       vertex_louvain_cluster_weights.begin(),
                                       handle.get_stream());

  if(debug) std::cout << (debug? " True": "False") << std::endl;
  #if 1
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("louvain_assignment_of_vertices",
                              louvain_assignment_of_vertices.data(),
                              louvain_assignment_of_vertices.size(),
                              std::cout);
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("vertex_louvain_cluster_weights",
                              vertex_louvain_cluster_weights.data(),
                              vertex_louvain_cluster_weights.size(),
                              std::cout);
  #endif
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
    [] __device__(auto src, auto dst, auto src_cluster, auto dst_cluster, auto wt) {
      weight_t weighted_cut_contribution{0};

      if (src == dst)  // self loop
        weighted_cut_contribution = 0;
      else if (src_cluster == dst_cluster)
        weighted_cut_contribution = wt;

      return weighted_cut_contribution;
    },
    weight_t{0},
    cugraph::reduce_op::plus<weight_t>{},
    weighted_cut_of_vertices_to_louvain.begin());

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

  if(debug) std::cout << "Before calling transform to compute  singleton_and_connected_flags" << std::endl;
  thrust::transform(handle.get_thrust_policy(),
                    wcut_deg_and_cluster_vol_triple_begin,
                    wcut_deg_and_cluster_vol_triple_end,
                    singleton_and_connected_flags.begin(),
                    [gamma = resolution] __device__(auto wcut_wdeg_and_louvain_volume) {
                      auto wcut           = thrust::get<0>(wcut_wdeg_and_louvain_volume);
                      auto wdeg           = thrust::get<1>(wcut_wdeg_and_louvain_volume);
                      auto louvain_volume = thrust::get<2>(wcut_wdeg_and_louvain_volume);
                      return wcut > (gamma * wdeg * (louvain_volume - wdeg));
                    });


#if 1                    
  if (GraphViewType::is_multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();

    for (int k = 0; k < comm_size; k++) {
      comm.barrier();
      if (comm_rank == k) {
        if (comm_rank == 0) {
          if(debug) std::cout << "---------------------------------------------" << std::endl;
        }

        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("weighted_cut_of_vertices_to_louvain",
                                  weighted_cut_of_vertices_to_louvain.data(),
                                  weighted_cut_of_vertices_to_louvain.size(),
                                  std::cout);
      
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("weighted_degree_of_vertices",
                                  weighted_degree_of_vertices.data(),
                                  weighted_degree_of_vertices.size(),
                                  std::cout);
      
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("vertex_louvain_cluster_weights",
                                  vertex_louvain_cluster_weights.data(),
                                  vertex_louvain_cluster_weights.size(),
                                  std::cout);
      

        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) std::cout << "Rank: " << comm_rank << std::endl;

        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("singleton_and_connected_flags",
                                  singleton_and_connected_flags.data(),
                                  singleton_and_connected_flags.size(),
                                  std::cout);
      

      }
      comm.barrier();
    }
  } else {
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("weighted_cut_of_vertices_to_louvain",
                              weighted_cut_of_vertices_to_louvain.data(),
                              weighted_cut_of_vertices_to_louvain.size(),
                              std::cout);
  
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("weighted_degree_of_vertices",
                              weighted_degree_of_vertices.data(),
                              weighted_degree_of_vertices.size(),
                              std::cout);
  
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("vertex_louvain_cluster_weights",
                              vertex_louvain_cluster_weights.data(),
                              vertex_louvain_cluster_weights.size(),
                              std::cout);
  
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("singleton_and_connected_flags",
                              singleton_and_connected_flags.data(),
                              singleton_and_connected_flags.size(),
                              std::cout);                   
  }
#endif

  edge_src_property_t<GraphViewType, weight_t> src_louvain_cluster_weight_cache(handle);
  edge_src_property_t<GraphViewType, weight_t> src_cut_to_louvain_cache(handle);

  if (GraphViewType::is_multi_gpu) {
    // Update cluster weight, weighted degree and cut for edge sources
    src_louvain_cluster_weight_cache =
      edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, vertex_louvain_cluster_weights.begin(), src_louvain_cluster_weight_cache);

    src_cut_to_louvain_cache = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
    update_edge_src_property(
      handle, graph_view, weighted_cut_of_vertices_to_louvain.begin(), src_cut_to_louvain_cache);

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

  kv_store_t<vertex_t, vertex_t, false> leiden_to_louvain_map(
    leiden_assignment.begin(),
    leiden_assignment.end(),
    louvain_assignment_of_vertices.begin(),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value,
    handle.get_stream());

  if(debug) std::cout << (GraphViewType::is_multi_gpu ? "MG-Graph" : "SG-Graph") << std::endl;

#if 1
  if (GraphViewType::is_multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();

    for (int i = 0; i < comm_size; ++i) {
      handle.get_comms().barrier();
      if (comm_rank == i) {
        if (comm_rank == 0) {
          if(debug) std::cout << "---------------------------------------------" << std::endl;
        }
        if(debug) std::cout << "Rank: " << i << std::endl;
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("leiden_assignment",
                                  leiden_assignment.data(),
                                  leiden_assignment.size(),
                                  std::cout);
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("louvain_assignment_of_vertices",
                                  louvain_assignment_of_vertices.data(),
                                  louvain_assignment_of_vertices.size(),
                                  std::cout);
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      }
      handle.get_comms().barrier();
    }
  }else{

    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("leiden_assignment",
                              leiden_assignment.data(),
                              leiden_assignment.size(),
                              std::cout);
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("louvain_assignment_of_vertices",
                              louvain_assignment_of_vertices.data(),
                              louvain_assignment_of_vertices.size(),
                              std::cout);
  }
#endif

  while (true) {
    vertex_t nr_remaining_active_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       singleton_and_connected_flags.begin(),
                       singleton_and_connected_flags.end(),
                       [] __device__(auto flag) { return flag > 0; });

    if(debug) std::cout << "nr_remaining_active_vertices : " << nr_remaining_active_vertices
              << std::endl;
    
    if (GraphViewType::is_multi_gpu) {
      nr_remaining_active_vertices = host_scalar_allreduce(handle.get_comms(),
                                                           nr_remaining_active_vertices,
                                                           raft::comms::op_t::SUM,
                                                           handle.get_stream());
    if(debug) std::cout << "nr_remaining_active_vertices(MG): " << nr_remaining_active_vertices << std::endl;
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
        handle, graph_view, leiden_assignment.begin(), src_leiden_assignment_cache);

      update_edge_dst_property(
        handle, graph_view, leiden_assignment.begin(), dst_leiden_assignment_cache);

      update_edge_src_property(handle,
                               graph_view,
                               singleton_and_connected_flags.begin(),
                               src_singleton_and_connected_flag_cache);
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
          // weight_t refined_partition_volume_contribution{0};
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


#if 1

        if (GraphViewType::is_multi_gpu) {
          auto& comm           = handle.get_comms();
          auto const comm_rank = comm.get_rank();
          auto const comm_size = comm.get_size();
    
          for (int k = 0; k < comm_size; k++) {
            comm.barrier();
            if (comm_rank == k) {
              if (comm_rank == 0) {
                if(debug) std::cout << "---------------------------------------------" << std::endl;
              }
              if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
              if(debug) std::cout << "Rank: " << comm_rank << std::endl;
    
              if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
              if(debug) raft::print_device_vector("refined_community_volumes",
                                        refined_community_volumes.data(),
                                        refined_community_volumes.size(),
                                        std::cout);
              if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
              if(debug) raft::print_device_vector("refined_community_cuts",
                                        refined_community_cuts.data(),
                                        refined_community_cuts.size(),
                                        std::cout);
              if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
              if(debug) raft::print_device_vector("leiden_keys_used_in_edge_reduction",
                                        leiden_keys_used_in_edge_reduction.data(),
                                        leiden_keys_used_in_edge_reduction.size(),
                                        std::cout);
    
            }
            comm.barrier();
          }
        } else {
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector("refined_community_volumes",
                                    refined_community_volumes.data(),
                                    refined_community_volumes.size(),
                                    std::cout);
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector("refined_community_cuts",
                                    refined_community_cuts.data(),
                                    refined_community_cuts.size(),
                                    std::cout);
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector("leiden_keys_used_in_edge_reduction",
                                    leiden_keys_used_in_edge_reduction.data(),
                                    leiden_keys_used_in_edge_reduction.size(),
                                    std::cout);
                              
        }

#endif


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

    rmm::device_uvector<vertex_t> louvain_of_leiden_keys_used_in_edge_reduction(0, handle.get_stream());


//------
if (GraphViewType::is_multi_gpu) {
  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  auto partitions_range_lasts = graph_view.vertex_partition_range_lasts();
  rmm::device_uvector<vertex_t> d_partitions_range_lasts(partitions_range_lasts.size(),
                                                         handle.get_stream());

  raft::update_device(d_partitions_range_lasts.data(),
                      partitions_range_lasts.data(),
                      partitions_range_lasts.size(),
                      handle.get_stream());

  cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> vertex_to_gpu_id_op{
    raft::device_span<vertex_t const>(d_partitions_range_lasts.data(),
                                      d_partitions_range_lasts.size()),
    major_comm_size,
    minor_comm_size};

  // cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
  //   comm_size, major_comm_size, minor_comm_size};

    louvain_of_leiden_keys_used_in_edge_reduction =
    cugraph::collect_values_for_keys(handle,
                                     leiden_to_louvain_map.view(),
                                     leiden_keys_used_in_edge_reduction.begin(),
                                     leiden_keys_used_in_edge_reduction.end(),
                                     vertex_to_gpu_id_op);
} else {
    louvain_of_leiden_keys_used_in_edge_reduction.resize(
      leiden_keys_used_in_edge_reduction.size(), handle.get_stream());

    leiden_to_louvain_map.view().find(
      leiden_keys_used_in_edge_reduction.begin(),
      leiden_keys_used_in_edge_reduction.end(),
      louvain_of_leiden_keys_used_in_edge_reduction.begin(),
      handle.get_stream());
}
//------

#if 1
  if (GraphViewType::is_multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();

    for (int k = 0; k < comm_size; k++) {
      comm.barrier();
      if (comm_rank == k) {
        if (comm_rank == 0) {
          if(debug) std::cout << "---------------------------------------------" << std::endl;
        }
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) std::cout << "Rank: " << comm_rank << std::endl;

        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("louvain_of_leiden_keys_used_in_edge_reduction",
                                  louvain_of_leiden_keys_used_in_edge_reduction.data(),
                                  louvain_of_leiden_keys_used_in_edge_reduction.size(),
                                  std::cout);
      }
      comm.barrier();
    }
  } else {
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("louvain_of_leiden_keys_used_in_edge_reduction",
                              louvain_of_leiden_keys_used_in_edge_reduction.data(),
                              louvain_of_leiden_keys_used_in_edge_reduction.size(),
                              std::cout);                        
  }
#endif


    // ||Cr|| //f(Cr)
    // E(Cr, louvain(v) - Cr) //f(Cr)
    // leiden(Cr) // f(Cr)
    // louvain(Cr) // f(Cr)
    auto values_for_leiden_cluster_keys = thrust::make_zip_iterator(
      thrust::make_tuple(refined_community_volumes.begin(),
                         refined_community_cuts.begin(),
                         leiden_keys_used_in_edge_reduction.begin(),  // redundant
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
      detail::leiden_key_aggregated_edge_op_t<vertex_t, weight_t, value_t>{total_edge_weight,
                                                                           resolution,
                                                                           debug},
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

#if 1
    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          if (comm_rank == 0) {
            if(debug) std::cout << "---------------------------------------------" << std::endl;
          }
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) std::cout << "Rank: " << comm_rank << std::endl;

          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector("dst",
                                    std::get<1>(gain_and_dst_output_pairs).data(),
                                    cugraph::size_dataframe_buffer(gain_and_dst_output_pairs),
                                    std::cout);

          if(debug) raft::print_device_vector("gain",
                                    std::get<0>(gain_and_dst_output_pairs).data(),
                                    cugraph::size_dataframe_buffer(gain_and_dst_output_pairs),
                                    std::cout);

        }
        comm.barrier();
      }
    } else {
      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector("dst",
                                std::get<1>(gain_and_dst_output_pairs).data(),
                                cugraph::size_dataframe_buffer(gain_and_dst_output_pairs),
                                std::cout);
      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector("gain",
                                std::get<0>(gain_and_dst_output_pairs).data(),
                                cugraph::size_dataframe_buffer(gain_and_dst_output_pairs),
                                std::cout);                
    }
#endif

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
                                                [debug] __device__(auto gain_dst_pair) {
                                                  vertex_t dst  = thrust::get<1>(gain_dst_pair);
                                                  weight_t gain = thrust::get<0>(gain_dst_pair);
                                                  if (gain > POSITIVE_GAIN) {
                                                    int idst = static_cast<int>(dst);
                                                    int igain = static_cast<int>(gain);
                                                    int igain_p = static_cast<int>(gain*100.0);

                                                   if(debug) printf("\ndst = %d gain = %d\n gain=%d/100", idst, igain, igain_p);
                                                  }
                                                  return (gain > POSITIVE_GAIN) && (dst >= 0);
                                                });

    if(debug) std::cout << "nr_valid_tuples: " << nr_valid_tuples << std::endl;

    vertex_t total_nr_valid_tuples = nr_valid_tuples;
    if (GraphViewType::is_multi_gpu) {
      total_nr_valid_tuples = host_scalar_allreduce(
        handle.get_comms(), total_nr_valid_tuples, raft::comms::op_t::SUM, handle.get_stream());

    if(debug) std::cout << "total_nr_valid_tuples(MG): " << total_nr_valid_tuples << std::endl;

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

    // debug = graph_view.number_of_vertices() < 50;
    thrust::copy_if(handle.get_thrust_policy(),
                    edge_begin,
                    edge_end,
                    d_src_dst_gain_iterator,
                    [debug] __device__(thrust::tuple<vertex_t, vertex_t, weight_t> src_dst_gain) {
                      vertex_t src  = thrust::get<0>(src_dst_gain);
                      vertex_t dst  = thrust::get<1>(src_dst_gain);
                      weight_t gain = thrust::get<2>(src_dst_gain);

                      if (gain > POSITIVE_GAIN) {

                        int isrc = static_cast<int>(src);
                        int idst = static_cast<int>(dst);
                        int igain = static_cast<int>(gain);
                        int igain_p = static_cast<int>(gain*100.0);


                       if(debug)
                        printf("=>> src = %d dst = %d gain=%d gain = %d/100\n", isrc, idst, igain, igain_p);
                      }

                      return (gain > POSITIVE_GAIN) && (dst >= 0);
                    });

#if 1
    
    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k && d_srcs.size() > 0) {
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) std::cout << "Rank :" << comm_rank << std::endl;

          if (comm_rank == 0) {
            if(debug) std::cout << "---------------------------------------------" << std::endl;
          }

          if(debug) std::cout << " d_srcs.size(): " << d_srcs.size() << " d_dsts.size(): " << d_dsts.size()
                    << " (*d_weights).size(): " << (*d_weights).size() << std::endl;
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector("d_srcs", d_srcs.data(), d_srcs.size(), std::cout);

          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector("d_dsts", d_dsts.data(), d_dsts.size(), std::cout);

          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector(
            "(*d_weights)", (*d_weights).data(), (*d_weights).size(), std::cout);

          if(debug) std::cout << "------------------" << std::endl;
        }
        comm.barrier();
      }
    }else{
      if(debug) std::cout << " d_srcs.size(): " << d_srcs.size() << " d_dsts.size(): " << d_dsts.size()
      << " (*d_weights).size(): " << (*d_weights).size() << std::endl;
      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector("d_srcs", d_srcs.data(), d_srcs.size(), std::cout);

      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector("d_dsts", d_dsts.data(), d_dsts.size(), std::cout);

      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector(
      "(*d_weights)", (*d_weights).data(), (*d_weights).size(), std::cout);
      
    }
#endif
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
      std::tie(store_transposed ? d_dsts : d_srcs,
               store_transposed ? d_srcs : d_dsts,
               d_weights,
               std::ignore,
               std::ignore) =
        cugraph::detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
          vertex_t,
          vertex_t,
          weight_t,
          int32_t>(handle,
                   store_transposed ? std::move(d_dsts) : std::move(d_srcs),
                   store_transposed ? std::move(d_srcs) : std::move(d_dsts),
                   std::move(d_weights),
                   std::nullopt,
                   std::nullopt);
    }

    if(debug) std::cout << "Before create_graph_from_edgelist ... " << std::endl;
    std::tie(decision_graph, coarse_edge_weights, std::ignore, std::ignore, renumber_map) =
      create_graph_from_edgelist<vertex_t,
                                 edge_t,
                                 weight_t,
                                 edge_t,
                                 int32_t,
                                 store_transposed,
                                 multi_gpu>(handle,
                                            std::nullopt,
                                            std::move(d_srcs),
                                            std::move(d_dsts),
                                            std::move(d_weights),
                                            std::nullopt,
                                            std::nullopt,
                                            cugraph::graph_properties_t{false, false},
                                            true,
                                            true  // FIXME: set it to false
      );

    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) std::cout << "Returned from create_graph_from_edgelist" << std::endl;

    
    auto decision_graph_view = decision_graph.view();

#if 0  
    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      if (comm_rank == 0) {
        if(debug) std::cout << std::endl;
        if(debug) std::cout << "comm_size : " << comm_size << " major_comm_size : " << major_comm_size
                  << " minor_comm_size : " << minor_comm_size << std::endl;
        if(debug) std::cout << std::endl;
      }

      for (int r = 0; r < comm_size; ++r) {
        handle.get_comms().barrier();
        if (comm_rank == r) {
          if (comm_rank == 0) {
            if(debug) std::cout << "---------------------------------------------" << std::endl;
          }
          if(debug) std::cout << "------------comm_rank = " << r
                    << " #EPar = " << decision_graph_view.number_of_local_edge_partitions()
                    << std::endl;

          if(debug) std::cout << "#V = " << decision_graph_view.number_of_vertices()
                    << " #E = " << decision_graph_view.number_of_edges() << std::endl;

          for (size_t lpidx = 0; lpidx < decision_graph_view.number_of_local_edge_partitions();
               lpidx++) {
            if (decision_graph_view.number_of_vertices() < 35) {
              if(debug) std::cout << "####--edge partition ---- " << lpidx << " #E: "
                        << decision_graph_view.number_of_local_edge_partition_edges(lpidx)
                        << std::endl;
              auto local_edge_partition_view = decision_graph_view.local_edge_partition_view(lpidx);

              if(debug) std::cout << "DCS" << (decision_graph_view.use_dcs() ? " yes" : " no") << std::endl;

              auto offsets = local_edge_partition_view.offsets();
              auto indices = local_edge_partition_view.indices();

              if(debug) raft::print_device_vector("offsets", offsets.data(), offsets.size(), std::cout);

              if(debug) raft::print_device_vector("indices", indices.data(), indices.size(), std::cout);

              if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());

              /*
              auto major_hypersparse_first = *(local_edge_partition_view.major_hypersparse_first());

              auto dcs_nzd_vertices = *(local_edge_partition_view.dcs_nzd_vertices());

              if(debug) std::cout << "major_hypersparse_first: " << major_hypersparse_first << std::endl;


              for (int idx = 0; idx < offsets.size() - 1; idx++) {
                if(debug) std::cout << (idx + decision_graph_view.local_edge_partition_src_range_first(lpidx))
                          << ": ";

                if (idx < (major_hypersparse_first -
                           decision_graph_view.local_edge_partition_src_range_first(lpidx))) {
                  if(debug) raft::print_device_vector(
                    "", indices.data(), offsets[idx + 1] - offsets[idx], std::cout);

                } else {
                  auto hs_idx = idx - major_hypersparse_first;

                  auto src = dcs_nzd_vertices[hs_idx];

                  // if(debug) std::cout << std::endl << src << ":";

                  if(debug) raft::print_device_vector(
                    "", indices.data(), offsets[idx + 1] - offsets[idx], std::cout);
                }
              }*/
            }
          }
        }

        handle.get_comms().barrier();
      }
    }
#endif
    //
    // Determine a set of moves using MIS of the decision_graph
    //

    auto vertices_in_mis = compute_mis<vertex_t, edge_t, weight_t, multi_gpu>(
      handle,
      decision_graph_view,
      coarse_edge_weights ? std::make_optional(coarse_edge_weights->view()) : std::nullopt);

#if 1 
       
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector(
      "vertices_in_mis", vertices_in_mis.data(), vertices_in_mis.size(), std::cout);
#endif

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

  #if 1
    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          if (comm_rank == 0) {
            if(debug) std::cout << "---------------------------------------------" << std::endl;
          }
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) std::cout << "Rank: " << comm_rank << std::endl;
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector(
            "numbering_indices", numbering_indices.data(), numbering_indices.size(), std::cout);

          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector(
            "*renumber_map", (*renumber_map).data(), (*renumber_map).size(), std::cout);
        }
        comm.barrier();
      }
    }else{
      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector(
        "numbering_indices", numbering_indices.data(), numbering_indices.size(), std::cout);

      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector(
        "*renumber_map", (*renumber_map).data(), (*renumber_map).size(), std::cout);
    }
  #endif

    numbering_indices.resize(0, handle.get_stream());
    numbering_indices.shrink_to_fit(handle.get_stream());

    (*renumber_map).resize(0, handle.get_stream());
    (*renumber_map).shrink_to_fit(handle.get_stream());

    if (GraphViewType::is_multi_gpu) {
      vertices_in_mis = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(vertices_in_mis), graph_view.vertex_partition_range_lasts());
    }

  #if 1
    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int k = 0; k < comm_size; k++) {
        comm.barrier();
        if (comm_rank == k) {
          if (comm_rank == 0) {
            if(debug) std::cout << "---------------------------------------------" << std::endl;
          }

          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) std::cout << "Rank: " << comm_rank << std::endl;
          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector("vertices_in_mis_mapped",
                                    vertices_in_mis.data(),
                                    vertices_in_mis.size(),
                                    std::cout);
        }
        comm.barrier();
      }
    }else{
      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector("vertices_in_mis_mapped",
                                vertices_in_mis.data(),
                                vertices_in_mis.size(),
                                std::cout);
    }
  #endif 

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
        auto v_offset = v - v_first;
        auto dst      = *(dst_first + v_offset);
        singleton_and_connected_flags[v_offset] = false;
        leiden_assignment[v_offset]             = dst;
      });

      
#if 1
if(debug) std::cout << "Print updated leiden assignment" <<std::endl;
  if(debug){
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("updated_leiden_assignment",
                            leiden_assignment.data(),
                            leiden_assignment.size(),
                            std::cout);
  }
  debug = false;
#endif

    //
    // Find the set of dest vertices
    //
    rmm::device_uvector<vertex_t> dst_vertices(vertices_in_mis.size(), handle.get_stream());

    thrust::transform(
      handle.get_thrust_policy(),
      vertices_in_mis.begin(),
      vertices_in_mis.end(),
      dst_vertices.begin(),
      [dst_first = thrust::get<1>(gain_and_dst_first.get_iterator_tuple()),
       v_first   = graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
        auto dst = *(dst_first + v - v_first);
        return dst;
      });


    cugraph::resize_dataframe_buffer(gain_and_dst_output_pairs, 0, handle.get_stream());
    cugraph::shrink_to_fit_dataframe_buffer(gain_and_dst_output_pairs, handle.get_stream());

    vertices_in_mis.resize(0, handle.get_stream());
    vertices_in_mis.shrink_to_fit(handle.get_stream());

    thrust::sort(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end());

    dst_vertices.resize(
      static_cast<size_t>(thrust::distance(
        dst_vertices.begin(),
        thrust::unique(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end()))),
      handle.get_stream());

  #if 1
    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int i = 0; i < comm_size; ++i) {
        handle.get_comms().barrier();
        if (comm_rank == i) {
          if (comm_rank == 0) {
            if(debug) std::cout << "---------------------------------------------" << std::endl;
          }

          if(debug) std::cout << "Rank: " << i << std::endl;

          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector(
            "dst_vertices", dst_vertices.data(), dst_vertices.size(), std::cout);
        }
        handle.get_comms().barrier();
      }
    } else {
      if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      if(debug) raft::print_device_vector(
        "dst_vertices", dst_vertices.data(), dst_vertices.size(), std::cout);
    }
  #endif


    // Shuffle dst vertices to owner GPU, according to vetex partitioning
    if constexpr (GraphViewType::is_multi_gpu) {
      dst_vertices = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(dst_vertices), graph_view.vertex_partition_range_lasts());

      thrust::sort(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end());

      dst_vertices.resize(
        static_cast<size_t>(thrust::distance(
          dst_vertices.begin(),
          thrust::unique(handle.get_thrust_policy(), dst_vertices.begin(), dst_vertices.end()))),
        handle.get_stream());
    }

#if 1
    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto const comm_size = comm.get_size();

      for (int i = 0; i < comm_size; ++i) {
        handle.get_comms().barrier();
        if (comm_rank == i) {
          if (comm_rank == 0) {
            if(debug) std::cout << "---------------------------------------------" << std::endl;
          }

          if(debug) std::cout << "Rank: " << i << std::endl;

          if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
          if(debug) raft::print_device_vector(
            "dst_vertices_shuffled", dst_vertices.data(), dst_vertices.size(), std::cout);
        }
        handle.get_comms().barrier();
      }
    }
#endif

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

    if(debug) std::cout << "End of current iteration" << std::endl;
  }

  if(debug) std::cout << "Out of refine while loop" << std::endl;

  src_louvain_cluster_weight_cache.clear(handle);
  src_cut_to_louvain_cache.clear(handle);

  // louvain_assignment_of_vertices.resize(0, handle.get_stream());
  // louvain_assignment_of_vertices.shrink_to_fit(handle.get_stream());

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
    static_cast<size_t>(thrust::distance(leiden_keys_to_read_louvain.begin(),
                                         thrust::unique(handle.get_thrust_policy(),
                                                        leiden_keys_to_read_louvain.begin(),
                                                        leiden_keys_to_read_louvain.end())));

  leiden_keys_to_read_louvain.resize(nr_unique_leiden_clusters, handle.get_stream());

#if 1
  if (GraphViewType::is_multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();

    for (int i = 0; i < comm_size; ++i) {
      handle.get_comms().barrier();
      if (comm_rank == i) {
        if(debug) std::cout << "Rank: " << i << std::endl;

        if (comm_rank == 0) {
          if(debug) std::cout << "---------------------------------------------" << std::endl;
        }
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("leiden_assignment",
                                  leiden_assignment.data(),
                                  leiden_assignment.size(),
                                  std::cout);
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("louvain_assignment_of_vertices",
                                  louvain_assignment_of_vertices.data(),
                                  louvain_assignment_of_vertices.size(),
                                  std::cout);
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
      }
      handle.get_comms().barrier();
    }
  } else {
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector(
      "leiden_assignment", leiden_assignment.data(), leiden_assignment.size(), std::cout);
    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("louvain_assignment_of_vertices",
                              louvain_assignment_of_vertices.data(),
                              louvain_assignment_of_vertices.size(),
                              std::cout);
  }
#endif

  if constexpr (GraphViewType::is_multi_gpu) {
    // leiden_keys_to_read_louvain =
    //   cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
    //     handle, std::move(leiden_keys_to_read_louvain));

    leiden_keys_to_read_louvain =
      cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(leiden_keys_to_read_louvain), graph_view.vertex_partition_range_lasts());

    thrust::sort(handle.get_thrust_policy(),
                 leiden_keys_to_read_louvain.begin(),
                 leiden_keys_to_read_louvain.end());

    nr_unique_leiden_clusters =
      static_cast<size_t>(thrust::distance(leiden_keys_to_read_louvain.begin(),
                                           thrust::unique(handle.get_thrust_policy(),
                                                          leiden_keys_to_read_louvain.begin(),
                                                          leiden_keys_to_read_louvain.end())));
    leiden_keys_to_read_louvain.resize(nr_unique_leiden_clusters, handle.get_stream());
  }

#if 1
  if (GraphViewType::is_multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();

    for (int i = 0; i < comm_size; ++i) {
      handle.get_comms().barrier();
      if (comm_rank == i) {
        if (comm_rank == 0) {
          if(debug) std::cout << "---------------------------------------------" << std::endl;
        }
        if(debug) std::cout << "Rank: " << i << std::endl;
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) std::cout << "nr_unique_leiden_clusters: " << nr_unique_leiden_clusters
                  << std::endl;
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("leiden_keys_to_read_louvain",
                                  leiden_keys_to_read_louvain.data(),
                                  leiden_keys_to_read_louvain.size(),
                                  std::cout);
      }
      handle.get_comms().barrier();
    }
  }else{
    if(debug) std::cout << "nr_unique_leiden_clusters: " << nr_unique_leiden_clusters
    << std::endl;

    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("leiden_keys_to_read_louvain",
                              leiden_keys_to_read_louvain.data(),
                              leiden_keys_to_read_louvain.size(),
                              std::cout);
  }
#endif

  rmm::device_uvector<vertex_t> lovain_of_leiden_cluster_keys(0, handle.get_stream());

  if (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    auto partitions_range_lasts = graph_view.vertex_partition_range_lasts();
    rmm::device_uvector<vertex_t> d_partitions_range_lasts(partitions_range_lasts.size(),
                                                           handle.get_stream());

    raft::update_device(d_partitions_range_lasts.data(),
                        partitions_range_lasts.data(),
                        partitions_range_lasts.size(),
                        handle.get_stream());

    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> vertex_to_gpu_id_op{
      raft::device_span<vertex_t const>(d_partitions_range_lasts.data(),
                                        d_partitions_range_lasts.size()),
      major_comm_size,
      minor_comm_size};

    // cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> vertex_to_gpu_id_op{
    //   comm_size, major_comm_size, minor_comm_size};

    lovain_of_leiden_cluster_keys =
      cugraph::collect_values_for_keys(handle,
                                       leiden_to_louvain_map.view(),
                                       leiden_keys_to_read_louvain.begin(),
                                       leiden_keys_to_read_louvain.end(),
                                       vertex_to_gpu_id_op);

  #if 1
    auto const comm_rank = comm.get_rank();
    for (int i = 0; i < comm_size; ++i) {
      handle.get_comms().barrier();
      if (comm_rank == i) {
        if (comm_rank == 0) {
          if(debug) std::cout << "---------------------------------------------" << std::endl;
        }
        if(debug) std::cout << "Rank: " << i << std::endl;
        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());

        if(debug) raft::print_device_vector("leiden_keys_to_read_louvain",
                                  leiden_keys_to_read_louvain.data(),
                                  leiden_keys_to_read_louvain.size(),
                                  std::cout);

        if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if(debug) raft::print_device_vector("lovain_of_leiden_cluster_keys",
                                  lovain_of_leiden_cluster_keys.data(),
                                  lovain_of_leiden_cluster_keys.size(),
                                  std::cout);
      }
      handle.get_comms().barrier();
    }
  #endif
    

  } else {
    lovain_of_leiden_cluster_keys.resize(leiden_keys_to_read_louvain.size(), handle.get_stream());

    leiden_to_louvain_map.view().find(leiden_keys_to_read_louvain.begin(),
                                      leiden_keys_to_read_louvain.end(),
                                      lovain_of_leiden_cluster_keys.begin(),
                                      handle.get_stream());

#if 1

    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());

    if(debug) raft::print_device_vector("leiden_keys_to_read_louvain",
                              leiden_keys_to_read_louvain.data(),
                              leiden_keys_to_read_louvain.size(),
                              std::cout);

    if(debug) RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if(debug) raft::print_device_vector("lovain_of_leiden_cluster_keys",
                              lovain_of_leiden_cluster_keys.data(),
                              lovain_of_leiden_cluster_keys.size(),
                              std::cout);
#endif
  }
  return std::make_tuple(std::move(leiden_assignment),
                         std::make_pair(std::move(leiden_keys_to_read_louvain),
                                        std::move(lovain_of_leiden_cluster_keys)));
}
}  // namespace detail
}  // namespace cugraph
