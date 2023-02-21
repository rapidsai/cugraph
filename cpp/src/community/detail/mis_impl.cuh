
/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <community/detail/mis.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/distance.h>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> rank_vertices(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  return compute_out_weight_sums<vertex_t, edge_t, weight_t, false, multi_gpu>(
    handle, graph_view, *edge_weight_view);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  using FlagType      = vertex_t;

  vertex_t number_of_vertices = graph_view.local_vertex_partition_range_size();

  rmm::device_uvector<vertex_t> mis(number_of_vertices, handle.get_stream());

  bool debug = graph_view.local_vertex_partition_range_size() < 40;
  if (debug) {
    auto offsets = graph_view.local_edge_partition_view(0).offsets();
    auto indices = graph_view.local_edge_partition_view(0).indices();
    cudaDeviceSynchronize();
    std::cout << "---- MIS Graph -----" << std::endl;
    raft::print_device_vector("offsets: ", offsets.data(), offsets.size(), std::cout);
    raft::print_device_vector("indices: ", indices.data(), indices.size(), std::cout);
  }
  //
  // Flag to indicate if a vertex is included in the MIS
  //
  rmm::device_uvector<FlagType> mis_inclusion_flags(number_of_vertices, handle.get_stream());
  thrust::uninitialized_fill(handle.get_thrust_policy(),
                             mis_inclusion_flags.begin(),
                             mis_inclusion_flags.end(),
                             FlagType{0});

  //
  // Falg to indicate if a vertex should be discarded from consideration, either
  // because itself or one of its neighbor is included in the MIS
  //
  rmm::device_uvector<FlagType> discard_flags(number_of_vertices, handle.get_stream());
  thrust::uninitialized_fill(
    handle.get_thrust_policy(), discard_flags.begin(), discard_flags.end(), FlagType{0});

  rmm::device_uvector<vertex_t> remaining_candidates(0, handle.get_stream());

  auto vertex_begin =
    thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first());
  auto vertex_end = thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last());

  rmm::device_uvector<vertex_t> included_list(graph_view.local_vertex_partition_range_size(),
                                              handle.get_stream());
  rmm::device_uvector<vertex_t> discard_list(graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());

  auto out_degrees = graph_view.compute_out_degrees(handle);

  if (debug) {
    cudaDeviceSynchronize();
    raft::print_device_vector("degrees: ", out_degrees.data(), out_degrees.size(), std::cout);
  }

  while (true) {
    cudaDeviceSynchronize();
    std::cout << " mis loop .." << std::endl;
    // Select a random set of eligible vertices

    vertex_t nr_remaining_candidates =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(thrust::make_tuple(
                         mis_inclusion_flags.begin(), discard_flags.begin(), out_degrees.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(
                         mis_inclusion_flags.end(), discard_flags.end(), out_degrees.end())),
                       [] __device__(auto flags) {
                         return (thrust::get<0>(flags) == 0) && (thrust::get<1>(flags) == 0) &&
                                (thrust::get<2>(flags) > 0);
                       });

    remaining_candidates.resize(nr_remaining_candidates, handle.get_stream());

    thrust::copy_if(handle.get_thrust_policy(),
                    vertex_begin,
                    vertex_end,
                    thrust::make_zip_iterator(thrust::make_tuple(
                      mis_inclusion_flags.begin(), discard_flags.begin(), out_degrees.begin())),
                    remaining_candidates.begin(),
                    [] __device__(auto flags) {
                      return (thrust::get<0>(flags) == 0) && (thrust::get<1>(flags) == 0) &&
                             (thrust::get<2>(flags) > 0);
                    });

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("remaining_candidates: ",
                                remaining_candidates.data(),
                                remaining_candidates.size(),
                                std::cout);
    }

    thrust::default_random_engine g;
    g.seed(0);
    thrust::shuffle(
      handle.get_thrust_policy(), remaining_candidates.begin(), remaining_candidates.end(), g);
    remaining_candidates.resize(
      std::max(vertex_t{1}, static_cast<vertex_t>(0.50 * nr_remaining_candidates)),
      handle.get_stream());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("remaining_candidates: (shuffle and picked 30%) ",
                                remaining_candidates.data(),
                                remaining_candidates.size(),
                                std::cout);
    }

    thrust::sort(
      handle.get_thrust_policy(), remaining_candidates.begin(), remaining_candidates.end());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("remaining_candidates(sorted): ",
                                remaining_candidates.data(),
                                remaining_candidates.size(),
                                std::cout);
    }

    remaining_candidates.resize(thrust::distance(remaining_candidates.begin(),
                                                 thrust::unique(handle.get_thrust_policy(),
                                                                remaining_candidates.begin(),
                                                                remaining_candidates.end())),
                                handle.get_stream());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("remaining_candidates: unique ",
                                remaining_candidates.data(),
                                remaining_candidates.size(),
                                std::cout);
    }

    // Flag vertices that are selected to be part of the MIS upon validity checks
    rmm::device_uvector<FlagType> selection_flags(graph_view.local_vertex_partition_range_size(),
                                                  handle.get_stream());

    thrust::uninitialized_fill(
      handle.get_thrust_policy(), selection_flags.begin(), selection_flags.end(), FlagType{0});

    thrust::transform(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      selection_flags.begin(),
                      [selected_nodes    = remaining_candidates.data(),
                       nr_selected_nodes = remaining_candidates.size()] __device__(auto node) {
                        return thrust::binary_search(
                          thrust::seq, selected_nodes, selected_nodes + nr_selected_nodes, node);
                      });

    vertex_t nr_selected_vertices = thrust::count_if(handle.get_thrust_policy(),
                                                     selection_flags.begin(),
                                                     selection_flags.end(),
                                                     [] __device__(auto flag) { return flag > 0; });

    rmm::device_uvector<vertex_t> selection_list(graph_view.local_vertex_partition_range_size(),
                                                 handle.get_stream());

    if (debug) {
      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      selection_flags.begin(),
                      selection_list.begin(),
                      [] __device__(auto flag) { return flag > 0; });

      selection_list.resize(nr_selected_vertices, handle.get_stream());
      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector(
        "slected vertices ", selection_list.data(), selection_list.size(), std::cout);
    }

    edge_src_property_t<GraphViewType, FlagType> src_inclusion_flag_cache(handle);
    edge_dst_property_t<GraphViewType, FlagType> dst_inclusion_flag_cache(handle);

    edge_src_property_t<GraphViewType, FlagType> src_selection_flag_cache(handle);
    edge_dst_property_t<GraphViewType, FlagType> dst_selection_flag_cache(handle);

    auto ranks = compute_out_weight_sums<vertex_t, edge_t, weight_t, false, multi_gpu>(
      handle, graph_view, *edge_weight_view);
    // rank_vertices<vertex_t, edge_t, weight_t, multi_gpu>(handle, graph_view, edge_weight_view);

    edge_src_property_t<GraphViewType, weight_t> src_rank_cache(handle);
    edge_dst_property_t<GraphViewType, weight_t> dst_rank_cache(handle);

    if constexpr (multi_gpu) {
      src_inclusion_flag_cache = edge_src_property_t<GraphViewType, FlagType>(handle, graph_view);
      dst_inclusion_flag_cache = edge_dst_property_t<GraphViewType, FlagType>(handle, graph_view);
      update_edge_src_property(
        handle, graph_view, mis_inclusion_flags.begin(), src_inclusion_flag_cache);
      update_edge_dst_property(
        handle, graph_view, mis_inclusion_flags.begin(), dst_inclusion_flag_cache);

      src_selection_flag_cache = edge_src_property_t<GraphViewType, FlagType>(handle, graph_view);
      dst_selection_flag_cache = edge_dst_property_t<GraphViewType, FlagType>(handle, graph_view);
      update_edge_src_property(
        handle, graph_view, selection_flags.begin(), src_selection_flag_cache);
      update_edge_dst_property(
        handle, graph_view, selection_flags.begin(), dst_selection_flag_cache);

      src_rank_cache = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<GraphViewType, weight_t>(handle, graph_view);
      update_edge_src_property(handle, graph_view, ranks.begin(), src_rank_cache);
      update_edge_dst_property(handle, graph_view, ranks.begin(), dst_rank_cache);
    }

    rmm::device_uvector<FlagType> de_selection_flags(graph_view.local_vertex_partition_range_size(),
                                                     handle.get_stream());

    rmm::device_uvector<FlagType> newly_discarded_flags(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::uninitialized_fill(handle.get_thrust_policy(),
                               de_selection_flags.begin(),
                               de_selection_flags.end(),
                               FlagType{0});

    thrust::uninitialized_fill(handle.get_thrust_policy(),
                               newly_discarded_flags.begin(),
                               newly_discarded_flags.end(),
                               FlagType{0});

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      multi_gpu
        ? view_concat(
            src_inclusion_flag_cache.view(), src_selection_flag_cache.view(), src_rank_cache.view())
        : view_concat(
            detail::edge_major_property_view_t<vertex_t, FlagType const*>(
              mis_inclusion_flags.data()),
            detail::edge_major_property_view_t<vertex_t, FlagType const*>(selection_flags.data()),
            detail::edge_major_property_view_t<vertex_t, weight_t const*>(ranks.data())),
      multi_gpu
        ? view_concat(
            dst_inclusion_flag_cache.view(), dst_selection_flag_cache.view(), dst_rank_cache.view())
        : view_concat(detail::edge_minor_property_view_t<vertex_t, FlagType const*>(
                        mis_inclusion_flags.data(), vertex_t{0}),
                      detail::edge_minor_property_view_t<vertex_t, FlagType const*>(
                        selection_flags.data(), vertex_t{0}),
                      detail::edge_minor_property_view_t<vertex_t, weight_t const*>(ranks.data(),
                                                                                    vertex_t{0})),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_info, auto dst_info, auto wt) {
        auto is_src_selected = thrust::get<1>(src_info);
        auto src_rank        = thrust::get<2>(src_info);

        auto is_dst_included = thrust::get<0>(dst_info);
        auto is_dst_selected = thrust::get<1>(dst_info);
        auto dst_rank        = thrust::get<2>(dst_info);

        // If a neighbor of a vertex is already included in the MIS,
        // then the vertex can't be included.

        // printf(
        //   "\n(outgoing) src = %d, is_src_selected = %d, src_rank = %f : dst = %d, is_dst_selected
        //   "
        //   "= %d, "
        //   "is_dst_included = %d, dst_rank = %f",
        //   src,
        //   uint32_t{is_src_selected},
        //   src_rank,
        //   dst,
        //   uint32_t{is_dst_selected},
        //   uint32_t{is_dst_included},
        //   dst_rank);

        if (is_dst_included) {
          // deselect and discard
          return thrust::make_tuple(FlagType{1}, FlagType{1});
        }

        if (is_src_selected && is_dst_selected) {
          // printf("\n (src=%d ---> dst=%d)  ss=%d ds=%d,  sr=%f dr=%f",
          //        uint32_t{src},
          //        uint32_t{dst},
          //        uint32_t{is_src_selected},
          //        uint32_t{is_dst_selected},
          //        src_rank,
          //        dst_rank);
          // Give priority to high rank, high id vertex
          if (src_rank < dst_rank) {
            // smaller rank, deselect it
            return thrust::make_tuple(FlagType{1}, FlagType{0});
          } else if (fabs(src_rank - dst_rank) < 1e-9) {
            if (src < dst) {
              // equal rank, but smaller id
              return thrust::make_tuple(FlagType{1}, FlagType{0});
            }
          }
        }

        // printf("\n (src=%d ---> dst=%d) returning %d %d",
        //        uint32_t{src},
        //        uint32_t{dst},
        //        uint32_t{0},
        //        uint32_t{0});
        return thrust::make_tuple(FlagType{0}, FlagType{0});
      },
      thrust::make_tuple(FlagType{0}, FlagType{0}),
      thrust::make_zip_iterator(
        thrust::make_tuple(de_selection_flags.begin(), newly_discarded_flags.begin())));

    // Update selection flags
    thrust::transform(
      handle.get_thrust_policy(),
      selection_flags.begin(),
      selection_flags.end(),
      de_selection_flags.begin(),
      selection_flags.begin(),
      [] __device__(auto status, auto deselected) { return (status > 0) && (!(deselected > 0)); });

    if (debug) {
      vertex_t nr_deselected = thrust::count_if(handle.get_thrust_policy(),
                                                de_selection_flags.begin(),
                                                de_selection_flags.end(),
                                                [] __device__(auto flag) { return flag > 0; });

      rmm::device_uvector<vertex_t> de_selection_list(nr_deselected, handle.get_stream());

      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      de_selection_flags.begin(),
                      de_selection_list.begin(),
                      [] __device__(auto flag) { return flag > 0; });

      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector(
        "\nde_selection_list ", de_selection_list.data(), de_selection_list.size(), std::cout);
    }

    thrust::transform(
      handle.get_thrust_policy(),
      discard_flags.begin(),
      discard_flags.end(),
      newly_discarded_flags.begin(),
      discard_flags.begin(),
      [] __device__(auto prev, auto current) { return (prev > 0) || (current > 0); });

    vertex_t nr_discarded_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       discard_flags.begin(),
                       discard_flags.end(),
                       [] __device__(auto flag) { return flag > 0; });

    if (debug) {
      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      discard_flags.begin(),
                      discard_list.begin(),
                      [] __device__(auto flag) { return flag > 0; });

      discard_list.resize(nr_discarded_vertices, handle.get_stream());
      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector(
        "\nupdated discarded vertices ", discard_list.data(), discard_list.size(), std::cout);

      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      selection_flags.begin(),
                      selection_list.begin(),
                      [] __device__(auto flag) { return flag > 0; });

      nr_selected_vertices = thrust::count_if(handle.get_thrust_policy(),
                                              selection_flags.begin(),
                                              selection_flags.end(),
                                              [] __device__(auto flag) { return flag > 0; });

      selection_list.resize(nr_selected_vertices, handle.get_stream());
      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector(
        "updated slected vertices ", selection_list.data(), selection_list.size(), std::cout);
    }

    if constexpr (multi_gpu) {
      update_edge_src_property(
        handle, graph_view, selection_flags.begin(), src_selection_flag_cache);
      update_edge_dst_property(
        handle, graph_view, selection_flags.begin(), dst_selection_flag_cache);
    }

    // rmm::device_uvector<FlagType> final_selection_flags(
    //   graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      multi_gpu
        ? view_concat(
            src_inclusion_flag_cache.view(), src_selection_flag_cache.view(), src_rank_cache.view())
        : view_concat(
            detail::edge_major_property_view_t<vertex_t, FlagType const*>(
              mis_inclusion_flags.data()),
            detail::edge_major_property_view_t<vertex_t, FlagType const*>(selection_flags.data()),
            detail::edge_major_property_view_t<vertex_t, weight_t const*>(ranks.data())),
      multi_gpu
        ? view_concat(
            dst_inclusion_flag_cache.view(), dst_selection_flag_cache.view(), dst_rank_cache.view())
        : view_concat(

            detail::edge_minor_property_view_t<vertex_t, FlagType const*>(
              mis_inclusion_flags.data(), vertex_t{0}),
            detail::edge_minor_property_view_t<vertex_t, FlagType const*>(selection_flags.data(),
                                                                          vertex_t{0}),
            detail::edge_minor_property_view_t<vertex_t, weight_t const*>(ranks.data(),
                                                                          vertex_t{0})),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_info, auto dst_info, auto wt) {
        auto is_src_included = thrust::get<0>(src_info);
        auto is_src_selected = thrust::get<1>(src_info);
        auto src_rank        = thrust::get<2>(src_info);

        auto is_dst_selected = thrust::get<1>(dst_info);
        auto dst_rank        = thrust::get<2>(dst_info);

        // printf(
        //   "\n (incoming) src = %d, is_src_selected = %d, is_src_included = %d, src_rank = %f :
        //   dst "
        //   "= %d, "
        //   "is_dst_selected = %d,  dst_rank = %f",
        //   src,
        //   uint32_t{is_src_selected},
        //   uint32_t{is_src_included},
        //   src_rank,
        //   dst,
        //   uint32_t{is_dst_selected},
        //   dst_rank);

        if (is_src_included) {
          // deselect and discard
          return thrust::make_tuple(FlagType{1}, FlagType{1});
        }

        // Give priority to high rank, high id vertex
        if (is_src_selected && is_dst_selected) {
          // printf("\n (in) (src=%d ---> dst=%d)  ss=%d ds=%d,  sr=%f dr=%f",
          //        uint32_t{src},
          //        uint32_t{dst},
          //        uint32_t{is_src_selected},
          //        uint32_t{is_dst_selected},
          //        src_rank,
          //        dst_rank);
          return thrust::make_tuple(static_cast<FlagType>(thrust::make_tuple(src_rank, src) >
                                                          thrust::make_tuple(dst_rank, dst)),
                                    FlagType{0});
        }

        // printf("\n(src=%d ---> dst=%d) returning %d %d",
        //        uint32_t{src},
        //        uint32_t{dst},
        //        uint32_t{0},
        //        uint32_t{0});
        return thrust::make_tuple(FlagType{0}, FlagType{0});
      },
      thrust::make_tuple(FlagType{0}, FlagType{0}),
      thrust::make_zip_iterator(
        thrust::make_tuple(de_selection_flags.begin(), newly_discarded_flags.begin())));

    if (debug) {
      vertex_t nr_deselected = thrust::count_if(handle.get_thrust_policy(),
                                                de_selection_flags.begin(),
                                                de_selection_flags.end(),
                                                [] __device__(auto flag) { return flag > 0; });

      rmm::device_uvector<vertex_t> de_selection_list(nr_deselected, handle.get_stream());

      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      de_selection_flags.begin(),
                      de_selection_list.begin(),
                      [] __device__(auto flag) { return flag > 0; });

      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector(
        "\n(in)de_selection_list ", de_selection_list.data(), de_selection_list.size(), std::cout);
    }

    // Update selection flags
    thrust::transform(
      handle.get_thrust_policy(),
      selection_flags.begin(),
      selection_flags.end(),
      de_selection_flags.begin(),
      selection_flags.begin(),
      [] __device__(auto status, auto deselected) { return (status > 0) && (!(deselected > 0)); });

    thrust::transform(
      handle.get_thrust_policy(),
      discard_flags.begin(),
      discard_flags.end(),
      newly_discarded_flags.begin(),
      discard_flags.begin(),
      [] __device__(auto prev, auto current) { return (prev > 0) || (current > 0); });

    // Add selected vertices to the MIS and discard in the next selection round
    thrust::transform(handle.get_thrust_policy(),
                      selection_flags.begin(),
                      selection_flags.end(),
                      mis_inclusion_flags.begin(),
                      mis_inclusion_flags.begin(),
                      [] __device__(auto selected, auto prev_inclusion_flag) {
                        return static_cast<FlagType>(selected || prev_inclusion_flag);
                      });

    vertex_t nr_include_vertices = thrust::count_if(handle.get_thrust_policy(),
                                                    mis_inclusion_flags.begin(),
                                                    mis_inclusion_flags.end(),
                                                    [] __device__(auto flag) { return flag > 0; });

    if (debug) {
      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      mis_inclusion_flags.begin(),
                      included_list.begin(),
                      [] __device__(auto flag) { return flag > 0; });

      included_list.resize(nr_include_vertices, handle.get_stream());

      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector(
        "included vertices: ", included_list.data(), included_list.size(), std::cout);
    }

    nr_discarded_vertices = thrust::count_if(handle.get_thrust_policy(),
                                             discard_flags.begin(),
                                             discard_flags.end(),
                                             [] __device__(auto flag) { return flag > 0; });

    if (debug) {
      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      discard_flags.begin(),
                      discard_list.begin(),
                      [] __device__(auto flag) { return flag > 0; });

      discard_list.resize(nr_discarded_vertices, handle.get_stream());
      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector(
        "discarded vertices ", discard_list.data(), discard_list.size(), std::cout);
    }

    vertex_t nr_remaining_vertices_to_check =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(thrust::make_tuple(
                         mis_inclusion_flags.begin(), discard_flags.begin(), out_degrees.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(
                         mis_inclusion_flags.end(), discard_flags.end(), out_degrees.end())),
                       [] __device__(auto flags) {
                         return (thrust::get<0>(flags) == 0) && (thrust::get<1>(flags) == 0) &&
                                (thrust::get<2>(flags) > 0);
                       });

    if (multi_gpu) {
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    }

    cudaDeviceSynchronize();

    std::cout << " number_of_vertices: " << number_of_vertices << std::endl;
    std::cout << " nr_include_vertices: " << nr_include_vertices << std::endl;
    std::cout << " nr_discarded_vertices: " << nr_discarded_vertices << std::endl;
    std::cout << " nr_remaining_vertices_to_check: " << nr_remaining_vertices_to_check << std::endl;

    if (nr_remaining_vertices_to_check == 0) { break; }
  }

  vertex_t nr_vertices_in_mis = thrust::count_if(handle.get_thrust_policy(),
                                                 mis_inclusion_flags.begin(),
                                                 mis_inclusion_flags.end(),
                                                 [] __device__(auto flag) { return flag > 0; });

  thrust::copy_if(handle.get_thrust_policy(),
                  vertex_begin,
                  vertex_end,
                  mis_inclusion_flags.begin(),
                  mis.begin(),
                  [] __device__(auto flag) { return flag > 0; });

  mis.resize(nr_vertices_in_mis, handle.get_stream());

  if (debug) {
    cudaDeviceSynchronize();
    raft::print_device_vector("mis", mis.data(), mis.size(), std::cout);
  }

  return mis;
}
}  // namespace detail
}  // namespace cugraph