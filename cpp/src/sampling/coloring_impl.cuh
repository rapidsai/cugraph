/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

// #include "mis_impl.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <cuda/functional>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/merge.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cmath>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> coloring(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  using graph_view_t = decltype(graph_view);

  auto num_vertices = graph_view.number_of_vertices();
  auto num_edges    = graph_view.compute_number_of_edges(handle);

  std::cout << "#E : " << num_edges << std::endl;
  std::cout << "#V:  " << num_vertices << std::endl;

  auto out_degrees      = graph_view.compute_out_degrees(handle);
  auto out_degrees_tile = std::string("out_degrees_").append(std::to_string(comm_rank));

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  raft::print_device_vector(
    out_degrees_tile.c_str(), out_degrees.begin(), out_degrees.size(), std::cout);


  // cugraph::edge_property_t<graph_view_t, bool> edge_masks(handle, graph_view);
  // cugraph::fill_edge_property(handle, graph_view, bool{true}, edge_masks);

  rmm::device_uvector<vertex_t> colors = rmm::device_uvector<vertex_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  thrust::fill(
    handle.get_thrust_policy(), colors.begin(), colors.end(), std::numeric_limits<vertex_t>::max());

  vertex_t color_id = 0;

  while (true) {
    auto mis =
      cugraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(handle, graph_view, rng_state);

    auto mis_tile = std::string("mis_").append(std::to_string(comm_rank));

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector(mis_tile.c_str(), mis.begin(), mis.size(), std::cout);
    /*
        using falg_type = vertex_t;

        rmm::device_uvector<falg_type> is_vertex_in_mis = rmm::device_uvector<falg_type>(
          graph_view.local_vertex_partition_range_size(), handle.get_stream());

        thrust::fill(handle.get_thrust_policy(), is_vertex_in_mis.begin(), is_vertex_in_mis.end(),
       0);

        thrust::for_each(
          handle.get_thrust_policy(),
          mis.begin(),
          mis.end(),
          [color_id,
           colors           = colors.data(),
           is_vertex_in_mis = is_vertex_in_mis.data(),
           v_first          = graph_view.local_vertex_partition_range_first()] __device__(vertex_t
       v) { auto v_offset              = v - v_first; is_vertex_in_mis[v_offset] = falg_type{1};
            colors[v_offset]           = (colors[v_offset] < color_id) ? colors[v_offset] :
       color_id;
          });

        auto is_vertex_in_mis_tile =
       std::string("is_vertex_in_mis").append(std::to_string(comm_rank));

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector(
          is_vertex_in_mis_tile.c_str(), is_vertex_in_mis.begin(), is_vertex_in_mis.size(),
       std::cout);

        cugraph::edge_src_property_t<graph_view_t, falg_type> src_mis_flags(handle, graph_view);
        cugraph::edge_dst_property_t<graph_view_t, falg_type> dst_mis_flags(handle, graph_view);

        cugraph::update_edge_src_property(handle, graph_view, is_vertex_in_mis.begin(),
       src_mis_flags);

        cugraph::update_edge_dst_property(handle, graph_view, is_vertex_in_mis.begin(),
       dst_mis_flags);

        if (graph_view.compute_number_of_edges(handle) == 0) { break; }
        cugraph::transform_e(
          handle,
          graph_view,
          src_mis_flags.view(),
          dst_mis_flags.view(),
          edge_masks.view(),
          [] __device__(
            auto src, auto dst, auto is_src_to_remove, auto is_dst_to_remove, auto current_mask) {
            auto rt = (is_src_to_remove == 0) && (is_dst_to_remove == 0);

            printf(
              "src = %d ---> dst = %d : is_src_to_remove = %d is_dst_to_remove = %d current = %d =>
       " "next = %d\n", static_cast<int>(src), static_cast<int>(dst),
              static_cast<int>(is_src_to_remove),
              static_cast<int>(is_dst_to_remove),
              static_cast<int>(current_mask),
              static_cast<int>(rt));
            return rt;
          },
          edge_masks.mutable_view());

        graph_view.clear_edge_mask();
        graph_view.attach_edge_mask(edge_masks.view());

        num_vertices = graph_view.number_of_vertices();
        num_edges    = graph_view.compute_number_of_edges(handle);

        color_id++;
        std::cout << "(with new mask) #E : " << num_edges << std::endl;
        std::cout << "(with new mask) #V:  " << num_vertices << std::endl;
        */
    break;
  }

  std::cout << "Used " << color_id << " colors\n";
  return colors;
}
}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> coloring(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  return detail::coloring(handle, graph_view, rng_state);
}

}  // namespace cugraph