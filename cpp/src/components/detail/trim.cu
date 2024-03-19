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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>
#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct extract_one_core_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(vertex_t src,
                                                                            vertex_t dst,
                                                                            bool src_one_core,
                                                                            bool dst_one_core,
                                                                            thrust::nullopt_t) const
  {
    return (src_one_core ==  true) && (dst_one_core == true)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};

template <typename edge_t>
struct is_one_or_greater_t {
  __device__ bool operator()(edge_t core_number) const
  {
    return core_number >= edge_t{1} ? true : false;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>>
trim(raft::handle_t const& handle,
       cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
       bool do_expensive_check)
{
    rmm::device_uvector<edge_t> core_numbers(graph_view.number_of_vertices(), handle.get_stream());

    core_number(
      handle, graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{1}, size_t{1});

   auto out_one_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_one_or_greater_t<edge_t>{});

    rmm::device_uvector<uint8_t> out_one_core_flags(core_numbers.size(), handle.get_stream());

    thrust::copy(handle.get_thrust_policy(),
                 out_one_core_first,
                 out_one_core_first + core_numbers.size(),
                 out_one_core_flags.begin());

    edge_src_property_t<decltype(graph_view), uint8_t> edge_src_out_one_cores(handle,
                                                                                 graph_view);
    edge_dst_property_t<decltype(graph_view), uint8_t> edge_dst_out_one_cores(handle,
                                                                                 graph_view);
    update_edge_src_property(
      handle, graph_view, out_one_core_flags.begin(), edge_src_out_one_cores);
    update_edge_dst_property(
      handle, graph_view, out_one_core_flags.begin(), edge_dst_out_one_cores);

    rmm::device_uvector<size_t> d_subgraph_offsets_out(2, handle.get_stream());
    std::vector<size_t> h_subgraph_offsets_out{{0, out_one_core_flags.size()}};
    raft::update_device(d_subgraph_offsets_out.data(),
                      h_subgraph_offsets_out.data(),
                      h_subgraph_offsets_out.size(),
                      handle.get_stream());

    graph_t<vertex_t, edge_t, false, multi_gpu> subgraph_out(handle);
    auto subgraph_out_view = subgraph_out.view(); 

    auto [src, dst, wgt, offset] = extract_induced_subgraphs(
    handle,
    subgraph_out_view,
    std::nullopt,
    raft::device_span<size_t const>{d_subgraph_offsets_out.data(), d_subgraph_offsets_out.size()},
    raft::device_span<vertex_t const>{edge_src_out_one_cores.data(), edge_src_out_one_cores.size()},
    do_expensive_check);


    std::optional<rmm::device_uvector<vertex_t>> renumber_map_out{std::nullopt};

    std::tie(subgraph_out, std::ignore, 
      std::ignore, std::ignore, renumber_map_out) = 
    create_graph_from_edgelist<vertex_t,
                               edge_t,
                               weight_t,
                               edge_t,
                               int32_t,
                               false,
                               multi_gpu>(handle, graph_view.local_vertex_partition_range_first(), src, dst, std::nullopt);

    
    // in degree 1 core
    core_number(
      handle, subgraph_out_view, core_numbers.data(), k_core_degree_type_t::IN, size_t{1}, size_t{1});

   auto in_one_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_one_or_greater_t<edge_t>{});
    rmm::device_uvector<uint8_t> in_one_core_flags(core_numbers.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 in_one_core_first,
                 in_one_core_first + core_numbers.size(),
                 in_one_core_flags.begin());

    edge_src_property_t<decltype(subgraph_out_view), uint8_t> edge_src_in_one_cores(handle,
                                                                                 subgraph_out_view);
    edge_dst_property_t<decltype(subgraph_out_view), uint8_t> edge_dst_in_one_cores(handle,
                                                                                 subgraph_out_view);
 
    update_edge_src_property(
      handle, subgraph_out_view, in_one_core_flags.begin(), edge_src_in_one_cores);
    update_edge_dst_property(
      handle, subgraph_out_view, in_one_core_flags.begin(), edge_dst_in_one_cores);
    rmm::device_uvector<size_t> d_subgraph_offsets_in(2, handle.get_stream());
    std::vector<size_t> h_subgraph_offsets_in{{0, in_one_core_flags.size()}};
    raft::update_device(d_subgraph_offsets_in.data(),
                      h_subgraph_offsets_in.data(),
                      h_subgraph_offsets_in.size(),
                      handle.get_stream());
    graph_t<vertex_t, edge_t, false, multi_gpu> subgraph_in(handle);
    std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>> subgraph_edge_weights_in{};
    std::optional<rmm::device_uvector<vertex_t>> renumber_map_in{std::nullopt};

    auto [src_, dst_, wgt_, offsets_] = extract_induced_subgraphs(
    handle,
    subgraph_out_view,
    std::nullopt,
    raft::device_span<size_t const>{d_subgraph_offsets_in.data(), d_subgraph_offsets_in.size()},
    raft::device_span<vertex_t const>{subgraph_in.data(), subgraph_in.size()},
    do_expensive_check);

    std::tie(subgraph_in, std::ignore, std::ignore, std::ignore, renumber_map_in) = 
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_t,
                                        int32_t,
                                        false,
                                        multi_gpu>(handle, subgraph_out_view.local_vertex_partition_range_first(), src_, dst_, std::nullopt);
 
    auto subgraph_in_view = subgraph_in.view();
    std::vector<vertex_t*> major_ptrs_in(subgraph_in_view.local_vertex_partition_range_size());
    std::vector<vertex_t*> minor_ptrs_in(major_ptrs_in.size());

    unrenumber_local_int_edges<vertex_t, false, multi_gpu>(
      handle,
      major_ptrs_in,
      minor_ptrs_in,
      major_ptrs_in.size(),
      (*renumber_map_in).data(),
      (*renumber_map_in).size());

   
    return  std::make_tuple(std::move(src_), std::move(dst_), std::move(wgt_));
}

}
}
