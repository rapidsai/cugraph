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

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>
#include <cugraph/mtmg/edge_property.hpp>
#include <cugraph/mtmg/graph_view.hpp>
#include <cugraph/mtmg/handle.hpp>
#include <cugraph/mtmg/renumber_map.hpp>

// DEBUGGING
#include <cugraph/utilities/device_functors.cuh>
#include <detail/graph_partition_utils.cuh>
// END DEBUGGING

namespace cugraph {
namespace mtmg {

/**
 * @brief Graph object for each GPU
 */
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
class graph_t : public detail::device_shared_wrapper_t<
                  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>> {
 public:
  graph_t()
    : detail::device_shared_wrapper_t<
        cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>>()
  {
  }

  void set_view(handle_t const& handle,
                cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>& views)
  {
    views.initialize_pointer(handle, this->get_pointer(handle)->view());
  }
};

/**
 * @brief Create an MTMG graph from an edgelist
 *
 * @param[in]  handle             Resource handle
 * @param[in]  edgelist           Edgelist
 * @param[in]  graph_properties   Graph properties
 * @param[in]  renumber           If true, renumber graph (must be true for MG)
 * @param[out] graph              MTMG graph is stored here
 * @param[out] edge_weights       MTMG edge weights is stored here
 * @param[out] edge_ids           MTMG edge ids is stored here
 * @param[out] edge_types         MTMG edge types is stored here
 * @param[in]  renumber_map       MTMG renumber_map is stored here
 * @param[in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
void create_graph_from_edgelist(
  handle_t const& handle,
  cugraph::mtmg::edgelist_t<vertex_t, weight_t, edge_id_t, edge_type_t>& edgelist,
  graph_properties_t graph_properties,
  bool renumber,
  cugraph::mtmg::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>& graph,
  std::optional<cugraph::mtmg::edge_property_t<
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    weight_t>>& edge_weights,
  std::optional<cugraph::mtmg::edge_property_t<
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    edge_id_t>>& edge_ids,
  std::optional<cugraph::mtmg::edge_property_t<
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    edge_type_t>>& edge_types,
  std::optional<cugraph::mtmg::renumber_map_t<vertex_t>>& renumber_map,
  bool do_expensive_check = false)
{
  if (handle.get_thread_rank() > 0) return;

  CUGRAPH_EXPECTS(renumber_map.has_value() == renumber,
                  "Renumbering set to true, but no space for renumber map");

  auto my_edgelist = edgelist.get_pointer(handle);

  CUGRAPH_EXPECTS(my_edgelist->get_src().size() > 0, "Cannot create graph without an edge list");
  CUGRAPH_EXPECTS(my_edgelist->get_src().size() == 1,
                  "Must consolidate edges into a single list before creating graph");

  // DEBUGGING... pulled the error check out of graph construction to experiment a bit...
  if constexpr (multi_gpu) {
    auto& comm           = handle.raft_handle().get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();
    auto& major_comm =
      handle.raft_handle().get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm =
      handle.raft_handle().get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    std::cout << "in expensive_check_edgelist, multi_gpu block, rank = " << comm_rank
              << ", size = " << comm_size << std::endl;

    std::cout << " checking edges, rank = " << comm_rank
              << ", majors size = " << my_edgelist->get_dst()[0].size() << std::endl;
    raft::print_device_vector(" edgelist_majors",
                              my_edgelist->get_dst()[0].data(),
                              my_edgelist->get_dst()[0].size(),
                              std::cout);
    raft::print_device_vector(" edgelist_minors",
                              my_edgelist->get_src()[0].data(),
                              my_edgelist->get_src()[0].size(),
                              std::cout);
    handle.raft_handle().sync_stream();
    std::cout << "after sync" << std::endl;
    int xxx;
    RAFT_CUDA_TRY(cudaGetDevice(&xxx));
    std::cout << "   device = " << xxx << std::endl;
    std::cout << "   stream = " << handle.raft_handle().get_stream() << std::endl;

    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(my_edgelist->get_dst()[0].begin(), my_edgelist->get_src()[0].begin()));
    CUGRAPH_EXPECTS(
      thrust::count_if(handle.raft_handle().get_thrust_policy(),
                       edge_first,
                       edge_first + my_edgelist->get_src()[0].size(),
                       [comm_rank,
                        gpu_id_key_func =
                          cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
                            comm_size, major_comm_size, minor_comm_size}] __device__(auto e) {
                         printf("(%d,%d) on rank %d, expected %d\n",
                                (int)thrust::get<0>(e),
                                (int)thrust::get<1>(e),
                                comm_rank,
                                gpu_id_key_func(e));

                         return (gpu_id_key_func(e) != comm_rank);
                       }) == 0,
      "Invalid input argument: edgelist_majors & edgelist_minors should be pre-shuffled.");
  }

  sleep(10);
  // END DEBUGGING

  auto [local_graph, local_edge_weights, local_edge_ids, local_edge_types, local_renumber_map] =
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_id_t,
                                        edge_type_t,
                                        store_transposed,
                                        multi_gpu>(
      handle.raft_handle(),
      std::nullopt,
      std::move(my_edgelist->get_src()[0]),
      std::move(my_edgelist->get_dst()[0]),
      my_edgelist->get_wgt() ? std::make_optional(std::move((*my_edgelist->get_wgt())[0]))
                             : std::nullopt,
      my_edgelist->get_edge_id() ? std::make_optional(std::move((*my_edgelist->get_edge_id())[0]))
                                 : std::nullopt,
      my_edgelist->get_edge_type()
        ? std::make_optional(std::move((*my_edgelist->get_edge_type())[0]))
        : std::nullopt,
      graph_properties,
      renumber,
      // DEBUGGING... force this to true
      true);
  // do_expensive_check);

  std::cout << "after cugraph::create_graph_from_edgelist call, rank = " << handle.get_rank()
            << std::endl;

  (*graph.get_pointer(handle)) = std::move(local_graph);
  if (edge_weights) (*edge_weights->get_pointer(handle)) = std::move(*local_edge_weights);
  if (edge_ids) (*edge_ids->get_pointer(handle)) = std::move(*local_edge_ids);
  if (edge_types) (*edge_types->get_pointer(handle)) = std::move(*local_edge_types);
  if (renumber) (*renumber_map->get_pointer(handle)) = std::move(*local_renumber_map);
}

}  // namespace mtmg
}  // namespace cugraph
