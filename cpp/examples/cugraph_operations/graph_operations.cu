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

#include "../tests/utilities/base_fixture.hpp"
#include "../tests/utilities/test_utilities.hpp"

#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <thrust/for_each.h>

#include <iostream>
#include <string>

void initialize_mpi_and_set_device(int argc, char** argv)
{
  RAFT_MPI_TRY(MPI_Init(&argc, &argv));

  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  int comm_size{};
  RAFT_MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

  int num_gpus_per_node{};
  RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  RAFT_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));
}

std::unique_ptr<raft::handle_t> initialize_mg_handle(std::string const& allocation_mode = "cuda")
{
  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  std::set<std::string> possible_allocation_modes = {"cuda", "pool", "binning", "managed"};

  if (possible_allocation_modes.find(allocation_mode) == possible_allocation_modes.end()) {
    if (!comm_rank) {
      std::cout << "'" << allocation_mode
                << "' is not a valid allocation mode. It must be one of the followings -"
                << std::endl;
      std::for_each(possible_allocation_modes.cbegin(),
                    possible_allocation_modes.cend(),
                    [](std::string mode) { std::cout << mode << std::endl; });
    }
    RAFT_MPI_TRY(MPI_Finalize());

    exit(0);
  }

  if (!comm_rank) {
    std::cout << "Using '" << allocation_mode
              << "' allocation mode to create device memory resources." << std::endl;
  }
  std::shared_ptr<rmm::mr::device_memory_resource> resource =
    cugraph::test::create_memory_resource(allocation_mode);
  rmm::mr::set_current_device_resource(resource.get());

  std::unique_ptr<raft::handle_t> handle =
    std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread, resource);

  raft::comms::initialize_mpi_comms(handle.get(), MPI_COMM_WORLD);
  auto& comm           = handle->get_comms();
  auto const comm_size = comm.get_size();

  auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
  while (comm_size % gpu_row_comm_size != 0) {
    --gpu_row_comm_size;
  }

  cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

  return std::move(handle);
}

/**
 * @brief This function reads graph from an input csv file and
 * display vertex and edge partitions.
 */

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void perform_example_graph_operations(raft::handle_t const& handle,
                                      std::string const& csv_graph_file_path,
                                      const bool weighted = false)
{
  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  std::cout << "Rank_" << comm_rank << ", reading graph from " << csv_graph_file_path << std::endl;

  bool renumber = true;  // must be true for distributed graph.

  // Read a graph (along with edge properties e.g. edge weights, if provided) from
  // the input csv file

  auto [graph, edge_weights, renumber_map] =
    cugraph::test::read_graph_from_csv_file<vertex_t, edge_t, weight_t, false, multi_gpu>(
      handle, csv_graph_file_path, weighted, renumber);

  // Non-owning view of the graph object
  auto graph_view = graph.view();
  // Number of vertices mapped to this process, ie the size of
  // the vertex partition assigned to this process

  vertex_t size_of_the_vertex_partition_assigned_to_this_process =
    graph_view.local_vertex_partition_range_size();

  // Non-owning of the edge edge_weights object
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  //
  // As an example operation, compute the weighted average of the properties of
  // neighboring vertices, weighted by the edge weights, if the input graph is weighted;
  // Otherwise, compute the simple average.
  //
  if (weighted) {
    using result_t      = weight_t;
    auto vertex_weights = compute_out_weight_sums(handle, graph_view, *edge_weight_view);

    cugraph::edge_src_property_t<graph_view_t, result_t> src_vertex_weights_cache(handle,
                                                                                  graph_view);

    cugraph::edge_dst_property_t<graph_view_t, result_t> dst_vertex_weights_cache(handle,
                                                                                  graph_view);

    update_edge_src_property(handle, graph_view, vertex_weights.begin(), src_vertex_weights_cache);

    update_edge_dst_property(handle, graph_view, vertex_weights.begin(), dst_vertex_weights_cache);

    rmm::device_uvector<result_t> outputs(size_of_the_vertex_partition_assigned_to_this_process,
                                          handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      src_vertex_weights_cache.view(),
      dst_vertex_weights_cache.view(),
      (*edge_weight_view),
      [] __device__(auto src, auto dst, auto src_prop, auto dst_prop, auto edge_prop) {
        printf("\nsrc ---> %d dst = %d :  src_prop = %f dst_prop = %f edge_prop = %f\n",
               static_cast<int>(src),
               static_cast<int>(dst),
               static_cast<float>(src_prop),
               static_cast<float>(dst_prop),
               static_cast<float>(edge_prop));
        return dst_prop * edge_prop;
      },
      result_t{0},
      cugraph::reduce_op::plus<result_t>{},
      outputs.begin());

    auto outputs_title                 = std::string("outputs_").append(std::to_string(comm_rank));
    size_t max_nr_of_elements_to_print = 10;
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector(outputs_title.c_str(),
                              outputs.begin(),
                              std::min<size_t>(outputs.size(), max_nr_of_elements_to_print),
                              std::cout);
  } else {
    using result_t      = edge_t;
    auto vertex_weights = graph_view.compute_out_degrees(handle);

    cugraph::edge_src_property_t<graph_view_t, result_t> src_vertex_weights_cache(handle,
                                                                                  graph_view);
    cugraph::edge_dst_property_t<graph_view_t, result_t> dst_vertex_weights_cache(handle,
                                                                                  graph_view);

    update_edge_src_property(handle, graph_view, vertex_weights.begin(), src_vertex_weights_cache);

    update_edge_dst_property(handle, graph_view, vertex_weights.begin(), dst_vertex_weights_cache);

    rmm::device_uvector<result_t> outputs(size_of_the_vertex_partition_assigned_to_this_process,
                                          handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      src_vertex_weights_cache.view(),
      dst_vertex_weights_cache.view(),
      cugraph::edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_prop, auto dst_prop, auto) {
        printf("\nsrc ---> %d dst = %d :  src_prop = %f dst_prop = %f\n",
               static_cast<int>(src),
               static_cast<int>(dst),
               static_cast<float>(src_prop),
               static_cast<float>(dst_prop));
        return dst_prop;
      },
      result_t{0},
      cugraph::reduce_op::plus<result_t>{},
      outputs.begin());

    auto outputs_title                 = std::string("outputs_").append(std::to_string(comm_rank));
    size_t max_nr_of_elements_to_print = 10;
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector(outputs_title.c_str(),
                              outputs.begin(),
                              std::min<size_t>(outputs.size(), max_nr_of_elements_to_print),
                              std::cout);
  }
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cout << "Usage: ./sg_examples path_to_your_csv_graph_file [memory allocation mode]"
              << std::endl;
    exit(0);
  }

  std::string const& csv_graph_file_path = argv[1];
  std::string const& allocation_mode     = argc < 3 ? "cuda" : argv[2];

  initialize_mpi_and_set_device(argc, argv);
  std::unique_ptr<raft::handle_t> handle = initialize_mg_handle(allocation_mode);

  auto const comm_rank = handle->get_comms().get_rank();
  auto const comm_size = handle->get_comms().get_size();

  using vertex_t           = int32_t;
  using edge_t             = int32_t;
  using weight_t           = float;
  constexpr bool multi_gpu = true;

  perform_example_graph_operations<vertex_t, edge_t, weight_t, multi_gpu>(
    *handle, csv_graph_file_path, false);
}
