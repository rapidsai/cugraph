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

#include <cugraph/algorithms.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include "iostream"
#include "string"
using namespace std;

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
                    [](string mode) { std::cout << mode << std::endl; });
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
 */

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void look_into_vertex_and_edge_partitions(raft::handle_t const& handle,
                                          std::string const& csv_graph_file_path)
{
  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  std::cout << "Rank_" << comm_rank << ", reading graph from " << csv_graph_file_path << std::endl;

  auto [graph, edge_weights, renumber_map] =
    cugraph::test::read_graph_from_csv_file<vertex_t, edge_t, weight_t, false, multi_gpu>(
      handle, csv_graph_file_path, true, true);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;
  assert(graph_view.local_vertex_partition_range_size() == (*renumber_map).size());

  // Renumber map

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  if (renumber_map) {
    auto renumber_map_title =
      std::string("renumber_map_:").append(std::to_string(comm_rank)).c_str();
    raft::print_device_vector(
      renumber_map_title, (*renumber_map).data(), (*renumber_map).size(), std::cout);
  }

  // Look into edge partitions

  for (size_t ep_idx = 0; ep_idx < graph_view.number_of_local_edge_partitions(); ++ep_idx) {
    // Toplogy
    auto edge_partition = graph_view.local_edge_partition_view(ep_idx);

    auto number_of_edges = edge_partition.number_of_edges();
    auto offsets         = edge_partition.offsets();
    auto indices         = edge_partition.indices();

    auto offsets_title = std::string("offsets_")
                           .append(std::to_string(comm_rank))
                           .append("_")
                           .append(std::to_string(ep_idx))
                           .c_str();
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector(offsets_title, offsets.begin(), offsets.size(), std::cout);

    auto indices_title = std::string("indices_")
                           .append(std::to_string(comm_rank))
                           .append("_")
                           .append(std::to_string(ep_idx))
                           .c_str();
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector(indices_title, indices.begin(), indices.size(), std::cout);

    // Edge property values
    if (edge_weight_view) {
      auto value_firsts = edge_weight_view->value_firsts();
      auto edge_counts  = edge_weight_view->edge_counts();

      assert(number_of_edges == edge_counts[ep_idx]);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto weights_title = std::string("weights_")
                             .append(std::to_string(comm_rank))
                             .append("_")
                             .append(std::to_string(ep_idx))
                             .c_str();
      raft::print_device_vector(
        weights_title, value_firsts[ep_idx], edge_counts[ep_idx], std::cout);
    }
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

  look_into_vertex_and_edge_partitions<vertex_t, edge_t, weight_t, multi_gpu>(*handle,
                                                                              csv_graph_file_path);
}
