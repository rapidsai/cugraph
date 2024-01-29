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

#include "../tests/utilities/test_utilities.hpp"
#include <cugraph/algorithms.hpp>

// #include <cugraph/detail/shuffle_wrappers.hpp>

// #include <cugraph/graph_functions.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include "iostream"
#include "string"
using namespace std;

/**
 * @brief This function initializes MPI and set device for each MPI process.
 */

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

/**
 * @brief This function initializes RAFT handle object that encapsulates CUDA stream,
 * communicator and handles to various CUDA libraries.
 */

std::unique_ptr<raft::handle_t> initialize_handle(size_t pool_size = 64)
{
  std::unique_ptr<raft::handle_t> handle{nullptr};

  // auto resource = std::make_shared<rmm::mr::cuda_memory_resource>();
  // rmm::mr::set_current_device_resource(resource.get());

  handle = std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread,
                                            std::make_shared<rmm::cuda_stream_pool>(pool_size));

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
 * @brief This function reads graph and edge properties from an input csv file.
 */

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
read_graph_and_edge_property(raft::handle_t const& handle,
                             std::string const& csv_graph_file_path,
                             bool renumber = true)
{
  constexpr bool is_symmetric = true;

  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  // Create graph
  cugraph::graph_t<vertex_t, edge_t, false, multi_gpu> graph(handle);
  std::optional<cugraph::edge_property_t<decltype(graph.view()), weight_t>> edge_weights{
    std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, edge_weights, renumber_map) =
    cugraph::test::read_graph_from_csv_file<vertex_t, edge_t, weight_t, false, multi_gpu>(
      handle, csv_graph_file_path, true, true);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  assert(graph_view.local_vertex_partition_range_size() == (*renumber_map).size());
  // Renumber map
  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      if (renumber_map) {
        std::cout << "rank " << r << " : " << std::endl;
        auto renumber_map_title =
          std::string("renumber_map:").append(std::to_string(comm_rank)).c_str();
        raft::print_device_vector(
          renumber_map_title, (*renumber_map).data(), (*renumber_map).size(), std::cout);
      }
    }
  }

  // Look into edge partitions
  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      for (size_t ep_idx = 0; ep_idx < graph_view.number_of_local_edge_partitions(); ++ep_idx) {
        // Toplogy
        auto edge_partition = graph_view.local_edge_partition_view(ep_idx);

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        std::cout << "rank: " << r << " ep: " << ep_idx
                  << ", #edges = " << edge_partition.number_of_edges() << std::endl;

        auto number_of_edges = edge_partition.number_of_edges();
        auto offsets         = edge_partition.offsets();
        auto indices         = edge_partition.indices();

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        auto offsets_title = std::string("offsets_")
                               .append(std::to_string(comm_rank))
                               .append(std::to_string(ep_idx))
                               .c_str();
        raft::print_device_vector(offsets_title, offsets.begin(), offsets.size(), std::cout);

        auto indices_title = std::string("indices_")
                               .append(std::to_string(comm_rank))
                               .append(std::to_string(ep_idx))
                               .c_str();
        raft::print_device_vector(indices_title, indices.begin(), indices.size(), std::cout);

        // Edge property values
        if (edge_weight_view) {
          auto value_firsts = edge_weight_view->value_firsts();
          auto edge_counts  = edge_weight_view->edge_counts();

          assert(number_of_edges == edge_counts[ep_idx]);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          auto weights_title = std::string("weights_")
                                 .append(std::to_string(comm_rank))
                                 .append(std::to_string(ep_idx))
                                 .c_str();
          raft::print_device_vector(
            weights_title, value_firsts[ep_idx], edge_counts[ep_idx], std::cout);
        }
      }
    }
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
  }

  // Compute out degrees

  auto d_in_degrees  = graph_view.compute_in_degrees(handle);
  auto d_out_degrees = graph_view.compute_out_degrees(handle);

  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      std::cout << "rank " << r << " : " << std::endl;

      auto in_degrees_title = std::string("in_degrees:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(
        in_degrees_title, d_in_degrees.data(), d_in_degrees.size(), std::cout);
      auto out_degrees_title =
        std::string("out_degrees:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(
        out_degrees_title, d_out_degrees.data(), d_out_degrees.size(), std::cout);
    }
  }

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void run_graph_algos(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  // Run a cuGaraph algorithm

  rmm::device_uvector<vertex_t> d_distances(graph_view.local_vertex_partition_range_size(),
                                            handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.local_vertex_partition_range_size(),
                                               handle.get_stream());

  rmm::device_uvector<vertex_t> d_sources(2, handle.get_stream());
  std::vector<vertex_t> h_sources = {0, 1};
  raft::update_device(d_sources.data(), h_sources.data(), h_sources.size(), handle.get_stream());

  cugraph::bfs(handle,
               graph_view,
               d_distances.data(),
               d_predecessors.data(),
               d_sources.data(),
               d_sources.size(),
               false,
               std::numeric_limits<vertex_t>::max());

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  raft::print_device_vector("d_distances", d_distances.data(), d_distances.size(), std::cout);

  raft::print_device_vector(
    "d_predecessors", d_predecessors.data(), d_predecessors.size(), std::cout);

  rmm::device_uvector<vertex_t> d_sg_cluster_v(graph_view.local_vertex_partition_range_size(),
                                               handle.get_stream());

  weight_t threshold  = 1e-7;
  weight_t resolution = 1.0;
  size_t max_level    = 10;

  weight_t modularity{-1.0};
  std::tie(std::ignore, modularity) =
    cugraph::louvain(handle,
                     std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
                     graph_view,
                     edge_weight_view,
                     d_sg_cluster_v.data(),
                     max_level,
                     threshold,
                     resolution);

  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      std::cout << "rank " << r << " : " << std::endl;
      auto cluster_title = std::string("cluster_v:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(
        cluster_title, d_sg_cluster_v.data(), d_sg_cluster_v.size(), std::cout);
    }
  }

  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      std::cout << "rank : " << comm_rank << ", modularity : " << modularity << std::endl;
    }
  }
}

int main(int argc, char** argv)
{
  initialize_mpi_and_set_device(argc, argv);
  std::unique_ptr<raft::handle_t> handle = initialize_handle();

  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  if (argc < 2) {
    std::cout << "./example1 [full path to your csv graph file]\n";
    std::cout << "mpirun -np [number of process] ./example1 [full path to your csv graph file]\n";

    RAFT_MPI_TRY(MPI_Finalize());
    exit(0);
  }
  std::cout << argv[1] << "\n";
  std::string const& csv_graph_file_path = argv[1];
  constexpr bool multi_gpu               = true;
  auto [graph, edge_weights, renumber_map] =
    read_graph_and_edge_property<vertex_t, edge_t, weight_t, multi_gpu>(*handle,
                                                                        csv_graph_file_path);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  run_graph_algos<vertex_t, edge_t, weight_t, multi_gpu>(*handle, graph_view, edge_weight_view);

  RAFT_MPI_TRY(MPI_Finalize());
}