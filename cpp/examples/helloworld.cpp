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

#include <cugraph/algorithms.hpp>
// #include </home/mnaim/cugraph/cpp/tests/utilities/test_utilities.hpp>

#include "../tests/utilities/test_utilities.hpp"

#include <cugraph/detail/shuffle_wrappers.hpp>

#include <cugraph/graph_functions.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <raft/random/rng_state.hpp>

#include "iostream"
#include "string"
using namespace std;

/**
 * @brief This function initializes RAFT handle object that encapsulates CUDA stream, communicator
 */

std::unique_ptr<raft::handle_t> initialize_mg_handle(size_t pool_size = 64)
{
  std::unique_ptr<raft::handle_t> handle{nullptr};

  handle = std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread,
                                            std::make_shared<rmm::cuda_stream_pool>(pool_size));

  raft::comms::initialize_mpi_comms(handle.get(), MPI_COMM_WORLD);
  auto& comm           = handle->get_comms();
  auto const comm_size = comm.get_size();

  std::cout << comm.get_rank() << " " << comm_size << std::endl;

  auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
  while (comm_size % gpu_row_comm_size != 0) {
    --gpu_row_comm_size;
  }

  cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

  return std::move(handle);
}

/**
 * @brief Create a graph from input sources, destinitions, and optional weights
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>

std::tuple<
  cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
create_graph(raft::handle_t const& handle,
             std::vector<vertex_t>&& edge_srcs,
             std::vector<vertex_t>&& edge_dsts,
             std::optional<std::vector<weight_t>>&& edge_wgts,
             bool renumber,
             bool is_symmetric)
{
  std::cout << "\n>>>>>>>>>>>> Inside create graph \n";
  constexpr bool store_transposed = false;

  assert(edge_dsts.size() == edge_srcs.size());
  assert((*edge_wgts).size() == edge_srcs.size());
  size_t num_edges = edge_srcs.size();

  std::cout << "#E = " << num_edges << std::endl;

  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();
  std::cout << "Comm size: " << comm_size << std::endl;

  // To keep it simple, each GPU will own part of the edge-list to start with
  auto start = comm_rank * (num_edges / comm_size) +
               (comm_rank < (num_edges % comm_size) ? comm_rank : num_edges % comm_size);
  auto end = start + (num_edges / comm_size) + ((comm_rank + 1) <= (num_edges % comm_size) ? 1 : 0);
  auto work_size = end - start;

  std::cout << "Edges for GPU_" << comm_rank << " starts at " << start << " ends at " << end
            << std::endl;

  rmm::device_uvector<vertex_t> d_src_v(work_size, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(work_size, handle.get_stream());

  auto d_wgt_v = edge_wgts.has_value() ? std::make_optional<rmm::device_uvector<weight_t>>(
                                           work_size, handle.get_stream())
                                       : std::nullopt;

  raft::update_device(d_src_v.data(), edge_srcs.data() + start, work_size, handle.get_stream());
  raft::update_device(d_dst_v.data(), edge_dsts.data() + start, work_size, handle.get_stream());
  if (d_wgt_v) {
    raft::update_device(
      (*d_wgt_v).data(), (*edge_wgts).data() + start, work_size, handle.get_stream());
  }

  // Print edges
  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      std::cout << "rank " << r << " : "
                << "is_symmetric: " << is_symmetric << std::endl;
      auto src_title = std::string("d_src_v:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(src_title, d_src_v.data(), d_src_v.size(), std::cout);

      auto dst_title = std::string("d_dst_v:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(dst_title, d_dst_v.data(), d_dst_v.size(), std::cout);

      if (d_wgt_v) {
        auto wgt_title = std::string("d_wgt_v:").append(std::to_string(comm_rank)).c_str();
        raft::print_device_vector(wgt_title, d_wgt_v->data(), d_wgt_v->size(), std::cout);
      }
    }
  }

  // Shuffle edges

  if (multi_gpu) {
    std::tie(store_transposed ? d_dst_v : d_src_v,
             store_transposed ? d_src_v : d_dst_v,
             d_wgt_v,
             std::ignore,
             std::ignore) =
      cugraph::detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
        vertex_t,
        vertex_t,
        weight_t,
        int32_t>(handle,
                 store_transposed ? std::move(d_dst_v) : std::move(d_src_v),
                 store_transposed ? std::move(d_src_v) : std::move(d_dst_v),
                 std::move(d_wgt_v),
                 std::nullopt,
                 std::nullopt);
  }

  // Print Shuffled edges
  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      std::cout << "rank " << r << " : "
                << "is_symmetric: " << is_symmetric << std::endl;
      auto src_title = std::string("d_src_v:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(src_title, d_src_v.data(), d_src_v.size(), std::cout);

      auto dst_title = std::string("d_dst_v:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(dst_title, d_dst_v.data(), d_dst_v.size(), std::cout);

      if (d_wgt_v) {
        auto wgt_title = std::string("d_wgt_v:").append(std::to_string(comm_rank)).c_str();
        raft::print_device_vector(wgt_title, d_wgt_v->data(), d_wgt_v->size(), std::cout);
      }
    }
  }

  // Create graph
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);

  std::optional<cugraph::edge_property_t<decltype(graph.view()), weight_t>> edge_weights{
    std::nullopt};

  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, edge_weights, std::ignore, std::ignore, renumber_map) =
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_t,
                                        int32_t,
                                        store_transposed,
                                        multi_gpu>(handle,
                                                   std::nullopt,
                                                   std::move(d_src_v),
                                                   std::move(d_dst_v),
                                                   std::move(d_wgt_v),
                                                   std::nullopt,
                                                   std::nullopt,
                                                   cugraph::graph_properties_t{is_symmetric, false},
                                                   renumber,
                                                   true);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

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
        auto offsets_title = std::string("o_r_")
                               .append(std::to_string(comm_rank))
                               .append("_ep_")
                               .append(std::to_string(ep_idx))
                               .c_str();
        raft::print_device_vector(offsets_title, offsets.begin(), offsets.size(), std::cout);

        auto indices_title = std::string("i_r_")
                               .append(std::to_string(comm_rank))
                               .append("_ep_")
                               .append(std::to_string(ep_idx))
                               .c_str();
        raft::print_device_vector(indices_title, indices.begin(), indices.size(), std::cout);

        // Edge property values
        if (edge_weight_view) {
          auto value_firsts = edge_weight_view->value_firsts();
          auto edge_counts  = edge_weight_view->edge_counts();

          assert(number_of_edges == edge_counts[ep_idx]);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          auto weights_title = std::string("w_r_")
                                 .append(std::to_string(comm_rank))
                                 .append("_ep_")
                                 .append(std::to_string(ep_idx))
                                 .c_str();
          raft::print_device_vector(
            weights_title, value_firsts[ep_idx], edge_counts[ep_idx], std::cout);
        }
      }
    }
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
  }

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

void run_graph_algos(raft::handle_t const& handle, std::string const& csv_graph_file)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  constexpr bool multi_gpu    = true;
  constexpr bool renumber     = true;
  constexpr bool is_symmetric = true;

  // Edge-list to create a graph.
  std::vector<vertex_t> edge_srcs = {3, 5, 7, 8, 8, 8};
  std::vector<vertex_t> edge_dsts = {8, 8, 8, 3, 5, 7};
  std::vector<weight_t> edge_wgts = {38.0, 58.0, 78.0, 38.0, 58.0, 78.0};

  // std::vector<vertex_t> edge_srcs = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  // std::vector<vertex_t> edge_dsts = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  // std::vector<weight_t> edge_wgts = {
  //   0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f,
  //   0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  auto [d_src_v, d_dst_v, d_wgt_v, is_input_symmetric] =
    cugraph::test::read_edgelist_from_csv_file<vertex_t, weight_t>(
      handle, csv_graph_file, true, false, multi_gpu);

  for (size_t r = 0; r < comm_size; r++) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    if (comm_rank == r) {
      std::cout << "rank " << r << " : "
                << "is_symmetric: " << is_input_symmetric << std::endl;
      auto src_title = std::string("d_src_v:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(src_title, d_src_v.data(), d_src_v.size(), std::cout);

      auto dst_title = std::string("d_dst_v:").append(std::to_string(comm_rank)).c_str();
      raft::print_device_vector(dst_title, d_dst_v.data(), d_dst_v.size(), std::cout);

      if (d_wgt_v) {
        auto wgt_title = std::string("d_wgt_v:").append(std::to_string(comm_rank)).c_str();
        raft::print_device_vector(wgt_title, d_wgt_v->data(), d_wgt_v->size(), std::cout);
      }
    }
  }

  // Create graph
  cugraph::graph_t<vertex_t, edge_t, false, multi_gpu> graph(handle);
  std::optional<cugraph::edge_property_t<decltype(graph.view()), weight_t>> edge_weights{
    std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, edge_weights, renumber_map) =
    create_graph<vertex_t, edge_t, weight_t, multi_gpu>(handle,
                                                        std::move(edge_srcs),
                                                        std::move(edge_dsts),
                                                        std::make_optional(std::move(edge_wgts)),
                                                        renumber,
                                                        is_symmetric);

  std::tie(graph, edge_weights, renumber_map) =
    cugraph::test::read_graph_from_csv_file<vertex_t, edge_t, weight_t, false, multi_gpu>(
      handle, csv_graph_file, true, true);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

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
        auto offsets_title = std::string("off_r_")
                               .append(std::to_string(comm_rank))
                               .append("_ep_")
                               .append(std::to_string(ep_idx))
                               .c_str();
        raft::print_device_vector(offsets_title, offsets.begin(), offsets.size(), std::cout);

        auto indices_title = std::string("ind_r_")
                               .append(std::to_string(comm_rank))
                               .append("_ep_")
                               .append(std::to_string(ep_idx))
                               .c_str();
        raft::print_device_vector(indices_title, indices.begin(), indices.size(), std::cout);

        // Edge property values
        if (edge_weight_view) {
          auto value_firsts = edge_weight_view->value_firsts();
          auto edge_counts  = edge_weight_view->edge_counts();

          assert(number_of_edges == edge_counts[ep_idx]);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          auto weights_title = std::string("wgts_r_")
                                 .append(std::to_string(comm_rank))
                                 .append("_ep_")
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
      auto out_degrees_title =
        std::string("out_degrees:").append(std::to_string(comm_rank)).c_str();

      raft::print_device_vector(
        in_degrees_title, d_in_degrees.data(), d_in_degrees.size(), std::cout);
      raft::print_device_vector(
        out_degrees_title, d_out_degrees.data(), d_out_degrees.size(), std::cout);
    }
  }

  assert(graph_view.local_vertex_partition_range_size() == (*renumber_map).size());

  // Run a cuGaraph algorithm
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
int main(int argc, char** argv)
{
  initialize_mpi_and_set_device(argc, argv);

  // auto resource = std::make_shared<rmm::mr::cuda_memory_resource>();
  // rmm::mr::set_current_device_resource(resource.get());

  raft::handle_t handle = *(initialize_mg_handle());

  run_graph_algos(handle, "graph.csv");

  RAFT_MPI_TRY(MPI_Finalize());
}