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

#include <cugraph/edge_partition_view.hpp>
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
    if (comm_rank == 0) {
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

  if (comm_rank == 0) {
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

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
void look_into_vertex_and_edge_partitions(raft::handle_t const& handle,
                                          std::string const& csv_graph_file_path,
                                          bool weighted = false)
{
  auto const comm_rank = handle.get_comms().get_rank();

  std::cout << "Rank_" << comm_rank << ", reading graph from " << csv_graph_file_path << std::endl;

  bool renumber = true;  // must be true for multi-gpu

  // Read a graph (along with edge properties e.g. edge weights, if provided) from
  // the input csv file
  auto [graph, edge_weights, renumber_map] =
    cugraph::test::read_graph_from_csv_file<vertex_t, edge_t, weight_t, false, multi_gpu>(
      handle, csv_graph_file_path, weighted, renumber);

  // Meta of the non-owning view of the graph object store vertex/edge partitioning map
  auto graph_view = graph.view();

  // Non-owning of the edge edge_weights object
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  // Total number of vertices
  vertex_t global_number_of_vertices = graph_view.number_of_vertices();

  //
  // Look into vertex partitions
  //

  // Number of vertices mapped to this process, ie the size of
  // the vertex partition assigned to this process
  vertex_t size_of_the_vertex_partition_assigned_to_this_process =
    graph_view.local_vertex_partition_range_size();

  // NOTE: The `renumber_map` contains the vertices assigned to this process.

  //
  // Print verties mapped to this process
  //

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  size_t max_nr_of_elements_to_print = 10;
  if (renumber_map) {
    auto vertex_partition_title = std::string("vertices@rank_").append(std::to_string(comm_rank));
    raft::print_device_vector(vertex_partition_title.c_str(),
                              (*renumber_map).data(),
                              std::min<size_t>((*renumber_map).size(), max_nr_of_elements_to_print),
                              std::cout);
  }

  std::vector<vertex_t> h_vertices_in_this_proces((*renumber_map).size());

  raft::update_host(h_vertices_in_this_proces.data(),
                    (*renumber_map).data(),
                    (*renumber_map).size(),
                    handle.get_stream());
  handle.sync_stream();

  assert(size_of_the_vertex_partition_assigned_to_this_process == (*renumber_map).size());

  // The position of a vertex in the `renumber_map` is indicative of its new (aka renumberd)
  // vertex id. The new (aka renumbered) id of the first vertex, ie the vertex at position 0
  // of `renumber_map`, assigned to this process
  vertex_t renumbered_vertex_id_of_local_first = graph_view.local_vertex_partition_range_first();

  // The new (aka renumbered) id of the last vertex, ie the vertex at position
  // `size_of_the_vertex_partition_assigned_to_this_process` - 1 of `renumber_map`,
  // assigned to this process
  vertex_t renumbered_vertex_id_of_local_last = graph_view.local_vertex_partition_range_last();

  //
  // Print original vertex ids, new (aka renumbered) vertex ids and the ranks of the owner processes
  //

  if (renumber_map) {
    thrust::for_each(thrust::host,
                     thrust::make_zip_iterator(thrust::make_tuple(
                       h_vertices_in_this_proces.begin(),
                       thrust::make_counting_iterator(renumbered_vertex_id_of_local_first))),
                     thrust::make_zip_iterator(thrust::make_tuple(
                       h_vertices_in_this_proces.end(),
                       thrust::make_counting_iterator(renumbered_vertex_id_of_local_last))),
                     [comm_rank](auto old_and_new_id_pair) {
                       auto old_id = thrust::get<0>(old_and_new_id_pair);
                       auto new_id = thrust::get<1>(old_and_new_id_pair);
                       printf("owner rank = %d, original vertex id %d is renumbered to  %d\n",
                              comm_rank,
                              static_cast<int>(old_id),
                              static_cast<int>(new_id));
                     });
  }

  //
  // Look into edge partitions and their associated edge properties (if any)
  //

  bool is_weighted = false;
  if (edge_weight_view.has_value()) { is_weighted = true; }

  for (size_t ep_idx = 0; ep_idx < graph_view.number_of_local_edge_partitions(); ++ep_idx) {
    auto edge_partition_view = graph_view.local_edge_partition_view(ep_idx);

    auto number_of_edges_in_edge_partition = edge_partition_view.number_of_edges();
    auto offsets                           = edge_partition_view.offsets();
    auto indices                           = edge_partition_view.indices();

    assert(number_of_edges_in_edge_partition == indices.size());

    auto major_range_first = edge_partition_view.major_range_first();
    auto major_range_last  = edge_partition_view.major_range_last();

    auto major_hypersparse_first = edge_partition_view.major_hypersparse_first();
    auto dcs_nzd_vertices        = edge_partition_view.dcs_nzd_vertices();

    //
    // Print sources and destinitions of edges stored in current edge partition
    //

    // print sources and destinitions stored in CSR/CSC format
    raft::device_span<weight_t const> weights_of_edges_stored_in_this_edge_partition{};

    if (is_weighted) {
      auto value_firsts = edge_weight_view->value_firsts();
      auto edge_counts  = edge_weight_view->edge_counts();

      weights_of_edges_stored_in_this_edge_partition =
        raft::device_span<weight_t const>(value_firsts[ep_idx], edge_counts[ep_idx]);
    }

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(vertex_t{0}),
      thrust::make_counting_iterator(
        (major_hypersparse_first ? (*major_hypersparse_first) : major_range_last) -
        major_range_first),
      [comm_rank,
       ep_idx,
       offsets,
       indices,
       major_range_first,
       is_weighted,
       weights = weights_of_edges_stored_in_this_edge_partition.begin()] __device__(auto i) {
        auto v                               = major_range_first + i;
        auto deg_of_v_in_this_edge_partition = offsets[i + 1] - offsets[i];

        thrust::for_each(
          thrust::seq,
          thrust::make_counting_iterator(edge_t{offsets[i]}),
          thrust::make_counting_iterator(edge_t{offsets[i + 1]}),
          [comm_rank, ep_idx, v, indices, is_weighted, weights] __device__(auto pos) {
            if (is_weighted) {
              printf(
                "\n[comm_rank = %d local edge partition id = %d]  edge: source = %d "
                "destination = %d weight = %f\n",
                static_cast<int>(comm_rank),
                static_cast<int>(ep_idx),
                static_cast<int>(v),
                static_cast<int>(indices[pos]),
                static_cast<float>(weights[pos]));

            } else {
              printf(
                "\n[comm_rank = %d local edge partition id = %d]  edge: source = %d "
                "destination = %d\n",
                static_cast<int>(comm_rank),
                static_cast<int>(ep_idx),
                static_cast<int>(v),
                static_cast<int>(indices[pos]));
            }
          });
      });

    // print sources and destinitions stored in DCSR/DCSC format
    if (major_hypersparse_first.has_value()) {
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(static_cast<vertex_t>((*dcs_nzd_vertices).size())),
        [comm_rank,
         ep_idx,
         offsets,
         indices,
         major_range_first,
         is_weighted,
         weights                 = weights_of_edges_stored_in_this_edge_partition.begin(),
         dcs_nzd_vertices        = (*dcs_nzd_vertices),
         major_hypersparse_first = (*major_hypersparse_first)] __device__(auto i) {
          auto v                               = dcs_nzd_vertices[i];
          auto major_idx                       = (major_hypersparse_first - major_range_first) + i;
          auto deg_of_v_in_this_edge_partition = offsets[major_idx + 1] - offsets[major_idx];

          thrust::for_each(
            thrust::seq,
            thrust::make_counting_iterator(edge_t{offsets[major_idx]}),
            thrust::make_counting_iterator(edge_t{offsets[major_idx + 1]}),
            [comm_rank, ep_idx, v, indices, is_weighted, weights] __device__(auto pos) {
              if (is_weighted) {
                printf(
                  "\n[comm_rank = %d local edge partition id = %d]  edge: source = %d "
                  "destination = %d weight = %f\n",
                  static_cast<int>(comm_rank),
                  static_cast<int>(ep_idx),
                  static_cast<int>(v),
                  static_cast<int>(indices[pos]),
                  static_cast<float>(weights[pos]));

              } else {
                printf(
                  "\n[comm_rank = %d local edge partition id = %d]  edge: source = %d "
                  "destination = %d\n",
                  static_cast<int>(comm_rank),
                  static_cast<int>(ep_idx),
                  static_cast<int>(v),
                  static_cast<int>(indices[pos]));
              }
            });
        });
    }

    // Edge property values
    if (edge_weight_view) {
      auto value_firsts = edge_weight_view->value_firsts();
      auto edge_counts  = edge_weight_view->edge_counts();

      assert(number_of_edges_in_edge_partition == edge_counts[ep_idx]);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto weights_title = std::string("weights_")
                             .append(std::to_string(comm_rank))
                             .append("_")
                             .append(std::to_string(ep_idx));
      raft::print_device_vector(
        weights_title.c_str(),
        value_firsts[ep_idx],
        std::min<size_t>(number_of_edges_in_edge_partition, max_nr_of_elements_to_print),
        std::cout);
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

  using vertex_t                  = int32_t;
  using edge_t                    = int32_t;
  using weight_t                  = float;
  constexpr bool multi_gpu        = true;
  constexpr bool store_transposed = false;

  look_into_vertex_and_edge_partitions<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    *handle, csv_graph_file_path, false);
}
