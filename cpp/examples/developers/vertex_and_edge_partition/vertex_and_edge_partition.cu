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

#include <cugraph/edge_partition_view.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
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

std::unique_ptr<raft::handle_t> initialize_mg_handle()
{
  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  std::shared_ptr<rmm::mr::device_memory_resource> resource =
    std::make_shared<rmm::mr::cuda_memory_resource>();
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
 * @brief Create a graph from edge sources, destinations, and optional weights
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>

std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<cugraph::edge_property_t<
             cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
             weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
create_graph(raft::handle_t const& handle,
             std::vector<vertex_t>&& edge_srcs,
             std::vector<vertex_t>&& edge_dsts,
             std::optional<std::vector<weight_t>>&& edge_wgts,
             bool renumber,
             bool is_symmetric)
{
  size_t num_edges = edge_srcs.size();
  assert(edge_dsts.size() == num_edges);
  if (edge_wgts.has_value()) { assert((*edge_wgts).size() == num_edges); }

  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  //
  // Assign part of the edge list to each GPU. If there are N edges and P GPUs, each GPU except the
  // one with rank P-1 reads N/P edges and the GPU with rank P -1 reads (N/P + N%P) edges into GPU
  // memory.
  //

  auto start = comm_rank * (num_edges / comm_size);
  auto end   = (comm_rank + 1) * (num_edges / comm_size);
  if (comm_rank == comm_size - 1) { end = num_edges; }

  auto work_size = end - start;
  rmm::device_uvector<vertex_t> d_edge_srcs(work_size, handle.get_stream());
  rmm::device_uvector<vertex_t> d_edge_dsts(work_size, handle.get_stream());

  auto d_edge_wgts = edge_wgts.has_value() ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               work_size, handle.get_stream())
                                           : std::nullopt;

  raft::update_device(d_edge_srcs.data(), edge_srcs.data() + start, work_size, handle.get_stream());
  raft::update_device(d_edge_dsts.data(), edge_dsts.data() + start, work_size, handle.get_stream());
  if (d_edge_wgts.has_value()) {
    raft::update_device(
      (*d_edge_wgts).data(), (*edge_wgts).data() + start, work_size, handle.get_stream());
  }

  //
  // In cugraph, each vertex and edge is assigned to a specific GPU using hash functions. Before
  // creating a graph from edges, we need to ensure that all edges are already assigned to the
  // proper GPU.
  //

  if (multi_gpu) {
    std::tie(d_edge_srcs, d_edge_dsts, d_edge_wgts, std::ignore, std::ignore) =
      cugraph::shuffle_external_edges<vertex_t, vertex_t, weight_t, int32_t>(handle,
                                                                             std::move(d_edge_srcs),
                                                                             std::move(d_edge_dsts),
                                                                             std::move(d_edge_wgts),
                                                                             std::nullopt,
                                                                             std::nullopt);
  }

  //
  // Create graph
  //

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
                                                   std::move(d_edge_srcs),
                                                   std::move(d_edge_dsts),
                                                   std::move(d_edge_wgts),
                                                   std::nullopt,
                                                   std::nullopt,
                                                   cugraph::graph_properties_t{is_symmetric, false},
                                                   renumber,
                                                   true);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

/**
 * @brief This function prints vertex and edge partitions.
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
void look_into_vertex_and_edge_partitions(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<rmm::device_uvector<vertex_t>>&& renumber_map)
{
  auto const comm_rank = handle.get_comms().get_rank();

  // Total number of vertices
  vertex_t global_number_of_vertices = graph_view.number_of_vertices();

  //
  // Look into vertex partitions
  //

  // Number of vertices mapped to this process, ie the size of
  // the vertex partition assigned to this process. We are using
  // one-process-per-GPU model
  vertex_t size_of_the_vertex_partition_assigned_to_this_process =
    graph_view.local_vertex_partition_range_size();

  // The `renumber_map` contains the vertices assigned to this process.
  // Print verties mapped to this process
  //

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  if (renumber_map) {
    auto vertex_partition_title = std::string("vertices@rank_").append(std::to_string(comm_rank));
    raft::print_device_vector(
      vertex_partition_title.c_str(), (*renumber_map).data(), (*renumber_map).size(), std::cout);
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
    // Print sources and destinations of edges stored in current edge partition
    //

    // print sources and destinations stored in CSR/CSC format
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

    // print sources and destinations stored in DCSR/DCSC format
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
        weights_title.c_str(), value_firsts[ep_idx], number_of_edges_in_edge_partition, std::cout);
    }
  }
}

int main(int argc, char** argv)
{
  initialize_mpi_and_set_device(argc, argv);
  std::unique_ptr<raft::handle_t> handle = initialize_mg_handle();

  using vertex_t    = int32_t;
  using edge_t      = int32_t;
  using weight_t    = float;
  bool is_symmetric = true;

  std::vector<vertex_t> edge_srcs = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<vertex_t> edge_dsts = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<weight_t> edge_wgts = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  constexpr bool multi_gpu        = true;
  constexpr bool store_transposed = false;
  bool renumber                   = true;  // must be true for multi-GPU applications

  auto [graph, edge_weights, renumber_map] =
    create_graph<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      *handle,
      std::move(edge_srcs),
      std::move(edge_dsts),
      std::move(std::make_optional(edge_wgts)),
      renumber,
      is_symmetric);

  // Non-owning view of the graph object
  auto graph_view = graph.view();

  // Non-owning of the edge_weights object
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  look_into_vertex_and_edge_partitions<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    *handle, graph_view, edge_weight_view, std::move(renumber_map));
}
