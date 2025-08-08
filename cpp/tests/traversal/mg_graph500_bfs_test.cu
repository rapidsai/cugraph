/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "detail/graph500_forest_pruning_utils.cuh"
#include "detail/graph_partition_utils.cuh"
#include "detail/shuffle_wrappers.hpp"
#include "nbr_unrenumber_cache.cuh"
#include "prims/count_if_e.cuh"
#include "prims/extract_transform_if_e.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/kv_store.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/collect_comm.cuh"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <gtest/gtest.h>

#include <random>

struct Graph500_BFS_Usecase {
  bool use_pruned_graph_unrenumber_cache{
    false};  // use cache to locally unrenumber (at the expense of additional memory usage)
  bool use_large_buffer{false};
  bool validate{true};
};

void init_nccl_env_variables() {}

template <typename input_usecase_t>
class Tests_GRAPH500_MGBFS
  : public ::testing::TestWithParam<std::tuple<Graph500_BFS_Usecase, input_usecase_t>> {
 public:
  Tests_GRAPH500_MGBFS() {}

  static void SetUpTestCase()
  {
    init_nccl_env_variables();

    size_t pool_size =
      12;  // note that CUDA_DEVICE_MAX_CONNECTIONS (default: 8) should be set to a value larger
           // than pool_size to avoid false dependency among different streams
    handle_ = cugraph::test::initialize_mg_handle(pool_size);

    cugraph::large_buffer_manager::init(
      *handle_, cugraph::large_buffer_manager::create_memory_buffer_resource(), std::nullopt);
  }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(Graph500_BFS_Usecase const& bfs_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t    = float;    // dummy
    using edge_type_t = int32_t;  // dummy
    using edge_time_t = int32_t;  // dummy

    bool constexpr store_transposed = false;
    bool constexpr multi_gpu        = true;
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = false;
    bool constexpr shuffle = false;  // Graph 500 requirement (edges can't be pre-shuffled, edges
                                     // should be shuffled in Kernel 1)
    size_t num_warmup_starting_vertices = 1;   // to enforce all CUDA & NCCL initializations
    size_t num_timed_starting_vertices  = 64;  // Graph 500 requirement (64)

    HighResTimer hr_timer{};
    raft::random::RngState rng_state{0};

    auto& comm           = handle_->get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();
    auto& major_comm     = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();
    auto vertex_partition_id =
      cugraph::partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);

    std::cout << "comm_size=" << comm_size << " major_comm_size=" << major_comm_size
              << " minor_comm_size=" << minor_comm_size << std::endl;

    constexpr auto invalid_distance = std::numeric_limits<vertex_t>::max();
    constexpr auto invalid_vertex   = cugraph::invalid_vertex_id<vertex_t>::value;

    // 1. force NCCL P2P initialization

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.start("NCCL P2P buffer initialization");
    }

    cugraph::test::enforce_p2p_initialization(comm, handle_->get_stream());
    cugraph::test::enforce_p2p_initialization(major_comm, handle_->get_stream());
    cugraph::test::enforce_p2p_initialization(minor_comm, handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 2. create a graph & meta data

    rmm::device_uvector<vertex_t> mg_renumber_map(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> components(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> parents(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> unrenumbered_parents(0, handle_->get_stream());
    std::vector<vertex_t> vertex_partition_range_offsets(comm_size + 1);
    vertex_t local_vertex_partition_range_first{};
    vertex_t local_vertex_partition_range_last{};

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_pruned_graph(*handle_);
    rmm::device_uvector<vertex_t> mg_pruned_graph_renumber_map(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> mg_graph_to_pruned_graph_map(
      0, handle_->get_stream());  // we may store this in host buffer to save HBM
    rmm::device_uvector<vertex_t> mg_pruned_graph_to_graph_map(0, handle_->get_stream());

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_isolated_trees(*handle_);
    rmm::device_uvector<vertex_t> mg_isolated_trees_renumber_map(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> mg_graph_to_isolated_trees_map(
      0, handle_->get_stream());  // we may store this in host buffer to save HBM
    rmm::device_uvector<vertex_t> mg_isolated_trees_to_graph_map(0, handle_->get_stream());

    std::optional<cugraph::test::nbr_unrenumber_cache_t<vertex_t>>
      mg_pruned_graph_pred_unrenumber_cache{std::nullopt};
    {
      edge_t num_input_edges{};
      edge_t num_edges{};  // after removing self-loops and multi-edges

      // 2-1. create an edge list

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.start("MG Construct edge list");
      }

      std::vector<rmm::device_uvector<vertex_t>> src_chunks{};
      std::vector<rmm::device_uvector<vertex_t>> dst_chunks{};
      std::tie(src_chunks, dst_chunks, std::ignore, std::ignore, std::ignore) =
        input_usecase.template construct_edgelist<vertex_t, weight_t>(
          *handle_, test_weighted, store_transposed, multi_gpu, shuffle);
      ASSERT_TRUE(input_usecase.undirected());

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      num_input_edges = 0;
      for (size_t i = 0; i < src_chunks.size(); ++i) {
        num_input_edges += static_cast<edge_t>(src_chunks[i].size());
      }
      num_input_edges = cugraph::host_scalar_allreduce(
        comm, num_input_edges, raft::comms::op_t::SUM, handle_->get_stream());

      // 2-2. create an MG graph

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.start("MG Construct graph (Kernel 1)");
      }

      for (size_t i = 0; i < src_chunks.size(); ++i) {
        std::tie(src_chunks[i],
                 dst_chunks[i],
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore) =
          cugraph::remove_self_loops<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
            *handle_,
            std::move(src_chunks[i]),
            std::move(dst_chunks[i]),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);

        std::vector<cugraph::arithmetic_device_uvector_t> dummy_edge_property_chunk{};

        std::tie(src_chunks[i], dst_chunks[i], dummy_edge_property_chunk, std::ignore) =
          cugraph::shuffle_ext_edges(*handle_,
                                     std::move(src_chunks[i]),
                                     std::move(dst_chunks[i]),
                                     std::move(dummy_edge_property_chunk),
                                     store_transposed);
      }

      std::tie(
        src_chunks, dst_chunks, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          *handle_,
          std::move(src_chunks),
          std::move(dst_chunks),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          true /* keep_min_value_edge */);

      num_edges = 0;
      for (size_t i = 0; i < src_chunks.size(); ++i) {
        num_edges += static_cast<edge_t>(src_chunks[i].size());
      }
      num_edges = cugraph::host_scalar_allreduce(
        comm, num_edges, raft::comms::op_t::SUM, handle_->get_stream());

      cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_graph(*handle_);
      std::optional<rmm::device_uvector<vertex_t>> tmp_map{};
      std::tie(mg_graph, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, tmp_map) =
        cugraph::create_graph_from_edgelist<vertex_t,
                                            edge_t,
                                            weight_t,
                                            edge_type_t,
                                            edge_time_t,
                                            store_transposed,
                                            multi_gpu>(
          *handle_,
          std::nullopt,
          std::move(src_chunks),
          std::move(dst_chunks),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cugraph::graph_properties_t{true /* symmetric */, false /* multi-graph */},
          renumber);
      mg_renumber_map = std::move(*tmp_map);
      {
        auto mg_graph_view = mg_graph.view();
        auto offsets       = mg_graph_view.vertex_partition_range_offsets();
        std::copy(offsets.begin(), offsets.end(), vertex_partition_range_offsets.begin());
        local_vertex_partition_range_first = mg_graph_view.local_vertex_partition_range_first();
        local_vertex_partition_range_last  = mg_graph_view.local_vertex_partition_range_last();
      }

      // 2-3. Forest pruning

      {
        auto mg_graph_view = mg_graph.view();

        components.resize(mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
        cugraph::weakly_connected_components(
          *handle_, mg_graph_view, components.data(), components.size());
        parents = find_trees_from_2cores(*handle_, mg_graph_view, invalid_vertex);
      }

      std::optional<rmm::device_uvector<vertex_t>> tmp_components{std::nullopt};
      if (bfs_usecase
            .use_large_buffer) {  // temporarily store components in host buffer to free up HBM
                                  // before extracting sub-graphs (which uses a lot of HBM)
        tmp_components = cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
          components.size(), handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     components.begin(),
                     components.end(),
                     tmp_components->begin());
        components.resize(0, handle_->get_stream());
        components.shrink_to_fit(handle_->get_stream());
      }

      std::tie(mg_pruned_graph,
               std::ignore,
               mg_pruned_graph_renumber_map,
               mg_graph_to_pruned_graph_map,
               mg_pruned_graph_to_graph_map,
               mg_isolated_trees,
               std::ignore,
               mg_isolated_trees_renumber_map,
               mg_graph_to_isolated_trees_map,
               mg_isolated_trees_to_graph_map) =
        extract_forest_pruned_graph_and_isolated_trees<vertex_t,
                                                       edge_t,
                                                       weight_t,
                                                       edge_type_t,
                                                       edge_time_t,
                                                       store_transposed,
                                                       multi_gpu>(
          *handle_,
          std::move(mg_graph),
          std::nullopt,
          raft::device_span<vertex_t const>(mg_renumber_map.data(), mg_renumber_map.size()),
          raft::device_span<vertex_t const>(parents.data(), parents.size()),
          invalid_vertex,
          bfs_usecase.use_large_buffer ? std::make_optional(cugraph::large_buffer_type_t::MEMORY)
                                       : std::nullopt);

      if (bfs_usecase.use_large_buffer) {
        components.resize(tmp_components->size(), handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     tmp_components->begin(),
                     tmp_components->end(),
                     components.begin());
        tmp_components = std::nullopt;
      }

      unrenumbered_parents.resize(parents.size(), handle_->get_stream());
      thrust::copy(
        handle_->get_thrust_policy(), parents.begin(), parents.end(), unrenumbered_parents.begin());
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        *handle_,
        unrenumbered_parents.data(),
        unrenumbered_parents.size(),
        mg_renumber_map.data(),
        raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                        vertex_partition_range_offsets.size() - 1));

      if (bfs_usecase.use_pruned_graph_unrenumber_cache) {
        mg_pruned_graph_pred_unrenumber_cache = cugraph::test::build_nbr_unrenumber_cache(
          *handle_,
          mg_pruned_graph.view(),
          raft::device_span<vertex_t const>(mg_pruned_graph_renumber_map.data(),
                                            mg_pruned_graph_renumber_map.size()),
          invalid_vertex,
          bfs_usecase.use_large_buffer ? std::make_optional(cugraph::large_buffer_type_t::MEMORY)
                                       : std::nullopt);
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      std::cout << "num_input_edges=" << num_input_edges
                << " V=" << vertex_partition_range_offsets.back() << " E=" << num_edges
                << " undirected E=" << (num_edges / 2) << std::endl;
    }
    auto mg_pruned_graph_view   = mg_pruned_graph.view();
    auto mg_isolated_trees_view = mg_isolated_trees.view();
    std::cout << "mg_pruned_graph V=" << mg_pruned_graph_view.number_of_vertices()
              << " E=" << mg_pruned_graph_view.compute_number_of_edges(*handle_)
              << " mg_isolated_trees_view V=" << mg_isolated_trees_view.number_of_vertices()
              << " E=" << mg_isolated_trees_view.compute_number_of_edges(*handle_) << std::endl;

    // 3. randomly select starting vertices

    std::vector<vertex_t> starting_vertices{};
    {
      ASSERT_TRUE(vertex_partition_range_offsets.back() > 0)
        << "Invalid input graph, the input graph should have at least one vertex";
      rmm::device_uvector<vertex_t> d_starting_vertices(
        num_warmup_starting_vertices + num_timed_starting_vertices, handle_->get_stream());
      if (comm_rank == 0) {
        cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                             d_starting_vertices.data(),
                                             d_starting_vertices.size(),
                                             vertex_partition_range_offsets[0],
                                             vertex_partition_range_offsets.back(),
                                             rng_state);
        raft::print_device_vector(
          "d_starting_vertices", d_starting_vertices.data(), d_starting_vertices.size(), std::cout);
      }
      cugraph::device_bcast(comm,
                            d_starting_vertices.data(),
                            d_starting_vertices.data(),
                            d_starting_vertices.size(),
                            int{0},
                            handle_->get_stream());
      starting_vertices = cugraph::test::to_host(*handle_, d_starting_vertices);
    }

    // 4. run MG BFS

    rmm::device_uvector<vertex_t> d_mg_distances(
      mg_renumber_map.size(),
      handle_->get_stream());  // Graph500 doesn't require computing distances (so we can update
                               // this outside the timed region)
    rmm::device_uvector<vertex_t> d_mg_unrenumbered_predecessors(mg_renumber_map.size(),
                                                                 handle_->get_stream());

    double total_elapsed{0.0};
    double tteps_sum{0.0};
    double one_over_tteps_sum{0.0};  // to compute harmonic mean
    for (size_t i = 0; i < (num_warmup_starting_vertices + num_timed_starting_vertices); ++i) {
      double elapsed{0.0};

      thrust::fill(handle_->get_thrust_policy(),
                   d_mg_distances.begin(),
                   d_mg_distances.end(),
                   invalid_distance);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.start("MG BFS (Kernel 2)");
      }

      auto starting_vertex = starting_vertices[i];
      auto starting_vertex_vertex_partition_id =
        static_cast<int>(std::distance(vertex_partition_range_offsets.begin() + 1,
                                       std::upper_bound(vertex_partition_range_offsets.begin() + 1,
                                                        vertex_partition_range_offsets.end(),
                                                        starting_vertex)));

      vertex_t unrenumbered_starting_vertex{};
      vertex_t starting_vertex_parent{starting_vertex};
      vertex_t starting_vertex_component{};
      if (starting_vertex_vertex_partition_id == vertex_partition_id) {
        unrenumbered_starting_vertex = mg_renumber_map.element(
          starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
        starting_vertex_parent = parents.element(
          starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
        starting_vertex_component = components.element(
          starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
      }
      thrust::tie(unrenumbered_starting_vertex, starting_vertex_parent, starting_vertex_component) =
        cugraph::host_scalar_bcast(
          comm,
          thrust::make_tuple(
            unrenumbered_starting_vertex, starting_vertex_parent, starting_vertex_component),
          cugraph::partition_manager::compute_global_comm_rank_from_vertex_partition_id(
            major_comm_size, minor_comm_size, starting_vertex_vertex_partition_id),
          handle_->get_stream());
      bool reachable_from_2cores{starting_vertex_parent != invalid_vertex};
      bool in_2cores{starting_vertex == starting_vertex_parent};

      if (reachable_from_2cores) {
        thrust::transform(handle_->get_thrust_policy(),
                          unrenumbered_parents.begin(),
                          unrenumbered_parents.end(),
                          components.begin(),
                          d_mg_unrenumbered_predecessors.begin(),
                          [starting_vertex_component, invalid_vertex] __device__(auto p, auto c) {
                            return (c == starting_vertex_component)
                                     ? p /* for the vertices in 2-cores (or the vertices in the
                                            path from the starting vertex to the first reachable
                                            2-core vertex), this will be over-written */
                                     : invalid_vertex;
                          });
      } else {
        thrust::fill(handle_->get_thrust_policy(),
                     d_mg_unrenumbered_predecessors.begin(),
                     d_mg_unrenumbered_predecessors.end(),
                     invalid_vertex);
      }

      vertex_t subgraph_starting_vertex{starting_vertex};
      int subgraph_starting_vertex_vertex_partition_id{starting_vertex_vertex_partition_id};
      vertex_t subgraph_starting_vertex_distance{0};
      vertex_t unrenumbered_subgraph_starting_vertex_parent{};
      if (reachable_from_2cores && !in_2cores) {  // find the path from starting_vertex to a 2-core
        if (starting_vertex_vertex_partition_id == vertex_partition_id) {
          d_mg_unrenumbered_predecessors.set_element_async(
            starting_vertex - local_vertex_partition_range_first,
            unrenumbered_starting_vertex,  // Graph 500 requires the predecessor of a starting
                                           // vertex to be itself
            handle_->get_stream());
          d_mg_distances.set_element_to_zero_async(
            starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
          handle_->sync_stream();
        }

        auto unrenumbered_v = unrenumbered_starting_vertex;
        auto n = starting_vertex_parent;  // reverse the parent child relationship till we reach
                                          // a 2-core
        while (true) {
          assert(v != n);  // in this case, v is already a 2-core vertex
          ++subgraph_starting_vertex_distance;
          auto n_vertex_partition_id = static_cast<int>(
            std::distance(vertex_partition_range_offsets.begin() + 1,
                          std::upper_bound(vertex_partition_range_offsets.begin() + 1,
                                           vertex_partition_range_offsets.end(),
                                           n)));
          vertex_t unrenumbered_n{};
          vertex_t nn{};
          if (n_vertex_partition_id == vertex_partition_id) {
            unrenumbered_n = mg_renumber_map.element(n - local_vertex_partition_range_first,
                                                     handle_->get_stream());
            nn = parents.element(n - local_vertex_partition_range_first, handle_->get_stream());
          }
          thrust::tie(unrenumbered_n, nn) = cugraph::host_scalar_bcast(
            comm,
            thrust::make_tuple(unrenumbered_n, nn),
            cugraph::partition_manager::compute_global_comm_rank_from_vertex_partition_id(
              major_comm_size, minor_comm_size, n_vertex_partition_id),
            handle_->get_stream());

          if (n == nn) {  // reached a 2-core
            subgraph_starting_vertex                     = n;
            subgraph_starting_vertex_vertex_partition_id = n_vertex_partition_id;
            unrenumbered_subgraph_starting_vertex_parent = unrenumbered_v;
            break;
          }

          if (n_vertex_partition_id == vertex_partition_id) {
            d_mg_unrenumbered_predecessors.set_element_async(
              n - local_vertex_partition_range_first, unrenumbered_v, handle_->get_stream());
            d_mg_distances.set_element_async(n - local_vertex_partition_range_first,
                                             subgraph_starting_vertex_distance,
                                             handle_->get_stream());
            handle_->sync_stream();
          }

          unrenumbered_v = unrenumbered_n;
          n              = nn;
        }
      }

      std::optional<rmm::device_scalar<vertex_t>> d_bfs_starting_vertex{std::nullopt};
      if (subgraph_starting_vertex_vertex_partition_id == vertex_partition_id) {
        auto bfs_starting_vertex =
          reachable_from_2cores ? mg_pruned_graph_view.local_vertex_partition_range_first() +
                                    mg_graph_to_pruned_graph_map.element(
                                      subgraph_starting_vertex - local_vertex_partition_range_first,
                                      handle_->get_stream())
                                : mg_isolated_trees_view.local_vertex_partition_range_first() +
                                    mg_graph_to_isolated_trees_map.element(
                                      subgraph_starting_vertex - local_vertex_partition_range_first,
                                      handle_->get_stream());
        d_bfs_starting_vertex =
          rmm::device_scalar<vertex_t>(bfs_starting_vertex, handle_->get_stream());
      }

      rmm::device_uvector<vertex_t> d_mg_bfs_predecessors(
        reachable_from_2cores ? mg_pruned_graph_view.local_vertex_partition_range_size()
                              : mg_isolated_trees_view.local_vertex_partition_range_size(),
        handle_->get_stream());
      rmm::device_uvector<vertex_t> d_mg_bfs_distances(d_mg_bfs_predecessors.size(),
                                                       handle_->get_stream());

      cugraph::bfs(*handle_,
                   reachable_from_2cores ? mg_pruned_graph_view : mg_isolated_trees_view,
                   d_mg_bfs_distances.data(),
                   d_mg_bfs_predecessors.data(),
                   d_bfs_starting_vertex ? d_bfs_starting_vertex->data()
                                         : static_cast<vertex_t const*>(nullptr),
                   d_bfs_starting_vertex ? size_t{1} : size_t{0},
                   true /* direction_optimizing */,
                   std::numeric_limits<vertex_t>::max() /* depth limit */);

      if (reachable_from_2cores && mg_pruned_graph_pred_unrenumber_cache) {
        mg_pruned_graph_pred_unrenumber_cache->unrenumber(
          *handle_,
          raft::device_span<vertex_t>(d_mg_bfs_predecessors.data(), d_mg_bfs_predecessors.size()));
      } else {
        cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
          *handle_,
          d_mg_bfs_predecessors.data(),
          d_mg_bfs_predecessors.size(),
          reachable_from_2cores ? mg_pruned_graph_renumber_map.data()
                                : mg_isolated_trees_renumber_map.data(),
          reachable_from_2cores ? mg_pruned_graph_view.vertex_partition_range_lasts()
                                : mg_isolated_trees_view.vertex_partition_range_lasts());
      }

      thrust::scatter(handle_->get_thrust_policy(),
                      d_mg_bfs_predecessors.begin(),
                      d_mg_bfs_predecessors.end(),
                      reachable_from_2cores ? mg_pruned_graph_to_graph_map.begin()
                                            : mg_isolated_trees_to_graph_map.begin(),
                      d_mg_unrenumbered_predecessors.begin());

      {  // update the starting vertex's parent
        if (subgraph_starting_vertex_vertex_partition_id ==
            vertex_partition_id) {  // cugraph::bfs sets the predecessor of the starting vertex to
                                    // invalid_vertex
          if (subgraph_starting_vertex_distance > vertex_t{0}) {
            d_mg_unrenumbered_predecessors.set_element_async(
              subgraph_starting_vertex - local_vertex_partition_range_first,
              unrenumbered_subgraph_starting_vertex_parent,
              handle_->get_stream());
          } else {
            assert(starting_vertex == subgraph_starting_vertex);
            d_mg_unrenumbered_predecessors.set_element_async(
              starting_vertex - local_vertex_partition_range_first,
              unrenumbered_starting_vertex,
              handle_->get_stream());
          }
          handle_->sync_stream();
        }
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        elapsed = hr_timer.stop();
        if (i >= num_warmup_starting_vertices) { total_elapsed += elapsed; }
        hr_timer.display_and_clear(std::cout);
      }

      // update d_mg_distances (for validation, Graph 500 doesn't require computing distances)

      if (reachable_from_2cores) {
        if (subgraph_starting_vertex_distance > vertex_t{0}) {
          thrust::transform(handle_->get_thrust_policy(),
                            d_mg_bfs_distances.begin(),
                            d_mg_bfs_distances.end(),
                            d_mg_bfs_distances.begin(),
                            [delta = subgraph_starting_vertex_distance] __device__(auto d) {
                              if (d != invalid_distance) {
                                return d + delta;
                              } else {
                                return invalid_distance;
                              }
                            });
        }
        thrust::scatter(handle_->get_thrust_policy(),
                        d_mg_bfs_distances.begin(),
                        d_mg_bfs_distances.end(),
                        mg_pruned_graph_to_graph_map.begin(),
                        d_mg_distances.begin());
        rmm::device_uvector<vertex_t> remaining_vertices(
          mg_renumber_map.size(),
          handle_->get_stream());  // unvisited vertices in the forest (but in the same connected
                                   // component with the starting vertex)
        remaining_vertices.resize(
          thrust::distance(
            remaining_vertices.begin(),
            thrust::copy_if(handle_->get_thrust_policy(),
                            thrust::make_counting_iterator(local_vertex_partition_range_first),
                            thrust::make_counting_iterator(local_vertex_partition_range_last),
                            thrust::make_zip_iterator(components.begin(), d_mg_distances.begin()),
                            remaining_vertices.begin(),
                            [starting_vertex_component, invalid_distance] __device__(auto pair) {
                              return thrust::get<0>(pair) == starting_vertex_component &&
                                     thrust::get<1>(pair) == invalid_distance;
                            })),
          handle_->get_stream());
        while (true) {
          auto tot_remaining_vertex_count = cugraph::host_scalar_allreduce(
            comm, remaining_vertices.size(), raft::comms::op_t::SUM, handle_->get_stream());
          if (tot_remaining_vertex_count == 0) { break; }
          rmm::device_uvector<vertex_t> preds(remaining_vertices.size(), handle_->get_stream());
          auto gather_offset_first = thrust::make_transform_iterator(
            remaining_vertices.begin(),
            cugraph::detail::shift_left_t<vertex_t>{local_vertex_partition_range_first});
          thrust::gather(handle_->get_thrust_policy(),
                         gather_offset_first,
                         gather_offset_first + remaining_vertices.size(),
                         parents.begin(),
                         preds.begin());
          auto dists = cugraph::collect_values_for_int_vertices(
            *handle_,
            preds.begin(),
            preds.end(),
            d_mg_distances.begin(),
            raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                            vertex_partition_range_offsets.size() - 1),
            local_vertex_partition_range_first);
          auto pair_first = thrust::make_zip_iterator(remaining_vertices.begin(), dists.begin());
          auto remaining_last = thrust::partition(
            handle_->get_thrust_policy(),
            pair_first,
            pair_first + remaining_vertices.size(),
            [] __device__(auto pair) { return thrust::get<1>(pair) == invalid_distance; });
          auto new_size = thrust::distance(pair_first, remaining_last);
          auto dist_first =
            thrust::make_transform_iterator(
              dists.begin(),
              cuda::proclaim_return_type<vertex_t>([invalid_distance] __device__(auto d) {
                return (d != invalid_distance) ? static_cast<vertex_t>(d + 1) : invalid_distance;
              })) +
            new_size;
          auto scatter_offset_first =
            thrust::make_transform_iterator(
              remaining_vertices.begin(),
              cugraph::detail::shift_left_t<vertex_t>{local_vertex_partition_range_first}) +
            new_size;
          thrust::scatter(handle_->get_thrust_policy(),
                          dist_first,
                          dist_first + (remaining_vertices.size() - new_size),
                          scatter_offset_first,
                          d_mg_distances.begin());
          remaining_vertices.resize(new_size, handle_->get_stream());
        }
      } else {
        assert(subgraph_starting_vertex_distance == vertex_t{0});
        thrust::scatter(handle_->get_thrust_policy(),
                        d_mg_bfs_distances.begin(),
                        d_mg_bfs_distances.end(),
                        mg_isolated_trees_to_graph_map.begin(),
                        d_mg_distances.begin());
      }

      /* compute the number of visisted edges */

      {
        edge_t tot_edge_count{};
        if (reachable_from_2cores) {
          rmm::device_uvector<bool> visited(
            mg_pruned_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
          thrust::transform(handle_->get_thrust_policy(),
                            d_mg_bfs_distances.begin(),
                            d_mg_bfs_distances.end(),
                            visited.begin(),
                            cuda::proclaim_return_type<bool>([invalid_distance] __device__(auto d) {
                              return d != invalid_distance;
                            }));
          cugraph::edge_src_property_t<vertex_t, bool> edge_src_visited(*handle_,
                                                                        mg_pruned_graph_view);
          cugraph::update_edge_src_property(
            *handle_, mg_pruned_graph_view, visited.begin(), edge_src_visited.mutable_view());
          tot_edge_count =
            cugraph::count_if_e(
              *handle_,
              mg_pruned_graph_view,
              edge_src_visited.view(),
              cugraph::edge_dst_dummy_property_t{}.view(),
              cugraph::edge_dummy_property_t{}.view(),
              [] __device__(auto, auto, auto src_visited, auto, auto) { return src_visited; }) /
            edge_t{2};
          auto forest_edge_count = thrust::count_if(
            handle_->get_thrust_policy(),
            thrust::make_zip_iterator(d_mg_distances.begin(), mg_graph_to_pruned_graph_map.begin()),
            thrust::make_zip_iterator(d_mg_distances.end(), mg_graph_to_pruned_graph_map.end()),
            [invalid_distance, invalid_vertex] __device__(auto pair) {
              return (thrust::get<0>(pair) != invalid_distance /* reachable */) &&
                     (thrust::get<1>(pair) == invalid_vertex /* not in the pruned graph */);
            });  // # vertices reachable from 2-cores but not in 2-cores
          forest_edge_count = cugraph::host_scalar_allreduce(
            comm, forest_edge_count, raft::comms::op_t::SUM, handle_->get_stream());
          tot_edge_count += forest_edge_count;
        } else {
          auto num_visited = thrust::count_if(
            handle_->get_thrust_policy(),
            d_mg_distances.begin(),
            d_mg_distances.end(),
            [invalid_distance] __device__(auto d) { return d != invalid_distance; });
          auto tot_num_visited = cugraph::host_scalar_allreduce(
            comm, num_visited, raft::comms::op_t::SUM, handle_->get_stream());
          tot_edge_count = tot_num_visited - 1;  // # edges in a tree is # vertices - 1
        }
        auto tteps = (static_cast<double>(tot_edge_count) / 1e12) / elapsed;
        if (i >= num_warmup_starting_vertices) {
          tteps_sum += tteps;
          one_over_tteps_sum +=
            (tteps > 0.0) ? 1.0 / tteps : std::numeric_limits<double>::infinity();
        }
        std::cout << "# visited undirected edges=" << tot_edge_count
                  << " TTEPS=" << (static_cast<double>(tot_edge_count) / 1e12) / elapsed
                  << std::endl;
      }

      if (bfs_usecase.validate) {
        /* renumber for validation */

        rmm::device_uvector<vertex_t> d_mg_predecessors(d_mg_unrenumbered_predecessors.size(),
                                                        handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     d_mg_unrenumbered_predecessors.begin(),
                     d_mg_unrenumbered_predecessors.end(),
                     d_mg_predecessors.begin());
        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(*handle_,
                                                            d_mg_predecessors.data(),
                                                            d_mg_predecessors.size(),
                                                            mg_renumber_map.data(),
                                                            local_vertex_partition_range_first,
                                                            local_vertex_partition_range_last);

        /* check starting vertex's predecessor */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (starting vertex's predecessor)");
        }

        {
          size_t num_invalids{0};
          if (starting_vertex_vertex_partition_id == vertex_partition_id) {
            auto starting_vertex_predecessor = d_mg_predecessors.element(
              starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
            if (starting_vertex_predecessor != starting_vertex) { ++num_invalids; }
          }
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
          ASSERT_EQ(num_invalids, 0) << "predecessor of a starting vertex should be itself";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check for cycles (update predecessor to predecessor's predecessor till reaching the
         * starting vertex, if there exists a cycle, this won't finish) */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (cycle)");
        }

        {
          cugraph::kv_store_t<vertex_t, vertex_t, true /* use_binary_search */> kv_store(
            thrust::make_counting_iterator(local_vertex_partition_range_first),
            thrust::make_counting_iterator(local_vertex_partition_range_last),
            d_mg_predecessors.begin(),
            invalid_vertex,
            true /* key_sorted */,
            handle_->get_stream());
          auto kv_store_view = kv_store.view();
          rmm::device_uvector<vertex_t> d_vertex_partition_range_offsets(
            vertex_partition_range_offsets.size(), handle_->get_stream());
          raft::update_device(d_vertex_partition_range_offsets.data(),
                              vertex_partition_range_offsets.data(),
                              vertex_partition_range_offsets.size(),
                              handle_->get_stream());

          rmm::device_uvector<vertex_t> ancestors(d_mg_predecessors.size(), handle_->get_stream());
          ancestors.resize(
            thrust::distance(
              ancestors.begin(),
              thrust::copy_if(handle_->get_thrust_policy(),
                              d_mg_predecessors.begin(),
                              d_mg_predecessors.end(),
                              ancestors.begin(),
                              cuda::proclaim_return_type<bool>(
                                [starting_vertex, invalid_vertex] __device__(auto pred) {
                                  return (pred != starting_vertex) && (pred != invalid_vertex);
                                }))),
            handle_->get_stream());

          size_t level{0};
          auto aggregate_size = cugraph::host_scalar_allreduce(
            comm, ancestors.size(), raft::comms::op_t::SUM, handle_->get_stream());
          while (aggregate_size > size_t{0}) {
            ASSERT_TRUE(level < vertex_partition_range_offsets.back() - 1)
              << "BFS predecessor tree has a cycle.";
            auto num_invalids = thrust::count(
              handle_->get_thrust_policy(), ancestors.begin(), ancestors.end(), invalid_vertex);
            num_invalids = cugraph::host_scalar_allreduce(
              comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
            ASSERT_EQ(num_invalids, 0) << "Invalid BFS predecessor tree, failed to backtrace from "
                                          "a reachable vertex to the starting vertex";
            ancestors = cugraph::collect_values_for_keys(
              comm,
              kv_store_view,
              ancestors.begin(),
              ancestors.end(),
              cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                raft::device_span<vertex_t const>(d_vertex_partition_range_offsets.data() + 1,
                                                  d_vertex_partition_range_offsets.size() - 1),
                major_comm_size,
                minor_comm_size},
              handle_->get_stream());
            ancestors.resize(
              thrust::distance(
                ancestors.begin(),
                thrust::remove_if(handle_->get_thrust_policy(),
                                  ancestors.begin(),
                                  ancestors.end(),
                                  cugraph::detail::is_equal_t<vertex_t>{starting_vertex})),
              handle_->get_stream());
            aggregate_size = cugraph::host_scalar_allreduce(
              comm, ancestors.size(), raft::comms::op_t::SUM, handle_->get_stream());
            ++level;
          }
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check that distance(v) = distance(predecssor(v)) + 1 */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (predecessor tree distances)");
        }

        {
          rmm::device_uvector<vertex_t> tree_srcs(mg_renumber_map.size(), handle_->get_stream());
          rmm::device_uvector<vertex_t> tree_dsts(tree_srcs.size(), handle_->get_stream());
          auto input_pair_first = thrust::make_zip_iterator(
            d_mg_predecessors.begin(),
            thrust::make_counting_iterator(local_vertex_partition_range_first));
          auto output_pair_first = thrust::make_zip_iterator(tree_srcs.begin(), tree_dsts.begin());
          tree_srcs.resize(
            cuda::std::distance(
              output_pair_first,
              thrust::copy_if(handle_->get_thrust_policy(),
                              input_pair_first,
                              input_pair_first + mg_renumber_map.size(),
                              output_pair_first,
                              cuda::proclaim_return_type<bool>(
                                [starting_vertex, invalid_vertex] __device__(auto pair) {
                                  auto pred = thrust::get<0>(pair);
                                  auto v    = thrust::get<1>(pair);
                                  return (pred != invalid_vertex) && (v != starting_vertex);
                                }))),
            handle_->get_stream());
          tree_dsts.resize(tree_srcs.size(), handle_->get_stream());

          auto tree_src_dists = cugraph::collect_values_for_int_vertices(
            *handle_,
            tree_srcs.begin(),
            tree_srcs.end(),
            d_mg_distances.begin(),
            raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                            vertex_partition_range_offsets.size() - 1),
            local_vertex_partition_range_first);

          rmm::device_uvector<vertex_t> tree_dst_dists(tree_dsts.size(), handle_->get_stream());
          thrust::transform(handle_->get_thrust_policy(),
                            tree_dsts.begin(),
                            tree_dsts.end(),
                            tree_dst_dists.begin(),
                            cuda::proclaim_return_type<vertex_t>(
                              [mg_distances = raft::device_span<vertex_t const>(
                                 d_mg_distances.data(), d_mg_distances.size()),
                               v_first = local_vertex_partition_range_first] __device__(auto v) {
                                return mg_distances[v - v_first];
                              }));

          ASSERT_EQ(tree_src_dists.size(), tree_dst_dists.size());
          auto dist_pair_first =
            thrust::make_zip_iterator(tree_src_dists.begin(), tree_dst_dists.begin());
          auto num_invalids =
            thrust::count_if(handle_->get_thrust_policy(),
                             dist_pair_first,
                             dist_pair_first + tree_src_dists.size(),
                             cuda::proclaim_return_type<bool>([] __device__(auto pair) {
                               auto src_dist = thrust::get<0>(pair);
                               auto dst_dist = thrust::get<1>(pair);
                               return (src_dist + 1) != dst_dist;
                             }));
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());

          ASSERT_EQ(num_invalids, 0) << " source and destination vertices in the BFS predecessor "
                                        "tree are not one hop away.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (graph distances)");
        }

        /* for every edge e = (u, v), abs(dist(u) - dist(v)) <= 1 or dist(u) == dist(v) ==
         * invalid_distance */

        {
          auto const& mg_subgraph_view =
            reachable_from_2cores ? mg_pruned_graph_view : mg_isolated_trees_view;
          cugraph::edge_src_property_t<vertex_t, bool> edge_src_flags(*handle_, mg_subgraph_view);
          cugraph::edge_dst_property_t<vertex_t, bool> edge_dst_flags(*handle_, mg_subgraph_view);

          // first, validate the edges in the subgraph

          auto max_distance = thrust::transform_reduce(
            handle_->get_thrust_policy(),
            d_mg_distances.begin(),
            d_mg_distances.end(),
            cuda::proclaim_return_type<vertex_t>([invalid_distance] __device__(auto d) {
              return d == invalid_distance ? vertex_t{0} : d;
            }),
            vertex_t{0},
            thrust::maximum<vertex_t>{});
          max_distance = cugraph::host_scalar_allreduce(
            comm, max_distance, raft::comms::op_t::MAX, handle_->get_stream());

          auto pair_first = thrust::make_zip_iterator(reachable_from_2cores
                                                        ? mg_graph_to_pruned_graph_map.begin()
                                                        : mg_graph_to_isolated_trees_map.begin(),
                                                      d_mg_distances.begin());
          for (vertex_t level = 0; level <= max_distance;
               ++level) {  // validate in multple round to cut peak memory usage (to store
                           // source|destination vertex properties using 1 bit per vertex)
            rmm::device_uvector<vertex_t> subgraph_level_v_offsets(
              mg_subgraph_view.local_vertex_partition_range_size(),
              handle_->get_stream());  // vertices with d_mg_distances[] == level
            rmm::device_uvector<vertex_t> subgraph_adjacent_level_v_offsets(
              mg_subgraph_view.local_vertex_partition_range_size(),
              handle_->get_stream());  // abs(vertices with d_mg_distances[] - level) <= 1
            subgraph_level_v_offsets.resize(
              thrust::distance(
                subgraph_level_v_offsets.begin(),
                thrust::copy_if(handle_->get_thrust_policy(),
                                reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                                      : mg_graph_to_isolated_trees_map.begin(),
                                reachable_from_2cores ? mg_graph_to_pruned_graph_map.end()
                                                      : mg_graph_to_isolated_trees_map.end(),
                                pair_first,
                                subgraph_level_v_offsets.begin(),
                                [level, invalid_distance] __device__(auto pair) {
                                  auto d = thrust::get<1>(pair);
                                  return (thrust::get<0>(pair) !=
                                          invalid_vertex /* in the subgraph */) &&
                                         (d == level);
                                })),
              handle_->get_stream());
            subgraph_level_v_offsets.shrink_to_fit(handle_->get_stream());
            subgraph_adjacent_level_v_offsets.resize(
              thrust::distance(
                subgraph_adjacent_level_v_offsets.begin(),
                thrust::copy_if(
                  handle_->get_thrust_policy(),
                  reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                        : mg_graph_to_isolated_trees_map.begin(),
                  reachable_from_2cores ? mg_graph_to_pruned_graph_map.end()
                                        : mg_graph_to_isolated_trees_map.end(),
                  pair_first,
                  subgraph_adjacent_level_v_offsets.begin(),
                  cuda::proclaim_return_type<bool>([level, invalid_distance] __device__(auto pair) {
                    auto d = thrust::get<1>(pair);
                    return (thrust::get<0>(pair) != invalid_vertex /* in the subgraph */) &&
                           (((d >= level) ? (d - level) : (level - d)) <= 1);
                  }))),
              handle_->get_stream());
            subgraph_adjacent_level_v_offsets.shrink_to_fit(handle_->get_stream());

            auto subgraph_level_vs = std::move(subgraph_level_v_offsets);
            thrust::transform(
              handle_->get_thrust_policy(),
              subgraph_level_vs.begin(),
              subgraph_level_vs.end(),
              subgraph_level_vs.begin(),
              cuda::proclaim_return_type<vertex_t>(
                [v_first = mg_subgraph_view.local_vertex_partition_range_first()] __device__(
                  auto v_offset) { return v_first + v_offset; }));
            thrust::sort(
              handle_->get_thrust_policy(), subgraph_level_vs.begin(), subgraph_level_vs.end());
            cugraph::fill_edge_src_property(
              *handle_, mg_subgraph_view, edge_src_flags.mutable_view(), false);
            cugraph::fill_edge_src_property(*handle_,
                                            mg_subgraph_view,
                                            subgraph_level_vs.begin(),
                                            subgraph_level_vs.end(),
                                            edge_src_flags.mutable_view(),
                                            true);  // true if the distance is level
            auto subgraph_adjacent_level_vs = std::move(subgraph_adjacent_level_v_offsets);
            thrust::transform(
              handle_->get_thrust_policy(),
              subgraph_adjacent_level_vs.begin(),
              subgraph_adjacent_level_vs.end(),
              subgraph_adjacent_level_vs.begin(),
              cuda::proclaim_return_type<vertex_t>(
                [v_first = mg_subgraph_view.local_vertex_partition_range_first()] __device__(
                  auto v_offset) { return v_first + v_offset; }));
            thrust::sort(handle_->get_thrust_policy(),
                         subgraph_adjacent_level_vs.begin(),
                         subgraph_adjacent_level_vs.end());
            cugraph::fill_edge_dst_property(
              *handle_, mg_subgraph_view, edge_dst_flags.mutable_view(), false);
            cugraph::fill_edge_dst_property(*handle_,
                                            mg_subgraph_view,
                                            subgraph_adjacent_level_vs.begin(),
                                            subgraph_adjacent_level_vs.end(),
                                            edge_dst_flags.mutable_view(),
                                            true);  // true if the abs(distance - level) <= 1
            auto num_invalids = cugraph::count_if_e(
              *handle_,
              mg_subgraph_view,
              edge_src_flags.view(),
              edge_dst_flags.view(),
              cugraph::edge_dummy_property_t{}.view(),
              cuda::proclaim_return_type<bool>(
                [level, invalid_distance] __device__(
                  auto src, auto dst, bool level_src, bool adjacent_level_dst, auto) {
                  return level_src && !adjacent_level_dst;
                }));
            ASSERT_EQ(num_invalids, 0)
              << "only one of the two connected vertices is reachable from the starting vertex or "
                 "the distances from the starting vertex differ by more than one.";
          }

          {
            rmm::device_uvector<vertex_t> unreachable_v_offsets(
              mg_subgraph_view.local_vertex_partition_range_size(), handle_->get_stream());
            unreachable_v_offsets.resize(
              thrust::distance(
                unreachable_v_offsets.begin(),
                thrust::copy_if(
                  handle_->get_thrust_policy(),
                  reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                        : mg_graph_to_isolated_trees_map.begin(),
                  reachable_from_2cores ? mg_graph_to_pruned_graph_map.end()
                                        : mg_graph_to_isolated_trees_map.end(),
                  pair_first,
                  unreachable_v_offsets.begin(),
                  cuda::proclaim_return_type<bool>(
                    [invalid_vertex, invalid_distance] __device__(auto pair) {
                      return (thrust::get<0>(pair) != invalid_vertex /* in the subgraph */) &&
                             (thrust::get<1>(pair) == invalid_distance /* unreachable */);
                    }))),
              handle_->get_stream());
            auto unreachable_vs = std::move(unreachable_v_offsets);
            thrust::transform(
              handle_->get_thrust_policy(),
              unreachable_vs.begin(),
              unreachable_vs.end(),
              unreachable_vs.begin(),
              cuda::proclaim_return_type<vertex_t>(
                [v_first = mg_subgraph_view.local_vertex_partition_range_first()] __device__(
                  auto v_offset) { return v_first + v_offset; }));
            cugraph::fill_edge_src_property(
              *handle_, mg_subgraph_view, edge_src_flags.mutable_view(), false);
            cugraph::fill_edge_src_property(*handle_,
                                            mg_subgraph_view,
                                            unreachable_vs.begin(),
                                            unreachable_vs.end(),
                                            edge_src_flags.mutable_view(),
                                            true);  // true if the distance is invalid_distance
            cugraph::fill_edge_dst_property(
              *handle_, mg_subgraph_view, edge_dst_flags.mutable_view(), false);
            cugraph::fill_edge_dst_property(*handle_,
                                            mg_subgraph_view,
                                            unreachable_vs.begin(),
                                            unreachable_vs.end(),
                                            edge_dst_flags.mutable_view(),
                                            true);  // true if the distance is invalid_distance
            auto num_invalids = cugraph::count_if_e(
              *handle_,
              mg_subgraph_view,
              edge_src_flags.view(),
              edge_dst_flags.view(),
              cugraph::edge_dummy_property_t{}.view(),
              cuda::proclaim_return_type<bool>(
                [] __device__(
                  auto src, auto dst, bool src_unreachable, bool dst_unreachable, auto) {
                  return src_unreachable != dst_unreachable;
                }));
            ASSERT_EQ(num_invalids, 0)
              << "only one of the two connected vertices is reachable from the starting vertex.";
          }

          // second, validate the edges in the pruned forest (if reachble_from_2cores is true)

          if (reachable_from_2cores) {
            rmm::device_uvector<vertex_t> forest_edge_parents(mg_renumber_map.size(),
                                                              handle_->get_stream());
            rmm::device_uvector<vertex_t> forest_edge_vertices(forest_edge_parents.size(),
                                                               handle_->get_stream());
            auto input_first = thrust::make_zip_iterator(
              parents.begin(), thrust::make_counting_iterator(local_vertex_partition_range_first));
            auto output_first =
              thrust::make_zip_iterator(forest_edge_parents.begin(), forest_edge_vertices.begin());
            forest_edge_parents.resize(
              thrust::distance(
                output_first,
                thrust::copy_if(handle_->get_thrust_policy(),
                                input_first,
                                input_first + forest_edge_parents.size(),
                                output_first,
                                cuda::proclaim_return_type<bool>([] __device__(auto pair) {
                                  auto p = thrust::get<0>(pair);
                                  auto v = thrust::get<1>(pair);
                                  return (p != invalid_vertex /* reachable from 2-cores */) &&
                                         (p != v /* not in a 2-core */);
                                }))),
              handle_->get_stream());
            forest_edge_vertices.resize(forest_edge_parents.size(), handle_->get_stream());
            auto forest_edge_src_dists = cugraph::collect_values_for_int_vertices(
              *handle_,
              forest_edge_parents.begin(),
              forest_edge_parents.end(),
              d_mg_distances.begin(),
              raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                              vertex_partition_range_offsets.size() - 1),
              local_vertex_partition_range_first);
            auto forest_edge_dst_dists = cugraph::collect_values_for_int_vertices(
              *handle_,
              forest_edge_vertices.begin(),
              forest_edge_vertices.end(),
              d_mg_distances.begin(),
              raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                              vertex_partition_range_offsets.size() - 1),
              local_vertex_partition_range_first);
            auto dist_pair_first = thrust::make_zip_iterator(forest_edge_src_dists.begin(),
                                                             forest_edge_dst_dists.begin());
            auto num_invalids =
              thrust::count_if(handle_->get_thrust_policy(),
                               dist_pair_first,
                               dist_pair_first + forest_edge_src_dists.size(),
                               cuda::proclaim_return_type<bool>([] __device__(auto pair) {
                                 auto src_dist = thrust::get<0>(pair);
                                 auto dst_dist = thrust::get<1>(pair);
                                 if (src_dist == invalid_distance) {
                                   return dst_dist != invalid_distance;
                                 } else {
                                   return (dst_dist == invalid_distance) ||
                                          (((src_dist >= dst_dist) ? (src_dist - dst_dist)
                                                                   : (dst_dist - src_dist)) > 1);
                                 }
                               }));
            num_invalids = cugraph::host_scalar_allreduce(
              comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
            ASSERT_EQ(num_invalids, 0)
              << "the distances from the starting vertex differ by more than one.";
          }
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* all the reachable vertices are in the same connected component, all the unreachable
         * vertices in different connected components */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (connected components)");
        }

        {
          auto pair_first =
            thrust::make_zip_iterator(components.begin(), d_mg_predecessors.begin());
          auto num_invalids =
            thrust::count_if(handle_->get_thrust_policy(),
                             pair_first,
                             pair_first + components.size(),
                             cuda::proclaim_return_type<bool>(
                               [starting_vertex_component, invalid_vertex] __device__(auto pair) {
                                 auto c    = thrust::get<0>(pair);
                                 auto pred = thrust::get<1>(pair);
                                 if (c == starting_vertex_component) {
                                   return pred == invalid_vertex;
                                 } else {
                                   return pred != invalid_vertex;
                                 }
                               }));
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
          ASSERT_EQ(num_invalids, 0) << "the BFS tree does not span the entire connected "
                                        "component of the starting vertex.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check that predecessor->v edges exist in the input graph */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (predecessor->v edge existence)");
        }

        {
          rmm::device_uvector<vertex_t> query_preds(d_mg_predecessors.size(),
                                                    handle_->get_stream());
          rmm::device_uvector<vertex_t> query_vertices(query_preds.size(), handle_->get_stream());
          auto input_edge_first = thrust::make_zip_iterator(
            d_mg_predecessors.begin(),
            thrust::make_counting_iterator(local_vertex_partition_range_first));
          auto output_edge_first =
            thrust::make_zip_iterator(query_preds.begin(), query_vertices.begin());
          query_preds.resize(
            thrust::distance(
              output_edge_first,
              thrust::copy_if(handle_->get_thrust_policy(),
                              input_edge_first,
                              input_edge_first + d_mg_predecessors.size(),
                              output_edge_first,
                              cuda::proclaim_return_type<bool>(
                                [invalid_vertex, starting_vertex] __device__(auto pair) {
                                  auto pred = thrust::get<0>(pair);
                                  auto v    = thrust::get<1>(pair);
                                  return (pred != invalid_vertex /* reachable */) &&
                                         (v != starting_vertex);
                                }))),
            handle_->get_stream());
          query_vertices.resize(query_preds.size(), handle_->get_stream());
          if (reachable_from_2cores) {  // exclude the edges in the forest (parents[v] -> v)
            auto query_edge_first =
              thrust::make_zip_iterator(query_preds.begin(), query_vertices.begin());
            query_preds.resize(
              thrust::distance(
                query_edge_first,
                thrust::remove_if(
                  handle_->get_thrust_policy(),
                  query_edge_first,
                  query_edge_first + query_preds.size(),
                  cuda::proclaim_return_type<bool>(
                    [parents = raft::device_span<vertex_t const>(parents.data(), parents.size()),
                     v_first = local_vertex_partition_range_first] __device__(auto pair) {
                      auto pred   = thrust::get<0>(pair);
                      auto v      = thrust::get<1>(pair);
                      auto parent = parents[v - v_first];
                      return parent == pred;  // the query edge exists in the forest
                    }))),
              handle_->get_stream());
            query_vertices.resize(query_preds.size(), handle_->get_stream());
            if (!in_2cores) {  // found BFS predecessor tree may contain edges from v ->
                               // parents[v] (instead of parents[v] -> v)
              rmm::device_uvector<vertex_t> forest_edge_vertices(parents.size(),
                                                                 handle_->get_stream());
              rmm::device_uvector<vertex_t> forest_edge_parents(forest_edge_vertices.size(),
                                                                handle_->get_stream());
              auto input_first = thrust::make_zip_iterator(
                thrust::make_counting_iterator(local_vertex_partition_range_first),
                parents.begin());
              auto output_first = thrust::make_zip_iterator(forest_edge_vertices.begin(),
                                                            forest_edge_parents.begin());
              forest_edge_vertices.resize(
                thrust::distance(
                  output_first,
                  thrust::copy_if(handle_->get_thrust_policy(),
                                  input_first,
                                  input_first + mg_renumber_map.size(),
                                  output_first,
                                  cuda::proclaim_return_type<bool>([] __device__(auto pair) {
                                    auto v      = thrust::get<0>(pair);
                                    auto parent = thrust::get<1>(pair);
                                    return (parent != invalid_vertex /* reachable */) &&
                                           (parent != v /* v is not in 2-cores */);
                                  }))),
                handle_->get_stream());
              forest_edge_parents.resize(forest_edge_vertices.size(), handle_->get_stream());
              std::tie(forest_edge_parents, forest_edge_vertices) =
                cugraph::detail::shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning<
                  vertex_t,
                  vertex_t>(
                  *handle_,
                  std::move(forest_edge_parents) /* vertex in (vertex, value) pair */,
                  std::move(forest_edge_vertices) /* value in (vertex, value) pair */,
                  raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                                  vertex_partition_range_offsets.size() - 1));
              auto forest_edge_first = thrust::make_zip_iterator(forest_edge_vertices.begin(),
                                                                 forest_edge_parents.begin());
              thrust::sort(handle_->get_thrust_policy(),
                           forest_edge_first,
                           forest_edge_first + forest_edge_vertices.size());
              query_edge_first =
                thrust::make_zip_iterator(query_preds.begin(), query_vertices.begin());
              query_preds.resize(
                thrust::distance(
                  query_edge_first,
                  thrust::remove_if(
                    handle_->get_thrust_policy(),
                    query_edge_first,
                    query_edge_first + query_preds.size(),
                    cuda::proclaim_return_type<bool>(
                      [forest_edge_first,
                       forest_edge_last =
                         forest_edge_first + forest_edge_vertices.size()] __device__(auto pair) {
                        auto pred = thrust::get<0>(pair);
                        auto v    = thrust::get<1>(pair);
                        auto key  = thrust::make_tuple(pred, v);
                        auto it   = thrust::lower_bound(
                          thrust::seq, forest_edge_first, forest_edge_last, key);
                        return (it != forest_edge_last) && (*it == key);
                      }))),
                handle_->get_stream());
              query_vertices.resize(query_preds.size(), handle_->get_stream());
            }
          }

          auto mg_graph_to_subgraph_map = raft::device_span<vertex_t const>(
            reachable_from_2cores ? mg_graph_to_pruned_graph_map.data()
                                  : mg_graph_to_isolated_trees_map.data(),
            reachable_from_2cores ? mg_graph_to_pruned_graph_map.size()
                                  : mg_graph_to_isolated_trees_map.size());
          auto mg_subgraph_view =
            reachable_from_2cores ? mg_pruned_graph_view : mg_isolated_trees_view;

          thrust::transform(
            handle_->get_thrust_policy(),
            query_vertices.begin(),
            query_vertices.end(),
            query_vertices.begin(),
            [mg_graph_to_subgraph_map,
             subgraph_v_first = mg_subgraph_view.local_vertex_partition_range_first(),
             v_first          = local_vertex_partition_range_first,
             invalid_vertex] __device__(auto v) {
              auto v_offset = mg_graph_to_subgraph_map[v - v_first];
              return (v_offset != invalid_vertex) ? (subgraph_v_first + v_offset) : invalid_vertex;
            });
          std::tie(query_preds, query_vertices) = cugraph::detail::
            shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning<vertex_t, vertex_t>(
              *handle_,
              std::move(query_preds) /* vertex in (vertex, value) pair */,
              std::move(query_vertices) /* value in (vertex, value) pair */,
              raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                              vertex_partition_range_offsets.size() - 1));
          thrust::transform(
            handle_->get_thrust_policy(),
            query_preds.begin(),
            query_preds.end(),
            query_preds.begin(),
            [mg_graph_to_subgraph_map,
             subgraph_v_first = mg_subgraph_view.local_vertex_partition_range_first(),
             v_first          = local_vertex_partition_range_first,
             invalid_vertex] __device__(auto v) {
              auto v_offset = mg_graph_to_subgraph_map[v - v_first];
              return (v_offset != invalid_vertex) ? (subgraph_v_first + v_offset) : invalid_vertex;
            });
          auto num_invalids = thrust::count_if(
            handle_->get_thrust_policy(),
            query_preds.begin(),
            query_preds.end(),
            [invalid_vertex] __device__(auto pred) { return pred == invalid_vertex; });
          num_invalids +=
            thrust::count_if(handle_->get_thrust_policy(),
                             query_vertices.begin(),
                             query_vertices.end(),
                             [invalid_vertex] __device__(auto v) { return v == invalid_vertex; });
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
          ASSERT_EQ(num_invalids, 0) << "predecessor->v missing in the input graph.";

          std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};

          std::tie(query_preds, query_vertices, std::ignore, std::ignore) =
            cugraph::shuffle_int_edges(*handle_,
                                       std::move(query_preds),
                                       std::move(query_vertices),
                                       std::move(edge_properties),
                                       store_transposed,
                                       mg_subgraph_view.vertex_partition_range_lasts());

          auto flags = mg_subgraph_view.has_edge(
            *handle_,
            raft::device_span<vertex_t const>(query_preds.data(), query_preds.size()),
            raft::device_span<vertex_t const>(query_vertices.data(), query_vertices.size()));
          num_invalids =
            thrust::count(handle_->get_thrust_policy(), flags.begin(), flags.end(), false);
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
          ASSERT_EQ(num_invalids, 0) << "predecessor->v missing in the input graph.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }
      }
    }

    std::cout << "average MG BFS (Kernel 2) time: " << (total_elapsed / num_timed_starting_vertices)
              << " TTEPS (arithmetic mean)=" << tteps_sum / num_timed_starting_vertices
              << " TTEPS (harmonic_mean)="
              << (one_over_tteps_sum > 0.0
                    ? static_cast<double>(num_timed_starting_vertices) / one_over_tteps_sum
                    : std::numeric_limits<double>::infinity())
              << std::endl;
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_GRAPH500_MGBFS<input_usecase_t>::handle_ = nullptr;

using Tests_GRAPH500_MGBFS_Rmat = Tests_GRAPH500_MGBFS<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_GRAPH500_MGBFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_GRAPH500_MGBFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(Graph500_BFS_Usecase{true, true, true},
                    cugraph::test::Rmat_Usecase(10,
                                                16,
                                                0.57,
                                                0.19,
                                                0.19,
                                                0 /* base RNG seed */,
                                                true /* undirected */,
                                                true /* scramble vertex ID */))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_GRAPH500_MGBFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(Graph500_BFS_Usecase{true, true, false},
                    cugraph::test::Rmat_Usecase(20,
                                                16,
                                                0.57,
                                                0.19,
                                                0.19,
                                                0 /* base RNG seed */,
                                                true /* undirected */,
                                                true /* scramble vertex IDs */))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
