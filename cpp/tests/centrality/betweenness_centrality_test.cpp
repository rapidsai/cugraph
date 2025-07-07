/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "betweenness_centrality_reference.hpp"
#include "betweenness_centrality_validate.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>

struct BetweennessCentrality_Usecase {
  size_t num_seeds{std::numeric_limits<size_t>::max()};
  bool normalized{false};
  bool include_endpoints{false};
  bool test_weighted{false};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_BetweennessCentrality
  : public ::testing::TestWithParam<std::tuple<BetweennessCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_BetweennessCentrality() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<BetweennessCentrality_Usecase, input_usecase_t> const& param)
  {
    constexpr bool renumber           = true;
    constexpr bool do_expensive_check = false;

    auto [betweenness_usecase, input_usecase] = param;

    std::cout << "\n=== DEBUG: Starting betweenness centrality test ===" << std::endl;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, betweenness_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    std::cout << "DEBUG: Number of vertices: " << graph_view.number_of_vertices() << std::endl;
    std::cout << "DEBUG: Number of edges: " << graph_view.compute_number_of_edges(handle) << std::endl;
    std::cout << "DEBUG: Is symmetric: " << graph_view.is_symmetric() << std::endl;
    std::cout << "DEBUG: Has edge weights: " << (edge_weight_view.has_value() ? "yes" : "no") << std::endl;

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (betweenness_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    std::optional<raft::device_span<vertex_t const>> seeds_span{std::nullopt};
    rmm::device_uvector<vertex_t> d_seeds(0, handle.get_stream());
    if (betweenness_usecase.num_seeds == std::numeric_limits<size_t>::max()) {
      // Use all vertices as sources (full betweenness centrality)
      std::cout << "DEBUG: Using all vertices as sources (full betweenness centrality)" << std::endl;
      // seeds_span remains std::nullopt, which signals to cugraph to use all vertices
    } else {
      raft::random::RngState rng_state(0);
      d_seeds = cugraph::select_random_vertices(
        handle,
        graph_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        rng_state,
        betweenness_usecase.num_seeds,
        false,
        true);
      seeds_span = raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()};
      std::cout << "DEBUG: Selected " << d_seeds.size() << " random vertices for seeds" << std::endl;
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Betweenness centrality");
    }

    auto d_centralities = cugraph::betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      seeds_span,
      betweenness_usecase.normalized,
      betweenness_usecase.include_endpoints,
      do_expensive_check);

    std::cout << "DEBUG: Betweenness centrality computation completed successfully" << std::endl;
    std::cout << "DEBUG: Centralities vector size: " << d_centralities.size() << std::endl;

    // Print first few centrality values for debugging
    if (d_centralities.size() > 0) {
      auto h_centralities_debug = cugraph::test::to_host(handle, d_centralities);
      std::cout << "DEBUG: First 5 centrality values: ";
      for (size_t i = 0; i < std::min(size_t(5), h_centralities_debug.size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << h_centralities_debug[i] << " ";
      }
      std::cout << std::endl;
      
      // Find max centrality vertex and print its coordinates
      auto max_it = std::max_element(h_centralities_debug.begin(), h_centralities_debug.end());
      if (max_it != h_centralities_debug.end()) {
        size_t max_vertex_idx = std::distance(h_centralities_debug.begin(), max_it);
        float max_centrality = *max_it;
        
        // Get the original vertex ID using the renumbering map
        vertex_t original_vertex_id = max_vertex_idx;  // Default to index if no renumbering
        if (renumber && d_renumber_map_labels.has_value()) {
          auto h_renumber_map = cugraph::test::to_host(handle, (*d_renumber_map_labels));
          if (max_vertex_idx < h_renumber_map.size()) {
            original_vertex_id = h_renumber_map[max_vertex_idx];
          }
        }
        
        // Read node coordinates from CSV file
        std::string node_file = "/home/nfs/howhuang/cugraph/manhattan_nodes.csv";
        std::ifstream node_stream(node_file);
        if (node_stream.is_open()) {
          std::string line;
          // Skip header
          std::getline(node_stream, line);
          
          // Find the line with the matching vertex ID
          bool found = false;
          while (std::getline(node_stream, line)) {
            std::stringstream ss(line);
            std::string id_str, x_str, y_str;
            std::getline(ss, id_str, ',');
            std::getline(ss, x_str, ',');
            std::getline(ss, y_str, ',');
            
            // Convert id_str to vertex_t for comparison
            vertex_t csv_vertex_id;
            std::stringstream id_ss(id_str);
            id_ss >> csv_vertex_id;
            
            if (csv_vertex_id == original_vertex_id) {
              std::cout << "DEBUG: Max centrality vertex: ID=" << id_str 
                        << ", x=" << x_str << ", y=" << y_str 
                        << ", centrality=" << std::fixed << std::setprecision(6) << max_centrality << std::endl;
              found = true;
              break;
            }
          }
          if (!found) {
            std::cout << "DEBUG: Could not find coordinates for vertex ID " << original_vertex_id 
                      << " (renumbered index " << max_vertex_idx << ")" << std::endl;
          }
        } else {
          // For datasets without coordinates (like California), just print the vertex ID
          std::cout << "DEBUG: Max centrality vertex: ID=" << original_vertex_id << " (renumbered index " << max_vertex_idx << ")" << ", centrality=" << std::fixed << std::setprecision(6) << max_centrality << std::endl;
        }
      }
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (betweenness_usecase.check_correctness) {
      auto [h_offsets, h_indices, h_wgt] = cugraph::test::graph_to_host_csr(
        handle,
        graph_view,
        edge_weight_view,
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));

      std::vector<vertex_t> h_seeds{};
      if (renumber) {
        rmm::device_uvector<vertex_t> d_unrenumbered_seeds(d_seeds.size(), handle.get_stream());
        raft::copy_async(
          d_unrenumbered_seeds.data(), d_seeds.data(), d_seeds.size(), handle.get_stream());
        cugraph::unrenumber_local_int_vertices(handle,
                                                d_unrenumbered_seeds.data(),
                                                d_unrenumbered_seeds.size(),
                                                (*d_renumber_map_labels).data(),
                                                vertex_t{0},
                                                graph_view.number_of_vertices());
        h_seeds = cugraph::test::to_host(handle, d_seeds);
        std::sort(h_seeds.begin(), h_seeds.end());
      } else {
        h_seeds = cugraph::test::to_host(handle, d_seeds);
      }
      auto h_reference_centralities =
        betweenness_centrality_reference(h_offsets,
                                          h_indices,
                                          h_wgt,
                                          h_seeds,
                                          betweenness_usecase.include_endpoints,
                                          !graph_view.is_symmetric(),
                                          betweenness_usecase.normalized);

      std::cout << "DEBUG: Reference centralities size: " << h_reference_centralities.size() << std::endl;

      // Print first few reference centrality values for debugging
      if (h_reference_centralities.size() > 0) {
        std::cout << "DEBUG: First 5 reference centrality values: ";
        for (size_t i = 0; i < std::min(size_t(5), h_reference_centralities.size()); ++i) {
          std::cout << std::fixed << std::setprecision(6) << h_reference_centralities[i] << " ";
        }
        std::cout << std::endl;
      }

      auto d_reference_centralities = cugraph::test::to_device(handle, h_reference_centralities);

      try {
        cugraph::test::betweenness_centrality_validate(
          handle, d_centralities, d_reference_centralities);
      } catch (const std::exception& e) {
        std::cout << "ERROR: Exception during betweenness centrality computation: " << e.what() << std::endl;
        throw;
      }
    }
  }
};

using Tests_BetweennessCentrality_File = Tests_BetweennessCentrality<cugraph::test::File_Usecase>;
using Tests_BetweennessCentrality_Rmat = Tests_BetweennessCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
// TEST_P(Tests_BetweennessCentrality_File, CheckInt32Int32FloatFloat)
// {
//   run_current_test<int32_t, int32_t, float>(
//     override_File_Usecase_with_cmd_line_arguments(GetParam()));
// }

TEST_P(Tests_BetweennessCentrality_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_BetweennessCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_BetweennessCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_BetweennessCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(BetweennessCentrality_Usecase{20, false, false, false, false},
                      BetweennessCentrality_Usecase{20, false, false, false, true},
                      BetweennessCentrality_Usecase{20, false, false, true, false},
                      BetweennessCentrality_Usecase{20, false, false, true, true},
                      BetweennessCentrality_Usecase{20, false, true, false, false},
                      BetweennessCentrality_Usecase{20, false, true, false, true},
                      BetweennessCentrality_Usecase{20, false, true, true, false},
                      BetweennessCentrality_Usecase{20, false, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_BetweennessCentrality_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(BetweennessCentrality_Usecase{50, false, false, false, false},
                      BetweennessCentrality_Usecase{50, false, false, false, true},
                      BetweennessCentrality_Usecase{50, false, false, true, false},
                      BetweennessCentrality_Usecase{50, false, false, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_BetweennessCentrality_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(BetweennessCentrality_Usecase{500, false, false, false, false, false},
                      BetweennessCentrality_Usecase{500, false, false, false, true, false},
                      BetweennessCentrality_Usecase{500, false, false, true, false, false},
                      BetweennessCentrality_Usecase{500, false, false, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  manhattan_test_pass,
  Tests_BetweennessCentrality_File,
  ::testing::Combine(
    // disable correctness checks for large dataset, use fewer seeds
    ::testing::Values(BetweennessCentrality_Usecase{50, false, false, false, false, false},
                      BetweennessCentrality_Usecase{100, false, false, false, false, false},
                      BetweennessCentrality_Usecase{200, false, false, false, false, false},
                      BetweennessCentrality_Usecase{std::numeric_limits<size_t>::max(), false, false, false, false, false}),
    ::testing::Values(cugraph::test::File_Usecase("/home/nfs/howhuang/cugraph/manhattan.csv"))));

INSTANTIATE_TEST_SUITE_P(
  newyork_test_pass,
  Tests_BetweennessCentrality_File,
  ::testing::Combine(
    // disable correctness checks for large dataset, use fewer seeds
    ::testing::Values(BetweennessCentrality_Usecase{50, false, false, false, false, false},
                      BetweennessCentrality_Usecase{100, false, false, false, false, false},
                      BetweennessCentrality_Usecase{200, false, false, false, false, false},
                      BetweennessCentrality_Usecase{std::numeric_limits<size_t>::max(), false, false, false, false, false}),
    ::testing::Values(cugraph::test::File_Usecase("/home/nfs/howhuang/cugraph/newyork.csv"))));

INSTANTIATE_TEST_SUITE_P(
  california_test_pass,
  Tests_BetweennessCentrality_File,
  ::testing::Combine(
    // disable correctness checks for large dataset, use fewer seeds
    ::testing::Values(BetweennessCentrality_Usecase{50, false, false, false, false, false},
                      BetweennessCentrality_Usecase{100, false, false, false, false, false},
                      BetweennessCentrality_Usecase{200, false, false, false, false, false},
                      BetweennessCentrality_Usecase{std::numeric_limits<size_t>::max(), false, false, false, false, false}),
    ::testing::Values(cugraph::test::File_Usecase("roadNet-CA.csv"))));
CUGRAPH_TEST_PROGRAM_MAIN()
