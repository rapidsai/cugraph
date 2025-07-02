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
    std::cout << "DEBUG: num_seeds = " << betweenness_usecase.num_seeds << std::endl;
    std::cout << "DEBUG: normalized = " << betweenness_usecase.normalized << std::endl;
    std::cout << "DEBUG: include_endpoints = " << betweenness_usecase.include_endpoints << std::endl;
    std::cout << "DEBUG: test_weighted = " << betweenness_usecase.test_weighted << std::endl;
    std::cout << "DEBUG: edge_masking = " << betweenness_usecase.edge_masking << std::endl;
    std::cout << "DEBUG: check_correctness = " << betweenness_usecase.check_correctness << std::endl;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    std::cout << "DEBUG: Constructing graph..." << std::endl;
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

    std::cout << "DEBUG: Graph constructed successfully" << std::endl;
    std::cout << "DEBUG: Number of vertices: " << graph_view.number_of_vertices() << std::endl;
    std::cout << "DEBUG: Number of edges: " << graph_view.compute_number_of_edges(handle) << std::endl;
    std::cout << "DEBUG: Is symmetric: " << graph_view.is_symmetric() << std::endl;
    std::cout << "DEBUG: Has edge weights: " << (edge_weight_view.has_value() ? "yes" : "no") << std::endl;

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (betweenness_usecase.edge_masking) {
      std::cout << "DEBUG: Creating edge mask..." << std::endl;
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
      std::cout << "DEBUG: Edge mask attached" << std::endl;
    }

    std::cout << "DEBUG: Selecting random vertices for seeds..." << std::endl;
    raft::random::RngState rng_state(0);
    auto d_seeds = cugraph::select_random_vertices(
      handle,
      graph_view,
      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      rng_state,
      betweenness_usecase.num_seeds,
      false,
      true);

    std::cout << "DEBUG: Selected " << d_seeds.size() << " seeds" << std::endl;
    
    // Print first few seeds for debugging
    if (d_seeds.size() > 0) {
      auto h_seeds_debug = cugraph::test::to_host(handle, d_seeds);
      std::cout << "DEBUG: First 5 seeds: ";
      for (size_t i = 0; i < std::min(size_t(5), h_seeds_debug.size()); ++i) {
        std::cout << h_seeds_debug[i] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "DEBUG: Starting betweenness centrality computation..." << std::endl;
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Betweenness centrality");
    }

    try {
      auto d_centralities = cugraph::betweenness_centrality(
        handle,
        graph_view,
        edge_weight_view,
        std::make_optional<raft::device_span<vertex_t const>>(
          raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()}),
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
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      if (betweenness_usecase.check_correctness) {
        std::cout << "DEBUG: Starting correctness validation..." << std::endl;
        
        std::cout << "DEBUG: Converting graph to host CSR format..." << std::endl;
        auto [h_offsets, h_indices, h_wgt] = cugraph::test::graph_to_host_csr(
          handle,
          graph_view,
          edge_weight_view,
          std::optional<raft::device_span<vertex_t const>>(std::nullopt));

        std::cout << "DEBUG: Host CSR conversion completed" << std::endl;
        std::cout << "DEBUG: Host offsets size: " << h_offsets.size() << std::endl;
        std::cout << "DEBUG: Host indices size: " << h_indices.size() << std::endl;
        if (h_wgt) {
          std::cout << "DEBUG: Host weights size: " << h_wgt->size() << std::endl;
        }

        std::vector<vertex_t> h_seeds{};
        if (renumber) {
          std::cout << "DEBUG: Unrenumbering seeds..." << std::endl;
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
          std::cout << "DEBUG: Seeds unrenumbered and sorted" << std::endl;
        } else {
          std::cout << "DEBUG: Copying seeds to host..." << std::endl;
          h_seeds = cugraph::test::to_host(handle, d_seeds);
        }

        std::cout << "DEBUG: Computing reference centralities..." << std::endl;
        auto h_reference_centralities =
          betweenness_centrality_reference(h_offsets,
                                           h_indices,
                                           h_wgt,
                                           h_seeds,
                                           betweenness_usecase.include_endpoints,
                                           !graph_view.is_symmetric(),
                                           betweenness_usecase.normalized);

        std::cout << "DEBUG: Reference centralities computed" << std::endl;
        std::cout << "DEBUG: Reference centralities size: " << h_reference_centralities.size() << std::endl;

        // Print first few reference centrality values for debugging
        if (h_reference_centralities.size() > 0) {
          std::cout << "DEBUG: First 5 reference centrality values: ";
          for (size_t i = 0; i < std::min(size_t(5), h_reference_centralities.size()); ++i) {
            std::cout << std::fixed << std::setprecision(6) << h_reference_centralities[i] << " ";
          }
          std::cout << std::endl;
        }

        std::cout << "DEBUG: Converting reference centralities to device..." << std::endl;
        auto d_reference_centralities = cugraph::test::to_device(handle, h_reference_centralities);

        std::cout << "DEBUG: Starting validation..." << std::endl;
        cugraph::test::betweenness_centrality_validate(
          handle, d_centralities, d_reference_centralities);
        
        std::cout << "DEBUG: Validation completed successfully" << std::endl;
      }
    } catch (const std::exception& e) {
      std::cout << "ERROR: Exception caught during betweenness centrality computation: " << e.what() << std::endl;
      throw;
    } catch (...) {
      std::cout << "ERROR: Unknown exception caught during betweenness centrality computation" << std::endl;
      throw;
    }

    std::cout << "DEBUG: Test completed successfully" << std::endl;
  }
};

using Tests_BetweennessCentrality_File = Tests_BetweennessCentrality<cugraph::test::File_Usecase>;
using Tests_BetweennessCentrality_Rmat = Tests_BetweennessCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_BetweennessCentrality_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
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

CUGRAPH_TEST_PROGRAM_MAIN()
