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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
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

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

struct HasEdgeAndComputeMultiplicity_Usecase {
  size_t num_vertex_pairs{};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_HasEdgeAndComputeMultiplicity
  : public ::testing::TestWithParam<
      std::tuple<HasEdgeAndComputeMultiplicity_Usecase, input_usecase_t>> {
 public:
  Tests_HasEdgeAndComputeMultiplicity() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, bool store_transposed>
  void run_current_test(
    HasEdgeAndComputeMultiplicity_Usecase const& has_edge_and_compute_multiplicity_usecase,
    input_usecase_t const& input_usecase)
  {
    using weight_t = float;

    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    raft::random::RngState rng_state(0);
    rmm::device_uvector<vertex_t> edge_srcs(
      has_edge_and_compute_multiplicity_usecase.num_vertex_pairs, handle.get_stream());
    rmm::device_uvector<vertex_t> edge_dsts(edge_srcs.size(), handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         edge_srcs.data(),
                                         edge_srcs.size(),
                                         vertex_t{0},
                                         graph_view.number_of_vertices(),
                                         rng_state);
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         edge_dsts.data(),
                                         edge_dsts.size(),
                                         vertex_t{0},
                                         graph_view.number_of_vertices(),
                                         rng_state);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Querying edge existence");
    }

    auto edge_exists =
      graph_view.has_edge(handle,
                          raft::device_span<vertex_t const>(edge_srcs.data(), edge_srcs.size()),
                          raft::device_span<vertex_t const>(edge_dsts.data(), edge_dsts.size()));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Computing multiplicity");
    }

    auto edge_multiplicities = graph_view.compute_multiplicity(
      handle,
      raft::device_span<vertex_t const>(edge_srcs.data(), edge_srcs.size()),
      raft::device_span<vertex_t const>(edge_dsts.data(), edge_dsts.size()));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (has_edge_and_compute_multiplicity_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, store_transposed, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle, input_usecase, false, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().offsets());
      std::vector<vertex_t> h_indices = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().indices());

      rmm::device_uvector<vertex_t> d_unrenumbered_edge_srcs(edge_srcs.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> d_unrenumbered_edge_dsts(edge_dsts.size(), handle.get_stream());
      raft::copy_async(
        d_unrenumbered_edge_srcs.data(), edge_srcs.data(), edge_srcs.size(), handle.get_stream());
      raft::copy_async(
        d_unrenumbered_edge_dsts.data(), edge_dsts.data(), edge_dsts.size(), handle.get_stream());
      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_unrenumbered_edge_srcs.data(),
                                               d_unrenumbered_edge_srcs.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices());
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_unrenumbered_edge_dsts.data(),
                                               d_unrenumbered_edge_dsts.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices());
      }
      auto h_unrenumbered_edge_srcs = cugraph::test::to_host(handle, d_unrenumbered_edge_srcs);
      auto h_unrenumbered_edge_dsts = cugraph::test::to_host(handle, d_unrenumbered_edge_dsts);

      auto h_cugraph_edge_exists         = cugraph::test::to_host(handle, edge_exists);
      auto h_cugraph_edge_multiplicities = cugraph::test::to_host(handle, edge_multiplicities);
      std::vector<bool> h_reference_edge_exists(edge_srcs.size());
      std::vector<edge_t> h_reference_edge_multiplicities(edge_srcs.size());
      for (size_t i = 0; i < edge_srcs.size(); ++i) {
        auto src      = h_unrenumbered_edge_srcs[i];
        auto dst      = h_unrenumbered_edge_dsts[i];
        auto major    = store_transposed ? dst : src;
        auto minor    = store_transposed ? src : dst;
        auto lower_it = std::lower_bound(
          h_indices.begin() + h_offsets[major], h_indices.begin() + h_offsets[major + 1], minor);
        auto upper_it = std::upper_bound(
          h_indices.begin() + h_offsets[major], h_indices.begin() + h_offsets[major + 1], minor);
        auto multiplicity                  = static_cast<edge_t>(std::distance(lower_it, upper_it));
        h_reference_edge_exists[i]         = multiplicity > 0 ? true : false;
        h_reference_edge_multiplicities[i] = multiplicity;
      }

      ASSERT_TRUE(std::equal(h_reference_edge_exists.begin(),
                             h_reference_edge_exists.end(),
                             h_cugraph_edge_exists.begin()))
        << "has_edge() return values do not match with the reference values.";
      ASSERT_TRUE(std::equal(h_reference_edge_multiplicities.begin(),
                             h_reference_edge_multiplicities.end(),
                             h_cugraph_edge_multiplicities.begin()))
        << "compute_multiplicity() return values do not match with the reference values.";
    }
  }
};

using Tests_HasEdgeAndComputeMultiplicity_File =
  Tests_HasEdgeAndComputeMultiplicity<cugraph::test::File_Usecase>;
using Tests_HasEdgeAndComputeMultiplicity_Rmat =
  Tests_HasEdgeAndComputeMultiplicity<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_HasEdgeAndComputeMultiplicity_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_HasEdgeAndComputeMultiplicity_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_HasEdgeAndComputeMultiplicity_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_HasEdgeAndComputeMultiplicity_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_HasEdgeAndComputeMultiplicity_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_HasEdgeAndComputeMultiplicity_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(HasEdgeAndComputeMultiplicity_Usecase{1024 * 128}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_HasEdgeAndComputeMultiplicity_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(HasEdgeAndComputeMultiplicity_Usecase{1024 * 128}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_HasEdgeAndComputeMultiplicity_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(HasEdgeAndComputeMultiplicity_Usecase{1024 * 1024 * 128, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
