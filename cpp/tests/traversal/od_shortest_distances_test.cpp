/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

// Dijkstra's algorithm
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_reference(edge_t const* offsets,
                    vertex_t const* indices,
                    weight_t const* weights,
                    weight_t* distances,
                    vertex_t num_vertices,
                    vertex_t source,
                    weight_t cutoff = std::numeric_limits<weight_t>::max())
{
  using queue_item_t = std::tuple<weight_t, vertex_t>;

  std::fill(distances, distances + num_vertices, std::numeric_limits<weight_t>::max());

  *(distances + source) = weight_t{0.0};
  std::priority_queue<queue_item_t, std::vector<queue_item_t>, std::greater<queue_item_t>> queue{};
  queue.push(std::make_tuple(weight_t{0.0}, source));

  while (queue.size() > 0) {
    weight_t distance{};
    vertex_t row{};
    std::tie(distance, row) = queue.top();
    queue.pop();
    if (distance <= *(distances + row)) {
      auto nbr_offsets     = *(offsets + row);
      auto nbr_offset_last = *(offsets + row + 1);
      for (auto nbr_offset = nbr_offsets; nbr_offset != nbr_offset_last; ++nbr_offset) {
        auto nbr          = *(indices + nbr_offset);
        auto new_distance = distance + *(weights + nbr_offset);
        auto threshold    = std::min(*(distances + nbr), cutoff);
        if (new_distance < threshold) {
          *(distances + nbr) = new_distance;
          queue.push(std::make_tuple(new_distance, nbr));
        }
      }
    }
  }
}

struct ODShortestDistances_Usecase {
  size_t num_origins{0};
  size_t num_destinations{0};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_ODShortestDistances
  : public ::testing::TestWithParam<std::tuple<ODShortestDistances_Usecase, input_usecase_t>> {
 public:
  Tests_ODShortestDistances() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(ODShortestDistances_Usecase const& od_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, true, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<decltype(graph_view), bool>> edge_mask{std::nullopt};
    if (od_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    raft::random::RngState rng_state(0);
    auto origins = cugraph::select_random_vertices<vertex_t, edge_t, false, false>(
      handle, graph_view, std::nullopt, rng_state, od_usecase.num_origins, false, true);
    auto destinations = cugraph::select_random_vertices<vertex_t, edge_t, false, false>(
      handle, graph_view, std::nullopt, rng_state, od_usecase.num_destinations, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("ODShortestDistances");
    }

    auto od_matrix = cugraph::od_shortest_distances(
      handle,
      graph_view,
      *edge_weight_view,
      raft::device_span<vertex_t const>(origins.data(), origins.size()),
      raft::device_span<vertex_t const>(destinations.data(), destinations.size()),
      std::numeric_limits<weight_t>::max(),
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (od_usecase.check_correctness) {
      auto [h_offsets, h_indices, h_weights] =
        cugraph::test::graph_to_host_csr<vertex_t, edge_t, weight_t, false, false>(
          handle,
          graph_view,
          edge_weight_view,
          d_renumber_map_labels
            ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
                                                                    (*d_renumber_map_labels).size())
            : std::nullopt);
      assert(h_weights.has_value());

      auto unrenumbered_origins      = cugraph::test::to_host(handle, origins);
      auto unrenumbered_destinations = cugraph::test::to_host(handle, destinations);
      if (renumber) {
        auto h_renumber_map_labels = cugraph::test::to_host(handle, *d_renumber_map_labels);
        std::transform(unrenumbered_origins.begin(),
                       unrenumbered_origins.end(),
                       unrenumbered_origins.begin(),
                       [&h_renumber_map_labels](auto v) { return h_renumber_map_labels[v]; });
        std::transform(unrenumbered_destinations.begin(),
                       unrenumbered_destinations.end(),
                       unrenumbered_destinations.begin(),
                       [&h_renumber_map_labels](auto v) { return h_renumber_map_labels[v]; });
      }

      std::vector<weight_t> h_reference_od_matrix(od_matrix.size());
      for (size_t i = 0; i < unrenumbered_origins.size(); ++i) {
        std::vector<weight_t> reference_distances(graph_view.number_of_vertices());

        sssp_reference(h_offsets.data(),
                       h_indices.data(),
                       (*h_weights).data(),
                       reference_distances.data(),
                       graph_view.number_of_vertices(),
                       unrenumbered_origins[i],
                       std::numeric_limits<weight_t>::max());

        for (size_t j = 0; j < unrenumbered_destinations.size(); ++j) {
          h_reference_od_matrix[i * unrenumbered_destinations.size() + j] =
            reference_distances[unrenumbered_destinations[j]];
        }
      }

      auto h_cugraph_od_matrix = cugraph::test::to_host(handle, od_matrix);

      auto max_weight_element = std::max_element((*h_weights).begin(), (*h_weights).end());
      auto epsilon            = (*max_weight_element) * weight_t{1e-6};
      auto nearly_equal = [epsilon](auto lhs, auto rhs) { return std::fabs(lhs - rhs) < epsilon; };

      ASSERT_TRUE(std::equal(h_reference_od_matrix.begin(),
                             h_reference_od_matrix.end(),
                             h_cugraph_od_matrix.begin(),
                             nearly_equal))
        << "distances do not match with the reference values.";
    }
  }
};

using Tests_ODShortestDistances_File = Tests_ODShortestDistances<cugraph::test::File_Usecase>;
using Tests_ODShortestDistances_Rmat = Tests_ODShortestDistances<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_ODShortestDistances_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_ODShortestDistances_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_ODShortestDistances_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_ODShortestDistances_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_ODShortestDistances_File,
  // enable correctness checks
  ::testing::Values(std::make_tuple(ODShortestDistances_Usecase{5, 5, false},
                                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
                    std::make_tuple(ODShortestDistances_Usecase{5, 5, true},
                                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
                    std::make_tuple(ODShortestDistances_Usecase{10, 20, false},
                                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
                    std::make_tuple(ODShortestDistances_Usecase{10, 20, true},
                                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
                    std::make_tuple(ODShortestDistances_Usecase{50, 100, false},
                                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
                    std::make_tuple(ODShortestDistances_Usecase{50, 100, true},
                                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_ODShortestDistances_Rmat,
  // enable correctness checks
  ::testing::Values(
    std::make_tuple(ODShortestDistances_Usecase{10, 100, false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(ODShortestDistances_Usecase{10, 100, true},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_ODShortestDistances_Rmat,
  // disable correctness checks for large graphs
  ::testing::Values(
    std::make_tuple(ODShortestDistances_Usecase{10, 100, false, false},
                    cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(ODShortestDistances_Usecase{10, 100, true, false},
                    cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
