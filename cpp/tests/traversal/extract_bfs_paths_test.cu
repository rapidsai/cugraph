/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include "randomly_select_destinations.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

struct ExtractBfsPaths_Usecase {
  size_t source{0};
  size_t num_paths_to_check{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_ExtractBfsPaths
  : public ::testing::TestWithParam<std::tuple<ExtractBfsPaths_Usecase, input_usecase_t>> {
 public:
  Tests_ExtractBfsPaths() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(ExtractBfsPaths_Usecase const& extract_bfs_paths_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;
    using weight_t          = float;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, true, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    ASSERT_TRUE(static_cast<vertex_t>(extract_bfs_paths_usecase.source) >= 0 &&
                static_cast<vertex_t>(extract_bfs_paths_usecase.source) <
                  graph_view.number_of_vertices())
      << "Invalid starting source.";

    ASSERT_TRUE(extract_bfs_paths_usecase.num_paths_to_check > 0) << "Invalid num_paths_to_check";
    ASSERT_TRUE(extract_bfs_paths_usecase.num_paths_to_check < graph_view.number_of_vertices())
      << "Invalid num_paths_to_check, more than number of vertices";

    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    rmm::device_scalar<vertex_t> const d_source(extract_bfs_paths_usecase.source,
                                                handle.get_stream());

    cugraph::bfs(handle,
                 graph_view,
                 d_distances.data(),
                 d_predecessors.data(),
                 d_source.data(),
                 size_t{1},
                 false,
                 std::numeric_limits<vertex_t>::max());

    auto h_distances    = cugraph::test::to_host(handle, d_distances);
    auto h_predecessors = cugraph::test::to_host(handle, d_predecessors);

    auto d_destinations = cugraph::test::randomly_select_destinations<false>(
      handle,
      graph_view.number_of_vertices(),
      vertex_t{0},
      d_predecessors,
      extract_bfs_paths_usecase.num_paths_to_check);

    rmm::device_uvector<vertex_t> d_paths(0, handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Extract BFS paths");
    }

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    int32_t max_path_length{};

    std::tie(d_paths, max_path_length) = extract_bfs_paths(handle,
                                                           graph_view,
                                                           d_distances.data(),
                                                           d_predecessors.data(),
                                                           d_destinations.data(),
                                                           d_destinations.size());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (extract_bfs_paths_usecase.check_correctness) {
      vertex_t invalid_vertex = cugraph::invalid_vertex_id<vertex_t>::value;

      std::vector<vertex_t> h_destinations  = cugraph::test::to_host(handle, d_destinations);
      std::vector<vertex_t> h_cugraph_paths = cugraph::test::to_host(handle, d_paths);
      std::vector<vertex_t> h_reference_paths(d_paths.size(), invalid_vertex);

      //
      //  Reference implementation.
      //
      for (size_t i = 0; i < h_destinations.size(); ++i) {
        vertex_t current_vertex = h_destinations[i];

        while (current_vertex != invalid_vertex) {
          h_reference_paths[max_path_length * i + h_distances[current_vertex]] = current_vertex;
          current_vertex = h_predecessors[current_vertex];
        }
      }

      ASSERT_TRUE(
        std::equal(h_reference_paths.begin(), h_reference_paths.end(), h_cugraph_paths.begin()))
        << "extracted paths do not match with the reference values.";
    }
  }
};

using Tests_ExtractBfsPaths_File = Tests_ExtractBfsPaths<cugraph::test::File_Usecase>;
using Tests_ExtractBfsPaths_Rmat = Tests_ExtractBfsPaths<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_ExtractBfsPaths_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_ExtractBfsPaths_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_ExtractBfsPaths_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_ExtractBfsPaths_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_ExtractBfsPaths_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(ExtractBfsPaths_Usecase{0, 10},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(ExtractBfsPaths_Usecase{0, 100},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(ExtractBfsPaths_Usecase{0, 100},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(ExtractBfsPaths_Usecase{100, 100},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(ExtractBfsPaths_Usecase{1000, 2000},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(ExtractBfsPaths_Usecase{1000, 20000},
                    cugraph::test::File_Usecase("test/datasets/wiki-Talk.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_ExtractBfsPaths_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(ExtractBfsPaths_Usecase{0, 20},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_ExtractBfsPaths_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_pair(ExtractBfsPaths_Usecase{0, 1000, false},
                   cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
