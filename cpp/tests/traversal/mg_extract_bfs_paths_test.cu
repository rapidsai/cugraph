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
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

struct ExtractBFSPaths_Usecase {
  size_t source{0};
  size_t num_paths_to_check{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGExtractBFSPaths
  : public ::testing::TestWithParam<std::tuple<ExtractBFSPaths_Usecase, input_usecase_t>> {
 public:
  Tests_MGExtractBFSPaths() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(ExtractBFSPaths_Usecase const& extract_bfs_paths_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    using weight_t = float;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> d_mg_renumber_map_labels{std::nullopt};
    std::tie(mg_graph, std::ignore, d_mg_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, true, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
    auto mg_graph_view = mg_graph.view();

    ASSERT_TRUE(static_cast<vertex_t>(extract_bfs_paths_usecase.source) >= 0 &&
                static_cast<vertex_t>(extract_bfs_paths_usecase.source) <
                  mg_graph_view.number_of_vertices())
      << "Invalid starting source.";

    ASSERT_TRUE(extract_bfs_paths_usecase.num_paths_to_check > 0) << "Invalid num_paths_to_check";
    ASSERT_TRUE(extract_bfs_paths_usecase.num_paths_to_check < mg_graph_view.number_of_vertices())
      << "Invalid num_paths_to_check, more than number of vertices";

    rmm::device_uvector<vertex_t> d_mg_distances(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle_->get_stream());
    rmm::device_uvector<vertex_t> d_mg_predecessors(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    auto const d_mg_source =
      mg_graph_view.in_local_vertex_partition_range_nocheck(extract_bfs_paths_usecase.source)
        ? std::make_optional<rmm::device_scalar<vertex_t>>(extract_bfs_paths_usecase.source,
                                                           handle_->get_stream())
        : std::nullopt;

    cugraph::bfs(*handle_,
                 mg_graph_view,
                 d_mg_distances.data(),
                 d_mg_predecessors.data(),
                 d_mg_source ? d_mg_source->data() : static_cast<vertex_t const*>(nullptr),
                 d_mg_source ? size_t{1} : size_t{0},
                 false,
                 std::numeric_limits<vertex_t>::max());

    auto h_mg_distances    = cugraph::test::to_host(*handle_, d_mg_distances);
    auto h_mg_predecessors = cugraph::test::to_host(*handle_, d_mg_predecessors);

    vertex_t invalid_vertex = cugraph::invalid_vertex_id<vertex_t>::value;

    auto d_mg_destinations = cugraph::test::randomly_select_destinations<false>(
      *handle_,
      mg_graph_view.local_vertex_partition_range_size(),
      mg_graph_view.local_vertex_partition_range_first(),
      d_mg_predecessors,
      extract_bfs_paths_usecase.num_paths_to_check);

    rmm::device_uvector<vertex_t> d_mg_paths(0, handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG extract_bfs_paths");
    }

    vertex_t mg_max_path_length{0};

    std::tie(d_mg_paths, mg_max_path_length) = extract_bfs_paths(*handle_,
                                                                 mg_graph_view,
                                                                 d_mg_distances.data(),
                                                                 d_mg_predecessors.data(),
                                                                 d_mg_destinations.data(),
                                                                 d_mg_destinations.size());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (extract_bfs_paths_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<rmm::device_uvector<vertex_t>> d_sg_renumber_map_labels{std::nullopt};
      std::tie(sg_graph, std::ignore, d_sg_renumber_map_labels) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
          *handle_, input_usecase, true, false);

      auto sg_graph_view = sg_graph.view();

      rmm::device_uvector<vertex_t> d_sg_destinations(d_mg_destinations.size(),
                                                      handle_->get_stream());
      raft::copy(d_sg_destinations.data(),
                 d_mg_destinations.data(),
                 d_mg_destinations.size(),
                 handle_->get_stream());

      rmm::device_uvector<vertex_t> d_sg_distances(sg_graph_view.number_of_vertices(),
                                                   handle_->get_stream());
      rmm::device_uvector<vertex_t> d_sg_predecessors(sg_graph_view.number_of_vertices(),
                                                      handle_->get_stream());
      rmm::device_uvector<vertex_t> d_sg_paths(0, handle_->get_stream());

      //
      // I think I can do this with allgatherv... think about this.
      // allgather distances and predecessors
      //
      {
        std::vector<size_t> rx_counts(handle_->get_comms().get_size(), size_t{0});
        std::vector<size_t> displacements(handle_->get_comms().get_size(), size_t{0});
        for (int i = 0; i < handle_->get_comms().get_size(); ++i) {
          rx_counts[i]     = mg_graph_view.vertex_partition_range_size(i);
          displacements[i] = (i == 0) ? 0 : displacements[i - 1] + rx_counts[i - 1];
        }
        handle_->get_comms().allgatherv(d_mg_distances.data(),
                                        d_sg_distances.data(),
                                        rx_counts.data(),
                                        displacements.data(),
                                        handle_->get_stream());

        handle_->get_comms().allgatherv(d_mg_predecessors.data(),
                                        d_sg_predecessors.data(),
                                        rx_counts.data(),
                                        displacements.data(),
                                        handle_->get_stream());
      }

      vertex_t sg_max_path_length;

      std::tie(d_sg_paths, sg_max_path_length) = extract_bfs_paths(*handle_,
                                                                   sg_graph_view,
                                                                   d_sg_distances.data(),
                                                                   d_sg_predecessors.data(),
                                                                   d_sg_destinations.data(),
                                                                   d_sg_destinations.size());

      std::vector<vertex_t> h_mg_paths = cugraph::test::to_host(*handle_, d_mg_paths);
      std::vector<vertex_t> h_sg_paths = cugraph::test::to_host(*handle_, d_sg_paths);

      ASSERT_EQ(d_mg_paths.size(), mg_max_path_length * d_mg_destinations.size());
      ASSERT_EQ(d_sg_paths.size(), sg_max_path_length * d_sg_destinations.size());

      for (size_t dest_id = 0; dest_id < d_mg_destinations.size(); ++dest_id) {
        for (vertex_t offset = 0; offset < std::min(sg_max_path_length, mg_max_path_length);
             ++offset) {
          ASSERT_EQ(h_mg_paths[dest_id * mg_max_path_length + offset],
                    h_sg_paths[dest_id * sg_max_path_length + offset]);
        }

        for (vertex_t offset = sg_max_path_length; offset < mg_max_path_length; ++offset) {
          ASSERT_EQ(h_mg_paths[dest_id * mg_max_path_length + offset], invalid_vertex);
        }

        for (vertex_t offset = mg_max_path_length; offset < sg_max_path_length; ++offset) {
          ASSERT_EQ(h_sg_paths[dest_id * sg_max_path_length + offset], invalid_vertex);
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGExtractBFSPaths<input_usecase_t>::handle_ = nullptr;

using Tests_MGExtractBFSPaths_File = Tests_MGExtractBFSPaths<cugraph::test::File_Usecase>;
using Tests_MGExtractBFSPaths_Rmat = Tests_MGExtractBFSPaths<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_MGExtractBFSPaths_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGExtractBFSPaths_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractBFSPaths_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractBFSPaths_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGExtractBFSPaths_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(ExtractBFSPaths_Usecase{0, 10},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 100},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 100},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{100, 100},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{1000, 2000},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{1000, 20000},
                    cugraph::test::File_Usecase("test/datasets/wiki-Talk.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGExtractBFSPaths_Rmat,
                         ::testing::Values(
                           // enable correctness checks
                           std::make_tuple(ExtractBFSPaths_Usecase{0, 20},
                                           cugraph::test::Rmat_Usecase(
                                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGExtractBFSPaths_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_pair(
      ExtractBFSPaths_Usecase{0, 1000, false},
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
