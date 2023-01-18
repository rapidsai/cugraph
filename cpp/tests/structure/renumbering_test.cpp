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
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

struct Renumbering_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Renumbering
  : public ::testing::TestWithParam<std::tuple<Renumbering_Usecase, input_usecase_t>> {
 public:
  Tests_Renumbering() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Renumbering_Usecase const& renumbering_usecase,
                        input_usecase_t const& input_usecase)
  {
    raft::handle_t handle{};
    HighResTimer hr_timer{};

    std::vector<vertex_t> h_original_src_v{};
    std::vector<vertex_t> h_original_dst_v{};
    std::vector<vertex_t> h_final_src_v{};
    std::vector<vertex_t> h_final_dst_v{};

    rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());
    rmm::device_uvector<vertex_t> renumber_map_labels_v(0, handle.get_stream());

    std::tie(src_v, dst_v, std::ignore, std::ignore, std::ignore) =
      input_usecase.template construct_edgelist<vertex_t, weight_t>(handle, false, false, false);

    if (renumbering_usecase.check_correctness) {
      h_original_src_v = cugraph::test::to_host(handle, src_v);
      h_original_dst_v = cugraph::test::to_host(handle, dst_v);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Renumbering");
    }

    std::tie(renumber_map_labels_v, std::ignore) =
      cugraph::renumber_edgelist<vertex_t, edge_t, false>(
        handle, std::nullopt, src_v.begin(), dst_v.begin(), src_v.size(), false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (renumbering_usecase.check_correctness) {
      cugraph::unrenumber_local_int_vertices(handle,
                                             src_v.data(),
                                             src_v.size(),
                                             renumber_map_labels_v.data(),
                                             0,
                                             static_cast<vertex_t>(renumber_map_labels_v.size()));
      cugraph::unrenumber_local_int_vertices(handle,
                                             dst_v.data(),
                                             dst_v.size(),
                                             renumber_map_labels_v.data(),
                                             0,
                                             static_cast<vertex_t>(renumber_map_labels_v.size()));

      h_final_src_v = cugraph::test::to_host(handle, src_v);
      h_final_dst_v = cugraph::test::to_host(handle, dst_v);

      EXPECT_EQ(h_original_src_v, h_original_src_v);
      EXPECT_EQ(h_original_dst_v, h_original_dst_v);
    }
  }
};

using Tests_Renumbering_File = Tests_Renumbering<cugraph::test::File_Usecase>;
using Tests_Renumbering_Rmat = Tests_Renumbering<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_Renumbering_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

// FIXME: add tests for type combinations
TEST_P(Tests_Renumbering_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Renumbering_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Renumbering_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("negative-vertex-id.csv"),
                      cugraph::test::File_Usecase("karate.csv"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_Renumbering_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Renumbering_Usecase{}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Renumbering_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Renumbering_Usecase{false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
