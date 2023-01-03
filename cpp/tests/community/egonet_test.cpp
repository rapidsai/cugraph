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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <community/egonet_validate.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

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

struct Egonet_Usecase {
  std::vector<int32_t> ego_sources_{};
  int32_t radius_{1};
  bool test_weighted_{false};
  bool check_correctness_{false};
};

template <typename input_usecase_t>
class Tests_Egonet : public ::testing::TestWithParam<std::tuple<Egonet_Usecase, input_usecase_t>> {
 public:
  Tests_Egonet() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Egonet_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [egonet_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    auto n_streams   = std::min(egonet_usecase.ego_sources_.size(), size_t{128});
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

    bool renumber = true;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, egonet_usecase.test_weighted_, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    rmm::device_uvector<vertex_t> d_ego_sources(egonet_usecase.ego_sources_.size(),
                                                handle.get_stream());

    raft::update_device(d_ego_sources.data(),
                        egonet_usecase.ego_sources_.data(),
                        egonet_usecase.ego_sources_.size(),
                        handle.get_stream());

    cugraph::renumber_ext_vertices<vertex_t, false>(handle,
                                                    d_ego_sources.data(),
                                                    d_ego_sources.size(),
                                                    d_renumber_map_labels->data(),
                                                    graph_view.local_vertex_partition_range_first(),
                                                    graph_view.local_vertex_partition_range_last());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Egonet");
    }

    auto [d_ego_edgelist_src, d_ego_edgelist_dst, d_ego_edgelist_wgt, d_ego_edgelist_offsets] =
      cugraph::extract_ego(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<vertex_t const>{d_ego_sources.data(), egonet_usecase.ego_sources_.size()},
        egonet_usecase.radius_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (egonet_usecase.check_correctness_) {
      auto [d_reference_src, d_reference_dst, d_reference_wgt, d_reference_offsets] =
        cugraph::test::egonet_reference(
          handle,
          graph_view,
          edge_weight_view,
          raft::device_span<vertex_t const>{d_ego_sources.data(), d_ego_sources.size()},
          egonet_usecase.radius_);

      cugraph::test::egonet_validate(handle,
                                     d_ego_edgelist_src,
                                     d_ego_edgelist_dst,
                                     d_ego_edgelist_wgt,
                                     d_ego_edgelist_offsets,
                                     d_reference_src,
                                     d_reference_dst,
                                     d_reference_wgt,
                                     d_reference_offsets);
    }
  }
};

using Tests_Egonet_File = Tests_Egonet<cugraph::test::File_Usecase>;
using Tests_Egonet_Rmat = Tests_Egonet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Egonet_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Egonet_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_Egonet_File,
  ::testing::Combine(
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{0}, 1, false, true},
                      Egonet_Usecase{std::vector<int32_t>{0}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{1}, 3, false, true},
                      Egonet_Usecase{std::vector<int32_t>{10, 0, 5}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{9, 3, 10}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test,
  Tests_Egonet_File,
  ::testing::Combine(
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Egonet_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{0}, 1, false, true},
                      Egonet_Usecase{std::vector<int32_t>{0}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{1}, 3, false, true},
                      Egonet_Usecase{std::vector<int32_t>{10, 0, 5}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{9, 3, 10}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Egonet_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
