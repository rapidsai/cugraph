/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "property_generator.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <prims/count_if_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <cuco/detail/hash_functions.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

struct Prims_Usecase {
  bool check_correctness{true};
  bool test_weighted{false};
};

template <typename input_usecase_t>
class Tests_MGCountIfE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGCountIfE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Verify the results of count_if_e primitive
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename result_t,
            bool store_transposed>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }
    cugraph::graph_t<vertex_t, edge_t, store_transposed, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> d_mg_renumber_map_labels{std::nullopt};
    std::tie(mg_graph, std::ignore, d_mg_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, prims_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG count_if_e

    const int hash_bin_count = 5;

    auto mg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
      *handle_, *d_mg_renumber_map_labels, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG count_if_e");
    }

    auto result = count_if_e(
      *handle_,
      mg_graph_view,
      mg_src_prop.view(),
      mg_dst_prop.view(),
      cugraph::edge_dummy_property_t{}.view(),
      [] __device__(auto row, auto col, auto src_property, auto dst_property, thrust::nullopt_t) {
        return src_property < dst_property;
      });

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (prims_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
          *handle_, input_usecase, prims_usecase.test_weighted, false);
      auto sg_graph_view = sg_graph.view();

      auto sg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
        *handle_,
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
        hash_bin_count);
      auto sg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
        *handle_, sg_graph_view, sg_vertex_prop);
      auto sg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
        *handle_, sg_graph_view, sg_vertex_prop);

      auto expected_result = count_if_e(
        *handle_,
        sg_graph_view,
        sg_src_prop.view(),
        sg_dst_prop.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto row, auto col, auto src_property, auto dst_property, thrust::nullopt_t) {
          return src_property < dst_property;
        });
      ASSERT_TRUE(expected_result == result);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGCountIfE<input_usecase_t>::handle_ = nullptr;

using Tests_MGCountIfE_File = Tests_MGCountIfE<cugraph::test::File_Usecase>;
using Tests_MGCountIfE_Rmat = Tests_MGCountIfE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGCountIfE_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int64FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt64Int64FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(std::get<0>(param),
                                                                             std::get<1>(param));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int64FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt64Int64FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountIfE_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGCountIfE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGCountIfE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGCountIfE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
