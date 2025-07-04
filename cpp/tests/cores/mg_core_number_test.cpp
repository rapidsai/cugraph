/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct CoreNumber_Usecase {
  cugraph::k_core_degree_type_t degree_type{cugraph::k_core_degree_type_t::OUT};
  size_t k_first{0};  // vertices that does not belong to k_first cores will have core numbers of 0
  size_t k_last{std::numeric_limits<size_t>::max()};  // vertices that belong (k_last + 1)-core will
                                                      // have core numbers of k_last

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGCoreNumber
  : public ::testing::TestWithParam<std::tuple<CoreNumber_Usecase, input_usecase_t>> {
 public:
  Tests_MGCoreNumber() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running CoreNumber on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(CoreNumber_Usecase const& core_number_usecase,
                        input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;
    using weight_t    = float;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (core_number_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG CoreNumber

    rmm::device_uvector<edge_t> d_mg_core_numbers(mg_graph_view.local_vertex_partition_range_size(),
                                                  handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Core number");
    }

    cugraph::core_number(*handle_,
                         mg_graph_view,
                         d_mg_core_numbers.data(),
                         core_number_usecase.degree_type,
                         core_number_usecase.k_first,
                         core_number_usecase.k_last);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. copmare SG & MG results

    if (core_number_usecase.check_correctness) {
      // 3-1. aggregate MG results

      rmm::device_uvector<edge_t> d_mg_aggregate_core_numbers(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_core_numbers) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<edge_t const>(d_mg_core_numbers.data(), d_mg_core_numbers.size()));

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);
      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. run SG CoreNumber

        auto sg_graph_view = sg_graph.view();

        ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

        rmm::device_uvector<edge_t> d_sg_core_numbers(sg_graph_view.number_of_vertices(),
                                                      handle_->get_stream());

        cugraph::core_number(*handle_,
                             sg_graph_view,
                             d_sg_core_numbers.data(),
                             core_number_usecase.degree_type,
                             core_number_usecase.k_first,
                             core_number_usecase.k_last);

        // 3-3. compare

        auto h_mg_aggregate_core_numbers =
          cugraph::test::to_host(*handle_, d_mg_aggregate_core_numbers);
        auto h_sg_core_numbers = cugraph::test::to_host(*handle_, d_sg_core_numbers);

        ASSERT_TRUE(std::equal(h_mg_aggregate_core_numbers.begin(),
                               h_mg_aggregate_core_numbers.end(),
                               h_sg_core_numbers.begin()));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGCoreNumber<input_usecase_t>::handle_ = nullptr;

using Tests_MGCoreNumber_File = Tests_MGCoreNumber<cugraph::test::File_Usecase>;
using Tests_MGCoreNumber_Rmat = Tests_MGCoreNumber<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGCoreNumber_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGCoreNumber_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCoreNumber_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGCoreNumber_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::IN, size_t{0}, std::numeric_limits<size_t>::max(), false},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::OUT, size_t{0}, std::numeric_limits<size_t>::max(), false},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::INOUT, size_t{0}, std::numeric_limits<size_t>::max(), false},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::IN, size_t{2}, size_t{2}, false},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::OUT, size_t{1}, size_t{3}, false},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::INOUT, size_t{2}, size_t{4}, false},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::IN, size_t{0}, std::numeric_limits<size_t>::max(), true},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::OUT, size_t{0}, std::numeric_limits<size_t>::max(), true},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::INOUT, size_t{0}, std::numeric_limits<size_t>::max(), true},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::IN, size_t{2}, size_t{2}, true},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::OUT, size_t{1}, size_t{3}, true},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::INOUT, size_t{2}, size_t{4}, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/polbooks.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_MGCoreNumber_Rmat,
  ::testing::Combine(
    ::testing::Values(
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::IN, size_t{0}, std::numeric_limits<size_t>::max(), false},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::OUT, size_t{0}, std::numeric_limits<size_t>::max(), false},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::INOUT, size_t{0}, std::numeric_limits<size_t>::max(), false},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::IN, size_t{0}, std::numeric_limits<size_t>::max(), true},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::OUT, size_t{0}, std::numeric_limits<size_t>::max(), true},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::INOUT, size_t{0}, std::numeric_limits<size_t>::max(), true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGCoreNumber_Rmat,
  ::testing::Combine(
    ::testing::Values(CoreNumber_Usecase{cugraph::k_core_degree_type_t::OUT,
                                         size_t{0},
                                         std::numeric_limits<size_t>::max(),
                                         false,
                                         false},
                      CoreNumber_Usecase{cugraph::k_core_degree_type_t::OUT,
                                         size_t{0},
                                         std::numeric_limits<size_t>::max(),
                                         true,
                                         false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
