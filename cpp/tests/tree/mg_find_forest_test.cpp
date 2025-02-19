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

struct FindForest_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGFindForest
  : public ::testing::TestWithParam<std::tuple<FindForest_Usecase, input_usecase_t>> {
 public:
  Tests_MGFindForest() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running find_forest on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(FindForest_Usecase const& find_forest_usecase,
                        input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;
    using weight_t    = float;

    bool constexpr renumber         = true;
    bool constexpr test_weighted    = false;
    bool constexpr drop_self_loops  = true;
    bool constexpr drop_multi_edges = true;

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
        *handle_, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (find_forest_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG find_forest

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG find_forest");
    }

    auto d_mg_parents = cugraph::find_forest(*handle_, mg_graph_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (find_forest_usecase.check_correctness) {
      // 3-1. unrenumber & aggregate MG source & results

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_parents.data(),
        d_mg_parents.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      // 3-2. aggregate MG results

      rmm::device_uvector<vertex_t> d_mg_aggregate_parents(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_parents) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<vertex_t const>(d_mg_parents.data(), d_mg_parents.size()));

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
        // 3-3. run SG find_forest

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        auto d_sg_parents = cugraph::find_forest(*handle_, sg_graph_view);

        // 3-4. compare

        std::vector<vertex_t> h_mg_aggregate_parents =
          cugraph::test::to_host(*handle_, d_mg_aggregate_parents);

        std::vector<vertex_t> h_sg_parents = cugraph::test::to_host(*handle_, d_sg_parents);

        ASSERT_TRUE(std::equal(
          h_mg_aggregate_parents.begin(), h_mg_aggregate_parents.end(), h_sg_parents.begin()));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGFindForest<input_usecase_t>::handle_ = nullptr;

using Tests_MGFindForest_File = Tests_MGFindForest<cugraph::test::File_Usecase>;
using Tests_MGFindForest_Rmat = Tests_MGFindForest<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGFindForest_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGFindForest_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGFindForest_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGFindForest_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(FindForest_Usecase{false}, FindForest_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGFindForest_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(
      FindForest_Usecase{false},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false)),
    std::make_tuple(
      FindForest_Usecase{true},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGFindForest_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(
      FindForest_Usecase{false, false},
      cugraph::test::Rmat_Usecase(
        20, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false /* scramble vertex IDs */)),
    std::make_tuple(
      FindForest_Usecase{true, false},
      cugraph::test::Rmat_Usecase(
        20, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false /* scramble vertex IDs */))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
