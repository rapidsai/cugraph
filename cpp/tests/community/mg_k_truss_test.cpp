/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

struct EdgeTriangleCount_Usecase {
  bool edge_masking_{false};
  bool check_correctness_{true};
};

template <typename input_usecase_t>
class Tests_MGEdgeTriangleCount
  : public ::testing::TestWithParam<std::tuple<EdgeTriangleCount_Usecase, input_usecase_t>> {
 public:
  Tests_MGEdgeTriangleCount() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running EdgeTriangleCount on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(EdgeTriangleCount_Usecase const& edge_triangle_count_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t = float;

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

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (edge_triangle_count_usecase.edge_masking_) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG EdgeTriangleCount

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG EdgeTriangleCount");
    }

    /*
    auto d_mg_cugraph_results =
      cugraph::edge_triangle_count<vertex_t, edge_t, true>(*handle_, mg_graph_view);
    */

    auto [d_cugraph_srcs, d_cugraph_dsts, d_cugraph_wgts] =
      cugraph::k_truss<vertex_t, edge_t, weight_t, true>(
        *handle_,
        mg_graph_view,
        // edge_weight ? std::make_optional((*edge_weight).view()) : std::nullopt,
        std::nullopt,  // FIXME: test weights
        // k_truss_usecase.k_,
        4,
        false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. Compare SG & MG results

#if 0
    if (edge_triangle_count_usecase.check_correctness_) {
      // 3-1. Convert to SG graph

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, edge_t>>
        d_sg_cugraph_results{std::nullopt};
      std::tie(sg_graph, std::ignore, d_sg_cugraph_results, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          // FIXME: Update 'create_graph_from_edgelist' to support int32_t and int64_t values
          std::make_optional(d_mg_cugraph_results.view()),
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. Convert the MG triangle counts stored as 'edge_property_t' to device vector

        auto [edgelist_srcs,
              edgelist_dsts,
              d_edgelist_weights,
              d_edge_triangle_counts,
              d_edgelist_type] =
          cugraph::decompress_to_edgelist(
            *handle_,
            sg_graph.view(),
            std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
            // FIXME: Update 'decompress_edgelist' to support int32_t and int64_t values
            std::make_optional((*d_sg_cugraph_results).view()),
            std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
            std::optional<raft::device_span<vertex_t const>>{
              std::nullopt});  // FIXME: No longer needed

        // 3-3. Run SG EdgeTriangleCount

        auto ref_d_sg_cugraph_results =
          cugraph::edge_triangle_count<vertex_t, edge_t, false>(*handle_, sg_graph.view());
        auto [ref_edgelist_srcs,
              ref_edgelist_dsts,
              ref_d_edgelist_weights,
              ref_d_edge_triangle_counts] =
          cugraph::decompress_to_edgelist(
            *handle_,
            sg_graph.view(),
            std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
            std::make_optional(ref_d_sg_cugraph_results.view()),
            std::optional<raft::device_span<vertex_t const>>{
              std::nullopt});  // FIXME: No longer needed

        // 3-4. Compare

        auto h_mg_edge_triangle_counts = cugraph::test::to_host(*handle_, *d_edge_triangle_counts);
        auto h_sg_edge_triangle_counts =
          cugraph::test::to_host(*handle_, *ref_d_edge_triangle_counts);

        ASSERT_TRUE(std::equal(h_mg_edge_triangle_counts.begin(),
                               h_mg_edge_triangle_counts.end(),
                               h_sg_edge_triangle_counts.begin()));
      }
    }
#endif
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGEdgeTriangleCount<input_usecase_t>::handle_ = nullptr;

using Tests_MGEdgeTriangleCount_File = Tests_MGEdgeTriangleCount<cugraph::test::File_Usecase>;
// using Tests_MGEdgeTriangleCount_Rmat = Tests_MGEdgeTriangleCount<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGEdgeTriangleCount_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}
#if 0
TEST_P(Tests_MGEdgeTriangleCount_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGEdgeTriangleCount_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGEdgeTriangleCount_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}
#endif

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGEdgeTriangleCount_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EdgeTriangleCount_Usecase{false, false}
                      // EdgeTriangleCount_Usecase{true, true}
                      ),
    ::testing::Values(
      cugraph::test::File_Usecase("/raid/jnke/optimize_ktruss/datasets/test_datasets.mtx")
      // cugraph::test::File_Usecase("test/datasets/dolphins.mtx")
      )));

#if 0
INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_MGEdgeTriangleCount_Rmat,
  ::testing::Combine(
    ::testing::Values(EdgeTriangleCount_Usecase{false, true},
                      EdgeTriangleCount_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGEdgeTriangleCount_Rmat,
  ::testing::Combine(
    ::testing::Values(EdgeTriangleCount_Usecase{false, false},
                      EdgeTriangleCount_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));
#endif

CUGRAPH_MG_TEST_PROGRAM_MAIN()
