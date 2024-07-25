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

struct KTruss_Usecase {
  int32_t k_{3};
  bool test_weighted_{false};
  // FIXME: test edge mask
  bool edge_masking_{false};
  bool check_correctness_{true};
};

template <typename input_usecase_t>
class Tests_MGKTruss
  : public ::testing::TestWithParam<std::tuple<KTruss_Usecase, input_usecase_t>> {
 public:
  Tests_MGKTruss() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running KTruss on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(KTruss_Usecase const& k_truss_usecase, input_usecase_t const& input_usecase)
  {
    using weight_t = float;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, edge_weight, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, k_truss_usecase.test_weighted_, true, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (k_truss_usecase.edge_masking_) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG KTruss

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG KTruss");
    }

    auto mg_edge_weight_view =
      edge_weight ? std::make_optional((*edge_weight).view()) : std::nullopt;
    auto [d_cugraph_srcs, d_cugraph_dsts, d_cugraph_wgts] =
      cugraph::k_truss<vertex_t, edge_t, weight_t, true>(
        *handle_, mg_graph_view, mg_edge_weight_view, k_truss_usecase.k_, false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. Compare SG & MG results

    if (k_truss_usecase.check_correctness_) {
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_cugraph_srcs.data(),
        d_cugraph_srcs.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_cugraph_dsts.data(),
        d_cugraph_dsts.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      auto global_d_cugraph_srcs = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(d_cugraph_srcs.data(), d_cugraph_srcs.size()));

      auto global_d_cugraph_dsts = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(d_cugraph_dsts.data(), d_cugraph_srcs.size()));

      rmm::device_uvector<vertex_t> d_sorted_cugraph_srcs{0, handle_->get_stream()};
      rmm::device_uvector<vertex_t> d_sorted_cugraph_dsts{0, handle_->get_stream()};
      rmm::device_uvector<weight_t> d_sorted_cugraph_wgts{0, handle_->get_stream()};

      if (edge_weight) {
        auto global_d_cugraph_wgts = cugraph::test::device_gatherv(
          *handle_,
          raft::device_span<weight_t const>((*d_cugraph_wgts).data(), (*d_cugraph_wgts).size()));

        std::tie(d_sorted_cugraph_srcs, d_sorted_cugraph_dsts, d_sorted_cugraph_wgts) =
          cugraph::test::sort_by_key<vertex_t, weight_t>(
            *handle_, global_d_cugraph_srcs, global_d_cugraph_dsts, global_d_cugraph_wgts);

      } else {
        std::tie(d_sorted_cugraph_srcs, d_sorted_cugraph_dsts) =
          cugraph::test::sort<vertex_t>(*handle_, global_d_cugraph_srcs, global_d_cugraph_dsts);
      }

      // 3-1. Convert to SG graph
      auto [sg_graph, sg_edge_weights, sg_edge_ids, sg_number_map] =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          mg_edge_weight_view,
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      auto sg_edge_weight_view =
        sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

      if (handle_->get_comms().get_rank() == int{0}) {
        auto sg_graph_view = sg_graph.view();

        // 3-2. Run SG KTruss
        auto [ref_d_cugraph_srcs, ref_d_cugraph_dsts, ref_d_cugraph_wgts] =
          cugraph::k_truss<vertex_t, edge_t, weight_t, false>(
            *handle_, sg_graph_view, sg_edge_weight_view, k_truss_usecase.k_, false);

        rmm::device_uvector<vertex_t> d_sorted_ref_cugraph_srcs{0, handle_->get_stream()};
        rmm::device_uvector<vertex_t> d_sorted_ref_cugraph_dsts{0, handle_->get_stream()};
        rmm::device_uvector<weight_t> d_sorted_ref_cugraph_wgts{0, handle_->get_stream()};

        if (edge_weight) {
          std::tie(
            d_sorted_ref_cugraph_srcs, d_sorted_ref_cugraph_dsts, d_sorted_ref_cugraph_wgts) =
            cugraph::test::sort_by_key<vertex_t, weight_t>(
              *handle_, ref_d_cugraph_srcs, ref_d_cugraph_dsts, *ref_d_cugraph_wgts);

        } else {
          std::tie(d_sorted_ref_cugraph_srcs, d_sorted_ref_cugraph_dsts) =
            cugraph::test::sort<vertex_t>(*handle_, ref_d_cugraph_srcs, ref_d_cugraph_dsts);
        }

        // 3-3. Compare
        auto h_cugraph_srcs     = cugraph::test::to_host(*handle_, d_sorted_cugraph_srcs);
        auto h_cugraph_dsts     = cugraph::test::to_host(*handle_, d_sorted_cugraph_dsts);
        auto ref_h_cugraph_srcs = cugraph::test::to_host(*handle_, d_sorted_ref_cugraph_srcs);
        auto ref_h_cugraph_dsts = cugraph::test::to_host(*handle_, d_sorted_ref_cugraph_dsts);

        ASSERT_TRUE(
          std::equal(h_cugraph_srcs.begin(), h_cugraph_srcs.end(), ref_h_cugraph_srcs.begin()));

        ASSERT_TRUE(
          std::equal(h_cugraph_dsts.begin(), h_cugraph_dsts.end(), ref_h_cugraph_dsts.begin()));

        if (edge_weight) {
          auto ref_h_cugraph_wgts = cugraph::test::to_host(*handle_, d_sorted_ref_cugraph_wgts);

          auto h_cugraph_wgts = cugraph::test::to_host(*handle_, d_sorted_cugraph_wgts);

          ASSERT_TRUE(
            std::equal(h_cugraph_wgts.begin(), h_cugraph_wgts.end(), ref_h_cugraph_wgts.begin()));
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGKTruss<input_usecase_t>::handle_ = nullptr;

using Tests_MGKTruss_File = Tests_MGKTruss<cugraph::test::File_Usecase>;
using Tests_MGKTruss_Rmat = Tests_MGKTruss<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGKTruss_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGKTruss_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGKTruss_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGKTruss_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGKTruss_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KTruss_Usecase{4, false, true, true}, KTruss_Usecase{5, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_MGKTruss_Rmat,
  ::testing::Combine(
    ::testing::Values(KTruss_Usecase{4, false, false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGKTruss_Rmat,
  ::testing::Combine(
    ::testing::Values(KTruss_Usecase{4, false, false, false},
                      KTruss_Usecase{5, false, false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
