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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include "k_core_validate.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

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

#include <numeric>
#include <vector>

struct KCore_Usecase {
  size_t k;
  cugraph::k_core_degree_type_t degree_type{cugraph::k_core_degree_type_t::OUT};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGKCore : public ::testing::TestWithParam<std::tuple<KCore_Usecase, input_usecase_t>> {
 public:
  Tests_MGKCore() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(std::tuple<KCore_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber              = true;
    auto [k_core_usecase, input_usecase] = param;

    using weight_t = float;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, renumber, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    rmm::device_uvector<edge_t> d_mg_core_numbers(mg_graph_view.local_vertex_partition_range_size(),
                                                  handle_->get_stream());

    cugraph::core_number(*handle_,
                         mg_graph_view,
                         d_mg_core_numbers.data(),
                         k_core_usecase.degree_type,
                         k_core_usecase.k,
                         k_core_usecase.k);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MG K-core");
    }

    auto [mg_subgraph_srcs, mg_subgraph_dsts, mg_subgraph_weights] =
      cugraph::k_core(*handle_,
                      mg_graph_view,
                      mg_edge_weight_view,
                      k_core_usecase.k,
                      std::nullopt,
                      std::make_optional<raft::device_span<edge_t const>>(
                        d_mg_core_numbers.data(), d_mg_core_numbers.size()));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (k_core_usecase.check_correctness) {
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        mg_subgraph_srcs.data(),
        mg_subgraph_srcs.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        mg_subgraph_dsts.data(),
        mg_subgraph_dsts.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      auto d_mg_aggregate_subgraph_srcs = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(mg_subgraph_srcs.data(), mg_subgraph_srcs.size()));

      auto d_mg_aggregate_subgraph_dsts = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(mg_subgraph_dsts.data(), mg_subgraph_dsts.size()));

      std::optional<rmm::device_uvector<weight_t>> d_mg_aggregate_subgraph_weights{std::nullopt};
      if (mg_subgraph_weights) {
        d_mg_aggregate_subgraph_weights = cugraph::test::device_gatherv(
          *handle_,
          raft::device_span<weight_t const>((*mg_subgraph_weights).data(),
                                            (*mg_subgraph_weights).size()));
      }

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

      auto [sg_graph, sg_edge_weights, sg_edge_ids, sg_number_map] =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          mg_edge_weight_view,
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();
        auto sg_edge_weight_view =
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

        cugraph::test::check_correctness(
          *handle_,
          sg_graph_view,
          sg_edge_weight_view,
          d_mg_aggregate_core_numbers,
          std::make_tuple(std::move(d_mg_aggregate_subgraph_srcs),
                          std::move(d_mg_aggregate_subgraph_dsts),
                          std::move(d_mg_aggregate_subgraph_weights)),
          k_core_usecase.k);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGKCore<input_usecase_t>::handle_ = nullptr;

using Tests_MGKCore_File = Tests_MGKCore<cugraph::test::File_Usecase>;
using Tests_MGKCore_Rmat = Tests_MGKCore<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGKCore_File, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGKCore_Rmat, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGKCore_Rmat, CheckInt32Int64)
{
  run_current_test<int32_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGKCore_Rmat, CheckInt64Int64)
{
  run_current_test<int64_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGKCore_File,
  ::testing::Combine(
    // enable correctness checks
    testing::Values(KCore_Usecase{3, cugraph::k_core_degree_type_t::IN},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::OUT},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::INOUT}),
    testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx"),
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGKCore_Rmat,
  ::testing::Combine(
    // enable correctness checks
    testing::Values(KCore_Usecase{3, cugraph::k_core_degree_type_t::IN},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::OUT},
                    KCore_Usecase{3, cugraph::k_core_degree_type_t::INOUT}),
    testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGKCore_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    testing::Values(KCore_Usecase{3, cugraph::k_core_degree_type_t::OUT, false}),
    testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
