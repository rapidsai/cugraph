/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/random/rng_state.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>

struct WeightedMatching_UseCase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGWeightedMatching
  : public ::testing::TestWithParam<std::tuple<WeightedMatching_UseCase, input_usecase_t>> {
 public:
  Tests_MGWeightedMatching() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<WeightedMatching_UseCase, input_usecase_t> const& param)
  {
    auto [weighted_matching_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    constexpr bool multi_gpu = true;

    bool test_weighted    = false;
    bool renumber         = true;
    bool drop_self_loops  = false;
    bool drop_multi_edges = false;

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        *handle_, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    std::tie(mg_graph, mg_edge_weights, mg_renumber_map) = cugraph::symmetrize_graph(
      *handle_,
      std::move(mg_graph),
      std::move(mg_edge_weights),
      mg_renumber_map ? std::optional<rmm::device_uvector<vertex_t>>(std::move(*mg_renumber_map))
                      : std::nullopt,
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    //
    CUGRAPH_EXPECTS(mg_edge_weight_view.has_value(), "Graph must be weighted");

    auto const comm_rank = handle_->get_comms().get_rank();
    for (size_t ep_idx = 0; ep_idx < mg_graph_view.number_of_local_edge_partitions(); ++ep_idx) {
      raft::device_span<weight_t const> weights_of_edges_stored_in_this_edge_partition{};

      auto value_firsts = mg_edge_weight_view->value_firsts();
      auto edge_counts  = mg_edge_weight_view->edge_counts();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto weights_title = std::string("weights_")
                             .append(std::to_string(comm_rank))
                             .append("_")
                             .append(std::to_string(ep_idx));
      raft::print_device_vector(
        weights_title.c_str(), value_firsts[ep_idx], edge_counts[ep_idx], std::cout);
    }

    //
    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (weighted_matching_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    rmm::device_uvector<vertex_t> mg_partners(0, handle_->get_stream());
    weight_t mg_matching_weights;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG approximate_weighted_matching");
    }

    std::forward_as_tuple(mg_partners, mg_matching_weights) =
      cugraph::approximate_weighted_matching<vertex_t, edge_t, weight_t, multi_gpu>(
        *handle_, mg_graph_view, (*mg_edge_weights).view());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
    if (weighted_matching_usecase.check_correctness) {
      auto h_mg_partners = cugraph::test::to_host(*handle_, mg_partners);

      auto constexpr invalid_partner = cugraph::invalid_vertex_id<vertex_t>::value;

      rmm::device_uvector<vertex_t> mg_aggregate_partners(0, handle_->get_stream());
      std::tie(std::ignore, mg_aggregate_partners) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<vertex_t const>(mg_partners.data(), mg_partners.size()));

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          mg_edge_weight_view,
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>(std::nullopt),
          false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();

        rmm::device_uvector<vertex_t> sg_partners(0, handle_->get_stream());
        weight_t sg_matching_weights;

        std::forward_as_tuple(sg_partners, sg_matching_weights) =
          cugraph::approximate_weighted_matching<vertex_t, edge_t, weight_t, false>(
            *handle_, sg_graph_view, (*sg_edge_weights).view());
        auto h_sg_partners           = cugraph::test::to_host(*handle_, sg_partners);
        auto h_mg_aggregate_partners = cugraph::test::to_host(*handle_, mg_aggregate_partners);

        ASSERT_FLOAT_EQ(mg_matching_weights, sg_matching_weights)
          << "SG and MG matching weights are different";
        ASSERT_TRUE(
          std::equal(h_sg_partners.begin(), h_sg_partners.end(), h_mg_aggregate_partners.begin()));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGWeightedMatching<input_usecase_t>::handle_ = nullptr;

using Tests_MGWeightedMatching_File = Tests_MGWeightedMatching<cugraph::test::File_Usecase>;
using Tests_MGWeightedMatching_Rmat = Tests_MGWeightedMatching<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGWeightedMatching_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGWeightedMatching_File, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGWeightedMatching_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGWeightedMatching_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGWeightedMatching_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGWeightedMatching_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGWeightedMatching_File,
  ::testing::Combine(::testing::Values(WeightedMatching_UseCase{false},
                                       WeightedMatching_UseCase{true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGWeightedMatching_Rmat,
                         ::testing::Combine(::testing::Values(WeightedMatching_UseCase{false},
                                                              WeightedMatching_UseCase{true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              3, 2, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGWeightedMatching_Rmat,
  ::testing::Combine(
    ::testing::Values(WeightedMatching_UseCase{false, false},
                      WeightedMatching_UseCase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
