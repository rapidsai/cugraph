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
#include <cugraph/utilities/high_res_timer.hpp>

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
class Tests_SGWeightedMatching
  : public ::testing::TestWithParam<std::tuple<WeightedMatching_UseCase, input_usecase_t>> {
 public:
  Tests_SGWeightedMatching() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<WeightedMatching_UseCase, input_usecase_t> const& param)
  {
    auto [weighted_matching_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      hr_timer.start("Construct graph");
    }

    constexpr bool multi_gpu = false;

    bool test_weighted    = true;
    bool renumber         = true;
    bool drop_self_loops  = false;
    bool drop_multi_edges = false;

    auto [sg_graph, sg_edge_weights, sg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    std::tie(sg_graph, sg_edge_weights, sg_renumber_map) = cugraph::symmetrize_graph(
      handle, std::move(sg_graph), std::move(sg_edge_weights), std::move(sg_renumber_map), false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto sg_graph_view = sg_graph.view();
    auto sg_edge_weight_view =
      sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<decltype(sg_graph_view), bool>> edge_mask{std::nullopt};
    if (weighted_matching_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(sg_graph_view), bool>::edge_property(
        handle, sg_graph_view, 2);
      sg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    rmm::device_uvector<vertex_t> d_partners(0, handle.get_stream());
    weight_t total_matching_weights;

    std::forward_as_tuple(d_partners, total_matching_weights) =
      cugraph::approximate_weighted_matching<vertex_t, edge_t, weight_t, multi_gpu>(
        handle, sg_graph_view, (*sg_edge_weights).view());

    if (weighted_matching_usecase.check_correctness) {
      auto h_partners                = cugraph::test::to_host(handle, d_partners);
      auto constexpr invalid_partner = cugraph::invalid_vertex_id<vertex_t>::value;

      std::for_each(h_partners.begin(), h_partners.end(), [&invalid_partner, h_partners](auto& v) {
        if (v != invalid_partner) ASSERT_TRUE(h_partners[h_partners[v]] == v);
      });
    }
  }
};

using Tests_SGWeightedMatching_File = Tests_SGWeightedMatching<cugraph::test::File_Usecase>;
using Tests_SGWeightedMatching_Rmat = Tests_SGWeightedMatching<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_SGWeightedMatching_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGWeightedMatching_File, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGWeightedMatching_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGWeightedMatching_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGWeightedMatching_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGWeightedMatching_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SGWeightedMatching_File,
  ::testing::Combine(::testing::Values(WeightedMatching_UseCase{false},
                                       WeightedMatching_UseCase{true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_SGWeightedMatching_Rmat,
                         ::testing::Combine(::testing::Values(WeightedMatching_UseCase{false},
                                                              WeightedMatching_UseCase{true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              3, 3, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_SGWeightedMatching_Rmat,
  ::testing::Combine(
    ::testing::Values(WeightedMatching_UseCase{false, false},
                      WeightedMatching_UseCase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
