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

#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_e.cuh"
#include "utilities/base_fixture.hpp"
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

struct GraphColoring_UseCase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_SGGraphColoring
  : public ::testing::TestWithParam<std::tuple<GraphColoring_UseCase, input_usecase_t>> {
 public:
  Tests_SGGraphColoring() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<GraphColoring_UseCase, input_usecase_t> const& param)
  {
    auto [coloring_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      hr_timer.start("Construct graph");
    }

    constexpr bool multi_gpu = false;

    auto [sg_graph, sg_edge_weights, sg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto sg_graph_view = sg_graph.view();
    auto sg_edge_weight_view =
      sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

    raft::random::RngState rng_state(0);
    auto d_colors =
      cugraph::vertex_coloring<vertex_t, edge_t, multi_gpu>(handle, sg_graph_view, rng_state);

    // Test Graph Coloring

    if (coloring_usecase.check_correctness) {
      std::vector<vertex_t> h_colors(d_colors.size());
      raft::update_host(h_colors.data(), d_colors.data(), d_colors.size(), handle.get_stream());

      std::for_each(h_colors.begin(),
                    h_colors.end(),
                    [num_vertices = sg_graph_view.number_of_vertices()](vertex_t color_id) {
                      ASSERT_TRUE(color_id <= num_vertices);
                    });

      rmm::device_uvector<uint8_t> d_color_conflict_flags(
        sg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

      per_v_transform_reduce_outgoing_e(
        handle,
        sg_graph_view,
        cugraph::detail::edge_major_property_view_t<vertex_t, vertex_t const*>(d_colors.data()),
        cugraph::detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(d_colors.data(),
                                                                               vertex_t{0}),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_color, auto dst_color, thrust::nullopt_t) {
          if ((src != dst) && (src_color == dst_color)) {
            return uint8_t{1};
          } else {
            return uint8_t{0};
          }
        },
        uint8_t{0},
        cugraph::reduce_op::maximum<uint8_t>{},
        d_color_conflict_flags.begin());

      std::vector<uint8_t> h_color_conflict_flags(d_color_conflict_flags.size());
      raft::update_host(h_color_conflict_flags.data(),
                        d_color_conflict_flags.data(),
                        d_color_conflict_flags.size(),
                        handle.get_stream());

      std::vector<vertex_t> h_vertices_in_this_proces((*sg_renumber_map).size());

      raft::update_host(h_vertices_in_this_proces.data(),
                        (*sg_renumber_map).data(),
                        (*sg_renumber_map).size(),
                        handle.get_stream());
      handle.sync_stream();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      edge_t nr_conflicts = cugraph::transform_reduce_e(
        handle,
        sg_graph_view,
        cugraph::detail::edge_major_property_view_t<vertex_t, vertex_t const*>(d_colors.begin()),
        cugraph::detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(d_colors.begin(),
                                                                               vertex_t{0}),
        cugraph::edge_dummy_property_t{}.view(),
        [renumber_map = (*sg_renumber_map).data()] __device__(
          auto src, auto dst, auto src_color, auto dst_color, thrust::nullopt_t) {
          if ((src != dst) && (src_color == dst_color)) {
            return vertex_t{1};
          } else {
            return vertex_t{0};
          }
        },
        vertex_t{0});

      ASSERT_TRUE(nr_conflicts == edge_t{0})
        << "adjacent vertices can't have same color." << std::endl;

      if (nr_conflicts >= 0) {
        thrust::for_each(
          thrust::host,
          thrust::make_zip_iterator(thrust::make_tuple(
            h_colors.begin(), h_vertices_in_this_proces.begin(), h_color_conflict_flags.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(
            h_colors.end(), h_vertices_in_this_proces.end(), h_color_conflict_flags.end())),
          [](auto color_vetex_and_conflict_flag) {
            auto color         = thrust::get<0>(color_vetex_and_conflict_flag);
            auto v             = thrust::get<1>(color_vetex_and_conflict_flag);
            auto conflict_flag = thrust::get<2>(color_vetex_and_conflict_flag);
            ASSERT_TRUE(conflict_flag == 0)
              << v << " got same color as one of its neighbor" << std::endl;
          });
      }
    }
  }
};

using Tests_SGGraphColoring_File = Tests_SGGraphColoring<cugraph::test::File_Usecase>;
using Tests_SGGraphColoring_Rmat = Tests_SGGraphColoring<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_SGGraphColoring_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGGraphColoring_File, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGGraphColoring_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGGraphColoring_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGGraphColoring_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGGraphColoring_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

bool constexpr check_correctness = false;

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SGGraphColoring_File,
  ::testing::Combine(::testing::Values(GraphColoring_UseCase{check_correctness},
                                       GraphColoring_UseCase{check_correctness}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SGGraphColoring_Rmat,
  ::testing::Combine(
    ::testing::Values(GraphColoring_UseCase{check_correctness}),
    ::testing::Values(cugraph::test::Rmat_Usecase(3, 4, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_SGGraphColoring_Rmat,
  ::testing::Combine(
    ::testing::Values(GraphColoring_UseCase{check_correctness},
                      GraphColoring_UseCase{check_correctness}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
