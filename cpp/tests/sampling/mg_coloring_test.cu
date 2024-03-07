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

#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/property_generator.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/test_utilities.hpp"

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

struct GraphColoring_UseCase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGGraphColoring
  : public ::testing::TestWithParam<std::tuple<GraphColoring_UseCase, input_usecase_t>> {
 public:
  Tests_MGGraphColoring() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<GraphColoring_UseCase, input_usecase_t> const& param)
  {
    auto [coloring_usecase, input_usecase] = param;

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    constexpr bool multi_gpu = true;

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    raft::random::RngState rng_state(multi_gpu ? handle_->get_comms().get_rank() : 0);
    auto d_colors =
      cugraph::coloring<vertex_t, edge_t, multi_gpu>(*handle_, mg_graph_view, rng_state);

    // Test Graph Coloring

    if (coloring_usecase.check_correctness) {
      std::vector<vertex_t> h_colors(d_colors.size());
      raft::update_host(h_colors.data(), d_colors.data(), d_colors.size(), handle_->get_stream());

      std::for_each(h_colors.begin(),
                    h_colors.end(),
                    [num_vertices = mg_graph_view.number_of_vertices()](vertex_t color_id) {
                      ASSERT_TRUE(color_id <= num_vertices);
                    });

      using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
      cugraph::edge_src_property_t<GraphViewType, vertex_t> src_color_cache(*handle_);
      cugraph::edge_dst_property_t<GraphViewType, vertex_t> dst_color_cache(*handle_);

      if constexpr (multi_gpu) {
        src_color_cache =
          cugraph::edge_src_property_t<GraphViewType, vertex_t>(*handle_, mg_graph_view);
        dst_color_cache =
          cugraph::edge_dst_property_t<GraphViewType, vertex_t>(*handle_, mg_graph_view);
        update_edge_src_property(*handle_, mg_graph_view, d_colors.begin(), src_color_cache);
        update_edge_dst_property(*handle_, mg_graph_view, d_colors.begin(), dst_color_cache);
      }

      rmm::device_uvector<uint8_t> d_color_conflicts(
        mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

      per_v_transform_reduce_outgoing_e(
        *handle_,
        mg_graph_view,
        multi_gpu
          ? src_color_cache.view()
          : cugraph::detail::edge_major_property_view_t<vertex_t, vertex_t const*>(d_colors.data()),
        multi_gpu ? dst_color_cache.view()
                  : cugraph::detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                      d_colors.data(), vertex_t{0}),
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
        d_color_conflicts.begin());

      std::vector<uint8_t> h_color_conflicts(d_color_conflicts.size());
      raft::update_host(h_color_conflicts.data(),
                        d_color_conflicts.data(),
                        d_color_conflicts.size(),
                        handle_->get_stream());

      std::vector<vertex_t> h_vertices_in_this_proces((*mg_renumber_map).size());

      raft::update_host(h_vertices_in_this_proces.data(),
                        (*mg_renumber_map).data(),
                        (*mg_renumber_map).size(),
                        handle_->get_stream());
      handle_->sync_stream();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      weight_t nr_conflicts = cugraph::transform_reduce_e(
        *handle_,
        mg_graph_view,
        multi_gpu ? src_color_cache.view()
                  : cugraph::detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
                      d_colors.begin()),
        multi_gpu ? dst_color_cache.view()
                  : cugraph::detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                      d_colors.begin(), vertex_t{0}),
        cugraph::edge_dummy_property_t{}.view(),
        [renumber_map = (*mg_renumber_map).data()] __device__(
          auto src, auto dst, auto src_color, auto dst_color, thrust::nullopt_t) {
          if ((src != dst) && (src_color == dst_color)) {
            return vertex_t{1};
          } else {
            return vertex_t{0};
          }
        },
        vertex_t{0});

      {
        thrust::for_each(
          thrust::host,
          thrust::make_zip_iterator(thrust::make_tuple(
            h_colors.begin(), h_vertices_in_this_proces.begin(), h_color_conflicts.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(
            h_colors.end(), h_vertices_in_this_proces.end(), h_color_conflicts.end())),
          [comm_rank](auto color_vetex_and_conflict_flag) {
            auto color         = thrust::get<0>(color_vetex_and_conflict_flag);
            auto v             = thrust::get<1>(color_vetex_and_conflict_flag);
            auto conflict_flag = thrust::get<2>(color_vetex_and_conflict_flag);
            if (conflict_flag != 0) {
              std::cout << "vertex: " << int{v} << " color:" << int{color}
                        << " conflicting?: " << int{conflict_flag} << std::endl;
            }

            ASSERT_TRUE(conflict_flag == 0)
              << v << " got same color as one of its neighbor" << std::endl;
          });
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGGraphColoring<input_usecase_t>::handle_ = nullptr;

using Tests_MGGraphColoring_File = Tests_MGGraphColoring<cugraph::test::File_Usecase>;
using Tests_MGGraphColoring_Rmat = Tests_MGGraphColoring<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGGraphColoring_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGGraphColoring_File, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGGraphColoring_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGGraphColoring_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGGraphColoring_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGGraphColoring_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

bool constexpr check_correctness = true;

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGGraphColoring_File,
  ::testing::Combine(::testing::Values(GraphColoring_UseCase{check_correctness},
                                       GraphColoring_UseCase{check_correctness}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGGraphColoring_Rmat,
  ::testing::Combine(
    ::testing::Values(GraphColoring_UseCase{check_correctness}),
    ::testing::Values(cugraph::test::Rmat_Usecase(3, 4, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGGraphColoring_Rmat,
  ::testing::Combine(
    ::testing::Values(GraphColoring_UseCase{check_correctness},
                      GraphColoring_UseCase{check_correctness}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
