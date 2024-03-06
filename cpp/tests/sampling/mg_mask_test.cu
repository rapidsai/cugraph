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

#include "prims/fill_edge_property.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/property_generator.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_e.cuh"
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

    using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

    // edge mask
    cugraph::edge_property_t<graph_view_t, bool> edge_masks(*handle_, mg_graph_view);
    cugraph::fill_edge_property(*handle_, mg_graph_view, bool{true}, edge_masks);

    cugraph::transform_e(
      *handle_,
      mg_graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_masks.view(),
      [] __device__(auto src, auto dst, auto, auto, auto current_mask) {
        return !(src == dst);  // mask out self-loop
      },
      edge_masks.mutable_view());
    mg_graph_view.attach_edge_mask(edge_masks.view());

    cugraph::transform_e(
      *handle_,
      mg_graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_masks.view(),
      [] __device__(auto src, auto dst, auto, auto, auto current_mask) {
        if (src == dst)
          printf("\nNO WAY %d %d  mask = %d\n",
                 static_cast<int>(src),
                 static_cast<int>(dst),
                 static_cast<int>(current_mask));

        return current_mask;
      },
      edge_masks.mutable_view());
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGGraphColoring<input_usecase_t>::handle_ = nullptr;

using Tests_MGGraphColoring_Rmat = Tests_MGGraphColoring<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGGraphColoring_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGGraphColoring_Rmat,
  ::testing::Combine(
    ::testing::Values(
      // GraphColoring_UseCase{check_correctness},
      GraphColoring_UseCase{true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(4, 10, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
