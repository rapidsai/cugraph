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
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/test_utilities.hpp"

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

struct MaximalIndependentSet_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_SGMaximalIndependentSet
  : public ::testing::TestWithParam<std::tuple<MaximalIndependentSet_Usecase, input_usecase_t>> {
 public:
  Tests_SGMaximalIndependentSet() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<MaximalIndependentSet_Usecase, input_usecase_t> const& param)
  {
    auto [mis_usecase, input_usecase] = param;

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
    auto d_mis = cugraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(
      handle, sg_graph_view, rng_state);

    // Test MIS
    if (mis_usecase.check_correctness) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::vector<vertex_t> h_mis(d_mis.size());
      raft::update_host(h_mis.data(), d_mis.data(), d_mis.size(), handle.get_stream());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      auto vertex_first = sg_graph_view.local_vertex_partition_range_first();
      auto vertex_last  = sg_graph_view.local_vertex_partition_range_last();

      std::for_each(h_mis.begin(), h_mis.end(), [vertex_first, vertex_last](vertex_t v) {
        ASSERT_TRUE((v >= vertex_first) && (v < vertex_last));
      });

      // If a vertex is included in MIS, then none of its neighbor should be

      vertex_t local_vtx_partitoin_size = sg_graph_view.local_vertex_partition_range_size();
      rmm::device_uvector<vertex_t> d_total_outgoing_nbrs_included_mis(local_vtx_partitoin_size,
                                                                       handle.get_stream());

      rmm::device_uvector<vertex_t> inclusiong_flags(local_vtx_partitoin_size, handle.get_stream());

      thrust::uninitialized_fill(
        handle.get_thrust_policy(), inclusiong_flags.begin(), inclusiong_flags.end(), vertex_t{0});

      thrust::for_each(
        handle.get_thrust_policy(),
        d_mis.begin(),
        d_mis.end(),
        [inclusiong_flags =
           raft::device_span<vertex_t>(inclusiong_flags.data(), inclusiong_flags.size()),
         v_first = sg_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          auto v_offset              = v - v_first;
          inclusiong_flags[v_offset] = vertex_t{1};
        });

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      per_v_transform_reduce_outgoing_e(
        handle,
        sg_graph_view,
        cugraph::detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
          inclusiong_flags.data()),
        cugraph::detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
          inclusiong_flags.data(), vertex_t{0}),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_included, auto dst_included, auto wt) {
          return (src == dst) ? 0 : dst_included;
        },
        vertex_t{0},
        cugraph::reduce_op::plus<vertex_t>{},
        d_total_outgoing_nbrs_included_mis.begin());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      std::vector<vertex_t> h_total_outgoing_nbrs_included_mis(
        d_total_outgoing_nbrs_included_mis.size());
      raft::update_host(h_total_outgoing_nbrs_included_mis.data(),
                        d_total_outgoing_nbrs_included_mis.data(),
                        d_total_outgoing_nbrs_included_mis.size(),
                        handle.get_stream());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      {
        auto vertex_first = sg_graph_view.local_vertex_partition_range_first();
        auto vertex_last  = sg_graph_view.local_vertex_partition_range_last();

        std::for_each(h_mis.begin(),
                      h_mis.end(),
                      [vertex_first, vertex_last, &h_total_outgoing_nbrs_included_mis](vertex_t v) {
                        ASSERT_TRUE((v >= vertex_first) && (v < vertex_last))
                          << v << " is not within vertex parition range" << std::endl;

                        ASSERT_TRUE(h_total_outgoing_nbrs_included_mis[v - vertex_first] == 0)
                          << v << "'s neighbor is included in MIS" << std::endl;
                      });
      }
    }
  }
};

using Tests_SGMaximalIndependentSet_File =
  Tests_SGMaximalIndependentSet<cugraph::test::File_Usecase>;
using Tests_SGMaximalIndependentSet_Rmat =
  Tests_SGMaximalIndependentSet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_SGMaximalIndependentSet_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_File, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

bool constexpr check_correctness = false;
INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SGMaximalIndependentSet_File,
  ::testing::Combine(::testing::Values(MaximalIndependentSet_Usecase{check_correctness},
                                       MaximalIndependentSet_Usecase{check_correctness}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SGMaximalIndependentSet_Rmat,
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{check_correctness}),
    ::testing::Values(cugraph::test::Rmat_Usecase(3, 4, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_SGMaximalIndependentSet_Rmat,
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{check_correctness},
                      MaximalIndependentSet_Usecase{check_correctness}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
