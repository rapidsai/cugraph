/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <prims/property_generator.cuh>

#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/random/rng_state.hpp>

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

struct MaximalIndependentSet_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGMaximalIndependentSet
  : public ::testing::TestWithParam<std::tuple<MaximalIndependentSet_Usecase, input_usecase_t>> {
 public:
  Tests_MGMaximalIndependentSet() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<MaximalIndependentSet_Usecase, input_usecase_t> const& param)
  {
    auto [mis_usecase, input_usecase] = param;

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
    auto d_mis = cugraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(
      *handle_, mg_graph_view, rng_state);

    // Test MIS
    if (mis_usecase.check_correctness) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::vector<vertex_t> h_mis(d_mis.size());
      raft::update_host(h_mis.data(), d_mis.data(), d_mis.size(), handle_->get_stream());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
      auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

      std::for_each(h_mis.begin(), h_mis.end(), [vertex_first, vertex_last](vertex_t v) {
        ASSERT_TRUE((v >= vertex_first) && (v < vertex_last));
      });

      // If a vertex is included in MIS, then none of its neighbor should be

      vertex_t local_vtx_partitoin_size = mg_graph_view.local_vertex_partition_range_size();
      rmm::device_uvector<vertex_t> d_total_outgoing_nbrs_included_mis(local_vtx_partitoin_size,
                                                                       handle_->get_stream());

      rmm::device_uvector<vertex_t> inclusiong_flags(local_vtx_partitoin_size,
                                                     handle_->get_stream());

      thrust::uninitialized_fill(handle_->get_thrust_policy(),
                                 inclusiong_flags.begin(),
                                 inclusiong_flags.end(),
                                 vertex_t{0});

      thrust::for_each(
        handle_->get_thrust_policy(),
        d_mis.begin(),
        d_mis.end(),
        [inclusiong_flags =
           raft::device_span<vertex_t>(inclusiong_flags.data(), inclusiong_flags.size()),
         v_first = mg_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          auto v_offset              = v - v_first;
          inclusiong_flags[v_offset] = vertex_t{1};
        });

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      // Cache for inclusiong_flags
      using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, true>;
      cugraph::edge_src_property_t<GraphViewType, vertex_t> src_inclusion_cache(*handle_);
      cugraph::edge_dst_property_t<GraphViewType, vertex_t> dst_inclusion_cache(*handle_);

      if constexpr (multi_gpu) {
        src_inclusion_cache =
          cugraph::edge_src_property_t<GraphViewType, vertex_t>(*handle_, mg_graph_view);
        dst_inclusion_cache =
          cugraph::edge_dst_property_t<GraphViewType, vertex_t>(*handle_, mg_graph_view);
        update_edge_src_property(
          *handle_, mg_graph_view, inclusiong_flags.begin(), src_inclusion_cache);
        update_edge_dst_property(
          *handle_, mg_graph_view, inclusiong_flags.begin(), dst_inclusion_cache);
      }

      per_v_transform_reduce_outgoing_e(
        *handle_,
        mg_graph_view,
        multi_gpu ? src_inclusion_cache.view()
                  : cugraph::detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
                      inclusiong_flags.data()),
        multi_gpu ? dst_inclusion_cache.view()
                  : cugraph::detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
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
                        handle_->get_stream());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      {
        auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
        auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

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

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGMaximalIndependentSet<input_usecase_t>::handle_ = nullptr;

using Tests_MGMaximalIndependentSet_File =
  Tests_MGMaximalIndependentSet<cugraph::test::File_Usecase>;
using Tests_MGMaximalIndependentSet_Rmat =
  Tests_MGMaximalIndependentSet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGMaximalIndependentSet_File,
  ::testing::Combine(::testing::Values(MaximalIndependentSet_Usecase{false},
                                       MaximalIndependentSet_Usecase{false}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGMaximalIndependentSet_Rmat,
                         ::testing::Combine(::testing::Values(MaximalIndependentSet_Usecase{false}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              3, 4, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGMaximalIndependentSet_Rmat,
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{false}, MaximalIndependentSet_Usecase{false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
