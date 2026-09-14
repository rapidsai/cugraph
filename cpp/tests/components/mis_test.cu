/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/per_v_transform_reduce_if_incoming_outgoing_e.cuh>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/random/rng_state.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/set_operations.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <vector>

struct MaximalIndependentSet_Usecase {
  bool renumber_{true};
  bool check_correctness_{true};
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

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<MaximalIndependentSet_Usecase, input_usecase_t> const& param)
  {
    auto [mis_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    constexpr bool multi_gpu = false;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("SG Construct graph");
    }

    auto [sg_graph, sg_edge_weights, sg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle, input_usecase, false, mis_usecase.renumber_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto sg_graph_view = sg_graph.view();

    raft::random::RngState rng_state(0);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MIS");
    }

    auto d_mis = cugraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(
      handle, sg_graph_view, rng_state);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);

      std::cout << "MIS size = " << d_mis.size() << ", "
                << (static_cast<double>(d_mis.size()) /
                    static_cast<double>(sg_graph_view.number_of_vertices())) *
                     100.0
                << "% of the vertices" << std::endl;
    }

    if (mis_usecase.check_correctness_) {
      auto vertex_first = sg_graph_view.local_vertex_partition_range_first();
      auto vertex_last  = sg_graph_view.local_vertex_partition_range_last();

      auto vertex_begin = thrust::make_counting_iterator(vertex_first);
      auto vertex_end   = thrust::make_counting_iterator(vertex_last);

      auto h_mis = cugraph::test::to_host(handle, d_mis);

      std::for_each(h_mis.begin(), h_mis.end(), [vertex_first, vertex_last](vertex_t v) {
        ASSERT_TRUE((v >= vertex_first) && (v < vertex_last))
          << v << " is not within the vertex partition range" << std::endl;
      });

      vertex_t local_vtx_partition_size = sg_graph_view.local_vertex_partition_range_size();

      // Flag the vertices included in the MIS
      rmm::device_uvector<vertex_t> inclusion_flags(local_vtx_partition_size, handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   inclusion_flags.begin(),
                   inclusion_flags.end(),
                   vertex_t{0});

      thrust::for_each(handle.get_thrust_policy(),
                       d_mis.begin(),
                       d_mis.end(),
                       [inclusion_flags = raft::device_span<vertex_t>(inclusion_flags.data(),
                                                                      inclusion_flags.size()),
                        v_first = vertex_first] __device__(auto v) {
                         inclusion_flags[v - v_first] = vertex_t{1};
                       });

      //
      // Independence: no two adjacent vertices are both included in the MIS
      //
      rmm::device_uvector<vertex_t> nr_nbrs_included_in_mis(local_vtx_partition_size,
                                                            handle.get_stream());

      per_v_transform_reduce_outgoing_e(
        handle,
        sg_graph_view,
        cugraph::make_edge_src_property_view<vertex_t, vertex_t>(
          sg_graph_view, inclusion_flags.begin(), inclusion_flags.size()),
        cugraph::make_edge_dst_property_view<vertex_t, vertex_t>(
          sg_graph_view, inclusion_flags.begin(), inclusion_flags.size()),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_included, auto dst_included, auto wt) {
          // both endpoints are included in the MIS, a self loop is not a violation
          return ((src != dst) && (src_included == vertex_t{1}) && (dst_included == vertex_t{1}))
                   ? vertex_t{1}
                   : vertex_t{0};
        },
        vertex_t{0},
        cugraph::reduce_op::plus<vertex_t>{},
        nr_nbrs_included_in_mis.begin());

      ASSERT_EQ(thrust::reduce(handle.get_thrust_policy(),
                               nr_nbrs_included_in_mis.begin(),
                               nr_nbrs_included_in_mis.end()),
                vertex_t{0})
        << "A vertex included in the MIS has a neighbor included in the MIS" << std::endl;

      //
      // Maximality: every vertex excluded from the MIS has a neighbor included in the MIS,
      // otherwise the MIS could be augmented with that vertex. Note that
      // per_v_transform_reduce_if_outgoing_e does not write to the vertices without a qualifying
      // edge, hence the output is initialized.
      //
      rmm::device_uvector<vertex_t> excluded_vertices(
        local_vtx_partition_size - static_cast<vertex_t>(d_mis.size()), handle.get_stream());

      thrust::set_difference(handle.get_thrust_policy(),
                             vertex_begin,
                             vertex_end,
                             d_mis.begin(),
                             d_mis.end(),
                             excluded_vertices.begin());

      cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier(handle, 1);
      vertex_frontier.bucket(0).insert(excluded_vertices.begin(), excluded_vertices.end());

      rmm::device_uvector<vertex_t> any_nbr_included_in_mis(excluded_vertices.size(),
                                                            handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   any_nbr_included_in_mis.begin(),
                   any_nbr_included_in_mis.end(),
                   vertex_t{0});

      per_v_transform_reduce_if_outgoing_e(
        handle,
        sg_graph_view,
        vertex_frontier.bucket(0),
        cugraph::make_edge_src_property_view<vertex_t, vertex_t>(
          sg_graph_view, inclusion_flags.begin(), inclusion_flags.size()),
        cugraph::make_edge_dst_property_view<vertex_t, vertex_t>(
          sg_graph_view, inclusion_flags.begin(), inclusion_flags.size()),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_included, auto dst_included, auto wt) {
          return vertex_t{1};
        },
        vertex_t{0},
        cugraph::reduce_op::any<vertex_t>{},
        [] __device__(auto src, auto dst, auto src_included, auto dst_included, auto wt) {
          return dst_included == vertex_t{1};
        },
        any_nbr_included_in_mis.begin());

      ASSERT_EQ(thrust::reduce(handle.get_thrust_policy(),
                               any_nbr_included_in_mis.begin(),
                               any_nbr_included_in_mis.end()),
                static_cast<vertex_t>(excluded_vertices.size()))
        << "A vertex excluded from the MIS has no neighbor included in the MIS" << std::endl;
    }
  }
};

using Tests_SGMaximalIndependentSet_File =
  Tests_SGMaximalIndependentSet<cugraph::test::File_Usecase>;
using Tests_SGMaximalIndependentSet_Rmat =
  Tests_SGMaximalIndependentSet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_SGMaximalIndependentSet_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SGMaximalIndependentSet_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SGMaximalIndependentSet_File,
  // enable correctness checks, and cover both the renumbered (degree ordered) and the
  // non-renumbered vertex partitions
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{true, true},
                      MaximalIndependentSet_Usecase{false, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"),
                      cugraph::test::File_Usecase("test/datasets/polbooks.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SGMaximalIndependentSet_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{true, true},
                      MaximalIndependentSet_Usecase{false, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_SGMaximalIndependentSet_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
