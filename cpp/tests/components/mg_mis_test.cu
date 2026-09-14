/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/random/rng_state.hpp>

#include <cuda/std/tuple>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <vector>

struct MaximalIndependentSet_Usecase {
  bool check_correctness_{true};
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

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<MaximalIndependentSet_Usecase, input_usecase_t> const& param)
  {
    auto [mis_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    constexpr bool multi_gpu = true;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    raft::random::RngState rng_state(handle_->get_comms().get_rank());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG MIS");
    }

    auto d_mis = cugraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(
      *handle_, mg_graph_view, rng_state);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);

      auto mis_size = cugraph::host_scalar_allreduce(
        handle_->get_comms(), d_mis.size(), raft::comms::op_t::SUM, handle_->get_stream());

      if (handle_->get_comms().get_rank() == 0) {
        std::cout << "MIS size = " << mis_size << ", "
                  << (static_cast<double>(mis_size) /
                      static_cast<double>(mg_graph_view.number_of_vertices())) *
                       100.0
                  << "% of the vertices" << std::endl;
      }
    }

    if (mis_usecase.check_correctness_) {
      auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
      auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

      auto h_mis = cugraph::test::to_host(*handle_, d_mis);

      std::for_each(h_mis.begin(), h_mis.end(), [vertex_first, vertex_last](vertex_t v) {
        ASSERT_TRUE((v >= vertex_first) && (v < vertex_last))
          << v << " is not within the vertex partition range" << std::endl;
      });

      vertex_t local_vtx_partition_size = mg_graph_view.local_vertex_partition_range_size();

      // Flag the vertices included in the MIS
      rmm::device_uvector<vertex_t> inclusion_flags(local_vtx_partition_size,
                                                    handle_->get_stream());
      thrust::fill(
        handle_->get_thrust_policy(), inclusion_flags.begin(), inclusion_flags.end(), vertex_t{0});

      thrust::for_each(handle_->get_thrust_policy(),
                       d_mis.begin(),
                       d_mis.end(),
                       [inclusion_flags = raft::device_span<vertex_t>(inclusion_flags.data(),
                                                                      inclusion_flags.size()),
                        v_first         = vertex_first] __device__(auto v) {
                         inclusion_flags[v - v_first] = vertex_t{1};
                       });

      // Edge endpoints may be owned by other GPUs, so the inclusion flags are materialized as
      // edge source and destination properties before the verification reductions
      cugraph::edge_src_property_t<vertex_t, vertex_t> src_inclusion_cache(*handle_, mg_graph_view);
      cugraph::edge_dst_property_t<vertex_t, vertex_t> dst_inclusion_cache(*handle_, mg_graph_view);
      update_edge_src_property(
        *handle_, mg_graph_view, inclusion_flags.begin(), src_inclusion_cache.mutable_view());
      update_edge_dst_property(
        *handle_, mg_graph_view, inclusion_flags.begin(), dst_inclusion_cache.mutable_view());

      //
      // Number of neighbors included in the MIS, for every vertex. A vertex is adjacent to both
      // its incoming and its outgoing neighbors, and a self loop is not an adjacency.
      //
      rmm::device_uvector<vertex_t> nr_nbrs_included_in_mis(local_vtx_partition_size,
                                                            handle_->get_stream());

      per_v_transform_reduce_outgoing_e(
        *handle_,
        mg_graph_view,
        src_inclusion_cache.view(),
        dst_inclusion_cache.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_included, auto dst_included, auto wt) {
          return (src != dst) ? dst_included : vertex_t{0};
        },
        vertex_t{0},
        cugraph::reduce_op::plus<vertex_t>{},
        nr_nbrs_included_in_mis.begin());

      if (!mg_graph_view.is_symmetric()) {
        rmm::device_uvector<vertex_t> nr_incoming_nbrs_included_in_mis(local_vtx_partition_size,
                                                                       handle_->get_stream());

        per_v_transform_reduce_incoming_e(
          *handle_,
          mg_graph_view,
          src_inclusion_cache.view(),
          dst_inclusion_cache.view(),
          cugraph::edge_dummy_property_t{}.view(),
          [] __device__(auto src, auto dst, auto src_included, auto dst_included, auto wt) {
            return (src != dst) ? src_included : vertex_t{0};
          },
          vertex_t{0},
          cugraph::reduce_op::plus<vertex_t>{},
          nr_incoming_nbrs_included_in_mis.begin());

        thrust::transform(handle_->get_thrust_policy(),
                          nr_nbrs_included_in_mis.begin(),
                          nr_nbrs_included_in_mis.end(),
                          nr_incoming_nbrs_included_in_mis.begin(),
                          nr_nbrs_included_in_mis.begin(),
                          thrust::plus<vertex_t>{});
      }

      //
      // Every GPU counts the violations in its own vertex partition, and the counts are summed
      // across the GPUs.
      //
      auto flag_nr_nbrs_pair_first =
        thrust::make_zip_iterator(inclusion_flags.begin(), nr_nbrs_included_in_mis.begin());

      // Independence: no vertex included in the MIS has a neighbor included in the MIS
      auto num_dependent_vertices =
        thrust::count_if(handle_->get_thrust_policy(),
                         flag_nr_nbrs_pair_first,
                         flag_nr_nbrs_pair_first + local_vtx_partition_size,
                         [] __device__(auto flag_and_nr_nbrs) {
                           return (cuda::std::get<0>(flag_and_nr_nbrs) == vertex_t{1}) &&
                                  (cuda::std::get<1>(flag_and_nr_nbrs) > vertex_t{0});
                         });

      num_dependent_vertices = cugraph::host_scalar_allreduce(handle_->get_comms(),
                                                              num_dependent_vertices,
                                                              raft::comms::op_t::SUM,
                                                              handle_->get_stream());

      ASSERT_EQ(num_dependent_vertices, 0)
        << "A vertex included in the MIS has a neighbor included in the MIS" << std::endl;

      // Maximality: every vertex excluded from the MIS has a neighbor included in the MIS,
      // otherwise the MIS could be augmented with that vertex
      auto num_augmenting_vertices =
        thrust::count_if(handle_->get_thrust_policy(),
                         flag_nr_nbrs_pair_first,
                         flag_nr_nbrs_pair_first + local_vtx_partition_size,
                         [] __device__(auto flag_and_nr_nbrs) {
                           return (cuda::std::get<0>(flag_and_nr_nbrs) == vertex_t{0}) &&
                                  (cuda::std::get<1>(flag_and_nr_nbrs) == vertex_t{0});
                         });

      num_augmenting_vertices = cugraph::host_scalar_allreduce(handle_->get_comms(),
                                                               num_augmenting_vertices,
                                                               raft::comms::op_t::SUM,
                                                               handle_->get_stream());

      ASSERT_EQ(num_augmenting_vertices, 0)
        << "A vertex excluded from the MIS has no neighbor included in the MIS" << std::endl;
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

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGMaximalIndependentSet_File,
  // enable correctness checks
  ::testing::Combine(::testing::Values(MaximalIndependentSet_Usecase{true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                                       cugraph::test::File_Usecase("test/datasets/dolphins.mtx"),
                                       cugraph::test::File_Usecase("test/datasets/netscience.mtx"),
                                       cugraph::test::File_Usecase("test/datasets/polbooks.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGMaximalIndependentSet_Rmat,
  // enable correctness checks, and cover both symmetric and asymmetric graphs
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false),
                      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGMaximalIndependentSet_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
