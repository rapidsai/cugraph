/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

namespace {

uint64_t bits_of(double x)
{
  uint64_t b{};
  std::memcpy(&b, &x, sizeof(b));
  return b;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_SUITE_P()
//
struct Leiden_Usecase {
  double resolution_{1.0};
  int32_t n_iterations_{2};
  bool test_weighted_{true};
  bool check_correctness_{true};
};

////////////////////////////////////////////////////////////////////////////////
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
template <typename input_usecase_t>
class Tests_MGLeiden
  : public ::testing::TestWithParam<std::tuple<Leiden_Usecase, input_usecase_t>> {
 public:
  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  // Run once for each test instance
  virtual void SetUp() {}
  virtual void TearDown() {}

  // Multi-GPU Leiden runs the single-GPU algorithm on the gathered graph with the seed of rank 0,
  // so its clustering (in internal vertex ids) and modularity must be bitwise identical to
  // single-GPU Leiden on the same graph with the same internal vertex ids, for any number of GPUs.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Leiden_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [usecase, input_usecase] = param;

    HighResTimer hr_timer{};
    auto& comm           = handle_->get_comms();
    auto const comm_rank = comm.get_rank();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, usecase.test_weighted_, true, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    cugraph::leiden_params_t params{};
    params.resolution   = usecase.resolution_;
    params.n_iterations = usecase.n_iterations_;

    // Every rank holds a different state (the C API even requires different seeds per rank); the
    // seed of rank 0 is used.
    uint64_t const seed = 42;
    auto run_mg         = [&]() {
      raft::random::RngState rng_state(seed + static_cast<uint64_t>(comm_rank));
      rmm::device_uvector<vertex_t> clustering(mg_graph_view.local_vertex_partition_range_size(),
                                               handle_->get_stream());
      auto result = cugraph::leiden<vertex_t, edge_t, weight_t, true>(
        *handle_,
        rng_state,
        mg_graph_view,
        mg_edge_weight_view,
        raft::device_span<vertex_t>(clustering.data(), clustering.size()),
        params,
        true);
      return std::make_tuple(std::move(clustering), result);
    };

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Leiden");
    }

    auto [mg_clustering, mg_result] = run_mg();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (!usecase.check_correctness_) return;

    // The result summary is identical on every rank.
    auto const q_bits_0 = cugraph::host_scalar_bcast(
      comm, bits_of(mg_result.modularity), int{0}, handle_->get_stream());
    auto const k_0 =
      cugraph::host_scalar_bcast(comm, mg_result.num_clusters, int{0}, handle_->get_stream());
    ASSERT_EQ(bits_of(mg_result.modularity), q_bits_0);
    ASSERT_EQ(mg_result.num_clusters, k_0);

    // Determinism: a second run gives the identical local clustering.
    {
      auto [mg_clustering2, mg_result2] = run_mg();
      ASSERT_EQ(cugraph::test::to_host(*handle_, mg_clustering),
                cugraph::test::to_host(*handle_, mg_clustering2));
      ASSERT_EQ(bits_of(mg_result.modularity), bits_of(mg_result2.modularity));
    }

    // Labels of every vertex in internal id order (the vertex partition ranges are consecutive in
    // rank order), on rank 0.
    auto mg_aggregate_clustering = cugraph::test::device_gatherv(
      *handle_, raft::device_span<vertex_t const>(mg_clustering.data(), mg_clustering.size()));

    cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
    std::optional<cugraph::edge_property_t<edge_t, weight_t>> sg_edge_weights{std::nullopt};
    std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, std::ignore) =
      cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        false);  // create an SG graph with MG graph vertex IDs

    if (comm_rank == 0) {
      auto sg_graph_view = sg_graph.view();
      auto sg_edge_weight_view =
        sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

      raft::random::RngState rng_state(seed);  // the state of rank 0
      rmm::device_uvector<vertex_t> sg_clustering(sg_graph_view.number_of_vertices(),
                                                  handle_->get_stream());
      auto sg_result = cugraph::leiden<vertex_t, edge_t, weight_t, false>(
        *handle_,
        rng_state,
        sg_graph_view,
        sg_edge_weight_view,
        raft::device_span<vertex_t>(sg_clustering.data(), sg_clustering.size()),
        params);

      ASSERT_EQ(cugraph::test::to_host(*handle_, mg_aggregate_clustering),
                cugraph::test::to_host(*handle_, sg_clustering))
        << "MG clustering differs from SG clustering on the same vertex ids";
      ASSERT_EQ(bits_of(mg_result.modularity), bits_of(sg_result.modularity));
      ASSERT_EQ(mg_result.num_clusters, sg_result.num_clusters);
      ASSERT_EQ(mg_result.num_iterations, sg_result.num_iterations);
      ASSERT_EQ(mg_result.num_levels, sg_result.num_levels);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGLeiden<input_usecase_t>::handle_ = nullptr;

using Tests_MGLeiden_File = Tests_MGLeiden<cugraph::test::File_Usecase>;
using Tests_MGLeiden_Rmat = Tests_MGLeiden<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGLeiden_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLeiden_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLeiden_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLeiden_Rmat, CheckInt64Int64Double)
{
  run_current_test<int64_t, int64_t, double>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGLeiden_File,
  ::testing::Combine(
    ::testing::Values(Leiden_Usecase{1.0, 2, true}, Leiden_Usecase{0.5, -1, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_tests,
                         Tests_MGLeiden_Rmat,
                         ::testing::Combine(::testing::Values(Leiden_Usecase{1.0, 2, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MGLeiden_File,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Leiden_Usecase{1.0, 2, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGLeiden_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Leiden_Usecase{1.0, 2, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(12, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
