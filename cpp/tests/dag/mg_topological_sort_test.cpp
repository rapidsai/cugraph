/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dag/dag_test_utilities.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

struct TopologicalSort_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTopologicalSort
  : public ::testing::TestWithParam<std::tuple<TopologicalSort_Usecase, input_usecase_t>> {
 public:
  Tests_MGTopologicalSort() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running topological sort on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(TopologicalSort_Usecase const& topological_sort_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t    = float;
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    ASSERT_FALSE(mg_graph_view.is_symmetric())
      << "Topological sort works only on directed (asymmetric) graphs.";

    std::optional<cugraph::edge_property_t<edge_t, bool>> random_mask{std::nullopt};
    if (topological_sort_usecase.edge_masking) {
      random_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask(random_mask->view());
    }

    // Mask out every edge that lives inside a non-trivial SCC (and every self-loop) so the graph
    // handed to topological_sort is acyclic.

    if (mg_graph_view.has_edge_mask()) { mg_graph_view.clear_edge_mask(); }

    auto acyclic_mask = cugraph::test::build_acyclic_edge_mask(*handle_, mg_graph_view);
    mg_graph_view.attach_edge_mask(acyclic_mask.view());

    // 2. run MG topological sort

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG topological_sort");
    }

    auto d_mg_levels = cugraph::topological_sort(*handle_, mg_graph_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (topological_sort_usecase.check_correctness) {
      // 3-1. aggregate MG results

      rmm::device_uvector<vertex_t> d_mg_aggregate_levels(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_levels) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<vertex_t const>(d_mg_levels.data(), d_mg_levels.size()));

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. run SG topological sort

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        auto d_sg_levels = cugraph::topological_sort(*handle_, sg_graph_view);

        // 3-3. compare

        auto h_mg_aggregate_levels = cugraph::test::to_host(*handle_, d_mg_aggregate_levels);
        auto h_sg_levels           = cugraph::test::to_host(*handle_, d_sg_levels);

        ASSERT_TRUE(
          std::equal(h_sg_levels.begin(), h_sg_levels.end(), h_mg_aggregate_levels.begin()))
          << "Topological sort levels do not match with the SG values.";
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTopologicalSort<input_usecase_t>::handle_ = nullptr;

using Tests_MGTopologicalSort_File = Tests_MGTopologicalSort<cugraph::test::File_Usecase>;
using Tests_MGTopologicalSort_Rmat = Tests_MGTopologicalSort<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTopologicalSort_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTopologicalSort_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTopologicalSort_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTopologicalSort_File,
  ::testing::Combine(
    ::testing::Values(TopologicalSort_Usecase{false}, TopologicalSort_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate-asymmetric.csv"),
                      cugraph::test::File_Usecase("test/datasets/cage6.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGTopologicalSort_Rmat,
  ::testing::Values(
    std::make_tuple(TopologicalSort_Usecase{false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(TopologicalSort_Usecase{true},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                       --gtest_filter to select only the rmat_benchmark_test with a specific
                       vertex & edge type combination) by command line arguments and do not
                       include more than one Rmat_Usecase that differ only in scale or edge
                       factor (to avoid running same benchmarks more than once) */
  Tests_MGTopologicalSort_Rmat,
  ::testing::Values(
    std::make_tuple(TopologicalSort_Usecase{false, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(TopologicalSort_Usecase{true, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
