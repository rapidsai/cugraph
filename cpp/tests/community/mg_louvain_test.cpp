/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_SUITE_P()
//
struct Louvain_Usecase {
  size_t max_level_{100};
  double resolution_{1};
  bool check_correctness_{true};
};

////////////////////////////////////////////////////////////////////////////////
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
template <typename input_usecase_t>
class Tests_MGLouvain
  : public ::testing::TestWithParam<std::tuple<Louvain_Usecase, input_usecase_t>> {
 public:
  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  // Run once for each test instance
  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of MNMG Louvain with the results of running
  // each step of SG Louvain, renumbering the coarsened graphs based
  // on the MNMG renumbering.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void compare_sg_results(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> mg_edge_weight_view,
    cugraph::Dendrogram<vertex_t> const& mg_dendrogram,
    weight_t resolution,
    weight_t mg_modularity)
  {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();

    cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(handle);
    std::optional<
      cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
      sg_edge_weights{std::nullopt};
    std::tie(sg_graph, sg_edge_weights, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::optional<rmm::device_uvector<vertex_t>>{std::nullopt},
      false);  // crate an SG graph with MG graph vertex IDs

    weight_t sg_modularity{-1.0};

    std::for_each(
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(mg_dendrogram.num_levels()),
      [&mg_dendrogram, &sg_graph, &sg_edge_weights, &sg_modularity, &handle, resolution, comm_rank](
        size_t i) {
        auto d_mg_aggregate_cluster_v =
          cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
            handle,
            std::optional<raft::device_span<vertex_t const>>{std::nullopt},
            std::optional<raft::device_span<vertex_t const>>{std::nullopt},
            raft::device_span<vertex_t const>(mg_dendrogram.get_level_ptr_nocheck(i),
                                              mg_dendrogram.get_level_size_nocheck(i)));

        if (comm_rank == 0) {
          auto sg_graph_view = sg_graph.view();
          auto sg_edge_weight_view =
            sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

          rmm::device_uvector<vertex_t> d_sg_cluster_v(sg_graph_view.number_of_vertices(),
                                                       handle_->get_stream());

          std::tie(std::ignore, sg_modularity) = cugraph::louvain(handle,
                                                                  sg_graph_view,
                                                                  sg_edge_weight_view,
                                                                  d_sg_cluster_v.data(),
                                                                  size_t{1},
                                                                  resolution);

          EXPECT_TRUE(cugraph::test::check_invertible(
            handle,
            raft::device_span<vertex_t const>(d_sg_cluster_v.data(), d_sg_cluster_v.size()),
            raft::device_span<vertex_t const>(d_mg_aggregate_cluster_v.data(),
                                              d_mg_aggregate_cluster_v.size())))
            << "(i = " << i << "), sg_modularity = " << sg_modularity;

          std::tie(sg_graph, sg_edge_weights, std::ignore) = cugraph::coarsen_graph(
            handle, sg_graph_view, sg_edge_weight_view, d_mg_aggregate_cluster_v.data(), false);
        }
      });

    if (comm_rank == 0)
      EXPECT_NEAR(mg_modularity, sg_modularity, std::max(mg_modularity, sg_modularity) * 1e-4);
  }

  // Compare the results of running louvain on multiple GPUs to that of a
  // single-GPU run for the configuration in param.  Note that MNMG Louvain
  // and single GPU Louvain are ONLY deterministic through a single
  // iteration of the outer loop.  Renumbering of the partitions when coarsening
  // the graph is a function of the number of GPUs in the GPU cluster.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Louvain_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [louvain_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Louvain");
    }

    auto [dendrogram, mg_modularity] =
      cugraph::louvain<vertex_t, edge_t, weight_t, true>(*handle_,
                                                         mg_graph_view,
                                                         mg_edge_weight_view,
                                                         louvain_usecase.max_level_,
                                                         louvain_usecase.resolution_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (louvain_usecase.check_correctness_) {
      SCOPED_TRACE("compare modularity input");

      compare_sg_results<vertex_t, edge_t, weight_t>(*handle_,
                                                     mg_graph_view,
                                                     mg_edge_weight_view,
                                                     *dendrogram,
                                                     louvain_usecase.resolution_,
                                                     mg_modularity);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGLouvain<input_usecase_t>::handle_ = nullptr;

////////////////////////////////////////////////////////////////////////////////
using Tests_MGLouvain_File = Tests_MGLouvain<cugraph::test::File_Usecase>;
using Tests_MGLouvain_Rmat = Tests_MGLouvain<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGLouvain_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLouvain_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLouvain_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLouvain_Rmat, CheckInt32Int64Float)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLouvain_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGLouvain_File,
  ::testing::Combine(
    // enable correctness checks for small graphs
    ::testing::Values(Louvain_Usecase{100, 1}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_tests,
                         Tests_MGLouvain_Rmat,
                         ::testing::Combine(::testing::Values(Louvain_Usecase{100, 1}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MGLouvain_File,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{100, 1, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGLouvain_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{100, 1, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
