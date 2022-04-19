/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "mg_louvain_helper.hpp"

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

void compare(float mg_modularity, float sg_modularity)
{
  ASSERT_FLOAT_EQ(mg_modularity, sg_modularity);
}
void compare(double mg_modularity, double sg_modularity)
{
  ASSERT_DOUBLE_EQ(mg_modularity, sg_modularity);
}

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_SUITE_P()
//
struct Louvain_Usecase {
  size_t max_level_{100};
  double resolution_{1};
  bool check_correctness_{false};
};

////////////////////////////////////////////////////////////////////////////////
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
template <typename input_usecase_t>
class Tests_MG_Louvain
  : public ::testing::TestWithParam<std::tuple<Louvain_Usecase, input_usecase_t>> {
 public:
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  // Run once for each test instance
  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of MNMG Louvain with the results of running
  // each step of SG Louvain, renumbering the coarsened graphs based
  // on the MNMG renumbering.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void compare_sg_results(raft::handle_t const& handle,
                          input_usecase_t const& input_usecase,
                          rmm::device_uvector<vertex_t>& d_renumber_map_gathered_v,
                          cugraph::Dendrogram<vertex_t> const& dendrogram,
                          weight_t resolution,
                          int rank,
                          weight_t mg_modularity)
  {
    auto sg_graph =
      std::make_unique<cugraph::graph_t<vertex_t, edge_t, weight_t, false, false>>(handle);
    rmm::device_uvector<vertex_t> d_clustering_v(0, handle.get_stream());
    weight_t sg_modularity{-1.0};

    if (rank == 0) {
      // Create initial SG graph, renumbered according to the MNMG renumber map

      auto [d_edgelist_srcs,
            d_edgelist_dsts,
            d_edgelist_weights,
            d_vertices,
            number_of_vertices,
            is_symmetric] =
        input_usecase.template construct_edgelist<vertex_t, edge_t, weight_t, false, false>(handle,
                                                                                            true);

      d_clustering_v.resize(d_vertices.size(), handle.get_stream());

      // renumber using d_renumber_map_gathered_v
      cugraph::test::single_gpu_renumber_edgelist_given_number_map(
        handle, d_edgelist_srcs, d_edgelist_dsts, d_renumber_map_gathered_v);

      std::tie(*sg_graph, std::ignore) =
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, false, false>(
          handle,
          std::move(d_vertices),
          std::move(d_edgelist_srcs),
          std::move(d_edgelist_dsts),
          std::move(d_edgelist_weights),
          cugraph::graph_properties_t{is_symmetric, false},
          false);
    }

    std::for_each(
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(dendrogram.num_levels()),
      [&dendrogram, &sg_graph, &d_clustering_v, &sg_modularity, &handle, resolution, rank](
        size_t i) {
        auto d_dendrogram_gathered_v = cugraph::test::device_gatherv(
          handle, dendrogram.get_level_ptr_nocheck(i), dendrogram.get_level_size_nocheck(i));

        if (rank == 0) {
          auto graph_view = sg_graph->view();

          d_clustering_v.resize(graph_view.number_of_vertices(), handle.get_stream());

          std::tie(std::ignore, sg_modularity) =
            cugraph::louvain(handle, graph_view, d_clustering_v.data(), size_t{1}, resolution);

          EXPECT_TRUE(cugraph::test::renumbered_vectors_same(
            handle, d_clustering_v, d_dendrogram_gathered_v));

          sg_graph =
            cugraph::test::coarsen_graph(handle, graph_view, d_dendrogram_gathered_v.data());
        }
      });

    if (rank == 0) compare(mg_modularity, sg_modularity);
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

    auto constexpr pool_size = 64;  // FIXME: tuning parameter
    raft::handle_t handle(rmm::cuda_stream_per_thread,
                          std::make_shared<rmm::cuda_stream_pool>(pool_size));
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto& comm     = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) {
      --row_comm_size;
    }

    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        handle, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto [dendrogram, mg_modularity] = cugraph::louvain(
      handle, mg_graph_view, louvain_usecase.max_level_, louvain_usecase.resolution_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG Louvain took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (louvain_usecase.check_correctness_) {
      SCOPED_TRACE("compare modularity input");

      auto d_renumber_map_gathered_v = cugraph::test::device_gatherv(
        handle, (*d_renumber_map_labels).data(), (*d_renumber_map_labels).size());

      compare_sg_results<vertex_t, edge_t, weight_t>(handle,
                                                     input_usecase,
                                                     d_renumber_map_gathered_v,
                                                     *dendrogram,
                                                     louvain_usecase.resolution_,
                                                     comm_rank,
                                                     mg_modularity);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
using Tests_MG_Louvain_File   = Tests_MG_Louvain<cugraph::test::File_Usecase>;
using Tests_MG_Louvain_File64 = Tests_MG_Louvain<cugraph::test::File_Usecase>;
using Tests_MG_Louvain_Rmat   = Tests_MG_Louvain<cugraph::test::Rmat_Usecase>;
using Tests_MG_Louvain_Rmat64 = Tests_MG_Louvain<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_Louvain_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MG_Louvain_File64, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MG_Louvain_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MG_Louvain_Rmat64, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_file_test,
  Tests_MG_Louvain_File,
  ::testing::Combine(
    // enable correctness checks for small graphs
    ::testing::Values(Louvain_Usecase{100, 1, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  simple_rmat_test,
  Tests_MG_Louvain_Rmat,
  ::testing::Combine(
    // enable correctness checks for small graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MG_Louvain_File,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file64_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MG_Louvain_File64,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MG_Louvain_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat64_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MG_Louvain_Rmat64,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
