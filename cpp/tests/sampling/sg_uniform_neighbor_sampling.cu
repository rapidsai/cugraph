/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "detail/nbr_sampling_utils.cuh"

#include <gtest/gtest.h>

struct Prims_Usecase {
  bool check_correctness{true};
  bool flag_replacement{true};
};

template <typename input_usecase_t>
class Tests_Uniform_Neighbor_Sampling
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_Uniform_Neighbor_Sampling() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    std::cout << "calling construct_graph" << std::endl;

    auto [graph, renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto graph_view                           = graph.view();
    constexpr edge_t indices_per_source       = 2;
    constexpr vertex_t repetitions_per_vertex = 5;
    constexpr vertex_t source_sample_count    = 3;

    // Generate random vertex ids in the range of current gpu

    std::cout << "get random sources" << std::endl;

    // Generate random sources to gather on
    auto random_sources =
      cugraph::test::random_vertex_ids(handle,
                                       graph_view.local_vertex_partition_range_first(),
                                       graph_view.local_vertex_partition_range_last(),
                                       source_sample_count,
                                       repetitions_per_vertex,
                                       uint64_t{0});

    std::vector<int> h_fan_out{indices_per_source};  // depth = 1

    std::cout << "run algorithm" << std::endl;

    auto&& [d_src_out, d_dst_out, d_indices] = cugraph::uniform_nbr_sample(
      handle,
      graph_view,
      raft::device_span<vertex_t>(random_sources.data(), random_sources.size()),
      raft::host_span<const int>(h_fan_out.data(), h_fan_out.size()),
      prims_usecase.flag_replacement);

    std::cout << "check correctness" << std::endl;

    if (prims_usecase.check_correctness) {
      // FIXME:
      //   Need an SG test.  How about:
      //     I) Validate the extracted paths make sense:
      //       1) Extract an SG directed graph from this set of edges
      //       2) For each seed run an SG BFS against this graph
      //       3) For each of the BFS runs, combine the results so that we
      //          mark the smallest shortest path for each vertex (the
      //          smallest distances value if the predecessor is not invalid)
      //       4) Validate that all vertices have a distance < the maximum
      //          number of hops
      //     II) Validate the extracted edges come from the graph
      //       1) Induce a subgraph using the sampled vertices
      //       2) Generate a COO from the subgraph view
      //       3) Consolidate the COO on rank 0
      //       4) Sort the COO by (src, dst)
      //       5) For each (src, dst) pair in the result, search in the
      //          sorted COO to make sure that it exists.  If any don't exist
      //          fail the test
      //

      // TODO:  Also should add a check that all edges are in the original graph...
      //        uniform_neighbor_sample could return random edges and I think that would pass
      //        this check.

#if 0
      if (handle.get_comms().get_rank() == int{0}) {
        std::cout << "results:" << std::endl;
        raft::print_device_vector(
          "  d_start_src", d_start_src.data(), d_start_src.size(), std::cout);
        raft::print_device_vector(
          "  d_aggregate_src", d_aggregate_src.data(), d_aggregate_src.size(), std::cout);
        raft::print_device_vector(
          "  d_aggregate_dst", d_aggregate_dst.data(), d_aggregate_dst.size(), std::cout);
      }
#endif

#if 0
      if (handle.get_comms().get_rank() == int{0}) {
        bool passed = cugraph::test::check_forest_trees_by_rank(
          h_start_in, h_ranks_in, h_src_out, h_dst_out, h_ranks_out);

        ASSERT_TRUE(passed);
      }
#endif
    }
  }
};

using Tests_Uniform_Neighbor_Sampling_File =
  Tests_Uniform_Neighbor_Sampling<cugraph::test::File_Usecase>;

using Tests_Uniform_Neighbor_Sampling_Rmat =
  Tests_Uniform_Neighbor_Sampling<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Uniform_Neighbor_Sampling_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_File, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_File, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Uniform_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true, true}, Prims_Usecase{true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Uniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Uniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
