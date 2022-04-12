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

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct BFS_Usecase {
  size_t source{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGBFS : public ::testing::TestWithParam<std::tuple<BFS_Usecase, input_usecase_t>> {
 public:
  Tests_MGBFS() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running BFS on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(BFS_Usecase const& bfs_usecase, input_usecase_t const& input_usecase)
  {
    using weight_t = float;

    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) {
      --row_comm_size;
    }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        handle, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    ASSERT_TRUE(static_cast<vertex_t>(bfs_usecase.source) >= 0 &&
                static_cast<vertex_t>(bfs_usecase.source) < mg_graph_view.number_of_vertices())
      << "Invalid starting source.";

    // 3. run MG BFS

    rmm::device_uvector<vertex_t> d_mg_distances(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle.get_stream());
    rmm::device_uvector<vertex_t> d_mg_predecessors(
      mg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto const d_mg_source =
      mg_graph_view.in_local_vertex_partition_range_nocheck(bfs_usecase.source)
        ? std::make_optional<rmm::device_scalar<vertex_t>>(bfs_usecase.source, handle.get_stream())
        : std::nullopt;

    cugraph::bfs(handle,
                 mg_graph_view,
                 d_mg_distances.data(),
                 d_mg_predecessors.data(),
                 d_mg_source ? (*d_mg_source).data() : static_cast<vertex_t const*>(nullptr),
                 d_mg_source ? size_t{1} : size_t{0},
                 false,
                 std::numeric_limits<vertex_t>::max());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG BFS took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 4. compare SG & MG results

    if (bfs_usecase.check_correctness) {
      // 4-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        handle, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_distances =
        cugraph::test::device_gatherv(handle, d_mg_distances.data(), d_mg_distances.size());
      auto d_mg_aggregate_predecessors =
        cugraph::test::device_gatherv(handle, d_mg_predecessors.data(), d_mg_predecessors.size());

      if (handle.get_comms().get_rank() == int{0}) {
        // 4-2. unrenumbr MG results

        cugraph::unrenumber_int_vertices<vertex_t, false>(
          handle,
          d_mg_aggregate_predecessors.data(),
          d_mg_aggregate_predecessors.size(),
          d_mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});

        std::tie(std::ignore, d_mg_aggregate_distances) = cugraph::test::sort_by_key(
          handle, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_distances);
        std::tie(std::ignore, d_mg_aggregate_predecessors) = cugraph::test::sort_by_key(
          handle, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_predecessors);

        // 4-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> sg_graph(handle);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false);

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        // 4-4. run SG BFS

        rmm::device_uvector<vertex_t> d_sg_distances(sg_graph_view.number_of_vertices(),
                                                     handle.get_stream());
        rmm::device_uvector<vertex_t> d_sg_predecessors(
          sg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

        vertex_t unrenumbered_source{};
        raft::update_host(&unrenumbered_source,
                          d_mg_aggregate_renumber_map_labels.data() + bfs_usecase.source,
                          size_t{1},
                          handle.get_stream());
        handle.sync_stream();

        rmm::device_scalar<vertex_t> const d_sg_source(unrenumbered_source, handle.get_stream());
        cugraph::bfs(handle,
                     sg_graph_view,
                     d_sg_distances.data(),
                     d_sg_predecessors.data(),
                     d_sg_source.data(),
                     size_t{1},
                     false,
                     std::numeric_limits<vertex_t>::max());
        // 4-5. compare

        std::vector<edge_t> h_sg_offsets(sg_graph_view.number_of_vertices() + 1);
        std::vector<vertex_t> h_sg_indices(sg_graph_view.number_of_edges());
        raft::update_host(h_sg_offsets.data(),
                          sg_graph_view.local_edge_partition_view().offsets(),
                          sg_graph_view.number_of_vertices() + 1,
                          handle.get_stream());
        raft::update_host(h_sg_indices.data(),
                          sg_graph_view.local_edge_partition_view().indices(),
                          sg_graph_view.number_of_edges(),
                          handle.get_stream());

        std::vector<vertex_t> h_mg_aggregate_distances(mg_graph_view.number_of_vertices());
        std::vector<vertex_t> h_mg_aggregate_predecessors(mg_graph_view.number_of_vertices());

        raft::update_host(h_mg_aggregate_distances.data(),
                          d_mg_aggregate_distances.data(),
                          d_mg_aggregate_distances.size(),
                          handle.get_stream());
        raft::update_host(h_mg_aggregate_predecessors.data(),
                          d_mg_aggregate_predecessors.data(),
                          d_mg_aggregate_predecessors.size(),
                          handle.get_stream());

        std::vector<vertex_t> h_sg_distances(sg_graph_view.number_of_vertices());
        std::vector<vertex_t> h_sg_predecessors(sg_graph_view.number_of_vertices());

        raft::update_host(
          h_sg_distances.data(), d_sg_distances.data(), d_sg_distances.size(), handle.get_stream());
        raft::update_host(h_sg_predecessors.data(),
                          d_sg_predecessors.data(),
                          d_sg_predecessors.size(),
                          handle.get_stream());
        handle.sync_stream();

        ASSERT_TRUE(std::equal(h_mg_aggregate_distances.begin(),
                               h_mg_aggregate_distances.end(),
                               h_sg_distances.begin()));
        for (size_t i = 0; i < h_mg_aggregate_predecessors.size(); ++i) {
          if (h_mg_aggregate_predecessors[i] == cugraph::invalid_vertex_id<vertex_t>::value) {
            ASSERT_TRUE(h_sg_predecessors[i] == h_mg_aggregate_predecessors[i])
              << "vertex reachability does not match with the SG result.";
          } else {
            ASSERT_TRUE(h_sg_distances[h_mg_aggregate_predecessors[i]] + 1 == h_sg_distances[i])
              << "distances to this vertex != distances to the predecessor vertex + 1.";
            bool found{false};
            for (auto j = h_sg_offsets[h_mg_aggregate_predecessors[i]];
                 j < h_sg_offsets[h_mg_aggregate_predecessors[i] + 1];
                 ++j) {
              if (h_sg_indices[j] == i) {
                found = true;
                break;
              }
            }
            ASSERT_TRUE(found) << "no edge from the predecessor vertex to this vertex.";
          }
        }
      }
    }
  }
};

using Tests_MGBFS_File = Tests_MGBFS<cugraph::test::File_Usecase>;
using Tests_MGBFS_Rmat = Tests_MGBFS<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGBFS_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGBFS_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGBFS_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGBFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGBFS_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(BFS_Usecase{0}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGBFS_Rmat,
                         ::testing::Values(
                           // enable correctness checks
                           std::make_tuple(BFS_Usecase{0},
                                           cugraph::test::Rmat_Usecase(
                                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGBFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(
      BFS_Usecase{0, false},
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
