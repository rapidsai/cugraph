/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

template <typename vertex_t, typename edge_t>
void bfs_reference(edge_t const* offsets,
                   vertex_t const* indices,
                   vertex_t* distances,
                   vertex_t* predecessors,
                   vertex_t num_vertices,
                   vertex_t source,
                   vertex_t depth_limit = std::numeric_limits<vertex_t>::max())
{
  vertex_t depth{0};

  std::fill(distances, distances + num_vertices, std::numeric_limits<vertex_t>::max());
  std::fill(predecessors, predecessors + num_vertices, cugraph::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = depth;
  std::vector<vertex_t> cur_frontier_rows{source};
  std::vector<vertex_t> new_frontier_rows{};

  while (cur_frontier_rows.size() > 0) {
    for (auto const row : cur_frontier_rows) {
      auto nbr_offset_first = *(offsets + row);
      auto nbr_offset_last  = *(offsets + row + 1);
      for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
        auto nbr = *(indices + nbr_offset);
        if (*(distances + nbr) == std::numeric_limits<vertex_t>::max()) {
          *(distances + nbr)    = depth + 1;
          *(predecessors + nbr) = row;
          new_frontier_rows.push_back(nbr);
        }
      }
    }
    std::swap(cur_frontier_rows, new_frontier_rows);
    new_frontier_rows.clear();
    ++depth;
    if (depth >= depth_limit) { break; }
  }

  return;
}

struct BFS_Usecase {
  size_t source{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_BFS : public ::testing::TestWithParam<std::tuple<BFS_Usecase, input_usecase_t>> {
 public:
  Tests_BFS() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(BFS_Usecase const& bfs_usecase, input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    using weight_t = float;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }
    auto graph_view = graph.view();

    ASSERT_TRUE(static_cast<vertex_t>(bfs_usecase.source) >= 0 &&
                static_cast<vertex_t>(bfs_usecase.source) < graph_view.number_of_vertices())
      << "Invalid starting source.";

    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    rmm::device_scalar<vertex_t> const d_source(bfs_usecase.source, handle.get_stream());

    cugraph::bfs(handle,
                 graph_view,
                 d_distances.data(),
                 d_predecessors.data(),
                 d_source.data(),
                 size_t{1},
                 false,
                 std::numeric_limits<vertex_t>::max());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "BFS took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (bfs_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets(unrenumbered_graph_view.number_of_vertices() + 1);
      std::vector<vertex_t> h_indices(unrenumbered_graph_view.number_of_edges());
      raft::update_host(h_offsets.data(),
                        unrenumbered_graph_view.local_edge_partition_view().offsets(),
                        unrenumbered_graph_view.number_of_vertices() + 1,
                        handle.get_stream());
      raft::update_host(h_indices.data(),
                        unrenumbered_graph_view.local_edge_partition_view().indices(),
                        unrenumbered_graph_view.number_of_edges(),
                        handle.get_stream());

      handle.sync_stream();

      auto unrenumbered_source = static_cast<vertex_t>(bfs_usecase.source);
      if (renumber) {
        std::vector<vertex_t> h_renumber_map_labels((*d_renumber_map_labels).size());
        raft::update_host(h_renumber_map_labels.data(),
                          (*d_renumber_map_labels).data(),
                          (*d_renumber_map_labels).size(),
                          handle.get_stream());

        handle.sync_stream();

        unrenumbered_source = h_renumber_map_labels[bfs_usecase.source];
      }

      std::vector<vertex_t> h_reference_distances(unrenumbered_graph_view.number_of_vertices());
      std::vector<vertex_t> h_reference_predecessors(unrenumbered_graph_view.number_of_vertices());

      bfs_reference(h_offsets.data(),
                    h_indices.data(),
                    h_reference_distances.data(),
                    h_reference_predecessors.data(),
                    unrenumbered_graph_view.number_of_vertices(),
                    unrenumbered_source,
                    std::numeric_limits<vertex_t>::max());

      std::vector<vertex_t> h_cugraph_distances(graph_view.number_of_vertices());
      std::vector<vertex_t> h_cugraph_predecessors(graph_view.number_of_vertices());
      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_predecessors.data(),
                                               d_predecessors.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices(),
                                               true);

        rmm::device_uvector<vertex_t> d_unrenumbered_distances(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_distances) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_distances);
        rmm::device_uvector<vertex_t> d_unrenumbered_predecessors(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_predecessors) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_predecessors);
        raft::update_host(h_cugraph_distances.data(),
                          d_unrenumbered_distances.data(),
                          d_unrenumbered_distances.size(),
                          handle.get_stream());
        raft::update_host(h_cugraph_predecessors.data(),
                          d_unrenumbered_predecessors.data(),
                          d_unrenumbered_predecessors.size(),
                          handle.get_stream());

        handle.sync_stream();
      } else {
        raft::update_host(
          h_cugraph_distances.data(), d_distances.data(), d_distances.size(), handle.get_stream());
        raft::update_host(h_cugraph_predecessors.data(),
                          d_predecessors.data(),
                          d_predecessors.size(),
                          handle.get_stream());

        handle.sync_stream();
      }

      ASSERT_TRUE(std::equal(
        h_reference_distances.begin(), h_reference_distances.end(), h_cugraph_distances.begin()))
        << "distances do not match with the reference values.";

      for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
        auto i = std::distance(h_cugraph_predecessors.begin(), it);
        if (*it == cugraph::invalid_vertex_id<vertex_t>::value) {
          ASSERT_TRUE(h_reference_predecessors[i] == *it)
            << "vertex reachability does not match with the reference.";
        } else {
          ASSERT_TRUE(h_reference_distances[*it] + 1 == h_reference_distances[i])
            << "distance to this vertex != distance to the predecessor vertex + 1.";
          bool found{false};
          for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
            if (h_indices[j] == i) {
              found = true;
              break;
            }
          }
          ASSERT_TRUE(found) << "no edge from the predecessor vertex to this vertex.";
        }
      }
    }
  }
};

using Tests_BFS_File = Tests_BFS<cugraph::test::File_Usecase>;
using Tests_BFS_Rmat = Tests_BFS<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_BFS_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_BFS_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_BFS_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_BFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_BFS_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(BFS_Usecase{0}, cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(BFS_Usecase{0}, cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(BFS_Usecase{0}, cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(BFS_Usecase{100}, cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(BFS_Usecase{1000}, cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(BFS_Usecase{1000},
                    cugraph::test::File_Usecase("test/datasets/wiki-Talk.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_BFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(BFS_Usecase{0},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_BFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_pair(BFS_Usecase{0, false},
                   cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
