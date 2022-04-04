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
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

// Dijkstra's algorithm
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_reference(edge_t const* offsets,
                    vertex_t const* indices,
                    weight_t const* weights,
                    weight_t* distances,
                    vertex_t* predecessors,
                    vertex_t num_vertices,
                    vertex_t source,
                    weight_t cutoff = std::numeric_limits<weight_t>::max())
{
  using queue_item_t = std::tuple<weight_t, vertex_t>;

  std::fill(distances, distances + num_vertices, std::numeric_limits<weight_t>::max());
  std::fill(predecessors, predecessors + num_vertices, cugraph::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = weight_t{0.0};
  std::priority_queue<queue_item_t, std::vector<queue_item_t>, std::greater<queue_item_t>> queue{};
  queue.push(std::make_tuple(weight_t{0.0}, source));

  while (queue.size() > 0) {
    weight_t distance{};
    vertex_t row{};
    std::tie(distance, row) = queue.top();
    queue.pop();
    if (distance > *(distances + row)) { continue; }
    auto nbr_offsets     = *(offsets + row);
    auto nbr_offset_last = *(offsets + row + 1);
    for (auto nbr_offset = nbr_offsets; nbr_offset != nbr_offset_last; ++nbr_offset) {
      auto nbr          = *(indices + nbr_offset);
      auto new_distance = distance + *(weights + nbr_offset);
      auto threshold    = std::min(*(distances + nbr), cutoff);
      if (new_distance < threshold) {
        *(distances + nbr)    = new_distance;
        *(predecessors + nbr) = row;
        queue.push(std::make_tuple(new_distance, nbr));
      }
    }
  }

  return;
}

struct SSSP_Usecase {
  size_t source{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_SSSP : public ::testing::TestWithParam<std::tuple<SSSP_Usecase, input_usecase_t>> {
 public:
  Tests_SSSP() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(SSSP_Usecase const& sssp_usecase, input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, true, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto graph_view = graph.view();

    ASSERT_TRUE(static_cast<vertex_t>(sssp_usecase.source) >= 0 &&
                static_cast<vertex_t>(sssp_usecase.source) < graph_view.number_of_vertices());

    rmm::device_uvector<weight_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    cugraph::sssp(handle,
                  graph_view,
                  d_distances.data(),
                  d_predecessors.data(),
                  static_cast<vertex_t>(sssp_usecase.source),
                  std::numeric_limits<weight_t>::max(),
                  false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "SSSP took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (sssp_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, true, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets(unrenumbered_graph_view.number_of_vertices() + 1);
      std::vector<vertex_t> h_indices(unrenumbered_graph_view.number_of_edges());
      std::vector<weight_t> h_weights(unrenumbered_graph_view.number_of_edges());
      raft::update_host(h_offsets.data(),
                        unrenumbered_graph_view.local_edge_partition_view().offsets(),
                        unrenumbered_graph_view.number_of_vertices() + 1,
                        handle.get_stream());
      raft::update_host(h_indices.data(),
                        unrenumbered_graph_view.local_edge_partition_view().indices(),
                        unrenumbered_graph_view.number_of_edges(),
                        handle.get_stream());
      raft::update_host(h_weights.data(),
                        *(unrenumbered_graph_view.local_edge_partition_view().weights()),
                        unrenumbered_graph_view.number_of_edges(),
                        handle.get_stream());

      handle.sync_stream();

      auto unrenumbered_source = static_cast<vertex_t>(sssp_usecase.source);
      if (renumber) {
        std::vector<vertex_t> h_renumber_map_labels((*d_renumber_map_labels).size());
        raft::update_host(h_renumber_map_labels.data(),
                          (*d_renumber_map_labels).data(),
                          (*d_renumber_map_labels).size(),
                          handle.get_stream());

        handle.sync_stream();

        unrenumbered_source = h_renumber_map_labels[sssp_usecase.source];
      }

      std::vector<weight_t> h_reference_distances(unrenumbered_graph_view.number_of_vertices());
      std::vector<vertex_t> h_reference_predecessors(unrenumbered_graph_view.number_of_vertices());

      sssp_reference(h_offsets.data(),
                     h_indices.data(),
                     h_weights.data(),
                     h_reference_distances.data(),
                     h_reference_predecessors.data(),
                     unrenumbered_graph_view.number_of_vertices(),
                     unrenumbered_source,
                     std::numeric_limits<weight_t>::max());

      std::vector<weight_t> h_cugraph_distances(graph_view.number_of_vertices());
      std::vector<vertex_t> h_cugraph_predecessors(graph_view.number_of_vertices());
      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_predecessors.data(),
                                               d_predecessors.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices(),
                                               true);

        rmm::device_uvector<weight_t> d_unrenumbered_distances(size_t{0}, handle.get_stream());
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

      auto max_weight_element = std::max_element(h_weights.begin(), h_weights.end());
      auto epsilon            = *max_weight_element * weight_t{1e-6};
      auto nearly_equal = [epsilon](auto lhs, auto rhs) { return std::fabs(lhs - rhs) < epsilon; };

      ASSERT_TRUE(std::equal(h_reference_distances.begin(),
                             h_reference_distances.end(),
                             h_cugraph_distances.begin(),
                             nearly_equal))
        << "distances do not match with the reference values.";

      for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
        auto i = std::distance(h_cugraph_predecessors.begin(), it);
        if (*it == cugraph::invalid_vertex_id<vertex_t>::value) {
          ASSERT_TRUE(h_reference_predecessors[i] == *it)
            << "vertex reachability do not match with the reference.";
        } else {
          auto pred_distance = h_reference_distances[*it];
          bool found{false};
          for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
            if (h_indices[j] == i) {
              if (nearly_equal(pred_distance + h_weights[j], h_reference_distances[i])) {
                found = true;
                break;
              }
            }
          }
          ASSERT_TRUE(found)
            << "no edge from the predecessor vertex to this vertex with the matching weight.";
        }
      }
    }
  }
};

using Tests_SSSP_File = Tests_SSSP<cugraph::test::File_Usecase>;
using Tests_SSSP_Rmat = Tests_SSSP<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_SSSP_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_SSSP_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_SSSP_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_SSSP_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SSSP_File,
  // enable correctness checks
  ::testing::Values(
    std::make_tuple(SSSP_Usecase{0}, cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(SSSP_Usecase{0}, cugraph::test::File_Usecase("test/datasets/dblp.mtx")),
    std::make_tuple(SSSP_Usecase{1000},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SSSP_Rmat,
  // enable correctness checks
  ::testing::Values(std::make_tuple(
    SSSP_Usecase{0}, cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_SSSP_Rmat,
  // disable correctness checks for large graphs
  ::testing::Values(
    std::make_tuple(SSSP_Usecase{0, false},
                    cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
