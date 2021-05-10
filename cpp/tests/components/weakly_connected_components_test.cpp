/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <algorithms.hpp>
#include <experimental/graph.hpp>
#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 1;

template <typename vertex_t, typename edge_t>
void weakly_connected_components_reference(edge_t const* offsets,
                                           vertex_t const* indices,
                                           vertex_t* components,
                                           vertex_t num_vertices)
{
  vertex_t depth{0};

  std::fill(components,
            components + num_vertices,
            cugraph::experimental::invalid_component_id<vertex_t>::value);

  vertex_t num_scanned{0};
  while (true) {
    auto it = std::find(components + num_scanned,
                        components + num_vertices,
                        cugraph::experimental::invalid_component_id<vertex_t>::value);
    if (it == components + num_vertices) { break; }
    num_scanned += static_cast<vertex_t>(std::distance(components + num_scanned, it));
    auto source            = num_scanned;
    *(components + source) = source;
    std::vector<vertex_t> cur_frontier_rows{source};
    std::vector<vertex_t> new_frontier_rows{};

    while (cur_frontier_rows.size() > 0) {
      for (auto const row : cur_frontier_rows) {
        auto nbr_offset_first = *(offsets + row);
        auto nbr_offset_last  = *(offsets + row + 1);
        for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
          auto nbr = *(indices + nbr_offset);
          if (*(components + nbr) == cugraph::experimental::invalid_component_id<vertex_t>::value) {
            *(components + nbr) = source;
            new_frontier_rows.push_back(nbr);
          }
        }
      }
      std::swap(cur_frontier_rows, new_frontier_rows);
      new_frontier_rows.clear();
    }
  }

  return;
}

struct WeaklyConnectedComponent_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_WeaklyConnectedComponent
  : public ::testing::TestWithParam<std::tuple<WeaklyConnectedComponent_Usecase, input_usecase_t>> {
 public:
  Tests_WeaklyConnectedComponent() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(WeaklyConnectedComponent_Usecase const& weakly_connected_component_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    using weight_t = float;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());
    std::tie(graph, d_renumber_map_labels) =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, true, renumber);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }
    auto graph_view = graph.view();
    ASSERT_TRUE(graph_view.is_symmetric())
      << "Weakly connected components works only on undirected (symmetric) graphs.";

    rmm::device_uvector<vertex_t> d_components(graph_view.get_number_of_vertices(),
                                               handle.get_stream());

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    cugraph::experimental::weakly_connected_components(handle, graph_view, d_components.data());

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "WeaklyConnectedComponent took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (weakly_connected_component_usecase.check_correctness) {
#if 1
      ASSERT_TRUE(false) << "unimplemented.";
#else
      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> unrenumbered_graph(
        handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore) =
          input_usecase.template construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, true, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets(unrenumbered_graph_view.get_number_of_vertices() + 1);
      std::vector<vertex_t> h_indices(unrenumbered_graph_view.get_number_of_edges());
      raft::update_host(h_offsets.data(),
                        unrenumbered_graph_view.offsets(),
                        unrenumbered_graph_view.get_number_of_vertices() + 1,
                        handle.get_stream());
      raft::update_host(h_indices.data(),
                        unrenumbered_graph_view.indices(),
                        unrenumbered_graph_view.get_number_of_edges(),
                        handle.get_stream());

      handle.get_stream_view().synchronize();

      auto unrenumbered_source = static_cast<vertex_t>(weakly_connected_component_usecase.source);
      if (renumber) {
        std::vector<vertex_t> h_renumber_map_labels(d_renumber_map_labels.size());
        raft::update_host(h_renumber_map_labels.data(),
                          d_renumber_map_labels.data(),
                          d_renumber_map_labels.size(),
                          handle.get_stream());

        handle.get_stream_view().synchronize();

        unrenumbered_source = h_renumber_map_labels[weakly_connected_component_usecase.source];
      }

      std::vector<vertex_t> h_reference_distances(unrenumbered_graph_view.get_number_of_vertices());
      std::vector<vertex_t> h_reference_predecessors(
        unrenumbered_graph_view.get_number_of_vertices());

      weakly_connected_component_reference(h_offsets.data(),
                                           h_indices.data(),
                                           h_reference_distances.data(),
                                           h_reference_predecessors.data(),
                                           unrenumbered_graph_view.get_number_of_vertices(),
                                           unrenumbered_source,
                                           std::numeric_limits<vertex_t>::max());

      std::vector<vertex_t> h_cugraph_distances(graph_view.get_number_of_vertices());
      std::vector<vertex_t> h_cugraph_predecessors(graph_view.get_number_of_vertices());
      if (renumber) {
        cugraph::experimental::unrenumber_local_int_vertices(handle,
                                                             d_predecessors.data(),
                                                             d_predecessors.size(),
                                                             d_renumber_map_labels.data(),
                                                             vertex_t{0},
                                                             graph_view.get_number_of_vertices(),
                                                             true);

        auto d_unrenumbered_distances = cugraph::test::sort_by_key(
          handle, d_renumber_map_labels.data(), d_distances.data(), d_renumber_map_labels.size());
        auto d_unrenumbered_predecessors = cugraph::test::sort_by_key(handle,
                                                                      d_renumber_map_labels.data(),
                                                                      d_predecessors.data(),
                                                                      d_renumber_map_labels.size());
        raft::update_host(h_cugraph_distances.data(),
                          d_unrenumbered_distances.data(),
                          d_unrenumbered_distances.size(),
                          handle.get_stream());
        raft::update_host(h_cugraph_predecessors.data(),
                          d_unrenumbered_predecessors.data(),
                          d_unrenumbered_predecessors.size(),
                          handle.get_stream());

        handle.get_stream_view().synchronize();
      } else {
        raft::update_host(
          h_cugraph_distances.data(), d_distances.data(), d_distances.size(), handle.get_stream());
        raft::update_host(h_cugraph_predecessors.data(),
                          d_predecessors.data(),
                          d_predecessors.size(),
                          handle.get_stream());

        handle.get_stream_view().synchronize();
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
#endif
    }
  }
};

using Tests_WeaklyConnectedComponent_File =
  Tests_WeaklyConnectedComponent<cugraph::test::File_Usecase>;
using Tests_WeaklyConnectedComponent_Rmat =
  Tests_WeaklyConnectedComponent<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_WeaklyConnectedComponent_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_WeaklyConnectedComponent_File,
  ::testing::Values(
    // enable correctness checks
#if 1
    std::make_tuple(WeaklyConnectedComponent_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")))
#else
    std::make_tuple(WeaklyConnectedComponent_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(WeaklyConnectedComponent_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(WeaklyConnectedComponent_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(WeaklyConnectedComponent_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(WeaklyConnectedComponent_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(WeaklyConnectedComponent_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/wiki-Talk.mtx")))
#endif
);

CUGRAPH_TEST_PROGRAM_MAIN()
