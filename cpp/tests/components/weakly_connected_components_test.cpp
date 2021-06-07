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
#include <utilities/test_graphs.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/experimental/graph.hpp>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/experimental/graph_view.hpp>

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
static int PERF = 0;

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

struct WeaklyConnectedComponents_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_WeaklyConnectedComponent
  : public ::testing::TestWithParam<
      std::tuple<WeaklyConnectedComponents_Usecase, input_usecase_t>> {
 public:
  Tests_WeaklyConnectedComponent() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(
    WeaklyConnectedComponents_Usecase const& weakly_connected_components_usecase,
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
        handle, false, renumber);

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
      std::cout << "weakly_connected_components took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (weakly_connected_components_usecase.check_correctness) {
      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> unrenumbered_graph(
        handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore) =
          input_usecase.template construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, false, false);
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

      std::vector<vertex_t> h_reference_components(
        unrenumbered_graph_view.get_number_of_vertices());

      weakly_connected_components_reference(h_offsets.data(),
                                            h_indices.data(),
                                            h_reference_components.data(),
                                            unrenumbered_graph_view.get_number_of_vertices());

      std::vector<vertex_t> h_cugraph_components(graph_view.get_number_of_vertices());
      if (renumber) {
        rmm::device_uvector<vertex_t> d_unrenumbered_components(size_t{0},
                                                                handle.get_stream_view());
        std::tie(std::ignore, d_unrenumbered_components) = cugraph::test::sort_by_key(
          handle, d_renumber_map_labels.data(), d_components.data(), d_renumber_map_labels.size());
        raft::update_host(h_cugraph_components.data(),
                          d_unrenumbered_components.data(),
                          d_unrenumbered_components.size(),
                          handle.get_stream());
      } else {
        raft::update_host(h_cugraph_components.data(),
                          d_components.data(),
                          d_components.size(),
                          handle.get_stream());
      }
      handle.get_stream_view().synchronize();

      std::unordered_map<vertex_t, vertex_t> cuda_to_reference_map{};
      for (size_t i = 0; i < h_reference_components.size(); ++i) {
        cuda_to_reference_map.insert({h_cugraph_components[i], h_reference_components[i]});
      }
      std::transform(
        h_cugraph_components.begin(),
        h_cugraph_components.end(),
        h_cugraph_components.begin(),
        [&cuda_to_reference_map](auto cugraph_c) { return cuda_to_reference_map[cugraph_c]; });

      ASSERT_TRUE(std::equal(
        h_reference_components.begin(), h_reference_components.end(), h_cugraph_components.begin()))
        << "components do not match with the reference values.";
    }
  }
};

using Tests_WeaklyConnectedComponents_File =
  Tests_WeaklyConnectedComponent<cugraph::test::File_Usecase>;
using Tests_WeaklyConnectedComponents_Rmat =
  Tests_WeaklyConnectedComponent<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_WeaklyConnectedComponents_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeaklyConnectedComponents_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_WeaklyConnectedComponents_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(WeaklyConnectedComponents_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(WeaklyConnectedComponents_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(WeaklyConnectedComponents_Usecase{},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_WeaklyConnectedComponents_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(WeaklyConnectedComponents_Usecase{},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_test,
  Tests_WeaklyConnectedComponents_Rmat,
  ::testing::Values(
    // disable correctness checks
    std::make_tuple(WeaklyConnectedComponents_Usecase{false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
