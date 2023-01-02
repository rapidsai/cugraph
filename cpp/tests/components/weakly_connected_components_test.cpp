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
#include <utilities/test_graphs.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

template <typename vertex_t, typename edge_t>
void weakly_connected_components_reference(edge_t const* offsets,
                                           vertex_t const* indices,
                                           vertex_t* components,
                                           vertex_t num_vertices)
{
  vertex_t depth{0};

  std::fill(components, components + num_vertices, cugraph::invalid_component_id<vertex_t>::value);

  vertex_t num_scanned{0};
  while (true) {
    auto it = std::find(components + num_scanned,
                        components + num_vertices,
                        cugraph::invalid_component_id<vertex_t>::value);
    if (it == components + num_vertices) { break; }
    num_scanned += static_cast<vertex_t>(std::distance(components + num_scanned, it));
    auto source            = num_scanned;
    *(components + source) = source;
    std::vector<vertex_t> cur_frontier_srcs{source};
    std::vector<vertex_t> new_frontier_srcs{};

    while (cur_frontier_srcs.size() > 0) {
      for (auto const src : cur_frontier_srcs) {
        auto nbr_offset_first = *(offsets + src);
        auto nbr_offset_last  = *(offsets + src + 1);
        for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
          auto nbr = *(indices + nbr_offset);
          if (*(components + nbr) == cugraph::invalid_component_id<vertex_t>::value) {
            *(components + nbr) = source;
            new_frontier_srcs.push_back(nbr);
          }
        }
      }
      std::swap(cur_frontier_srcs, new_frontier_srcs);
      new_frontier_srcs.clear();
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

  static void SetUpTestCase() {}
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
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    ASSERT_TRUE(graph_view.is_symmetric())
      << "Weakly connected components works only on undirected (symmetric) graphs.";

    rmm::device_uvector<vertex_t> d_components(graph_view.number_of_vertices(),
                                               handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Weakly_connected_components");
    }

    cugraph::weakly_connected_components(handle, graph_view, d_components.data());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (weakly_connected_components_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, false, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      auto h_offsets = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().offsets());
      auto h_indices = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().indices());

      std::vector<vertex_t> h_reference_components(unrenumbered_graph_view.number_of_vertices());

      weakly_connected_components_reference(h_offsets.data(),
                                            h_indices.data(),
                                            h_reference_components.data(),
                                            unrenumbered_graph_view.number_of_vertices());

      std::vector<vertex_t> h_cugraph_components{};
      if (renumber) {
        rmm::device_uvector<vertex_t> d_unrenumbered_components(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_components) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_components);
        h_cugraph_components = cugraph::test::to_host(handle, d_unrenumbered_components);
      } else {
        h_cugraph_components = cugraph::test::to_host(handle, d_components);
      }

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
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_WeaklyConnectedComponents_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_WeaklyConnectedComponents_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
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
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_WeaklyConnectedComponents_Rmat,
  ::testing::Values(
    // disable correctness checks
    std::make_tuple(WeaklyConnectedComponents_Usecase{false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
