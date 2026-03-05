/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <stack>
#include <unordered_map>
#include <vector>

// Tarjan's strongly connected components algorithm.
// (https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
template <typename vertex_t, typename edge_t>
void strongly_connected_components_reference(edge_t const* offsets,
                                             vertex_t const* indices,
                                             vertex_t* components,
                                             vertex_t num_vertices)
{
  using index_t                   = size_t;
  constexpr index_t invalid_index = std::numeric_limits<index_t>::max();

  std::vector<index_t> index(num_vertices, invalid_index);
  std::vector<index_t> lowlink(num_vertices, invalid_index);
  std::vector<bool> on_stack(num_vertices, false);
  std::stack<vertex_t> S{};
  index_t current_index{0};
  vertex_t next_component_id{0};

  std::fill(components, components + num_vertices, cugraph::invalid_component_id<vertex_t>::value);

  auto strongconnect = [&](vertex_t v, auto&& strongconnect_ref) -> void {
    index[v]   = current_index;
    lowlink[v] = current_index;
    ++current_index;
    S.push(v);
    on_stack[v] = true;

    // Consider successors of v (outgoing edges)
    edge_t nbr_begin = offsets[v];
    edge_t nbr_end   = offsets[v + 1];
    for (edge_t e = nbr_begin; e != nbr_end; ++e) {
      vertex_t w = indices[e];
      if (index[w] == invalid_index) {
        strongconnect_ref(w, strongconnect_ref);
        lowlink[v] = std::min(lowlink[v], lowlink[w]);
      } else if (on_stack[w]) {
        lowlink[v] = std::min(lowlink[v], index[w]);
      }
    }

    // If v is a root node, pop the stack and assign component id
    if (lowlink[v] == index[v]) {
      vertex_t w;
      do {
        w = S.top();
        S.pop();
        on_stack[w]   = false;
        components[w] = next_component_id;
      } while (w != v);
      ++next_component_id;
    }
  };

  for (vertex_t v = 0; v < num_vertices; ++v) {
    if (index[v] == invalid_index) { strongconnect(v, strongconnect); }
  }
}

struct StronglyConnectedComponents_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_StronglyConnectedComponent
  : public ::testing::TestWithParam<
      std::tuple<StronglyConnectedComponents_Usecase, input_usecase_t>> {
 public:
  Tests_StronglyConnectedComponent() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(
    StronglyConnectedComponents_Usecase const& strongly_connected_components_usecase,
    input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    using weight_t = float;  // dummy

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
    ASSERT_FALSE(graph_view.is_symmetric())
      << "Strongly connected components works only on directed (asymmetric) graphs.";

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (strongly_connected_components_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask(edge_mask->view());
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Strongly_connected_components");
    }

    auto d_components = cugraph::strongly_connected_components(handle, graph_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (strongly_connected_components_usecase.check_correctness) {
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

      strongly_connected_components_reference(h_offsets.data(),
                                              h_indices.data(),
                                              h_reference_components.data(),
                                              unrenumbered_graph_view.number_of_vertices());

      std::vector<vertex_t> h_cugraph_components{};
      if (renumber) {
        rmm::device_uvector<vertex_t> d_unrenumbered_components(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_components) =
          cugraph::test::sort_by_key<vertex_t, vertex_t>(
            handle, *d_renumber_map_labels, d_components);
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

using Tests_StronglyConnectedComponents_File =
  Tests_StronglyConnectedComponent<cugraph::test::File_Usecase>;
using Tests_StronglyConnectedComponents_Rmat =
  Tests_StronglyConnectedComponent<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_StronglyConnectedComponents_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_StronglyConnectedComponents_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_StronglyConnectedComponents_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_StronglyConnectedComponents_File,
  ::testing::Values(
    std::make_tuple(StronglyConnectedComponents_Usecase{false},
                    cugraph::test::File_Usecase("test/datasets/karate-asymmetric.csv")),
    std::make_tuple(StronglyConnectedComponents_Usecase{true},
                    cugraph::test::File_Usecase("test/datasets/karate-asymmetric.csv")),
    std::make_tuple(StronglyConnectedComponents_Usecase{false},
                    cugraph::test::File_Usecase("test/datasets/cage6.mtx")),
    std::make_tuple(StronglyConnectedComponents_Usecase{true},
                    cugraph::test::File_Usecase("test/datasets/cage6.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_StronglyConnectedComponents_Rmat,
  ::testing::Values(
    std::make_tuple(StronglyConnectedComponents_Usecase{false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(StronglyConnectedComponents_Usecase{true},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_StronglyConnectedComponents_Rmat,
  ::testing::Values(
    std::make_tuple(StronglyConnectedComponents_Usecase{false, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(StronglyConnectedComponents_Usecase{true, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
