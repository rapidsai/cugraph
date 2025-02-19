/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

template <typename vertex_t, typename edge_t>
std::vector<vertex_t> find_forest_reference(raft::host_span<edge_t const> offsets,
                                            raft::host_span<vertex_t const> indices)
{
  auto num_vertices = static_cast<vertex_t>(offsets.size() - 1);
  std::vector<vertex_t> parents(num_vertices, cugraph::invalid_vertex_id_v<vertex_t>);

  std::vector<vertex_t> degrees(num_vertices, 0);
  std::adjacent_difference(offsets.begin() + 1, offsets.end(), degrees.begin());

  while (true) {
    bool changed{false};
    for (vertex_t v = 0; v < num_vertices; ++v) {
      if (degrees[v] == 0) {
        parents[v] = v;
        changed    = true;
      } else if (degrees[v] == 1) {
        for (edge_t i = offsets[v]; i < offsets[v + 1]; ++i) {
          auto nbr = indices[i];
          if (parents[nbr] == cugraph::invalid_vertex_id_v<vertex_t>) {
            parents[v] = nbr;
            degrees[nbr]--;
            changed = true;
            break;
          }
        }
      }
    }
    if (!changed) { break; }
  }

  return parents;
}

struct FindForest_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_FindForest
  : public ::testing::TestWithParam<std::tuple<FindForest_Usecase, input_usecase_t>> {
 public:
  Tests_FindForest() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(FindForest_Usecase const& find_forest_usecase,
                        input_usecase_t const& input_usecase)
  {
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = false;
    bool constexpr drop_self_loops  = true;
    bool constexpr drop_multi_edges = true;

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
        handle, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
    auto graph_view = graph.view();

    std::optional<cugraph::edge_property_t<decltype(graph_view), bool>> edge_mask{std::nullopt};
    if (find_forest_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("find_forest");
    }

    auto d_parents = cugraph::find_forest(handle, graph_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (find_forest_usecase.check_correctness) {
      std::vector<edge_t> h_offsets{};
      std::vector<vertex_t> h_indices{};
      std::tie(h_offsets, h_indices, std::ignore) =
        cugraph::test::graph_to_host_csr<vertex_t, edge_t, weight_t, false, false>(
          handle,
          graph_view,
          std::nullopt,
          d_renumber_map_labels
            ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
                                                                    (*d_renumber_map_labels).size())
            : std::nullopt);

      auto h_reference_parents =
        find_forest_reference(raft::host_span(h_offsets.data(), h_offsets.size()),
                              raft::host_span(h_indices.data(), h_indices.size()));

      std::vector<vertex_t> h_cugraph_parents{};
      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_parents.data(),
                                               d_parents.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices());

        rmm::device_uvector<vertex_t> d_unrenumbered_parents(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_parents) =
          cugraph::test::sort_by_key<vertex_t, vertex_t>(handle, *d_renumber_map_labels, d_parents);
        h_cugraph_parents = cugraph::test::to_host(handle, d_unrenumbered_parents);
      } else {
        h_cugraph_parents = cugraph::test::to_host(handle, d_parents);
      }

      ASSERT_TRUE(std::equal(
        h_reference_parents.begin(), h_reference_parents.end(), h_cugraph_parents.begin()))
        << "parents do not match with the reference values.";
    }
  }
};

using Tests_FindForest_File = Tests_FindForest<cugraph::test::File_Usecase>;
using Tests_FindForest_Rmat = Tests_FindForest<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_FindForest_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_FindForest_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_FindForest_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_FindForest_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(FindForest_Usecase{false},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(FindForest_Usecase{true},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(FindForest_Usecase{false},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(FindForest_Usecase{true},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(FindForest_Usecase{false},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(FindForest_Usecase{true},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_FindForest_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(
      FindForest_Usecase{false},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false)),
    std::make_tuple(
      FindForest_Usecase{true},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_FindForest_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(
      FindForest_Usecase{false, false},
      cugraph::test::Rmat_Usecase(
        20, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false /* scramble vertex IDs */)),
    std::make_tuple(
      FindForest_Usecase{true, false},
      cugraph::test::Rmat_Usecase(
        20, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false /* scramble vertex IDs */))));

CUGRAPH_TEST_PROGRAM_MAIN()
