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
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

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

#include <numeric>
#include <vector>

// self-loops and multi-edges are masked out and do not participate in degree computation, this code
// assumes that every vertex's neighbor list is sorted.
template <typename vertex_t, typename edge_t>
std::vector<edge_t> core_number_reference(edge_t const* offsets,
                                          vertex_t const* indices,
                                          vertex_t num_vertices,
                                          cugraph::k_core_degree_type_t degree_type,
                                          size_t k_first = 0,
                                          size_t k_last  = std::numeric_limits<size_t>::max())
{
  // mask out self-loops and multi_edges

  std::vector<bool> edge_valids(offsets[num_vertices], true);

  for (vertex_t i = 0; i < num_vertices; ++i) {
    for (edge_t j = offsets[i]; j < offsets[i + 1]; j++) {
      if (indices[j] == i) {
        edge_valids[j] = false;
      } else if ((j > offsets[i]) && (indices[j] == indices[j - 1])) {
        edge_valids[j] = false;
      }
    }
  }

  // construct the CSC representation if necessary

  std::vector<edge_t> csc_offsets(num_vertices + 1, edge_t{0});
  std::vector<vertex_t> csc_indices(offsets[num_vertices], vertex_t{0});
  std::vector<bool> csc_edge_valids(offsets[num_vertices], true);
  std::vector<edge_t> counters(num_vertices, edge_t{0});

  for (edge_t i = 0; i < offsets[num_vertices]; ++i) {
    ++counters[indices[i]];
  }
  std::partial_sum(counters.begin(), counters.end(), csc_offsets.begin() + 1);
  std::fill(counters.begin(), counters.end(), edge_t{0});
  for (vertex_t i = 0; i < num_vertices; ++i) {
    for (edge_t j = offsets[i]; j < offsets[i + 1]; ++j) {
      auto dst                                      = indices[j];
      csc_indices[csc_offsets[dst] + counters[dst]] = i;
      if (!edge_valids[j]) { csc_edge_valids[csc_offsets[dst] + counters[dst]] = false; }
      ++counters[dst];
    }
  }

  // initialize core_numbers to degrees

  std::vector<edge_t> degrees(num_vertices, edge_t{0});
  if ((degree_type == cugraph::k_core_degree_type_t::OUT) ||
      (degree_type == cugraph::k_core_degree_type_t::INOUT)) {
    for (vertex_t i = 0; i < num_vertices; ++i) {
      for (edge_t j = offsets[i]; j < offsets[i + 1]; ++j) {
        if (edge_valids[j]) { ++degrees[i]; }
      }
    }
  }
  if ((degree_type == cugraph::k_core_degree_type_t::IN) ||
      (degree_type == cugraph::k_core_degree_type_t::INOUT)) {
    for (vertex_t i = 0; i < num_vertices; ++i) {
      for (edge_t j = csc_offsets[i]; j < csc_offsets[i + 1]; ++j) {
        if (csc_edge_valids[j]) { ++degrees[i]; }
      }
    }
  }
  std::vector<edge_t> core_numbers = std::move(degrees);

  // sort vertices based on degrees

  std::vector<vertex_t> sorted_vertices(num_vertices);
  std::iota(sorted_vertices.begin(), sorted_vertices.end(), vertex_t{0});
  std::sort(sorted_vertices.begin(), sorted_vertices.end(), [&core_numbers](auto lhs, auto rhs) {
    return core_numbers[lhs] < core_numbers[rhs];
  });

  // update initial bin boundaries

  std::vector<vertex_t> bin_start_offsets = {0};

  edge_t cur_degree{0};
  for (vertex_t i = 0; i < num_vertices; ++i) {
    auto degree = core_numbers[sorted_vertices[i]];
    if (degree > cur_degree) {
      bin_start_offsets.insert(bin_start_offsets.end(), degree - cur_degree, i);
      cur_degree = degree;
    }
  }

  // initialize vertex positions

  std::vector<vertex_t> v_positions(num_vertices);
  for (vertex_t i = 0; i < num_vertices; ++i) {
    v_positions[sorted_vertices[i]] = i;
  }

  // update core numbers

  for (vertex_t i = 0; i < num_vertices; ++i) {
    auto v = sorted_vertices[i];
    if (core_numbers[v] >= k_last) { break; }
    for (edge_t j = offsets[v]; j < offsets[v + 1]; ++j) {
      auto nbr = indices[j];
      if (edge_valids[j] && (core_numbers[nbr] > core_numbers[v])) {
        for (edge_t k = csc_offsets[nbr]; k < csc_offsets[nbr + 1]; ++k) {
          if (csc_indices[k] == v) {
            csc_edge_valids[k] = false;
            break;
          }
        }
        if ((degree_type == cugraph::k_core_degree_type_t::IN) ||
            (degree_type == cugraph::k_core_degree_type_t::INOUT)) {
          auto nbr_pos       = v_positions[nbr];
          auto bin_start_pos = bin_start_offsets[core_numbers[nbr]];
          std::swap(v_positions[nbr], v_positions[sorted_vertices[bin_start_pos]]);
          std::swap(sorted_vertices[nbr_pos], sorted_vertices[bin_start_pos]);
          ++bin_start_offsets[core_numbers[nbr]];
          --core_numbers[nbr];
        }
      }
    }
    for (edge_t j = csc_offsets[v]; j < csc_offsets[v + 1]; ++j) {
      auto nbr = csc_indices[j];
      if (csc_edge_valids[j] && (core_numbers[nbr] > core_numbers[v])) {
        for (edge_t k = offsets[nbr]; k < offsets[nbr + 1]; ++k) {
          if (indices[k] == v) {
            edge_valids[k] = false;
            break;
          }
        }
        if ((degree_type == cugraph::k_core_degree_type_t::OUT) ||
            (degree_type == cugraph::k_core_degree_type_t::INOUT)) {
          auto nbr_pos       = v_positions[nbr];
          auto bin_start_pos = bin_start_offsets[core_numbers[nbr]];
          std::swap(v_positions[nbr], v_positions[sorted_vertices[bin_start_pos]]);
          std::swap(sorted_vertices[nbr_pos], sorted_vertices[bin_start_pos]);
          ++bin_start_offsets[core_numbers[nbr]];
          --core_numbers[nbr];
        }
      }
    }
  }

  // clip core numbers

  std::transform(
    core_numbers.begin(), core_numbers.end(), core_numbers.begin(), [k_first, k_last](auto c) {
      if (c < k_first) {
        return edge_t{0};
      } else {
        return c;
      }
    });

  return core_numbers;
}

struct CoreNumber_Usecase {
  cugraph::k_core_degree_type_t degree_type{cugraph::k_core_degree_type_t::OUT};
  size_t k_first{0};  // vertices that does not belong to k_first cores will have core numbers of 0
  size_t k_last{std::numeric_limits<size_t>::max()};  // vertices that belong (k_last + 1)-core will
                                                      // have core numbers of k_last

  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_CoreNumber
  : public ::testing::TestWithParam<std::tuple<CoreNumber_Usecase, input_usecase_t>> {
 public:
  Tests_CoreNumber() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(CoreNumber_Usecase const& core_number_usecase,
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
        handle, input_usecase, false, renumber, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
    auto graph_view = graph.view();

    ASSERT_TRUE(core_number_usecase.k_first <= core_number_usecase.k_last)
      << "Invalid pair of (k_first, k_last).";

    rmm::device_uvector<edge_t> d_core_numbers(graph_view.number_of_vertices(),
                                               handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Core number");
    }

    cugraph::core_number(handle,
                         graph_view,
                         d_core_numbers.data(),
                         core_number_usecase.degree_type,
                         core_number_usecase.k_first,
                         core_number_usecase.k_last);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (core_number_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, false, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, true, false, true, true);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      auto h_offsets = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().offsets());
      auto h_indices = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().indices());

      auto h_reference_core_numbers = core_number_reference(h_offsets.data(),
                                                            h_indices.data(),
                                                            graph_view.number_of_vertices(),
                                                            core_number_usecase.degree_type,
                                                            core_number_usecase.k_first,
                                                            core_number_usecase.k_last);

      std::vector<edge_t> h_cugraph_core_numbers{};
      if (renumber) {
        rmm::device_uvector<edge_t> d_unrenumbered_core_numbers(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_core_numbers) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_core_numbers);
        h_cugraph_core_numbers = cugraph::test::to_host(handle, d_unrenumbered_core_numbers);
      } else {
        h_cugraph_core_numbers = cugraph::test::to_host(handle, d_core_numbers);
      }

      ASSERT_TRUE(std::equal(h_reference_core_numbers.begin(),
                             h_reference_core_numbers.end(),
                             h_cugraph_core_numbers.begin()))
        << "core numbers do not match with the reference values.";
    }
  }
};

using Tests_CoreNumber_File = Tests_CoreNumber<cugraph::test::File_Usecase>;
using Tests_CoreNumber_Rmat = Tests_CoreNumber<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_CoreNumber_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_CoreNumber_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_CoreNumber_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_CoreNumber_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_CoreNumber_File,
  ::testing::Combine(
    // enable correctness checks
    testing::Values(
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::IN, size_t{0}, std::numeric_limits<size_t>::max()},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::OUT, size_t{0}, std::numeric_limits<size_t>::max()},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::INOUT, size_t{0}, std::numeric_limits<size_t>::max()},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::IN, size_t{2}, size_t{2}},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::OUT, size_t{1}, size_t{3}},
      CoreNumber_Usecase{cugraph::k_core_degree_type_t::INOUT, size_t{2}, size_t{4}}),
    testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx"),
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_CoreNumber_Rmat,
  ::testing::Combine(
    // enable correctness checks
    testing::Values(
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::IN, size_t{0}, std::numeric_limits<size_t>::max()},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::OUT, size_t{0}, std::numeric_limits<size_t>::max()},
      CoreNumber_Usecase{
        cugraph::k_core_degree_type_t::INOUT, size_t{0}, std::numeric_limits<size_t>::max()}),
    testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_CoreNumber_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    testing::Values(CoreNumber_Usecase{
      cugraph::k_core_degree_type_t::OUT, size_t{0}, std::numeric_limits<size_t>::max(), false}),
    testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
