/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

template <typename result_t, typename vertex_t, typename edge_t>
std::tuple<std::vector<result_t>, std::vector<result_t>, double, size_t> hits_reference(
  edge_t const* h_offsets,
  vertex_t const* h_indices,
  vertex_t num_vertices,
  edge_t num_edges,
  size_t max_iterations,
  std::optional<result_t const*> starting_hub_values,
  bool normalized,
  double epsilon)
{
  CUGRAPH_EXPECTS(num_vertices > 1, "number of vertices expected to be non-zero");
  auto tolerance = static_cast<result_t>(num_vertices) * epsilon;

  std::vector<result_t> prev_hubs(num_vertices, result_t{1.0} / num_vertices);
  std::vector<result_t> prev_authorities(num_vertices, result_t{1.0} / num_vertices);
  std::vector<result_t> curr_hubs(num_vertices);
  std::vector<result_t> curr_authorities(num_vertices);
  double hubs_error{std::numeric_limits<double>::max()};
  size_t hubs_iterations{0};

  if (starting_hub_values) {
    std::copy((*starting_hub_values), (*starting_hub_values) + num_vertices, prev_hubs.begin());
    auto prev_hubs_norm = std::reduce(prev_hubs.begin(), prev_hubs.end());
    std::transform(prev_hubs.begin(), prev_hubs.end(), prev_hubs.begin(), [prev_hubs_norm](auto x) {
      return std::divides{}(x, prev_hubs_norm);
    });
  }

  for (; hubs_iterations < max_iterations; ++hubs_iterations) {
    std::fill(curr_hubs.begin(), curr_hubs.end(), result_t{0});
    std::fill(curr_authorities.begin(), curr_authorities.end(), result_t{0});
    for (vertex_t src = 0; src < num_vertices; ++src) {
      for (vertex_t dest_index = h_offsets[src]; dest_index < h_offsets[src + 1]; ++dest_index) {
        curr_authorities[h_indices[dest_index]] += prev_hubs[src];
      }
    }
    for (vertex_t src = 0; src < num_vertices; ++src) {
      for (vertex_t dest_index = h_offsets[src]; dest_index < h_offsets[src + 1]; ++dest_index) {
        curr_hubs[src] += curr_authorities[h_indices[dest_index]];
      }
    }
    auto curr_hubs_norm =
      std::reduce(curr_hubs.begin(), curr_hubs.end(), result_t{0}, [](auto x, auto y) {
        return std::max(x, y);
      });
    std::transform(curr_hubs.begin(), curr_hubs.end(), curr_hubs.begin(), [curr_hubs_norm](auto x) {
      return std::divides{}(x, curr_hubs_norm);
    });
    auto curr_authorities_norm = std::reduce(
      curr_authorities.begin(), curr_authorities.end(), result_t{0}, [](auto x, auto y) {
        return std::max(x, y);
      });
    std::transform(
      curr_authorities.begin(),
      curr_authorities.end(),
      curr_authorities.begin(),
      [curr_authorities_norm](auto x) { return std::divides{}(x, curr_authorities_norm); });
    hubs_error = std::transform_reduce(curr_hubs.begin(),
                                       curr_hubs.end(),
                                       prev_hubs.begin(),
                                       result_t{0},
                                       std::plus<result_t>{},
                                       [](auto x, auto y) { return std::abs(x - y); });
    if (hubs_error < tolerance) {
      break;
    } else {
      std::copy(curr_authorities.begin(), curr_authorities.end(), prev_authorities.begin());
      std::copy(curr_hubs.begin(), curr_hubs.end(), prev_hubs.begin());
    }
  }
  auto curr_hubs_norm = std::reduce(curr_hubs.begin(), curr_hubs.end());
  std::transform(curr_hubs.begin(), curr_hubs.end(), curr_hubs.begin(), [curr_hubs_norm](auto x) {
    return std::divides{}(x, curr_hubs_norm);
  });
  auto curr_authorities_norm = std::reduce(curr_authorities.begin(), curr_authorities.end());
  std::transform(
    curr_authorities.begin(),
    curr_authorities.end(),
    curr_authorities.begin(),
    [curr_authorities_norm](auto x) { return std::divides{}(x, curr_authorities_norm); });
  return std::make_tuple(
    std::move(curr_hubs), std::move(curr_authorities), hubs_error, hubs_iterations);
}

struct Hits_Usecase {
  bool check_initial_input{false};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Hits : public ::testing::TestWithParam<std::tuple<Hits_Usecase, input_usecase_t>> {
 public:
  Tests_Hits() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Hits_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber            = true;
    auto [hits_usecase, input_usecase] = param;

    // 1. initialize handle

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // 2. create SG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, true, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. run hits

    auto graph_view = graph.view();

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (hits_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    auto maximum_iterations = 400;
    weight_t epsilon        = 1e-7;
    rmm::device_uvector<weight_t> d_hubs(graph_view.local_vertex_partition_range_size(),
                                         handle.get_stream());

    rmm::device_uvector<weight_t> d_authorities(graph_view.local_vertex_partition_range_size(),
                                                handle.get_stream());

    std::optional<rmm::device_uvector<weight_t>> d_initial_random_hubs{std::nullopt};
    if (hits_usecase.check_initial_input) {
      raft::random::RngState rng_state(0);
      d_initial_random_hubs = rmm::device_uvector<weight_t>(d_hubs.size(), handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           (*d_initial_random_hubs).data(),
                                           (*d_initial_random_hubs).size(),
                                           weight_t{0.0},
                                           weight_t{1.0},
                                           rng_state);
      raft::copy(d_hubs.data(),
                 (*d_initial_random_hubs).data(),
                 (*d_initial_random_hubs).size(),
                 handle.get_stream());

      handle.sync_stream();  // before rng_state goes out-of-scope
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("HITS");
    }

    auto result = cugraph::hits(handle,
                                graph_view,
                                d_hubs.data(),
                                d_authorities.data(),
                                epsilon,
                                maximum_iterations,
                                hits_usecase.check_initial_input,
                                true,
                                hits_usecase.check_initial_input);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (hits_usecase.check_correctness) {
      std::vector<edge_t> h_offsets{};
      std::vector<vertex_t> h_indices{};
      std::tie(h_offsets, h_indices, std::ignore) =
        cugraph::test::graph_to_host_csr<vertex_t, edge_t, weight_t, true, false>(
          handle,
          graph_view,
          std::nullopt,
          d_renumber_map_labels
            ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
                                                                    (*d_renumber_map_labels).size())
            : std::nullopt);

      std::optional<std::vector<weight_t>> h_initial_random_hubs{std::nullopt};
      if (d_initial_random_hubs) {
        if (renumber) {
          rmm::device_uvector<weight_t> d_unrenumbered_initial_random_hubs(0, handle.get_stream());
          std::tie(std::ignore, d_unrenumbered_initial_random_hubs) =
            cugraph::test::sort_by_key<vertex_t, weight_t>(
              handle, *d_renumber_map_labels, *d_initial_random_hubs);
          h_initial_random_hubs =
            cugraph::test::to_host(handle, d_unrenumbered_initial_random_hubs);
        } else {
          h_initial_random_hubs = cugraph::test::to_host(handle, *d_initial_random_hubs);
        }
      }

      auto reference_result = hits_reference<weight_t>(
        h_offsets.data(),
        h_indices.data(),
        graph_view.number_of_vertices(),
        graph_view.compute_number_of_edges(handle),
        maximum_iterations,
        h_initial_random_hubs ? std::make_optional((*h_initial_random_hubs).data()) : std::nullopt,
        true,
        epsilon);

      std::vector<weight_t> h_cugraph_hits{};
      if (renumber) {
        rmm::device_uvector<weight_t> d_unrenumbered_hubs(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_hubs) =
          cugraph::test::sort_by_key<vertex_t, weight_t>(handle, *d_renumber_map_labels, d_hubs);
        h_cugraph_hits = cugraph::test::to_host(handle, d_unrenumbered_hubs);
      } else {
        h_cugraph_hits = cugraph::test::to_host(handle, d_hubs);
      }
      handle.sync_stream();
      auto threshold_ratio = 2e-3;
      auto threshold_magnitude =
        2e-6;  // skip comparison for low hits vertices (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        return std::abs(lhs - rhs) <=
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
      };

      ASSERT_TRUE(std::equal(std::get<0>(reference_result).begin(),
                             std::get<0>(reference_result).end(),
                             h_cugraph_hits.begin(),
                             nearly_equal))
        << "Hits values do not match with the reference values.";
    }
  }
};

using Tests_Hits_File = Tests_Hits<cugraph::test::File_Usecase>;
using Tests_Hits_Rmat = Tests_Hits<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Hits_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Hits_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Hits_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Hits_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Hits_Usecase{false, false, true},
                      Hits_Usecase{false, true, true},
                      Hits_Usecase{true, false, true},
                      Hits_Usecase{true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_Hits_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Hits_Usecase{false, false, true},
                      Hits_Usecase{false, true, true},
                      Hits_Usecase{true, false, true},
                      Hits_Usecase{true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_Hits_Rmat,
                         // enable correctness checks
                         ::testing::Combine(::testing::Values(Hits_Usecase{false, false, true},
                                                              Hits_Usecase{false, true, true},
                                                              Hits_Usecase{true, false, true},
                                                              Hits_Usecase{true, true, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Hits_File,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(Hits_Usecase{false, false, false},
                      Hits_Usecase{false, true, false},
                      Hits_Usecase{true, false, false},
                      Hits_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Hits_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(Hits_Usecase{false, false, false},
                      Hits_Usecase{false, true, false},
                      Hits_Usecase{true, false, false},
                      Hits_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
