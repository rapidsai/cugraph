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
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void katz_centrality_reference(edge_t const* offsets,
                               vertex_t const* indices,
                               std::optional<weight_t const*> weights,
                               result_t const* betas,
                               result_t* katz_centralities,
                               vertex_t num_vertices,
                               result_t alpha,
                               result_t beta,  // relevant only if betas == nullptr
                               result_t epsilon,
                               size_t max_iterations,
                               bool has_initial_guess,
                               bool normalize)
{
  if (num_vertices == 0) { return; }

  if (!has_initial_guess) {
    std::fill(katz_centralities, katz_centralities + num_vertices, result_t{0.0});
  }

  std::vector<result_t> old_katz_centralities(num_vertices, result_t{0.0});
  size_t iter{0};
  while (true) {
    std::copy(katz_centralities, katz_centralities + num_vertices, old_katz_centralities.begin());
    for (vertex_t i = 0; i < num_vertices; ++i) {
      katz_centralities[i] = betas != nullptr ? betas[i] : beta;
      for (auto j = *(offsets + i); j < *(offsets + i + 1); ++j) {
        auto nbr = indices[j];
        auto w   = weights ? (*weights)[j] : result_t{1.0};
        katz_centralities[i] += alpha * old_katz_centralities[nbr] * w;
      }
    }

    result_t diff_sum{0.0};
    for (vertex_t i = 0; i < num_vertices; ++i) {
      diff_sum += std::abs(katz_centralities[i] - old_katz_centralities[i]);
    }
    if (diff_sum < epsilon) { break; }
    iter++;
    ASSERT_TRUE(iter < max_iterations);
  }

  if (normalize) {
    auto l2_norm = std::sqrt(std::inner_product(
      katz_centralities, katz_centralities + num_vertices, katz_centralities, result_t{0.0}));
    std::transform(
      katz_centralities, katz_centralities + num_vertices, katz_centralities, [l2_norm](auto& val) {
        return val / l2_norm;
      });
  }

  return;
}

struct KatzCentrality_Usecase {
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_KatzCentrality
  : public ::testing::TestWithParam<std::tuple<KatzCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_KatzCentrality() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(KatzCentrality_Usecase const& katz_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
        handle, input_usecase, katz_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    auto degrees   = graph_view.compute_in_degrees(handle);
    auto h_degrees = cugraph::test::to_host(handle, degrees);
    auto max_it    = std::max_element(h_degrees.begin(), h_degrees.end());

    result_t const alpha = result_t{1.0} / static_cast<result_t>(*max_it + 1);
    result_t constexpr beta{1.0};
    result_t constexpr epsilon{1e-6};

    rmm::device_uvector<result_t> d_katz_centralities(graph_view.number_of_vertices(),
                                                      handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Katz centrality");
    }

    cugraph::katz_centrality(handle,
                             graph_view,
                             edge_weight_view,
                             static_cast<result_t*>(nullptr),
                             d_katz_centralities.data(),
                             alpha,
                             beta,
                             epsilon,
                             std::numeric_limits<size_t>::max(),
                             false,
                             true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (katz_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, true, false> unrenumbered_graph(handle);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, false>, weight_t>>
        unrenumbered_edge_weights{std::nullopt};
      if (renumber) {
        std::tie(unrenumbered_graph, unrenumbered_edge_weights, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
            handle, input_usecase, katz_usecase.test_weighted, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;
      auto unrenumbered_edge_weight_view =
        renumber
          ? (unrenumbered_edge_weights ? std::make_optional((*unrenumbered_edge_weights).view())
                                       : std::nullopt)
          : edge_weight_view;

      auto h_offsets = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().offsets());
      auto h_indices = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().indices());
      auto h_weights = cugraph::test::to_host(
        handle,
        unrenumbered_edge_weight_view ? std::make_optional(raft::device_span<weight_t const>(
                                          (*unrenumbered_edge_weight_view).value_firsts()[0],
                                          (*unrenumbered_edge_weight_view).edge_counts()[0]))
                                      : std::nullopt);

      std::vector<result_t> h_reference_katz_centralities(
        unrenumbered_graph_view.number_of_vertices());

      katz_centrality_reference(
        h_offsets.data(),
        h_indices.data(),
        h_weights ? std::optional<weight_t const*>{(*h_weights).data()} : std::nullopt,
        static_cast<result_t*>(nullptr),
        h_reference_katz_centralities.data(),
        unrenumbered_graph_view.number_of_vertices(),
        alpha,
        beta,
        epsilon,
        std::numeric_limits<size_t>::max(),
        false,
        true);

      std::vector<result_t> h_cugraph_katz_centralities{};
      if (renumber) {
        rmm::device_uvector<result_t> d_unrenumbered_katz_centralities(size_t{0},
                                                                       handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_katz_centralities) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_katz_centralities);
        h_cugraph_katz_centralities =
          cugraph::test::to_host(handle, d_unrenumbered_katz_centralities);
      } else {
        h_cugraph_katz_centralities = cugraph::test::to_host(handle, d_katz_centralities);
      }

      handle.sync_stream();

      auto threshold_ratio = 1e-3;
      auto threshold_magnitude =
        (1.0 / static_cast<result_t>(graph_view.number_of_vertices())) *
        threshold_ratio;  // skip comparison for low Katz Centrality verties (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        return std::abs(lhs - rhs) <
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
      };

      ASSERT_TRUE(std::equal(h_reference_katz_centralities.begin(),
                             h_reference_katz_centralities.end(),
                             h_cugraph_katz_centralities.begin(),
                             nearly_equal))
        << "Katz centrality values do not match with the reference values.";
    }
  }
};

using Tests_KatzCentrality_File = Tests_KatzCentrality<cugraph::test::File_Usecase>;
using Tests_KatzCentrality_Rmat = Tests_KatzCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_KatzCentrality_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_KatzCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_KatzCentrality_Rmat, CheckInt32Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_KatzCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_KatzCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KatzCentrality_Usecase{false}, KatzCentrality_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_KatzCentrality_Rmat,
                         // enable correctness checks
                         ::testing::Combine(::testing::Values(KatzCentrality_Usecase{false},
                                                              KatzCentrality_Usecase{true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_KatzCentrality_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(KatzCentrality_Usecase{false, false}, KatzCentrality_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
