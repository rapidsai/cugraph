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

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void pagerank_reference(edge_t const* offsets,
                        vertex_t const* indices,
                        std::optional<weight_t const*> weights,
                        std::optional<vertex_t const*> personalization_vertices,
                        std::optional<result_t const*> personalization_values,
                        std::optional<vertex_t> personalization_vector_size,
                        result_t* pageranks,
                        vertex_t num_vertices,
                        result_t alpha,
                        result_t epsilon,
                        size_t max_iterations,
                        bool has_initial_guess)
{
  if (num_vertices == 0) { return; }

  if (has_initial_guess) {
    // use a double type counter (instead of result_t) to accumulate as std::accumulate is
    // inaccurate in adding a large number of comparably sized numbers. In C++17 or later,
    // std::reduce may be a better option.
    auto sum =
      static_cast<result_t>(std::accumulate(pageranks, pageranks + num_vertices, double{0.0}));
    ASSERT_TRUE(sum > 0.0);
    std::for_each(pageranks, pageranks + num_vertices, [sum](auto& val) { val /= sum; });
  } else {
    std::for_each(pageranks, pageranks + num_vertices, [num_vertices](auto& val) {
      val = result_t{1.0} / static_cast<result_t>(num_vertices);
    });
  }

  result_t personalization_sum{0.0};
  if (personalization_vertices) {
    // use a double type counter (instead of result_t) to accumulate as std::accumulate is
    // inaccurate in adding a large number of comparably sized numbers. In C++17 or later,
    // std::reduce may be a better option.
    personalization_sum =
      static_cast<result_t>(std::accumulate(*personalization_values,
                                            *personalization_values + *personalization_vector_size,
                                            double{0.0}));
    ASSERT_TRUE(personalization_sum > 0.0);
  }

  std::vector<weight_t> out_weight_sums(num_vertices, result_t{0.0});
  for (vertex_t i = 0; i < num_vertices; ++i) {
    for (auto j = *(offsets + i); j < *(offsets + i + 1); ++j) {
      auto nbr = indices[j];
      auto w   = weights ? (*weights)[j] : weight_t{1.0};
      out_weight_sums[nbr] += w;
    }
  }

  std::vector<result_t> old_pageranks(num_vertices, result_t{0.0});
  size_t iter{0};
  while (true) {
    std::copy(pageranks, pageranks + num_vertices, old_pageranks.begin());
    result_t dangling_sum{0.0};
    for (vertex_t i = 0; i < num_vertices; ++i) {
      if (out_weight_sums[i] == result_t{0.0}) { dangling_sum += old_pageranks[i]; }
    }
    for (vertex_t i = 0; i < num_vertices; ++i) {
      pageranks[i] = result_t{0.0};
      for (auto j = *(offsets + i); j < *(offsets + i + 1); ++j) {
        auto nbr = indices[j];
        auto w   = weights ? (*weights)[j] : result_t{1.0};
        pageranks[i] += alpha * old_pageranks[nbr] * (w / out_weight_sums[nbr]);
      }
      if (!personalization_vertices) {
        pageranks[i] +=
          (dangling_sum * alpha + (1.0 - alpha)) / static_cast<result_t>(num_vertices);
      }
    }
    if (personalization_vertices) {
      for (vertex_t i = 0; i < *personalization_vector_size; ++i) {
        auto v = (*personalization_vertices)[i];
        pageranks[v] += (dangling_sum * alpha + (1.0 - alpha)) *
                        ((*personalization_values)[i] / personalization_sum);
      }
    }
    result_t diff_sum{0.0};
    for (vertex_t i = 0; i < num_vertices; ++i) {
      diff_sum += std::abs(pageranks[i] - old_pageranks[i]);
    }
    if (diff_sum < epsilon) { break; }
    iter++;
    ASSERT_TRUE(iter < max_iterations);
  }

  return;
}

struct PageRank_Usecase {
  double personalization_ratio{0.0};
  bool test_weighted{false};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_PageRank
  : public ::testing::TestWithParam<std::tuple<PageRank_Usecase, input_usecase_t>> {
 public:
  Tests_PageRank() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<PageRank_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber                = true;
    auto [pagerank_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
        handle, input_usecase, pagerank_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (pagerank_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    std::optional<rmm::device_uvector<vertex_t>> d_personalization_vertices{std::nullopt};
    std::optional<rmm::device_uvector<result_t>> d_personalization_values{std::nullopt};
    if (pagerank_usecase.personalization_ratio > 0.0) {
      raft::random::RngState rng_state(0);

      d_personalization_vertices = cugraph::select_random_vertices(
        handle,
        graph_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        rng_state,
        std::max(
          static_cast<size_t>(graph_view.number_of_vertices() *
                              pagerank_usecase.personalization_ratio),
          std::min(
            static_cast<size_t>(graph_view.number_of_vertices()),
            size_t{1})),  // there should be at least one vertex unless the graph is an empty graph
        false,
        false);
      d_personalization_values =
        rmm::device_uvector<result_t>((*d_personalization_vertices).size(), handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           (*d_personalization_values).data(),
                                           (*d_personalization_values).size(),
                                           result_t{0.0},
                                           result_t{1.0},
                                           rng_state);

      handle.sync_stream();  // before rng_state goes out-of-scope
    }

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("PageRank");
    }

    auto [d_pageranks, metadata] = cugraph::pagerank<vertex_t, edge_t, weight_t>(
      handle,
      graph_view,
      edge_weight_view,
      std::nullopt,
      d_personalization_vertices
        ? std::make_optional(
            std::make_tuple(raft::device_span<vertex_t const>{d_personalization_vertices->data(),
                                                              d_personalization_vertices->size()},
                            raft::device_span<result_t const>{d_personalization_values->data(),
                                                              d_personalization_values->size()}))
        : std::nullopt,
      std::optional<raft::device_span<result_t const>>{std::nullopt},
      alpha,
      epsilon,
      std::numeric_limits<size_t>::max(),
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (pagerank_usecase.check_correctness) {
      auto [h_offsets, h_indices, h_weights] =
        cugraph::test::graph_to_host_csc<vertex_t, edge_t, weight_t, true, false>(
          handle,
          graph_view,
          edge_weight_view,
          d_renumber_map_labels
            ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
                                                                    (*d_renumber_map_labels).size())
            : std::nullopt);

      std::optional<std::vector<vertex_t>> h_unrenumbered_personalization_vertices{std::nullopt};
      std::optional<std::vector<result_t>> h_unrenumbered_personalization_values{std::nullopt};
      if (d_personalization_vertices) {
        if (renumber) {
          rmm::device_uvector<vertex_t> d_unrenumbered_personalization_vertices(
            (*d_personalization_vertices).size(), handle.get_stream());
          rmm::device_uvector<result_t> d_unrenumbered_personalization_values(
            d_unrenumbered_personalization_vertices.size(), handle.get_stream());
          raft::copy_async(d_unrenumbered_personalization_vertices.data(),
                           (*d_personalization_vertices).data(),
                           (*d_personalization_vertices).size(),
                           handle.get_stream());
          raft::copy_async(d_unrenumbered_personalization_values.data(),
                           (*d_personalization_values).data(),
                           (*d_personalization_values).size(),
                           handle.get_stream());
          cugraph::unrenumber_local_int_vertices(handle,
                                                 d_unrenumbered_personalization_vertices.data(),
                                                 d_unrenumbered_personalization_vertices.size(),
                                                 (*d_renumber_map_labels).data(),
                                                 vertex_t{0},
                                                 graph_view.number_of_vertices());
          std::tie(d_unrenumbered_personalization_vertices, d_unrenumbered_personalization_values) =
            cugraph::test::sort_by_key<vertex_t, result_t>(handle,
                                                           d_unrenumbered_personalization_vertices,
                                                           d_unrenumbered_personalization_values);

          h_unrenumbered_personalization_vertices =
            cugraph::test::to_host(handle, d_unrenumbered_personalization_vertices);
          h_unrenumbered_personalization_values =
            cugraph::test::to_host(handle, d_unrenumbered_personalization_values);
        } else {
          h_unrenumbered_personalization_vertices =
            cugraph::test::to_host(handle, d_personalization_vertices);
          h_unrenumbered_personalization_values =
            cugraph::test::to_host(handle, d_personalization_values);
        }
      }

      handle.sync_stream();

      std::vector<result_t> h_reference_pageranks(graph_view.number_of_vertices());

      pagerank_reference(
        h_offsets.data(),
        h_indices.data(),
        h_weights ? std::optional<weight_t const*>{(*h_weights).data()} : std::nullopt,
        h_unrenumbered_personalization_vertices
          ? std::optional<vertex_t const*>{(*h_unrenumbered_personalization_vertices).data()}
          : std::nullopt,
        h_unrenumbered_personalization_values
          ? std::optional<result_t const*>{(*h_unrenumbered_personalization_values).data()}
          : std::nullopt,
        h_unrenumbered_personalization_vertices
          ? std::optional<vertex_t>{static_cast<vertex_t>(
              (*h_unrenumbered_personalization_vertices).size())}
          : std::nullopt,
        h_reference_pageranks.data(),
        graph_view.number_of_vertices(),
        alpha,
        epsilon,
        std::numeric_limits<size_t>::max(),
        false);

      std::vector<result_t> h_cugraph_pageranks{};
      if (renumber) {
        rmm::device_uvector<result_t> d_unrenumbered_pageranks(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_pageranks) =
          cugraph::test::sort_by_key<vertex_t, result_t>(
            handle, *d_renumber_map_labels, d_pageranks);
        h_cugraph_pageranks = cugraph::test::to_host(handle, d_unrenumbered_pageranks);
      } else {
        h_cugraph_pageranks = cugraph::test::to_host(handle, d_pageranks);
      }

      handle.sync_stream();

      auto threshold_ratio = 1e-3;
      auto threshold_magnitude =
        1e-6;  // skip comparison for low PageRank verties (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        return std::abs(lhs - rhs) <
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
      };

      ASSERT_TRUE(std::equal(h_reference_pageranks.begin(),
                             h_reference_pageranks.end(),
                             h_cugraph_pageranks.begin(),
                             nearly_equal))
        << "PageRank values do not match with the reference values.";
    }
  }
};

using Tests_PageRank_File = Tests_PageRank<cugraph::test::File_Usecase>;
using Tests_PageRank_Rmat = Tests_PageRank<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_PageRank_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_PageRank_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_PageRank_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(file_test,
                         Tests_PageRank_File,
                         ::testing::Combine(
                           // enable correctness checks
                           ::testing::Values(PageRank_Usecase{0.0, false, false},
                                             PageRank_Usecase{0.0, false, true},
                                             PageRank_Usecase{0.0, true, false},
                                             PageRank_Usecase{0.0, true, true},
                                             PageRank_Usecase{0.5, false, false},
                                             PageRank_Usecase{0.0, false, true},
                                             PageRank_Usecase{0.5, true, false},
                                             PageRank_Usecase{0.5, true, true}),
                           ::testing::Values(cugraph::test::File_Usecase("karate.csv"),
                                             cugraph::test::File_Usecase("dolphins.csv"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_PageRank_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(PageRank_Usecase{0.0, false, false},
                      PageRank_Usecase{0.0, false, true},
                      PageRank_Usecase{0.0, true, false},
                      PageRank_Usecase{0.0, true, true},
                      PageRank_Usecase{0.5, false, false},
                      PageRank_Usecase{0.0, false, true},
                      PageRank_Usecase{0.5, true, false},
                      PageRank_Usecase{0.5, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_PageRank_File,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(PageRank_Usecase{0.0, false, false, false},
                      PageRank_Usecase{0.0, false, true, false},
                      PageRank_Usecase{0.0, true, false, false},
                      PageRank_Usecase{0.0, true, true, false},
                      PageRank_Usecase{0.5, false, false, false},
                      PageRank_Usecase{0.5, false, true, false},
                      PageRank_Usecase{0.5, true, false, false},
                      PageRank_Usecase{0.5, true, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_PageRank_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(PageRank_Usecase{0.0, false, false, false},
                      PageRank_Usecase{0.0, false, true, false},
                      PageRank_Usecase{0.0, true, false, false},
                      PageRank_Usecase{0.0, true, true, false},
                      PageRank_Usecase{0.5, false, false, false},
                      PageRank_Usecase{0.5, false, true, false},
                      PageRank_Usecase{0.5, true, false, false},
                      PageRank_Usecase{0.5, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
