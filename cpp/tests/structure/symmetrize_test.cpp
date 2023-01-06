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
#include <utilities/test_utilities.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

// partition the edges to lower-triangular, diagonal, upper-triangular edges; flip sources and
// destinations of the upper triangular edges, and sort within each partition
template <typename EdgeIterator>
std::tuple<EdgeIterator, EdgeIterator> partition_and_sort_edges(
  EdgeIterator edge_first, EdgeIterator edge_last, bool flip_upper_triangular_before_sort)
{
  // partition

  auto lower_triangular_last =
    std::partition(edge_first, edge_last, [](auto e) { return std::get<0>(e) > std::get<1>(e); });
  auto diagonal_last = std::partition(
    lower_triangular_last, edge_last, [](auto e) { return std::get<0>(e) == std::get<1>(e); });

  // flip

  if (flip_upper_triangular_before_sort) {
    std::transform(diagonal_last, edge_last, diagonal_last, [](auto e) {
      auto ret         = e;
      std::get<0>(ret) = std::get<1>(e);
      std::get<1>(ret) = std::get<0>(e);
      return ret;
    });
  }

  // sort

  std::sort(edge_first, lower_triangular_last);
  std::sort(lower_triangular_last, diagonal_last);
  if (flip_upper_triangular_before_sort) {
    std::sort(diagonal_last, edge_last);
  } else {
    std::sort(diagonal_last, edge_last, [](auto lhs, auto rhs) {
      std::swap(std::get<0>(lhs), std::get<1>(lhs));
      std::swap(std::get<0>(rhs), std::get<1>(rhs));
      return lhs < rhs;
    });
  }

  return std::make_tuple(lower_triangular_last, diagonal_last);
}

template <typename EdgeIterator, typename IncludeIterator>
struct symmetrize_op_t {
  bool reciprocal{false};

  void operator()(
    EdgeIterator lower_first,
    size_t lower_run_length,
    EdgeIterator upper_first,
    size_t upper_run_length,
    IncludeIterator include_first /* size = lower_run_length + upper_run_Length */) const
  {
    using weight_t =
      typename std::tuple_element<2, typename std::iterator_traits<EdgeIterator>::value_type>::type;

    auto min_run_length = std::min(lower_run_length, upper_run_length);
    auto max_run_length = std::max(lower_run_length, upper_run_length);
    for (size_t i = 0; i < max_run_length; ++i) {
      if (i < min_run_length) {
        std::get<2>(*(lower_first + i)) =
          (std::get<2>(*(lower_first + i)) + std::get<2>(*(upper_first + i))) /
          weight_t{2.0};  // average
        *(include_first + i)                    = true;
        *(include_first + lower_run_length + i) = false;
      } else {
        if (lower_run_length > upper_run_length) {
          *(include_first + i) = !reciprocal;
        } else {
          *(include_first + lower_run_length + i) = !reciprocal;
        }
      }
    }
  }
};

template <typename EdgeIterator, typename IncludeIterator>
struct update_edge_weights_and_flags_t {
  EdgeIterator edge_first{};
  IncludeIterator include_first{nullptr};
  size_t num_edges{0};
  symmetrize_op_t<EdgeIterator, IncludeIterator> op{};

  void operator()(size_t i) const
  {
    bool first_in_run{};
    if (i == 0) {
      first_in_run = true;
    } else {
      auto cur       = *(edge_first + i);
      auto prev      = *(edge_first + (i - 1));
      auto cur_pair  = std::get<0>(cur) > std::get<1>(cur)
                         ? std::make_tuple(std::get<0>(cur), std::get<1>(cur))
                         : std::make_tuple(std::get<1>(cur), std::get<0>(cur));
      auto prev_pair = std::get<0>(prev) > std::get<1>(prev)
                         ? std::make_tuple(std::get<0>(prev), std::get<1>(prev))
                         : std::make_tuple(std::get<1>(prev), std::get<0>(prev));
      first_in_run   = cur_pair != prev_pair;
    }

    if (first_in_run) {
      auto first = *(edge_first + i);
      size_t lower_run_length{0};
      size_t upper_run_length{0};
      auto pair_first = std::get<0>(first) > std::get<1>(first)
                          ? std::make_tuple(std::get<0>(first), std::get<1>(first))
                          : std::make_tuple(std::get<1>(first), std::get<0>(first));
      while (i + lower_run_length < num_edges) {
        auto cur = *(edge_first + i + lower_run_length);
        if ((std::get<0>(cur) > std::get<1>(cur)) &&
            (std::make_tuple(std::get<0>(cur), std::get<1>(cur)) == pair_first)) {
          ++lower_run_length;
        } else {
          break;
        }
      }
      while (i + lower_run_length + upper_run_length < num_edges) {
        auto cur = *(edge_first + i + lower_run_length + upper_run_length);
        if ((std::get<0>(cur) < std::get<1>(cur)) &&
            (std::make_tuple(std::get<1>(cur), std::get<0>(cur)) == pair_first)) {
          ++upper_run_length;
        } else {
          break;
        }
      }

      op(edge_first + i,
         lower_run_length,
         edge_first + i + lower_run_length,
         upper_run_length,
         include_first + i);
    }
  }
};

typedef struct Symmetrize_Usecase_t {
  bool reciprocal{false};
  bool test_weighted{false};
  bool check_correctness{true};
} Symmetrize_Usecase;

template <typename input_usecase_t>
class Tests_Symmetrize
  : public ::testing::TestWithParam<std::tuple<Symmetrize_Usecase, input_usecase_t>> {
 public:
  Tests_Symmetrize() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(Symmetrize_Usecase const& symmetrize_usecase,
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
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, symmetrize_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    rmm::device_uvector<vertex_t> d_org_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_org_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> d_org_weights{std::nullopt};
    if (symmetrize_usecase.check_correctness) {
      std::tie(d_org_srcs, d_org_dsts, d_org_weights) = cugraph::decompress_to_edgelist(
        handle,
        graph.view(),
        edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt,
        d_renumber_map_labels ? std::make_optional<raft::device_span<vertex_t const>>(
                                  (*d_renumber_map_labels).data(), (*d_renumber_map_labels).size())
                              : std::nullopt);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Symmetrize");
    }

    std::tie(graph, edge_weights, d_renumber_map_labels) =
      cugraph::symmetrize_graph(handle,
                                std::move(graph),
                                std::move(edge_weights),
                                std::move(d_renumber_map_labels),
                                symmetrize_usecase.reciprocal);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (symmetrize_usecase.check_correctness) {
      auto [d_symm_srcs, d_symm_dsts, d_symm_weights] = cugraph::decompress_to_edgelist(
        handle,
        graph.view(),
        edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt,
        d_renumber_map_labels ? std::make_optional<raft::device_span<vertex_t const>>(
                                  (*d_renumber_map_labels).data(), (*d_renumber_map_labels).size())
                              : std::nullopt);

      auto h_org_srcs    = cugraph::test::to_host(handle, d_org_srcs);
      auto h_org_dsts    = cugraph::test::to_host(handle, d_org_dsts);
      auto h_org_weights = cugraph::test::to_host(handle, d_org_weights);

      auto h_symm_srcs    = cugraph::test::to_host(handle, d_symm_srcs);
      auto h_symm_dsts    = cugraph::test::to_host(handle, d_symm_dsts);
      auto h_symm_weights = cugraph::test::to_host(handle, d_symm_weights);

      if (symmetrize_usecase.test_weighted) {
        std::vector<std::tuple<vertex_t, vertex_t, weight_t>> org_edges(h_org_srcs.size());
        for (size_t i = 0; i < org_edges.size(); ++i) {
          org_edges[i] = std::make_tuple(h_org_srcs[i], h_org_dsts[i], (*h_org_weights)[i]);
        }

        auto [org_lower_triangular_last, org_diagonal_last] =
          partition_and_sort_edges(org_edges.begin(), org_edges.end(), false);

        std::vector<std::tuple<vertex_t, vertex_t, weight_t>> tmp_edges(
          std::distance(org_edges.begin(), org_lower_triangular_last) +
          std::distance(org_diagonal_last, org_edges.end()));
        std::merge(
          org_edges.begin(),
          org_lower_triangular_last,
          org_diagonal_last,
          org_edges.end(),
          tmp_edges.begin(),
          [](auto lhs, auto rhs) {
            auto lhs_in_lower = std::get<0>(lhs) > std::get<1>(lhs);
            auto rhs_in_lower = std::get<0>(rhs) > std::get<1>(rhs);
            return std::make_tuple(
                     lhs_in_lower ? std::get<0>(lhs) : std::get<1>(lhs),
                     lhs_in_lower ? std::get<1>(lhs) : std::get<0>(lhs),
                     !lhs_in_lower,  // lower triangular edges come before upper triangular edges
                     std::get<2>(lhs)) <
                   std::make_tuple(
                     rhs_in_lower ? std::get<0>(rhs) : std::get<1>(rhs),
                     rhs_in_lower ? std::get<1>(rhs) : std::get<0>(rhs),
                     !rhs_in_lower,  // lower triangular edges come before upper triangular edges
                     std::get<2>(rhs));
          });

        // Note that for multi-graphs with weighted edges, this correctness test assumes that the
        // reference code and device code are using the same math to symmetrize N lower_triangular
        // edges and M upper triangular edges having the same source and destination vertices
        // (specifically, 1) the lower triangular and upper triangular edges with the same source &
        // destination vertices are sorted by edge weights, respectively; 2) for the first min(N, M)
        // lower triangular and upper triangular edge pairs are reduced to the new min(N, M) edges
        // by averaging the edge weights; 3) and the remaining max(N, M) - min(N, M) lower
        // triangular (if N > M) or upper triangular (if N < M) edges are either discarded (if
        // reciprocal == true) or included as is (if reciprocal == false)
        std::vector<bool> includes(tmp_edges.size());
        symmetrize_op_t<decltype(tmp_edges.begin()), decltype(includes.begin())> symm_op{
          symmetrize_usecase.reciprocal};
        update_edge_weights_and_flags_t<decltype(tmp_edges.begin()), decltype(includes.begin())>
          edge_op{tmp_edges.begin(), includes.begin(), tmp_edges.size(), symm_op};
        for (size_t i = 0; i < tmp_edges.size(); ++i) {
          edge_op(i);
        }
        for (size_t i = 0; i < tmp_edges.size(); ++i) {
          if (includes[i] == true) {
            if (std::get<0>(tmp_edges[i]) <
                std::get<1>(
                  tmp_edges[i])) {  // flip the edges selected from the upper triangular part
              std::swap(std::get<0>(tmp_edges[i]), std::get<1>(tmp_edges[i]));
            }
          } else {
            std::get<0>(tmp_edges[i]) = cugraph::invalid_vertex_id<vertex_t>::value;
          }
        }
        tmp_edges.resize(std::distance(
          tmp_edges.begin(), std::remove_if(tmp_edges.begin(), tmp_edges.end(), [](auto e) {
            return std::get<0>(e) == cugraph::invalid_vertex_id<vertex_t>::value;
          })));
        std::sort(tmp_edges.begin(), tmp_edges.end());
        tmp_edges.insert(tmp_edges.end(), org_lower_triangular_last, org_diagonal_last);
        org_edges = std::move(tmp_edges);

        std::vector<std::tuple<vertex_t, vertex_t, weight_t>> symm_edges(h_symm_srcs.size());
        for (size_t i = 0; i < symm_edges.size(); ++i) {
          symm_edges[i] = std::make_tuple(h_symm_srcs[i], h_symm_dsts[i], (*h_symm_weights)[i]);
        }

        auto [symm_lower_triangular_last, symm_diagonal_last] =
          partition_and_sort_edges(symm_edges.begin(), symm_edges.end(), true);

        ASSERT_TRUE(std::equal(
          symm_edges.begin(), symm_lower_triangular_last, symm_diagonal_last));  // check symmetric

        symm_edges.resize(std::distance(symm_edges.begin(), symm_diagonal_last));

        ASSERT_TRUE(std::equal(org_edges.begin(), org_edges.end(), symm_edges.begin()));
      } else {
        std::vector<std::tuple<vertex_t, vertex_t>> org_edges(h_org_srcs.size());
        for (size_t i = 0; i < org_edges.size(); ++i) {
          org_edges[i] = std::make_tuple(h_org_srcs[i], h_org_dsts[i]);
        }

        auto [org_lower_triangular_last, org_diagonal_last] =
          partition_and_sort_edges(org_edges.begin(), org_edges.end(), true);

        std::vector<std::tuple<vertex_t, vertex_t>> tmp_edges(
          symmetrize_usecase.reciprocal
            ? std::min(std::distance(org_edges.begin(), org_lower_triangular_last),
                       std::distance(org_diagonal_last, org_edges.end()))
            : std::distance(org_edges.begin(), org_lower_triangular_last) +
                std::distance(org_diagonal_last, org_edges.end()));
        if (symmetrize_usecase.reciprocal) {
          tmp_edges.resize(std::distance(tmp_edges.begin(),
                                         std::set_intersection(org_edges.begin(),
                                                               org_lower_triangular_last,
                                                               org_diagonal_last,
                                                               org_edges.end(),
                                                               tmp_edges.begin())));
        } else {
          tmp_edges.resize(std::distance(tmp_edges.begin(),
                                         std::set_union(org_edges.begin(),
                                                        org_lower_triangular_last,
                                                        org_diagonal_last,
                                                        org_edges.end(),
                                                        tmp_edges.begin())));
        }
        tmp_edges.insert(tmp_edges.end(), org_lower_triangular_last, org_diagonal_last);
        org_edges = std::move(tmp_edges);

        std::vector<std::tuple<vertex_t, vertex_t>> symm_edges(h_symm_srcs.size());
        for (size_t i = 0; i < symm_edges.size(); ++i) {
          symm_edges[i] = std::make_tuple(h_symm_srcs[i], h_symm_dsts[i]);
        }

        auto [symm_lower_triangular_last, symm_diagonal_last] =
          partition_and_sort_edges(symm_edges.begin(), symm_edges.end(), true);

        ASSERT_TRUE(std::equal(
          symm_edges.begin(), symm_lower_triangular_last, symm_diagonal_last));  // check symmetric

        symm_edges.resize(std::distance(symm_edges.begin(), symm_diagonal_last));

        ASSERT_TRUE(std::equal(org_edges.begin(), org_edges.end(), symm_edges.begin()));
      }
    }
  }
};

using Tests_Symmetrize_File = Tests_Symmetrize<cugraph::test::File_Usecase>;
using Tests_Symmetrize_Rmat = Tests_Symmetrize<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Symmetrize_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Symmetrize_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Symmetrize_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Symmetrize_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Symmetrize_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Symmetrize_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Symmetrize_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Symmetrize_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Symmetrize_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Symmetrize_Usecase{false, false},
                      Symmetrize_Usecase{true, false},
                      Symmetrize_Usecase{false, true},
                      Symmetrize_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Symmetrize_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Symmetrize_Usecase{false, false},
                      Symmetrize_Usecase{true, false},
                      Symmetrize_Usecase{false, true},
                      Symmetrize_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Symmetrize_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Symmetrize_Usecase{false, false, false},
                      Symmetrize_Usecase{true, false, false},
                      Symmetrize_Usecase{false, true, false},
                      Symmetrize_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
