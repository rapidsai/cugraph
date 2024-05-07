/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include "utilities/check_utilities.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

struct EdgeTriangleCount_Usecase {
  bool test_weighted_{false};
  bool check_correctness_{true};
};

template <typename input_usecase_t>
class Tests_EdgeTriangleCount
  : public ::testing::TestWithParam<std::tuple<EdgeTriangleCount_Usecase, input_usecase_t>> {
 public:
  Tests_EdgeTriangleCount() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // FIXME: There is an utility equivalent functor not
  // supporting host vectors.
  template <typename type_t>
  struct host_nearly_equal {
    const type_t threshold_ratio;
    const type_t threshold_magnitude;

    bool operator()(type_t lhs, type_t rhs) const
    {
      return std::abs(lhs - rhs) <
             std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
    }
  };

  template <typename vertex_t, typename edge_t>
  std::vector<edge_t> edge_triangle_count_reference(std::vector<vertex_t> h_srcs,
                                                    std::vector<vertex_t> h_dsts)
  {
    std::vector<vertex_t> edge_triangle_counts(h_srcs.size());
    std::uninitialized_fill(edge_triangle_counts.begin(), edge_triangle_counts.end(), 0);

    for (int i = 0; i < h_srcs.size(); ++i) {  // edge centric implementation
      // for each edge, find the intersection
      auto src          = h_srcs[i];
      auto dst          = h_dsts[i];
      auto it_src_start = std::lower_bound(h_srcs.begin(), h_srcs.end(), src);
      auto src_start    = std::distance(h_srcs.begin(), it_src_start);

      auto src_end =
        src_start + std::distance(it_src_start, std::upper_bound(it_src_start, h_srcs.end(), src));

      auto it_dst_start = std::lower_bound(h_srcs.begin(), h_srcs.end(), dst);
      auto dst_start    = std::distance(h_srcs.begin(), it_dst_start);
      auto dst_end =
        dst_start + std::distance(it_dst_start, std::upper_bound(it_dst_start, h_srcs.end(), dst));

      std::set<vertex_t> nbr_intersection;
      std::set_intersection(h_dsts.begin() + src_start,
                            h_dsts.begin() + src_end,
                            h_dsts.begin() + dst_start,
                            h_dsts.begin() + dst_end,
                            std::inserter(nbr_intersection, nbr_intersection.end()));
      // Find the supporting edges
      for (auto v : nbr_intersection) {
        auto it_edge  = std::lower_bound(h_dsts.begin() + src_start, h_dsts.begin() + src_end, v);
        auto idx_edge = std::distance(h_dsts.begin(), it_edge);
        edge_triangle_counts[idx_edge] += 1;

        it_edge  = std::lower_bound(h_dsts.begin() + dst_start, h_dsts.begin() + dst_end, v);
        idx_edge = std::distance(h_dsts.begin(), it_edge);
      }
    }

    std::transform(edge_triangle_counts.begin(),
                   edge_triangle_counts.end(),
                   edge_triangle_counts.begin(),
                   [](auto count) { return count * 3; });
    return std::move(edge_triangle_counts);
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(
    std::tuple<EdgeTriangleCount_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber                           = false;
    auto [edge_triangle_count_usecase, input_usecase] = param;
    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("SG Construct graph");
    }

    auto [graph, edge_weight, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, edge_triangle_count_usecase.test_weighted_, renumber, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> d_edge_triangle_counts{std::nullopt};
    
    auto d_cugraph_results =
      cugraph::edge_triangle_count<vertex_t, edge_t, false>(handle, graph_view);

    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, d_edge_triangle_counts) =
      cugraph::decompress_to_edgelist(
        handle,
        graph_view,
        std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        std::make_optional(d_cugraph_results.view()),
        std::optional<raft::device_span<vertex_t const>>{std::nullopt});  // FIXME: No longer needed
  
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("EdgeTriangleCount");
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (edge_triangle_count_usecase.check_correctness_) {
      std::optional<cugraph::graph_t<vertex_t, edge_t, false, false>> modified_graph{std::nullopt};
      std::vector<vertex_t> h_srcs(edgelist_srcs.size());
      std::vector<vertex_t> h_dsts(edgelist_dsts.size());
      std::tie(h_srcs, h_dsts, std::ignore) = cugraph::test::graph_to_host_coo(
        handle,
        graph_view,
        edge_weight ? std::make_optional((*edge_weight).view()) : std::nullopt,
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));

      auto h_cugraph_edge_triangle_counts = cugraph::test::to_host(handle, *d_edge_triangle_counts);

      auto h_reference_edge_triangle_counts =
        edge_triangle_count_reference<vertex_t, edge_t>(h_srcs, h_dsts);

      for (size_t i = 0; i < h_srcs.size(); ++i) {
        ASSERT_EQ(h_cugraph_edge_triangle_counts[i], h_reference_edge_triangle_counts[i])
          << "Edge triangle count values do not match with the reference values.";
      }
    }
  }
};

using Tests_EdgeTriangleCount_File = Tests_EdgeTriangleCount<cugraph::test::File_Usecase>;
using Tests_EdgeTriangleCount_Rmat = Tests_EdgeTriangleCount<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_EdgeTriangleCount_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}
TEST_P(Tests_EdgeTriangleCount_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}
TEST_P(Tests_EdgeTriangleCount_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}
TEST_P(Tests_EdgeTriangleCount_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_EdgeTriangleCount_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EdgeTriangleCount_Usecase{false, true},
                      EdgeTriangleCount_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_EdgeTriangleCount_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(EdgeTriangleCount_Usecase{false, true},
                      EdgeTriangleCount_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_EdgeTriangleCount_Rmat,
  // disable correctness checks for large graphs
  // FIXME: High memory footprint. Perform nbr_intersection in chunks.
  ::testing::Combine(
    ::testing::Values(EdgeTriangleCount_Usecase{false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(16, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
