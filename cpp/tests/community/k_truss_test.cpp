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

#include "utilities/check_utilities.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

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

struct KTruss_Usecase {
  int32_t k_{3};
  bool test_weighted_{false};
  bool check_correctness_{true};
};

template <typename input_usecase_t>
class Tests_KTruss : public ::testing::TestWithParam<std::tuple<KTruss_Usecase, input_usecase_t>> {
 public:
  Tests_KTruss() {}

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

  template <typename vertex_t, typename edge_t, typename weight_t>
  std::tuple<std::vector<vertex_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
  k_truss_reference(
    std::vector<vertex_t> h_offsets,
    std::vector<vertex_t> h_indices,
    std::optional<std::vector<weight_t>> h_values,
    edge_t k)
  {
    std::vector<vertex_t> vertices(h_offsets.size() - 1);
    std::iota(vertices.begin(), vertices.end(), 0);

    auto n_dropped = 1;

    while (n_dropped > 0) {
      n_dropped = 0;
      std::set<vertex_t> seen;
      // Go over all the vertices
      for (auto u = vertices.begin(); u != vertices.end(); ++u) {
        std::set<vertex_t> nbrs_u;
        // Find all neighbors of u from the offsets and indices array
        auto idx_start = (h_offsets.begin() + (*u));
        auto idx_end   = idx_start + 1;

        for (edge_t i = *idx_start; i < *idx_end; ++i) {
          nbrs_u.insert(*(h_indices.begin() + i));
        }

        seen.insert(*u);
        std::set<vertex_t> new_nbrs;
        std::set_difference(nbrs_u.begin(),
                            nbrs_u.end(),
                            seen.begin(),
                            seen.end(),
                            std::inserter(new_nbrs, new_nbrs.end()));

        // Finding the neighbors of v
        for (auto v = new_nbrs.begin(); v != new_nbrs.end(); ++v) {
          std::set<vertex_t> nbrs_v;
          // Find all neighbors of v from the offsets and indices array
          idx_start = (h_offsets.begin() + (*v));
          idx_end   = idx_start + 1;
          for (edge_t i = *idx_start; i < *idx_end; ++i) {
            nbrs_v.insert(*(h_indices.begin() + i));
          }

          std::set<vertex_t> nbr_intersection_u_v;
          // Find the intersection of nbr_u and nbr_v
          std::set_intersection(nbrs_u.begin(),
                                nbrs_u.end(),
                                nbrs_v.begin(),
                                nbrs_v.end(),
                                std::inserter(nbr_intersection_u_v, nbr_intersection_u_v.end()));

          if (nbr_intersection_u_v.size() < (k - 2)) {
            auto del_v = std::find(
              h_indices.begin() + h_offsets[*u], h_indices.begin() + h_offsets[*u + 1], *v);

            if (h_values) {
              (*h_values).erase((*h_values).begin() + std::distance(h_indices.begin(), del_v));
            }

            std::transform(std::begin(h_offsets) + (*u) + 1,
                           std::end(h_offsets),
                           std::begin(h_offsets) + (*u) + 1,
                           [](int x) { return x - 1; });
            h_indices.erase(del_v);

            // Delete edge in both directions
            auto del_u = std::find(
              h_indices.begin() + h_offsets[*v], h_indices.begin() + h_offsets[*v + 1], *u);

            if (h_values) {
              (*h_values).erase((*h_values).begin() + std::distance(h_indices.begin(), del_u));
            }
            std::transform(std::begin(h_offsets) + (*v) + 1,
                           std::end(h_offsets),
                           std::begin(h_offsets) + (*v) + 1,
                           [](int x) { return x - 1; });
            h_indices.erase(del_u);
            n_dropped += 1;
          }
        }
      }
    }

    h_offsets.erase(std::unique(h_offsets.begin() + 1, h_offsets.end()),
                    h_offsets.end());  // CSR start from 0

    return std::make_tuple(
      std::move(h_offsets), std::move(h_indices), std::move(h_values));
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<KTruss_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber               = false;
    auto [k_truss_usecase, input_usecase] = param;
    raft::handle_t handle{};

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("SG Construct graph");
    }

    // NX k_truss is not implemented for graph with self loop and multi edges therefore dropped
    // them especially for rmat generated graphs.
    auto [graph, edge_weight, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, k_truss_usecase.test_weighted_, renumber, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("K-truss");
    }

    auto [d_cugraph_src, d_cugraph_dst, d_cugraph_wgt] =
      cugraph::k_truss<vertex_t, edge_t, weight_t, false>(
        handle,
        graph_view,
        edge_weight ? std::make_optional((*edge_weight).view()) : std::nullopt,
        k_truss_usecase.k_,
        false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (k_truss_usecase.check_correctness_) {
      std::optional<cugraph::graph_t<vertex_t, edge_t, false, false>> modified_graph{std::nullopt};

      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
        modified_edge_weight{std::nullopt};
      std::tie(*modified_graph, modified_edge_weight, std::ignore, std::ignore, std::ignore) =
        cugraph::
          create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, false>(
            handle,
            std::nullopt,
            std::move(d_cugraph_src),
            std::move(d_cugraph_dst),
            std::move(d_cugraph_wgt),
            std::nullopt,
            std::nullopt,
            cugraph::graph_properties_t{true, false},
            renumber);

      // Convert cugraph results to CSR
      auto [h_cugraph_offsets, h_cugraph_indices, h_cugraph_values] =
        cugraph::test::graph_to_host_csr(
          handle,
          (*modified_graph).view(),
          modified_edge_weight ? std::make_optional((*modified_edge_weight).view()) : std::nullopt,
          std::optional<raft::device_span<vertex_t const>>(std::nullopt));

      // Remove isolated vertices.
      h_cugraph_offsets.erase(std::unique(h_cugraph_offsets.begin() + 1, h_cugraph_offsets.end()),
                              h_cugraph_offsets.end());  // CSR start from 0

      auto [h_offsets, h_indices, h_values] = cugraph::test::graph_to_host_csr(
        handle,
        graph_view,
        edge_weight ? std::make_optional((*edge_weight).view()) : std::nullopt,
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));

      auto [h_reference_offsets, h_reference_indices, h_reference_values] =
        k_truss_reference<vertex_t, edge_t, weight_t>(
          h_offsets,
          h_indices,
          h_values,
          k_truss_usecase.k_);

      EXPECT_EQ(h_cugraph_offsets.size(), h_reference_offsets.size());

      ASSERT_TRUE(std::equal(
        h_cugraph_offsets.begin(), h_cugraph_offsets.end(), h_reference_offsets.begin()));

      ASSERT_TRUE(std::equal(
        h_cugraph_indices.begin(), h_cugraph_indices.end(), h_reference_indices.begin()));

      if (edge_weight) {
        auto compare_functor = host_nearly_equal<weight_t>{
          weight_t{1e-3},
          weight_t{(weight_t{1} / static_cast<weight_t>((*h_cugraph_values).size())) *
                   weight_t{1e-3}}};
        EXPECT_TRUE(std::equal((*h_cugraph_values).begin(),
                               (*h_cugraph_values).end(),
                               (*h_reference_values).begin(),
                               compare_functor));
      }
    }
  }
};

using Tests_KTruss_File = Tests_KTruss<cugraph::test::File_Usecase>;
using Tests_KTruss_Rmat = Tests_KTruss<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_KTruss_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_KTruss_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_KTruss_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_KTruss_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_KTruss_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KTruss_Usecase{5, true, false},
                      KTruss_Usecase{4, true, false},
                      KTruss_Usecase{9, true, true},
                      KTruss_Usecase{7, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_KTruss_Rmat,
                         // enable correctness checks
                         ::testing::Combine(::testing::Values(KTruss_Usecase{5, false, true},
                                                              KTruss_Usecase{4, false, true},
                                                              KTruss_Usecase{9, true, true},
                                                              KTruss_Usecase{7, true, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_KTruss_Rmat,
  // disable correctness checks for large graphs
  // FIXME: High memory footprint. Perform nbr_intersection in chunks.
  ::testing::Combine(
    ::testing::Values(KTruss_Usecase{4, false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(14, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
