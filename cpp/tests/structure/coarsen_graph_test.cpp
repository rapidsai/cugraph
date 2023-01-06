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
#include <map>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t>
void check_coarsened_graph_results(edge_t const* org_offsets,
                                   vertex_t const* org_indices,
                                   std::optional<weight_t const*> org_weights,
                                   vertex_t const* org_labels,
                                   edge_t const* coarse_offsets,
                                   vertex_t const* coarse_indices,
                                   std::optional<weight_t const*> coarse_weights,
                                   vertex_t const* coarse_vertex_labels,
                                   vertex_t num_org_vertices,
                                   vertex_t num_coarse_vertices)
{
  ASSERT_TRUE(org_weights.has_value() == coarse_weights.has_value());
  ASSERT_TRUE(std::is_sorted(org_offsets, org_offsets + num_org_vertices));
  ASSERT_TRUE(std::count_if(org_indices,
                            org_indices + org_offsets[num_org_vertices],
                            [num_org_vertices](auto nbr) {
                              return !cugraph::is_valid_vertex(num_org_vertices, nbr);
                            }) == 0);
  ASSERT_TRUE(std::is_sorted(coarse_offsets, coarse_offsets + num_coarse_vertices));
  ASSERT_TRUE(std::count_if(coarse_indices,
                            coarse_indices + coarse_offsets[num_coarse_vertices],
                            [num_coarse_vertices](auto nbr) {
                              return !cugraph::is_valid_vertex(num_coarse_vertices, nbr);
                            }) == 0);
  ASSERT_TRUE(num_coarse_vertices <= num_org_vertices);

  std::vector<vertex_t> org_unique_labels(num_org_vertices);
  std::iota(org_unique_labels.begin(), org_unique_labels.end(), vertex_t{0});
  std::transform(org_unique_labels.begin(),
                 org_unique_labels.end(),
                 org_unique_labels.begin(),
                 [org_labels](auto v) { return org_labels[v]; });
  std::sort(org_unique_labels.begin(), org_unique_labels.end());
  org_unique_labels.resize(std::distance(
    org_unique_labels.begin(), std::unique(org_unique_labels.begin(), org_unique_labels.end())));

  ASSERT_TRUE(org_unique_labels.size() == static_cast<size_t>(num_coarse_vertices));

  {
    std::vector<vertex_t> tmp_coarse_vertex_labels(coarse_vertex_labels,
                                                   coarse_vertex_labels + num_coarse_vertices);
    std::sort(tmp_coarse_vertex_labels.begin(), tmp_coarse_vertex_labels.end());
    ASSERT_TRUE(std::unique(tmp_coarse_vertex_labels.begin(), tmp_coarse_vertex_labels.end()) ==
                tmp_coarse_vertex_labels.end());
    ASSERT_TRUE(std::equal(
      org_unique_labels.begin(), org_unique_labels.end(), tmp_coarse_vertex_labels.begin()));
  }

  std::vector<std::tuple<vertex_t, vertex_t>> label_org_vertex_pairs(num_org_vertices);
  for (vertex_t i = 0; i < num_org_vertices; ++i) {
    label_org_vertex_pairs[i] = std::make_tuple(org_labels[i], i);
  }
  std::sort(label_org_vertex_pairs.begin(), label_org_vertex_pairs.end());

  std::map<vertex_t, vertex_t> label_to_coarse_vertex_map{};
  for (vertex_t i = 0; i < num_coarse_vertices; ++i) {
    label_to_coarse_vertex_map[coarse_vertex_labels[i]] = i;
  }

  auto threshold_ratio = org_weights ? weight_t{1e-4} : weight_t{1.0} /* irrelevant */;
  auto threshold_magnitude =
    org_weights
      ? (std::accumulate(
           *coarse_weights, *coarse_weights + coarse_offsets[num_coarse_vertices], weight_t{0.0}) /
         static_cast<weight_t>(coarse_offsets[num_coarse_vertices])) *
          threshold_ratio
      : weight_t{1.0} /* irrelevant */;

  for (size_t i = 0; i < org_unique_labels.size(); ++i) {  // for each vertex in the coarse graph
    auto lb =
      std::lower_bound(label_org_vertex_pairs.begin(),
                       label_org_vertex_pairs.end(),
                       std::make_tuple(org_unique_labels[i],
                                       cugraph::invalid_vertex_id<vertex_t>::value /* dummy */),
                       [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
    auto ub =
      std::upper_bound(label_org_vertex_pairs.begin(),
                       label_org_vertex_pairs.end(),
                       std::make_tuple(org_unique_labels[i],
                                       cugraph::invalid_vertex_id<vertex_t>::value /* dummy */),
                       [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
    auto count  = std::distance(lb, ub);
    auto offset = std::distance(label_org_vertex_pairs.begin(), lb);
    if (org_weights) {
      std::vector<std::tuple<vertex_t, weight_t>> coarse_nbr_weight_pairs0{};
      std::for_each(lb,
                    ub,
                    [org_offsets,
                     org_indices,
                     org_weights = (*org_weights),
                     org_labels,
                     &label_to_coarse_vertex_map,
                     &coarse_nbr_weight_pairs0](auto t) {
                      auto org_vertex = std::get<1>(t);
                      std::vector<std::tuple<vertex_t, weight_t>> tmp_pairs(
                        org_offsets[org_vertex + 1] - org_offsets[org_vertex]);
                      for (auto j = org_offsets[org_vertex]; j < org_offsets[org_vertex + 1]; ++j) {
                        tmp_pairs[j - org_offsets[org_vertex]] = std::make_tuple(
                          label_to_coarse_vertex_map[org_labels[org_indices[j]]], org_weights[j]);
                      }
                      coarse_nbr_weight_pairs0.insert(
                        coarse_nbr_weight_pairs0.end(), tmp_pairs.begin(), tmp_pairs.end());
                    });
      std::sort(coarse_nbr_weight_pairs0.begin(), coarse_nbr_weight_pairs0.end());
      // reduce by key
      {
        size_t run_start_idx = 0;
        for (size_t j = 1; j < coarse_nbr_weight_pairs0.size(); ++j) {
          auto& start = coarse_nbr_weight_pairs0[run_start_idx];
          auto& cur   = coarse_nbr_weight_pairs0[j];
          if (std::get<0>(start) == std::get<0>(cur)) {
            std::get<1>(start) += std::get<1>(cur);
            std::get<0>(cur) = cugraph::invalid_vertex_id<vertex_t>::value;
          } else {
            run_start_idx = j;
          }
        }
        coarse_nbr_weight_pairs0.erase(
          std::remove_if(
            coarse_nbr_weight_pairs0.begin(),
            coarse_nbr_weight_pairs0.end(),
            [](auto t) { return std::get<0>(t) == cugraph::invalid_vertex_id<vertex_t>::value; }),
          coarse_nbr_weight_pairs0.end());
      }

      auto coarse_vertex = label_to_coarse_vertex_map[org_unique_labels[i]];
      std::vector<std::tuple<vertex_t, weight_t>> coarse_nbr_weight_pairs1(
        coarse_offsets[coarse_vertex + 1] - coarse_offsets[coarse_vertex]);
      for (auto j = coarse_offsets[coarse_vertex]; j < coarse_offsets[coarse_vertex + 1]; ++j) {
        coarse_nbr_weight_pairs1[j - coarse_offsets[coarse_vertex]] =
          std::make_tuple(coarse_indices[j], (*coarse_weights)[j]);
      }
      std::sort(coarse_nbr_weight_pairs1.begin(), coarse_nbr_weight_pairs1.end());

      ASSERT_TRUE(coarse_nbr_weight_pairs0.size() == coarse_nbr_weight_pairs1.size());
      ASSERT_TRUE(std::equal(
        coarse_nbr_weight_pairs0.begin(),
        coarse_nbr_weight_pairs0.end(),
        coarse_nbr_weight_pairs1.begin(),
        [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
          return std::get<0>(lhs) == std::get<0>(rhs)
                   ? (std::abs(std::get<1>(lhs) - std::get<1>(rhs)) <=
                      std::max(std::max(std::abs(std::get<1>(lhs)), std::abs(std::get<1>(rhs))) *
                                 threshold_ratio,
                               threshold_magnitude))
                   : false;
        }));
    } else {
      std::vector<vertex_t> coarse_nbrs0{};
      std::for_each(
        lb,
        ub,
        [org_offsets, org_indices, org_labels, &label_to_coarse_vertex_map, &coarse_nbrs0](auto t) {
          auto org_vertex = std::get<1>(t);
          std::vector<vertex_t> tmp_nbrs(org_offsets[org_vertex + 1] - org_offsets[org_vertex]);
          std::transform(org_indices + org_offsets[org_vertex],
                         org_indices + org_offsets[org_vertex + 1],
                         tmp_nbrs.begin(),
                         [org_labels, &label_to_coarse_vertex_map](auto nbr) {
                           return label_to_coarse_vertex_map[org_labels[nbr]];
                         });
          coarse_nbrs0.insert(coarse_nbrs0.end(), tmp_nbrs.begin(), tmp_nbrs.end());
        });
      std::sort(coarse_nbrs0.begin(), coarse_nbrs0.end());
      coarse_nbrs0.resize(
        std::distance(coarse_nbrs0.begin(), std::unique(coarse_nbrs0.begin(), coarse_nbrs0.end())));

      auto coarse_vertex = label_to_coarse_vertex_map[org_unique_labels[i]];
      auto coarse_offset = coarse_offsets[coarse_vertex];
      auto coarse_count  = coarse_offsets[coarse_vertex + 1] - coarse_offset;
      std::vector<vertex_t> coarse_nbrs1(coarse_indices + coarse_offset,
                                         coarse_indices + coarse_offset + coarse_count);
      std::sort(coarse_nbrs1.begin(), coarse_nbrs1.end());

      ASSERT_TRUE(coarse_nbrs0.size() == coarse_nbrs1.size());
      ASSERT_TRUE(std::equal(coarse_nbrs0.begin(), coarse_nbrs0.end(), coarse_nbrs1.begin()));
    }
  }

  return;
}

struct CoarsenGraph_Usecase {
  double coarsen_ratio{0.0};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_CoarsenGraph
  : public ::testing::TestWithParam<std::tuple<CoarsenGraph_Usecase, input_usecase_t>> {
 public:
  Tests_CoarsenGraph() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(
    std::tuple<CoarsenGraph_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber                     = true;
    auto [coarsen_graph_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // FIXME: remove this once we drop Pascal support
    if (handle.get_device_properties().major < 7) {  // Pascal is not supported, skip testing
      return;
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, coarsen_graph_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    if (graph_view.number_of_vertices() == 0) { return; }

    std::vector<vertex_t> h_labels(graph_view.number_of_vertices());
    auto num_labels = std::max(
      static_cast<vertex_t>(h_labels.size() * coarsen_graph_usecase.coarsen_ratio), vertex_t{1});

    std::default_random_engine generator{};
    std::uniform_int_distribution<vertex_t> distribution{0, num_labels - 1};

    std::for_each(h_labels.begin(), h_labels.end(), [&distribution, &generator](auto& label) {
      label = distribution(generator);
    });

    rmm::device_uvector<vertex_t> d_labels(h_labels.size(), handle.get_stream());
    raft::update_device(d_labels.data(), h_labels.data(), h_labels.size(), handle.get_stream());

    handle.sync_stream();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Graph coarsening");
    }

    auto [coarse_graph, coarse_edge_weights, coarse_vertices_to_labels] =
      cugraph::coarsen_graph(handle, graph_view, edge_weight_view, d_labels.begin(), true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (coarsen_graph_usecase.check_correctness) {
      auto h_org_offsets =
        cugraph::test::to_host(handle, graph_view.local_edge_partition_view().offsets());
      auto h_org_indices =
        cugraph::test::to_host(handle, graph_view.local_edge_partition_view().indices());
      auto h_org_weights = cugraph::test::to_host(
        handle,
        edge_weight_view
          ? std::make_optional(raft::device_span<weight_t const>(
              (*edge_weight_view).value_firsts()[0], (*edge_weight_view).edge_counts()[0]))
          : std::nullopt);

      auto coarse_graph_view = coarse_graph.view();
      auto coarse_edge_weight_view =
        coarse_edge_weights ? std::make_optional((*coarse_edge_weights).view()) : std::nullopt;

      auto h_coarse_offsets =
        cugraph::test::to_host(handle, coarse_graph_view.local_edge_partition_view().offsets());
      auto h_coarse_indices =
        cugraph::test::to_host(handle, coarse_graph_view.local_edge_partition_view().indices());
      auto h_coarse_weights = cugraph::test::to_host(
        handle,
        coarse_edge_weight_view ? std::make_optional(raft::device_span<weight_t const>(
                                    (*coarse_edge_weight_view).value_firsts()[0],
                                    (*coarse_edge_weight_view).edge_counts()[0]))
                                : std::nullopt);

      auto h_coarse_vertices_to_labels = cugraph::test::to_host(handle, *coarse_vertices_to_labels);

      check_coarsened_graph_results(
        h_org_offsets.data(),
        h_org_indices.data(),
        h_org_weights ? std::optional<weight_t const*>{(*h_org_weights).data()} : std::nullopt,
        h_labels.data(),
        h_coarse_offsets.data(),
        h_coarse_indices.data(),
        h_coarse_weights ? std::optional<weight_t const*>{(*h_coarse_weights).data()}
                         : std::nullopt,
        h_coarse_vertices_to_labels.data(),
        graph_view.number_of_vertices(),
        coarse_graph_view.number_of_vertices());
    }
  }
};

using Tests_CoarsenGraph_File = Tests_CoarsenGraph<cugraph::test::File_Usecase>;
using Tests_CoarsenGraph_Rmat = Tests_CoarsenGraph<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_CoarsenGraph_File, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_CoarsenGraph_File, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_CoarsenGraph_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_CoarsenGraph_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_CoarsenGraph_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  run_current_test<int32_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_CoarsenGraph_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  run_current_test<int32_t, int64_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_CoarsenGraph_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  run_current_test<int64_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_CoarsenGraph_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  run_current_test<int64_t, int64_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_CoarsenGraph_File,
  ::testing::Combine(
    // enable correctness check
    ::testing::Values(CoarsenGraph_Usecase{0.2, false}, CoarsenGraph_Usecase{0.2, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_CoarsenGraph_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(CoarsenGraph_Usecase{0.2, false}, CoarsenGraph_Usecase{0.2, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false),
                      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_CoarsenGraph_File,
  ::testing::Combine(::testing::Values(CoarsenGraph_Usecase{0.2, false, false},
                                       CoarsenGraph_Usecase{0.2, true, false}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_CoarsenGraph_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(CoarsenGraph_Usecase{0.2, false, false},
                      CoarsenGraph_Usecase{0.2, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false),
                      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
