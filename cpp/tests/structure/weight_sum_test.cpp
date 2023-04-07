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

template <typename vertex_t, typename edge_t, typename weight_t>
void weight_sum_reference(edge_t const* offsets,
                          vertex_t const* indices,
                          weight_t const* weights,
                          weight_t* weight_sums,
                          vertex_t num_vertices,
                          bool major)
{
  if (!major) { std::fill(weight_sums, weight_sums + num_vertices, weight_t{0.0}); }
  for (vertex_t i = 0; i < num_vertices; ++i) {
    if (major) {
      weight_sums[i] =
        std::accumulate(weights + offsets[i], weights + offsets[i + 1], weight_t{0.0});
    } else {
      for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
        auto nbr = indices[j];
        weight_sums[nbr] += weights[j];
      }
    }
  }

  return;
}

typedef struct WeightSum_Usecase_t {
  std::string graph_file_full_path{};

  WeightSum_Usecase_t(std::string const& graph_file_path)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} WeightSum_Usecase;

class Tests_WeightSum : public ::testing::TestWithParam<WeightSum_Usecase> {
 public:
  Tests_WeightSum() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(WeightSum_Usecase const& configuration)
  {
    raft::handle_t handle{};

    cugraph::graph_t<vertex_t, edge_t, store_transposed, false> graph(handle);
    std::optional<
      cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, false>,
                               weight_t>>
      edge_weights{std::nullopt};
    std::tie(graph, edge_weights, std::ignore) = cugraph::test::
      read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, configuration.graph_file_full_path, true, false);
    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    auto h_offsets =
      cugraph::test::to_host(handle, graph_view.local_edge_partition_view().offsets());
    auto h_indices =
      cugraph::test::to_host(handle, graph_view.local_edge_partition_view().indices());
    auto h_weights = cugraph::test::to_host(
      handle,
      raft::device_span<weight_t const>((*edge_weight_view).value_firsts()[0],
                                        (*edge_weight_view).edge_counts()[0]));

    std::vector<weight_t> h_reference_in_weight_sums(graph_view.number_of_vertices());
    std::vector<weight_t> h_reference_out_weight_sums(graph_view.number_of_vertices());

    weight_sum_reference(h_offsets.data(),
                         h_indices.data(),
                         h_weights.data(),
                         h_reference_in_weight_sums.data(),
                         graph_view.number_of_vertices(),
                         store_transposed);

    weight_sum_reference(h_offsets.data(),
                         h_indices.data(),
                         h_weights.data(),
                         h_reference_out_weight_sums.data(),
                         graph_view.number_of_vertices(),
                         !store_transposed);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto d_in_weight_sums = cugraph::compute_in_weight_sums(handle, graph_view, *edge_weight_view);
    auto d_out_weight_sums =
      cugraph::compute_out_weight_sums(handle, graph_view, *edge_weight_view);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto h_cugraph_in_weight_sums  = cugraph::test::to_host(handle, d_in_weight_sums);
    auto h_cugraph_out_weight_sums = cugraph::test::to_host(handle, d_out_weight_sums);

    auto threshold_ratio     = weight_t{1e-4};
    auto threshold_magnitude = std::numeric_limits<weight_t>::min();
    auto nearly_equal        = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
      return std::abs(lhs - rhs) <
             std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
    };

    ASSERT_TRUE(std::equal(h_reference_in_weight_sums.begin(),
                           h_reference_in_weight_sums.end(),
                           h_cugraph_in_weight_sums.begin(),
                           nearly_equal))
      << "In-weight-sum values do not match with the reference values.";
    ASSERT_TRUE(std::equal(h_reference_out_weight_sums.begin(),
                           h_reference_out_weight_sums.end(),
                           h_cugraph_out_weight_sums.begin(),
                           nearly_equal))
      << "Out-weight-sum values do not match with the reference values.";
  }
};

// FIXME: add tests for type combinations

TEST_P(Tests_WeightSum, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
}

TEST_P(Tests_WeightSum, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(simple_test,
                         Tests_WeightSum,
                         ::testing::Values(WeightSum_Usecase("test/datasets/karate.mtx"),
                                           WeightSum_Usecase("test/datasets/web-Google.mtx"),
                                           WeightSum_Usecase("test/datasets/ljournal-2008.mtx"),
                                           WeightSum_Usecase("test/datasets/webbase-1M.mtx")));

CUGRAPH_TEST_PROGRAM_MAIN()
