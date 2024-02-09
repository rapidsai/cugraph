/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
        std::reduce(weights + offsets[i], weights + offsets[i + 1], weight_t{0.0});
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
  bool edge_masking{false};
  bool check_correctness{true};
} WeightSum_Usecase;

template <typename input_usecase_t>
class Tests_WeightSum : public ::testing::TestWithParam<std::tuple<WeightSum_Usecase, input_usecase_t>> {
 public:
  Tests_WeightSum() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(WeightSum_Usecase const& weight_sum_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber      = true;
    constexpr bool test_weighted = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Compute in-weight sums");
    }

    auto d_in_weight_sums = cugraph::compute_in_weight_sums(handle, graph_view, *edge_weight_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Compute out-weight sums");
    }

    auto d_out_weight_sums =
      cugraph::compute_out_weight_sums(handle, graph_view, *edge_weight_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (weight_sum_usecase.check_correctness) {
      auto [h_offsets, h_indices, h_weights] = cugraph::test::graph_to_host_csr(handle, graph_view, edge_weight_view);

      std::vector<weight_t> h_reference_in_weight_sums(graph_view.number_of_vertices());
      std::vector<weight_t> h_reference_out_weight_sums(graph_view.number_of_vertices());

      weight_sum_reference(h_offsets.data(),
                           h_indices.data(),
                           (*h_weights).data(),
                           h_reference_in_weight_sums.data(),
                           graph_view.number_of_vertices(),
                           store_transposed);

      weight_sum_reference(h_offsets.data(),
                           h_indices.data(),
                           (*h_weights).data(),
                           h_reference_out_weight_sums.data(),
                           graph_view.number_of_vertices(),
                           !store_transposed);

      auto h_cugraph_in_weight_sums  = cugraph::test::to_host(handle, d_in_weight_sums);
      auto h_cugraph_out_weight_sums = cugraph::test::to_host(handle, d_out_weight_sums);

      auto threshold_ratio     = weight_t{2.0 * 1e-4};
      auto threshold_magnitude = std::numeric_limits<weight_t>::min();
      auto nearly_equal        = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        auto ret = 
          std::abs(lhs - rhs) <
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
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
  }
};

using Tests_WeightSum_File = Tests_WeightSum<cugraph::test::File_Usecase>;
using Tests_WeightSum_Rmat = Tests_WeightSum<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_WeightSum_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeightSum_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeightSum_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeightSum_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeightSum_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeightSum_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeightSum_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_WeightSum_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, true>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_WeightSum_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(WeightSum_Usecase{false}, WeightSum_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_WeightSum_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(WeightSum_Usecase{false}, WeightSum_Usecase{true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_WeightSum_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(WeightSum_Usecase{false, false}, WeightSum_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
