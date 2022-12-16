/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

template <typename vertex_t, typename weight_t>
void eigenvector_centrality_reference(vertex_t const* src,
                                      vertex_t const* dst,
                                      std::optional<weight_t const*> weights,
                                      size_t num_edges,
                                      weight_t* centralities,
                                      vertex_t num_vertices,
                                      weight_t epsilon,
                                      size_t max_iterations)
{
  if (num_vertices == 0) { return; }

  std::vector<double> tmp_centralities(num_vertices, double{0});
  std::vector<double> old_centralities(num_vertices, double{0.0});

  std::fill(tmp_centralities.begin(),
            tmp_centralities.end(),
            double{1} / static_cast<double>(num_vertices));

  size_t iter{0};
  while (true) {
    std::copy(tmp_centralities.begin(), tmp_centralities.end(), old_centralities.begin());
    std::fill(tmp_centralities.begin(), tmp_centralities.end(), double{0});

    for (size_t e = 0; e < num_edges; ++e) {
      auto w = weights ? (*weights)[e] : weight_t{1.0};
      tmp_centralities[src[e]] += old_centralities[dst[e]] * w;
    }

    auto l2_norm = std::sqrt(std::inner_product(
      tmp_centralities.begin(), tmp_centralities.end(), tmp_centralities.begin(), double{0.0}));

    std::transform(tmp_centralities.begin(),
                   tmp_centralities.end(),
                   tmp_centralities.begin(),
                   [l2_norm](auto& val) { return val / l2_norm; });

    double diff_sum{0.0};
    double diff_max{0};
    for (vertex_t i = 0; i < num_vertices; ++i) {
      diff_sum += std::abs(tmp_centralities[i] - old_centralities[i]);
      if (std::abs(tmp_centralities[i] - old_centralities[i]) > diff_max)
        diff_max = std::abs(tmp_centralities[i] - old_centralities[i]);
    }

    if (diff_sum < (num_vertices * epsilon)) { break; }
    iter++;
    ASSERT_TRUE(iter < max_iterations);
  }

  std::transform(tmp_centralities.begin(), tmp_centralities.end(), centralities, [](auto v) {
    return static_cast<weight_t>(v);
  });

  return;
}

struct EigenvectorCentrality_Usecase {
  size_t max_iterations{std::numeric_limits<size_t>::max()};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_EigenvectorCentrality
  : public ::testing::TestWithParam<std::tuple<EigenvectorCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_EigenvectorCentrality() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(EigenvectorCentrality_Usecase const& eigenvector_usecase,
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
        handle, input_usecase, eigenvector_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    weight_t constexpr epsilon{1e-6};

    rmm::device_uvector<weight_t> d_centralities(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Eigenvector centrality");
    }

    d_centralities =
      cugraph::eigenvector_centrality(handle,
                                      graph_view,
                                      edge_weight_view,
                                      std::optional<raft::device_span<weight_t const>>{},
                                      epsilon,
                                      eigenvector_usecase.max_iterations,
                                      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (eigenvector_usecase.check_correctness) {
      auto [dst_v, src_v, opt_wgt_v] = cugraph::decompress_to_edgelist(
        handle,
        graph_view,
        edge_weight_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt});

      auto h_src     = cugraph::test::to_host(handle, src_v);
      auto h_dst     = cugraph::test::to_host(handle, dst_v);
      auto h_weights = cugraph::test::to_host(handle, opt_wgt_v);

      std::vector<weight_t> h_reference_centralities(graph_view.number_of_vertices());

      eigenvector_centrality_reference(
        h_src.data(),
        h_dst.data(),
        h_weights ? std::make_optional<weight_t const*>(h_weights->data()) : std::nullopt,
        h_src.size(),
        h_reference_centralities.data(),
        graph_view.number_of_vertices(),
        epsilon,
        eigenvector_usecase.max_iterations);

      auto h_cugraph_centralities = cugraph::test::to_host(handle, d_centralities);

      auto max_centrality =
        *std::max_element(h_cugraph_centralities.begin(), h_cugraph_centralities.end());

      // skip comparison for low Eigenvector Centrality vertices (lowly ranked vertices)
      auto threshold_magnitude = max_centrality * epsilon;

      auto nearly_equal = [epsilon, threshold_magnitude](auto lhs, auto rhs) {
        return std::abs(lhs - rhs) < std::max(std::max(lhs, rhs) * epsilon, threshold_magnitude);
      };

      // FIND DIFFERENCES...
      size_t count_differences{0};
      for (size_t i = 0; i < h_reference_centralities.size(); ++i) {
        if (nearly_equal(h_reference_centralities[i], h_cugraph_centralities[i])) {
        } else {
          if (count_differences < 10) {
            std::cout << "unequal [" << i << "] " << h_reference_centralities[i]
                      << " != " << h_cugraph_centralities[i] << std::endl;
          }
          ++count_differences;
        }
      }

      ASSERT_EQ(count_differences, size_t{0})
        << "Eigenvector centrality values do not match with the reference "
           "values.";
    }
  }
};

using Tests_EigenvectorCentrality_File = Tests_EigenvectorCentrality<cugraph::test::File_Usecase>;
using Tests_EigenvectorCentrality_Rmat = Tests_EigenvectorCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_EigenvectorCentrality_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_EigenvectorCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_EigenvectorCentrality_Rmat, CheckInt32Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_EigenvectorCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_EigenvectorCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EigenvectorCentrality_Usecase{500, false},
                      EigenvectorCentrality_Usecase{500, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_EigenvectorCentrality_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(EigenvectorCentrality_Usecase{500, false},
                      EigenvectorCentrality_Usecase{500, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_EigenvectorCentrality_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(EigenvectorCentrality_Usecase{500, false, false},
                      EigenvectorCentrality_Usecase{500, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
