/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <algorithms.hpp>
#include <experimental/graph.hpp>
#include <experimental/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void katz_centrality_reference(edge_t* offsets,
                               vertex_t* indices,
                               weight_t* weights,
                               result_t* betas,
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
        auto w   = weights != nullptr ? weights[j] : result_t{1.0};
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

typedef struct KatzCentrality_Usecase_t {
  std::string graph_file_full_path{};
  bool test_weighted{false};

  KatzCentrality_Usecase_t(std::string const& graph_file_path, bool test_weighted)
    : test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} KatzCentrality_Usecase;

class Tests_KatzCentrality : public ::testing::TestWithParam<KatzCentrality_Usecase> {
 public:
  Tests_KatzCentrality() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(KatzCentrality_Usecase const& configuration)
  {
    raft::handle_t handle{};

    auto graph =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, true>(
        handle, configuration.graph_file_full_path, configuration.test_weighted);
    auto graph_view = graph.view();

    std::vector<edge_t> h_offsets(graph_view.get_number_of_vertices() + 1);
    std::vector<vertex_t> h_indices(graph_view.get_number_of_edges());
    std::vector<weight_t> h_weights{};
    raft::update_host(h_offsets.data(),
                      graph_view.offsets(),
                      graph_view.get_number_of_vertices() + 1,
                      handle.get_stream());
    raft::update_host(h_indices.data(),
                      graph_view.indices(),
                      graph_view.get_number_of_edges(),
                      handle.get_stream());
    if (graph_view.is_weighted()) {
      h_weights.assign(graph_view.get_number_of_edges(), weight_t{0.0});
      raft::update_host(h_weights.data(),
                        graph_view.weights(),
                        graph_view.get_number_of_edges(),
                        handle.get_stream());
    }
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    std::vector<result_t> h_reference_katz_centralities(graph_view.get_number_of_vertices());

    std::vector<edge_t> tmps(h_offsets.size());
    std::adjacent_difference(h_offsets.begin(), h_offsets.end(), tmps.begin());
    auto max_it = std::max_element(tmps.begin(), tmps.end());

    result_t const alpha = result_t{1.0} / static_cast<result_t>(*max_it + 1);
    result_t constexpr beta{1.0};
    result_t constexpr epsilon{1e-6};

    katz_centrality_reference(
      h_offsets.data(),
      h_indices.data(),
      h_weights.size() > 0 ? h_weights.data() : static_cast<weight_t*>(nullptr),
      static_cast<result_t*>(nullptr),
      h_reference_katz_centralities.data(),
      graph_view.get_number_of_vertices(),
      alpha,
      beta,
      epsilon,
      std::numeric_limits<size_t>::max(),
      false,
      true);

    rmm::device_uvector<result_t> d_katz_centralities(graph_view.get_number_of_vertices(),
                                                      handle.get_stream());

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    cugraph::experimental::katz_centrality(handle,
                                           graph_view,
                                           static_cast<result_t*>(nullptr),
                                           d_katz_centralities.begin(),
                                           alpha,
                                           beta,
                                           epsilon,
                                           std::numeric_limits<size_t>::max(),
                                           false,
                                           true,
                                           false);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<result_t> h_cugraph_katz_centralities(graph_view.get_number_of_vertices());

    raft::update_host(h_cugraph_katz_centralities.data(),
                      d_katz_centralities.data(),
                      d_katz_centralities.size(),
                      handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    auto threshold_ratio = 1e-3;
    auto threshold_magnitude =
      (epsilon / static_cast<result_t>(graph_view.get_number_of_vertices())) * threshold_ratio;
    auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
      auto diff = std::abs(lhs - rhs);
      return (diff < std::max(lhs, rhs) * threshold_ratio) || (diff < threshold_magnitude);
    };

    ASSERT_TRUE(std::equal(h_reference_katz_centralities.begin(),
                           h_reference_katz_centralities.end(),
                           h_cugraph_katz_centralities.begin(),
                           nearly_equal))
      << "Katz centrality values do not match with the reference values.";
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_KatzCentrality, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_KatzCentrality,
  ::testing::Values(KatzCentrality_Usecase("test/datasets/karate.mtx", false),
                    KatzCentrality_Usecase("test/datasets/karate.mtx", true),
                    KatzCentrality_Usecase("test/datasets/web-Google.mtx", false),
                    KatzCentrality_Usecase("test/datasets/web-Google.mtx", true),
                    KatzCentrality_Usecase("test/datasets/ljournal-2008.mtx", false),
                    KatzCentrality_Usecase("test/datasets/ljournal-2008.mtx", true),
                    KatzCentrality_Usecase("test/datasets/webbase-1M.mtx", false),
                    KatzCentrality_Usecase("test/datasets/webbase-1M.mtx", true)));

CUGRAPH_TEST_PROGRAM_MAIN()
