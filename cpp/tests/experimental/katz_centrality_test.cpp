/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <utilities/renumber_utilities.hpp>
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

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 0;

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void katz_centrality_reference(edge_t const* offsets,
                               vertex_t const* indices,
                               weight_t const* weights,
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
  cugraph::test::input_graph_specifier_t input_graph_specifier{};

  bool test_weighted{false};
  bool check_correctness{false};

  KatzCentrality_Usecase_t(std::string const& graph_file_path,
                           bool test_weighted,
                           bool check_correctness = true)
    : test_weighted(test_weighted), check_correctness(check_correctness)
  {
    std::string graph_file_full_path{};
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
    input_graph_specifier.tag = cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH;
    input_graph_specifier.graph_file_full_path = graph_file_full_path;
  };

  KatzCentrality_Usecase_t(cugraph::test::rmat_params_t rmat_params,
                           bool test_weighted,
                           bool check_correctness = true)
    : test_weighted(test_weighted), check_correctness(check_correctness)
  {
    input_graph_specifier.tag         = cugraph::test::input_graph_specifier_t::RMAT_PARAMS;
    input_graph_specifier.rmat_params = rmat_params;
  }
} KatzCentrality_Usecase;

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, false>,
           rmm::device_uvector<vertex_t>>
read_graph(raft::handle_t const& handle, KatzCentrality_Usecase const& configuration, bool renumber)
{
  return configuration.input_graph_specifier.tag ==
             cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH
           ? cugraph::test::
               read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, true, false>(
                 handle,
                 configuration.input_graph_specifier.graph_file_full_path,
                 configuration.test_weighted,
                 renumber)
           : cugraph::test::
               generate_graph_from_rmat_params<vertex_t, edge_t, weight_t, true, false>(
                 handle,
                 configuration.input_graph_specifier.rmat_params.scale,
                 configuration.input_graph_specifier.rmat_params.edge_factor,
                 configuration.input_graph_specifier.rmat_params.a,
                 configuration.input_graph_specifier.rmat_params.b,
                 configuration.input_graph_specifier.rmat_params.c,
                 configuration.input_graph_specifier.rmat_params.seed,
                 configuration.input_graph_specifier.rmat_params.undirected,
                 configuration.input_graph_specifier.rmat_params.scramble_vertex_ids,
                 configuration.test_weighted,
                 renumber,
                 std::vector<size_t>{0},
                 size_t{1});
}

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
    constexpr bool renumber = true;

    raft::handle_t handle{};

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, false> graph(handle);
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());
    std::tie(graph, d_renumber_map_labels) =
      read_graph<vertex_t, edge_t, weight_t>(handle, configuration, renumber);
    auto graph_view = graph.view();

    auto degrees = graph_view.compute_in_degrees(handle);
    std::vector<edge_t> h_degrees(degrees.size());
    raft::update_host(h_degrees.data(), degrees.data(), degrees.size(), handle.get_stream());
    handle.get_stream_view().synchronize();
    auto max_it = std::max_element(h_degrees.begin(), h_degrees.end());

    result_t const alpha = result_t{1.0} / static_cast<result_t>(*max_it + 1);
    result_t constexpr beta{1.0};
    result_t constexpr epsilon{1e-6};

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

    if (configuration.check_correctness) {
      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, false> unrenumbered_graph(
        handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore) =
          read_graph<vertex_t, edge_t, weight_t>(handle, configuration, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets(unrenumbered_graph_view.get_number_of_vertices() + 1);
      std::vector<vertex_t> h_indices(unrenumbered_graph_view.get_number_of_edges());
      std::vector<weight_t> h_weights{};
      raft::update_host(h_offsets.data(),
                        unrenumbered_graph_view.offsets(),
                        unrenumbered_graph_view.get_number_of_vertices() + 1,
                        handle.get_stream());
      raft::update_host(h_indices.data(),
                        unrenumbered_graph_view.indices(),
                        unrenumbered_graph_view.get_number_of_edges(),
                        handle.get_stream());
      if (unrenumbered_graph_view.is_weighted()) {
        h_weights.assign(unrenumbered_graph_view.get_number_of_edges(), weight_t{0.0});
        raft::update_host(h_weights.data(),
                          unrenumbered_graph_view.weights(),
                          unrenumbered_graph_view.get_number_of_edges(),
                          handle.get_stream());
      }

      handle.get_stream_view().synchronize();

      std::vector<result_t> h_reference_katz_centralities(
        unrenumbered_graph_view.get_number_of_vertices());

      katz_centrality_reference(
        h_offsets.data(),
        h_indices.data(),
        h_weights.size() > 0 ? h_weights.data() : static_cast<weight_t*>(nullptr),
        static_cast<result_t*>(nullptr),
        h_reference_katz_centralities.data(),
        unrenumbered_graph_view.get_number_of_vertices(),
        alpha,
        beta,
        epsilon,
        std::numeric_limits<size_t>::max(),
        false,
        true);

      std::vector<result_t> h_cugraph_katz_centralities(graph_view.get_number_of_vertices());
      if (renumber) {
        auto d_unrenumbered_katz_centralities =
          cugraph::test::sort_values_by_key(handle,
                                            d_renumber_map_labels.data(),
                                            d_katz_centralities.data(),
                                            d_renumber_map_labels.size());
        raft::update_host(h_cugraph_katz_centralities.data(),
                          d_unrenumbered_katz_centralities.data(),
                          d_unrenumbered_katz_centralities.size(),
                          handle.get_stream());
      } else {
        raft::update_host(h_cugraph_katz_centralities.data(),
                          d_katz_centralities.data(),
                          d_katz_centralities.size(),
                          handle.get_stream());
      }

      handle.get_stream_view().synchronize();

      auto threshold_ratio = 1e-3;
      auto threshold_magnitude =
        (1.0 / static_cast<result_t>(graph_view.get_number_of_vertices())) *
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

// FIXME: add tests for type combinations
TEST_P(Tests_KatzCentrality, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_KatzCentrality,
  ::testing::Values(
    // enable correctness checks
    KatzCentrality_Usecase("test/datasets/karate.mtx", false),
    KatzCentrality_Usecase("test/datasets/karate.mtx", true),
    KatzCentrality_Usecase("test/datasets/web-Google.mtx", false),
    KatzCentrality_Usecase("test/datasets/web-Google.mtx", true),
    KatzCentrality_Usecase("test/datasets/ljournal-2008.mtx", false),
    KatzCentrality_Usecase("test/datasets/ljournal-2008.mtx", true),
    KatzCentrality_Usecase("test/datasets/webbase-1M.mtx", false),
    KatzCentrality_Usecase("test/datasets/webbase-1M.mtx", true),
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                           false),
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                           true),
    // disable correctness checks for large graphs
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{25, 32, 0.57, 0.19, 0.19, 0, false, false},
                           false,
                           false),
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{25, 32, 0.57, 0.19, 0.19, 0, false, false},
                           true,
                           false)));

CUGRAPH_TEST_PROGRAM_MAIN()
