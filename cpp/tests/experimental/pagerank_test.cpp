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

#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <graph.hpp>

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
void pagerank_reference(edge_t* offsets,
                        vertex_t* indices,
                        weight_t* weights,
                        vertex_t* personalization_vertices,
                        result_t* personalization_values,
                        result_t* pageranks,
                        vertex_t num_vertices,
                        vertex_t personalization_vector_size,
                        result_t alpha,
                        result_t epsilon,
                        size_t max_iterations,
                        bool has_initial_guess)
{
  if (num_vertices == 0) { return; }

  if (has_initial_guess) {
    auto sum = std::accumulate(pageranks, pageranks + num_vertices, result_t{0.0});
    ASSERT_TRUE(sum > 0.0);
    std::for_each(pageranks, pageranks + num_vertices, [sum](auto& val) { val /= sum; });
  } else {
    std::for_each(pageranks, pageranks + num_vertices, [num_vertices](auto& val) {
      val = result_t{1.0} / static_cast<result_t>(num_vertices);
    });
  }

  if (personalization_vertices != nullptr) {
    auto sum = std::accumulate(
      personalization_values, personalization_values + personalization_vector_size, result_t{0.0});
    ASSERT_TRUE(sum > 0.0);
    std::for_each(personalization_values,
                  personalization_values + personalization_vector_size,
                  [sum](auto& val) { val /= sum; });
  }

  std::vector<weight_t> out_weight_sums(num_vertices, result_t{0.0});
  for (vertex_t i = 0; i < num_vertices; ++i) {
    for (auto j = *(offsets + i); j < *(offsets + i + 1); ++j) {
      auto nbr = indices[j];
      auto w   = weights != nullptr ? weights[j] : 1.0;
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
        auto w   = weights != nullptr ? weights[j] : result_t{1.0};
        pageranks[i] += alpha * old_pageranks[nbr] * (w / out_weight_sums[nbr]);
      }
      if (personalization_vertices == nullptr) {
        pageranks[i] += (dangling_sum + (1.0 - alpha)) / static_cast<result_t>(num_vertices);
      }
    }
    if (personalization_vertices != nullptr) {
      for (vertex_t i = 0; i < personalization_vector_size; ++i) {
        auto v = personalization_vertices[i];
        pageranks[v] += (dangling_sum + (1.0 - alpha)) * personalization_values[i];
      }
    }
    result_t diff_sum{0.0};
    for (vertex_t i = 0; i < num_vertices; ++i) {
      diff_sum += fabs(pageranks[i] - old_pageranks[i]);
    }
    if (diff_sum < static_cast<result_t>(num_vertices) * epsilon) { break; }
    iter++;
    ASSERT_TRUE(iter < max_iterations);
  }

  return;
}

typedef struct PageRank_Usecase_t {
  std::string graph_file_path_;
  std::string graph_file_full_path_;

  PageRank_Usecase_t(std::string const& graph_file_path) : graph_file_path_(graph_file_path)
  {
    if ((graph_file_path_.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path_ = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path_;
    } else {
      graph_file_full_path_ = graph_file_path_;
    }
  };
} PageRank_Usecase;

class Tests_PageRank : public ::testing::TestWithParam<PageRank_Usecase> {
 public:
  Tests_PageRank() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(PageRank_Usecase const& configuration)
  {
    // FIXME: better read using a uiltity function (for CSR, we have generate_graph_csr_from_mm)
    MM_typecode mc{};
    vertex_t m{};
    vertex_t k{};
    edge_t nnz{};

    FILE* file = fopen(configuration.graph_file_full_path_.c_str(), "r");
    ASSERT_NE(file, nullptr) << "fopen (" << configuration.graph_file_full_path_ << ") failure.";

    ASSERT_EQ(cugraph::test::mm_properties<int>(file, 1, &mc, &m, &k, &nnz), 0)
      << "could not read Matrix Market file properties\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    std::vector<vertex_t> rows(nnz, vertex_t{0});
    std::vector<vertex_t> cols(nnz, vertex_t{0});
    std::vector<weight_t> weights(nnz, weight_t{0.0});

    ASSERT_EQ((cugraph::test::mm_to_coo<vertex_t, weight_t>(
                file, 1, nnz, rows.data(), cols.data(), weights.data(), nullptr)),
              0)
      << "could not read matrix data\n";
    ASSERT_EQ(fclose(file), 0);

    // FIXME: this is more of a hack than a proper implementation.
    cugraph::GraphCOOView<vertex_t, edge_t, weight_t> coo_graph(
      cols.data(), rows.data(), weights.data(), m, nnz);
    auto p_csc_graph = cugraph::coo_to_csr(coo_graph);
    cugraph::GraphCSCView<vertex_t, edge_t, weight_t> csc_graph_view(
      p_csc_graph->offsets(),
      p_csc_graph->indices(),
      p_csc_graph->edge_data(),
      p_csc_graph->number_of_vertices(),
      p_csc_graph->number_of_edges());

    std::vector<edge_t> h_offsets(csc_graph_view.number_of_vertices + 1);
    std::vector<vertex_t> h_indices(csc_graph_view.number_of_edges);
    std::vector<weight_t> h_weights{};
    std::vector<result_t> h_reference_pageranks(csc_graph_view.number_of_vertices);

    CUDA_TRY(cudaMemcpy(h_offsets.data(),
                        csc_graph_view.offsets,
                        sizeof(edge_t) * h_offsets.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_indices.data(),
                        csc_graph_view.indices,
                        sizeof(vertex_t) * h_indices.size(),
                        cudaMemcpyDeviceToHost));
    if (csc_graph_view.edge_data != nullptr) {
      h_weights.assign(csc_graph_view.number_of_edges, weight_t{0.0});
      CUDA_TRY(cudaMemcpy(h_weights.data(),
                          csc_graph_view.edge_data,
                          sizeof(weight_t) * h_weights.size(),
                          cudaMemcpyDeviceToHost));
    }

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    pagerank_reference(h_offsets.data(),
                       h_indices.data(),
                       h_weights.size() > 0 ? h_weights.data() : static_cast<weight_t*>(nullptr),
                       static_cast<vertex_t*>(nullptr),
                       static_cast<result_t*>(nullptr),
                       h_reference_pageranks.data(),
                       csc_graph_view.number_of_vertices,
                       vertex_t{0},
                       alpha,
                       epsilon,
                       std::numeric_limits<size_t>::max(),
                       false);

    raft::handle_t handle{};

    rmm::device_uvector<result_t> d_pageranks(csc_graph_view.number_of_vertices,
                                              handle.get_stream());

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    cugraph::experimental::pagerank(handle,
                                    csc_graph_view,
                                    static_cast<weight_t*>(nullptr),
                                    static_cast<vertex_t*>(nullptr),
                                    static_cast<result_t*>(nullptr),
                                    vertex_t{0},
                                    d_pageranks.begin(),
                                    alpha,
                                    epsilon,
                                    std::numeric_limits<size_t>::max(),
                                    false,
                                    false);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<result_t> h_cugraph_pageranks(csc_graph_view.number_of_vertices);

    CUDA_TRY(cudaMemcpy(h_cugraph_pageranks.data(),
                        d_pageranks.data(),
                        sizeof(result_t) * d_pageranks.size(),
                        cudaMemcpyDeviceToHost));

    auto nearly_equal = [epsilon](auto lhs, auto rhs) { return std::fabs(lhs - rhs) < epsilon; };

    ASSERT_TRUE(std::equal(h_reference_pageranks.begin(),
                           h_reference_pageranks.end(),
                           h_cugraph_pageranks.begin(),
                           nearly_equal))
      << "PageRank values do not match with the reference values.";
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_PageRank, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_PageRank,
                        ::testing::Values(PageRank_Usecase("test/datasets/karate.mtx"),
                                          PageRank_Usecase("test/datasets/web-Google.mtx"),
                                          PageRank_Usecase("test/datasets/ljournal-2008.mtx"),
                                          PageRank_Usecase("test/datasets/webbase-1M.mtx")));

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}
