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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#include <experimental/graph.hpp>
#else
#include <experimental/louvain.cuh>
#endif

#include <algorithms.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

typedef struct Louvain_Usecase_t {
  std::string graph_file_full_path{};
  bool test_weighted{false};

  Louvain_Usecase_t(std::string const& graph_file_path, bool test_weighted)
    : test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} Louvain_Usecase;

class Tests_Louvain : public ::testing::TestWithParam<Louvain_Usecase> {
 public:
  Tests_Louvain() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(Louvain_Usecase const& configuration)
  {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    CUGRAPH_FAIL("Louvain not supported on Pascal and older architectures");
#else
    raft::handle_t handle{};

    std::cout << "read graph file: " << configuration.graph_file_full_path << std::endl;

    auto graph =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false>(
        handle, configuration.graph_file_full_path, configuration.test_weighted);

    auto graph_view = graph.view();

    louvain(graph_view);
#endif
  }

  template <typename graph_t>
  void louvain(graph_t const& graph_view)
  {
    using vertex_t = typename graph_t::vertex_type;
    using weight_t = typename graph_t::weight_type;

    raft::handle_t handle{};

    rmm::device_vector<vertex_t> clustering_v(graph_view.get_number_of_local_vertices());
    size_t level;
    weight_t modularity;

    std::tie(level, modularity) =
      cugraph::louvain(handle, graph_view, clustering_v.data().get(), size_t{100}, weight_t{1});

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::cout << "level = " << level << std::endl;
    std::cout << "modularity = " << modularity << std::endl;
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_Louvain, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_Louvain,
                        ::testing::Values(Louvain_Usecase("test/datasets/karate.mtx", true)
#if 0
			,
                                          Louvain_Usecase("test/datasets/web-Google.mtx", true),
                                          Louvain_Usecase("test/datasets/ljournal-2008.mtx", true),
                                          Louvain_Usecase("test/datasets/webbase-1M.mtx", true)
#endif
                                            ));

CUGRAPH_TEST_PROGRAM_MAIN()
