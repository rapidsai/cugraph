/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_profiler_api.h"
#include "gtest/gtest.h"

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <rmm/exec_policy.hpp>

#include <cugraph/algorithms.hpp>

#include <raft/handle.hpp>

#include <topology/topology.cuh>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <utilities/high_res_timer.hpp>
#include <vector>

namespace topo = cugraph::topology;

struct RandomWalks_Usecase {
  std::string graph_file_full_path{};
  bool test_weighted{false};

  RandomWalks_Usecase(std::string const& graph_file_path, bool test_weighted)
    : test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
};

class Tests_RWSegSort : public ::testing::TestWithParam<RandomWalks_Usecase> {
 public:
  Tests_RWSegSort() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(RandomWalks_Usecase const& target)
  {
    raft::handle_t handle{};

    // debuf info:
    //
    // std::cout << "read graph file: " << configuration.graph_file_full_path << std::endl;
    cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    std::tie(graph, std::ignore) =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        handle, target.graph_file_full_path, target.test_weighted, false);

    size_t num_vertices = graph.number_of_vertices();
    size_t num_edges    = graph.number_of_edges();

    topo::segment_sorter_by_weights_t seg_sort(handle, num_vertices, num_edges);

    auto graph_view = graph.view();

    // NOTE: barring a graph.sort() method,
    // this const_cast<> is the only way to test
    // segmented weight sort for a graph;
    //

    edge_t* offsets = const_cast<edge_t*>(graph_view.local_edge_partition_view().offsets());

    vertex_t* indices = const_cast<vertex_t*>(graph_view.local_edge_partition_view().indices());
    weight_t* values  = const_cast<weight_t*>(*(graph_view.local_edge_partition_view().weights()));

    HighResTimer hr_timer;
    std::string label{};

    label = std::string("Biased RW: CUB Segmented Sort.");
    hr_timer.start(label);
    cudaProfilerStart();

    auto [d_srt_indices, d_srt_weights] = seg_sort(offsets, indices, values);

    cudaProfilerStop();
    hr_timer.stop();

    bool check_seg_sort =
      topo::check_segmented_sort(handle, offsets, d_srt_weights.data(), num_vertices, num_edges);
    ASSERT_TRUE(check_seg_sort);

    try {
      auto runtime = hr_timer.get_average_runtime(label);

      std::cout << "Segmented Sort for Biased RW:\n";

    } catch (std::exception const& ex) {
      std::cerr << ex.what() << '\n';
      return;

    } catch (...) {
      std::cerr << "ERROR: Unknown exception on timer label search." << '\n';
      return;
    }
    hr_timer.display(std::cout);
  }
};

TEST_P(Tests_RWSegSort, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_RWSegSort,
  ::testing::Values(RandomWalks_Usecase("test/datasets/karate.mtx", true),
                    RandomWalks_Usecase("test/datasets/web-Google.mtx", true),
                    RandomWalks_Usecase("test/datasets/ljournal-2008.mtx", true),
                    RandomWalks_Usecase("test/datasets/webbase-1M.mtx", true)));

CUGRAPH_TEST_PROGRAM_MAIN()
