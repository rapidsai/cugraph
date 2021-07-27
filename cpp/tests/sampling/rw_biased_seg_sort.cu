/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>

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
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    std::tie(graph, std::ignore) =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        handle, target.graph_file_full_path, target.test_weighted, false);

    size_t num_vertices = graph.get_number_of_vertices();
    size_t num_edges    = graph.get_number_of_edges();

    topo::segment_sorter_by_weights_t seg_sort(handle, num_vertices, num_edges);

    HighResTimer hr_timer;
    std::string label{};

    label = std::string("Biased RW: CUB Segmented Sort.");
    hr_timer.start(label);
    cudaProfilerStart();

    graph.sort(seg_sort);

    cudaProfilerStop();
    hr_timer.stop();

    auto graph_view = graph.view();

    edge_t const* offsets   = graph_view.get_matrix_partition_view().get_offsets();
    vertex_t const* indices = graph_view.get_matrix_partition_view().get_indices();
    weight_t const* values  = *(graph_view.get_matrix_partition_view().get_weights());

    bool check_seg_sort =
      topo::check_segmented_sort(handle, offsets, values, num_vertices, num_edges);
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
