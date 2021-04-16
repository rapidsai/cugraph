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

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_timer.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <experimental/graph.hpp>
#include <experimental/graph_view.hpp>
#include <graph.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cuda_profiler_api.h>
#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

typedef struct MsBfs_Usecase_t {
  std::string graph_file_full_path{};
  std::vector<int32_t> sources{};
  int32_t radius;
  bool test_weighted{false};

  MsBfs_Usecase_t(std::string const& graph_file_path,
                  std::vector<int32_t> const& sources,
                  int32_t radius,
                  bool test_weighted)
    : sources(sources), radius(radius), test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} MsBfs_Usecase;

class Tests_MsBfs : public ::testing::TestWithParam<MsBfs_Usecase> {
 public:
  Tests_MsBfs() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(MsBfs_Usecase const& configuration)
  {
    auto n_seeds  = configuration.sources.size();
    int n_streams = std::min(n_seeds, static_cast<std::size_t>(128));
    raft::handle_t handle(n_streams);

    // TODO RMAT multi component and ospc seeds

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> graph(
      handle);
    std::tie(graph, std::ignore) = cugraph::test::
      read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, configuration.graph_file_full_path, configuration.test_weighted, false);
    auto graph_view = graph.view();

    std::vector<rmm::device_uvector<vertex_t>> d_distances_ref{};
    std::vector<rmm::device_uvector<vertex_t>> d_predecessors_ref{};
    std::vector<std::vector<vertex_t>> h_distances_ref(n_seeds);
    std::vector<std::vector<vertex_t>> h_predecessors_ref(n_seeds);

    d_distances_ref.reserve(n_seeds);
    d_predecessors_ref.reserve(n_seeds);
    for (vertex_t i = 0; i < n_seeds; i++) {
      rmm::device_uvector<vertex_t> tmp_distances(graph_view.get_number_of_vertices(),
                                                  handle.get_internal_stream_view(i));
      rmm::device_uvector<vertex_t> tmp_predecessors(graph_view.get_number_of_vertices(),
                                                     handle.get_internal_stream_view(i));

      d_distances_ref.push_back(std::move(tmp_distances));
      d_predecessors_ref.push_back(std::move(tmp_predecessors));
    }

    // warm up
    vertex_t source = configuration.sources[0];
    cugraph::experimental::bfs(
      handle, graph_view, d_distances_ref[0].begin(), d_predecessors_ref[0].begin(), &source, 1);

    // one by one
    HighResTimer hr_timer;
    hr_timer.start("bfs");
    cudaProfilerStart();
    for (vertex_t i = 0; i < n_seeds; i++) {
      source = configuration.sources[i];
      cugraph::experimental::bfs(
        handle, graph_view, d_distances_ref[i].begin(), d_predecessors_ref[i].begin(), &source, 1);
    }
    cudaProfilerStop();
    hr_timer.stop();
    hr_timer.display(std::cout);

    // ms
    rmm::device_uvector<vertex_t> d_distances(graph_view.get_number_of_vertices(),
                                              handle.get_internal_stream_view(i));
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.get_number_of_vertices(),
                                                 handle.get_internal_stream_view(i));
    rmm::device_uvector<vertex_t> d_sources_v(configuration.sources.size(), handle.get_stream());
    raft::copy(d_sources, configuration.sources, configuration.sources.size(), handle.get_stream());

    hr_timer.start("msbfs");
    cudaProfilerStart();
    cugraph::experimental::bfs(handle,
                               graph_view,
                               d_distances.begin(),
                               d_predecessors.begin(),
                               d_sources,
                               configuration.sources.size());

    cudaProfilerStop();
    hr_timer.stop();
    hr_timer.display(std::cout);

    // checksum
    vertex_t ref_sum = 0;
    for (vertex_t i = 0; i < n_seeds; i++) {
      thrust::replace(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      d_distances_ref.begin(),
                      d_distances_ref.end(),
                      std::numeric_limits<vertex_t>::max(),
                      0);
      ref_sum += thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                d_distances_ref.begin(),
                                d_distances_ref.end(),
                                0);
    }
    thrust::replace(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    d_distances.begin(),
                    d_distances.end(),
                    std::numeric_limits<vertex_t>::max(),
                    0);
    vertex_t ms_sum = thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                     d_distances.begin(),
                                     d_distances.end(),
                                     0);
    ASSERT_TRUE(ref_sum > 0);
    ASSERT_TRUE(ref_sum < std::numeric_limits<vertex_t>::max());
    ASSERT_TRUE(ref_sum == ms_sum);
  }
};

TEST_P(Tests_MsBfs, CheckInt32Int32FloatUntransposed)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
}
// TODO
CUGRAPH_TEST_PROGRAM_MAIN()
