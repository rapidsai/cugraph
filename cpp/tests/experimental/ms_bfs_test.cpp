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

// todo remove
typedef struct MsBfs_Usecase {
  int32_t radius;
  bool test_weighted{false};
};

template <typename input_usecase_t>

  class Tests_MsBfs : public ::testing::TestWithParam<MsBfs_Usecase, input_usecase_t>> {
 public:
  Tests_MsBfs() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(MsBfs_Usecase const& configuration, input_usecase_t const& input_usecase))
  {
    constexpr bool renumber = true;
    using weight_t          = float;
    int n_streams           = std::min(n_sources, static_cast<std::size_t>(128));
    raft::handle_t handle(16);
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());
    std::tie(graph, d_renumber_map_labels) =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, true, renumber);

    // translate
    // auto n_sources = offsets.size();

    auto graph_view = graph.view();

    std::vector<rmm::device_uvector<vertex_t>> d_distances_ref{};
    std::vector<rmm::device_uvector<vertex_t>> d_predecessors_ref{};
    std::vector<std::vector<vertex_t>> h_distances_ref(n_sources);
    std::vector<std::vector<vertex_t>> h_predecessors_ref(n_sources);

    d_distances_ref.reserve(n_sources);
    d_predecessors_ref.reserve(n_sources);
    for (vertex_t i = 0; i < n_sources; i++) {
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
    for (vertex_t i = 0; i < n_sources; i++) {
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
    for (vertex_t i = 0; i < n_sources; i++) {
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

using Tests_BFS_Rmat = Tests_BFS<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_BFS_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

// todo multi components accepting n component parameter
INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MsBfs,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(MsBfs_Usecase{2, false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
