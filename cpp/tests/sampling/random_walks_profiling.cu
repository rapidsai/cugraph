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

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_timer.hpp>
#include <utilities/test_utilities.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/random.h>

#include <algorithms.hpp>
#include <graph.hpp>
#include <sampling/random_walks.cuh>

#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

namespace {  // anonym.
template <typename vertex_t, typename index_t>
void fill_start(raft::handle_t const& handle,
                rmm::device_uvector<vertex_t>& d_start,
                index_t num_vertices)
{
  index_t num_paths = d_start.size();

  thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    thrust::make_counting_iterator<index_t>(0),
                    thrust::make_counting_iterator<index_t>(num_paths),

                    d_start.begin(),
                    [num_vertices] __device__(auto indx) { return indx % num_vertices; });
}
}  // namespace

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

class Tests_RandomWalksProfiling : public ::testing::TestWithParam<RandomWalks_Usecase> {
 public:
  Tests_RandomWalksProfiling() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(RandomWalks_Usecase const& configuration)
  {
    raft::handle_t handle{};

    // debuf info:
    //
    // std::cout << "read graph file: " << configuration.graph_file_full_path << std::endl;

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    std::tie(graph, std::ignore) =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        handle, configuration.graph_file_full_path, configuration.test_weighted, false);

    auto graph_view = graph.view();

    // call random_walks:
    std::vector<edge_t> v_np{1, 10, 100};
    for (auto&& num_paths : v_np) { time_random_walks(graph_view, num_paths); }
  }

  template <typename graph_vt>
  void time_random_walks(graph_vt const& graph_view, typename graph_vt::edge_type num_paths)
  {
    using vertex_t = typename graph_vt::vertex_type;
    using edge_t   = typename graph_vt::edge_type;
    using weight_t = typename graph_vt::weight_type;

    raft::handle_t handle{};
    rmm::device_uvector<vertex_t> d_start(num_paths, handle.get_stream());

    vertex_t num_vertices = graph_view.get_number_of_vertices();
    fill_start(handle, d_start, num_vertices);

    // 0-copy const device view:
    //
    cugraph::experimental::detail::device_const_vector_view<vertex_t, edge_t> d_start_view{
      d_start.data(), num_paths};

    edge_t max_depth{10};

    HighResTimer hr_timer;
    std::string label("RandomWalks");
    hr_timer.start(label);
    cudaProfilerStart();
    auto ret_tuple =
      cugraph::experimental::detail::random_walks_impl(handle, graph_view, d_start_view, max_depth);
    cudaProfilerStop();
    hr_timer.stop();
    try {
      auto runtime = hr_timer.get_average_runtime(label);

      std::cout << "RW for num_paths: " << num_paths
                << ", runtime [ms] / path: " << runtime / num_paths << ":\n";

    } catch (std::exception const& ex) {
      std::cerr << ex.what() << '\n';
      ASSERT_TRUE(false);  // test has failed.

    } catch (...) {
      std::cerr << "ERROR: Unknown exception on timer label search." << '\n';
      ASSERT_TRUE(false);  // test has failed.
    }
    hr_timer.display(std::cout);
  }
};

TEST_P(Tests_RandomWalksProfiling, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_RandomWalksProfiling,
                        ::testing::Values(RandomWalks_Usecase("test/datasets/karate.mtx", true)));

CUGRAPH_TEST_PROGRAM_MAIN()
