/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include "cuda_profiler_api.h"
#include "gtest/gtest.h"

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/random.h>

#include <algorithms.hpp>
#include <experimental/random_walks.cuh>
#include <graph.hpp>

#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <utilities/high_res_timer.hpp>
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

class Tests_RandomWalks : public ::testing::TestWithParam<RandomWalks_Usecase> {
 public:
  Tests_RandomWalks() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(RandomWalks_Usecase const& configuration)
  {
    raft::handle_t handle{};

    std::cout << "read graph file: " << configuration.graph_file_full_path << std::endl;

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    std::tie(graph, std::ignore) =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        handle, configuration.graph_file_full_path, configuration.test_weighted, false);

    auto graph_view = graph.view();

    // call random_walks:
    start_random_walks(graph_view);
  }

  template <typename graph_t>
  void start_random_walks(graph_t const& graph)
  {
    using vertex_t = typename graph_t::vertex_type;
    using weight_t = typename graph_t::weight_type;

    raft::handle_t handle{};
    vertex_t num_paths = 10;
    rmm::device_uvector<vertex_t> d_start(num_paths, handle.get_stream());

    vertex_t num_vertices = graph.get_number_of_vertices();
    fill_start(handle, d_start, num_vertices);

    vertex_t max_d{10};

    auto ret_tuple = cugraph::experimental::random_walks(handle, graph, d_start, max_d);
  }
};

TEST_P(Tests_RandomWalks, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_RandomWalks,
                        ::testing::Values(RandomWalks_Usecase("test/datasets/karate.mtx", true)
#if 0
			,
                                          RandomWalks_Usecase("test/datasets/web-Google.mtx", true),
                                          RandomWalks_Usecase("test/datasets/ljournal-2008.mtx", true),
                                          RandomWalks_Usecase("test/datasets/webbase-1M.mtx", true)
#endif
                                            ));

CUGRAPH_TEST_PROGRAM_MAIN()

/*
struct RandomWalksTest : public ::testing::Test {
};

TEST_F(RandomWalksTest, CorrectInit)
{
  raft::handle_t handle{};

  ASSERT_TRUE(true);
}
*/
