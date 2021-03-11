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
    uint64_t seed{101201301401};
    raft::random::Rng rnd(seed);  // TODO: have it passed as argument from the test instance;

    // call random_walks:
    start_random_walks(graph_view, rnd);
  }

  template <typename graph_t, typename random_eng_t>
  void start_random_walks(graph_t const& graph, random_eng_t& rnd_eng)
  {
    using vertex_t = typename graph_t::vertex_type;
    using weight_t = typename graph_t::weight_type;

    raft::handle_t handle{};
    std::vector<vertex_t> empty_vec{};  // for now...
    size_t max_d{10};

    auto ret_tuple =
      cugraph::experimental::detail::random_walks(handle, graph, empty_vec, max_d, rnd_eng);
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
