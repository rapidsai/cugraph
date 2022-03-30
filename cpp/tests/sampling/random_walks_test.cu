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
#include <thrust/random.h>

#include <cugraph/algorithms.hpp>
#include <sampling/random_walks.cuh>

#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include "random_walks_utils.cuh"

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <utilities/high_res_timer.hpp>
#include <vector>

namespace {  // anonym.
template <typename vertex_t, typename index_t>
void fill_start(raft::handle_t const& handle,
                rmm::device_uvector<vertex_t>& d_start,
                index_t num_vertices)
{
  index_t num_paths = d_start.size();

  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<index_t>(0),
                    thrust::make_counting_iterator<index_t>(num_paths),

                    d_start.begin(),
                    [num_vertices] __device__(auto indx) { return indx % num_vertices; });
}
}  // namespace

namespace impl_details = cugraph::detail;

enum class traversal_id_t : int { HORIZONTAL = 0, VERTICAL };

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

class Tests_RandomWalks
  : public ::testing::TestWithParam<std::tuple<traversal_id_t, int, RandomWalks_Usecase>> {
 public:
  Tests_RandomWalks() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<traversal_id_t, int, RandomWalks_Usecase> const& configuration)
  {
    raft::handle_t handle{};

    // debuf info:
    //
    // std::cout << "read graph file: " << configuration.graph_file_full_path << std::endl;

    traversal_id_t trv_id = std::get<0>(configuration);
    int sampling_id       = std::get<1>(configuration);
    auto const& target    = std::get<2>(configuration);

    cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    std::tie(graph, std::ignore) =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        handle, target.graph_file_full_path, target.test_weighted, false);

    auto graph_view = graph.view();

    // call random_walks:
    start_random_walks(handle, graph_view, trv_id, sampling_id);
  }

  template <typename graph_vt>
  void start_random_walks(raft::handle_t const& handle,
                          graph_vt const& graph_view,
                          traversal_id_t trv_id,
                          int sampling_id)
  {
    using vertex_t = typename graph_vt::vertex_type;
    using edge_t   = typename graph_vt::edge_type;
    using weight_t = typename graph_vt::weight_type;
    using real_t   = float;

    edge_t num_paths = 10;
    rmm::device_uvector<vertex_t> d_start(num_paths, handle.get_stream());

    vertex_t num_vertices = graph_view.number_of_vertices();
    fill_start(handle, d_start, num_vertices);

    // 0-copy const device view:
    //
    impl_details::device_const_vector_view<vertex_t, edge_t> d_start_view{d_start.data(),
                                                                          num_paths};

    edge_t max_depth{10};

    weight_t p{4};
    weight_t q{8};

    if (trv_id == traversal_id_t::HORIZONTAL) {
      // `node2vec` without alpha buffer:
      //
      if (sampling_id == 2) {
        auto ret_tuple = cugraph::random_walks(
          handle,
          graph_view,
          d_start_view.begin(),
          num_paths,
          max_depth,
          false,
          std::make_unique<cugraph::sampling_params_t>(sampling_id, p, q, false));

        // check results:
        //
        bool test_all_paths = cugraph::test::host_check_rw_paths(handle,
                                                                 graph_view,
                                                                 std::get<0>(ret_tuple),
                                                                 std::get<1>(ret_tuple),
                                                                 std::get<2>(ret_tuple));

        ASSERT_TRUE(test_all_paths);
      }

      // the alpha buffer case should also be tested for `node2vec`
      // and for the others is irrelevant, so this block is necessary
      // for any sampling method:
      //
      {
        auto ret_tuple = cugraph::random_walks(
          handle,
          graph_view,
          d_start_view.begin(),
          num_paths,
          max_depth,
          false,
          std::make_unique<cugraph::sampling_params_t>(sampling_id, p, q, true));

        // check results:
        //
        bool test_all_paths = cugraph::test::host_check_rw_paths(handle,
                                                                 graph_view,
                                                                 std::get<0>(ret_tuple),
                                                                 std::get<1>(ret_tuple),
                                                                 std::get<2>(ret_tuple));

        ASSERT_TRUE(test_all_paths);
      }
    } else {  // VERTICAL: needs to be force-called via detail
      if (sampling_id == 0) {
        impl_details::uniform_selector_t<graph_vt, real_t> selector{handle, graph_view, real_t{0}};

        auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                         decltype(selector),
                                                         impl_details::vertical_traversal_t>(
          handle,  // required to prevent clang-format to separate functin name from its namespace
          graph_view,
          d_start_view,
          max_depth,
          selector);

        // check results:
        //
        bool test_all_paths = cugraph::test::host_check_rw_paths(handle,
                                                                 graph_view,
                                                                 std::get<0>(ret_tuple),
                                                                 std::get<1>(ret_tuple),
                                                                 std::get<2>(ret_tuple));

        if (!test_all_paths)
          std::cout << "starting seed on failure: " << std::get<3>(ret_tuple) << '\n';

        ASSERT_TRUE(test_all_paths);
      } else if (sampling_id == 1) {
        impl_details::biased_selector_t<graph_vt, real_t> selector{handle, graph_view, real_t{0}};

        auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                         decltype(selector),
                                                         impl_details::vertical_traversal_t>(
          handle,  // required to prevent clang-format to separate functin name from its namespace
          graph_view,
          d_start_view,
          max_depth,
          selector);

        // check results:
        //
        bool test_all_paths = cugraph::test::host_check_rw_paths(handle,
                                                                 graph_view,
                                                                 std::get<0>(ret_tuple),
                                                                 std::get<1>(ret_tuple),
                                                                 std::get<2>(ret_tuple));

        if (!test_all_paths)
          std::cout << "starting seed on failure: " << std::get<3>(ret_tuple) << '\n';

        ASSERT_TRUE(test_all_paths);
      } else {
        impl_details::node2vec_selector_t<graph_vt, real_t> selector{
          handle, graph_view, real_t{0}, p, q, num_paths};

        auto ret_tuple = impl_details::random_walks_impl<graph_vt,
                                                         decltype(selector),
                                                         impl_details::vertical_traversal_t>(
          handle,  // required to prevent clang-format to separate functin name from its namespace
          graph_view,
          d_start_view,
          max_depth,
          selector);

        // check results:
        //
        bool test_all_paths = cugraph::test::host_check_rw_paths(handle,
                                                                 graph_view,
                                                                 std::get<0>(ret_tuple),
                                                                 std::get<1>(ret_tuple),
                                                                 std::get<2>(ret_tuple));

        if (!test_all_paths)
          std::cout << "starting seed on failure: " << std::get<3>(ret_tuple) << '\n';

        ASSERT_TRUE(test_all_paths);
      }
    }
  }
};

TEST_P(Tests_RandomWalks, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_RandomWalks,
  ::testing::Combine(::testing::Values(traversal_id_t::HORIZONTAL, traversal_id_t::VERTICAL),
                     ::testing::Values(int{0}, int{1}, int{2}),
                     ::testing::Values(RandomWalks_Usecase("test/datasets/karate.mtx", true),
                                       RandomWalks_Usecase("test/datasets/web-Google.mtx", true),
                                       RandomWalks_Usecase("test/datasets/ljournal-2008.mtx", true),
                                       RandomWalks_Usecase("test/datasets/webbase-1M.mtx", true))));

CUGRAPH_TEST_PROGRAM_MAIN()
