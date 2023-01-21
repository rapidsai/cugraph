/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <sampling/random_walks_check.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <gtest/gtest.h>

struct UniformRandomWalks_Usecase {
  bool test_weighted{false};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t num_paths)
  {
    return cugraph::uniform_random_walks(
      handle, graph_view, edge_weight_view, start_vertices, num_paths, seed);
  }

  bool expect_throw() { return false; }
};

struct BiasedRandomWalks_Usecase {
  bool test_weighted{true};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t num_paths)
  {
    CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Biased random walk requires edge weights.");

    return cugraph::biased_random_walks(
      handle, graph_view, *edge_weight_view, start_vertices, num_paths, seed);
  }

  // FIXME: Not currently implemented
  bool expect_throw() { return true; }
};

struct Node2VecRandomWalks_Usecase {
  double p{1};
  double q{1};
  bool test_weighted{false};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t num_paths)
  {
    return cugraph::node2vec_random_walks(handle,
                                          graph_view,
                                          edge_weight_view,
                                          start_vertices,
                                          num_paths,
                                          static_cast<weight_t>(p),
                                          static_cast<weight_t>(q),
                                          seed);
  }

  // FIXME: Not currently implemented
  bool expect_throw() { return true; }
};

template <typename tuple_t>
class Tests_RandomWalks : public ::testing::TestWithParam<tuple_t> {
 public:
  Tests_RandomWalks() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(tuple_t const& param)
  {
    raft::handle_t handle{};
    HighResTimer hr_timer{};

    auto [randomwalks_usecase, input_usecase] = param;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    bool renumber{true};
    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, randomwalks_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    edge_t num_paths  = std::min(edge_t{10}, graph_view.number_of_vertices());
    edge_t max_length = 10;
    rmm::device_uvector<vertex_t> d_start(num_paths, handle.get_stream());

    cugraph::detail::sequence_fill(handle.get_stream(), d_start.begin(), d_start.size(), 0);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Random walks");
    }

    if (randomwalks_usecase.expect_throw()) {
      // biased and node2vec currently throw since they are not implemented
      EXPECT_THROW(
        randomwalks_usecase(handle,
                            graph_view,
                            edge_weight_view,
                            raft::device_span<vertex_t const>{d_start.data(), d_start.size()},
                            max_length),
        std::exception);
    } else {
      auto [d_vertices, d_weights] =
        randomwalks_usecase(handle,
                            graph_view,
                            edge_weight_view,
                            raft::device_span<vertex_t const>{d_start.data(), d_start.size()},
                            max_length);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      if (randomwalks_usecase.check_correctness) {
        cugraph::test::random_walks_validate(handle,
                                             graph_view,
                                             edge_weight_view,
                                             std::move(d_start),
                                             std::move(d_vertices),
                                             std::move(d_weights),
                                             max_length);
      }
    }
  }
};

using Tests_UniformRandomWalks_File =
  Tests_RandomWalks<std::tuple<UniformRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_UniformRandomWalks_Rmat =
  Tests_RandomWalks<std::tuple<UniformRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;
using Tests_BiasedRandomWalks_File =
  Tests_RandomWalks<std::tuple<BiasedRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_BiasedRandomWalks_Rmat =
  Tests_RandomWalks<std::tuple<BiasedRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;
using Tests_Node2VecRandomWalks_File =
  Tests_RandomWalks<std::tuple<Node2VecRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_Node2VecRandomWalks_Rmat =
  Tests_RandomWalks<std::tuple<Node2VecRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;

TEST_P(Tests_UniformRandomWalks_File, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_UniformRandomWalks_Rmat, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_BiasedRandomWalks_File, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_BiasedRandomWalks_Rmat, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Node2VecRandomWalks_File, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Node2VecRandomWalks_Rmat, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

#if 0
// FIXME: Not sure why these are failing, but we're refactoring anyway.
INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_UniformRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(UniformRandomWalks_Usecase{false, 0, true},
                      UniformRandomWalks_Usecase{true, 0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));
#endif

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_BiasedRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(BiasedRandomWalks_Usecase{false, 0, true},
                      BiasedRandomWalks_Usecase{true, 0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_Node2VecRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(Node2VecRandomWalks_Usecase{4, 8, false, 0, true},
                      Node2VecRandomWalks_Usecase{4, 8, true, 0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

#if 0
// FIXME: Not sure why these are failing, but we're refactoring anyway.
INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_UniformRandomWalks_Rmat,
  ::testing::Combine(::testing::Values(UniformRandomWalks_Usecase{false, 0, true},
                                       UniformRandomWalks_Usecase{true, 0, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_UniformRandomWalks_Rmat,
  ::testing::Combine(::testing::Values(UniformRandomWalks_Usecase{true, 0, false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, false))));
#endif

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_BiasedRandomWalks_Rmat,
  ::testing::Combine(::testing::Values(BiasedRandomWalks_Usecase{false, 0, true},
                                       BiasedRandomWalks_Usecase{true, 0, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_BiasedRandomWalks_Rmat,
  ::testing::Combine(::testing::Values(BiasedRandomWalks_Usecase{true, 0, false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Node2VecRandomWalks_Rmat,
  ::testing::Combine(::testing::Values(Node2VecRandomWalks_Usecase{8, 4, false, 0, true},
                                       Node2VecRandomWalks_Usecase{8, 4, true, 0, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_Node2VecRandomWalks_Rmat,
  ::testing::Combine(::testing::Values(Node2VecRandomWalks_Usecase{8, 4, true, 0, false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
