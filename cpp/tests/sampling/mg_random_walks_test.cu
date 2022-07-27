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

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>

#include <gtest/gtest.h>

struct UniformRandomWalks_Usecase {
  bool test_weighted{false};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t max_depth)
  {
    return cugraph::uniform_random_walks(handle, graph_view, start_vertices, max_depth, seed);
  }
};

struct BiasedRandomWalks_Usecase {
  bool test_weighted{true};
  uint64_t seed{0};
  bool check_correctness{false};

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t max_depth)
  {
    return cugraph::biased_random_walks(handle, graph_view, start_vertices, max_depth, seed);
  }
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
             cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
             raft::device_span<vertex_t const> start_vertices,
             size_t max_depth)
  {
    return cugraph::node2vec_random_walks(
      handle, graph_view, start_vertices, max_depth, p, q, seed);
  }
};

template <typename tuple_t>
class Tests_MGRandomWalks : public ::testing::TestWithParam<tuple_t> {
 public:
  Tests_MGRandomWalks() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(tuple_t const& param)
  {
    HighResClock hr_clock{};

    auto [randomwalks_usecase, input_usecase] = param;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    bool renumber{true};
    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, randomwalks_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto graph_view = graph.view();

    edge_t num_paths = 10;
    rmm::device_uvector<vertex_t> d_start(num_paths, handle_->get_stream());

    thrust::tabulate(handle_->get_thrust_policy(),
                     d_start.begin(),
                     d_start.end(),
                     [num_vertices = graph_view.number_of_vertices()] __device__(auto idx) {
                       return (idx % num_vertices);
                     });

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

#if 0
    auto [vertices, weights] = randomwalks_usecase(
      *handle_, graph_view, raft::device_span<vertex_t const>{d_start.data(), d_start.size()}, size_t{10});
#else
    EXPECT_THROW(
      randomwalks_usecase(*handle_,
                          graph_view,
                          raft::device_span<vertex_t const>{d_start.data(), d_start.size()},
                          size_t{10}),
      std::exception);
#endif

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "PageRank took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (randomwalks_usecase.check_correctness) {
#if 0
      // FIXME: Need an MG test
#endif
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename tuple_t>
std::unique_ptr<raft::handle_t> Tests_MGRandomWalks<tuple_t>::handle_ = nullptr;

using Tests_UniformRandomWalks_File =
  Tests_MGRandomWalks<std::tuple<UniformRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_UniformRandomWalks_Rmat =
  Tests_MGRandomWalks<std::tuple<UniformRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;
using Tests_BiasedRandomWalks_File =
  Tests_MGRandomWalks<std::tuple<BiasedRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_BiasedRandomWalks_Rmat =
  Tests_MGRandomWalks<std::tuple<BiasedRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;
using Tests_Node2VecRandomWalks_File =
  Tests_MGRandomWalks<std::tuple<Node2VecRandomWalks_Usecase, cugraph::test::File_Usecase>>;
using Tests_Node2VecRandomWalks_Rmat =
  Tests_MGRandomWalks<std::tuple<Node2VecRandomWalks_Usecase, cugraph::test::Rmat_Usecase>>;

TEST_P(Tests_UniformRandomWalks_File, Initialize_i32_i32_f)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_UniformRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(UniformRandomWalks_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_BiasedRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(BiasedRandomWalks_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_Node2VecRandomWalks_File,
  ::testing::Combine(
    ::testing::Values(Node2VecRandomWalks_Usecase{4, 8}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
