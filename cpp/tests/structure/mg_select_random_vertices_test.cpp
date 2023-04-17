/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <cugraph/utilities/high_res_timer.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <gtest/gtest.h>

struct SelectRandomVertices_Usecase {
  size_t select_count{std::numeric_limits<size_t>::max()};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGSelectRandomVertices
  : public ::testing::TestWithParam<std::tuple<SelectRandomVertices_Usecase, input_usecase_t>> {
 public:
  Tests_MGSelectRandomVertices() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<SelectRandomVertices_Usecase, input_usecase_t> const& param)
  {
    auto [select_random_vertices_usecase, input_usecase] = param;

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    raft::random::RngState rng_state(comm_rank);

    //
    // Test sampling from a distributed set
    //

    {
      // Generate distributed vertex set to sample from
      srand((unsigned)time(NULL));
      std::vector<vertex_t> h_given_set(rand() % mg_graph_view.local_vertex_partition_range_size() +
                                        1);

      for (int i = 0; i < h_given_set.size(); i++) {
        h_given_set[i] = mg_graph_view.local_vertex_partition_range_first() +
                         rand() % mg_graph_view.local_vertex_partition_range_size();
      }

      std::sort(h_given_set.begin(), h_given_set.end());
      auto last = std::unique(h_given_set.begin(), h_given_set.end());
      h_given_set.erase(last, h_given_set.end());

      // Compute size of the distributed vertex set
      int num_of_given_set = static_cast<int>(h_given_set.size());

      rmm::device_uvector<int> d_num_of_given_set(1, handle_->get_stream());
      raft::update_device(d_num_of_given_set.data(), &num_of_given_set, 1, handle_->get_stream());
      handle_->get_comms().allreduce(d_num_of_given_set.data(),
                                     d_num_of_given_set.data(),
                                     1,
                                     raft::comms::op_t::SUM,
                                     handle_->get_stream());
      raft::update_host(&num_of_given_set, d_num_of_given_set.data(), 1, handle_->get_stream());
      auto status = handle_->get_comms().sync_stream(handle_->get_stream());
      CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

      // Move the distributed vertex set to GPUs
      std::optional<rmm::device_uvector<vertex_t>> d_given_set{std::nullopt};
      d_given_set = rmm::device_uvector<vertex_t>(h_given_set.size(), handle_->get_stream());
      raft::update_device(
        (*d_given_set).data(), h_given_set.data(), h_given_set.size(), handle_->get_stream());

      // Sampling size should not exceed the size of distributed vertex set
      size_t select_count = num_of_given_set > select_random_vertices_usecase.select_count
                              ? select_random_vertices_usecase.select_count
                              : num_of_given_set / 2 + 1;

      auto d_sampled_vertices = cugraph::select_random_vertices(
        *handle_,
        mg_graph_view,
        d_given_set ? std::move(d_given_set)
                    : std::optional<rmm::device_uvector<vertex_t>>{std::nullopt},
        rng_state,
        select_count,
        false,
        true);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      std::vector<vertex_t> h_sampled_vertices(d_sampled_vertices.size());
      raft::update_host(h_sampled_vertices.data(),
                        d_sampled_vertices.data(),
                        d_sampled_vertices.size(),
                        handle_->get_stream());

      if (select_random_vertices_usecase.check_correctness) {
        std::for_each(
          h_sampled_vertices.begin(), h_sampled_vertices.end(), [&h_given_set](vertex_t v) {
            ASSERT_TRUE(std::binary_search(h_given_set.begin(), h_given_set.end(), v));
          });
      }
    }

    //
    // Test sampling from [0, V)
    //
    {
      auto d_sampled_vertices =
        cugraph::select_random_vertices(*handle_,
                                        mg_graph_view,
                                        std::optional<rmm::device_uvector<vertex_t>>{std::nullopt},
                                        rng_state,
                                        select_random_vertices_usecase.select_count,
                                        false,
                                        true);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      std::vector<vertex_t> h_sampled_vertices(d_sampled_vertices.size());
      raft::update_host(h_sampled_vertices.data(),
                        d_sampled_vertices.data(),
                        d_sampled_vertices.size(),
                        handle_->get_stream());

      if (select_random_vertices_usecase.check_correctness) {
        auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
        auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

        std::for_each(h_sampled_vertices.begin(),
                      h_sampled_vertices.end(),
                      [vertex_first, vertex_last](vertex_t v) {
                        ASSERT_TRUE((v >= vertex_first) && (v < vertex_last));
                      });
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGSelectRandomVertices<input_usecase_t>::handle_ = nullptr;

using Tests_MGSelectRandomVertices_File = Tests_MGSelectRandomVertices<cugraph::test::File_Usecase>;
using Tests_MGSelectRandomVertices_Rmat = Tests_MGSelectRandomVertices<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_MGSelectRandomVertices_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGSelectRandomVertices_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGSelectRandomVertices_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGSelectRandomVertices_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_MGSelectRandomVertices_File,
  ::testing::Combine(::testing::Values(SelectRandomVertices_Usecase{20, true},
                                       SelectRandomVertices_Usecase{20, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGSelectRandomVertices_Rmat,
  ::testing::Combine(
    ::testing::Values(SelectRandomVertices_Usecase{50, true},
                      SelectRandomVertices_Usecase{50, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(6, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGSelectRandomVertices_Rmat,
  ::testing::Combine(
    ::testing::Values(SelectRandomVertices_Usecase{500, true},
                      SelectRandomVertices_Usecase{500, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
