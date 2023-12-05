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

#include <chrono>
#include <random>

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

    std::vector<bool> with_replacement_flags = {true, false};
    std::vector<bool> shuffle_flags          = {true, false};

    {
      // Generate distributed vertex set to sample from
      std::srand((unsigned)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count());

      std::vector<vertex_t> h_given_set(mg_graph_view.local_vertex_partition_range_size());

      std::iota(
        h_given_set.begin(), h_given_set.end(), mg_graph_view.local_vertex_partition_range_first());
      std::shuffle(h_given_set.begin(), h_given_set.end(), std::mt19937{std::random_device{}()});
      h_given_set.resize(std::rand() % (mg_graph_view.local_vertex_partition_range_size() + 1));

      // Compute size of the distributed vertex set
      int num_of_elements_in_given_set = static_cast<int>(h_given_set.size());
      num_of_elements_in_given_set     = cugraph::host_scalar_allreduce(handle_->get_comms(),
                                                                    num_of_elements_in_given_set,
                                                                    raft::comms::op_t::SUM,
                                                                    handle_->get_stream());
      // Move the distributed vertex set to GPUs
      auto d_given_set = cugraph::test::to_device(*handle_, h_given_set);

      // Sampling size should not exceed the size of distributed vertex set
      size_t select_count =
        num_of_elements_in_given_set > select_random_vertices_usecase.select_count
          ? select_random_vertices_usecase.select_count
          : std::rand() % (num_of_elements_in_given_set + 1);

      for (int idx = 0; idx < with_replacement_flags.size(); idx++) {
        bool with_replacement = with_replacement_flags[idx];
        auto d_sampled_vertices =
          cugraph::select_random_vertices(*handle_,
                                          mg_graph_view,
                                          std::make_optional(raft::device_span<vertex_t const>{
                                            d_given_set.data(), d_given_set.size()}),
                                          rng_state,
                                          select_count,
                                          with_replacement,
                                          true);

        RAFT_CUDA_TRY(cudaDeviceSynchronize());

        auto h_sampled_vertices = cugraph::test::to_host(*handle_, d_sampled_vertices);

        if (select_random_vertices_usecase.check_correctness) {
          if (!with_replacement) {
            std::sort(h_sampled_vertices.begin(), h_sampled_vertices.end());

            auto nr_duplicates =
              std::distance(std::unique(h_sampled_vertices.begin(), h_sampled_vertices.end()),
                            h_sampled_vertices.end());

            ASSERT_EQ(nr_duplicates, 0);
          }

          std::sort(h_given_set.begin(), h_given_set.end());
          std::for_each(
            h_sampled_vertices.begin(), h_sampled_vertices.end(), [&h_given_set](vertex_t v) {
              ASSERT_TRUE(std::binary_search(h_given_set.begin(), h_given_set.end(), v));
            });
        }
      }
    }

    //
    // Test sampling from [0, V)
    //
    std::vector<size_t> select_counts = {select_random_vertices_usecase.select_count,
                                         static_cast<size_t>(mg_graph_view.number_of_vertices())};

    for (int idx = 0; idx < with_replacement_flags.size(); idx++) {
      for (int k = 0; k < shuffle_flags.size(); k++) {
        for (int l = 0; l < select_counts.size(); l++) {
          bool with_replacement               = with_replacement_flags[idx];
          bool shuffle_using_vertex_partition = shuffle_flags[k];
          auto select_count                   = select_counts[l];

          auto d_sampled_vertices = cugraph::select_random_vertices(
            *handle_,
            mg_graph_view,
            std::optional<raft::device_span<vertex_t const>>{std::nullopt},
            rng_state,
            select_count,
            with_replacement,
            true,
            shuffle_using_vertex_partition);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());

          auto h_sampled_vertices = cugraph::test::to_host(*handle_, d_sampled_vertices);

          if (select_random_vertices_usecase.check_correctness) {
            if (!with_replacement) {
              std::sort(h_sampled_vertices.begin(), h_sampled_vertices.end());

              auto nr_duplicates =
                std::distance(std::unique(h_sampled_vertices.begin(), h_sampled_vertices.end()),
                              h_sampled_vertices.end());

              ASSERT_EQ(nr_duplicates, 0);
            }

            if (shuffle_using_vertex_partition) {
              auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
              auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

              std::for_each(h_sampled_vertices.begin(),
                            h_sampled_vertices.end(),
                            [vertex_first, vertex_last](vertex_t v) {
                              ASSERT_TRUE((v >= vertex_first) && (v < vertex_last));
                            });

            } else {
              if (select_count == static_cast<size_t>(mg_graph_view.number_of_vertices())) {
                ASSERT_EQ(h_sampled_vertices.size(),
                          mg_graph_view.local_vertex_partition_range_size());
              }
            }
          }
        }
      }
    }

    std::vector<bool> sort_vertices_flags = {true, false};

    for (int i = 0; i < with_replacement_flags.size(); i++) {
      for (int j = 0; j < sort_vertices_flags.size(); j++) {
        for (int k = 0; k < shuffle_flags.size(); k++) {
          for (int l = 0; l < select_counts.size(); l++) {
            bool with_replacement               = with_replacement_flags[i];
            bool sort_vertices                  = sort_vertices_flags[j];
            bool shuffle_using_vertex_partition = shuffle_flags[k];
            auto select_count                   = static_cast<size_t>(select_counts[l]);

            auto d_sampled_vertices = cugraph::select_random_vertices(
              *handle_,
              mg_graph_view,
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              rng_state,
              select_count,
              with_replacement,
              sort_vertices,
              shuffle_using_vertex_partition);

            RAFT_CUDA_TRY(cudaDeviceSynchronize());

            auto h_sampled_vertices = cugraph::test::to_host(*handle_, d_sampled_vertices);

            if (select_random_vertices_usecase.check_correctness) {
              if (!with_replacement) {
                std::sort(h_sampled_vertices.begin(), h_sampled_vertices.end());

                auto nr_duplicates =
                  std::distance(std::unique(h_sampled_vertices.begin(), h_sampled_vertices.end()),
                                h_sampled_vertices.end());

                ASSERT_EQ(nr_duplicates, 0);
              }

              if (shuffle_using_vertex_partition) {
                auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
                auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

                std::for_each(h_sampled_vertices.begin(),
                              h_sampled_vertices.end(),
                              [vertex_first, vertex_last](vertex_t v) {
                                ASSERT_TRUE((v >= vertex_first) && (v < vertex_last));
                              });
              } else {
                if (select_count == static_cast<size_t>(mg_graph_view.number_of_vertices())) {
                  ASSERT_EQ(h_sampled_vertices.size(),
                            mg_graph_view.local_vertex_partition_range_size());
                }
              }
            }
          }
        }
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
  ::testing::Combine(::testing::Values(SelectRandomVertices_Usecase{20, false},
                                       SelectRandomVertices_Usecase{20, false}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGSelectRandomVertices_Rmat,
  ::testing::Combine(
    ::testing::Values(SelectRandomVertices_Usecase{50, false},
                      SelectRandomVertices_Usecase{50, false}),
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
