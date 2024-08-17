/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/validation_utilities.hpp"

#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <gtest/gtest.h>

struct Negative_Sampling_Usecase {
  float sample_multiplier{2};
  bool use_src_bias{false};
  bool use_dst_bias{false};
  bool remove_duplicates{false};
  bool remove_existing_edges{false};
  bool exact_number_of_samples{false};
  bool check_correctness{true};
};

template <typename input_usecase_t, typename vertex_t, typename edge_t, typename weight_t>
class Tests_MGNegative_Sampling : public ::testing::TestWithParam<input_usecase_t> {
 public:
  using graph_t      = cugraph::graph_t<vertex_t, edge_t, false, true>;
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, true>;

  Tests_MGNegative_Sampling() : graph_(*handle_) {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  template <typename input_t>
  void load_graph(input_t const& param)
  {
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    std::tie(graph_, edge_weights_, renumber_map_labels_) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, param, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
  }

  virtual void SetUp() {}
  virtual void TearDown() {}

  void run_current_test(raft::random::RngState& rng_state,
                        Negative_Sampling_Usecase const& negative_sampling_usecase)
  {
    constexpr bool do_expensive_check{false};

    HighResTimer hr_timer{};

    auto graph_view = graph_.view();

    size_t num_samples = graph_view.number_of_edges() * negative_sampling_usecase.sample_multiplier;

    rmm::device_uvector<weight_t> src_bias_v(0, handle_->get_stream());
    rmm::device_uvector<weight_t> dst_bias_v(0, handle_->get_stream());

    std::optional<raft::device_span<weight_t const>> src_bias{std::nullopt};
    std::optional<raft::device_span<weight_t const>> dst_bias{std::nullopt};

    if (negative_sampling_usecase.use_src_bias) {
      src_bias_v.resize(graph_view.local_vertex_partition_range_size(), handle_->get_stream());

      cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                           src_bias_v.data(),
                                           src_bias_v.size(),
                                           weight_t{1},
                                           weight_t{10},
                                           rng_state);

      src_bias = raft::device_span<weight_t const>{src_bias_v.data(), src_bias_v.size()};
    }

    if (negative_sampling_usecase.use_dst_bias) {
      dst_bias_v.resize(graph_view.local_vertex_partition_range_size(), handle_->get_stream());

      cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                           dst_bias_v.data(),
                                           dst_bias_v.size(),
                                           weight_t{1},
                                           weight_t{10},
                                           rng_state);

      dst_bias = raft::device_span<weight_t const>{dst_bias_v.data(), dst_bias_v.size()};
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Negative sampling");
    }

    auto&& [src_out, dst_out] =
      cugraph::negative_sampling(*handle_,
                                 rng_state,
                                 graph_view,
                                 num_samples,
                                 src_bias,
                                 dst_bias,
                                 negative_sampling_usecase.remove_duplicates,
                                 negative_sampling_usecase.remove_existing_edges,
                                 negative_sampling_usecase.exact_number_of_samples,
                                 do_expensive_check);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (negative_sampling_usecase.check_correctness) {
      ASSERT_EQ(src_out.size(), dst_out.size()) << "Result size (src, dst) mismatch";

      cugraph::test::sort(*handle_,
                          raft::device_span<vertex_t>{src_out.data(), src_out.size()},
                          raft::device_span<vertex_t>{dst_out.data(), dst_out.size()});

      // TODO:  Move this to validation_utilities...
      auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
      rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
        h_vertex_partition_range_lasts.size(), handle_->get_stream());
      raft::update_device(d_vertex_partition_range_lasts.data(),
                          h_vertex_partition_range_lasts.data(),
                          h_vertex_partition_range_lasts.size(),
                          handle_->get_stream());

      size_t error_count = cugraph::test::count_edges_on_wrong_int_gpu(
        *handle_,
        raft::device_span<vertex_t const>{src_out.data(), src_out.size()},
        raft::device_span<vertex_t const>{dst_out.data(), dst_out.size()},
        raft::device_span<vertex_t const>{d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()});

      ASSERT_EQ(error_count, 0) << "generate edges out of range > 0";

      if ((negative_sampling_usecase.remove_duplicates) && (src_out.size() > 0)) {
        error_count = cugraph::test::count_duplicate_vertex_pairs_sorted(
          *handle_,
          raft::device_span<vertex_t const>{src_out.data(), src_out.size()},
          raft::device_span<vertex_t const>{dst_out.data(), dst_out.size()});
        ASSERT_EQ(error_count, 0) << "Remove duplicates specified, found duplicate entries";
      }

      if (negative_sampling_usecase.remove_existing_edges) {
        rmm::device_uvector<vertex_t> graph_src(0, handle_->get_stream());
        rmm::device_uvector<vertex_t> graph_dst(0, handle_->get_stream());

        std::tie(graph_src, graph_dst, std::ignore, std::ignore, std::ignore) =
          cugraph::decompress_to_edgelist<vertex_t, edge_t, float, int, false, true>(
            *handle_, graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

        error_count = cugraph::test::count_intersection<vertex_t, edge_t, weight_t, int32_t>(
          *handle_,
          raft::device_span<vertex_t const>{graph_src.data(), graph_src.size()},
          raft::device_span<vertex_t const>{graph_dst.data(), graph_dst.size()},
          std::nullopt,
          std::nullopt,
          std::nullopt,
          raft::device_span<vertex_t const>{src_out.data(), src_out.size()},
          raft::device_span<vertex_t const>{dst_out.data(), dst_out.size()},
          std::nullopt,
          std::nullopt,
          std::nullopt);
        ASSERT_EQ(error_count, 0) << "Remove existing edges specified, found existing edges";
      }

      if (negative_sampling_usecase.exact_number_of_samples) {
        size_t sz = cugraph::host_scalar_allreduce(
          handle_->get_comms(), src_out.size(), raft::comms::op_t::SUM, handle_->get_stream());
        ASSERT_EQ(sz, num_samples) << "Expected exact number of samples";
      }

      //  TBD: How do we determine if we have properly reflected the biases?
    }
  }

 public:
  static std::unique_ptr<raft::handle_t> handle_;

 private:
  graph_t graph_;
  std::optional<cugraph::edge_property_t<graph_view_t, weight_t>> edge_weights_{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map_labels_{std::nullopt};
};

template <typename input_usecase_t, typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<raft::handle_t>
  Tests_MGNegative_Sampling<input_usecase_t, vertex_t, edge_t, weight_t>::handle_ = nullptr;

using Tests_MGNegative_Sampling_File_i64_i64_float =
  Tests_MGNegative_Sampling<cugraph::test::File_Usecase, int64_t, int64_t, float>;

using Tests_MGNegative_Sampling_Rmat_i64_i64_float =
  Tests_MGNegative_Sampling<cugraph::test::Rmat_Usecase, int64_t, int64_t, float>;

template <typename CurrentTest>
void run_all_tests(CurrentTest* current_test)
{
  raft::random::RngState rng_state{
    static_cast<uint64_t>(current_test->handle_->get_comms().get_rank())};

  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, false, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, false, false, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, true, false, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, true, false, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, true, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, false, true, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, true, true, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, true, true, false, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, false, true, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, false, false, true, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, true, false, true, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, true, false, true, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, true, true, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, false, true, true, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, true, true, true, false, true});
  current_test->run_current_test(rng_state,
                                 Negative_Sampling_Usecase{2, true, true, true, true, false, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, false, false, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, false, false, false, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, true, false, false, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, true, false, false, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, true, false, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, false, true, false, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, true, true, false, true, true});
  current_test->run_current_test(rng_state,
                                 Negative_Sampling_Usecase{2, true, true, true, false, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, false, true, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, true, false, false, true, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, true, false, true, true, true});
  current_test->run_current_test(rng_state,
                                 Negative_Sampling_Usecase{2, true, true, false, true, true, true});
  current_test->run_current_test(
    rng_state, Negative_Sampling_Usecase{2, false, false, true, true, true, true});
  current_test->run_current_test(rng_state,
                                 Negative_Sampling_Usecase{2, true, false, true, true, true, true});
  current_test->run_current_test(rng_state,
                                 Negative_Sampling_Usecase{2, false, true, true, true, true, true});
  current_test->run_current_test(rng_state,
                                 Negative_Sampling_Usecase{2, true, true, true, true, true, true});
}

TEST_P(Tests_MGNegative_Sampling_File_i64_i64_float, CheckInt64Int64Float)
{
  load_graph(override_File_Usecase_with_cmd_line_arguments(GetParam()));
  run_all_tests(this);
}

TEST_P(Tests_MGNegative_Sampling_Rmat_i64_i64_float, CheckInt64Int64Float)
{
  load_graph(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
  run_all_tests(this);
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGNegative_Sampling_File_i64_i64_float,
  ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx")));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGNegative_Sampling_File_i64_i64_float,
  ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                    cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                    cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx")));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGNegative_Sampling_Rmat_i64_i64_float,
  ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false, 0)));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGNegative_Sampling_Rmat_i64_i64_float,
  ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0)));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
