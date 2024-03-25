/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

struct Hits_Usecase {
  bool check_initial_input{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGHits : public ::testing::TestWithParam<std::tuple<Hits_Usecase, input_usecase_t>> {
 public:
  Tests_MGHits() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running HITS on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(Hits_Usecase const& hits_usecase, input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, true, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 2. run hits

    auto mg_graph_view = mg_graph.view();

    auto maximum_iterations = 200;
    weight_t epsilon        = 1e-7;
    rmm::device_uvector<weight_t> d_mg_hubs(mg_graph_view.local_vertex_partition_range_size(),
                                            handle_->get_stream());

    rmm::device_uvector<weight_t> d_mg_authorities(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    std::optional<rmm::device_uvector<weight_t>> d_mg_initial_random_hubs{std::nullopt};
    if (hits_usecase.check_initial_input) {
      raft::random::RngState rng_state(handle_->get_comms().get_rank());
      d_mg_initial_random_hubs =
        rmm::device_uvector<weight_t>(d_mg_hubs.size(), handle_->get_stream());
      cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                           (*d_mg_initial_random_hubs).data(),
                                           (*d_mg_initial_random_hubs).size(),
                                           weight_t{0.0},
                                           weight_t{1.0},
                                           rng_state);
      raft::copy(d_mg_hubs.data(),
                 (*d_mg_initial_random_hubs).data(),
                 (*d_mg_initial_random_hubs).size(),
                 handle_->get_stream());

      handle_->sync_stream();  // before rng_state goes out-of-scope
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG HITS");
    }

    auto result = cugraph::hits(*handle_,
                                mg_graph_view,
                                d_mg_hubs.data(),
                                d_mg_authorities.data(),
                                epsilon,
                                maximum_iterations,
                                hits_usecase.check_initial_input,
                                true,
                                hits_usecase.check_initial_input);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (hits_usecase.check_correctness) {
      // 3-1. aggregate MG results

      rmm::device_uvector<weight_t> d_mg_aggregate_hubs(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_hubs) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<weight_t const>(d_mg_hubs.data(), d_mg_hubs.size()));

      rmm::device_uvector<weight_t> d_mg_aggregate_authorities(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_authorities) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<weight_t const>(d_mg_authorities.data(), d_mg_authorities.size()));

      rmm::device_uvector<weight_t> d_mg_aggregate_initial_random_hubs(0, handle_->get_stream());
      if (hits_usecase.check_initial_input) {
        std::tie(std::ignore, d_mg_aggregate_initial_random_hubs) =
          cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
            *handle_,
            std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                  (*mg_renumber_map).size()),
            mg_graph_view.local_vertex_partition_range(),
            std::optional<raft::device_span<vertex_t const>>{std::nullopt},
            std::optional<raft::device_span<vertex_t const>>{std::nullopt},
            raft::device_span<weight_t const>((*d_mg_initial_random_hubs).data(),
                                              (*d_mg_initial_random_hubs).size()));
      }

      cugraph::graph_t<vertex_t, edge_t, true, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                              (*mg_renumber_map).size()),
        false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-3. run SG Hits

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        rmm::device_uvector<weight_t> d_sg_hubs(sg_graph_view.number_of_vertices(),
                                                handle_->get_stream());
        rmm::device_uvector<weight_t> d_sg_authorities(sg_graph_view.number_of_vertices(),
                                                       handle_->get_stream());
        if (hits_usecase.check_initial_input) {
          raft::copy(d_sg_hubs.begin(),
                     d_mg_aggregate_initial_random_hubs.data(),
                     d_mg_aggregate_initial_random_hubs.size(),
                     handle_->get_stream());
        }

        auto result = cugraph::hits(*handle_,
                                    sg_graph_view,
                                    d_sg_hubs.data(),
                                    d_sg_authorities.data(),
                                    epsilon,
                                    maximum_iterations,
                                    hits_usecase.check_initial_input,
                                    true,
                                    hits_usecase.check_initial_input);

        // 3-3. compare

        auto h_mg_aggregate_hubs = cugraph::test::to_host(*handle_, d_mg_aggregate_hubs);
        auto h_sg_hubs           = cugraph::test::to_host(*handle_, d_sg_hubs);

        auto threshold_ratio = 1e-3;
        auto threshold_magnitude =
          1e-6;  // skip comparison for low Hits verties (lowly ranked vertices)
        auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
          return std::abs(lhs - rhs) <
                 std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
        };

        ASSERT_TRUE(std::equal(
          h_mg_aggregate_hubs.begin(), h_mg_aggregate_hubs.end(), h_sg_hubs.begin(), nearly_equal));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGHits<input_usecase_t>::handle_ = nullptr;

using Tests_MGHits_File = Tests_MGHits<cugraph::test::File_Usecase>;
using Tests_MGHits_Rmat = Tests_MGHits<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGHits_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGHits_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGHits_Rmat, CheckInt32Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGHits_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGHits_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Hits_Usecase{false, true}, Hits_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGHits_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Hits_Usecase{false, true}, Hits_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGHits_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Hits_Usecase{false, false}, Hits_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
