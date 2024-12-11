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
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct HasEdgeAndComputeMultiplicity_Usecase {
  size_t num_vertex_pairs{};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGHasEdgeAndComputeMultiplicity
  : public ::testing::TestWithParam<
      std::tuple<HasEdgeAndComputeMultiplicity_Usecase, input_usecase_t>> {
 public:
  Tests_MGHasEdgeAndComputeMultiplicity() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running has_edge & compute_multiplicity on multiple GPUs to that of
  // a single-GPU run
  template <typename vertex_t, typename edge_t, bool store_transposed>
  void run_current_test(
    HasEdgeAndComputeMultiplicity_Usecase const& has_edge_and_compute_multiplicity_usecase,
    input_usecase_t const& input_usecase)
  {
    using weight_t       = float;
    using edge_type_id_t = int32_t;

    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. create an edge list to query

    raft::random::RngState rng_state(comm_rank);
    size_t num_vertex_pairs_this_gpu =
      (has_edge_and_compute_multiplicity_usecase.num_vertex_pairs / comm_size) +
      ((comm_rank < has_edge_and_compute_multiplicity_usecase.num_vertex_pairs % comm_size)
         ? size_t{1}
         : size_t{0});
    rmm::device_uvector<vertex_t> d_mg_edge_srcs(num_vertex_pairs_this_gpu, handle_->get_stream());
    rmm::device_uvector<vertex_t> d_mg_edge_dsts(d_mg_edge_srcs.size(), handle_->get_stream());
    cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                         d_mg_edge_srcs.data(),
                                         d_mg_edge_srcs.size(),
                                         vertex_t{0},
                                         mg_graph_view.number_of_vertices(),
                                         rng_state);
    cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                         d_mg_edge_dsts.data(),
                                         d_mg_edge_dsts.size(),
                                         vertex_t{0},
                                         mg_graph_view.number_of_vertices(),
                                         rng_state);

    std::tie(store_transposed ? d_mg_edge_dsts : d_mg_edge_srcs,
             store_transposed ? d_mg_edge_srcs : d_mg_edge_dsts,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore) =
      cugraph::detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
        vertex_t,
        edge_t,
        weight_t,
        edge_type_id_t>(*handle_,
                        std::move(store_transposed ? d_mg_edge_dsts : d_mg_edge_srcs),
                        std::move(store_transposed ? d_mg_edge_srcs : d_mg_edge_dsts),
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        mg_graph_view.vertex_partition_range_lasts());

    // 3. run MG has_edge & compute_multiplicity

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Querying edge existence");
    }

    auto d_mg_edge_exists = mg_graph_view.has_edge(
      *handle_,
      raft::device_span<vertex_t const>(d_mg_edge_srcs.data(), d_mg_edge_srcs.size()),
      raft::device_span<vertex_t const>(d_mg_edge_dsts.data(), d_mg_edge_dsts.size()));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Computing multiplicity");
    }

    auto d_mg_edge_multiplicities = mg_graph_view.compute_multiplicity(
      *handle_,
      raft::device_span<vertex_t const>(d_mg_edge_srcs.data(), d_mg_edge_srcs.size()),
      raft::device_span<vertex_t const>(d_mg_edge_dsts.data(), d_mg_edge_dsts.size()));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 4. copmare SG & MG results

    if (has_edge_and_compute_multiplicity_usecase.check_correctness) {
      // 4-1. aggregate MG results

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_edge_srcs.data(),
        d_mg_edge_srcs.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_edge_dsts.data(),
        d_mg_edge_dsts.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      auto d_mg_aggregate_edge_srcs = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(d_mg_edge_srcs.data(), d_mg_edge_srcs.size()));
      auto d_mg_aggregate_edge_dsts = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(d_mg_edge_dsts.data(), d_mg_edge_dsts.size()));
      auto d_mg_aggregate_edge_exists = cugraph::test::device_gatherv(
        *handle_, raft::device_span<bool const>(d_mg_edge_exists.data(), d_mg_edge_exists.size()));
      auto d_mg_aggregate_edge_multiplicities = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<edge_t const>(d_mg_edge_multiplicities.data(),
                                        d_mg_edge_multiplicities.size()));

      cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();

        // 4-2. run SG count_self_loops & count_multi_edges

        auto d_sg_edge_exists = sg_graph_view.has_edge(
          *handle_,
          raft::device_span<vertex_t const>(d_mg_aggregate_edge_srcs.data(),
                                            d_mg_aggregate_edge_srcs.size()),
          raft::device_span<vertex_t const>(d_mg_aggregate_edge_dsts.data(),
                                            d_mg_aggregate_edge_dsts.size()));
        auto d_sg_edge_multiplicities = sg_graph_view.compute_multiplicity(
          *handle_,
          raft::device_span<vertex_t const>(d_mg_aggregate_edge_srcs.data(),
                                            d_mg_aggregate_edge_srcs.size()),
          raft::device_span<vertex_t const>(d_mg_aggregate_edge_dsts.data(),
                                            d_mg_aggregate_edge_dsts.size()));

        // 4-3. compare

        auto h_mg_aggregate_edge_exists =
          cugraph::test::to_host(*handle_, d_mg_aggregate_edge_exists);
        auto h_mg_aggregate_edge_multiplicities =
          cugraph::test::to_host(*handle_, d_mg_aggregate_edge_multiplicities);
        auto h_sg_edge_exists         = cugraph::test::to_host(*handle_, d_sg_edge_exists);
        auto h_sg_edge_multiplicities = cugraph::test::to_host(*handle_, d_sg_edge_multiplicities);

        ASSERT_TRUE(std::equal(h_mg_aggregate_edge_exists.begin(),
                               h_mg_aggregate_edge_exists.end(),
                               h_sg_edge_exists.begin()));
        ASSERT_TRUE(std::equal(h_mg_aggregate_edge_multiplicities.begin(),
                               h_mg_aggregate_edge_multiplicities.end(),
                               h_sg_edge_multiplicities.begin()));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGHasEdgeAndComputeMultiplicity<input_usecase_t>::handle_ =
  nullptr;

using Tests_MGHasEdgeAndComputeMultiplicity_File =
  Tests_MGHasEdgeAndComputeMultiplicity<cugraph::test::File_Usecase>;
using Tests_MGHasEdgeAndComputeMultiplicity_Rmat =
  Tests_MGHasEdgeAndComputeMultiplicity<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGHasEdgeAndComputeMultiplicity_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGHasEdgeAndComputeMultiplicity_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGHasEdgeAndComputeMultiplicity_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGHasEdgeAndComputeMultiplicity_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGHasEdgeAndComputeMultiplicity_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGHasEdgeAndComputeMultiplicity_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(HasEdgeAndComputeMultiplicity_Usecase{1024 * 128}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_MGHasEdgeAndComputeMultiplicity_Rmat,
  ::testing::Combine(
    ::testing::Values(HasEdgeAndComputeMultiplicity_Usecase{1024 * 128}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGHasEdgeAndComputeMultiplicity_Rmat,
  ::testing::Combine(
    ::testing::Values(HasEdgeAndComputeMultiplicity_Usecase{1024 * 1024 * 128, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
