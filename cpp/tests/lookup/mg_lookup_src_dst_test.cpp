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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/random/rng_state.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <execution>
#include <iostream>
#include <random>

struct EdgeSrcDstLookup_UseCase {
  // FIXME: Test with edge mask once the graph generator is updated to generate edge ids and types
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGLookupEdgeSrcDst
  : public ::testing::TestWithParam<std::tuple<EdgeSrcDstLookup_UseCase, input_usecase_t>> {
 public:
  Tests_MGLookupEdgeSrcDst() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<EdgeSrcDstLookup_UseCase, input_usecase_t> const& param)
  {
    auto [lookup_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    constexpr bool multi_gpu = true;

    bool test_weighted    = false;
    bool renumber         = true;
    bool drop_self_loops  = false;
    bool drop_multi_edges = false;

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        *handle_, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    std::tie(mg_graph, mg_edge_weights, mg_renumber_map) = cugraph::symmetrize_graph(
      *handle_,
      std::move(mg_graph),
      std::move(mg_edge_weights),
      mg_renumber_map ? std::optional<rmm::device_uvector<vertex_t>>(std::move(*mg_renumber_map))
                      : std::nullopt,
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};

    //
    // FIXME: As the graph generator doesn't generate edge ids and types at the moment, generate
    // edge ids and types for now and remove the code for generating edge ids and types from this
    // file once the graph generator is updated to generate edge ids and types.
    //

    int number_of_edge_types = std::max(
      1 << 8,
      static_cast<int>(std::rand() % (1 + (mg_graph_view.number_of_vertices() / (1 << 20)))));

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), int32_t>> edge_types{
      std::nullopt};
    edge_types = cugraph::test::generate<decltype(mg_graph_view), int32_t>::edge_property(
      *handle_, mg_graph_view, number_of_edge_types);

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), edge_t>> edge_ids{std::nullopt};

    edge_ids = cugraph::test::generate<decltype(mg_graph_view), edge_t>::edge_property(
      *handle_, mg_graph_view, 1);

    auto edge_counts = (*edge_ids).view().edge_counts();

    auto const comm_rank = (*handle_).get_comms().get_rank();
    auto const comm_size = (*handle_).get_comms().get_size();

    std::vector<size_t> type_freqs(number_of_edge_types, 0);
    std::mutex mtx[number_of_edge_types];

    for (size_t ep_idx = 0; ep_idx < edge_counts.size(); ep_idx++) {
      auto ep_types =
        cugraph::test::to_host(*handle_,
                               raft::device_span<int32_t const>(
                                 (*edge_types).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      std::for_each(std::execution::par, ep_types.begin(), ep_types.end(), [&](int32_t et) {
        std::lock_guard<std::mutex> guard(mtx[et]);
        type_freqs[et]++;
      });

      auto ep_ids =
        cugraph::test::to_host(*handle_,
                               raft::device_span<edge_t const>(
                                 (*edge_ids).view().value_firsts()[ep_idx], edge_counts[ep_idx]));
    }

    assert(std::reduce(type_freqs.cbegin(), type_freqs.cend()) ==
           std::reduce(edge_counts.cbegin(), edge_counts.cend()));

    auto d_type_freqs = cugraph::test::to_device(*handle_, type_freqs);
    d_type_freqs =
      cugraph::test::device_allgatherv(*handle_, d_type_freqs.data(), d_type_freqs.size());
    type_freqs = cugraph::test::to_host(*handle_, d_type_freqs);

    std::vector<size_t> distributed_type_offsets(comm_size * number_of_edge_types);

    for (size_t i = 0; i < number_of_edge_types; i++) {
      for (size_t j = 0; j < comm_size; j++) {
        distributed_type_offsets[j + comm_size * i] = type_freqs[number_of_edge_types * j + i];
      }
    }

    // prefix sum for each type
    for (size_t i = 0; i < number_of_edge_types; i++) {
      auto start = distributed_type_offsets.begin() + i * comm_size;
      std::exclusive_scan(start, start + comm_size, start, 0);
    }

    assert(std::reduce(distributed_type_offsets.cbegin(), distributed_type_offsets.cend()) ==
           mg_graph_view.compute_number_of_edges(*handle_));

    auto number_of_local_edges = std::reduce(edge_counts.cbegin(), edge_counts.cend());

    for (size_t ep_idx = 0; ep_idx < edge_counts.size(); ep_idx++) {
      auto ep_types =
        cugraph::test::to_host(*handle_,
                               raft::device_span<int32_t const>(
                                 (*edge_types).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      auto ep_ids =
        cugraph::test::to_host(*handle_,
                               raft::device_span<edge_t const>(
                                 (*edge_ids).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      std::transform(ep_types.cbegin(), ep_types.cend(), ep_ids.begin(), [&](int32_t et) {
        edge_t val = distributed_type_offsets[(comm_size * et + comm_rank)];
        distributed_type_offsets[(comm_size * et + comm_rank)]++;
        return val;
      });

      raft::update_device((*edge_ids).mutable_view().value_firsts()[ep_idx],
                          ep_ids.data(),
                          ep_ids.size(),
                          handle_->get_stream());
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Build Lookup Map");
    }

    auto search_container =
      cugraph::build_edge_id_and_type_to_src_dst_lookup_map<vertex_t, edge_t, int32_t, multi_gpu>(
        *handle_, mg_graph_view, (*edge_ids).view(), (*edge_types).view());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (lookup_usecase.check_correctness) {
      rmm::device_uvector<vertex_t> d_mg_srcs(0, handle_->get_stream());
      rmm::device_uvector<vertex_t> d_mg_dsts(0, handle_->get_stream());

      std::optional<rmm::device_uvector<edge_t>> d_mg_edge_ids{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> d_mg_edge_types{std::nullopt};

      std::tie(d_mg_srcs, d_mg_dsts, std::ignore, d_mg_edge_ids, d_mg_edge_types) =
        cugraph::decompress_to_edgelist(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::make_optional((*edge_ids).view()),
          std::make_optional((*edge_types).view()),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt});

      auto number_of_edges = mg_graph_view.compute_number_of_edges(*handle_);

      auto h_mg_edge_ids   = cugraph::test::to_host(*handle_, d_mg_edge_ids);
      auto h_mg_edge_types = cugraph::test::to_host(*handle_, d_mg_edge_types);

      auto h_srcs_expected = cugraph::test::to_host(*handle_, d_mg_srcs);
      auto h_dsts_expected = cugraph::test::to_host(*handle_, d_mg_dsts);

      if (number_of_local_edges > 0) {
        int nr_wrong_ids_or_types = (std::rand() % number_of_local_edges);

        for (int k = 0; k < nr_wrong_ids_or_types; k++) {
          auto id_or_type = std::rand() % 2;
          auto random_idx = std::rand() % number_of_local_edges;
          if (id_or_type)
            (*h_mg_edge_ids)[random_idx] = std::numeric_limits<edge_t>::max();
          else
            (*h_mg_edge_types)[random_idx] = std::numeric_limits<int32_t>::max() - 2;

          h_srcs_expected[random_idx] = cugraph::invalid_vertex_id<vertex_t>::value;
          h_dsts_expected[random_idx] = cugraph::invalid_vertex_id<vertex_t>::value;
        }
      }

      d_mg_edge_ids   = cugraph::test::to_device(*handle_, h_mg_edge_ids);
      d_mg_edge_types = cugraph::test::to_device(*handle_, h_mg_edge_types);

      auto [srcs, dsts] =
        cugraph::lookup_endpoints_from_edge_ids_and_types<vertex_t, edge_t, int32_t, multi_gpu>(
          *handle_,
          search_container,
          raft::device_span<edge_t>((*d_mg_edge_ids).begin(), (*d_mg_edge_ids).size()),
          raft::device_span<int32_t>((*d_mg_edge_types).begin(), (*d_mg_edge_types).size()));

      auto h_srcs_results = cugraph::test::to_host(*handle_, srcs);
      auto h_dsts_results = cugraph::test::to_host(*handle_, dsts);

      EXPECT_EQ(h_srcs_expected.size(), h_srcs_results.size());
      ASSERT_TRUE(
        std::equal(h_srcs_expected.begin(), h_srcs_expected.end(), h_srcs_results.begin()));

      EXPECT_EQ(h_dsts_expected.size(), h_dsts_results.size());
      ASSERT_TRUE(
        std::equal(h_dsts_expected.begin(), h_dsts_expected.end(), h_dsts_results.begin()));
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGLookupEdgeSrcDst<input_usecase_t>::handle_ = nullptr;

using Tests_MGLookupEdgeSrcDst_File = Tests_MGLookupEdgeSrcDst<cugraph::test::File_Usecase>;
using Tests_MGLookupEdgeSrcDst_Rmat = Tests_MGLookupEdgeSrcDst<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGLookupEdgeSrcDst_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLookupEdgeSrcDst_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLookupEdgeSrcDst_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLookupEdgeSrcDst_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGLookupEdgeSrcDst_File,
  ::testing::Combine(::testing::Values(EdgeSrcDstLookup_UseCase{}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGLookupEdgeSrcDst_Rmat,
                         ::testing::Combine(::testing::Values(EdgeSrcDstLookup_UseCase{}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              3, 2, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGLookupEdgeSrcDst_Rmat,
  ::testing::Combine(
    ::testing::Values(EdgeSrcDstLookup_UseCase{false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
