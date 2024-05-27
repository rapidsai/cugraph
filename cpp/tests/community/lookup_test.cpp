
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
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/random/rng_state.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <execution>
#include <iostream>
#include <random>

struct EdgeSrcDstLookup_UseCase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_SGLookupEdgeSrcDst
  : public ::testing::TestWithParam<std::tuple<EdgeSrcDstLookup_UseCase, input_usecase_t>> {
 public:
  Tests_SGLookupEdgeSrcDst() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<EdgeSrcDstLookup_UseCase, input_usecase_t> const& param)
  {
    auto [lookup_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      hr_timer.start("Construct graph");
    }

    constexpr bool multi_gpu = false;

    bool test_weighted    = true;
    bool renumber         = true;
    bool drop_self_loops  = false;
    bool drop_multi_edges = false;

    auto comm_rank = 0;

      auto [sg_graph, sg_edge_weights, sg_renumber_map] =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
          handle, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    std::tie(sg_graph, sg_edge_weights, sg_renumber_map) = cugraph::symmetrize_graph(
      handle, std::move(sg_graph), std::move(sg_edge_weights), std::move(sg_renumber_map), false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto sg_graph_view = sg_graph.view();
    auto sg_edge_weight_view =
      sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<decltype(sg_graph_view), bool>> edge_mask{std::nullopt};
    if (lookup_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(sg_graph_view), bool>::edge_property(
        handle, sg_graph_view, 2);
      // sg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    ///////////////

    int32_t nr_hash_bins = 6 + static_cast<int>(sg_graph_view.number_of_vertices() / (1 << 10));

    std::cout << "nrbins: " << nr_hash_bins << "\n";
    std::optional<cugraph::edge_property_t<decltype(sg_graph_view), int32_t>> edge_types{
      std::nullopt};
    edge_types = cugraph::test::generate<decltype(sg_graph_view), int32_t>::edge_property(
      handle, sg_graph_view, nr_hash_bins);

    std::optional<cugraph::edge_property_t<decltype(sg_graph_view), edge_t>> edge_ids{std::nullopt};

    edge_ids = cugraph::test::generate<decltype(sg_graph_view), edge_t>::edge_property(
      handle, sg_graph_view, 1);

    auto edge_counts = (*edge_ids).view().edge_counts();

    bool debug = (sg_graph_view.number_of_vertices() < 16);
    if (debug) {
      std::for_each(edge_counts.cbegin(), edge_counts.cend(), [](const edge_t& cnt) {
        std::cout << cnt << " ";
      });
      std::cout << "\n";
    }

    std::vector<size_t> type_freqs(nr_hash_bins, 0);
    std::mutex mtx[nr_hash_bins];

    for (size_t ep_idx = 0; ep_idx < edge_counts.size(); ep_idx++) {
      auto ep_types =
        cugraph::test::to_host(handle,
                               raft::device_span<int32_t const>(
                                 (*edge_types).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      if (debug) {
        std::for_each(ep_types.cbegin(), ep_types.cend(), [nr_hash_bins](const int32_t& cnt) {
          if ((cnt < 0) || cnt > nr_hash_bins) std::cout << "Error: " << cnt << "\n";
          std::cout << cnt << " ";
        });
        std::cout << "\n";
      }

      std::for_each(std::execution::par, ep_types.begin(), ep_types.end(), [&](int32_t et) {
        std::lock_guard<std::mutex> guard(mtx[et]);
        type_freqs[et]++;
      });

      if (debug) {
        std::for_each(
          type_freqs.cbegin(), type_freqs.cend(), [](const edge_t& f) { std::cout << f << " "; });
        std::cout << "\n";
      }

      auto ep_ids =
        cugraph::test::to_host(handle,
                               raft::device_span<edge_t const>(
                                 (*edge_ids).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      if (debug) {
        std::for_each(
          ep_ids.cbegin(), ep_ids.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
        std::cout << "\n";
      }
    }

    assert(std::reduce(type_freqs.cbegin(), type_freqs.cend()) ==
           std::reduce(edge_counts.cbegin(), edge_counts.cend()));

    auto d_type_freqs = cugraph::test::to_device(handle, type_freqs);

    if (debug) {
      std::for_each(
        type_freqs.cbegin(), type_freqs.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
      std::cout << "\n";
    }

    std::vector<size_t> distributed_type_offsets(nr_hash_bins);

    for (size_t i = 0; i < nr_hash_bins; i++) {
      distributed_type_offsets[i] = type_freqs[i];
    }
    std::cout << "\n";

    if (debug) {
      std::for_each(distributed_type_offsets.cbegin(),
                    distributed_type_offsets.cend(),
                    [](const edge_t& cnt) { std::cout << cnt << " "; });
      std::cout << "\n";
    }

    assert(std::reduce(distributed_type_offsets.cbegin(), distributed_type_offsets.cend()) ==
           sg_graph_view.compute_number_of_edges(handle));

    std::cout << " sg_graph_view: #V " << sg_graph_view.number_of_vertices() << " C#E "
              << sg_graph_view.compute_number_of_edges(handle) << "\n";

    auto number_of_local_edges = std::reduce(edge_counts.cbegin(), edge_counts.cend());

    for (size_t ep_idx = 0; ep_idx < edge_counts.size(); ep_idx++) {
      auto ep_types =
        cugraph::test::to_host(handle,
                               raft::device_span<int32_t const>(
                                 (*edge_types).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      if (debug) {
        std::cout << " *ep_types: ";
        std::for_each(ep_types.cbegin(), ep_types.cend(), [nr_hash_bins](const int32_t& cnt) {
          if ((cnt < 0) || cnt > nr_hash_bins) std::cout << "Error: " << cnt << "\n";
          std::cout << cnt << " ";
        });
        std::cout << "\n";
      }

      auto ep_ids =
        cugraph::test::to_host(handle,
                               raft::device_span<edge_t const>(
                                 (*edge_ids).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      std::transform(ep_types.cbegin(), ep_types.cend(), ep_ids.begin(), [&](int32_t et) {
        edge_t val = distributed_type_offsets[et];
        distributed_type_offsets[et]++;
        return val;
      });

      raft::update_device((*edge_ids).mutable_view().value_firsts()[ep_idx],
                          ep_ids.data(),
                          ep_ids.size(),
                          handle.get_stream());

      if (debug) {
        std::cout << "Rank: " << comm_rank << " *ep_ids: ";
        std::for_each(
          ep_ids.cbegin(), ep_ids.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
        std::cout << "\n";
      }
    }

    auto search_container =
      cugraph::build_edge_id_and_type_to_src_dst_lookup_map<vertex_t, edge_t, int32_t, multi_gpu>(
        handle, sg_graph_view, (*edge_ids).view(), (*edge_types).view());

    search_container.print();

    if (lookup_usecase.check_correctness) {
      rmm::device_uvector<vertex_t> d_mg_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> d_mg_dsts(0, handle.get_stream());

      std::optional<rmm::device_uvector<edge_t>> d_mg_edge_ids{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> d_mg_edge_types{std::nullopt};

      std::tie(d_mg_srcs, d_mg_dsts, std::ignore, d_mg_edge_ids, d_mg_edge_types) =
        cugraph::decompress_to_edgelist(
          handle,
          sg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::make_optional((*edge_ids).view()),
          std::make_optional((*edge_types).view()),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt});

      /* */

      if (debug) {
        auto srcs_title = std::string("srcs_").append(std::to_string(comm_rank));

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector(
          srcs_title.c_str(), d_mg_srcs.begin(), d_mg_srcs.size(), std::cout);

        auto dsts_title = std::string("dsts_").append(std::to_string(comm_rank));

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector(
          dsts_title.c_str(), d_mg_dsts.begin(), d_mg_dsts.size(), std::cout);

        auto edge_types_title = std::string("edge_types_").append(std::to_string(comm_rank));

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector(edge_types_title.c_str(),
                                  (*d_mg_edge_types).begin(),
                                  (*d_mg_edge_types).size(),
                                  std::cout);

        auto edge_ids_title = std::string("edge_ids_").append(std::to_string(comm_rank));

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector(
          edge_ids_title.c_str(), (*d_mg_edge_ids).begin(), (*d_mg_edge_ids).size(), std::cout);
      }

      auto number_of_edges = sg_graph_view.compute_number_of_edges(handle);

      auto h_mg_edge_ids   = cugraph::test::to_host(handle, d_mg_edge_ids);
      auto h_mg_edge_types = cugraph::test::to_host(handle, d_mg_edge_types);

      auto h_srcs_expected = cugraph::test::to_host(handle, d_mg_srcs);
      auto h_dsts_expected = cugraph::test::to_host(handle, d_mg_dsts);

      if (number_of_local_edges > 0) {
        int nr_wrong_ids_or_types = (std::rand() % number_of_local_edges);

        for (int k = 0; k < nr_wrong_ids_or_types; k++) {
          auto id_or_type = std::rand() % 2;
          auto random_idx = std::rand() % number_of_local_edges;
          if (id_or_type)
            (*h_mg_edge_ids)[random_idx] = number_of_edges;
          else
            (*h_mg_edge_types)[random_idx] = nr_hash_bins;

          h_srcs_expected[random_idx] = cugraph::invalid_vertex_id<vertex_t>::value;
          h_dsts_expected[random_idx] = cugraph::invalid_vertex_id<vertex_t>::value;
        }
      }

      d_mg_edge_ids   = cugraph::test::to_device(handle, h_mg_edge_ids);
      d_mg_edge_types = cugraph::test::to_device(handle, h_mg_edge_types);

      if (debug) {
        std::cout << "Rank: " << comm_rank << " h_mg_edge_ids: ";
        std::for_each((*h_mg_edge_ids).cbegin(), (*h_mg_edge_ids).cend(), [](const auto& f) {
          std::cout << f << " ";
        });
        std::cout << "\n";

        std::cout << "Rank: " << comm_rank << " (*h_mg_edge_types): ";
        std::for_each((*h_mg_edge_types).cbegin(), (*h_mg_edge_types).cend(), [](const auto& f) {
          std::cout << f << " ";
        });
        std::cout << "\n";
      }

      auto [srcs, dsts] = cugraph::
        cugraph_lookup_src_dst_from_edge_id_and_type_pub<vertex_t, edge_t, int32_t, multi_gpu>(
          handle,
          search_container,
          raft::device_span<edge_t>((*d_mg_edge_ids).begin(), (*d_mg_edge_ids).size()),
          raft::device_span<int32_t>((*d_mg_edge_types).begin(), (*d_mg_edge_types).size()));

      auto h_srcs_results = cugraph::test::to_host(handle, srcs);
      auto h_dsts_results = cugraph::test::to_host(handle, dsts);

      if (debug) {
        std::cout << "Rank: " << comm_rank << " check correctness .........\n";

        std::cout << "Rank: " << comm_rank << " h_srcs_expected: ";
        std::for_each(h_srcs_expected.cbegin(), h_srcs_expected.cend(), [](const auto& f) {
          std::cout << f << " ";
        });
        std::cout << "\n";

        std::cout << "Rank: " << comm_rank << " h_srcs_results: ";
        std::for_each(h_srcs_results.cbegin(), h_srcs_results.cend(), [](const auto& f) {
          std::cout << f << " ";
        });
        std::cout << "\n";

        std::cout << "Rank: " << comm_rank << " h_dsts_expected: ";
        std::for_each(h_dsts_expected.cbegin(), h_dsts_expected.cend(), [](const auto& f) {
          std::cout << f << " ";
        });
        std::cout << "\n";

        std::cout << "Rank: " << comm_rank << " h_dsts_results: ";
        std::for_each(h_dsts_results.cbegin(), h_dsts_results.cend(), [](const auto& f) {
          std::cout << f << " ";
        });
        std::cout << "\n";
      }

      EXPECT_EQ(h_srcs_expected.size(), h_srcs_results.size());
      ASSERT_TRUE(
        std::equal(h_srcs_expected.begin(), h_srcs_expected.end(), h_srcs_results.begin()));

      EXPECT_EQ(h_dsts_expected.size(), h_dsts_results.size());
      ASSERT_TRUE(
        std::equal(h_dsts_expected.begin(), h_dsts_expected.end(), h_dsts_results.begin()));
    }

    ///////////////
  }
};

using Tests_SGLookupEdgeSrcDst_File = Tests_SGLookupEdgeSrcDst<cugraph::test::File_Usecase>;
using Tests_SGLookupEdgeSrcDst_Rmat = Tests_SGLookupEdgeSrcDst<cugraph::test::Rmat_Usecase>;

// TEST_P(Tests_SGLookupEdgeSrcDst_File, CheckInt32Int32FloatFloat)
// {
//   run_current_test<int32_t, int32_t, float, int>(
//     override_File_Usecase_with_cmd_line_arguments(GetParam()));
// }

// TEST_P(Tests_SGLookupEdgeSrcDst_File, CheckInt32Int64FloatFloat)
// {
//   run_current_test<int32_t, int64_t, float, int>(
//     override_File_Usecase_with_cmd_line_arguments(GetParam()));
// }

// TEST_P(Tests_SGLookupEdgeSrcDst_File, CheckInt64Int64FloatFloat)
// {
//   run_current_test<int64_t, int64_t, float, int>(
//     override_File_Usecase_with_cmd_line_arguments(GetParam()));
// }

TEST_P(Tests_SGLookupEdgeSrcDst_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

// TEST_P(Tests_SGLookupEdgeSrcDst_Rmat, CheckInt32Int64FloatFloat)
// {
//   run_current_test<int32_t, int64_t, float, int>(
//     override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
// }

// TEST_P(Tests_SGLookupEdgeSrcDst_Rmat, CheckInt64Int64FloatFloat)
// {
//   run_current_test<int64_t, int64_t, float, int>(
//     override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
// }

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SGLookupEdgeSrcDst_File,
  ::testing::Combine(::testing::Values(EdgeSrcDstLookup_UseCase{false},
                                       EdgeSrcDstLookup_UseCase{true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_SGLookupEdgeSrcDst_Rmat,
                         ::testing::Combine(::testing::Values(EdgeSrcDstLookup_UseCase{false},
                                                              EdgeSrcDstLookup_UseCase{true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              3, 3, 0.57, 0.19, 0.19, 0, true, false))));

// INSTANTIATE_TEST_SUITE_P(
//   rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
//                           --gtest_filter to select only the rmat_benchmark_test with a specific
//                           vertex & edge type combination) by command line arguments and do not
//                           include more than one Rmat_Usecase that differ only in scale or edge
//                           factor (to avoid running same benchmarks more than once) */
//   Tests_SGLookupEdgeSrcDst_Rmat,
//   ::testing::Combine(
//     ::testing::Values(EdgeSrcDstLookup_UseCase{false, false},
//                       EdgeSrcDstLookup_UseCase{true, false}),
//     ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
