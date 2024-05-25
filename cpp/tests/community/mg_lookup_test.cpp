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
#include <cugraph/utilities/dataframe_buffer.hpp>
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
    bool drop_multi_edges = false;  // FIXME: Need to check

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
    if (lookup_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      // mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    int32_t nr_hash_bins = 6 + static_cast<int>(mg_graph_view.number_of_vertices() / 100);

    std::cout << "nrbins: " << nr_hash_bins << "\n";
    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), int32_t>> edge_types{
      std::nullopt};
    edge_types = cugraph::test::generate<decltype(mg_graph_view), int32_t>::edge_property(
      *handle_, mg_graph_view, nr_hash_bins);

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), edge_t>> edge_ids{std::nullopt};

    edge_ids = cugraph::test::generate<decltype(mg_graph_view), edge_t>::edge_property(
      *handle_, mg_graph_view, 1);

    auto edge_counts = (*edge_ids).view().edge_counts();

    auto const comm_rank = (*handle_).get_comms().get_rank();
    auto const comm_size = (*handle_).get_comms().get_size();

    /*
    std::cout << "Rank: " << comm_rank << " edge couts: ";
    std::for_each(
      edge_counts.cbegin(), edge_counts.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
    std::cout << "\n";
    */

    std::vector<size_t> type_freqs(nr_hash_bins, 0);
    std::mutex mtx[nr_hash_bins];

    for (size_t ep_idx = 0; ep_idx < edge_counts.size(); ep_idx++) {
      auto ep_types =
        cugraph::test::to_host(*handle_,
                               raft::device_span<int32_t const>(
                                 (*edge_types).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      /*
      std::cout << "Rank: " << comm_rank << " ep_types: ";
      std::for_each(ep_types.cbegin(), ep_types.cend(), [nr_hash_bins](const int32_t& cnt) {
        if ((cnt < 0) || cnt > nr_hash_bins) std::cout << "Error: " << cnt << "\n";
        std::cout << cnt << " ";
      });
      std::cout << "\n";
      */

      std::for_each(std::execution::par, ep_types.begin(), ep_types.end(), [&](int32_t et) {
        std::lock_guard<std::mutex> guard(mtx[et]);
        type_freqs[et]++;
      });

    /*
      std::cout << "Rank: " << comm_rank << " type_freqs: ";
      std::for_each(
        type_freqs.cbegin(), type_freqs.cend(), [](const edge_t& f) { std::cout << f << " "; });
      std::cout << "\n";
    */
      auto ep_ids =
        cugraph::test::to_host(*handle_,
                               raft::device_span<edge_t const>(
                                 (*edge_ids).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      /*
      std::cout << "Rank: " << comm_rank << " ep_ids: ";
      std::for_each(
        ep_ids.cbegin(), ep_ids.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
      std::cout << "\n";
      */
    }

    assert(std::reduce(type_freqs.cbegin(), type_freqs.cend()) ==
           std::reduce(edge_counts.cbegin(), edge_counts.cend()));

    auto d_type_freqs = cugraph::test::to_device(*handle_, type_freqs);
    d_type_freqs =
      cugraph::test::device_allgatherv(*handle_, d_type_freqs.data(), d_type_freqs.size());
    type_freqs = cugraph::test::to_host(*handle_, d_type_freqs);

    /*
    std::cout << "Rank: " << comm_rank << " gathered_type_freqs: ";
    std::for_each(
      type_freqs.cbegin(), type_freqs.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
    std::cout << "\n";
    */
    std::vector<size_t> distributed_type_offsets(comm_size * nr_hash_bins);

    std::cout << "Rank: " << comm_rank << " copying to indices: ";
    for (size_t i = 0; i < nr_hash_bins; i++) {
      for (size_t j = 0; j < comm_size; j++) {
        // std::cout << (nr_hash_bins * j + i) << " ";
        distributed_type_offsets[j + comm_size * i] = type_freqs[nr_hash_bins * j + i];
        // auto xx                                     = (j + comm_size * i);
        // std::cout << xx << " ";
      }
    }
    std::cout << "\n";

    // prefix sum for each type
    for (size_t i = 0; i < nr_hash_bins; i++) {
      auto start = distributed_type_offsets.begin() + i * comm_size;
      std::exclusive_scan(start, start + comm_size, start, 0);
    }

    /*
    std::cout << "Rank: " << comm_rank << " prefix sum: ";
    std::for_each(distributed_type_offsets.cbegin(),
                  distributed_type_offsets.cend(),
                  [](const edge_t& cnt) { std::cout << cnt << " "; });
    std::cout << "\n";
    */

    assert(std::reduce(distributed_type_offsets.cbegin(), distributed_type_offsets.cend()) ==
           mg_graph_view.compute_number_of_edges(*handle_));

    std::cout << "Rank: " << comm_rank << " mg_graph_view: #V "
              << mg_graph_view.number_of_vertices()
              // << " #E " << mg_graph_view.number_of_edges()
              << " C#E " << mg_graph_view.compute_number_of_edges(*handle_) << "\n";

    auto number_of_local_edges = std::reduce(edge_counts.cbegin(), edge_counts.cend());

    for (size_t ep_idx = 0; ep_idx < edge_counts.size(); ep_idx++) {
      auto ep_types =
        cugraph::test::to_host(*handle_,
                               raft::device_span<int32_t const>(
                                 (*edge_types).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      
      /*
      std::cout << "Rank: " << comm_rank << " *ep_types: ";
      std::for_each(ep_types.cbegin(), ep_types.cend(), [nr_hash_bins](const int32_t& cnt) {
        if ((cnt < 0) || cnt > nr_hash_bins) std::cout << "Error: " << cnt << "\n";
        std::cout << cnt << " ";
      });
      std::cout << "\n";
      */

      auto ep_ids =
        cugraph::test::to_host(*handle_,
                               raft::device_span<edge_t const>(
                                 (*edge_ids).view().value_firsts()[ep_idx], edge_counts[ep_idx]));

      // std::transform(
      //   std::execution::par, ep_types.cbegin(), ep_types.cend(), ep_ids.begin(), [&](int32_t et)
      //   {
      //     edge_t val;
      //     std::lock_guard<std::mutex> guard(mtx[et]);
      //     {
      //       val = distributed_type_offsets[(comm_size * et + comm_rank)];
      //       distributed_type_offsets[(comm_size * et + comm_rank)]++;
      //     }
      //     return val;
      //   });

      std::transform(ep_types.cbegin(), ep_types.cend(), ep_ids.begin(), [&](int32_t et) {
        edge_t val = distributed_type_offsets[(comm_size * et + comm_rank)];
        distributed_type_offsets[(comm_size * et + comm_rank)]++;
        return val;
      });

      raft::update_device((*edge_ids).mutable_view().value_firsts()[ep_idx],
                          ep_ids.data(),
                          ep_ids.size(),
                          handle_->get_stream());

      /*
      std::cout << "Rank: " << comm_rank << " *ep_ids: ";
      std::for_each(
        ep_ids.cbegin(), ep_ids.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
      std::cout << "\n";
      */
    }

    /*
        std::vector<edge_t> edge_ids_to_lookup = {55, 55, 77, 77};

        rmm::device_uvector<edge_t> d_edge_ids_to_lookup(edge_ids_to_lookup.size(),
                                                         handle_->get_stream());
        raft::update_device(d_edge_ids_to_lookup.data(),
                            edge_ids_to_lookup.data(),
                            edge_ids_to_lookup.size(),
                            handle_->get_stream());

        std::vector<edge_t> edge_types_to_lookup = {1, 1, 0, 0};
        rmm::device_uvector<edge_t> d_edge_types_to_lookup(edge_types_to_lookup.size(),
                                                           handle_->get_stream());
        raft::update_device(d_edge_types_to_lookup.data(),
                            edge_types_to_lookup.data(),
                            edge_types_to_lookup.size(),
                            handle_->get_stream());
    */

    auto search_container =
      cugraph::build_edge_id_and_type_to_src_dst_lookup_map<vertex_t, edge_t, int32_t, multi_gpu>(
        *handle_, mg_graph_view, (*edge_ids).view(), (*edge_types).view());

    std::cout << "Rank: " << comm_rank << ">>>>>>>  Back to test code \n";
    // search_container.print();

    /*
        auto [srcs, dsts] = cugraph::
          cugraph_lookup_src_dst_from_edge_id_and_type_pub<vertex_t, edge_t, edge_t, multi_gpu>(
            *handle_,
            search_container,
            raft::device_span<edge_t>(d_edge_ids_to_lookup.begin(), d_edge_ids_to_lookup.size()),
            raft::device_span<edge_t>(d_edge_types_to_lookup.begin(),
       d_edge_types_to_lookup.size()));
    */
    if (lookup_usecase.check_correctness) {
      // std::vector<bool> flag_ids_exist   = {true, false};
      // std::vector<bool> flag_types_exist = {true, false};

      // for (int i = 0; i < flag_ids_exist.size(); i++) {
      //   for (int j = 0; j < flag_types_exist.size(); j++) {

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

      /*
      auto srcs_title = std::string("srcs_").append(std::to_string(comm_rank));

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      raft::print_device_vector(srcs_title.c_str(), d_mg_srcs.begin(), d_mg_srcs.size(), std::cout);

      auto dsts_title = std::string("dsts_").append(std::to_string(comm_rank));

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      raft::print_device_vector(dsts_title.c_str(), d_mg_dsts.begin(), d_mg_dsts.size(), std::cout);

      auto edge_types_title = std::string("edge_types_").append(std::to_string(comm_rank));

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      raft::print_device_vector(
        edge_types_title.c_str(), (*d_mg_edge_types).begin(), (*d_mg_edge_types).size(), std::cout);

      auto edge_ids_title = std::string("edge_ids_").append(std::to_string(comm_rank));

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      raft::print_device_vector(
        edge_ids_title.c_str(), (*d_mg_edge_ids).begin(), (*d_mg_edge_ids).size(), std::cout);

      */
      auto number_of_edges = mg_graph_view.compute_number_of_edges(*handle_);

      int nr_wrong_ids_or_types = (std::rand() % number_of_local_edges);

      auto h_mg_edge_ids   = cugraph::test::to_host(*handle_, d_mg_edge_ids);
      auto h_mg_edge_types = cugraph::test::to_host(*handle_, d_mg_edge_types);

      auto h_srcs_expected = cugraph::test::to_host(*handle_, d_mg_srcs);
      auto h_dsts_expected = cugraph::test::to_host(*handle_, d_mg_dsts);

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

      d_mg_edge_ids   = cugraph::test::to_device(*handle_, h_mg_edge_ids);
      d_mg_edge_types = cugraph::test::to_device(*handle_, h_mg_edge_types);

      /*
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
      */
      auto [srcs, dsts] = cugraph::
        cugraph_lookup_src_dst_from_edge_id_and_type_pub<vertex_t, edge_t, int32_t, multi_gpu>(
          *handle_,
          search_container,
          raft::device_span<edge_t>((*d_mg_edge_ids).begin(), (*d_mg_edge_ids).size()),
          raft::device_span<int32_t>((*d_mg_edge_types).begin(), (*d_mg_edge_types).size()));

      auto h_srcs_results = cugraph::test::to_host(*handle_, srcs);
      auto h_dsts_results = cugraph::test::to_host(*handle_, dsts);

      /*
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
      */

      EXPECT_EQ(h_srcs_expected.size(), h_srcs_results.size());
      ASSERT_TRUE(
        std::equal(h_srcs_expected.begin(), h_srcs_expected.end(), h_srcs_results.begin()));

      EXPECT_EQ(h_dsts_expected.size(), h_dsts_results.size());
      ASSERT_TRUE(
        std::equal(h_dsts_expected.begin(), h_dsts_expected.end(), h_dsts_results.begin()));

      //   }
      // }
    }

    // std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
    // std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
    // d_renumber_map_labels
    //   ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
    //                                                           (*d_renumber_map_labels).size())
    //   : std::nullopt

    //   for (size_t ep_idx = 0; ep_idx < edge_counts.size(); ep_idx++) {
    //     auto [srcs, dsts] =
    //       cugraph::cugraph_lookup_src_dst_from_edge_id_and_type_pub<vertex_t, edge_t, edge_t,
    //       multi_gpu>(
    //         handle,
    //         m,
    //         raft::device_span<edge_t>(d_edge_ids_to_lookup.begin(),
    //         d_edge_ids_to_lookup.size()),
    //         raft::device_span<edge_t>(d_edge_types_to_lookup.begin(),
    //         d_edge_types_to_lookup.size()));

    //   auto ep_types =
    //     cugraph::test::to_host(*handle_,
    //                            raft::device_span<int32_t const>(
    //                              (*edge_types).view().value_firsts()[ep_idx],
    //                              edge_counts[ep_idx]));

    //   auto ep_ids =
    //     cugraph::test::to_host(*handle_,
    //                            raft::device_span<edge_t const>(
    //                              (*edge_ids).view().value_firsts()[ep_idx],
    //                              edge_counts[ep_idx]));
    //   raft::update_device((*edge_ids).mutable_view().value_firsts()[ep_idx],
    //                       ep_ids.data(),
    //                       ep_ids.size(),
    //                       handle_->get_stream());

    //   std::cout << "Rank: " << comm_rank << " *ep_ids: ";
    //   std::for_each(
    //     ep_ids.cbegin(), ep_ids.cend(), [](const edge_t& cnt) { std::cout << cnt << " "; });
    //   std::cout << "\n";
    // }

    /*
    if (lookup_usecase.check_correctness) {
      weight_t mg_matching_weights;
      rmm::device_uvector<vertex_t> mg_partners(0, handle_->get_stream());
      auto h_mg_partners = cugraph::test::to_host(*handle_, mg_partners);

      auto constexpr invalid_partner = cugraph::invalid_vertex_id<vertex_t>::value;

      rmm::device_uvector<vertex_t> mg_aggregate_partners(0, handle_->get_stream());
      std::tie(std::ignore, mg_aggregate_partners) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<vertex_t const>(mg_partners.data(), mg_partners.size()));

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<raft::device_span<vertex_t const>>(std::nullopt),
        false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();

        rmm::device_uvector<vertex_t> sg_partners(0, handle_->get_stream());
        weight_t sg_matching_weights;

        std::forward_as_tuple(sg_partners, sg_matching_weights) =
          cugraph::approximate_weighted_matching<vertex_t, edge_t, weight_t, false>(
            *handle_, sg_graph_view, (*sg_edge_weights).view());
        auto h_sg_partners           = cugraph::test::to_host(*handle_, sg_partners);
        auto h_mg_aggregate_partners = cugraph::test::to_host(*handle_, mg_aggregate_partners);

        ASSERT_FLOAT_EQ(mg_matching_weights, sg_matching_weights)
          << "SG and MG matching weights are different";
        ASSERT_TRUE(
          std::equal(h_sg_partners.begin(), h_sg_partners.end(), h_mg_aggregate_partners.begin()));
      }
    }*/
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

// TEST_P(Tests_MGLookupEdgeSrcDst_File, CheckInt32Int64FloatFloat)
// {
//   run_current_test<int32_t, int64_t, float, int>(
//     override_File_Usecase_with_cmd_line_arguments(GetParam()));
// }

// TEST_P(Tests_MGLookupEdgeSrcDst_File, CheckInt64Int64FloatFloat)
// {
//   run_current_test<int64_t, int64_t, float, int>(
//     override_File_Usecase_with_cmd_line_arguments(GetParam()));
// }

TEST_P(Tests_MGLookupEdgeSrcDst_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

// TEST_P(Tests_MGLookupEdgeSrcDst_Rmat, CheckInt32Int64FloatFloat)
// {
//   run_current_test<int32_t, int64_t, float, int>(
//     override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
// }

// TEST_P(Tests_MGLookupEdgeSrcDst_Rmat, CheckInt64Int64FloatFloat)
// {
//   run_current_test<int64_t, int64_t, float, int>(
//     override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
// }

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGLookupEdgeSrcDst_File,
  ::testing::Combine(::testing::Values(EdgeSrcDstLookup_UseCase{false},
                                       EdgeSrcDstLookup_UseCase{true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGLookupEdgeSrcDst_Rmat,
                         ::testing::Combine(::testing::Values(EdgeSrcDstLookup_UseCase{false}
                                                              //  , EdgeSrcDstLookup_UseCase{true}
                                                              ),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              3, 2, 0.57, 0.19, 0.19, 0, true, false))));

// INSTANTIATE_TEST_SUITE_P(
//   rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
//                           --gtest_filter to select only the rmat_benchmark_test with a specific
//                           vertex & edge type combination) by command line arguments and do not
//                           include more than one Rmat_Usecase that differ only in scale or edge
//                           factor (to avoid running same benchmarks more than once) */
//   Tests_MGLookupEdgeSrcDst_Rmat,
//   ::testing::Combine(
//     ::testing::Values(EdgeSrcDstLookup_UseCase{false, false},
//                       EdgeSrcDstLookup_UseCase{true, false}),
//     ::testing::Values(cugraph::test::Rmat_Usecase(5, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
