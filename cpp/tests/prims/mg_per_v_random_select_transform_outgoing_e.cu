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

#include "prims/per_v_random_select_transform_outgoing_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_kernels.cuh"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename bias_t>
struct e_bias_op_t {
  __device__ bias_t
  operator()(vertex_t, vertex_t, thrust::nullopt_t, thrust::nullopt_t, bias_t bias) const
  {
    return bias;
  }
};

template <typename vertex_t, typename weight_t, typename property_t>
struct e_op_t {
  using result_t = decltype(cugraph::thrust_tuple_cat(thrust::tuple<vertex_t, vertex_t>{},
                                                      cugraph::to_thrust_tuple(property_t{}),
                                                      cugraph::to_thrust_tuple(property_t{})));

  __device__ result_t operator()(
    vertex_t src, vertex_t dst, property_t src_prop, property_t dst_prop, thrust::nullopt_t) const
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
      static_assert(thrust::tuple_size<property_t>::value == size_t{2});
      return thrust::make_tuple(src,
                                dst,
                                thrust::get<0>(src_prop),
                                thrust::get<1>(src_prop),
                                thrust::get<0>(dst_prop),
                                thrust::get<1>(dst_prop));
    } else {
      return thrust::make_tuple(src, dst, src_prop, dst_prop);
    }
  }

  __device__ result_t
  operator()(vertex_t src, vertex_t dst, property_t src_prop, property_t dst_prop, weight_t w) const
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
      static_assert(thrust::tuple_size<property_t>::value == size_t{2});
      return thrust::make_tuple(src,
                                dst,
                                thrust::get<0>(src_prop),
                                thrust::get<1>(src_prop),
                                thrust::get<0>(dst_prop),
                                thrust::get<1>(dst_prop));
    } else {
      return thrust::make_tuple(src, dst, src_prop, dst_prop);
    }
  }
};

struct Prims_Usecase {
  size_t num_seeds{0};
  size_t K{0};
  bool with_replacement{false};
  bool use_invalid_value{false};
  bool use_weight_as_bias{false};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPerVRandomSelectTransformOutgoingE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGPerVRandomSelectTransformOutgoingE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Verify the results of per_v_random_select_transform_outgoing_e primitive
  template <typename vertex_t, typename edge_t, typename weight_t, typename property_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, prims_usecase.use_weight_as_bias, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (prims_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG per_v_random_select_transform_outgoing_e primitive

    const int hash_bin_count = 5;

    auto mg_vertex_prop =
      cugraph::test::generate<decltype(mg_graph_view), property_t>::vertex_property(
        *handle_, *mg_renumber_map, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<decltype(mg_graph_view), property_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<decltype(mg_graph_view), property_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    raft::random::RngState rng_state(static_cast<uint64_t>(handle_->get_comms().get_rank()));

    auto select_count     = prims_usecase.with_replacement
                              ? prims_usecase.num_seeds
                              : std::min(prims_usecase.num_seeds,
                                     static_cast<size_t>(mg_graph_view.number_of_vertices()));
    auto mg_vertex_buffer = cugraph::select_random_vertices(
      *handle_,
      mg_graph_view,
      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      rng_state,
      select_count,
      prims_usecase.with_replacement,
      false);

    constexpr size_t bucket_idx_cur = 0;
    constexpr size_t num_buckets    = 1;

    cugraph::vertex_frontier_t<vertex_t, void, true, false> mg_vertex_frontier(*handle_,
                                                                               num_buckets);
    mg_vertex_frontier.bucket(bucket_idx_cur)
      .insert(cugraph::get_dataframe_buffer_begin(mg_vertex_buffer),
              cugraph::get_dataframe_buffer_end(mg_vertex_buffer));

    using result_t = decltype(cugraph::thrust_tuple_cat(thrust::tuple<vertex_t, vertex_t>{},
                                                        cugraph::to_thrust_tuple(property_t{}),
                                                        cugraph::to_thrust_tuple(property_t{})));

    std::optional<result_t> invalid_value{std::nullopt};
    if (prims_usecase.use_invalid_value) {
      invalid_value                  = result_t{};
      thrust::get<0>(*invalid_value) = cugraph::invalid_vertex_id<vertex_t>::value;
      thrust::get<1>(*invalid_value) = cugraph::invalid_vertex_id<vertex_t>::value;
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG per_v_random_select_transform_outgoing_e");
    }

    auto [mg_sample_offsets, mg_sample_e_op_results] =
      prims_usecase.use_weight_as_bias ? cugraph::per_v_random_select_transform_outgoing_e(
                                           *handle_,
                                           mg_graph_view,
                                           mg_vertex_frontier.bucket(bucket_idx_cur),
                                           cugraph::edge_src_dummy_property_t{}.view(),
                                           cugraph::edge_dst_dummy_property_t{}.view(),
                                           *mg_edge_weight_view,
                                           e_bias_op_t<vertex_t, weight_t>{},
                                           mg_src_prop.view(),
                                           mg_dst_prop.view(),
                                           *mg_edge_weight_view,
                                           e_op_t<vertex_t, weight_t, property_t>{},
                                           rng_state,
                                           prims_usecase.K,
                                           prims_usecase.with_replacement,
                                           invalid_value)
                                       : cugraph::per_v_random_select_transform_outgoing_e(
                                           *handle_,
                                           mg_graph_view,
                                           mg_vertex_frontier.bucket(bucket_idx_cur),
                                           mg_src_prop.view(),
                                           mg_dst_prop.view(),
                                           cugraph::edge_dummy_property_t{}.view(),
                                           e_op_t<vertex_t, weight_t, property_t>{},
                                           rng_state,
                                           prims_usecase.K,
                                           prims_usecase.with_replacement,
                                           invalid_value);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. validate MG results

    if (prims_usecase.check_correctness) {
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        mg_vertex_frontier.bucket(bucket_idx_cur).begin(),
        mg_vertex_frontier.bucket(bucket_idx_cur).size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      std::optional<rmm::device_uvector<size_t>> mg_sample_counts{std::nullopt};
      if (mg_sample_offsets) {
        mg_sample_counts = rmm::device_uvector<size_t>(
          mg_vertex_frontier.bucket(bucket_idx_cur).size(), handle_->get_stream());
        thrust::adjacent_difference(handle_->get_thrust_policy(),
                                    (*mg_sample_offsets).begin() + 1,
                                    (*mg_sample_offsets).end(),
                                    (*mg_sample_counts).begin());
      }

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        std::get<0>(mg_sample_e_op_results).data(),
        std::get<0>(mg_sample_e_op_results).size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        std::get<1>(mg_sample_e_op_results).data(),
        std::get<1>(mg_sample_e_op_results).size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      auto mg_aggregate_frontier_vertices = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(mg_vertex_frontier.bucket(bucket_idx_cur).begin(),
                                          mg_vertex_frontier.bucket(bucket_idx_cur).size()));

      std::optional<rmm::device_uvector<size_t>> mg_aggregate_sample_counts{std::nullopt};
      if (mg_sample_counts) {
        mg_aggregate_sample_counts = cugraph::test::device_gatherv(
          *handle_,
          raft::device_span<size_t const>((*mg_sample_counts).data(), (*mg_sample_counts).size()));
      }

      auto mg_aggregate_sample_e_op_results =
        cugraph::allocate_dataframe_buffer<result_t>(0, handle_->get_stream());
      std::get<0>(mg_aggregate_sample_e_op_results) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<0>(mg_sample_e_op_results).data(),
                                      std::get<0>(mg_sample_e_op_results).size());
      std::get<1>(mg_aggregate_sample_e_op_results) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<1>(mg_sample_e_op_results).data(),
                                      std::get<1>(mg_sample_e_op_results).size());
      std::get<2>(mg_aggregate_sample_e_op_results) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<2>(mg_sample_e_op_results).data(),
                                      std::get<2>(mg_sample_e_op_results).size());
      std::get<3>(mg_aggregate_sample_e_op_results) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<3>(mg_sample_e_op_results).data(),
                                      std::get<3>(mg_sample_e_op_results).size());
      if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
        std::get<4>(mg_aggregate_sample_e_op_results) =
          cugraph::test::device_gatherv(*handle_,
                                        std::get<4>(mg_sample_e_op_results).data(),
                                        std::get<4>(mg_sample_e_op_results).size());
        std::get<5>(mg_aggregate_sample_e_op_results) =
          cugraph::test::device_gatherv(*handle_,
                                        std::get<5>(mg_sample_e_op_results).data(),
                                        std::get<5>(mg_sample_e_op_results).size());
      }

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                              (*mg_renumber_map).size()),
        false);

      if (handle_->get_comms().get_rank() == 0) {
        std::optional<rmm::device_uvector<size_t>> mg_aggregate_sample_offsets{std::nullopt};
        if (mg_aggregate_sample_counts) {
          mg_aggregate_sample_offsets = rmm::device_uvector<size_t>(
            (*mg_aggregate_sample_counts).size() + 1, handle_->get_stream());
          (*mg_aggregate_sample_offsets).set_element_to_zero_async(0, handle_->get_stream());
          thrust::inclusive_scan(handle_->get_thrust_policy(),
                                 (*mg_aggregate_sample_counts).begin(),
                                 (*mg_aggregate_sample_counts).end(),
                                 (*mg_aggregate_sample_offsets).begin() + 1);
        }

        auto sg_graph_view = sg_graph.view();

        rmm::device_uvector<edge_t> sg_offsets(sg_graph_view.number_of_vertices() + vertex_t{1},
                                               handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     sg_graph_view.local_edge_partition_view().offsets().begin(),
                     sg_graph_view.local_edge_partition_view().offsets().end(),
                     sg_offsets.begin());
        rmm::device_uvector<vertex_t> sg_indices(
          sg_graph_view.local_edge_partition_view().indices().size(), handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     sg_graph_view.local_edge_partition_view().indices().begin(),
                     sg_graph_view.local_edge_partition_view().indices().end(),
                     sg_indices.begin());

        auto num_invalids = static_cast<size_t>(thrust::count_if(
          handle_->get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(mg_aggregate_frontier_vertices.size()),
          [frontier_vertex_first = mg_aggregate_frontier_vertices.begin(),
           sample_offsets = mg_aggregate_sample_offsets ? thrust::make_optional<size_t const*>(
                                                            (*mg_aggregate_sample_offsets).data())
                                                        : thrust::nullopt,
           sample_e_op_result_first =
             cugraph::get_dataframe_buffer_begin(mg_aggregate_sample_e_op_results),
           sg_offsets       = sg_offsets.begin(),
           sg_indices       = sg_indices.begin(),
           K                = prims_usecase.K,
           with_replacement = prims_usecase.with_replacement,
           invalid_value =
             invalid_value ? thrust::make_optional<result_t>(*invalid_value) : thrust::nullopt,
           property_transform =
             cugraph::test::detail::vertex_property_transform<vertex_t, property_t>{
               hash_bin_count}] __device__(size_t i) {
            auto v = *(frontier_vertex_first + i);

            // check sample_offsets

            auto offset_first = sample_offsets ? *(*sample_offsets + i) : K * i;
            auto offset_last  = sample_offsets ? *(*sample_offsets + (i + 1)) : K * (i + 1);
            if (!sample_offsets) {
              size_t num_valids{0};
              for (size_t j = offset_first; j < offset_last; ++j) {
                auto e_op_result = *(sample_e_op_result_first + j);
                if (e_op_result == *invalid_value) { break; }
                ++num_valids;
              }
              for (size_t j = offset_first + num_valids; j < offset_last; ++j) {
                auto e_op_result = *(sample_e_op_result_first + j);
                if (e_op_result != *invalid_value) { return true; }
              }
              offset_last = offset_first + num_valids;
            }
            auto count = offset_last - offset_first;

            auto out_degree = *(sg_offsets + v + 1) - *(sg_offsets + v);
            if (with_replacement) {
              if ((out_degree > 0 && count != K) || (out_degree == 0 && count != 0)) {
                return true;
              }
            } else {
              if (count != std::min(static_cast<size_t>(out_degree), K)) { return true; }
            }

            // check sample_e_op_results

            for (size_t j = offset_first; j < offset_last; ++j) {
              auto e_op_result  = *(sample_e_op_result_first + j);
              auto sg_src       = thrust::get<0>(e_op_result);
              auto sg_dst       = thrust::get<1>(e_op_result);
              auto sg_nbr_first = sg_indices + *(sg_offsets + sg_src);
              auto sg_nbr_last  = sg_indices + *(sg_offsets + (sg_src + vertex_t{1}));
              if (!thrust::binary_search(thrust::seq,
                                         sg_nbr_first,
                                         sg_nbr_last,
                                         sg_dst)) {  // assumed neighbor lists are sorted
                return true;
              }
              property_t src_val{};
              property_t dst_val{};
              if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
                src_val =
                  thrust::make_tuple(thrust::get<2>(e_op_result), thrust::get<3>(e_op_result));
                dst_val =
                  thrust::make_tuple(thrust::get<4>(e_op_result), thrust::get<5>(e_op_result));
              } else {
                src_val = thrust::get<2>(e_op_result);
                dst_val = thrust::get<3>(e_op_result);
              }
              if (src_val != property_transform(sg_src)) { return true; }
              if (dst_val != property_transform(sg_dst)) { return true; }

              if (!with_replacement) {
                auto sg_dst_first =
                  thrust::get<1>(sample_e_op_result_first.get_iterator_tuple()) + offset_first;
                auto sg_dst_last =
                  thrust::get<1>(sample_e_op_result_first.get_iterator_tuple()) + offset_last;
                auto dst_count =
                  thrust::count(thrust::seq,
                                sg_dst_first,
                                sg_dst_last,
                                sg_dst);  // this could be inefficient for high-degree vertices, if
                                          // we sort [sg_dst_first, sg_dst_last) we can use binary
                                          // search but we may better not modify the sampling output
                                          // and allow inefficiency as this is just for testing
                auto multiplicity = thrust::distance(
                  thrust::lower_bound(thrust::seq, sg_nbr_first, sg_nbr_last, sg_dst),
                  thrust::upper_bound(thrust::seq,
                                      sg_nbr_first,
                                      sg_nbr_last,
                                      sg_dst));  // this assumes neighbor lists are sorted
                if (dst_count > multiplicity) { return true; }
              }
            }

            return false;
          }));

        ASSERT_TRUE(num_invalids == 0);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGPerVRandomSelectTransformOutgoingE<input_usecase_t>::handle_ = nullptr;

using Tests_MGPerVRandomSelectTransformOutgoingE_File =
  Tests_MGPerVRandomSelectTransformOutgoingE<cugraph::test::File_Usecase>;
using Tests_MGPerVRandomSelectTransformOutgoingE_Rmat =
  Tests_MGPerVRandomSelectTransformOutgoingE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_File, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(std::get<0>(param),
                                                                       std::get<1>(param));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt32Int64FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt64Int64FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVRandomSelectTransformOutgoingE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{size_t{1000}, size_t{4}, false, false, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, false, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, false, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, false, true, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, true, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, true, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGPerVRandomSelectTransformOutgoingE_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{size_t{1000}, size_t{4}, false, false, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, false, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, false, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, false, true, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, false, true, true, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, false, true, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, false, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, false, true},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, true, false},
                      Prims_Usecase{size_t{1000}, size_t{4}, true, true, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGPerVRandomSelectTransformOutgoingE_Rmat,
  ::testing::Combine(
    ::testing::Values(
      Prims_Usecase{size_t{10000000}, size_t{25}, false, false, false, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, false, false, false, true, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, false, false, true, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, false, false, true, true, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, false, true, false, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, false, true, false, true, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, false, true, true, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, false, true, true, true, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, false, false, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, false, false, true, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, false, true, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, false, true, true, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, true, false, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, true, false, true, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, true, true, false, false},
      Prims_Usecase{size_t{10000000}, size_t{25}, true, true, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
