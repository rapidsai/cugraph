/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "prims/per_v_transform_reduce_dst_key_aggregated_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "result_compare.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cuco/hash_functions.cuh>

#include <gtest/gtest.h>

#include <random>
#include <sstream>

template <typename vertex_t, typename edge_value_t, typename result_t>
struct key_aggregated_e_op_t {
  __device__ result_t operator()(vertex_t src,
                                 vertex_t key,
                                 result_t src_property,
                                 result_t key_property,
                                 edge_value_t edge_property) const
  {
    if (src_property < key_property) {
      return src_property;
    } else {
      return key_property;
    }
  }
};

struct Prims_Usecase {
  bool test_weighted{false};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of per_v_transform_reduce_incoming|outgoing_e primitive
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, prims_usecase.test_weighted, true);

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

    // 2. run MG per_v_transform_reduce_dst_key_aggregated_outgoing_e

    const int vertex_prop_hash_bin_count = 5;
    const int key_hash_bin_count         = 10;
    const int key_prop_hash_bin_count    = 20;
    const int initial_value              = 4;

    auto property_initial_value =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::initial_value(initial_value);

    auto mg_vertex_prop =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::vertex_property(
        *handle_, *mg_renumber_map, vertex_prop_hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<decltype(mg_graph_view), result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    auto mg_vertex_key =
      cugraph::test::generate<decltype(mg_graph_view), vertex_t>::vertex_property(
        *handle_, *mg_renumber_map, key_hash_bin_count);
    auto mg_dst_key = cugraph::test::generate<decltype(mg_graph_view), vertex_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_key);

    rmm::device_uvector<vertex_t> mg_kv_store_keys(comm_rank == 0 ? key_hash_bin_count : int{0},
                                                   handle_->get_stream());
    thrust::sequence(
      handle_->get_thrust_policy(), mg_kv_store_keys.begin(), mg_kv_store_keys.end(), vertex_t{0});
    mg_kv_store_keys = cugraph::shuffle_ext_vertices(*handle_, std::move(mg_kv_store_keys));
    auto mg_kv_store_values =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::vertex_property(
        *handle_, mg_kv_store_keys, key_prop_hash_bin_count);

    static_assert(std::is_same_v<result_t, int> ||
                  std::is_same_v<result_t, thrust::tuple<int, float>>);
    result_t invalid_value{};
    if constexpr (std::is_same_v<result_t, int>) {
      invalid_value = std::numeric_limits<int>::max();
    } else {
      invalid_value =
        thrust::make_tuple(std::numeric_limits<int>::max(), std::numeric_limits<float>::max());
    }
    cugraph::kv_store_t<vertex_t, result_t, false> mg_kv_store(
      mg_kv_store_keys.begin(),
      mg_kv_store_keys.end(),
      cugraph::get_dataframe_buffer_begin(mg_kv_store_values),
      cugraph::invalid_vertex_id<vertex_t>::value,
      invalid_value,
      handle_->get_stream());

    enum class reduction_type_t { PLUS, ELEMWISE_MIN, ELEMWISE_MAX };
    std::array<reduction_type_t, 3> reduction_types = {
      reduction_type_t::PLUS, reduction_type_t::ELEMWISE_MIN, reduction_type_t::ELEMWISE_MAX};

    std::vector<decltype(cugraph::allocate_dataframe_buffer<result_t>(0, rmm::cuda_stream_view{}))>
      mg_results{};
    mg_results.reserve(reduction_types.size());

    for (size_t i = 0; i < reduction_types.size(); ++i) {
      mg_results.push_back(cugraph::allocate_dataframe_buffer<result_t>(
        mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream()));

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle_->get_comms().barrier();
        hr_timer.start("MG per_v_transform_reduce_outgoing_e");
      }

      switch (reduction_types[i]) {
        case reduction_type_t::PLUS:
          if (mg_edge_weight_view) {
            per_v_transform_reduce_dst_key_aggregated_outgoing_e(
              *handle_,
              mg_graph_view,
              mg_src_prop.view(),
              *mg_edge_weight_view,
              mg_dst_key.view(),
              mg_kv_store.view(),
              key_aggregated_e_op_t<vertex_t, weight_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::plus<result_t>{},
              cugraph::get_dataframe_buffer_begin(mg_results[i]));
          } else {
            per_v_transform_reduce_dst_key_aggregated_outgoing_e(
              *handle_,
              mg_graph_view,
              mg_src_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              mg_dst_key.view(),
              mg_kv_store.view(),
              key_aggregated_e_op_t<vertex_t, cuda::std::nullopt_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::plus<result_t>{},
              cugraph::get_dataframe_buffer_begin(mg_results[i]));
          }
          break;
        case reduction_type_t::ELEMWISE_MIN:
          if (mg_edge_weight_view) {
            per_v_transform_reduce_dst_key_aggregated_outgoing_e(
              *handle_,
              mg_graph_view,
              mg_src_prop.view(),
              *mg_edge_weight_view,
              mg_dst_key.view(),
              mg_kv_store.view(),
              key_aggregated_e_op_t<vertex_t, weight_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::elementwise_minimum<result_t>{},
              cugraph::get_dataframe_buffer_begin(mg_results[i]));
          } else {
            per_v_transform_reduce_dst_key_aggregated_outgoing_e(
              *handle_,
              mg_graph_view,
              mg_src_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              mg_dst_key.view(),
              mg_kv_store.view(),
              key_aggregated_e_op_t<vertex_t, cuda::std::nullopt_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::elementwise_minimum<result_t>{},
              cugraph::get_dataframe_buffer_begin(mg_results[i]));
          }
          break;
        case reduction_type_t::ELEMWISE_MAX:
          if (mg_edge_weight_view) {
            per_v_transform_reduce_dst_key_aggregated_outgoing_e(
              *handle_,
              mg_graph_view,
              mg_src_prop.view(),
              *mg_edge_weight_view,
              mg_dst_key.view(),
              mg_kv_store.view(),
              key_aggregated_e_op_t<vertex_t, weight_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::elementwise_maximum<result_t>{},
              cugraph::get_dataframe_buffer_begin(mg_results[i]));
          } else {
            per_v_transform_reduce_dst_key_aggregated_outgoing_e(
              *handle_,
              mg_graph_view,
              mg_src_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              mg_dst_key.view(),
              mg_kv_store.view(),
              key_aggregated_e_op_t<vertex_t, cuda::std::nullopt_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::elementwise_maximum<result_t>{},
              cugraph::get_dataframe_buffer_begin(mg_results[i]));
          }
          break;
        default: FAIL() << "should not be reached.";
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle_->get_comms().barrier();
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }
    }

    // 3. compare SG & MG results

    if (prims_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      for (size_t i = 0; i < reduction_types.size(); ++i) {
        auto mg_aggregate_results =
          cugraph::allocate_dataframe_buffer<result_t>(0, handle_->get_stream());

        static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<result_t>::value);
        if constexpr (std::is_arithmetic_v<result_t>) {
          std::tie(std::ignore, mg_aggregate_results) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<result_t const>(mg_results[i].data(), mg_results[i].size()));
        } else {
          std::tie(std::ignore, std::get<0>(mg_aggregate_results)) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<typename thrust::tuple_element<0, result_t>::type const>(
                std::get<0>(mg_results[i]).data(), std::get<0>(mg_results[i]).size()));

          std::tie(std::ignore, std::get<1>(mg_aggregate_results)) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<typename thrust::tuple_element<1, result_t>::type const>(
                std::get<1>(mg_results[i]).data(), std::get<1>(mg_results[i]).size()));
        }

        if (handle_->get_comms().get_rank() == int{0}) {
          auto sg_graph_view = sg_graph.view();
          auto sg_edge_weight_view =
            sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

          auto sg_vertex_prop =
            cugraph::test::generate<decltype(sg_graph_view), result_t>::vertex_property(
              *handle_,
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
              vertex_prop_hash_bin_count);
          auto sg_src_prop =
            cugraph::test::generate<decltype(sg_graph_view), result_t>::src_property(
              *handle_, sg_graph_view, sg_vertex_prop);

          auto sg_vertex_key =
            cugraph::test::generate<decltype(sg_graph_view), vertex_t>::vertex_property(
              *handle_,
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
              key_hash_bin_count);
          auto sg_dst_key =
            cugraph::test::generate<decltype(sg_graph_view), vertex_t>::dst_property(
              *handle_, sg_graph_view, sg_vertex_key);

          rmm::device_uvector<vertex_t> sg_kv_store_keys(key_hash_bin_count, handle_->get_stream());
          thrust::sequence(handle_->get_thrust_policy(),
                           sg_kv_store_keys.begin(),
                           sg_kv_store_keys.end(),
                           vertex_t{0});
          auto sg_kv_store_values =
            cugraph::test::generate<decltype(sg_graph_view), result_t>::vertex_property(
              *handle_, sg_kv_store_keys, key_prop_hash_bin_count);

          cugraph::kv_store_t<vertex_t, result_t, false> sg_kv_store(
            sg_kv_store_keys.begin(),
            sg_kv_store_keys.end(),
            cugraph::get_dataframe_buffer_begin(sg_kv_store_values),
            cugraph::invalid_vertex_id<vertex_t>::value,
            invalid_value,
            handle_->get_stream());

          cugraph::test::vector_result_compare compare{*handle_};

          auto global_result = cugraph::allocate_dataframe_buffer<result_t>(
            sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

          switch (reduction_types[i]) {
            case reduction_type_t::PLUS:
              if (sg_edge_weight_view) {
                per_v_transform_reduce_dst_key_aggregated_outgoing_e(
                  *handle_,
                  sg_graph_view,
                  sg_src_prop.view(),
                  *sg_edge_weight_view,
                  sg_dst_key.view(),
                  sg_kv_store.view(),
                  key_aggregated_e_op_t<vertex_t, weight_t, result_t>{},
                  property_initial_value,
                  cugraph::reduce_op::plus<result_t>{},
                  cugraph::get_dataframe_buffer_begin(global_result));
              } else {
                per_v_transform_reduce_dst_key_aggregated_outgoing_e(
                  *handle_,
                  sg_graph_view,
                  sg_src_prop.view(),
                  cugraph::edge_dummy_property_t{}.view(),
                  sg_dst_key.view(),
                  sg_kv_store.view(),
                  key_aggregated_e_op_t<vertex_t, cuda::std::nullopt_t, result_t>{},
                  property_initial_value,
                  cugraph::reduce_op::plus<result_t>{},
                  cugraph::get_dataframe_buffer_begin(global_result));
              }
              break;
            case reduction_type_t::ELEMWISE_MIN:
              if (sg_edge_weight_view) {
                per_v_transform_reduce_dst_key_aggregated_outgoing_e(
                  *handle_,
                  sg_graph_view,
                  sg_src_prop.view(),
                  *sg_edge_weight_view,
                  sg_dst_key.view(),
                  sg_kv_store.view(),
                  key_aggregated_e_op_t<vertex_t, weight_t, result_t>{},
                  property_initial_value,
                  cugraph::reduce_op::elementwise_minimum<result_t>{},
                  cugraph::get_dataframe_buffer_begin(global_result));
              } else {
                per_v_transform_reduce_dst_key_aggregated_outgoing_e(
                  *handle_,
                  sg_graph_view,
                  sg_src_prop.view(),
                  cugraph::edge_dummy_property_t{}.view(),
                  sg_dst_key.view(),
                  sg_kv_store.view(),
                  key_aggregated_e_op_t<vertex_t, cuda::std::nullopt_t, result_t>{},
                  property_initial_value,
                  cugraph::reduce_op::elementwise_minimum<result_t>{},
                  cugraph::get_dataframe_buffer_begin(global_result));
              }
              break;
            case reduction_type_t::ELEMWISE_MAX:
              if (sg_edge_weight_view) {
                per_v_transform_reduce_dst_key_aggregated_outgoing_e(
                  *handle_,
                  sg_graph_view,
                  sg_src_prop.view(),
                  *sg_edge_weight_view,
                  sg_dst_key.view(),
                  sg_kv_store.view(),
                  key_aggregated_e_op_t<vertex_t, weight_t, result_t>{},
                  property_initial_value,
                  cugraph::reduce_op::elementwise_maximum<result_t>{},
                  cugraph::get_dataframe_buffer_begin(global_result));
              } else {
                per_v_transform_reduce_dst_key_aggregated_outgoing_e(
                  *handle_,
                  sg_graph_view,
                  sg_src_prop.view(),
                  cugraph::edge_dummy_property_t{}.view(),
                  sg_dst_key.view(),
                  sg_kv_store.view(),
                  key_aggregated_e_op_t<vertex_t, cuda::std::nullopt_t, result_t>{},
                  property_initial_value,
                  cugraph::reduce_op::elementwise_maximum<result_t>{},
                  cugraph::get_dataframe_buffer_begin(global_result));
              }
              break;
            default: FAIL() << "should not be reached.";
          }

          ASSERT_TRUE(compare(mg_aggregate_results, global_result));
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE<input_usecase_t>::handle_ = nullptr;

using Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_File =
  Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE<cugraph::test::File_Usecase>;
using Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_Rmat =
  Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_File,
       CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(std::get<0>(param),
                                                                       std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_Rmat,
       CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_Rmat,
       CheckInt64Int64FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_File,
       CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_Rmat,
       CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_Rmat,
       CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, false, true},
                                       Prims_Usecase{false, true, true},
                                       Prims_Usecase{true, false, true},
                                       Prims_Usecase{true, true, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, true},
                      Prims_Usecase{false, true, true},
                      Prims_Usecase{true, false, true},
                      Prims_Usecase{true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_Rmat,
                         ::testing::Combine(::testing::Values(Prims_Usecase{false, false, true},
                                                              Prims_Usecase{false, true, true},
                                                              Prims_Usecase{true, false, true},
                                                              Prims_Usecase{true, true, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGPerVTransformReduceDstKeyAggregatedOutgoingE_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, false},
                      Prims_Usecase{false, true, false},
                      Prims_Usecase{true, false, false},
                      Prims_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
