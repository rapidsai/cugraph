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

#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
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

template <typename vertex_t, typename result_t>
struct e_op_t {
  __device__ result_t operator()(vertex_t src,
                                 vertex_t dst,
                                 result_t src_property,
                                 result_t dst_property,
                                 cuda::std::nullopt_t) const
  {
    if (src_property < dst_property) {
      return src_property;
    } else {
      return dst_property;
    }
  }
};

struct Prims_Usecase {
  bool test_weighted{false};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPerVTransformReduceIncomingOutgoingE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGPerVTransformReduceIncomingOutgoingE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of per_v_transform_reduce_incoming|outgoing_e primitive
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename result_t,
            bool store_transposed>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

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
        *handle_, input_usecase, prims_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (prims_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG transform reduce

    const int hash_bin_count = 5;
    const int initial_value  = 4;

    auto property_initial_value =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::initial_value(initial_value);

    auto mg_vertex_prop =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::vertex_property(
        *handle_, *mg_renumber_map, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<decltype(mg_graph_view), result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<decltype(mg_graph_view), result_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    enum class reduction_type_t { PLUS, ELEMWISE_MIN, ELEMWISE_MAX };
    std::array<reduction_type_t, 3> reduction_types = {
      reduction_type_t::PLUS, reduction_type_t::ELEMWISE_MIN, reduction_type_t::ELEMWISE_MAX};

    std::vector<decltype(cugraph::allocate_dataframe_buffer<result_t>(0, rmm::cuda_stream_view{}))>
      mg_in_results{};
    std::vector<decltype(cugraph::allocate_dataframe_buffer<result_t>(0, rmm::cuda_stream_view{}))>
      mg_out_results{};
    mg_in_results.reserve(reduction_types.size());
    mg_out_results.reserve(reduction_types.size());

    for (size_t i = 0; i < reduction_types.size(); ++i) {
      mg_in_results.push_back(cugraph::allocate_dataframe_buffer<result_t>(
        mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream()));

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle_->get_comms().barrier();
        hr_timer.start("MG per_v_transform_reduce_incoming_e");
      }

      switch (reduction_types[i]) {
        case reduction_type_t::PLUS:
          per_v_transform_reduce_incoming_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::plus<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(mg_in_results[i]));
          break;
        case reduction_type_t::ELEMWISE_MIN:
          per_v_transform_reduce_incoming_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::elementwise_minimum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(mg_in_results[i]));
          break;
        case reduction_type_t::ELEMWISE_MAX:
          per_v_transform_reduce_incoming_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::elementwise_maximum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(mg_in_results[i]));
          break;
        default: FAIL() << "should not be reached.";
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle_->get_comms().barrier();
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      mg_out_results.push_back(cugraph::allocate_dataframe_buffer<result_t>(
        mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream()));

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle_->get_comms().barrier();
        hr_timer.start("MG per_v_transform_reduce_outgoing_e");
      }

      switch (reduction_types[i]) {
        case reduction_type_t::PLUS:
          per_v_transform_reduce_outgoing_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::plus<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(mg_out_results[i]));
          break;
        case reduction_type_t::ELEMWISE_MIN:
          per_v_transform_reduce_outgoing_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::elementwise_minimum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(mg_out_results[i]));
          break;
        case reduction_type_t::ELEMWISE_MAX:
          per_v_transform_reduce_outgoing_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::elementwise_maximum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(mg_out_results[i]));
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
      cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore, std::ignore) =
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
        auto mg_aggregate_in_results =
          cugraph::allocate_dataframe_buffer<result_t>(0, handle_->get_stream());
        auto mg_aggregate_out_results =
          cugraph::allocate_dataframe_buffer<result_t>(0, handle_->get_stream());

        static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<result_t>::value);
        if constexpr (std::is_arithmetic_v<result_t>) {
          std::tie(std::ignore, mg_aggregate_in_results) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<result_t const>(mg_in_results[i].data(), mg_in_results[i].size()));

          std::tie(std::ignore, mg_aggregate_out_results) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<result_t const>(mg_out_results[i].data(),
                                                mg_out_results[i].size()));
        } else {
          std::tie(std::ignore, std::get<0>(mg_aggregate_in_results)) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<typename thrust::tuple_element<0, result_t>::type const>(
                std::get<0>(mg_in_results[i]).data(), std::get<0>(mg_in_results[i]).size()));

          std::tie(std::ignore, std::get<1>(mg_aggregate_in_results)) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<typename thrust::tuple_element<1, result_t>::type const>(
                std::get<1>(mg_in_results[i]).data(), std::get<1>(mg_in_results[i]).size()));

          std::tie(std::ignore, std::get<0>(mg_aggregate_out_results)) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<typename thrust::tuple_element<0, result_t>::type const>(
                std::get<0>(mg_out_results[i]).data(), std::get<0>(mg_out_results[i]).size()));

          std::tie(std::ignore, std::get<1>(mg_aggregate_out_results)) =
            cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
              *handle_,
              std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                    (*mg_renumber_map).size()),
              mg_graph_view.local_vertex_partition_range(),
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              std::optional<raft::device_span<vertex_t const>>{std::nullopt},
              raft::device_span<typename thrust::tuple_element<1, result_t>::type const>(
                std::get<1>(mg_out_results[i]).data(), std::get<1>(mg_out_results[i]).size()));
        }

        if (handle_->get_comms().get_rank() == int{0}) {
          auto sg_graph_view = sg_graph.view();

          auto sg_vertex_prop =
            cugraph::test::generate<decltype(sg_graph_view), result_t>::vertex_property(
              *handle_,
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
              hash_bin_count);
          auto sg_src_prop =
            cugraph::test::generate<decltype(sg_graph_view), result_t>::src_property(
              *handle_, sg_graph_view, sg_vertex_prop);
          auto sg_dst_prop =
            cugraph::test::generate<decltype(sg_graph_view), result_t>::dst_property(
              *handle_, sg_graph_view, sg_vertex_prop);
          cugraph::test::vector_result_compare compare{*handle_};

          auto global_in_result = cugraph::allocate_dataframe_buffer<result_t>(
            sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

          switch (reduction_types[i]) {
            case reduction_type_t::PLUS:
              per_v_transform_reduce_incoming_e(
                *handle_,
                sg_graph_view,
                sg_src_prop.view(),
                sg_dst_prop.view(),
                cugraph::edge_dummy_property_t{}.view(),
                e_op_t<vertex_t, result_t>{},
                property_initial_value,
                cugraph::reduce_op::plus<result_t>{},
                cugraph::get_dataframe_buffer_begin(global_in_result));
              break;
            case reduction_type_t::ELEMWISE_MIN:
              per_v_transform_reduce_incoming_e(
                *handle_,
                sg_graph_view,
                sg_src_prop.view(),
                sg_dst_prop.view(),
                cugraph::edge_dummy_property_t{}.view(),
                e_op_t<vertex_t, result_t>{},
                property_initial_value,
                cugraph::reduce_op::elementwise_minimum<result_t>{},
                cugraph::get_dataframe_buffer_begin(global_in_result));
              break;
            case reduction_type_t::ELEMWISE_MAX:
              per_v_transform_reduce_incoming_e(
                *handle_,
                sg_graph_view,
                sg_src_prop.view(),
                sg_dst_prop.view(),
                cugraph::edge_dummy_property_t{}.view(),
                e_op_t<vertex_t, result_t>{},
                property_initial_value,
                cugraph::reduce_op::elementwise_maximum<result_t>{},
                cugraph::get_dataframe_buffer_begin(global_in_result));
              break;
            default: FAIL() << "should not be reached.";
          }

          auto global_out_result = cugraph::allocate_dataframe_buffer<result_t>(
            sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

          switch (reduction_types[i]) {
            case reduction_type_t::PLUS:
              per_v_transform_reduce_outgoing_e(
                *handle_,
                sg_graph_view,
                sg_src_prop.view(),
                sg_dst_prop.view(),
                cugraph::edge_dummy_property_t{}.view(),
                e_op_t<vertex_t, result_t>{},
                property_initial_value,
                cugraph::reduce_op::plus<result_t>{},
                cugraph::get_dataframe_buffer_begin(global_out_result));
              break;
            case reduction_type_t::ELEMWISE_MIN:
              per_v_transform_reduce_outgoing_e(
                *handle_,
                sg_graph_view,
                sg_src_prop.view(),
                sg_dst_prop.view(),
                cugraph::edge_dummy_property_t{}.view(),
                e_op_t<vertex_t, result_t>{},
                property_initial_value,
                cugraph::reduce_op::elementwise_minimum<result_t>{},
                cugraph::get_dataframe_buffer_begin(global_out_result));
              break;
            case reduction_type_t::ELEMWISE_MAX:
              per_v_transform_reduce_outgoing_e(
                *handle_,
                sg_graph_view,
                sg_src_prop.view(),
                sg_dst_prop.view(),
                cugraph::edge_dummy_property_t{}.view(),
                e_op_t<vertex_t, result_t>{},
                property_initial_value,
                cugraph::reduce_op::elementwise_maximum<result_t>{},
                cugraph::get_dataframe_buffer_begin(global_out_result));
              break;
            default: FAIL() << "should not be reached.";
          }

          ASSERT_TRUE(compare(mg_aggregate_in_results, global_in_result));
          ASSERT_TRUE(compare(mg_aggregate_out_results, global_out_result));
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGPerVTransformReduceIncomingOutgoingE<input_usecase_t>::handle_ = nullptr;

using Tests_MGPerVTransformReduceIncomingOutgoingE_File =
  Tests_MGPerVTransformReduceIncomingOutgoingE<cugraph::test::File_Usecase>;
using Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat =
  Tests_MGPerVTransformReduceIncomingOutgoingE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File,
       CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
       CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
       CheckInt64Int64FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File,
       CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(std::get<0>(param),
                                                                             std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
       CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
       CheckInt64Int64FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVTransformReduceIncomingOutgoingE_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, false, true},
                                       Prims_Usecase{false, true, true},
                                       Prims_Usecase{true, false, true},
                                       Prims_Usecase{true, true, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGPerVTransformReduceIncomingOutgoingE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, true},
                      Prims_Usecase{false, true, true},
                      Prims_Usecase{true, false, true},
                      Prims_Usecase{true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
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
  Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, false},
                      Prims_Usecase{false, true, false},
                      Prims_Usecase{true, false, false},
                      Prims_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
