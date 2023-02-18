/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "property_generator.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <cuco/detail/hash_functions.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <sstream>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename result_t>
struct e_op_t {
  __device__ result_t operator()(vertex_t src,
                                 vertex_t dst,
                                 result_t src_property,
                                 result_t dst_property,
                                 thrust::nullopt_t) const
  {
    if (src_property < dst_property) {
      return src_property;
    } else {
      return dst_property;
    }
  }
};

template <typename T>
struct comparator {
  static constexpr double threshold_ratio{1e-2};
  __host__ __device__ bool operator()(T t1, T t2) const
  {
    if constexpr (std::is_floating_point_v<T>) {
      bool passed = (t1 == t2)  // when t1 == t2 == 0
                    ||
                    (std::abs(t1 - t2) < (std::max(std::abs(t1), std::abs(t2)) * threshold_ratio));
      return passed;
    }
    return t1 == t2;
  }
};

struct result_compare {
  const raft::handle_t& handle_;
  result_compare(raft::handle_t const& handle) : handle_(handle) {}

  template <typename... Args>
  auto operator()(const std::tuple<rmm::device_uvector<Args>...>& t1,
                  const std::tuple<rmm::device_uvector<Args>...>& t2)
  {
    using type = thrust::tuple<Args...>;
    return equality_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<type>::value>());
  }

  template <typename T>
  auto operator()(const rmm::device_uvector<T>& t1, const rmm::device_uvector<T>& t2)
  {
    return thrust::equal(
      handle_.get_thrust_policy(), t1.begin(), t1.end(), t2.begin(), comparator<T>());
  }

 private:
  template <typename T, std::size_t... I>
  auto equality_impl(T& t1, T& t2, std::index_sequence<I...>)
  {
    return (... && (result_compare::operator()(std::get<I>(t1), std::get<I>(t2))));
  }
};

template <typename buffer_type>
buffer_type aggregate(const raft::handle_t& handle, const buffer_type& result)
{
  auto aggregated_result =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<buffer_type>>(
      0, handle.get_stream());
  cugraph::transform(result, aggregated_result, [&handle](auto& input, auto& output) {
    output = cugraph::test::device_gatherv(handle, input.data(), input.size());
  });
  return aggregated_result;
}

struct Prims_Usecase {
  bool check_correctness{true};
  bool test_weighted{false};
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
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> d_mg_renumber_map_labels{std::nullopt};
    std::tie(mg_graph, std::ignore, d_mg_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, prims_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG transform reduce

    const int hash_bin_count = 5;
    const int initial_value  = 4;

    auto property_initial_value =
      cugraph::test::generate<vertex_t, result_t>::initial_value(initial_value);

    auto mg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
      *handle_, *d_mg_renumber_map_labels, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    enum class reduction_type_t { PLUS, MINIMUM, MAXIMUM };
    std::array<reduction_type_t, 3> reduction_types = {
      reduction_type_t::PLUS, reduction_type_t::MINIMUM, reduction_type_t::MAXIMUM};

    std::vector<decltype(cugraph::allocate_dataframe_buffer<result_t>(0, rmm::cuda_stream_view{}))>
      out_results{};
    std::vector<decltype(cugraph::allocate_dataframe_buffer<result_t>(0, rmm::cuda_stream_view{}))>
      in_results{};
    out_results.reserve(reduction_types.size());
    in_results.reserve(reduction_types.size());

    for (size_t i = 0; i < reduction_types.size(); ++i) {
      in_results.push_back(cugraph::allocate_dataframe_buffer<result_t>(
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
                                            cugraph::get_dataframe_buffer_begin(in_results[i]));
          break;
        case reduction_type_t::MINIMUM:
          per_v_transform_reduce_incoming_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::minimum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(in_results[i]));
          break;
        case reduction_type_t::MAXIMUM:
          per_v_transform_reduce_incoming_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::maximum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(in_results[i]));
          break;
        default: FAIL() << "should not be reached.";
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle_->get_comms().barrier();
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      out_results.push_back(cugraph::allocate_dataframe_buffer<result_t>(
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
                                            cugraph::get_dataframe_buffer_begin(out_results[i]));
          break;
        case reduction_type_t::MINIMUM:
          per_v_transform_reduce_outgoing_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::minimum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(out_results[i]));
          break;
        case reduction_type_t::MAXIMUM:
          per_v_transform_reduce_outgoing_e(*handle_,
                                            mg_graph_view,
                                            mg_src_prop.view(),
                                            mg_dst_prop.view(),
                                            cugraph::edge_dummy_property_t{}.view(),
                                            e_op_t<vertex_t, result_t>{},
                                            property_initial_value,
                                            cugraph::reduce_op::maximum<result_t>{},
                                            cugraph::get_dataframe_buffer_begin(out_results[i]));
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
      std::tie(sg_graph, std::ignore, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
          *handle_, input_usecase, true, false);

      auto sg_graph_view = sg_graph.view();

      auto sg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
        *handle_,
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
        hash_bin_count);
      auto sg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
        *handle_, sg_graph_view, sg_vertex_prop);
      auto sg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
        *handle_, sg_graph_view, sg_vertex_prop);
      result_compare comp{*handle_};

      for (size_t i = 0; i < reduction_types.size(); ++i) {
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
          case reduction_type_t::MINIMUM:
            per_v_transform_reduce_outgoing_e(
              *handle_,
              sg_graph_view,
              sg_src_prop.view(),
              sg_dst_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              e_op_t<vertex_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::minimum<result_t>{},
              cugraph::get_dataframe_buffer_begin(global_out_result));
            break;
          case reduction_type_t::MAXIMUM:
            per_v_transform_reduce_outgoing_e(
              *handle_,
              sg_graph_view,
              sg_src_prop.view(),
              sg_dst_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              e_op_t<vertex_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::maximum<result_t>{},
              cugraph::get_dataframe_buffer_begin(global_out_result));
            break;
          default: FAIL() << "should not be reached.";
        }

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
          case reduction_type_t::MINIMUM:
            per_v_transform_reduce_incoming_e(
              *handle_,
              sg_graph_view,
              sg_src_prop.view(),
              sg_dst_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              e_op_t<vertex_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::minimum<result_t>{},
              cugraph::get_dataframe_buffer_begin(global_in_result));
            break;
          case reduction_type_t::MAXIMUM:
            per_v_transform_reduce_incoming_e(
              *handle_,
              sg_graph_view,
              sg_src_prop.view(),
              sg_dst_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              e_op_t<vertex_t, result_t>{},
              property_initial_value,
              cugraph::reduce_op::maximum<result_t>{},
              cugraph::get_dataframe_buffer_begin(global_in_result));
            break;
          default: FAIL() << "should not be reached.";
        }

        auto aggregate_labels      = aggregate(*handle_, *d_mg_renumber_map_labels);
        auto aggregate_out_results = aggregate(*handle_, out_results[i]);
        auto aggregate_in_results  = aggregate(*handle_, in_results[i]);
        if (handle_->get_comms().get_rank() == int{0}) {
          std::tie(std::ignore, aggregate_out_results) =
            cugraph::test::sort_by_key(*handle_, aggregate_labels, aggregate_out_results);
          std::tie(std::ignore, aggregate_in_results) =
            cugraph::test::sort_by_key(*handle_, aggregate_labels, aggregate_in_results);
          ASSERT_TRUE(comp(aggregate_out_results, global_out_result));
          ASSERT_TRUE(comp(aggregate_in_results, global_in_result));
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
       CheckInt32Int64FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, thrust::tuple<int, float>, false>(
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
       CheckInt32Int64FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, thrust::tuple<int, float>, true>(
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

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, false>(
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

TEST_P(Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, true>(
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
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGPerVTransformReduceIncomingOutgoingE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
