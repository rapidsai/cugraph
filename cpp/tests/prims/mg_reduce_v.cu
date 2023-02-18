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

#include <prims/property_op_utils.cuh>
#include <prims/reduce_v.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <cuco/detail/hash_functions.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename T>
struct result_compare {
  static constexpr double threshold_ratio{1e-2};
  constexpr auto operator()(const T& t1, const T& t2)
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

template <typename... Args>
struct result_compare<thrust::tuple<Args...>> {
  static constexpr double threshold_ratio{1e-3};

  using Type = thrust::tuple<Args...>;
  constexpr auto operator()(const Type& t1, const Type& t2)
  {
    return equality_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<Type>::value>());
  }

 private:
  template <typename T>
  constexpr bool equal(T t1, T t2)
  {
    if constexpr (std::is_floating_point_v<T>) {
      bool passed = (t1 == t2)  // when t1 == t2 == 0
                    ||
                    (std::abs(t1 - t2) < (std::max(std::abs(t1), std::abs(t2)) * threshold_ratio));
      return passed;
    }
    return t1 == t2;
  }
  template <typename T, std::size_t... I>
  constexpr auto equality_impl(T& t1, T& t2, std::index_sequence<I...>)
  {
    return (... && (equal(thrust::get<I>(t1), thrust::get<I>(t2))));
  }
};

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGReduceV
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGReduceV() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of reduce_v primitive and thrust reduce on a single GPU
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
        *handle_, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG reduce_v

    const int hash_bin_count = 5;
    const int initial_value  = 10;

    auto property_initial_value =
      cugraph::test::generate<vertex_t, result_t>::initial_value(initial_value);

    auto mg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
      *handle_, (*d_mg_renumber_map_labels), hash_bin_count);
    auto property_iter = cugraph::get_dataframe_buffer_begin(mg_vertex_prop);

    enum class reduction_type_t { PLUS, MINIMUM, MAXIMUM };
    std::array<reduction_type_t, 3> reduction_types = {
      reduction_type_t::PLUS, reduction_type_t::MINIMUM, reduction_type_t::MAXIMUM};

    std::unordered_map<reduction_type_t, result_t> results;

    for (auto reduction_type : reduction_types) {
      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle_->get_comms().barrier();
        hr_timer.start("MG reduce_v");
      }

      switch (reduction_type) {
        case reduction_type_t::PLUS:
          results[reduction_type] = reduce_v(*handle_,
                                             mg_graph_view,
                                             property_iter,
                                             property_initial_value,
                                             cugraph::reduce_op::plus<result_t>{});
          break;
        case reduction_type_t::MINIMUM:
          results[reduction_type] = reduce_v(*handle_,
                                             mg_graph_view,
                                             property_iter,
                                             property_initial_value,
                                             cugraph::reduce_op::minimum<result_t>{});
          break;
        case reduction_type_t::MAXIMUM:
          results[reduction_type] = reduce_v(*handle_,
                                             mg_graph_view,
                                             property_iter,
                                             property_initial_value,
                                             cugraph::reduce_op::maximum<result_t>{});
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
      auto sg_property_iter = cugraph::get_dataframe_buffer_begin(sg_vertex_prop);

      for (auto reduction_type : reduction_types) {
        result_t expected_result{};
        switch (reduction_type) {
          case reduction_type_t::PLUS:
            expected_result = reduce_v(*handle_,
                                       sg_graph_view,
                                       sg_property_iter,
                                       property_initial_value,
                                       cugraph::reduce_op::plus<result_t>{});
            break;
          case reduction_type_t::MINIMUM:
            expected_result = reduce_v(*handle_,
                                       sg_graph_view,
                                       sg_property_iter,
                                       property_initial_value,
                                       cugraph::reduce_op::minimum<result_t>{});
            break;
          case reduction_type_t::MAXIMUM:
            expected_result = reduce_v(*handle_,
                                       sg_graph_view,
                                       sg_property_iter,
                                       property_initial_value,
                                       cugraph::reduce_op::maximum<result_t>{});
            break;
          default: FAIL() << "should not be reached.";
        }
        result_compare<result_t> compare{};
        ASSERT_TRUE(compare(expected_result, results[reduction_type]));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGReduceV<input_usecase_t>::handle_ = nullptr;

using Tests_MGReduceV_File = Tests_MGReduceV<cugraph::test::File_Usecase>;
using Tests_MGReduceV_Rmat = Tests_MGReduceV<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGReduceV_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int32_t, float>, false>(
    std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int32_t, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGReduceV_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int32_t, float>, true>(
    std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int32_t, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGReduceV_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int32_t, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGReduceV_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int32_t, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGReduceV_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGReduceV_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));
INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGReduceV_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGReduceV_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
