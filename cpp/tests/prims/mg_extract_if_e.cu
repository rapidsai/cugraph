/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <prims/extract_if_e.cuh>
#include <prims/property_op_utils.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <cuco/detail/hash_functions.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename T, std::size_t... Is>
__device__ bool compare_equal_scalar(T const& lhs, T const& rhs)
{
  static_assert(std::is_arithmetic_v<T>);
  bool ret{false};
  if constexpr (std::is_floating_point_v<T>) {
    static constexpr double threshold_ratio{1e-3};
    ret = (std::abs(lhs - rhs) < (std::max(std::abs(lhs), std::abs(rhs)) * threshold_ratio));
  } else {
    ret = (lhs == rhs);
  }
  return ret;
}

template <typename TupleType, std::size_t... Is>
__device__ bool compare_equal_tuple(TupleType const& lhs,
                                    TupleType const& rhs,
                                    std::index_sequence<Is...>)
{
  return (... && compare_equal_scalar(thrust::get<Is>(lhs), thrust::get<Is>(rhs)));
}

template <typename property_t>
struct compare_equal_t {
  static constexpr double threshold_ratio{1e-3};

  __device__ bool operator()(property_t const& lhs, property_t const& rhs) const
  {
    static_assert(cugraph::is_thrust_tuple_of_arithmetic<property_t>::value ||
                  std::is_arithmetic_v<property_t>);
    bool ret{false};
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
      ret = compare_equal_tuple(
        lhs, rhs, std::make_index_sequence<thrust::tuple_size<property_t>::value>{});
    } else {
      ret = compare_equal_scalar(lhs, rhs);
    }
    return ret;
  }
};

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGExtractIfE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGExtractIfE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of extract_if_e primitive and thrust reduce on a single GPU
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

    // 2. run MG extract_if_e

    constexpr int hash_bin_count = 5;

    auto mg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
      *handle_, *d_mg_renumber_map_labels, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG extract_if_e");
    }

    auto [mg_edgelist_srcs, mg_edgelist_dsts] = extract_if_e(
      *handle_,
      mg_graph_view,
      mg_src_prop.view(),
      mg_dst_prop.view(),
      [] __device__(vertex_t src, vertex_t dst, auto src_val, auto dst_val, thrust::nullopt_t) {
        return src_val < dst_val;
      });

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (prims_usecase.check_correctness) {
      // 3-1. aggregate MG results

      auto mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto mg_aggregate_edgelist_srcs =
        cugraph::test::device_gatherv(*handle_, mg_edgelist_srcs.data(), mg_edgelist_srcs.size());
      auto mg_aggregate_edgelist_dsts =
        cugraph::test::device_gatherv(*handle_, mg_edgelist_dsts.data(), mg_edgelist_dsts.size());

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. unrenumber MG results

        cugraph::unrenumber_int_vertices<vertex_t, false>(
          *handle_,
          mg_aggregate_edgelist_srcs.data(),
          mg_aggregate_edgelist_srcs.size(),
          mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
        cugraph::unrenumber_int_vertices<vertex_t, false>(
          *handle_,
          mg_aggregate_edgelist_dsts.data(),
          mg_aggregate_edgelist_dsts.size(),
          mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});

        // 3-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
        std::tie(sg_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
            *handle_, input_usecase, true, false);

        auto sg_graph_view = sg_graph.view();

        // 3-4. run SG extract_if_e

        auto sg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
          *handle_,
          thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
          thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
          hash_bin_count);
        auto sg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
          *handle_, sg_graph_view, sg_vertex_prop);
        auto sg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
          *handle_, sg_graph_view, sg_vertex_prop);

        auto [sg_edgelist_srcs, sg_edgelist_dsts] = extract_if_e(
          *handle_,
          sg_graph_view,
          sg_src_prop.view(),
          sg_dst_prop.view(),
          [] __device__(vertex_t src, vertex_t dst, auto src_val, auto dst_val, thrust::nullopt_t) {
            return src_val < dst_val;
          });

        // 3-5. compare

        auto mg_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
          mg_aggregate_edgelist_srcs.begin(), mg_aggregate_edgelist_dsts.begin()));
        auto sg_edge_first = thrust::make_zip_iterator(
          thrust::make_tuple(sg_edgelist_srcs.begin(), sg_edgelist_dsts.begin()));
        thrust::sort(handle_->get_thrust_policy(),
                     mg_edge_first,
                     mg_edge_first + mg_aggregate_edgelist_srcs.size());
        thrust::sort(
          handle_->get_thrust_policy(), sg_edge_first, sg_edge_first + sg_edgelist_srcs.size());
        ASSERT_TRUE(thrust::equal(
          handle_->get_thrust_policy(),
          mg_edge_first,
          mg_edge_first + mg_aggregate_edgelist_srcs.size(),
          sg_edge_first,
          compare_equal_t<
            typename thrust::iterator_traits<decltype(mg_edge_first)>::value_type>{}));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGExtractIfE<input_usecase_t>::handle_ = nullptr;

using Tests_MGExtractIfE_File = Tests_MGExtractIfE<cugraph::test::File_Usecase>;
using Tests_MGExtractIfE_Rmat = Tests_MGExtractIfE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGExtractIfE_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractIfE_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(std::get<0>(param),
                                                                             std::get<1>(param));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractIfE_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractIfE_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractIfE_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGExtractIfE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGExtractIfE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGExtractIfE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
