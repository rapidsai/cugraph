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

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/partition_manager.hpp>

#include <cuco/detail/hash_functions.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/extract_if_e.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/equal.h>
#include <thrust/reduce.h>

#include <gtest/gtest.h>

#include <random>

template <typename TupleType, typename T, std::size_t... Is>
__device__ auto make_type_casted_tuple_from_scalar(T val, std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    static_cast<typename thrust::tuple_element<Is, TupleType>::type>(val)...);
}

template <typename property_t, typename T>
__device__ __host__ auto make_property_value(T val)
{
  property_t ret{};
  if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
    ret = make_type_casted_tuple_from_scalar<property_t>(
      val, std::make_index_sequence<thrust::tuple_size<property_t>::value>{});
  } else {
    ret = static_cast<property_t>(val);
  }
  return ret;
}

template <typename vertex_t, typename property_t>
struct property_transform_t {
  int mod{};

  constexpr __device__ property_t operator()(vertex_t const v) const
  {
    static_assert(cugraph::is_thrust_tuple_of_arithmetic<property_t>::value ||
                  std::is_arithmetic_v<property_t>);
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    return make_property_value<property_t>(hash_func(v) % mod);
  }
};

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
class Tests_MG_ExtractIfE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_ExtractIfE() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of extract_if_e primitive and thrust reduce on a single GPU
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename property_t,
            bool store_transposed>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) {
      --row_comm_size;
    }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_graph, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        handle, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. run MG extract_if_e

    constexpr int hash_bin_count = 5;

    auto mg_property_buffer = cugraph::allocate_dataframe_buffer<property_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::transform(handle.get_thrust_policy(),
                      (*mg_renumber_map_labels).begin(),
                      (*mg_renumber_map_labels).end(),
                      cugraph::get_dataframe_buffer_begin(mg_property_buffer),
                      property_transform_t<vertex_t, property_t>{hash_bin_count});

    cugraph::edge_partition_src_property_t<decltype(mg_graph_view), property_t> mg_src_properties(
      handle, mg_graph_view);
    cugraph::edge_partition_dst_property_t<decltype(mg_graph_view), property_t> mg_dst_properties(
      handle, mg_graph_view);

    update_edge_partition_src_property(handle,
                                       mg_graph_view,
                                       cugraph::get_dataframe_buffer_cbegin(mg_property_buffer),
                                       mg_src_properties);
    update_edge_partition_dst_property(handle,
                                       mg_graph_view,
                                       cugraph::get_dataframe_buffer_cbegin(mg_property_buffer),
                                       mg_dst_properties);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_edgelist_srcs, mg_edgelist_dsts, mg_edgelist_weights] =
      extract_if_e(handle,
                   mg_graph_view,
                   mg_src_properties.device_view(),
                   mg_dst_properties.device_view(),
                   [] __device__(vertex_t src, vertex_t dst, auto src_val, auto dst_val) {
                     return src_val < dst_val;
                   });

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG extract_if_e took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 4. compare SG & MG results

    if (prims_usecase.check_correctness) {
      // 4-1. aggregate MG results

      auto mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        handle, (*mg_renumber_map_labels).data(), (*mg_renumber_map_labels).size());
      auto mg_aggregate_edgelist_srcs =
        cugraph::test::device_gatherv(handle, mg_edgelist_srcs.data(), mg_edgelist_srcs.size());
      auto mg_aggregate_edgelist_dsts =
        cugraph::test::device_gatherv(handle, mg_edgelist_dsts.data(), mg_edgelist_dsts.size());
      std::optional<rmm::device_uvector<weight_t>> mg_aggregate_edgelist_weights{std::nullopt};
      if (mg_edgelist_weights) {
        mg_aggregate_edgelist_weights = cugraph::test::device_gatherv(
          handle, (*mg_edgelist_weights).data(), (*mg_edgelist_weights).size());
      }

      if (handle.get_comms().get_rank() == int{0}) {
        // 4-2. unrenumber MG results

        cugraph::unrenumber_int_vertices<vertex_t, false>(
          handle,
          mg_aggregate_edgelist_srcs.data(),
          mg_aggregate_edgelist_srcs.size(),
          mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
        cugraph::unrenumber_int_vertices<vertex_t, false>(
          handle,
          mg_aggregate_edgelist_dsts.data(),
          mg_aggregate_edgelist_dsts.size(),
          mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});

        // 4-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> sg_graph(handle);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle, input_usecase, true, false);
        auto sg_graph_view = sg_graph.view();

        // 4-4. run SG extract_if_e

        auto sg_property_buffer = cugraph::allocate_dataframe_buffer<property_t>(
          sg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

        thrust::transform(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
          thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
          cugraph::get_dataframe_buffer_begin(sg_property_buffer),
          property_transform_t<vertex_t, property_t>{hash_bin_count});

        cugraph::edge_partition_src_property_t<decltype(sg_graph_view), property_t>
          sg_src_properties(handle, sg_graph_view);
        cugraph::edge_partition_dst_property_t<decltype(sg_graph_view), property_t>
          sg_dst_properties(handle, sg_graph_view);

        update_edge_partition_src_property(handle,
                                           sg_graph_view,
                                           cugraph::get_dataframe_buffer_cbegin(sg_property_buffer),
                                           sg_src_properties);
        update_edge_partition_dst_property(handle,
                                           sg_graph_view,
                                           cugraph::get_dataframe_buffer_cbegin(sg_property_buffer),
                                           sg_dst_properties);

        auto [sg_edgelist_srcs, sg_edgelist_dsts, sg_edgelist_weights] =
          extract_if_e(handle,
                       sg_graph_view,
                       sg_src_properties.device_view(),
                       sg_dst_properties.device_view(),
                       [] __device__(vertex_t src, vertex_t dst, auto src_val, auto dst_val) {
                         return src_val < dst_val;
                       });

        // 4-5. compare

        if (mg_graph_view.is_weighted()) {
          auto mg_edge_first =
            thrust::make_zip_iterator(thrust::make_tuple(mg_aggregate_edgelist_srcs.begin(),
                                                         mg_aggregate_edgelist_dsts.begin(),
                                                         (*mg_aggregate_edgelist_weights).begin()));
          auto sg_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
            sg_edgelist_srcs.begin(), sg_edgelist_dsts.begin(), (*sg_edgelist_weights).begin()));
          thrust::sort(handle.get_thrust_policy(),
                       mg_edge_first,
                       mg_edge_first + mg_aggregate_edgelist_srcs.size());
          thrust::sort(
            handle.get_thrust_policy(), sg_edge_first, sg_edge_first + sg_edgelist_srcs.size());
          ASSERT_TRUE(thrust::equal(
            handle.get_thrust_policy(),
            mg_edge_first,
            mg_edge_first + mg_aggregate_edgelist_srcs.size(),
            sg_edge_first,
            compare_equal_t<
              typename thrust::iterator_traits<decltype(mg_edge_first)>::value_type>{}));
        } else {
          auto mg_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
            mg_aggregate_edgelist_srcs.begin(), mg_aggregate_edgelist_dsts.begin()));
          auto sg_edge_first = thrust::make_zip_iterator(
            thrust::make_tuple(sg_edgelist_srcs.begin(), sg_edgelist_dsts.begin()));
          thrust::sort(handle.get_thrust_policy(),
                       mg_edge_first,
                       mg_edge_first + mg_aggregate_edgelist_srcs.size());
          thrust::sort(
            handle.get_thrust_policy(), sg_edge_first, sg_edge_first + sg_edgelist_srcs.size());
          ASSERT_TRUE(thrust::equal(
            handle.get_thrust_policy(),
            mg_edge_first,
            mg_edge_first + mg_aggregate_edgelist_srcs.size(),
            sg_edge_first,
            compare_equal_t<
              typename thrust::iterator_traits<decltype(mg_edge_first)>::value_type>{}));
        }
      }
    }
  }
};

using Tests_MG_ExtractIfE_File = Tests_MG_ExtractIfE<cugraph::test::File_Usecase>;
using Tests_MG_ExtractIfE_Rmat = Tests_MG_ExtractIfE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_ExtractIfE_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_ExtractIfE_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(std::get<0>(param),
                                                                             std::get<1>(param));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_ExtractIfE_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_ExtractIfE_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_ExtractIfE_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_ExtractIfE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_ExtractIfE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MG_ExtractIfE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
