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
#include <cugraph/utilities/dataframe_buffer.cuh>

#include <cuco/detail/hash_functions.cuh>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v_transform_reduce_in_out_nbr.cuh>
#include <cugraph/prims/edge_partition_src_dst_property.cuh>
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <sstream>
#include <thrust/count.h>
#include <thrust/equal.h>

#include <gtest/gtest.h>

#include <random>

template <typename... Args>
struct property_type {
  using type = std::conditional_t<(sizeof...(Args) > 1),
                                  thrust::tuple<Args...>,
                                  typename thrust::tuple_element<0, thrust::tuple<Args...>>::type>;
};

template <typename vertex_t, typename... Args>
struct property_transform
  : public thrust::unary_function<vertex_t, typename property_type<Args...>::type> {
  int mod{};
  property_transform(int mod_count) : mod(mod_count) {}

  template <typename type = typename property_type<Args...>::type>
  constexpr __device__
    typename std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<type>::value, type>
    operator()(const vertex_t& val)
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto value = hash_func(val) % mod;
    return thrust::make_tuple(static_cast<Args>(value)...);
  }

  template <typename type = typename property_type<Args...>::type>
  constexpr __device__ typename std::enable_if_t<std::is_arithmetic<type>::value, type> operator()(
    const vertex_t& val)
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto value = hash_func(val) % mod;
    return static_cast<type>(value);
  }
};

template <typename vertex_t, template <typename...> typename Tuple, typename... Args>
struct property_transform<vertex_t, Tuple<Args...>> : public property_transform<vertex_t, Args...> {
};

template <typename... Args>
struct generate_impl {
 private:
  using property_buffer_type = std::conditional_t<
    (sizeof...(Args) > 1),
    std::tuple<rmm::device_uvector<Args>...>,
    rmm::device_uvector<typename thrust::tuple_element<0, thrust::tuple<Args...>>::type>>;

 public:
  using type = typename property_type<Args...>::type;
  static thrust::tuple<Args...> initial_value(int init)
  {
    return thrust::make_tuple(static_cast<Args>(init)...);
  }
  template <typename label_t>
  static auto vertex_property(rmm::device_uvector<label_t>& labels,
                              int hash_bin_count,
                              raft::handle_t const& handle)
  {
    auto data = cugraph::allocate_dataframe_buffer<type>(labels.size(), handle.get_stream());
    auto zip  = cugraph::get_dataframe_buffer_begin(data);
    thrust::transform(handle.get_thrust_policy(),
                      labels.begin(),
                      labels.end(),
                      zip,
                      property_transform<label_t, Args...>(hash_bin_count));
    return data;
  }
  template <typename label_t>
  static auto vertex_property(thrust::counting_iterator<label_t> begin,
                              thrust::counting_iterator<label_t> end,
                              int hash_bin_count,
                              raft::handle_t const& handle)
  {
    auto length = thrust::distance(begin, end);
    auto data   = cugraph::allocate_dataframe_buffer<type>(length, handle.get_stream());
    auto zip    = cugraph::get_dataframe_buffer_begin(data);
    thrust::transform(handle.get_thrust_policy(),
                      begin,
                      end,
                      zip,
                      property_transform<label_t, Args...>(hash_bin_count));
    return data;
  }

  template <typename graph_view_type>
  static auto column_property(raft::handle_t const& handle,
                              graph_view_type const& graph_view,
                              property_buffer_type& property)
  {
    auto output_property =
      cugraph::edge_partition_dst_property_t<graph_view_type, type>(handle, graph_view);
    update_edge_partition_dst_property(
      handle, graph_view, cugraph::get_dataframe_buffer_begin(property), output_property);
    return output_property;
  }

  template <typename graph_view_type>
  static auto row_property(raft::handle_t const& handle,
                           graph_view_type const& graph_view,
                           property_buffer_type& property)
  {
    auto output_property =
      cugraph::edge_partition_src_property_t<graph_view_type, type>(handle, graph_view);
    update_edge_partition_src_property(
      handle, graph_view, cugraph::get_dataframe_buffer_begin(property), output_property);
    return output_property;
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

template <typename T>
struct generate : public generate_impl<T> {
  static T initial_value(int init) { return static_cast<T>(init); }
};
template <typename... Args>
struct generate<std::tuple<Args...>> : public generate_impl<Args...> {
};

struct Prims_Usecase {
  bool check_correctness{true};
  bool test_weighted{false};
};

template <typename input_usecase_t>
class Tests_MG_CopyVTransformReduceInOutNbr
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_CopyVTransformReduceInOutNbr() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of copy_v_transform_reduce_out_nbr primitive
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename result_t,
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
    auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        handle, input_usecase, prims_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. run MG transform reduce

    const int hash_bin_count = 5;
    const int initial_value  = 4;

    auto property_initial_value = generate<result_t>::initial_value(initial_value);
    using property_t            = decltype(property_initial_value);
    auto vertex_property_data =
      generate<result_t>::vertex_property((*d_mg_renumber_map_labels), hash_bin_count, handle);
    auto col_prop =
      generate<result_t>::column_property(handle, mg_graph_view, vertex_property_data);
    auto row_prop   = generate<result_t>::row_property(handle, mg_graph_view, vertex_property_data);
    auto out_result = cugraph::allocate_dataframe_buffer<property_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle.get_stream());
    auto in_result = cugraph::allocate_dataframe_buffer<property_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    copy_v_transform_reduce_in_nbr(
      handle,
      mg_graph_view,
      row_prop.device_view(),
      col_prop.device_view(),
      [] __device__(auto row, auto col, weight_t wt, auto row_property, auto col_property) {
        if (row_property < col_property) {
          return row_property;
        } else {
          return col_property;
        }
      },
      property_initial_value,
      cugraph::get_dataframe_buffer_begin(in_result));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG copy v transform reduce in took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    copy_v_transform_reduce_out_nbr(
      handle,
      mg_graph_view,
      row_prop.device_view(),
      col_prop.device_view(),
      [] __device__(auto row, auto col, weight_t wt, auto row_property, auto col_property) {
        if (row_property < col_property) {
          return row_property;
        } else {
          return col_property;
        }
      },
      property_initial_value,
      cugraph::get_dataframe_buffer_begin(out_result));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG copy v transform reduce out took " << elapsed_time * 1e-6 << " s.\n";
    }

    //// 4. compare SG & MG results

    if (prims_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> sg_graph(handle);
      std::tie(sg_graph, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
          handle, input_usecase, true, false);
      auto sg_graph_view = sg_graph.view();

      auto sg_vertex_property_data = generate<result_t>::vertex_property(
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
        hash_bin_count,
        handle);
      auto sg_col_prop =
        generate<result_t>::column_property(handle, sg_graph_view, sg_vertex_property_data);
      auto sg_row_prop =
        generate<result_t>::row_property(handle, sg_graph_view, sg_vertex_property_data);
      result_compare comp{handle};

      auto global_out_result = cugraph::allocate_dataframe_buffer<property_t>(
        sg_graph_view.local_vertex_partition_range_size(), handle.get_stream());
      copy_v_transform_reduce_out_nbr(
        handle,
        sg_graph_view,
        sg_row_prop.device_view(),
        sg_col_prop.device_view(),
        [] __device__(auto row, auto col, weight_t wt, auto row_property, auto col_property) {
          if (row_property < col_property) {
            return row_property;
          } else {
            return col_property;
          }
        },
        property_initial_value,
        cugraph::get_dataframe_buffer_begin(global_out_result));

      auto global_in_result = cugraph::allocate_dataframe_buffer<property_t>(
        sg_graph_view.local_vertex_partition_range_size(), handle.get_stream());
      copy_v_transform_reduce_in_nbr(
        handle,
        sg_graph_view,
        sg_row_prop.device_view(),
        sg_col_prop.device_view(),
        [] __device__(auto row, auto col, weight_t wt, auto row_property, auto col_property) {
          if (row_property < col_property) {
            return row_property;
          } else {
            return col_property;
          }
        },
        property_initial_value,
        cugraph::get_dataframe_buffer_begin(global_in_result));
      auto aggregate_labels      = aggregate(handle, *d_mg_renumber_map_labels);
      auto aggregate_out_results = aggregate(handle, out_result);
      auto aggregate_in_results  = aggregate(handle, in_result);
      if (handle.get_comms().get_rank() == int{0}) {
        std::tie(std::ignore, aggregate_out_results) =
          cugraph::test::sort_by_key(handle, aggregate_labels, aggregate_out_results);
        std::tie(std::ignore, aggregate_in_results) =
          cugraph::test::sort_by_key(handle, aggregate_labels, aggregate_in_results);
        ASSERT_TRUE(comp(aggregate_out_results, global_out_result));
        ASSERT_TRUE(comp(aggregate_in_results, global_in_result));
      }
    }
  }
};

using Tests_MG_CopyVTransformReduceInOutNbr_File =
  Tests_MG_CopyVTransformReduceInOutNbr<cugraph::test::File_Usecase>;
using Tests_MG_CopyVTransformReduceInOutNbr_Rmat =
  Tests_MG_CopyVTransformReduceInOutNbr<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(std::get<0>(param),
                                                                           std::get<1>(param));
}

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(std::get<0>(param),
                                                                          std::get<1>(param));
}

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_CopyVTransformReduceInOutNbr_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_CopyVTransformReduceInOutNbr_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_CopyVTransformReduceInOutNbr_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_test,
  Tests_MG_CopyVTransformReduceInOutNbr_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
