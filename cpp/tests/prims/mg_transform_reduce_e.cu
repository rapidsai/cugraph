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
#include <cugraph/prims/edge_partition_src_dst_property.cuh>
#include <cugraph/prims/transform_reduce_e.cuh>
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/count.h>

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
  using type                 = typename property_type<Args...>::type;
  using property_buffer_type = std::conditional_t<
    (sizeof...(Args) > 1),
    std::tuple<rmm::device_uvector<Args>...>,
    rmm::device_uvector<typename thrust::tuple_element<0, thrust::tuple<Args...>>::type>>;

 public:
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
struct result_compare {
  static constexpr double threshold_ratio{1e-3};
  constexpr auto operator()(const T& t1, const T& t2)
  {
    if constexpr (std::is_floating_point_v<T>) {
      return std::abs(t1 - t2) < (std::max(t1, t2) * threshold_ratio);
    }
    return t1 == t2;
  }
};

template <typename... Args>
struct result_compare<thrust::tuple<Args...>> {
  static constexpr double threshold_ratio{1e-3};

  using type = thrust::tuple<Args...>;
  constexpr auto operator()(const type& t1, const type& t2)
  {
    return equality_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<type>::value>());
  }

 private:
  template <typename T>
  constexpr bool equal(T t1, T t2)
  {
    if constexpr (std::is_floating_point_v<T>) {
      return std::abs(t1 - t2) < (std::max(t1, t2) * threshold_ratio);
    }
    return t1 == t2;
  }
  template <typename T, std::size_t... I>
  constexpr auto equality_impl(T& t1, T& t2, std::index_sequence<I...>)
  {
    return (... && (equal(thrust::get<I>(t1), thrust::get<I>(t2))));
  }
};

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
class Tests_MG_TransformReduceE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_TransformReduceE() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of reduce_if_v primitive and thrust reduce on a single GPU
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
    auto row_prop = generate<result_t>::row_property(handle, mg_graph_view, vertex_property_data);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto result = transform_reduce_e(
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
      property_initial_value);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG transform reduce took " << elapsed_time * 1e-6 << " s.\n";
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

      auto expected_result = transform_reduce_e(
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
        property_initial_value);
      result_compare<property_t> compare{};
      ASSERT_TRUE(compare(expected_result, result));
    }
  }
};

using Tests_MG_TransformReduceE_File = Tests_MG_TransformReduceE<cugraph::test::File_Usecase>;
using Tests_MG_TransformReduceE_Rmat = Tests_MG_TransformReduceE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_TransformReduceE_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(std::get<0>(param),
                                                                           std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_TransformReduceE_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(std::get<0>(param),
                                                                          std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_TransformReduceE_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceE_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_TransformReduceE_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceE_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_TransformReduceE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_TransformReduceE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_test,
  Tests_MG_TransformReduceE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
