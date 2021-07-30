/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/partition_manager.hpp>

#include <cuco/detail/hash_functions.cuh>
#include <cugraph/experimental/graph_view.hpp>
#include <cugraph/prims/reduce_v.cuh>

#include <thrust/count.h>
#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 0;

template <typename vertex_t, typename... T>
struct property_transform : public thrust::unary_function<vertex_t, thrust::tuple<T...>> {
  int mod{};
  property_transform(int mod_count) : mod(mod_count) {}
  __device__ auto operator()(const vertex_t& val)
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto value = hash_func(val) % mod;
    return thrust::make_tuple(static_cast<T>(value)...);
  }
};
template <typename vertex_t, template <typename...> typename Tuple, typename... T>
struct property_transform<vertex_t, Tuple<T...>> : public property_transform<vertex_t, T...> {
};

template <typename Tuple, std::size_t... I>
auto make_iterator_tuple(Tuple& data, std::index_sequence<I...>)
{
  return thrust::make_tuple((std::get<I>(data).begin())...);
}

template <typename... T>
auto get_zip_iterator(std::tuple<T...>& data)
{
  return thrust::make_zip_iterator(make_iterator_tuple(
    data, std::make_index_sequence<std::tuple_size<std::tuple<T...>>::value>()));
}

template <typename T>
auto get_property_iterator(std::tuple<T>& data)
{
  return (std::get<0>(data)).begin();
}

template <typename T0, typename... T>
auto get_property_iterator(std::tuple<T0, T...>& data)
{
  return get_zip_iterator(data);
}

template <typename... T>
struct generate_impl {
  static thrust::tuple<T...> initial_value(int init)
  {
    return thrust::make_tuple(static_cast<T>(init)...);
  }
  template <typename label_t>
  static std::tuple<rmm::device_uvector<T>...> property(rmm::device_uvector<label_t>& labels,
                                                        int hash_bin_count,
                                                        raft::handle_t const& handle)
  {
    auto data = std::make_tuple(rmm::device_uvector<T>(labels.size(), handle.get_stream())...);
    auto zip  = get_zip_iterator(data);
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      labels.begin(),
                      labels.end(),
                      zip,
                      property_transform<label_t, T...>(hash_bin_count));
    return data;
  }
  template <typename label_t>
  static std::tuple<rmm::device_uvector<T>...> property(thrust::counting_iterator<label_t> begin,
                                                        thrust::counting_iterator<label_t> end,
                                                        int hash_bin_count,
                                                        raft::handle_t const& handle)
  {
    auto length = thrust::distance(begin, end);
    auto data   = std::make_tuple(rmm::device_uvector<T>(length, handle.get_stream())...);
    auto zip    = get_zip_iterator(data);
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      begin,
                      end,
                      zip,
                      property_transform<label_t, T...>(hash_bin_count));
    return data;
  }
};

template <typename T>
struct result_compare {
  constexpr auto operator()(const T& t1, const T& t2) { return (t1 == t2); }
};

template <typename... Args>
struct result_compare<thrust::tuple<Args...>> {
  static constexpr double threshold_ratio{1e-3};

 private:
  template <typename T>
  bool equal(T t1, T t2)
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

 public:
  using Type = thrust::tuple<Args...>;
  constexpr auto operator()(const Type& t1, const Type& t2)
  {
    return equality_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<Type>::value>());
  }
};

template <typename T>
struct generate : public generate_impl<T> {
  static T initial_value(int init) { return static_cast<T>(init); }
};
template <typename... T>
struct generate<std::tuple<T...>> : public generate_impl<T...> {
};

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MG_ReduceIfV
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_ReduceIfV() {}
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

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }
    auto [mg_graph, d_mg_renumber_map_labels] =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        handle, true, true);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. run MG count if

    const int hash_bin_count = 5;
    const int initial_value  = 10;

    auto property_initial_value = generate<result_t>::initial_value(initial_value);
    auto property_data =
      generate<result_t>::property((*d_mg_renumber_map_labels), hash_bin_count, handle);
    auto property_iter = get_property_iterator(property_data);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto result = reduce_v(handle,
                           mg_graph_view,
                           property_iter,
                           property_iter + (*d_mg_renumber_map_labels).size(),
                           property_initial_value);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG count if took " << elapsed_time * 1e-6 << " s.\n";
    }

    //// 4. compare SG & MG results

    if (prims_usecase.check_correctness) {
      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> sg_graph(
        handle);
      std::tie(sg_graph, std::ignore) =
        input_usecase.template construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
          handle, true, false);
      auto sg_graph_view = sg_graph.view();

      auto sg_property_data = generate<result_t>::property(
        thrust::make_counting_iterator(sg_graph_view.get_local_vertex_first()),
        thrust::make_counting_iterator(sg_graph_view.get_local_vertex_last()),
        hash_bin_count,
        handle);
      auto sg_property_iter = get_property_iterator(sg_property_data);
      using property_t      = decltype(property_initial_value);

      auto expected_result =
        thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       sg_property_iter,
                       sg_property_iter + sg_graph_view.get_number_of_local_vertices(),
                       property_initial_value,
                       cugraph::experimental::property_add<property_t>());
      result_compare<property_t> compare;
      ASSERT_TRUE(compare(expected_result, result));
    }
  }
};

using Tests_MG_ReduceIfV_File = Tests_MG_ReduceIfV<cugraph::test::File_Usecase>;
using Tests_MG_ReduceIfV_Rmat = Tests_MG_ReduceIfV<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_ReduceIfV_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(std::get<0>(param),
                                                                           std::get<1>(param));
}

TEST_P(Tests_MG_ReduceIfV_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(std::get<0>(param),
                                                                           std::get<1>(param));
}

TEST_P(Tests_MG_ReduceIfV_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(std::get<0>(param),
                                                                          std::get<1>(param));
}

TEST_P(Tests_MG_ReduceIfV_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(std::get<0>(param),
                                                                          std::get<1>(param));
}

TEST_P(Tests_MG_ReduceIfV_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_ReduceIfV_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_ReduceIfV_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_ReduceIfV_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_ReduceIfV_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_ReduceIfV_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_test,
  Tests_MG_ReduceIfV_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
