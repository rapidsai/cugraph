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
#include <cugraph/prims/transform_reduce_v.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/count.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename T>
struct property_transform : public thrust::unary_function<vertex_t, T> {
  int mod{};
  property_transform(int mod_count) : mod(mod_count) {}
  constexpr __device__ auto operator()(const vertex_t& val)
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto value = hash_func(val) % mod;
    return static_cast<T>(value);
  }
};

template <typename vertex_t, typename... Args>
struct property_transform<vertex_t, std::tuple<Args...>>
  : public thrust::unary_function<vertex_t, thrust::tuple<Args...>> {
  int mod{};
  property_transform(int mod_count) : mod(mod_count) {}
  constexpr __device__ auto operator()(const vertex_t& val)
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto value = hash_func(val) % mod;
    return thrust::make_tuple(static_cast<Args>(value)...);
  }
};

template <typename T>
struct result_compare {
  static constexpr double threshold_ratio{1e-3};
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

template <typename T>
struct generate {
  static T initial_value(int init) { return static_cast<T>(init); }
};
template <typename... Args>
struct generate<std::tuple<Args...>> {
  static thrust::tuple<Args...> initial_value(int init)
  {
    return thrust::make_tuple(static_cast<Args>(init)...);
  }
};

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MG_TransformReduceV
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_TransformReduceV() {}
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
        handle, input_usecase, true, true);

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
    const int initial_value  = 10;

    property_transform<vertex_t, result_t> prop(hash_bin_count);
    auto property_initial_value = generate<result_t>::initial_value(initial_value);
    using property_t            = decltype(property_initial_value);
    raft::comms::op_t ops[]     = {
      raft::comms::op_t::SUM, raft::comms::op_t::MIN, raft::comms::op_t::MAX};

    std::unordered_map<raft::comms::op_t, property_t> results;

    for (auto op : ops) {
      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle.get_comms().barrier();
        hr_clock.start();
      }

      results[op] = transform_reduce_v(
        handle, mg_graph_view, d_mg_renumber_map_labels->begin(), prop, property_initial_value, op);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        handle.get_comms().barrier();
        double elapsed_time{0.0};
        hr_clock.stop(&elapsed_time);
        std::cout << "MG transform reduce took " << elapsed_time * 1e-6 << " s.\n";
      }
    }

    //// 4. compare SG & MG results

    if (prims_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> sg_graph(handle);
      std::tie(sg_graph, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
          handle, input_usecase, true, false);
      auto sg_graph_view = sg_graph.view();

      for (auto op : ops) {
        auto expected_result = cugraph::op_dispatch<property_t>(
          op, [&handle, &sg_graph_view, prop, property_initial_value](auto op) {
            return thrust::transform_reduce(
              handle.get_thrust_policy(),
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
              thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
              prop,
              property_initial_value,
              op);
          });
        result_compare<property_t> compare{};
        ASSERT_TRUE(compare(expected_result, results[op]));
      }
    }
  }
};

using Tests_MG_TransformReduceV_File = Tests_MG_TransformReduceV<cugraph::test::File_Usecase>;
using Tests_MG_TransformReduceV_Rmat = Tests_MG_TransformReduceV<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_TransformReduceV_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(std::get<0>(param),
                                                                           std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceV_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_TransformReduceV_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(std::get<0>(param),
                                                                          std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceV_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, std::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_TransformReduceV_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceV_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_TransformReduceV_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_TransformReduceV_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_TransformReduceV_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_TransformReduceV_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_test,
  Tests_MG_TransformReduceV_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
