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
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>
#include <cugraph/prims/update_frontier_v_push_if_out_nbr.cuh>
#include <cugraph/prims/vertex_frontier.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <sstream>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/iterator/permutation_iterator.h>

#include <gtest/gtest.h>

#include <random>

template <typename TupleType, typename T, std::size_t... Is>
__device__ __host__ auto make_type_casted_tuple_from_scalar(T val, std::index_sequence<Is...>)
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

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MG_UpdateFrontierVPushIfOutNbr
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_UpdateFrontierVPushIfOutNbr() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of update_frontier_v_push_if_out_nbr primitive
  template <typename vertex_t, typename edge_t, typename weight_t, typename property_t>
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

    constexpr bool is_multi_gpu = true;
    constexpr bool renumber     = true;  // needs to be true for multi gpu case
    constexpr bool store_transposed =
      false;  // needs to be false for using update_frontier_v_push_if_out_nbr
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }
    auto [mg_graph, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, is_multi_gpu>(
        handle, input_usecase, false, renumber);

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
    // const int initial_value  = 4;

    auto mg_property_buffer = cugraph::allocate_dataframe_buffer<property_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::transform(handle.get_thrust_policy(),
                      (*mg_renumber_map_labels).begin(),
                      (*mg_renumber_map_labels).end(),
                      cugraph::get_dataframe_buffer_begin(mg_property_buffer),
                      property_transform_t<vertex_t, property_t>{hash_bin_count});
    rmm::device_uvector<vertex_t> sources(mg_graph_view.number_of_vertices(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(),
                     sources.begin(),
                     sources.end(),
                     mg_graph_view.local_vertex_partition_range_first());

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

    enum class Bucket { cur, next, num_buckets };
    cugraph::VertexFrontier<vertex_t, void, is_multi_gpu, static_cast<size_t>(Bucket::num_buckets)>
      mg_vertex_frontier(handle);
    mg_vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
      .insert(sources.begin(), sources.end());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    // prims call
    update_frontier_v_push_if_out_nbr(
      handle,
      mg_graph_view,
      mg_vertex_frontier,
      static_cast<size_t>(Bucket::cur),
      std::vector<size_t>{static_cast<size_t>(Bucket::next)},
      mg_src_properties.device_view(),
      mg_dst_properties.device_view(),
      [] __device__(vertex_t src, vertex_t dst, auto src_val, auto dst_val) {
        thrust::optional<vertex_t> result;
        if (src_val < dst_val) { result.emplace(src); }
        return result;
      },
      cugraph::reduce_op::any<vertex_t>(),
      cugraph::get_dataframe_buffer_cbegin(mg_property_buffer),
      thrust::make_discard_iterator() /* dummy */,
      [] __device__(auto v, auto v_val, auto pushed_val) {
        return thrust::optional<thrust::tuple<size_t, std::byte>>{
          thrust::make_tuple(static_cast<size_t>(Bucket::next), std::byte{0} /* dummy */)};
      });

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG update_frontier_v_push_if_out_nbr took " << elapsed_time * 1e-6 << " s.\n";
    }

    //// 4. compare SG & MG results

    if (prims_usecase.check_correctness) {
      auto mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        handle, (*mg_renumber_map_labels).data(), (*mg_renumber_map_labels).size());

      auto& next_bucket = mg_vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next));
      auto mg_aggregate_frontier_dsts =
        cugraph::test::device_gatherv(handle, next_bucket.begin(), next_bucket.size());

      if (handle.get_comms().get_rank() == int{0}) {
        cugraph::unrenumber_int_vertices<vertex_t, !is_multi_gpu>(
          handle,
          mg_aggregate_frontier_dsts.begin(),
          mg_aggregate_frontier_dsts.size(),
          mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
        thrust::sort(handle.get_thrust_policy(),
                     mg_aggregate_frontier_dsts.begin(),
                     mg_aggregate_frontier_dsts.end());

        cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, !is_multi_gpu> sg_graph(
          handle);
        std::tie(sg_graph, std::ignore) = cugraph::test::
          construct_graph<vertex_t, edge_t, weight_t, store_transposed, !is_multi_gpu>(
            handle, input_usecase, false, false);
        auto sg_graph_view = sg_graph.view();

        auto sg_property_buffer = cugraph::allocate_dataframe_buffer<property_t>(
          sg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

        thrust::transform(handle.get_thrust_policy(),
                          sources.begin(),
                          sources.end(),
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
        cugraph::
          VertexFrontier<vertex_t, void, !is_multi_gpu, static_cast<size_t>(Bucket::num_buckets)>
            sg_vertex_frontier(handle);
        sg_vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
          .insert(sources.begin(), sources.end());

        update_frontier_v_push_if_out_nbr(
          handle,
          sg_graph_view,
          sg_vertex_frontier,
          static_cast<size_t>(Bucket::cur),
          std::vector<size_t>{static_cast<size_t>(Bucket::next)},
          sg_src_properties.device_view(),
          sg_dst_properties.device_view(),
          [] __device__(vertex_t src, vertex_t dst, auto src_val, auto dst_val) {
            thrust::optional<vertex_t> result;
            if (src_val < dst_val) { result.emplace(src); }
            return result;
          },
          cugraph::reduce_op::any<vertex_t>(),
          cugraph::get_dataframe_buffer_cbegin(sg_property_buffer),
          thrust::make_discard_iterator() /* dummy */,
          [] __device__(auto v, auto v_val, auto pushed_val) {
            return thrust::optional<thrust::tuple<size_t, std::byte>>{
              thrust::make_tuple(static_cast<size_t>(Bucket::next), std::byte{0} /* dummy */)};
          });

        thrust::sort(handle.get_thrust_policy(),
                     sg_vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).begin(),
                     sg_vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).end());
        bool passed =
          thrust::equal(handle.get_thrust_policy(),
                        sg_vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).begin(),
                        sg_vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).end(),
                        mg_aggregate_frontier_dsts.begin());
        ASSERT_TRUE(passed);
      }
    }
  }
};

using Tests_MG_update_frontier_v_push_if_out_nbr_File =
  Tests_MG_UpdateFrontierVPushIfOutNbr<cugraph::test::File_Usecase>;
using Tests_MG_update_frontier_v_push_if_out_nbr_Rmat =
  Tests_MG_UpdateFrontierVPushIfOutNbr<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_File, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(std::get<0>(param),
                                                                       std::get<1>(param));
}

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_Rmat, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_File, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_File, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_update_frontier_v_push_if_out_nbr_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_update_frontier_v_push_if_out_nbr_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_update_frontier_v_push_if_out_nbr_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MG_update_frontier_v_push_if_out_nbr_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
