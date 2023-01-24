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

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <prims/per_v_pair_transform_dst_nbr_intersection.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename edge_t>
struct intersection_op_t {
  __device__ thrust::tuple<edge_t, edge_t> operator()(
    vertex_t v1,
    vertex_t v2,
    edge_t v0_prop /* out degree */,
    edge_t v1_prop /* out degree */,
    raft::device_span<vertex_t const> intersection) const
  {
    return thrust::make_tuple(v0_prop + v1_prop, static_cast<edge_t>(intersection.size()));
  }
};

struct Prims_Usecase {
  size_t num_vertex_pairs{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPerVPairTransformDstNbrIntersection
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGPerVPairTransformDstNbrIntersection() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Verify the results of per_v_pair_transform_dst_nbr_intersection primitive
  template <typename vertex_t, typename edge_t, typename weight_t, typename property_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();
    auto const row_comm_size =
      handle_->get_subcomm(cugraph::partition_2d::key_naming_t().row_name()).get_size();
    auto const col_comm_size =
      handle_->get_subcomm(cugraph::partition_2d::key_naming_t().col_name()).get_size();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> d_mg_renumber_map_labels{std::nullopt};
    std::tie(mg_graph, std::ignore, d_mg_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG per_v_pair_transform_dst_nbr_intersection primitive

    ASSERT_TRUE(
      mg_graph_view.number_of_vertices() >
      vertex_t{0});  // the code below to generate vertex pairs is invalid for an empty graph.

    auto mg_vertex_pair_buffer =
      cugraph::allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prims_usecase.num_vertex_pairs / comm_size +
          (static_cast<size_t>(comm_rank) < prims_usecase.num_vertex_pairs % comm_size ? 1 : 0),
        handle_->get_stream());
    thrust::tabulate(
      handle_->get_thrust_policy(),
      cugraph::get_dataframe_buffer_begin(mg_vertex_pair_buffer),
      cugraph::get_dataframe_buffer_end(mg_vertex_pair_buffer),
      [comm_rank, num_vertices = mg_graph_view.number_of_vertices()] __device__(size_t i) {
        cuco::detail::MurmurHash3_32<size_t>
          hash_func{};  // use hash_func to generate arbitrary vertex pairs
        auto v0 = static_cast<vertex_t>(hash_func(i + comm_rank) % num_vertices);
        auto v1 = static_cast<vertex_t>(hash_func(i + num_vertices + comm_rank) % num_vertices);
        return thrust::make_tuple(v0, v1);
      });

    auto h_vertex_partition_range_lasts = mg_graph_view.vertex_partition_range_lasts();
    std::tie(std::get<0>(mg_vertex_pair_buffer),
             std::get<1>(mg_vertex_pair_buffer),
             std::ignore,
             std::ignore) =
      cugraph::detail::shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                  edge_t,
                                                                                  weight_t,
                                                                                  int32_t>(
        *handle_,
        std::move(std::get<0>(mg_vertex_pair_buffer)),
        std::move(std::get<1>(mg_vertex_pair_buffer)),
        std::nullopt,
        std::nullopt,
        h_vertex_partition_range_lasts);

    auto mg_result_buffer = cugraph::allocate_dataframe_buffer<thrust::tuple<edge_t, edge_t>>(
      cugraph::size_dataframe_buffer(mg_vertex_pair_buffer), handle_->get_stream());
    auto mg_out_degrees = mg_graph_view.compute_out_degrees(*handle_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG per_v_pair_transform_dst_nbr_intersection");
    }

    cugraph::per_v_pair_transform_dst_nbr_intersection(
      *handle_,
      mg_graph_view,
      cugraph::get_dataframe_buffer_begin(mg_vertex_pair_buffer),
      cugraph::get_dataframe_buffer_end(mg_vertex_pair_buffer),
      mg_out_degrees.begin(),
      intersection_op_t<vertex_t, edge_t>{},
      cugraph::get_dataframe_buffer_begin(mg_result_buffer));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. validate MG results

    if (prims_usecase.check_correctness) {
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        std::get<0>(mg_vertex_pair_buffer).data(),
        cugraph::size_dataframe_buffer(mg_vertex_pair_buffer),
        (*d_mg_renumber_map_labels).data(),
        h_vertex_partition_range_lasts);
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        std::get<1>(mg_vertex_pair_buffer).data(),
        cugraph::size_dataframe_buffer(mg_vertex_pair_buffer),
        (*d_mg_renumber_map_labels).data(),
        h_vertex_partition_range_lasts);

      cugraph::graph_t<vertex_t, edge_t, false, false> unrenumbered_graph(*handle_);
      std::tie(unrenumbered_graph, std::ignore, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
          *handle_, input_usecase, false, false);

      auto unrenumbered_graph_view = unrenumbered_graph.view();

      auto sg_result_buffer = cugraph::allocate_dataframe_buffer<thrust::tuple<edge_t, edge_t>>(
        cugraph::size_dataframe_buffer(mg_vertex_pair_buffer), handle_->get_stream());
      auto sg_out_degrees = unrenumbered_graph_view.compute_out_degrees(*handle_);

      cugraph::per_v_pair_transform_dst_nbr_intersection(
        *handle_,
        unrenumbered_graph_view,
        cugraph::get_dataframe_buffer_begin(mg_vertex_pair_buffer /* now unrenumbered */),
        cugraph::get_dataframe_buffer_end(mg_vertex_pair_buffer /* now unrenumbered */),
        sg_out_degrees.begin(),
        intersection_op_t<vertex_t, edge_t>{},
        cugraph::get_dataframe_buffer_begin(sg_result_buffer));

      bool valid = thrust::equal(handle_->get_thrust_policy(),
                                 cugraph::get_dataframe_buffer_begin(mg_result_buffer),
                                 cugraph::get_dataframe_buffer_end(mg_result_buffer),
                                 cugraph::get_dataframe_buffer_begin(sg_result_buffer));

      valid = static_cast<bool>(cugraph::host_scalar_allreduce(handle_->get_comms(),
                                                               static_cast<int>(valid),
                                                               raft::comms::op_t::MIN,
                                                               handle_->get_stream()));
      ASSERT_TRUE(valid);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGPerVPairTransformDstNbrIntersection<input_usecase_t>::handle_ = nullptr;

using Tests_MGPerVPairTransformDstNbrIntersection_File =
  Tests_MGPerVPairTransformDstNbrIntersection<cugraph::test::File_Usecase>;
using Tests_MGPerVPairTransformDstNbrIntersection_Rmat =
  Tests_MGPerVPairTransformDstNbrIntersection<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_File, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(std::get<0>(param),
                                                                       std::get<1>(param));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int64FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt64Int64FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVPairTransformDstNbrIntersection_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{size_t{1024}, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGPerVPairTransformDstNbrIntersection_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{size_t{1024}, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGPerVPairTransformDstNbrIntersection_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{size_t{1024 * 1024}, false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
