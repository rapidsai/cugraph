/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "prims/per_v_pair_transform_dst_nbr_intersection.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
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
#include <rmm/exec_policy.hpp>

#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename edge_t, typename weight_t>
struct intersection_op_t {
  __device__ thrust::tuple<weight_t, weight_t> operator()(
    vertex_t a,
    vertex_t b,
    weight_t weight_a /* weighted out degree */,
    weight_t weight_b /* weighted out degree */,
    raft::device_span<vertex_t const> intersection,
    raft::device_span<weight_t const> intersected_property_values_a,
    raft::device_span<weight_t const> intersected_property_values_b) const
  {
    weight_t min_weight_a_intersect_b = weight_t{0};
    weight_t max_weight_a_intersect_b = weight_t{0};
    weight_t sum_of_intersected_a     = weight_t{0};
    weight_t sum_of_intersected_b     = weight_t{0};

    for (size_t k = 0; k < intersection.size(); k++) {
      min_weight_a_intersect_b +=
        min(intersected_property_values_a[k], intersected_property_values_b[k]);
      max_weight_a_intersect_b +=
        max(intersected_property_values_a[k], intersected_property_values_b[k]);
      sum_of_intersected_a += intersected_property_values_a[k];
      sum_of_intersected_b += intersected_property_values_b[k];
    }

    weight_t sum_of_uniq_a = weight_a - sum_of_intersected_a;
    weight_t sum_of_uniq_b = weight_b - sum_of_intersected_b;

    max_weight_a_intersect_b += sum_of_uniq_a + sum_of_uniq_b;

    return thrust::make_tuple(min_weight_a_intersect_b, max_weight_a_intersect_b);
  }
};

struct Prims_Usecase {
  size_t num_vertex_pairs{0};
  bool edge_masking{false};
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
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    constexpr bool store_transposed = false;

    constexpr bool test_weighted    = true;
    constexpr bool renumber         = true;
    constexpr bool drop_self_loops  = false;
    constexpr bool drop_multi_edges = true;

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weight, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view       = mg_graph.view();
    auto mg_edge_weight_view = (*mg_edge_weight).view();

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (prims_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

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
        cuco::murmurhash3_32<size_t>
          hash_func{};  // use hash_func to generate arbitrary vertex pairs
        auto v0 = static_cast<vertex_t>(hash_func(i + comm_rank) % num_vertices);
        auto v1 = static_cast<vertex_t>(hash_func(i + num_vertices + comm_rank) % num_vertices);
        return thrust::make_tuple(v0, v1);
      });

    auto h_vertex_partition_range_lasts = mg_graph_view.vertex_partition_range_lasts();
    std::tie(std::get<0>(mg_vertex_pair_buffer),
             std::get<1>(mg_vertex_pair_buffer),
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore) =
      cugraph::detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<
        vertex_t,
        edge_t,
        weight_t,
        int32_t,
        int32_t>(*handle_,
                 std::move(std::get<0>(mg_vertex_pair_buffer)),
                 std::move(std::get<1>(mg_vertex_pair_buffer)),
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 h_vertex_partition_range_lasts);

    auto mg_result_buffer = cugraph::allocate_dataframe_buffer<thrust::tuple<weight_t, weight_t>>(
      cugraph::size_dataframe_buffer(mg_vertex_pair_buffer), handle_->get_stream());
    auto mg_out_weight_sums = compute_out_weight_sums(*handle_, mg_graph_view, mg_edge_weight_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG per_v_pair_transform_dst_nbr_intersection");
    }

    cugraph::per_v_pair_transform_dst_nbr_intersection(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      cugraph::get_dataframe_buffer_begin(mg_vertex_pair_buffer),
      cugraph::get_dataframe_buffer_end(mg_vertex_pair_buffer),
      mg_out_weight_sums.begin(),
      intersection_op_t<vertex_t, edge_t, weight_t>{},
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
        (*mg_renumber_map).data(),
        h_vertex_partition_range_lasts);
      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        std::get<1>(mg_vertex_pair_buffer).data(),
        cugraph::size_dataframe_buffer(mg_vertex_pair_buffer),
        (*mg_renumber_map).data(),
        h_vertex_partition_range_lasts);

      auto mg_aggregate_vertex_pair_buffer =
        cugraph::allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          0, handle_->get_stream());
      std::get<0>(mg_aggregate_vertex_pair_buffer) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<0>(mg_vertex_pair_buffer).data(),
                                      std::get<0>(mg_vertex_pair_buffer).size());
      std::get<1>(mg_aggregate_vertex_pair_buffer) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<1>(mg_vertex_pair_buffer).data(),
                                      std::get<1>(mg_vertex_pair_buffer).size());

      auto mg_aggregate_result_buffer =
        cugraph::allocate_dataframe_buffer<thrust::tuple<weight_t, weight_t>>(
          0, handle_->get_stream());
      std::get<0>(mg_aggregate_result_buffer) = cugraph::test::device_gatherv(
        *handle_, std::get<0>(mg_result_buffer).data(), std::get<0>(mg_result_buffer).size());
      std::get<1>(mg_aggregate_result_buffer) = cugraph::test::device_gatherv(
        *handle_, std::get<1>(mg_result_buffer).data(), std::get<1>(mg_result_buffer).size());

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);

      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, false>,
                                 weight_t>>
        sg_edge_weight{std::nullopt};

      std::tie(sg_graph, sg_edge_weight, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          mg_edge_weight
            ? std::make_optional(mg_edge_weight_view)
            : std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();
        auto sg_result_buffer =
          cugraph::allocate_dataframe_buffer<thrust::tuple<weight_t, weight_t>>(
            cugraph::size_dataframe_buffer(mg_aggregate_vertex_pair_buffer), handle_->get_stream());

        rmm::device_uvector<weight_t> sg_out_weight_sums =
          compute_out_weight_sums(*handle_, sg_graph_view, (*sg_edge_weight).view());

        cugraph::per_v_pair_transform_dst_nbr_intersection(
          *handle_,
          sg_graph_view,
          (*sg_edge_weight).view(),
          cugraph::get_dataframe_buffer_begin(
            mg_aggregate_vertex_pair_buffer /* now unrenumbered */),
          cugraph::get_dataframe_buffer_end(mg_aggregate_vertex_pair_buffer /* now unrenumbered
          */), sg_out_weight_sums.begin(),  intersection_op_t<vertex_t, edge_t, weight_t>{},
          cugraph::get_dataframe_buffer_begin(sg_result_buffer));

        auto threshold_ratio     = weight_t{1e-4};
        auto threshold_magnitude = std::numeric_limits<weight_t>::min();
        auto nearly_equal = [threshold_ratio, threshold_magnitude] __device__(auto lhs, auto rhs) {
          return (fabs(thrust::get<0>(lhs) - thrust::get<0>(rhs)) <
                  max(max(thrust::get<0>(lhs), thrust::get<0>(rhs)) * threshold_ratio,
                      threshold_magnitude)) &&
                 (fabs(thrust::get<1>(lhs) - thrust::get<1>(rhs)) <
                  max(max(thrust::get<1>(lhs), thrust::get<1>(rhs)) * threshold_ratio,
                      threshold_magnitude));
        };
        bool valid = thrust::equal(handle_->get_thrust_policy(),
                                   cugraph::get_dataframe_buffer_begin(mg_aggregate_result_buffer),
                                   cugraph::get_dataframe_buffer_end(mg_aggregate_result_buffer),
                                   cugraph::get_dataframe_buffer_begin(sg_result_buffer),
                                   nearly_equal);
        ASSERT_TRUE(valid);
      }
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

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVPairTransformDstNbrIntersection_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVPairTransformDstNbrIntersection_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{size_t{10}, false, true},
                      Prims_Usecase{size_t{10}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGPerVPairTransformDstNbrIntersection_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{size_t{1024}, false, true},
                      Prims_Usecase{size_t{1024}, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGPerVPairTransformDstNbrIntersection_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{size_t{1024 * 1024}, false, false},
                      Prims_Usecase{size_t{1024 * 1024}, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
