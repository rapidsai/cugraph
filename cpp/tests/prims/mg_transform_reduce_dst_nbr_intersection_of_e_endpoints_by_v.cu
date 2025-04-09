/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "prims/transform_e.cuh"
#include "prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
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

#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename edge_t>
struct intersection_op_t {
  __device__ thrust::tuple<edge_t, edge_t, edge_t> operator()(
    vertex_t v0,
    vertex_t v1,
    edge_t v0_prop,
    edge_t v1_prop,
    raft::device_span<vertex_t const> intersection) const
  {
    return thrust::make_tuple(
      v0_prop + v1_prop, v0_prop + v1_prop, static_cast<edge_t>(intersection.size()));
  }
};

struct Prims_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Verify the results of transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v primitive
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (prims_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v primitive

    const int hash_bin_count = 5;
    const int initial_value  = 4;

    auto property_initial_value =
      cugraph::test::generate<decltype(mg_graph_view), edge_t>::initial_value(initial_value);

    auto mg_vertex_prop = cugraph::test::generate<decltype(mg_graph_view), edge_t>::vertex_property(
      *handle_, *mg_renumber_map, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<decltype(mg_graph_view), edge_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<decltype(mg_graph_view), edge_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    auto mg_result_buffer = rmm::device_uvector<edge_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v");
    }

    cugraph::transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v(
      *handle_,
      mg_graph_view,
      mg_src_prop.view(),
      mg_dst_prop.view(),
      intersection_op_t<vertex_t, edge_t>{},
      property_initial_value,
      mg_result_buffer.begin());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. validate MG results

    if (prims_usecase.check_correctness) {
      rmm::device_uvector<edge_t> mg_aggregate_result_buffer(0, handle_->get_stream());
      std::tie(std::ignore, mg_aggregate_result_buffer) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<edge_t const>(mg_result_buffer.data(), mg_result_buffer.size()));

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();

        auto sg_vertex_prop =
          cugraph::test::generate<decltype(sg_graph_view), edge_t>::vertex_property(
            *handle_,
            thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
            thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
            hash_bin_count);
        auto sg_src_prop = cugraph::test::generate<decltype(sg_graph_view), edge_t>::src_property(
          *handle_, sg_graph_view, sg_vertex_prop);
        auto sg_dst_prop = cugraph::test::generate<decltype(sg_graph_view), edge_t>::dst_property(
          *handle_, sg_graph_view, sg_vertex_prop);

        auto sg_result_buffer = cugraph::allocate_dataframe_buffer<edge_t>(
          sg_graph_view.number_of_vertices(), handle_->get_stream());

        cugraph::transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v(
          *handle_,
          sg_graph_view,
          sg_src_prop.view(),
          sg_dst_prop.view(),
          intersection_op_t<vertex_t, edge_t>{},
          property_initial_value,
          sg_result_buffer.begin());

        bool valid = thrust::equal(handle_->get_thrust_policy(),
                                   mg_aggregate_result_buffer.begin(),
                                   mg_aggregate_result_buffer.end(),
                                   sg_result_buffer.begin());

        ASSERT_TRUE(valid);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV<input_usecase_t>::handle_ = nullptr;

using Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_File =
  Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV<cugraph::test::File_Usecase>;
using Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_Rmat =
  Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, true}, Prims_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_Rmat,
                         ::testing::Combine(::testing::Values(Prims_Usecase{false, true},
                                                              Prims_Usecase{true, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTransformReduceDstNbrIntersectionOfEEndpointsByV_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false}, Prims_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
