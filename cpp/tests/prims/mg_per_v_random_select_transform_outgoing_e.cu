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
#include <utilities/high_res_clock.h>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <prims/per_v_random_select_transform_outgoing_e.cuh>
#include <prims/vertex_frontier.cuh>

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <cugraph/graph_view.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename property_t>
struct e_op_t {
  using result_t = decltype(cugraph::thrust_tuple_cat(thrust::tuple<vertex_t, vertex_t>{},
                                                      cugraph::to_thrust_tuple(property_t{}),
                                                      cugraph::to_thrust_tuple(property_t{})));

  __device__ result_t operator()(vertex_t src,
                                 vertex_t dst,
                                 property_t src_prop,
                                 property_t dst_prop) const
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
      static_assert(thrust::tuple_size<property_t>::value == size_t{2});
      return thrust::make_tuple(src,
                                dst,
                                thrust::get<0>(src_prop),
                                thrust::get<1>(src_prop),
                                thrust::get<0>(dst_prop),
                                thrust::get<1>(dst_prop));
    } else {
      return thrust::make_tuple(src, dst, src_prop, dst_prop);
    }
  }
};

struct Prims_Usecase {
  size_t K{0};
  bool with_replacement{false};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPerVRandomSelectTransformOutgoingE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGPerVRandomSelectTransformOutgoingE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Verify the results of per_v_random_select_transform_outgoing_e primitive
  template <typename vertex_t, typename edge_t, typename weight_t, typename property_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResClock hr_clock{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, prims_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG per_v_random_select_transform_outgoing_e primitive

    const int hash_bin_count = 5;

    auto mg_vertex_prop = cugraph::test::generate<vertex_t, property_t>::vertex_property(
      *handle_, *d_mg_renumber_map_labels, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<vertex_t, property_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<vertex_t, property_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    auto mg_vertex_buffer = rmm::device_uvector<vertex_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
    thrust::sequence(handle_->get_thrust_policy(),
                     cugraph::get_dataframe_buffer_begin(mg_vertex_buffer),
                     cugraph::get_dataframe_buffer_end(mg_vertex_buffer),
                     mg_graph_view.local_vertex_partition_range_first());

    constexpr size_t bucket_idx_cur = 0;
    constexpr size_t num_buckets    = 1;

    cugraph::vertex_frontier_t<vertex_t, void, true, false> mg_vertex_frontier(*handle_,
                                                                               num_buckets);
    mg_vertex_frontier.bucket(bucket_idx_cur)
      .insert(cugraph::get_dataframe_buffer_begin(mg_vertex_buffer),
              cugraph::get_dataframe_buffer_end(mg_vertex_buffer));

    raft::random::RngState rng_state(static_cast<uint64_t>(handle_->get_comms().get_rank()));

    using result_t = decltype(cugraph::thrust_tuple_cat(thrust::tuple<vertex_t, vertex_t>{},
                                                        cugraph::to_thrust_tuple(property_t{}),
                                                        cugraph::to_thrust_tuple(property_t{})));

    std::optional<result_t> invalid_value{std::nullopt};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    auto [sample_offsets, sample_e_op_results] =
      cugraph::per_v_random_select_transform_outgoing_e(*handle_,
                                                        mg_graph_view,
                                                        mg_vertex_frontier.bucket(bucket_idx_cur),
                                                        mg_src_prop.view(),
                                                        mg_dst_prop.view(),
                                                        e_op_t<vertex_t, property_t>{},
                                                        rng_state,
                                                        prims_usecase.K,
                                                        prims_usecase.with_replacement,
                                                        invalid_value);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG per_v_random_select_transform_outgoing_e took " << elapsed_time * 1e-6
                << " s.\n";
    }

    // 3. validate MG results

    if (prims_usecase.check_correctness) {
#if 0
      cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
          *handle_, input_usecase, prims_usecase.test_weighted, false);
      auto sg_graph_view = sg_graph.view();

// 1. check whether sources coincide with the local vertices in the frontier or not

// 2. check sample counts

// 3. check destinations exist in the input graph

// 4. check source/destination property values

      auto sg_vertex_property_data = generate<property_t>::vertex_property(
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
        hash_bin_count,
        *handle_);
      auto sg_dst_prop =
        generate<property_t>::dst_property(*handle_, sg_graph_view, sg_vertex_property_data);
      auto sg_src_prop =
        generate<property_t>::src_property(*handle_, sg_graph_view, sg_vertex_property_data);

      auto expected_result = count_if_e(
        *handle_,
        sg_graph_view,
        sg_src_prop.view(),
        sg_dst_prop.view(),
        [] __device__(auto src, auto dst, weight_t, auto src_property, auto dst_property) {
          return src_property < dst_property;
        });
      ASSERT_TRUE(expected_result == result);
#endif
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGPerVRandomSelectTransformOutgoingE<input_usecase_t>::handle_ = nullptr;

using Tests_MGPerVRandomSelectTransformOutgoingE_File =
  Tests_MGPerVRandomSelectTransformOutgoingE<cugraph::test::File_Usecase>;
using Tests_MGPerVRandomSelectTransformOutgoingE_Rmat =
  Tests_MGPerVRandomSelectTransformOutgoingE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_File, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(std::get<0>(param),
                                                                       std::get<1>(param));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt32Int32FloatTupleIntFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPerVRandomSelectTransformOutgoingE_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGPerVRandomSelectTransformOutgoingE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGPerVRandomSelectTransformOutgoingE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_test,
  Tests_MGPerVRandomSelectTransformOutgoingE_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
