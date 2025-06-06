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

#include "prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cuco/hash_functions.cuh>

#include <gtest/gtest.h>

#include <random>
#include <sstream>

template <typename key_t, typename vertex_t, typename payload_t>
struct e_op_t {
  __device__ auto operator()(key_t optionally_tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t) const
  {
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      if constexpr (std::is_same_v<payload_t, void>) {
        return;
      } else {
        return static_cast<payload_t>(1);
      }
    } else {
      auto tag = thrust::get<1>(optionally_tagged_src);
      if constexpr (std::is_same_v<payload_t, void>) {
        return tag;
      } else {
        return thrust::make_tuple(tag, static_cast<payload_t>(1));
      }
    }
  }
};

struct Prims_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTransformReduceVFrontierOutgoingEBySrcDst
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGTransformReduceVFrontierOutgoingEBySrcDst() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of transform_reduce_v_frontier_outgoing_e_by_dst primitive
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename tag_t,
            typename payload_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;

    using key_t =
      std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;

    static_assert(std::is_same_v<tag_t, void> || std::is_arithmetic_v<tag_t>);
    static_assert(std::is_same_v<payload_t, void> ||
                  cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<payload_t>::value);
    if constexpr (cugraph::is_thrust_tuple<payload_t>::value) {
      static_assert(thrust::tuple_size<payload_t>::value == size_t{2});
    }

    HighResTimer hr_timer{};

    // 1. create MG graph

    constexpr bool renumber = true;  // needs to be true for multi gpu case
    constexpr bool store_transposed =
      false;  // needs to be false for using transform_reduce_v_frontier_outgoing_e_by_dst
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (prims_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG transform reduce

    auto mg_key_buffer = cugraph::allocate_dataframe_buffer<key_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
    if constexpr (std::is_same_v<tag_t, void>) {
      thrust::sequence(handle_->get_thrust_policy(),
                       cugraph::get_dataframe_buffer_begin(mg_key_buffer),
                       cugraph::get_dataframe_buffer_end(mg_key_buffer),
                       mg_graph_view.local_vertex_partition_range_first());
    } else {
      thrust::tabulate(handle_->get_thrust_policy(),
                       cugraph::get_dataframe_buffer_begin(mg_key_buffer),
                       cugraph::get_dataframe_buffer_end(mg_key_buffer),
                       [mg_renumber_map_labels = (*mg_renumber_map).data(),
                        local_vertex_partition_range_first =
                          mg_graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
                         return thrust::make_tuple(
                           static_cast<vertex_t>(local_vertex_partition_range_first + i),
                           static_cast<tag_t>(*(mg_renumber_map_labels + i) % size_t{10}));
                       });
    }

    constexpr size_t bucket_idx_cur = 0;
    constexpr size_t num_buckets    = 1;

    cugraph::vertex_frontier_t<vertex_t, tag_t, true, true> mg_vertex_frontier(*handle_,
                                                                               num_buckets);
    mg_vertex_frontier.bucket(bucket_idx_cur)
      .insert(cugraph::get_dataframe_buffer_begin(mg_key_buffer),
              cugraph::get_dataframe_buffer_end(mg_key_buffer));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transform_reduce_v_frontier_outgoing_e_by_dst");
    }

    auto mg_reduce_by_dst_new_frontier_key_buffer =
      cugraph::allocate_dataframe_buffer<key_t>(0, handle_->get_stream());
    [[maybe_unused]] auto mg_reduce_by_dst_payload_buffer =
      cugraph::detail::allocate_optional_dataframe_buffer<payload_t>(0, handle_->get_stream());

    if constexpr (std::is_same_v<payload_t, void>) {
      mg_reduce_by_dst_new_frontier_key_buffer =
        cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
          *handle_,
          mg_graph_view,
          mg_vertex_frontier.bucket(bucket_idx_cur),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          e_op_t<key_t, vertex_t, payload_t>{},
          cugraph::reduce_op::null{});
    } else {
      std::tie(mg_reduce_by_dst_new_frontier_key_buffer, mg_reduce_by_dst_payload_buffer) =
        cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
          *handle_,
          mg_graph_view,
          mg_vertex_frontier.bucket(bucket_idx_cur),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          e_op_t<key_t, vertex_t, payload_t>{},
          cugraph::reduce_op::plus<payload_t>{});
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (prims_usecase.check_correctness) {
      auto mg_reduce_by_dst_aggregate_new_frontier_key_buffer =
        cugraph::allocate_dataframe_buffer<key_t>(0, handle_->get_stream());
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        cugraph::unrenumber_local_int_vertices(*handle_,
                                               mg_reduce_by_dst_new_frontier_key_buffer.data(),
                                               mg_reduce_by_dst_new_frontier_key_buffer.size(),
                                               (*mg_renumber_map).data(),
                                               mg_graph_view.local_vertex_partition_range_first(),
                                               mg_graph_view.local_vertex_partition_range_last());
        mg_reduce_by_dst_aggregate_new_frontier_key_buffer =
          cugraph::test::device_gatherv(*handle_,
                                        mg_reduce_by_dst_new_frontier_key_buffer.data(),
                                        mg_reduce_by_dst_new_frontier_key_buffer.size());
      } else {
        cugraph::unrenumber_local_int_vertices(
          *handle_,
          std::get<0>(mg_reduce_by_dst_new_frontier_key_buffer).data(),
          std::get<0>(mg_reduce_by_dst_new_frontier_key_buffer).size(),
          (*mg_renumber_map).data(),
          mg_graph_view.local_vertex_partition_range_first(),
          mg_graph_view.local_vertex_partition_range_last());
        std::get<0>(mg_reduce_by_dst_aggregate_new_frontier_key_buffer) =
          cugraph::test::device_gatherv(
            *handle_,
            std::get<0>(mg_reduce_by_dst_new_frontier_key_buffer).data(),
            std::get<0>(mg_reduce_by_dst_new_frontier_key_buffer).size());
        std::get<1>(mg_reduce_by_dst_aggregate_new_frontier_key_buffer) =
          cugraph::test::device_gatherv(
            *handle_,
            std::get<1>(mg_reduce_by_dst_new_frontier_key_buffer).data(),
            std::get<1>(mg_reduce_by_dst_new_frontier_key_buffer).size());
      }

      [[maybe_unused]] auto mg_reduce_by_dst_aggregate_payload_buffer =
        cugraph::detail::allocate_optional_dataframe_buffer<payload_t>(0, handle_->get_stream());
      if constexpr (!std::is_same_v<payload_t, void>) {
        if constexpr (std::is_arithmetic_v<payload_t>) {
          mg_reduce_by_dst_aggregate_payload_buffer =
            cugraph::test::device_gatherv(*handle_,
                                          mg_reduce_by_dst_payload_buffer.data(),
                                          mg_reduce_by_dst_payload_buffer.size());
        } else {
          std::get<0>(mg_reduce_by_dst_aggregate_payload_buffer) =
            cugraph::test::device_gatherv(*handle_,
                                          std::get<0>(mg_reduce_by_dst_payload_buffer).data(),
                                          std::get<0>(mg_reduce_by_dst_payload_buffer).size());
          std::get<1>(mg_reduce_by_dst_aggregate_payload_buffer) =
            cugraph::test::device_gatherv(*handle_,
                                          std::get<1>(mg_reduce_by_dst_payload_buffer).data(),
                                          std::get<1>(mg_reduce_by_dst_payload_buffer).size());
        }
      }

      cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
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

      if (handle_->get_comms().get_rank() == int{0}) {
        if constexpr (std::is_same_v<payload_t, void>) {
          thrust::sort(
            handle_->get_thrust_policy(),
            cugraph::get_dataframe_buffer_begin(mg_reduce_by_dst_aggregate_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_end(mg_reduce_by_dst_aggregate_new_frontier_key_buffer));
        } else {
          thrust::sort_by_key(
            handle_->get_thrust_policy(),
            cugraph::get_dataframe_buffer_begin(mg_reduce_by_dst_aggregate_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_end(mg_reduce_by_dst_aggregate_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_begin(mg_reduce_by_dst_aggregate_payload_buffer));
        }

        auto sg_graph_view = sg_graph.view();

        auto sg_key_buffer = cugraph::allocate_dataframe_buffer<key_t>(
          sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
        if constexpr (std::is_same_v<tag_t, void>) {
          thrust::sequence(handle_->get_thrust_policy(),
                           cugraph::get_dataframe_buffer_begin(sg_key_buffer),
                           cugraph::get_dataframe_buffer_end(sg_key_buffer),
                           sg_graph_view.local_vertex_partition_range_first());
        } else {
          thrust::tabulate(handle_->get_thrust_policy(),
                           cugraph::get_dataframe_buffer_begin(sg_key_buffer),
                           cugraph::get_dataframe_buffer_end(sg_key_buffer),
                           [] __device__(size_t i) {
                             return thrust::make_tuple(
                               static_cast<vertex_t>(i),
                               static_cast<tag_t>(static_cast<vertex_t>(i) % size_t{10}));
                           });
        }

        cugraph::vertex_frontier_t<vertex_t, tag_t, false, true> sg_vertex_frontier(*handle_,
                                                                                    num_buckets);
        sg_vertex_frontier.bucket(bucket_idx_cur)
          .insert(cugraph::get_dataframe_buffer_begin(sg_key_buffer),
                  cugraph::get_dataframe_buffer_end(sg_key_buffer));

        auto sg_reduce_by_dst_new_frontier_key_buffer =
          cugraph::allocate_dataframe_buffer<key_t>(0, handle_->get_stream());
        [[maybe_unused]] auto sg_reduce_by_dst_payload_buffer =
          cugraph::detail::allocate_optional_dataframe_buffer<payload_t>(0, handle_->get_stream());
        if constexpr (std::is_same_v<payload_t, void>) {
          sg_reduce_by_dst_new_frontier_key_buffer =
            cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
              *handle_,
              sg_graph_view,
              sg_vertex_frontier.bucket(bucket_idx_cur),
              cugraph::edge_src_dummy_property_t{}.view(),
              cugraph::edge_dst_dummy_property_t{}.view(),
              cugraph::edge_dummy_property_t{}.view(),
              e_op_t<key_t, vertex_t, payload_t>{},
              cugraph::reduce_op::null{});
        } else {
          std::tie(sg_reduce_by_dst_new_frontier_key_buffer, sg_reduce_by_dst_payload_buffer) =
            cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
              *handle_,
              sg_graph_view,
              sg_vertex_frontier.bucket(bucket_idx_cur),
              cugraph::edge_src_dummy_property_t{}.view(),
              cugraph::edge_dst_dummy_property_t{}.view(),
              cugraph::edge_dummy_property_t{}.view(),
              e_op_t<key_t, vertex_t, payload_t>{},
              cugraph::reduce_op::plus<payload_t>{});
        }

        if constexpr (std::is_same_v<payload_t, void>) {
          thrust::sort(
            handle_->get_thrust_policy(),
            cugraph::get_dataframe_buffer_begin(sg_reduce_by_dst_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_end(sg_reduce_by_dst_new_frontier_key_buffer));
        } else {
          thrust::sort_by_key(
            handle_->get_thrust_policy(),
            cugraph::get_dataframe_buffer_begin(sg_reduce_by_dst_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_end(sg_reduce_by_dst_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_begin(sg_reduce_by_dst_payload_buffer));
        }

        auto key_passed = thrust::equal(
          handle_->get_thrust_policy(),
          cugraph::get_dataframe_buffer_begin(sg_reduce_by_dst_new_frontier_key_buffer),
          cugraph::get_dataframe_buffer_end(sg_reduce_by_dst_new_frontier_key_buffer),
          cugraph::get_dataframe_buffer_begin(mg_reduce_by_dst_aggregate_new_frontier_key_buffer));
        ASSERT_TRUE(key_passed);

        if constexpr (!std::is_same_v<payload_t, void>) {
          bool payload_passed = thrust::equal(
            handle_->get_thrust_policy(),
            cugraph::get_dataframe_buffer_begin(sg_reduce_by_dst_payload_buffer),
            cugraph::get_dataframe_buffer_begin(sg_reduce_by_dst_payload_buffer),
            cugraph::get_dataframe_buffer_end(mg_reduce_by_dst_aggregate_payload_buffer));
          ASSERT_TRUE(payload_passed);
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t>
  Tests_MGTransformReduceVFrontierOutgoingEBySrcDst<input_usecase_t>::handle_ = nullptr;

using Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File =
  Tests_MGTransformReduceVFrontierOutgoingEBySrcDst<cugraph::test::File_Usecase>;
using Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat =
  Tests_MGTransformReduceVFrontierOutgoingEBySrcDst<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File, CheckInt32Int32FloatVoidVoid)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, void>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat, CheckInt32Int32FloatVoidVoid)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, void>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File, CheckInt32Int32FloatVoidInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat, CheckInt32Int32FloatVoidInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File,
       CheckInt32Int32FloatVoidTupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, thrust::tuple<float, int32_t>>(
    std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat,
       CheckInt32Int32FloatVoidTupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, thrust::tuple<float, int32_t>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File, CheckInt32Int32FloatInt32Void)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, void>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat, CheckInt32Int32FloatInt32Void)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, void>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File, CheckInt32Int32FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, int32_t>(std::get<0>(param),
                                                              std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat, CheckInt32Int32FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File,
       CheckInt32Int32FloatInt32TupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, thrust::tuple<float, int32_t>>(
    std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat,
       CheckInt32Int32FloatInt32TupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, thrust::tuple<float, int32_t>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File, CheckInt64Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t, int32_t>(std::get<0>(param),
                                                              std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat, CheckInt64Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}, Prims_Usecase{true, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, true}, Prims_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat,
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
  Tests_MGTransformReduceVFrontierOutgoingEBySrcDst_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false}, Prims_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
