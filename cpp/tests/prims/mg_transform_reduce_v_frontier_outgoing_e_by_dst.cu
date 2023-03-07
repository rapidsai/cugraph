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
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <prims/vertex_frontier.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <cuco/detail/hash_functions.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <sstream>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

template <typename key_t, typename vertex_t, typename property_t, typename payload_t>
struct e_op_t {
  __device__ auto operator()(key_t optionally_tagged_src,
                             vertex_t dst,
                             property_t src_val,
                             property_t dst_val,
                             thrust::nullopt_t) const
  {
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      if constexpr (std::is_same_v<payload_t, void>) {
        return src_val < dst_val ? thrust::optional<std::byte>{std::byte{0}} /* dummy */
                                 : thrust::nullopt;
      } else {
        return src_val < dst_val ? thrust::optional<payload_t>{static_cast<payload_t>(1)}
                                 : thrust::nullopt;
      }
    } else {
      auto tag = thrust::get<1>(optionally_tagged_src);
      if constexpr (std::is_same_v<payload_t, void>) {
        return src_val < dst_val ? thrust::optional<decltype(tag)>{tag} : thrust::nullopt;
      } else {
        return src_val < dst_val
                 ? thrust::optional<thrust::tuple<decltype(tag), payload_t>>{thrust::make_tuple(
                     tag, static_cast<payload_t>(1))}
                 : thrust::nullopt;
      }
    }
  }
};

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTransformReduceVFrontierOutgoingEByDst
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGTransformReduceVFrontierOutgoingEByDst() {}

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
    using property_t = int32_t;

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

    constexpr bool is_multi_gpu = true;
    constexpr bool renumber     = true;  // needs to be true for multi gpu case
    constexpr bool store_transposed =
      false;  // needs to be false for using transform_reduce_v_frontier_outgoing_e_by_dst
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, is_multi_gpu> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> d_mg_renumber_map_labels{std::nullopt};
    std::tie(mg_graph, std::ignore, d_mg_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, is_multi_gpu>(
        *handle_, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG transform reduce

    const int hash_bin_count = 5;

    auto mg_vertex_prop = cugraph::test::generate<vertex_t, property_t>::vertex_property(
      *handle_, *d_mg_renumber_map_labels, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<vertex_t, property_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<vertex_t, property_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

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
                       [mg_renumber_map_labels = (*d_mg_renumber_map_labels).data(),
                        local_vertex_partition_range_first =
                          mg_graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
                         return thrust::make_tuple(
                           static_cast<vertex_t>(local_vertex_partition_range_first + i),
                           static_cast<tag_t>(*(mg_renumber_map_labels + i) % size_t{10}));
                       });
    }

    constexpr size_t bucket_idx_cur = 0;
    constexpr size_t num_buckets    = 1;

    cugraph::vertex_frontier_t<vertex_t, tag_t, is_multi_gpu, true> mg_vertex_frontier(*handle_,
                                                                                       num_buckets);
    mg_vertex_frontier.bucket(bucket_idx_cur)
      .insert(cugraph::get_dataframe_buffer_begin(mg_key_buffer),
              cugraph::get_dataframe_buffer_end(mg_key_buffer));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transform_reduce_v_frontier_outgoing_e_by_dst");
    }

    auto mg_new_frontier_key_buffer =
      cugraph::allocate_dataframe_buffer<key_t>(0, handle_->get_stream());
    [[maybe_unused]] auto mg_payload_buffer =
      cugraph::detail::allocate_optional_dataframe_buffer<payload_t>(0, handle_->get_stream());

    if constexpr (std::is_same_v<payload_t, void>) {
      mg_new_frontier_key_buffer = cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
        *handle_,
        mg_graph_view,
        mg_vertex_frontier.bucket(bucket_idx_cur),
        mg_src_prop.view(),
        mg_dst_prop.view(),
        cugraph::edge_dummy_property_t{}.view(),
        e_op_t<key_t, vertex_t, property_t, payload_t>{},
        cugraph::reduce_op::null{});
    } else {
      std::tie(mg_new_frontier_key_buffer, mg_payload_buffer) =
        cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
          *handle_,
          mg_graph_view,
          mg_vertex_frontier.bucket(bucket_idx_cur),
          mg_src_prop.view(),
          mg_dst_prop.view(),
          cugraph::edge_dummy_property_t{}.view(),
          e_op_t<key_t, vertex_t, property_t, payload_t>{},
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
      auto mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());

      auto mg_aggregate_new_frontier_key_buffer =
        cugraph::allocate_dataframe_buffer<key_t>(0, handle_->get_stream());
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        mg_aggregate_new_frontier_key_buffer = cugraph::test::device_gatherv(
          *handle_, mg_new_frontier_key_buffer.data(), mg_new_frontier_key_buffer.size());
      } else {
        std::get<0>(mg_aggregate_new_frontier_key_buffer) =
          cugraph::test::device_gatherv(*handle_,
                                        std::get<0>(mg_new_frontier_key_buffer).data(),
                                        std::get<0>(mg_new_frontier_key_buffer).size());
        std::get<1>(mg_aggregate_new_frontier_key_buffer) =
          cugraph::test::device_gatherv(*handle_,
                                        std::get<1>(mg_new_frontier_key_buffer).data(),
                                        std::get<1>(mg_new_frontier_key_buffer).size());
      }

      [[maybe_unused]] auto mg_aggregate_payload_buffer =
        cugraph::detail::allocate_optional_dataframe_buffer<payload_t>(0, handle_->get_stream());
      if constexpr (!std::is_same_v<payload_t, void>) {
        if constexpr (std::is_arithmetic_v<payload_t>) {
          mg_aggregate_payload_buffer = cugraph::test::device_gatherv(
            *handle_, mg_payload_buffer.data(), mg_payload_buffer.size());
        } else {
          std::get<0>(mg_aggregate_payload_buffer) = cugraph::test::device_gatherv(
            *handle_, std::get<0>(mg_payload_buffer).data(), std::get<0>(mg_payload_buffer).size());
          std::get<1>(mg_aggregate_payload_buffer) = cugraph::test::device_gatherv(
            *handle_, std::get<1>(mg_payload_buffer).data(), std::get<1>(mg_payload_buffer).size());
        }
      }

      if (handle_->get_comms().get_rank() == int{0}) {
        if constexpr (std::is_same_v<key_t, vertex_t>) {
          cugraph::unrenumber_int_vertices<vertex_t, !is_multi_gpu>(
            *handle_,
            mg_aggregate_new_frontier_key_buffer.begin(),
            mg_aggregate_new_frontier_key_buffer.size(),
            mg_aggregate_renumber_map_labels.data(),
            std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
        } else {
          cugraph::unrenumber_int_vertices<vertex_t, !is_multi_gpu>(
            *handle_,
            std::get<0>(mg_aggregate_new_frontier_key_buffer).begin(),
            std::get<0>(mg_aggregate_new_frontier_key_buffer).size(),
            mg_aggregate_renumber_map_labels.data(),
            std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
        }

        if constexpr (std::is_same_v<payload_t, void>) {
          thrust::sort(handle_->get_thrust_policy(),
                       cugraph::get_dataframe_buffer_begin(mg_aggregate_new_frontier_key_buffer),
                       cugraph::get_dataframe_buffer_end(mg_aggregate_new_frontier_key_buffer));
        } else {
          thrust::sort_by_key(
            handle_->get_thrust_policy(),
            cugraph::get_dataframe_buffer_begin(mg_aggregate_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_end(mg_aggregate_new_frontier_key_buffer),
            cugraph::get_dataframe_buffer_begin(mg_aggregate_payload_buffer));
        }

        cugraph::graph_t<vertex_t, edge_t, store_transposed, !is_multi_gpu> sg_graph(*handle_);
        std::tie(sg_graph, std::ignore, std::ignore) = cugraph::test::
          construct_graph<vertex_t, edge_t, weight_t, store_transposed, !is_multi_gpu>(
            *handle_, input_usecase, false, false);

        auto sg_graph_view = sg_graph.view();

        auto sg_vertex_prop = cugraph::test::generate<vertex_t, property_t>::vertex_property(
          *handle_,
          thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_first()),
          thrust::make_counting_iterator(sg_graph_view.local_vertex_partition_range_last()),
          hash_bin_count);
        auto sg_src_prop = cugraph::test::generate<vertex_t, property_t>::src_property(
          *handle_, sg_graph_view, sg_vertex_prop);
        auto sg_dst_prop = cugraph::test::generate<vertex_t, property_t>::dst_property(
          *handle_, sg_graph_view, sg_vertex_prop);

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

        cugraph::vertex_frontier_t<vertex_t, tag_t, !is_multi_gpu, true> sg_vertex_frontier(
          *handle_, num_buckets);
        sg_vertex_frontier.bucket(bucket_idx_cur)
          .insert(cugraph::get_dataframe_buffer_begin(sg_key_buffer),
                  cugraph::get_dataframe_buffer_end(sg_key_buffer));

        auto sg_new_frontier_key_buffer =
          cugraph::allocate_dataframe_buffer<key_t>(0, handle_->get_stream());
        [[maybe_unused]] auto sg_payload_buffer =
          cugraph::detail::allocate_optional_dataframe_buffer<payload_t>(0, handle_->get_stream());
        if constexpr (std::is_same_v<payload_t, void>) {
          sg_new_frontier_key_buffer = cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
            *handle_,
            sg_graph_view,
            sg_vertex_frontier.bucket(bucket_idx_cur),
            sg_src_prop.view(),
            sg_dst_prop.view(),
            cugraph::edge_dummy_property_t{}.view(),
            e_op_t<key_t, vertex_t, property_t, payload_t>{},
            cugraph::reduce_op::null{});
        } else {
          std::tie(sg_new_frontier_key_buffer, sg_payload_buffer) =
            cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
              *handle_,
              sg_graph_view,
              sg_vertex_frontier.bucket(bucket_idx_cur),
              sg_src_prop.view(),
              sg_dst_prop.view(),
              cugraph::edge_dummy_property_t{}.view(),
              e_op_t<key_t, vertex_t, property_t, payload_t>{},
              cugraph::reduce_op::plus<payload_t>{});
        }

        if constexpr (std::is_same_v<payload_t, void>) {
          thrust::sort(handle_->get_thrust_policy(),
                       cugraph::get_dataframe_buffer_begin(sg_new_frontier_key_buffer),
                       cugraph::get_dataframe_buffer_end(sg_new_frontier_key_buffer));
        } else {
          thrust::sort_by_key(handle_->get_thrust_policy(),
                              cugraph::get_dataframe_buffer_begin(sg_new_frontier_key_buffer),
                              cugraph::get_dataframe_buffer_end(sg_new_frontier_key_buffer),
                              cugraph::get_dataframe_buffer_begin(sg_payload_buffer));
        }

        bool key_passed =
          thrust::equal(handle_->get_thrust_policy(),
                        cugraph::get_dataframe_buffer_begin(sg_new_frontier_key_buffer),
                        cugraph::get_dataframe_buffer_end(sg_new_frontier_key_buffer),
                        cugraph::get_dataframe_buffer_begin(mg_aggregate_new_frontier_key_buffer));
        ASSERT_TRUE(key_passed);

        if constexpr (!std::is_same_v<payload_t, void>) {
          bool payload_passed =
            thrust::equal(handle_->get_thrust_policy(),
                          cugraph::get_dataframe_buffer_begin(sg_payload_buffer),
                          cugraph::get_dataframe_buffer_begin(sg_payload_buffer),
                          cugraph::get_dataframe_buffer_end(mg_aggregate_payload_buffer));
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
  Tests_MGTransformReduceVFrontierOutgoingEByDst<input_usecase_t>::handle_ = nullptr;

using Tests_MGTransformReduceVFrontierOutgoingEByDst_File =
  Tests_MGTransformReduceVFrontierOutgoingEByDst<cugraph::test::File_Usecase>;
using Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat =
  Tests_MGTransformReduceVFrontierOutgoingEByDst<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File, CheckInt32Int32FloatVoidVoid)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, void>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat, CheckInt32Int32FloatVoidVoid)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, void>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File, CheckInt32Int32FloatVoidInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat, CheckInt32Int32FloatVoidInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File, CheckInt32Int32FloatVoidTupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, thrust::tuple<float, int32_t>>(
    std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat, CheckInt32Int32FloatVoidTupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, void, thrust::tuple<float, int32_t>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File, CheckInt32Int32FloatInt32Void)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, void>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat, CheckInt32Int32FloatInt32Void)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, void>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File, CheckInt32Int32FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, int32_t>(std::get<0>(param),
                                                              std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat, CheckInt32Int32FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File,
       CheckInt32Int32FloatInt32TupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, thrust::tuple<float, int32_t>>(
    std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat,
       CheckInt32Int32FloatInt32TupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t, thrust::tuple<float, int32_t>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File, CheckInt32Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int32_t, int32_t>(std::get<0>(param),
                                                              std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat, CheckInt32Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, int32_t, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_File, CheckInt64Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t, int32_t>(std::get<0>(param),
                                                              std::get<1>(param));
}

TEST_P(Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat, CheckInt64Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTransformReduceVFrontierOutgoingEByDst_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTransformReduceVFrontierOutgoingEByDst_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
