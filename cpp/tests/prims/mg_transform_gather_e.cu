/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_e.cuh"
#include "prims/transform_gather_e.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <cuco/hash_functions.cuh>

#include <gtest/gtest.h>

#include <random>

struct Prims_Usecase {
  bool use_sorted_unique_edgelist{false};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTransformGatherE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGTransformGatherE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename result_t,
            bool store_transposed>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, false, true);

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

    // 2. run MG transform_gather_e

    const int hash_bin_count = 5;

    auto mg_vertex_prop =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::vertex_property(
        *handle_, *mg_renumber_map, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<decltype(mg_graph_view), result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<decltype(mg_graph_view), result_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    rmm::device_uvector<vertex_t> srcs(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle_->get_stream());
    std::optional<rmm::device_uvector<edge_t>> multi_edge_indices{std::nullopt};
    auto should_be_gathered_values =
      cugraph::allocate_dataframe_buffer<result_t>(0, handle_->get_stream());
    {
      if (mg_graph_view.is_multigraph()) {
        cugraph::edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(*handle_,
                                                                                  mg_graph_view);
        auto ret = cugraph::extract_transform_if_e(
          *handle_,
          mg_graph_view,
          mg_src_prop.view(),
          mg_dst_prop.view(),
          edge_multi_indices.view(),
          cuda::proclaim_return_type<decltype(cugraph::thrust_tuple_cat(
            cuda::std::tuple<vertex_t, vertex_t, edge_t>{}, cugraph::to_thrust_tuple(result_t{})))>(
            [] __device__(
              auto src, auto dst, auto src_property, auto dst_property, auto multi_edge_index) {
              if (src_property < dst_property) {
                return cugraph::thrust_tuple_cat(cuda::std::make_tuple(src, dst, multi_edge_index),
                                                 cugraph::to_thrust_tuple(src_property));
              } else {
                return cugraph::thrust_tuple_cat(cuda::std::make_tuple(src, dst, multi_edge_index),
                                                 cugraph::to_thrust_tuple(dst_property));
              }
            }),
          cuda::proclaim_return_type<bool>(
            [] __device__(auto src, auto dst, auto, auto, auto multi_edge_index) {
              return ((src + dst) % 2) == 0;
            }));
        srcs               = std::move(std::get<0>(ret));
        dsts               = std::move(std::get<1>(ret));
        multi_edge_indices = std::move(std::get<2>(ret));
        if constexpr (std::is_arithmetic_v<result_t>) {
          should_be_gathered_values = std::move(std::get<3>(ret));
        } else {
          static_assert(std::tuple_size_v<result_t> == 2);
          should_be_gathered_values =
            std::make_tuple(std::move(std::get<3>(ret)), std::move(std::get<4>(ret)));
        }
        auto tuple_first =
          thrust::make_zip_iterator(store_transposed ? dsts.begin() : srcs.begin(),
                                    store_transposed ? srcs.begin() : dsts.begin(),
                                    multi_edge_indices->begin(),
                                    cugraph::get_dataframe_buffer_begin(should_be_gathered_values));
        if (prims_usecase.use_sorted_unique_edgelist) {
          thrust::sort(handle_->get_thrust_policy(), tuple_first, tuple_first + srcs.size());
        } else {
          thrust::shuffle(handle_->get_thrust_policy(),
                          tuple_first,
                          tuple_first + srcs.size(),
                          thrust::default_random_engine());
        }
      } else {
        auto ret = cugraph::extract_transform_if_e(
          *handle_,
          mg_graph_view,
          mg_src_prop.view(),
          mg_dst_prop.view(),
          cugraph::edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<decltype(cugraph::thrust_tuple_cat(
            cuda::std::tuple<vertex_t, vertex_t>{}, cugraph::to_thrust_tuple(result_t{})))>(
            [] __device__(
              auto src, auto dst, auto src_property, auto dst_property, cuda::std::nullopt_t) {
              if (src_property < dst_property) {
                return cugraph::thrust_tuple_cat(cuda::std::make_tuple(src, dst),
                                                 cugraph::to_thrust_tuple(src_property));
              } else {
                return cugraph::thrust_tuple_cat(cuda::std::make_tuple(src, dst),
                                                 cugraph::to_thrust_tuple(dst_property));
              }
            }),
          cuda::proclaim_return_type<bool>([] __device__(auto src, auto dst, auto, auto, auto) {
            return ((src + dst) % 2) == 0;
          }));
        srcs = std::move(std::get<0>(ret));
        dsts = std::move(std::get<1>(ret));
        if constexpr (std::is_arithmetic_v<result_t>) {
          should_be_gathered_values = std::move(std::get<2>(ret));
        } else {
          static_assert(std::tuple_size_v<result_t> == 2);
          should_be_gathered_values =
            std::make_tuple(std::move(std::get<2>(ret)), std::move(std::get<3>(ret)));
        }
        auto tuple_first =
          thrust::make_zip_iterator(store_transposed ? dsts.begin() : srcs.begin(),
                                    store_transposed ? srcs.begin() : dsts.begin(),
                                    cugraph::get_dataframe_buffer_begin(should_be_gathered_values));
        if (prims_usecase.use_sorted_unique_edgelist) {
          thrust::sort(handle_->get_thrust_policy(), tuple_first, tuple_first + srcs.size());
        } else {
          thrust::shuffle(handle_->get_thrust_policy(),
                          tuple_first,
                          tuple_first + srcs.size(),
                          thrust::default_random_engine());
        }
      }
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transform_gather_e");
    }

    auto gathered_values =
      cugraph::allocate_dataframe_buffer<result_t>(srcs.size(), handle_->get_stream());
    if (prims_usecase.use_sorted_unique_edgelist) {
      cugraph::edge_bucket_t<vertex_t,
                             edge_t,
                             !store_transposed /* src_major */,
                             true,
                             true /* sorted_unique */>
        edge_list(*handle_, mg_graph_view.is_multigraph());

      edge_list.insert(
        srcs.begin(),
        srcs.end(),
        dsts.begin(),
        multi_edge_indices ? std::make_optional(multi_edge_indices->begin()) : std::nullopt);

      cugraph::transform_gather_e(
        *handle_,
        mg_graph_view,
        edge_list,
        mg_src_prop.view(),
        mg_dst_prop.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<result_t>(
          [] __device__(
            auto src, auto dst, auto src_property, auto dst_property, cuda::std::nullopt_t) {
            if (src_property < dst_property) {
              return src_property;
            } else {
              return dst_property;
            }
          }),
        cugraph::get_dataframe_buffer_begin(gathered_values));
    } else {
      cugraph::edge_bucket_t<vertex_t,
                             edge_t,
                             !store_transposed /* src_major */,
                             true,
                             false /* sorted_unique */>
        edge_list(*handle_, mg_graph_view.is_multigraph());

      edge_list.insert(
        srcs.begin(),
        srcs.end(),
        dsts.begin(),
        multi_edge_indices ? std::make_optional(multi_edge_indices->begin()) : std::nullopt);

      cugraph::transform_gather_e(
        *handle_,
        mg_graph_view,
        edge_list,
        mg_src_prop.view(),
        mg_dst_prop.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<result_t>(
          [] __device__(
            auto src, auto dst, auto src_property, auto dst_property, cuda::std::nullopt_t) {
            if (src_property < dst_property) {
              return src_property;
            } else {
              return dst_property;
            }
          }),
        cugraph::get_dataframe_buffer_begin(gathered_values));
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. validate MG results

    if (prims_usecase.check_correctness) {
      ASSERT_TRUE(thrust::equal(handle_->get_thrust_policy(),
                                cugraph::get_dataframe_buffer_begin(gathered_values),
                                cugraph::get_dataframe_buffer_end(gathered_values),
                                cugraph::get_dataframe_buffer_begin(should_be_gathered_values)));
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTransformGatherE<input_usecase_t>::handle_ = nullptr;

using Tests_MGTransformGatherE_File = Tests_MGTransformGatherE<cugraph::test::File_Usecase>;
using Tests_MGTransformGatherE_Rmat = Tests_MGTransformGatherE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<int, float>, false>(
    std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, cuda::std::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<int, float>, true>(std::get<0>(param),
                                                                                std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, cuda::std::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatBoolTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatBoolTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatBoolTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, bool, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatBoolTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatBoolTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatBoolTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, bool, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTransformGatherE_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, false, true},
                                       Prims_Usecase{false, true, true},
                                       Prims_Usecase{true, false, true},
                                       Prims_Usecase{true, true, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGTransformGatherE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, true},
                      Prims_Usecase{false, true, true},
                      Prims_Usecase{true, false, true},
                      Prims_Usecase{true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGTransformGatherE_Rmat,
                         ::testing::Combine(::testing::Values(Prims_Usecase{false, false, true},
                                                              Prims_Usecase{false, true, true},
                                                              Prims_Usecase{true, false, true},
                                                              Prims_Usecase{true, true, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTransformGatherE_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, false},
                      Prims_Usecase{false, true, false},
                      Prims_Usecase{true, false, false},
                      Prims_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
