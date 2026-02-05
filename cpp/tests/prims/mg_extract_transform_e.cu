
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "prims/extract_transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
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
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuco/hash_functions.cuh>

#include <gtest/gtest.h>

#include <random>
#include <sstream>
#include <type_traits>

template <typename vertex_t, typename output_payload_t>
struct e_op_t {
  static_assert(std::is_same_v<output_payload_t, int32_t> ||
                std::is_same_v<output_payload_t, cuda::std::tuple<float, int32_t>>);

  using return_type = std::conditional_t<std::is_arithmetic_v<output_payload_t>,
                                         cuda::std::tuple<vertex_t, vertex_t, int32_t>,
                                         cuda::std::tuple<vertex_t, vertex_t, float, int32_t>>;

  __device__ return_type operator()(vertex_t src,
                                    vertex_t dst,
                                    cuda::std::nullopt_t,
                                    cuda::std::nullopt_t,
                                    cuda::std::nullopt_t) const
  {
    auto output_payload = static_cast<output_payload_t>(1);
    if constexpr (std::is_arithmetic_v<output_payload_t>) {
      return cuda::std::make_tuple(src, dst, output_payload);
    } else {
      static_assert(cuda::std::tuple_size<output_payload_t>::value == size_t{2});
      return cuda::std::make_tuple(
        src, dst, cuda::std::get<0>(output_payload), cuda::std::get<1>(output_payload));
    }
  }
};

struct Prims_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGExtractTransformE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGExtractTransformE() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of extract_transform_e primitive
  template <typename vertex_t, typename edge_t, typename weight_t, typename output_payload_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;
    using result_t    = int32_t;

    static_assert(std::is_same_v<output_payload_t, void> ||
                  cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<output_payload_t>::value);
    if constexpr (cugraph::is_thrust_tuple<output_payload_t>::value) {
      static_assert(cuda::std::tuple_size<output_payload_t>::value == size_t{2});
    }

    HighResTimer hr_timer{};

    // 1. create MG graph

    constexpr bool is_multi_gpu     = true;
    constexpr bool renumber         = true;   // needs to be true for multi gpu case
    constexpr bool store_transposed = false;  // needs to be false for using extract_transform_e
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

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (prims_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG extract_transform_e

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG extract_transform_e");
    }

    auto mg_extract_transform_output_buffer =
      cugraph::extract_transform_e(*handle_,
                                   mg_graph_view,
                                   cugraph::edge_src_dummy_property_t{}.view(),
                                   cugraph::edge_dst_dummy_property_t{}.view(),
                                   cugraph::edge_dummy_property_t{}.view(),
                                   e_op_t<vertex_t, output_payload_t>{});

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (prims_usecase.check_correctness) {
      auto mg_aggregate_extract_transform_output_buffer = cugraph::allocate_dataframe_buffer<
        typename e_op_t<vertex_t, output_payload_t>::return_type>(size_t{0}, handle_->get_stream());
      std::get<0>(mg_aggregate_extract_transform_output_buffer) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<0>(mg_extract_transform_output_buffer).data(),
                                      std::get<0>(mg_extract_transform_output_buffer).size());
      std::get<1>(mg_aggregate_extract_transform_output_buffer) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<1>(mg_extract_transform_output_buffer).data(),
                                      std::get<1>(mg_extract_transform_output_buffer).size());
      std::get<2>(mg_aggregate_extract_transform_output_buffer) =
        cugraph::test::device_gatherv(*handle_,
                                      std::get<2>(mg_extract_transform_output_buffer).data(),
                                      std::get<2>(mg_extract_transform_output_buffer).size());
      if constexpr (!std::is_arithmetic_v<output_payload_t>) {
        std::get<3>(mg_aggregate_extract_transform_output_buffer) =
          cugraph::test::device_gatherv(*handle_,
                                        std::get<3>(mg_extract_transform_output_buffer).data(),
                                        std::get<3>(mg_extract_transform_output_buffer).size());
      }

      cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*d_mg_renumber_map_labels).data(),
                                                                (*d_mg_renumber_map_labels).size()),
          false);

      if (handle_->get_comms().get_rank() == int{0}) {
        thrust::sort(
          handle_->get_thrust_policy(),
          cugraph::get_dataframe_buffer_begin(mg_aggregate_extract_transform_output_buffer),
          cugraph::get_dataframe_buffer_end(mg_aggregate_extract_transform_output_buffer));

        auto sg_graph_view = sg_graph.view();

        auto sg_extract_transform_output_buffer =
          cugraph::extract_transform_e(*handle_,
                                       sg_graph_view,
                                       cugraph::edge_src_dummy_property_t{}.view(),
                                       cugraph::edge_dst_dummy_property_t{}.view(),
                                       cugraph::edge_dummy_property_t{}.view(),
                                       e_op_t<vertex_t, output_payload_t>{});

        thrust::sort(handle_->get_thrust_policy(),
                     cugraph::get_dataframe_buffer_begin(sg_extract_transform_output_buffer),
                     cugraph::get_dataframe_buffer_end(sg_extract_transform_output_buffer));

        bool e_op_result_passed = thrust::equal(
          handle_->get_thrust_policy(),
          cugraph::get_dataframe_buffer_begin(sg_extract_transform_output_buffer),
          cugraph::get_dataframe_buffer_begin(sg_extract_transform_output_buffer),
          cugraph::get_dataframe_buffer_end(mg_aggregate_extract_transform_output_buffer));
        ASSERT_TRUE(e_op_result_passed);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGExtractTransformE<input_usecase_t>::handle_ = nullptr;

using Tests_MGExtractTransformE_File = Tests_MGExtractTransformE<cugraph::test::File_Usecase>;
using Tests_MGExtractTransformE_Rmat = Tests_MGExtractTransformE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGExtractTransformE_File, CheckInt32Int32FloatVoidInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGExtractTransformE_Rmat, CheckInt32Int32FloatVoidInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractTransformE_File, CheckInt32Int32FloatVoidTupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<float, int32_t>>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MGExtractTransformE_Rmat, CheckInt32Int32FloatVoidTupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<float, int32_t>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractTransformE_File, CheckInt32Int32FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGExtractTransformE_Rmat, CheckInt32Int32FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractTransformE_File, CheckInt32Int32FloatInt32TupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<float, int32_t>>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MGExtractTransformE_Rmat, CheckInt32Int32FloatInt32TupleFloatInt32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, cuda::std::tuple<float, int32_t>>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractTransformE_File, CheckInt64Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGExtractTransformE_Rmat, CheckInt64Int64FloatInt32Int32)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int32_t>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGExtractTransformE_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}, Prims_Usecase{true, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGExtractTransformE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, true}, Prims_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGExtractTransformE_Rmat,
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
  Tests_MGExtractTransformE_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false}, Prims_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
