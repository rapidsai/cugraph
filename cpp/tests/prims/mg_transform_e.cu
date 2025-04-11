/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "prims/count_if_e.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/transform_e.cuh"
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
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cuco/hash_functions.cuh>

#include <gtest/gtest.h>

#include <random>

struct Prims_Usecase {
  bool use_edgelist{false};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTransformE
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MGTransformE() {}

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

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (prims_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. run MG transform_e

    const int hash_bin_count = 5;
    const int initial_value  = 4;

    auto property_initial_value =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::initial_value(initial_value);
    auto mg_vertex_prop =
      cugraph::test::generate<decltype(mg_graph_view), result_t>::vertex_property(
        *handle_, *mg_renumber_map, hash_bin_count);
    auto mg_src_prop = cugraph::test::generate<decltype(mg_graph_view), result_t>::src_property(
      *handle_, mg_graph_view, mg_vertex_prop);
    auto mg_dst_prop = cugraph::test::generate<decltype(mg_graph_view), result_t>::dst_property(
      *handle_, mg_graph_view, mg_vertex_prop);

    cugraph::edge_bucket_t<vertex_t, void, !store_transposed /* src_major */, true, true> edge_list(
      *handle_);
    if (prims_usecase.use_edgelist) {
      rmm::device_uvector<vertex_t> srcs(0, handle_->get_stream());
      rmm::device_uvector<vertex_t> dsts(0, handle_->get_stream());
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) = cugraph::decompress_to_edgelist(
        *handle_,
        mg_graph_view,
        std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>{std::nullopt});
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(store_transposed ? dsts.begin() : srcs.begin(),
                           store_transposed ? srcs.begin() : dsts.begin()));
      srcs.resize(cuda::std::distance(
                    edge_first,
                    thrust::remove_if(handle_->get_thrust_policy(),
                                      edge_first,
                                      edge_first + srcs.size(),
                                      [] __device__(thrust::tuple<vertex_t, vertex_t> e) {
                                        return ((thrust::get<0>(e) + thrust::get<1>(e)) % 2) != 0;
                                      })),
                  handle_->get_stream());
      dsts.resize(srcs.size(), handle_->get_stream());
      edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(store_transposed ? dsts.begin() : srcs.begin(),
                           store_transposed ? srcs.begin() : dsts.begin()));
      thrust::sort(handle_->get_thrust_policy(), edge_first, edge_first + srcs.size());

      edge_list.insert(srcs.begin(),
                       srcs.end(),
                       dsts.begin());  // now edge_list stores edge pairs with (src + dst) % 2 == 0
    }

    cugraph::edge_property_t<decltype(mg_graph_view), result_t> edge_value_output(*handle_,
                                                                                  mg_graph_view);

    cugraph::fill_edge_property(
      *handle_, mg_graph_view, edge_value_output.mutable_view(), property_initial_value);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transform_e");
    }

    if (prims_usecase.use_edgelist) {
      cugraph::transform_e(
        *handle_,
        mg_graph_view,
        edge_list,
        mg_src_prop.view(),
        mg_dst_prop.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(
          auto src, auto dst, auto src_property, auto dst_property, cuda::std::nullopt_t) {
          if (src_property < dst_property) {
            return src_property;
          } else {
            return dst_property;
          }
        },
        edge_value_output.mutable_view());
    } else {
      cugraph::transform_e(
        *handle_,
        mg_graph_view,
        mg_src_prop.view(),
        mg_dst_prop.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(
          auto src, auto dst, auto src_property, auto dst_property, cuda::std::nullopt_t) {
          if (src_property < dst_property) {
            return src_property;
          } else {
            return dst_property;
          }
        },
        edge_value_output.mutable_view());
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. validate MG results

    if (prims_usecase.check_correctness) {
      size_t num_invalids{};
      if (prims_usecase.use_edgelist) {
        num_invalids = cugraph::count_if_e(
          *handle_,
          mg_graph_view,
          mg_src_prop.view(),
          mg_dst_prop.view(),
          edge_value_output.view(),
          [property_initial_value] __device__(
            auto src, auto dst, auto src_property, auto dst_property, auto edge_property) {
            if (((src + dst) % 2) == 0) {
              if (src_property < dst_property) {
                return edge_property != src_property;
              } else {
                return edge_property != dst_property;
              }
            } else {
              return edge_property != property_initial_value;
            }
          });
      } else {
        num_invalids = cugraph::count_if_e(
          *handle_,
          mg_graph_view,
          mg_src_prop.view(),
          mg_dst_prop.view(),
          edge_value_output.view(),
          [property_initial_value] __device__(
            auto src, auto dst, auto src_property, auto dst_property, auto edge_property) {
            if (src_property < dst_property) {
              return edge_property != src_property;
            } else {
              return edge_property != dst_property;
            }
          });
      }

      ASSERT_TRUE(num_invalids == 0);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTransformE<input_usecase_t>::handle_ = nullptr;

using Tests_MGTransformE_File = Tests_MGTransformE<cugraph::test::File_Usecase>;
using Tests_MGTransformE_Rmat = Tests_MGTransformE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTransformE_File, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(std::get<0>(param),
                                                                              std::get<1>(param));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt64Int64FloatTupleIntFloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_File, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(std::get<0>(param),
                                                                             std::get<1>(param));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt32Int32FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt64Int64FloatTupleIntFloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, thrust::tuple<int, float>, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_File, CheckInt32Int32FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt32Int32FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt64Int64FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_File, CheckInt32Int32FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt32Int32FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt64Int64FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, int, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_File, CheckInt32Int32FloatBoolTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt32Int32FloatBoolTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt64Int64FloatBoolTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, bool, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_File, CheckInt32Int32FloatBoolTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt32Int32FloatBoolTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, bool, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformE_Rmat, CheckInt64Int64FloatBoolTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, bool, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTransformE_File,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, false, true},
                                       Prims_Usecase{false, true, true},
                                       Prims_Usecase{true, false, true},
                                       Prims_Usecase{true, true, true}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGTransformE_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, true},
                      Prims_Usecase{false, true, true},
                      Prims_Usecase{true, false, true},
                      Prims_Usecase{true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGTransformE_Rmat,
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
  Tests_MGTransformE_Rmat,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{false, false, false},
                      Prims_Usecase{false, true, false},
                      Prims_Usecase{true, false, false},
                      Prims_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
