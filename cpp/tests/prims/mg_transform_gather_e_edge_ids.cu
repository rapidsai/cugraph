/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_e.cuh"
#include "prims/transform_gather_e.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
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

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
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
    std::optional<cugraph::edge_property_t<edge_t, edge_t>> edge_ids;
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};

    std::vector<rmm::device_uvector<vertex_t>> src_chunks{};
    std::vector<rmm::device_uvector<vertex_t>> dst_chunks{};
    std::vector<rmm::device_uvector<vertex_t>> edge_id_chunks{};
    bool is_symmetric;

    std::tie(src_chunks, dst_chunks, std::ignore, std::ignore, is_symmetric) =
      input_usecase.template construct_edgelist<vertex_t, weight_t>(
        *handle_, false, store_transposed, true, true);

    edge_t num_input_edges = 0;
    for (size_t i = 0; i < src_chunks.size(); ++i) {
      num_input_edges += static_cast<edge_t>(src_chunks[i].size());
    }

    auto edge_counts =
      cugraph::host_scalar_allgather(handle_->get_comms(), num_input_edges, handle_->get_stream());
    std::exclusive_scan(edge_counts.begin(), edge_counts.end(), edge_counts.begin(), edge_t{0});

    edge_t base_edge_idx = edge_counts[handle_->get_comms().get_rank()];

    for (size_t i = 0; i < src_chunks.size(); ++i) {
      rmm::device_uvector<edge_t> tmp_ids(src_chunks[i].size(), handle_->get_stream());
      cugraph::detail::sequence_fill(
        handle_->get_stream(), tmp_ids.data(), tmp_ids.size(), base_edge_idx);
      base_edge_idx += src_chunks[i].size();
      edge_id_chunks.push_back(std::move(tmp_ids));
    }

    std::tie(
      mg_graph, std::ignore, edge_ids, std::ignore, std::ignore, std::ignore, mg_renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t,
                                          edge_t,
                                          weight_t,
                                          int32_t,
                                          int32_t,
                                          store_transposed,
                                          true>(*handle_,
                                                std::nullopt,
                                                std::move(src_chunks),
                                                std::move(dst_chunks),
                                                std::nullopt,
                                                std::make_optional(std::move(edge_id_chunks)),
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                cugraph::graph_properties_t{is_symmetric, true},
                                                true,
                                                std::nullopt,
                                                std::nullopt);

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

    // 2. run MG transform_gather_e to identify extracted edges

    rmm::device_uvector<vertex_t> srcs(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle_->get_stream());
    std::optional<rmm::device_uvector<edge_t>> multi_edge_indices{std::nullopt};
    rmm::device_uvector<edge_t> should_be_gathered_ids(0, handle_->get_stream());

    if (mg_graph_view.is_multigraph()) {
      std::cout << "calling multigraph extract_transform_if_e" << std::endl;
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(*handle_,
                                                                                mg_graph_view);
      std::tie(srcs, dsts, multi_edge_indices) = cugraph::extract_transform_if_e(
        *handle_,
        mg_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_multi_indices.view(),
        cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t, edge_t>>(
          [] __device__(auto src, auto dst, auto, auto, edge_t index) {
            if (src == 0)  //((src == 0) && (dst == 480))
              printf("multigraph, edge (%d, %d, %d)\n", (int)src, (int)dst, (int)index);
            return cuda::std::make_tuple(src, dst, index);
          }),
        cuda::proclaim_return_type<bool>(
          [] __device__(auto src, auto dst, auto, auto, auto) { return ((src + dst) % 20) == 0; }));

      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      std::cout << "back from multigraph extract_transform_if_e" << std::endl;
    }

    std::cout << "calling edge id extract_transform_if_e" << std::endl;

    std::tie(srcs, dsts, should_be_gathered_ids) = cugraph::extract_transform_if_e(
      *handle_,
      mg_graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_ids->view(),
      cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t, edge_t>>(
        [] __device__(auto src, auto dst, auto, auto, auto id) {
          if (src == 0)  //((src == 0) && (dst == 480))
            printf("edge id, edge (%d, %d, %d)\n", (int)src, (int)dst, (int)id);

          return cuda::std::make_tuple(src, dst, id);
        }),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto src, auto dst, auto, auto, auto) { return ((src + dst) % 20) == 0; }));

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
    std::cout << "back from edge id extract_transform_if_e" << std::endl;

    if (mg_graph_view.is_multigraph()) {
      auto tuple_first = thrust::make_zip_iterator(store_transposed ? dsts.begin() : srcs.begin(),
                                                   store_transposed ? srcs.begin() : dsts.begin(),
                                                   multi_edge_indices->begin(),
                                                   should_be_gathered_ids.begin());
      if (prims_usecase.use_sorted_unique_edgelist) {
        thrust::sort(handle_->get_thrust_policy(), tuple_first, tuple_first + srcs.size());
      } else {
        thrust::shuffle(handle_->get_thrust_policy(),
                        tuple_first,
                        tuple_first + srcs.size(),
                        thrust::default_random_engine());
      }
    } else {
      auto tuple_first = thrust::make_zip_iterator(store_transposed ? dsts.begin() : srcs.begin(),
                                                   store_transposed ? srcs.begin() : dsts.begin(),
                                                   should_be_gathered_ids.begin());
      if (prims_usecase.use_sorted_unique_edgelist) {
        thrust::sort(handle_->get_thrust_policy(), tuple_first, tuple_first + srcs.size());
      } else {
        thrust::shuffle(handle_->get_thrust_policy(),
                        tuple_first,
                        tuple_first + srcs.size(),
                        thrust::default_random_engine());
      }
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transform_gather_e");
    }

    rmm::device_uvector<edge_t> gathered_ids(srcs.size(), handle_->get_stream());

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

      cugraph::transform_gather_e(*handle_,
                                  mg_graph_view,
                                  edge_list,
                                  cugraph::edge_src_dummy_property_t{}.view(),
                                  cugraph::edge_dst_dummy_property_t{}.view(),
                                  edge_ids->view(),
                                  cuda::proclaim_return_type<edge_t>(
                                    [] __device__(auto, auto, auto, auto, auto id) { return id; }),
                                  gathered_ids.begin());
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

      cugraph::transform_gather_e(*handle_,
                                  mg_graph_view,
                                  edge_list,
                                  cugraph::edge_src_dummy_property_t{}.view(),
                                  cugraph::edge_dst_dummy_property_t{}.view(),
                                  edge_ids->view(),
                                  cuda::proclaim_return_type<edge_t>(
                                    [] __device__(auto, auto, auto, auto, auto id) { return id; }),
                                  gathered_ids.begin());
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. validate MG results

    auto h_srcs                   = cugraph::test::to_host(*handle_, srcs);
    auto h_dsts                   = cugraph::test::to_host(*handle_, dsts);
    auto h_gathered_ids           = cugraph::test::to_host(*handle_, gathered_ids);
    auto h_should_be_gathered_ids = cugraph::test::to_host(*handle_, should_be_gathered_ids);

    std::cout << "Validation... sorted = "
              << (prims_usecase.use_sorted_unique_edgelist ? "TRUE" : "FALSE")
              << ", multigraph = " << (mg_graph_view.is_multigraph() ? "TRUE" : "FALSE")
              << std::endl;

    for (size_t i = 0; i < h_srcs.size(); ++i) {
      if (h_gathered_ids[i] != h_should_be_gathered_ids[i]) {
        std::cout << "  (" << h_srcs[i] << ", " << h_dsts[i] << ") got " << h_gathered_ids[i]
                  << ", expected " << h_should_be_gathered_ids[i] << std::endl;
      }
    }

    if (prims_usecase.check_correctness) {
      ASSERT_TRUE(thrust::equal(handle_->get_thrust_policy(),
                                gathered_ids.begin(),
                                gathered_ids.end(),
                                should_be_gathered_ids.begin()));
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTransformGatherE<input_usecase_t>::handle_ = nullptr;

using Tests_MGTransformGatherE_File = Tests_MGTransformGatherE<cugraph::test::File_Usecase>;
using Tests_MGTransformGatherE_Rmat = Tests_MGTransformGatherE<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatIntTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_File, CheckInt32Int32FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt32Int32FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param),
    cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransformGatherE_Rmat, CheckInt64Int64FloatIntTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, true>(
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
