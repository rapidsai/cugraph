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

#include "egonet_validate.hpp"

#include <structure/detail/structure_utils.cuh>

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <gtest/gtest.h>

struct Egonet_Usecase {
  std::vector<int32_t> ego_sources_{};
  int32_t radius_{1};
  bool test_weighted_{false};
  bool check_correctness_{false};
};

template <typename input_usecase_t>
class Tests_MGEgonet
  : public ::testing::TestWithParam<std::tuple<Egonet_Usecase, input_usecase_t>> {
 public:
  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(128); }
  static void TearDownTestCase() { handle_.reset(); }

  // Run once for each test instance
  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Egonet_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [egonet_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, egonet_usecase.test_weighted_, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    int my_rank = handle_->get_comms().get_rank();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Egonet");
    }

    rmm::device_uvector<vertex_t> d_ego_sources(0, handle_->get_stream());

    if (my_rank == 0) {
      d_ego_sources.resize(egonet_usecase.ego_sources_.size(), handle_->get_stream());

      if constexpr (std::is_same<int32_t, vertex_t>::value) {
        raft::update_device(d_ego_sources.data(),
                            egonet_usecase.ego_sources_.data(),
                            egonet_usecase.ego_sources_.size(),
                            handle_->get_stream());
      } else {
        std::vector<vertex_t> h_ego_sources(d_ego_sources.size());
        std::transform(egonet_usecase.ego_sources_.begin(),
                       egonet_usecase.ego_sources_.end(),
                       h_ego_sources.begin(),
                       [](auto v) { return static_cast<vertex_t>(v); });
        raft::update_device(
          d_ego_sources.data(), h_ego_sources.data(), h_ego_sources.size(), handle_->get_stream());
      }
    }

    cugraph::renumber_ext_vertices<vertex_t, true>(
      *handle_,
      d_ego_sources.data(),
      d_ego_sources.size(),
      d_renumber_map_labels->data(),
      mg_graph_view.local_vertex_partition_range_first(),
      mg_graph_view.local_vertex_partition_range_last());

    d_ego_sources = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
      *handle_, std::move(d_ego_sources), mg_graph_view.vertex_partition_range_lasts());

    auto [d_ego_edgelist_src, d_ego_edgelist_dst, d_ego_edgelist_wgt, d_ego_edgelist_offsets] =
      cugraph::extract_ego(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        raft::device_span<vertex_t const>{d_ego_sources.data(), d_ego_sources.size()},
        static_cast<vertex_t>(egonet_usecase.radius_));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (egonet_usecase.check_correctness_) {
      d_ego_edgelist_src = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(d_ego_edgelist_src.data(), d_ego_edgelist_src.size()));
      d_ego_edgelist_dst = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(d_ego_edgelist_dst.data(), d_ego_edgelist_dst.size()));

      if (d_ego_edgelist_wgt) {
        *d_ego_edgelist_wgt =
          cugraph::test::device_gatherv(*handle_,
                                        raft::device_span<weight_t const>(
                                          d_ego_edgelist_wgt->data(), d_ego_edgelist_wgt->size()));
      }

      size_t offsets_size = d_ego_edgelist_offsets.size();

      auto graph_ids_v = cugraph::detail::expand_sparse_offsets(
        raft::device_span<size_t const>(d_ego_edgelist_offsets.data(),
                                        d_ego_edgelist_offsets.size()),
        vertex_t{0},
        handle_->get_stream());

      graph_ids_v = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(graph_ids_v.data(), graph_ids_v.size()));

      if (d_ego_edgelist_wgt) {
        thrust::sort_by_key(
          handle_->get_thrust_policy(),
          thrust::make_zip_iterator(
            graph_ids_v.begin(), d_ego_edgelist_src.begin(), d_ego_edgelist_dst.begin()),
          thrust::make_zip_iterator(
            graph_ids_v.end(), d_ego_edgelist_src.end(), d_ego_edgelist_dst.end()),
          d_ego_edgelist_wgt->begin());
      } else {
        thrust::sort(handle_->get_thrust_policy(),
                     thrust::make_zip_iterator(
                       graph_ids_v.begin(), d_ego_edgelist_src.begin(), d_ego_edgelist_dst.begin()),
                     thrust::make_zip_iterator(
                       graph_ids_v.end(), d_ego_edgelist_src.end(), d_ego_edgelist_dst.end()));
      }

      d_ego_edgelist_offsets = cugraph::detail::compute_sparse_offsets<size_t>(
        graph_ids_v.begin(), graph_ids_v.end(), size_t{0}, offsets_size - 1, handle_->get_stream());

      auto [sg_graph, sg_edge_weights, sg_number_map] = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<rmm::device_uvector<vertex_t>>{std::nullopt},
        false);

      d_ego_sources = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(d_ego_sources.data(), d_ego_sources.size()));

      if (my_rank == 0) {
        auto [d_reference_src, d_reference_dst, d_reference_wgt, d_reference_offsets] =
          cugraph::extract_ego(
            *handle_,
            sg_graph.view(),
            sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt,
            raft::device_span<vertex_t const>{d_ego_sources.data(), d_ego_sources.size()},
            static_cast<vertex_t>(egonet_usecase.radius_));

        cugraph::test::egonet_validate(*handle_,
                                       d_ego_edgelist_src,
                                       d_ego_edgelist_dst,
                                       d_ego_edgelist_wgt,
                                       d_ego_edgelist_offsets,
                                       d_reference_src,
                                       d_reference_dst,
                                       d_reference_wgt,
                                       d_reference_offsets);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGEgonet<input_usecase_t>::handle_ = nullptr;

using Tests_MGEgonet_File   = Tests_MGEgonet<cugraph::test::File_Usecase>;
using Tests_MGEgonet_File64 = Tests_MGEgonet<cugraph::test::File_Usecase>;
using Tests_MGEgonet_Rmat   = Tests_MGEgonet<cugraph::test::Rmat_Usecase>;
using Tests_MGEgonet_Rmat64 = Tests_MGEgonet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGEgonet_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEgonet_File64, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEgonet_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEgonet_Rmat64, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_file_test,
  Tests_MGEgonet_File,
  ::testing::Combine(
    // enable correctness checks for small graphs
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{0}, 1, false, true},
                      Egonet_Usecase{std::vector<int32_t>{0}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{0}, 3, false, true},
                      Egonet_Usecase{std::vector<int32_t>{10, 0, 5}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{9, 3, 10}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  simple_rmat_test,
  Tests_MGEgonet_Rmat,
  ::testing::Combine(
    // enable correctness checks for small graphs
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{0}, 1, false, true},
                      Egonet_Usecase{std::vector<int32_t>{0}, 2, false, true},
                      Egonet_Usecase{std::vector<int32_t>{0}, 3, false, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MGEgonet_File,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file64_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MGEgonet_File64,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGEgonet_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat64_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGEgonet_Rmat64,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Egonet_Usecase{std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
