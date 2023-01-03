/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "nbr_sampling_utils.cuh"

#include <sampling/detail/graph_functions.hpp>

#include <utilities/mg_utilities.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>

#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MG_GatherEdges
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_GatherEdges() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    constexpr bool sort_adjacency_list = true;

    auto [mg_graph, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, true, true, false, sort_adjacency_list);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view                        = mg_graph.view();
    constexpr vertex_t repetitions_per_vertex = 5;
    constexpr vertex_t source_sample_count    = 3;

    // 2. Gather mnmg call
    // Generate random vertex ids in the range of current gpu

    auto [global_degree_offsets, global_out_degrees] =
      cugraph::detail::get_global_degree_information(*handle_, mg_graph_view);

    // Generate random sources to gather on
    auto random_sources =
      cugraph::test::random_vertex_ids(*handle_,
                                       mg_graph_view.local_vertex_partition_range_first(),
                                       mg_graph_view.local_vertex_partition_range_last(),
                                       std::min(mg_graph_view.local_vertex_partition_range_size() *
                                                  (repetitions_per_vertex + vertex_t{1}),
                                                source_sample_count),
                                       repetitions_per_vertex);

    // FIXME: allgather is probably a poor name for this function.
    //        It's really an allgather across the row communicator
    auto active_sources =
      cugraph::detail::allgather_active_majors(*handle_, std::move(random_sources));

    auto [src, dst, edge_ids] =
      cugraph::detail::gather_one_hop_edgelist(*handle_, mg_graph_view, active_sources);

    if (prims_usecase.check_correctness) {
      // Gather outputs to gpu 0
      auto mg_out_srcs = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{src.data(), src.size()});
      auto mg_out_dsts = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{dst.data(), dst.size()});

      // Gather relevant edges from graph
      auto& col_comm      = handle_->get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_rank = col_comm.get_rank();
      auto all_active_sources = cugraph::test::device_allgatherv(
        *handle_,
        raft::device_span<vertex_t const>{active_sources.data(),
                                          col_rank == 0 ? active_sources.size() : 0});

      thrust::sort(
        handle_->get_thrust_policy(), all_active_sources.begin(), all_active_sources.end());

      // Gather input graph edgelist
      rmm::device_uvector<vertex_t> sg_src(0, handle_->get_stream());
      rmm::device_uvector<vertex_t> sg_dst(0, handle_->get_stream());
      std::tie(sg_src, sg_dst, std::ignore) =
        mg_graph_view.decompress_to_edgelist(*handle_, std::nullopt);

      auto begin_iter = thrust::make_zip_iterator(sg_src.begin(), sg_dst.begin());
      auto new_end    = thrust::remove_if(
        handle_->get_thrust_policy(),
        begin_iter,
        begin_iter + sg_src.size(),
        [sources = all_active_sources.data(), size = all_active_sources.size()] __device__(auto t) {
          auto src = thrust::get<0>(t);
          return !thrust::binary_search(thrust::seq, sources, sources + size, src);
        });

      sg_src.resize(thrust::distance(begin_iter, new_end), handle_->get_stream());
      sg_dst.resize(thrust::distance(begin_iter, new_end), handle_->get_stream());

      auto aggregated_sg_src = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{sg_src.begin(), sg_src.size()});
      auto aggregated_sg_dst = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{sg_dst.begin(), sg_dst.size()});

      thrust::sort(handle_->get_thrust_policy(),
                   thrust::make_zip_iterator(mg_out_srcs.begin(), mg_out_dsts.begin()),
                   thrust::make_zip_iterator(mg_out_srcs.end(), mg_out_dsts.end()));

      thrust::sort(handle_->get_thrust_policy(),
                   thrust::make_zip_iterator(aggregated_sg_src.begin(), aggregated_sg_dst.begin()),
                   thrust::make_zip_iterator(aggregated_sg_src.end(), aggregated_sg_dst.end()));

      // FIXME:  This is ignoring the case of the same seed being specified multiple
      //         times.  Not sure that's worth worrying about, so taking the easy way out here.
      auto unique_end =
        thrust::unique(handle_->get_thrust_policy(),
                       thrust::make_zip_iterator(mg_out_srcs.begin(), mg_out_dsts.begin()),
                       thrust::make_zip_iterator(mg_out_srcs.end(), mg_out_dsts.end()));

      mg_out_srcs.resize(
        thrust::distance(thrust::make_zip_iterator(mg_out_srcs.begin(), mg_out_dsts.begin()),
                         unique_end),
        handle_->get_stream());
      mg_out_dsts.resize(
        thrust::distance(thrust::make_zip_iterator(mg_out_srcs.begin(), mg_out_dsts.begin()),
                         unique_end),
        handle_->get_stream());

      auto passed = thrust::equal(handle_->get_thrust_policy(),
                                  mg_out_srcs.begin(),
                                  mg_out_srcs.end(),
                                  aggregated_sg_src.begin());
      passed &= thrust::equal(handle_->get_thrust_policy(),
                              mg_out_dsts.begin(),
                              mg_out_dsts.end(),
                              aggregated_sg_dst.begin());
      ASSERT_TRUE(passed);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MG_GatherEdges<input_usecase_t>::handle_ = nullptr;

using Tests_MG_GatherEdges_File = Tests_MG_GatherEdges<cugraph::test::File_Usecase>;

using Tests_MG_GatherEdges_Rmat = Tests_MG_GatherEdges<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_GatherEdges_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_File, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_File, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_GatherEdges_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_GatherEdges_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_GatherEdges_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MG_GatherEdges_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
