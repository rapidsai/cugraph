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
#include <sampling/detail/sampling_utils_impl.cuh>

#include <utilities/mg_utilities.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>

#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

struct Prims_Usecase {
  bool check_correctness{true};
};

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<vertex_t>, std::vector<vertex_t>> test_gather_local_edges(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, true> const& mg_graph_view,
  rmm::device_uvector<vertex_t> const& sources,
  rmm::device_uvector<edge_t> const& destination_offsets,
  edge_t indices_per_source)
{
  auto& col_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_rank = col_comm.get_rank();

  // logic relies on gather_one_hop not having duplicates
  rmm::device_uvector<vertex_t> sources_copy(sources.size(), handle.get_stream());
  raft::copy(sources_copy.data(), sources.data(), sources.size(), handle.get_stream());
  thrust::sort(handle.get_thrust_policy(), sources_copy.begin(), sources_copy.end());
  auto sources_copy_end =
    thrust::unique(handle.get_thrust_policy(), sources_copy.begin(), sources_copy.end());
  sources_copy.resize(thrust::distance(sources_copy.begin(), sources_copy_end),
                      handle.get_stream());

  auto [one_hop_src, one_hop_dst, one_hop_edge_ids] =
    cugraph::detail::gather_one_hop_edgelist(handle, mg_graph_view, sources_copy);

  rmm::device_uvector<int> one_hop_gpu_id(one_hop_src.size(), handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               one_hop_gpu_id.begin(),
               one_hop_gpu_id.end(),
               handle.get_comms().get_rank());

  // Pull everything to rank 0
  auto sg_src = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{one_hop_src.data(), one_hop_src.size()});
  auto sg_dst = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{one_hop_dst.data(), one_hop_dst.size()});
  auto sg_gpu_id = cugraph::test::device_gatherv(
    handle, raft::device_span<int const>{one_hop_gpu_id.data(), one_hop_gpu_id.size()});
  auto sg_sources = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{sources.data(), col_rank == 0 ? sources.size() : 0});
  auto sg_destination_offsets = cugraph::test::device_gatherv(
    handle,
    raft::device_span<edge_t const>{destination_offsets.data(),
                                    col_rank == 0 ? destination_offsets.size() : 0});

  thrust::sort(handle.get_thrust_policy(),
               thrust::make_zip_iterator(sg_src.begin(), sg_gpu_id.begin(), sg_dst.begin()),
               thrust::make_zip_iterator(sg_src.end(), sg_gpu_id.end(), sg_dst.end()));

  std::vector<vertex_t> h_sources(sg_sources.size());
  std::vector<vertex_t> h_src(sg_src.size());
  std::vector<vertex_t> h_dst(sg_dst.size());
  std::vector<vertex_t> h_result_src(sg_destination_offsets.size());
  std::vector<vertex_t> h_result_dst(sg_destination_offsets.size());
  std::vector<edge_t> h_destination_offsets(sg_destination_offsets.size());

  raft::update_host(h_sources.data(), sg_sources.data(), sg_sources.size(), handle.get_stream());
  raft::update_host(h_src.data(), sg_src.data(), sg_src.size(), handle.get_stream());
  raft::update_host(h_dst.data(), sg_dst.data(), sg_dst.size(), handle.get_stream());
  raft::update_host(h_destination_offsets.data(),
                    sg_destination_offsets.data(),
                    sg_destination_offsets.size(),
                    handle.get_stream());

  thrust::for_each(thrust::host,
                   thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(sg_destination_offsets.size()),
                   [&] __host__(auto i) {
                     h_result_src[i] = h_sources[i / indices_per_source];
                     h_result_dst[i] = mg_graph_view.number_of_vertices();
                     edge_t offset   = h_destination_offsets[i];

                     for (size_t j = 0; j < h_src.size(); ++j) {
                       if (h_result_src[i] == h_src[j]) {
                         if (offset == 0) {
                           h_result_dst[i] = h_dst[j];
                           break;
                         }
                         --offset;
                       }
                     }
                   });

  auto new_end =
    thrust::remove_if(thrust::host,
                      thrust::make_zip_iterator(h_result_src.begin(), h_result_dst.begin()),
                      thrust::make_zip_iterator(h_result_src.end(), h_result_dst.end()),
                      [invalid_vertex = mg_graph_view.number_of_vertices()] __host__(auto p) {
                        return (thrust::get<1>(p) == invalid_vertex);
                      });

  h_result_src.resize(thrust::distance(
    thrust::make_zip_iterator(h_result_src.begin(), h_result_dst.begin()), new_end));
  h_result_dst.resize(h_result_src.size());

  return std::make_tuple(std::move(h_result_src), std::move(h_result_dst));
}

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
    constexpr edge_t indices_per_source       = 2;
    constexpr vertex_t repetitions_per_vertex = 5;
    constexpr vertex_t source_sample_count    = 3;

    // 2. Gather mnmg call
    // Generate random vertex ids in the range of current gpu

    auto&& [global_degree_offsets, global_out_degrees] =
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

    // get source global out degrees to generate indices
    auto active_source_degrees = cugraph::detail::get_active_major_global_degrees(
      *handle_, mg_graph_view, active_sources, global_out_degrees);

    auto random_destination_offsets =
      cugraph::test::generate_random_destination_indices(*handle_,
                                                         active_source_degrees,
                                                         mg_graph_view.number_of_vertices(),
                                                         edge_t{-1},
                                                         indices_per_source);

    rmm::device_uvector<edge_t> input_destination_offsets(random_destination_offsets.size(),
                                                          handle_->get_stream());
    raft::copy(input_destination_offsets.data(),
               random_destination_offsets.data(),
               random_destination_offsets.size(),
               handle_->get_stream());

    auto [src, dst, dst_map] =
      cugraph::detail::gather_local_edges(*handle_,
                                          mg_graph_view,
                                          active_sources,
                                          std::move(random_destination_offsets),
                                          indices_per_source,
                                          global_degree_offsets);

    if (prims_usecase.check_correctness) {
      // NOTE: This test assumes that edgea within the data structure are sorted
      //  We'll use gather_one_hop_edgelist to pull out the relevant edges
      auto [h_src, h_dst] = test_gather_local_edges(
        *handle_, mg_graph_view, active_sources, input_destination_offsets, indices_per_source);

      auto agg_src = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{src.data(), src.size()});
      auto agg_dst = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{dst.data(), dst.size()});

      thrust::sort(handle_->get_thrust_policy(),
                   thrust::make_zip_iterator(agg_src.begin(), agg_dst.begin()),
                   thrust::make_zip_iterator(agg_src.end(), agg_dst.end()));
      thrust::sort(thrust::host,
                   thrust::make_zip_iterator(h_src.begin(), h_dst.begin()),
                   thrust::make_zip_iterator(h_src.end(), h_dst.end()));

      std::vector<vertex_t> h_agg_src(agg_src.size());
      std::vector<vertex_t> h_agg_dst(agg_dst.size());
      raft::update_host(h_agg_src.data(), agg_src.data(), agg_src.size(), handle_->get_stream());
      raft::update_host(h_agg_dst.data(), agg_dst.data(), agg_dst.size(), handle_->get_stream());

      // FIXME:  Why are the randomly selected vertices on each GPU so similar??

      auto passed = thrust::equal(thrust::host, h_src.begin(), h_src.end(), h_agg_src.begin());
      passed &= thrust::equal(thrust::host, h_dst.begin(), h_dst.end(), h_agg_dst.begin());
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
