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

#include "detail/nbr_sampling_utils.cuh"
#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>

#include <gtest/gtest.h>

#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

struct Prims_Usecase {
  bool check_correctness{true};
  bool flag_replacement{true};
};

template <typename input_usecase_t>
class Tests_MG_Nbr_Sampling
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_MG_Nbr_Sampling() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    using namespace cugraph::test;

    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) {
      --row_comm_size;
    }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    constexpr bool sort_adjacency_list = true;

    auto [mg_graph, mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        handle, input_usecase, true, true, false, sort_adjacency_list);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view                        = mg_graph.view();
    constexpr edge_t indices_per_source       = 2;
    constexpr vertex_t repetitions_per_vertex = 5;
    constexpr vertex_t source_sample_count    = 3;

    // Generate random vertex ids in the range of current gpu
    auto random_sources =
      random_vertex_ids(handle,
                        mg_graph_view.local_vertex_partition_range_first(),
                        mg_graph_view.local_vertex_partition_range_last(),
                        std::min(mg_graph_view.local_vertex_partition_range_size() *
                                   (repetitions_per_vertex + vertex_t{1}),
                                 source_sample_count),
                        repetitions_per_vertex,
                        comm_rank);

    std::vector<int> h_fan_out{indices_per_source};  // depth = 1

#ifdef NO_CUGRAPH_OPS
    EXPECT_THROW(cugraph::uniform_nbr_sample(
                   handle,
                   mg_graph_view,
                   raft::device_span<vertex_t>(random_sources.data(), random_sources.size()),
                   raft::host_span<const int>(h_fan_out.data(), h_fan_out.size()),
                   prims_usecase.flag_replacement),
                 std::exception);
#else
    auto&& [d_src_out, d_dst_out, d_indices, d_counts] = cugraph::uniform_nbr_sample(
      handle,
      mg_graph_view,
      raft::device_span<vertex_t>(random_sources.data(), random_sources.size()),
      raft::host_span<const int>(h_fan_out.data(), h_fan_out.size()),
      prims_usecase.flag_replacement);

    if (prims_usecase.check_correctness) {
      // Consolidate results on GPU 0
      auto d_mg_start_src = cugraph::test::device_gatherv(
        handle, raft::device_span<vertex_t const>{random_sources.data(), random_sources.size()});
      auto d_mg_aggregate_src = cugraph::test::device_gatherv(
        handle, raft::device_span<vertex_t const>{d_src_out.data(), d_src_out.size()});
      auto d_mg_aggregate_dst = cugraph::test::device_gatherv(
        handle, raft::device_span<vertex_t const>{d_dst_out.data(), d_dst_out.size()});
      auto d_mg_aggregate_indices = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_indices.data(), d_indices.size()});

#if 0
      // FIXME:  extract_induced_subgraphs not currently support MG, so we'll skip this validation
      //         step

      //  First validate that the extracted edges are actually a subset of the
      //  edges in the input graph
      rmm::device_uvector<vertex_t> d_vertices(2 * d_mg_aggregate_src.size(), handle.get_stream());
      raft::copy(d_vertices.data(), d_mg_aggregate_src.data(), d_mg_aggregate_src.size(), handle.get_stream());
      raft::copy(d_vertices.data() + d_mg_aggregate_src.size(),
                 d_mg_aggregate_dst.data(),
                 d_mg_aggregate_dst.size(),
                 handle.get_stream());
      thrust::sort(handle.get_thrust_policy(), d_vertices.begin(), d_vertices.end());
      auto vertices_end =
        thrust::unique(handle.get_thrust_policy(), d_vertices.begin(), d_vertices.end());
      d_vertices.resize(thrust::distance(d_vertices.begin(), vertices_end), handle.get_stream());

      d_vertices = cugraph::detail::shuffle_int_vertices_by_gpu_id(handle, std::move(d_vertices), mg_graph_view.vertex_partition_range_lasts());

      thrust::sort(handle.get_thrust_policy(), d_vertices.begin(), d_vertices.end());

      rmm::device_uvector<size_t> d_subgraph_offsets(2, handle.get_stream());
      std::vector<size_t> h_subgraph_offsets({0, d_vertices.size()});

      raft::update_device(d_subgraph_offsets.data(),
                          h_subgraph_offsets.data(),
                          h_subgraph_offsets.size(),
                          handle.get_stream());

      auto [d_src_in, d_dst_in, d_indices_in, d_ignore] = extract_induced_subgraphs(
        handle, mg_graph_view, d_subgraph_offsets.data(), d_vertices.data(), 1, true);

      cugraph::test::validate_extracted_graph_is_subgraph(
        handle, d_src_in, d_dst_in, *d_indices_in, d_src_out, d_dst_out, d_indices);
#endif

      if (d_mg_aggregate_src.size() > 0) {
        cugraph::test::validate_sampling_depth(handle,
                                               std::move(d_mg_aggregate_src),
                                               std::move(d_mg_aggregate_dst),
                                               std::move(d_mg_aggregate_indices),
                                               std::move(d_mg_start_src),
                                               h_fan_out.size());
      }
    }
#endif
  }
};

using Tests_MG_Nbr_Sampling_File = Tests_MG_Nbr_Sampling<cugraph::test::File_Usecase>;

using Tests_MG_Nbr_Sampling_Rmat = Tests_MG_Nbr_Sampling<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_Nbr_Sampling_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_Nbr_Sampling_File, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_Nbr_Sampling_File, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_Nbr_Sampling_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_Nbr_Sampling_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MG_Nbr_Sampling_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_Nbr_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true, true}, Prims_Usecase{true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MG_Nbr_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MG_Nbr_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
