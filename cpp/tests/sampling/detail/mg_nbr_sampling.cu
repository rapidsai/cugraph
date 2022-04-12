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
#include <sampling/nbr_sampling_impl.cuh>

#include <gtest/gtest.h>

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
    using gpu_t = int;

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

    // Generate random sources to gather on
    auto random_sources = random_vertex_ids(handle,
                                            mg_graph_view.local_vertex_partition_range_first(),
                                            mg_graph_view.local_vertex_partition_range_last(),
                                            source_sample_count,
                                            repetitions_per_vertex);
    rmm::device_uvector<gpu_t> random_source_gpu_ids(random_sources.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 random_source_gpu_ids.begin(),
                 random_source_gpu_ids.end(),
                 comm_rank);

    std::vector<int> h_fan_out{indices_per_source};  // depth = 1

    auto begin_in_pairs = thrust::make_zip_iterator(
      thrust::make_tuple(random_sources.begin(), random_source_gpu_ids.begin()));
    auto end_in_pairs = thrust::make_zip_iterator(
      thrust::make_tuple(random_sources.end(), random_source_gpu_ids.end()));

    // gather input:
    //
    auto&& [tuple_vertex_ranks, counts] = cugraph::detail::shuffle_to_gpus(
      handle, mg_graph_view, begin_in_pairs, end_in_pairs, gpu_t{});

    auto&& [tuple_quad, v_sizes] = cugraph::uniform_nbr_sample(handle,
                                                               mg_graph_view,
                                                               random_sources.begin(),
                                                               random_source_gpu_ids.begin(),
                                                               random_sources.size(),
                                                               h_fan_out,
                                                               prims_usecase.flag_replacement);

    auto&& d_src_out = std::get<0>(tuple_quad);
    auto&& d_dst_out = std::get<1>(tuple_quad);
    auto&& d_gpu_ids = std::get<2>(tuple_quad);

    if (prims_usecase.check_correctness) {
      auto self_rank = handle.get_comms().get_rank();

      // bring inputs and outputs on one rank
      // and check if test passed:
      //
      if (self_rank == gpu_t{0}) {
        auto num_ranks = v_sizes.size();
        ASSERT_TRUE(counts.size() == num_ranks);  // == #ranks

        // CAVEAT: in size << out_size;
        //
        auto total_in_sizes  = std::accumulate(counts.begin(), counts.end(), 0);
        auto total_out_sizes = std::accumulate(v_sizes.begin(), v_sizes.end(), 0);

        // merge inputs / outputs to be checked on host:
        //
        std::vector<vertex_t> h_start_in{};
        h_start_in.reserve(total_in_sizes);

        std::vector<gpu_t> h_ranks_in{};
        h_ranks_in.reserve(total_in_sizes);

        std::vector<vertex_t> h_src_out{};
        h_src_out.reserve(total_out_sizes);

        std::vector<vertex_t> h_dst_out{};
        h_dst_out.reserve(total_out_sizes);

        std::vector<gpu_t> h_ranks_out{};
        h_ranks_out.reserve(total_out_sizes);

        auto filler = [&handle](auto const& coalesced_in,
                                auto& accumulator,
                                auto& v_per_rank,
                                auto count,
                                auto offset) {
          auto start_offset_in = coalesced_in.cbegin() + offset;

          raft::update_host(
            v_per_rank.data(), start_offset_in, static_cast<size_t>(count), handle.get_stream());

          accumulator.insert(accumulator.begin() + offset, v_per_rank.begin(), v_per_rank.end());
        };

        size_t in_offset  = 0;
        size_t out_offset = 0;
        for (size_t index_rank = 0; index_rank < num_ranks; ++index_rank) {
          auto in_sz = counts[index_rank];
          std::vector<vertex_t> per_rank_start_in(in_sz);
          std::vector<gpu_t> per_rank_in(in_sz);

          filler(std::get<0>(tuple_vertex_ranks), h_start_in, per_rank_start_in, in_sz, in_offset);

          filler(std::get<1>(tuple_vertex_ranks), h_ranks_in, per_rank_in, in_sz, in_offset);

          auto out_sz = v_sizes[index_rank];
          std::vector<vertex_t> per_rank_src_out(out_sz);
          std::vector<vertex_t> per_rank_dst_out(out_sz);
          std::vector<gpu_t> per_rank_out(out_sz);

          filler(d_src_out, h_src_out, per_rank_src_out, out_sz, out_offset);
          filler(d_dst_out, h_dst_out, per_rank_dst_out, out_sz, out_offset);
          filler(d_gpu_ids, h_ranks_out, per_rank_out, out_sz, out_offset);

          in_offset += in_sz;
          out_offset += out_sz;
        }

        bool passed = cugraph::test::check_forest_trees_by_rank(
          h_start_in, h_ranks_in, h_src_out, h_dst_out, h_ranks_out);

        ASSERT_TRUE(passed);
      }
    }
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
