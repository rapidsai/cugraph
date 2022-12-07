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

#include <gtest/gtest.h>

#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

struct Prims_Usecase {
  bool check_correctness{true};
  bool flag_replacement{true};
};

template <typename input_usecase_t>
class Tests_Uniform_Neighbor_Sampling
  : public ::testing::TestWithParam<std::tuple<Prims_Usecase, input_usecase_t>> {
 public:
  Tests_Uniform_Neighbor_Sampling() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Prims_Usecase const& prims_usecase, input_usecase_t const& input_usecase)
  {
    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    constexpr edge_t indices_per_source       = 2;
    constexpr vertex_t repetitions_per_vertex = 5;
    constexpr vertex_t source_sample_count    = 3;

    // Generate random vertex ids in the range of current gpu

    // Generate random sources to gather on
    auto random_sources =
      cugraph::test::random_vertex_ids(handle,
                                       graph_view.local_vertex_partition_range_first(),
                                       graph_view.local_vertex_partition_range_last(),
                                       std::min(graph_view.local_vertex_partition_range_size() *
                                                  (repetitions_per_vertex + vertex_t{1}),
                                                source_sample_count),
                                       repetitions_per_vertex,
                                       uint64_t{0});

    std::vector<int> h_fan_out{indices_per_source};  // depth = 1

#ifdef NO_CUGRAPH_OPS
    EXPECT_THROW(cugraph::uniform_nbr_sample(
                   handle,
                   graph_view,
                   edge_weight_view,
                   raft::device_span<vertex_t>(random_sources.data(), random_sources.size()),
                   raft::host_span<const int>(h_fan_out.data(), h_fan_out.size()),
                   prims_usecase.flag_replacement),
                 std::exception);
#else
    auto&& [d_src_out, d_dst_out, d_indices, d_counts] = cugraph::uniform_nbr_sample(
      handle,
      graph_view,
      edge_weight_view,
      raft::device_span<vertex_t>(random_sources.data(), random_sources.size()),
      raft::host_span<const int>(h_fan_out.data(), h_fan_out.size()),
      prims_usecase.flag_replacement);

    if (prims_usecase.check_correctness) {
      //  First validate that the extracted edges are actually a subset of the
      //  edges in the input graph
      rmm::device_uvector<vertex_t> d_vertices(2 * d_src_out.size(), handle.get_stream());
      raft::copy(d_vertices.data(), d_src_out.data(), d_src_out.size(), handle.get_stream());
      raft::copy(d_vertices.data() + d_src_out.size(),
                 d_dst_out.data(),
                 d_dst_out.size(),
                 handle.get_stream());
      thrust::sort(handle.get_thrust_policy(), d_vertices.begin(), d_vertices.end());
      auto vertices_end =
        thrust::unique(handle.get_thrust_policy(), d_vertices.begin(), d_vertices.end());
      d_vertices.resize(thrust::distance(d_vertices.begin(), vertices_end), handle.get_stream());

      rmm::device_uvector<size_t> d_subgraph_offsets(2, handle.get_stream());
      std::vector<size_t> h_subgraph_offsets({0, d_vertices.size()});

      raft::update_device(d_subgraph_offsets.data(),
                          h_subgraph_offsets.data(),
                          h_subgraph_offsets.size(),
                          handle.get_stream());

      auto [d_src_in, d_dst_in, d_indices_in, d_ignore] = extract_induced_subgraphs(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<size_t const>(d_subgraph_offsets.data(), 2),
        raft::device_span<vertex_t const>(d_vertices.data(), d_vertices.size()),
        true);

      cugraph::test::validate_extracted_graph_is_subgraph(
        handle, d_src_in, d_dst_in, *d_indices_in, d_src_out, d_dst_out, d_indices);

      cugraph::test::validate_sampling_depth(handle,
                                             std::move(d_src_out),
                                             std::move(d_dst_out),
                                             std::move(d_indices),
                                             std::move(random_sources),
                                             h_fan_out.size());
    }
#endif
  }
};

using Tests_Uniform_Neighbor_Sampling_File =
  Tests_Uniform_Neighbor_Sampling<cugraph::test::File_Usecase>;

using Tests_Uniform_Neighbor_Sampling_Rmat =
  Tests_Uniform_Neighbor_Sampling<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Uniform_Neighbor_Sampling_File, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_File, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_File, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_Rmat, CheckInt32Int32Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_Rmat, CheckInt32Int64Float)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Uniform_Neighbor_Sampling_Rmat, CheckInt64Int64Float)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Uniform_Neighbor_Sampling_File,
  ::testing::Combine(
    ::testing::Values(Prims_Usecase{true, true}, Prims_Usecase{true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Uniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Uniform_Neighbor_Sampling_Rmat,
  ::testing::Combine(::testing::Values(Prims_Usecase{false, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
