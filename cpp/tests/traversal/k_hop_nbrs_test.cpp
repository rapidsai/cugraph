/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

template <typename vertex_t, typename edge_t>
std::tuple<std::vector<size_t>, std::vector<vertex_t>> k_hop_nbrs_reference(
  edge_t const* offsets,
  vertex_t const* indices,
  vertex_t const* start_vertices,
  size_t num_start_vertices,
  size_t k)
{
  std::vector<std::tuple<vertex_t, size_t>> cur_tagged_vertex_buffer(num_start_vertices);
  for (size_t i = 0; i < num_start_vertices; ++i) {
    cur_tagged_vertex_buffer[i] = std::make_tuple(start_vertices[i], i);
  }

  std::vector<size_t> start_vertex_indices{};
  std::vector<vertex_t> nbrs{};
  for (size_t iter = 0; iter < k; ++iter) {
    std::vector<std::tuple<vertex_t, size_t>> new_tagged_vertex_buffer{};
    for (size_t i = 0; i < cur_tagged_vertex_buffer.size(); ++i) {
      auto [v, tag] = cur_tagged_vertex_buffer[i];
      for (edge_t j = offsets[v]; j < offsets[v + 1]; ++j) {
        new_tagged_vertex_buffer.push_back(std::make_tuple(indices[j], tag));
      }
    }
    std::sort(new_tagged_vertex_buffer.begin(), new_tagged_vertex_buffer.end());
    new_tagged_vertex_buffer.resize(
      std::distance(new_tagged_vertex_buffer.begin(),
                    std::unique(new_tagged_vertex_buffer.begin(), new_tagged_vertex_buffer.end())));
    new_tagged_vertex_buffer.shrink_to_fit();
    if (iter < (k - 1)) {
      cur_tagged_vertex_buffer.clear();
      cur_tagged_vertex_buffer.shrink_to_fit();
      std::swap(cur_tagged_vertex_buffer, new_tagged_vertex_buffer);
    } else {
      std::sort(
        new_tagged_vertex_buffer.begin(), new_tagged_vertex_buffer.end(), [](auto lhs, auto rhs) {
          return std::make_tuple(std::get<1>(lhs), std::get<0>(lhs)) <
                 std::make_tuple(std::get<1>(rhs), std::get<0>(rhs));
        });
      start_vertex_indices.resize(new_tagged_vertex_buffer.size());
      nbrs.resize(new_tagged_vertex_buffer.size());
      for (size_t i = 0; i < new_tagged_vertex_buffer.size(); ++i) {
        start_vertex_indices[i] = std::get<1>(new_tagged_vertex_buffer[i]);
        nbrs[i]                 = std::get<0>(new_tagged_vertex_buffer[i]);
      }
    }
  }

  std::vector<size_t> nbr_offsets(num_start_vertices + 1, 0);
  for (size_t i = 0; i < start_vertex_indices.size(); ++i) {
    auto idx = start_vertex_indices[i];
    ++nbr_offsets[idx];
  }
  std::exclusive_scan(nbr_offsets.begin(), nbr_offsets.end(), nbr_offsets.begin(), size_t{0});

  return std::make_tuple(std::move(nbr_offsets), std::move(nbrs));
}

struct KHopNbrs_Usecase {
  size_t num_start_vertices{0};
  size_t k{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_KHopNbrs
  : public ::testing::TestWithParam<std::tuple<KHopNbrs_Usecase, input_usecase_t>> {
 public:
  Tests_KHopNbrs() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(KHopNbrs_Usecase const& k_hop_nbrs_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    using weight_t = float;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    std::vector<vertex_t> h_start_vertices(k_hop_nbrs_usecase.num_start_vertices);
    for (size_t i = 0; i < h_start_vertices.size(); ++i) {
      h_start_vertices[i] =
        static_cast<vertex_t>(std::hash<size_t>{}(i) % graph_view.number_of_vertices());
    }
    rmm::device_uvector<vertex_t> d_start_vertices(h_start_vertices.size(), handle.get_stream());
    raft::update_device(d_start_vertices.data(),
                        h_start_vertices.data(),
                        h_start_vertices.size(),
                        handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("K-hop neighbors");
    }

    auto [offsets, nbrs] = cugraph::k_hop_nbrs(
      handle,
      graph_view,
      raft::device_span<vertex_t const>(d_start_vertices.data(), d_start_vertices.size()),
      k_hop_nbrs_usecase.k);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (k_hop_nbrs_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, false, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      auto h_offsets = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().offsets());
      auto h_indices = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().indices());

      auto unrenumbered_start_vertices = std::vector<vertex_t>(h_start_vertices.size());
      if (renumber) {
        auto h_renumber_map_labels = cugraph::test::to_host(handle, *d_renumber_map_labels);
        for (size_t i = 0; i < unrenumbered_start_vertices.size(); ++i) {
          unrenumbered_start_vertices[i] = h_renumber_map_labels[h_start_vertices[i]];
        }
      }

      auto [h_reference_offsets, h_reference_nbrs] =
        k_hop_nbrs_reference(h_offsets.data(),
                             h_indices.data(),
                             unrenumbered_start_vertices.data(),
                             unrenumbered_start_vertices.size(),
                             k_hop_nbrs_usecase.k);

      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               nbrs.data(),
                                               nbrs.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices(),
                                               true);
      }
      auto h_cugraph_offsets = cugraph::test::to_host(handle, offsets);
      auto h_cugraph_nbrs    = cugraph::test::to_host(handle, nbrs);

      ASSERT_TRUE(std::equal(
        h_reference_offsets.begin(), h_reference_offsets.end(), h_cugraph_offsets.begin()))
        << "offsets do not match with the reference values.";

      for (size_t i = 0; i < k_hop_nbrs_usecase.num_start_vertices; ++i) {
        std::sort(h_reference_nbrs.begin() + h_reference_offsets[i],
                  h_reference_nbrs.begin() + h_reference_offsets[i + 1]);
        std::sort(h_cugraph_nbrs.begin() + h_cugraph_offsets[i],
                  h_cugraph_nbrs.begin() + h_cugraph_offsets[i + 1]);
      }
      ASSERT_TRUE(
        std::equal(h_reference_nbrs.begin(), h_reference_nbrs.end(), h_cugraph_nbrs.begin()))
        << "neighbors do not match with the reference values.";
    }
  }
};

using Tests_KHopNbrs_File = Tests_KHopNbrs<cugraph::test::File_Usecase>;
using Tests_KHopNbrs_Rmat = Tests_KHopNbrs<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_KHopNbrs_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_KHopNbrs_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_KHopNbrs_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_KHopNbrs_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_KHopNbrs_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(KHopNbrs_Usecase{1024, 5},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(KHopNbrs_Usecase{1024, 4},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(KHopNbrs_Usecase{1024, 3},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(KHopNbrs_Usecase{1024, 2},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(KHopNbrs_Usecase{1024, 1},
                    cugraph::test::File_Usecase("test/datasets/wiki-Talk.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_KHopNbrs_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(KHopNbrs_Usecase{1024, 2},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_KHopNbrs_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_pair(KHopNbrs_Usecase{4, 2, false},
                   cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
