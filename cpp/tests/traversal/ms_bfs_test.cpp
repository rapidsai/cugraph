/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/check_utilities.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

struct MsBfs_Usecase {
  size_t radius;
  size_t max_seeds;
  bool test_weighted_{false};
  bool edge_masking_{false}; // FIXME: Not Supported
  bool check_correctness_{true};
};

template <typename input_usecase_t>
class Tests_MsBfs : public ::testing::TestWithParam<std::tuple<MsBfs_Usecase, input_usecase_t>> {
 public:
  Tests_MsBfs() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<MsBfs_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber             = false;
    auto [MsBfs_usecase, input_usecase] = param;
    raft::handle_t handle{};

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("SG Construct graph");
    }

    // NX MsBfs is not implemented for graph with self loop and multi edges therefore dropped
    // them especially for rmat generated graphs.
    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, MsBfs_usecase.test_weighted_, renumber, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;
    
    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (MsBfs_usecase.edge_masking_) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    rmm::device_uvector<vertex_t> d_labels(graph_view.number_of_vertices(), handle.get_stream());

    ASSERT_TRUE(graph_view.is_symmetric())
      << "Weakly connected components works only on undirected (symmetric) graphs.";

    // Call WCC
    cugraph::weakly_connected_components(handle, graph_view, d_labels.data());

    auto d_vertices = cugraph::test::sequence<vertex_t>(
      handle, graph_view.number_of_vertices(), size_t{1}, vertex_t{0});

    rmm::device_uvector<vertex_t> d_sorted_vertices{0, handle.get_stream()};
    rmm::device_uvector<vertex_t> d_sorted_labels{0, handle.get_stream()};

    std::tie(d_sorted_labels, d_sorted_vertices) =
      cugraph::test::sort_by_key<vertex_t, vertex_t>(handle, d_labels, d_vertices);

    auto num_unique_labels = cugraph::test::unique_count<vertex_t>(handle, d_sorted_labels);

    // Should run this test on datasets with more than 1 component
    ASSERT_TRUE(num_unique_labels > 1);

    rmm::device_uvector<vertex_t> d_first_components(num_unique_labels, handle.get_stream());

    std::tie(std::ignore, d_first_components) = cugraph::test::reduce_by_key<vertex_t, vertex_t>(
      handle, d_sorted_labels, d_sorted_vertices, num_unique_labels);

    // Select random seeds from different components
    raft::random::RngState rng_state(0);
    auto d_sources =
      cugraph::select_random_vertices(handle,
                                      graph_view,
                                      std::make_optional(raft::device_span<vertex_t const>{
                                        d_first_components.data(), d_first_components.size()}),
                                      rng_state,
                                      std::min(MsBfs_usecase.max_seeds, num_unique_labels),
                                      false,
                                      true);

    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    bool direction_optimizing = false;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("MsBfs");
    }

    cugraph::bfs(handle,
                 graph_view,
                 d_distances.begin(),
                 d_predecessors.begin(),
                 d_sources.data(),
                 d_sources.size(),
                 direction_optimizing,
                 static_cast<vertex_t>(MsBfs_usecase.radius));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (MsBfs_usecase.check_correctness_) {
      auto h_sources = cugraph::test::to_host(handle, d_sources);

      std::vector<rmm::device_uvector<vertex_t>> d_distances_ref{};
      std::vector<rmm::device_uvector<vertex_t>> d_predecessors_ref{};
      std::vector<std::vector<vertex_t>> h_distances_ref(h_sources.size());
      std::vector<std::vector<vertex_t>> h_predecessors_ref(h_sources.size());

      d_distances_ref.reserve(h_sources.size());
      d_predecessors_ref.reserve(h_sources.size());
      for (size_t i = 0; i < h_sources.size(); i++) {
        rmm::device_uvector<vertex_t> tmp_distances(graph_view.number_of_vertices(),
                                                    handle.get_next_usable_stream(i));
        rmm::device_uvector<vertex_t> tmp_predecessors(graph_view.number_of_vertices(),
                                                       handle.get_next_usable_stream(i));

        d_distances_ref.push_back(std::move(tmp_distances));
        d_predecessors_ref.push_back(std::move(tmp_predecessors));
      }

      // one by one
      for (size_t i = 0; i < h_sources.size(); i++) {
        auto source = h_sources[i];
        rmm::device_scalar<vertex_t> const d_source_i(source, handle.get_stream());
        cugraph::bfs(handle,
                     graph_view,
                     d_distances_ref[i].begin(),
                     d_predecessors_ref[i].begin(),
                     d_source_i.data(),
                     size_t{1},
                     direction_optimizing,
                     static_cast<vertex_t>(MsBfs_usecase.radius));
      }

      // checksum
      vertex_t ref_sum = 0;
      for (size_t i = 0; i < h_sources.size(); i++) {
        d_distances_ref[i] = cugraph::test::replace<vertex_t>(handle,
                                                              std::move(d_distances_ref[i]),
                                                              std::numeric_limits<vertex_t>::max(),
                                                              static_cast<vertex_t>(0));

        ref_sum +=
          cugraph::test::reduce(handle, d_distances_ref[i], static_cast<vertex_t>(0));
      }

      d_distances = cugraph::test::replace<vertex_t>(handle,
                                                     std::move(d_distances),
                                                     std::numeric_limits<vertex_t>::max(),
                                                     static_cast<vertex_t>(0));

      vertex_t ms_sum =
        cugraph::test::reduce(handle, d_distances, static_cast<vertex_t>(0));

      auto d_vertex_degree = graph_view.compute_out_degrees(handle);

      d_sources = cugraph::test::sort<vertex_t>(handle, std::move(d_sources));

      size_t seeds_degree_size = 0;

      std::tie(d_vertex_degree, d_vertices, seeds_degree_size) =
        cugraph::test::partition<vertex_t, vertex_t>(
          handle, std::move(d_vertex_degree), std::move(d_vertices), d_sources);

      d_vertex_degree.resize(seeds_degree_size, handle.get_stream());

      auto seeds_degree_sum =
        cugraph::test::reduce(handle, d_vertex_degree, static_cast<vertex_t>(0));

      // Check the degree of the seed vertices
      // If degree of all seeds is zero hence ref_sum is zero otherwise ref_sum > 0
      ASSERT_TRUE(seeds_degree_sum ? ref_sum > 0 : ref_sum == 0);
      ASSERT_TRUE(ref_sum < std::numeric_limits<vertex_t>::max());
      ASSERT_TRUE(ref_sum == ms_sum);
    }
  }
};

using Tests_MsBfs_File = Tests_MsBfs<cugraph::test::File_Usecase>;
using Tests_MsBfs_Rmat = Tests_MsBfs<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MsBfs_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MsBfs_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MsBfs_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MsBfs_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_MsBfs_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(MsBfs_Usecase{2, 5, false, false, true}, MsBfs_Usecase{4, 9, true, false, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MsBfs_Rmat,
                         // enable correctness checks
                         ::testing::Combine(::testing::Values(MsBfs_Usecase{2, 5, false, false, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MsBfs_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(MsBfs_Usecase{10, 150, false, false}, MsBfs_Usecase{12, 170, true, false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
