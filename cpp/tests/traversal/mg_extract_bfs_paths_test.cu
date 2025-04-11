/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <cuda/std/iterator>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

struct ExtractBFSPaths_Usecase {
  size_t source{0};
  size_t num_paths_to_check{0};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGExtractBFSPaths
  : public ::testing::TestWithParam<std::tuple<ExtractBFSPaths_Usecase, input_usecase_t>> {
 public:
  Tests_MGExtractBFSPaths() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(ExtractBFSPaths_Usecase const& extract_bfs_paths_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t    = float;
    using edge_type_t = int32_t;

    constexpr bool renumber = true;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, true, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
    auto mg_graph_view = mg_graph.view();

    ASSERT_TRUE(static_cast<vertex_t>(extract_bfs_paths_usecase.source) >= 0 &&
                static_cast<vertex_t>(extract_bfs_paths_usecase.source) <
                  mg_graph_view.number_of_vertices())
      << "Invalid starting source.";

    ASSERT_TRUE(extract_bfs_paths_usecase.num_paths_to_check > 0) << "Invalid num_paths_to_check";
    ASSERT_TRUE(extract_bfs_paths_usecase.num_paths_to_check < mg_graph_view.number_of_vertices())
      << "Invalid num_paths_to_check, more than number of vertices";

    rmm::device_uvector<vertex_t> d_mg_distances(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle_->get_stream());
    rmm::device_uvector<vertex_t> d_mg_predecessors(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    auto const d_mg_source =
      mg_graph_view.in_local_vertex_partition_range_nocheck(extract_bfs_paths_usecase.source)
        ? std::make_optional<rmm::device_scalar<vertex_t>>(extract_bfs_paths_usecase.source,
                                                           handle_->get_stream())
        : std::nullopt;

    cugraph::bfs(*handle_,
                 mg_graph_view,
                 d_mg_distances.data(),
                 d_mg_predecessors.data(),
                 d_mg_source ? d_mg_source->data() : static_cast<vertex_t const*>(nullptr),
                 d_mg_source ? size_t{1} : size_t{0},
                 false,
                 std::numeric_limits<vertex_t>::max());

    auto h_mg_distances    = cugraph::test::to_host(*handle_, d_mg_distances);
    auto h_mg_predecessors = cugraph::test::to_host(*handle_, d_mg_predecessors);

    rmm::device_uvector<vertex_t> d_vertices(mg_graph_view.local_vertex_partition_range_size(),
                                             handle_->get_stream());
    {
      constexpr vertex_t invalid_vertex = cugraph::invalid_vertex_id<vertex_t>::value;
      auto local_vertex_first           = mg_graph_view.local_vertex_partition_range_first();
      cugraph::detail::sequence_fill(
        handle_->get_stream(), d_vertices.begin(), d_vertices.size(), local_vertex_first);
      auto end_iter = thrust::remove_if(
        handle_->get_thrust_policy(),
        d_vertices.begin(),
        d_vertices.end(),
        [invalid_vertex, predecessors = d_mg_predecessors.data(), local_vertex_first] __device__(
          auto v) { return predecessors[v - local_vertex_first] == invalid_vertex; });
      d_vertices.resize(cuda::std::distance(d_vertices.begin(), end_iter), handle_->get_stream());
    }

    // Compute size of the distributed vertex set
    auto num_of_paths_in_given_set = d_vertices.size();
    num_of_paths_in_given_set      = cugraph::host_scalar_allreduce(handle_->get_comms(),
                                                               num_of_paths_in_given_set,
                                                               raft::comms::op_t::SUM,
                                                               handle_->get_stream());

    raft::random::RngState rng_state(0);
    auto d_mg_destinations = cugraph::select_random_vertices(
      *handle_,
      mg_graph_view,
      std::make_optional(raft::device_span<vertex_t const>{d_vertices.data(), d_vertices.size()}),
      rng_state,
      std::min(num_of_paths_in_given_set, extract_bfs_paths_usecase.num_paths_to_check),
      false,
      false);

    rmm::device_uvector<vertex_t> d_mg_paths(0, handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG extract_bfs_paths");
    }

    vertex_t mg_max_path_length{0};

    std::tie(d_mg_paths, mg_max_path_length) = extract_bfs_paths(*handle_,
                                                                 mg_graph_view,
                                                                 d_mg_distances.data(),
                                                                 d_mg_predecessors.data(),
                                                                 d_mg_destinations.data(),
                                                                 d_mg_destinations.size());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (extract_bfs_paths_usecase.check_correctness) {
      // unrenumber & aggregate MG destination vertices to extract paths, & results
      // collect MG BFS results instead of re-running SG BFS as BFS is non-deterministic

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_predecessors.data(),
        d_mg_predecessors.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_destinations.data(),
        d_mg_destinations.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_paths.data(),
        d_mg_paths.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      rmm::device_uvector<vertex_t> d_mg_aggregate_distances(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_distances) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<vertex_t const>(d_mg_distances.data(), d_mg_distances.size()));

      rmm::device_uvector<vertex_t> d_mg_aggregate_predecessors(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_predecessors) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<vertex_t const>(d_mg_predecessors.data(), d_mg_predecessors.size()));

      auto d_mg_aggregate_destinations =
        cugraph::test::device_gatherv(*handle_, d_mg_destinations.data(), d_mg_destinations.size());
      auto d_mg_aggregate_paths =
        cugraph::test::device_gatherv(*handle_, d_mg_paths.data(), d_mg_paths.size());

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::tie(sg_graph, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // run SG extract_bfs_paths

        auto sg_graph_view = sg_graph.view();

        auto [d_sg_paths, sg_max_path_length] =
          extract_bfs_paths(*handle_,
                            sg_graph_view,
                            d_mg_aggregate_distances.data(),
                            d_mg_aggregate_predecessors.data(),
                            d_mg_aggregate_destinations.data(),
                            d_mg_aggregate_destinations.size());

        // compare

        ASSERT_EQ(mg_max_path_length, sg_max_path_length);
        ASSERT_EQ(d_mg_aggregate_paths.size(), d_sg_paths.size());

        auto h_mg_aggregate_paths = cugraph::test::to_host(*handle_, d_mg_aggregate_paths);
        auto h_sg_paths           = cugraph::test::to_host(*handle_, d_sg_paths);

        ASSERT_TRUE(
          std::equal(h_mg_aggregate_paths.begin(), h_mg_aggregate_paths.end(), h_sg_paths.begin()));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGExtractBFSPaths<input_usecase_t>::handle_ = nullptr;

using Tests_MGExtractBFSPaths_File = Tests_MGExtractBFSPaths<cugraph::test::File_Usecase>;
using Tests_MGExtractBFSPaths_Rmat = Tests_MGExtractBFSPaths<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_MGExtractBFSPaths_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGExtractBFSPaths_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGExtractBFSPaths_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGExtractBFSPaths_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(ExtractBFSPaths_Usecase{0, 10, false},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 10, true},
                    cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 100, false},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 100, true},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 100, false},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 100, true},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{100, 100, false},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(ExtractBFSPaths_Usecase{100, 100, true},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGExtractBFSPaths_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(ExtractBFSPaths_Usecase{0, 20, false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, true)),
    std::make_tuple(ExtractBFSPaths_Usecase{0, 20, true},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGExtractBFSPaths_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_pair(ExtractBFSPaths_Usecase{0, 1000, false, false},
                   cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, true)),
    std::make_pair(ExtractBFSPaths_Usecase{0, 1000, true, false},
                   cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
