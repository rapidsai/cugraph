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
 * See the License for the specific language governing permissions and
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

#include <gtest/gtest.h>

#include <random>

struct BFS_Usecase {
  size_t source{0};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGBFS : public ::testing::TestWithParam<std::tuple<BFS_Usecase, input_usecase_t>> {
 public:
  Tests_MGBFS() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running BFS on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(BFS_Usecase const& bfs_usecase, input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;
    using weight_t    = float;

    bool constexpr renumber         = true;
    bool constexpr test_weighted    = false;
    bool constexpr drop_self_loops  = false;
    bool constexpr drop_multi_edges = false;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> mg_renumber_map{std::nullopt};
    std::tie(mg_graph, std::ignore, mg_renumber_map) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (bfs_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    ASSERT_TRUE(static_cast<vertex_t>(bfs_usecase.source) >= 0 &&
                static_cast<vertex_t>(bfs_usecase.source) < mg_graph_view.number_of_vertices())
      << "Invalid starting source.";

    // 2. run MG BFS

    rmm::device_uvector<vertex_t> d_mg_distances(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle_->get_stream());
    rmm::device_uvector<vertex_t> d_mg_predecessors(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    auto d_mg_source = mg_graph_view.in_local_vertex_partition_range_nocheck(bfs_usecase.source)
                         ? std::make_optional<rmm::device_scalar<vertex_t>>(bfs_usecase.source,
                                                                            handle_->get_stream())
                         : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG BFS");
    }

    cugraph::bfs(*handle_,
                 mg_graph_view,
                 d_mg_distances.data(),
                 d_mg_predecessors.data(),
                 d_mg_source ? (*d_mg_source).data() : static_cast<vertex_t const*>(nullptr),
                 d_mg_source ? size_t{1} : size_t{0},
                 mg_graph_view.is_symmetric() ? true : false,
                 std::numeric_limits<vertex_t>::max());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (bfs_usecase.check_correctness) {
      // 3-1. unrenumber & aggregate MG source & results

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_predecessors.data(),
        d_mg_predecessors.size(),
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      cugraph::unrenumber_int_vertices<vertex_t, true>(
        *handle_,
        d_mg_source ? (*d_mg_source).data() : static_cast<vertex_t*>(nullptr),
        d_mg_source ? size_t{1} : size_t{0},
        (*mg_renumber_map).data(),
        mg_graph_view.vertex_partition_range_lasts());

      // 3-2. aggregate MG results

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

      auto d_mg_aggregate_sources = cugraph::test::device_gatherv(
        *handle_,
        d_mg_source ? (*d_mg_source).data() : static_cast<vertex_t const*>(nullptr),
        d_mg_source ? size_t{1} : size_t{0});

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
        // 3-3. run SG BFS

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        rmm::device_uvector<vertex_t> d_sg_distances(sg_graph_view.number_of_vertices(),
                                                     handle_->get_stream());
        rmm::device_uvector<vertex_t> d_sg_predecessors(
          sg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

        cugraph::bfs(*handle_,
                     sg_graph_view,
                     d_sg_distances.data(),
                     d_sg_predecessors.data(),
                     d_mg_aggregate_sources.data(),
                     d_mg_aggregate_sources.size(),
                     false,
                     std::numeric_limits<vertex_t>::max());

        // 3-4. compare

        std::vector<edge_t> h_sg_offsets =
          cugraph::test::to_host(*handle_, sg_graph_view.local_edge_partition_view().offsets());
        std::vector<vertex_t> h_sg_indices =
          cugraph::test::to_host(*handle_, sg_graph_view.local_edge_partition_view().indices());

        std::vector<vertex_t> h_mg_aggregate_distances =
          cugraph::test::to_host(*handle_, d_mg_aggregate_distances);
        std::vector<vertex_t> h_mg_aggregate_predecessors =
          cugraph::test::to_host(*handle_, d_mg_aggregate_predecessors);

        std::vector<vertex_t> h_sg_distances = cugraph::test::to_host(*handle_, d_sg_distances);
        std::vector<vertex_t> h_sg_predecessors =
          cugraph::test::to_host(*handle_, d_sg_predecessors);

        ASSERT_TRUE(std::equal(h_mg_aggregate_distances.begin(),
                               h_mg_aggregate_distances.end(),
                               h_sg_distances.begin()));
        for (size_t i = 0; i < h_mg_aggregate_predecessors.size(); ++i) {
          if (h_mg_aggregate_predecessors[i] == cugraph::invalid_vertex_id<vertex_t>::value) {
            ASSERT_TRUE(h_sg_predecessors[i] == h_mg_aggregate_predecessors[i])
              << "vertex reachability does not match with the SG result.";
          } else {
            ASSERT_TRUE(h_sg_distances[h_mg_aggregate_predecessors[i]] + 1 == h_sg_distances[i])
              << "distances to this vertex != distances to the predecessor vertex + 1.";
            bool found{false};
            for (auto j = h_sg_offsets[h_mg_aggregate_predecessors[i]];
                 j < h_sg_offsets[h_mg_aggregate_predecessors[i] + 1];
                 ++j) {
              if (h_sg_indices[j] == i) {
                found = true;
                break;
              }
            }
            ASSERT_TRUE(found) << "no edge from the predecessor vertex to this vertex.";
          }
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGBFS<input_usecase_t>::handle_ = nullptr;

using Tests_MGBFS_File = Tests_MGBFS<cugraph::test::File_Usecase>;
using Tests_MGBFS_Rmat = Tests_MGBFS<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGBFS_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGBFS_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGBFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGBFS_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(BFS_Usecase{0, false}, BFS_Usecase{0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGBFS_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(BFS_Usecase{0, false}, BFS_Usecase{0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGBFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(
      BFS_Usecase{0, false},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false)),
    std::make_tuple(
      BFS_Usecase{0, true},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGBFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(BFS_Usecase{0, false, false},
                    cugraph::test::Rmat_Usecase(
                      20, 16, 0.57, 0.19, 0.19, 0, false, false /* scramble vertex IDs */)),
    std::make_tuple(BFS_Usecase{0, true, false},
                    cugraph::test::Rmat_Usecase(
                      20, 16, 0.57, 0.19, 0.19, 0, false, false /* scramble vertex IDs */))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
