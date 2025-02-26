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
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGBFS<input_usecase_t>::handle_ = nullptr;

using Tests_MGBFS_Rmat = Tests_MGBFS<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGBFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGBFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(
      BFS_Usecase{0},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false)),
    std::make_tuple(
      BFS_Usecase{0},
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
    std::make_tuple(BFS_Usecase{0, false},
                    cugraph::test::Rmat_Usecase(
                      20, 16, 0.57, 0.19, 0.19, 0, false, false /* scramble vertex IDs */)),
    std::make_tuple(BFS_Usecase{0, false},
                    cugraph::test::Rmat_Usecase(
                      20, 16, 0.57, 0.19, 0.19, 0, false, false /* scramble vertex IDs */))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
