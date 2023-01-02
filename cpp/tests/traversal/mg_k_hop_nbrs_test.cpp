/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct KHopNbrs_Usecase {
  size_t num_start_vertices{0};
  size_t k{0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGKHopNbrs
  : public ::testing::TestWithParam<std::tuple<KHopNbrs_Usecase, input_usecase_t>> {
 public:
  Tests_MGKHopNbrs() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running K-hop neighbors on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(KHopNbrs_Usecase const& k_hop_nbrs_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t = float;

    HighResTimer hr_timer{};

    // 1. create MG graph

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> d_mg_renumber_map_labels{std::nullopt};
    std::tie(mg_graph, std::ignore, d_mg_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    std::vector<vertex_t> h_mg_start_vertices(
      std::min(static_cast<size_t>(
                 (k_hop_nbrs_usecase.num_start_vertices / comm_size) +
                 (comm_rank < (k_hop_nbrs_usecase.num_start_vertices % comm_size) ? 1 : 0)),
               static_cast<size_t>(mg_graph_view.local_vertex_partition_range_size())));
    for (size_t i = 0; i < h_mg_start_vertices.size(); ++i) {
      h_mg_start_vertices[i] =
        mg_graph_view.local_vertex_partition_range_first() +
        static_cast<vertex_t>(std::hash<size_t>{}(i) %
                              mg_graph_view.local_vertex_partition_range_size());
    }
    rmm::device_uvector<vertex_t> d_mg_start_vertices(h_mg_start_vertices.size(),
                                                      handle_->get_stream());
    raft::update_device(d_mg_start_vertices.data(),
                        h_mg_start_vertices.data(),
                        h_mg_start_vertices.size(),
                        handle_->get_stream());

    // 2. run MG K-hop neighbors

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG K-hop neighbors");
    }

    auto [d_mg_offsets, d_mg_nbrs] = cugraph::k_hop_nbrs(
      *handle_,
      mg_graph_view,
      raft::device_span<vertex_t const>(d_mg_start_vertices.data(), d_mg_start_vertices.size()),
      k_hop_nbrs_usecase.k);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (k_hop_nbrs_usecase.check_correctness) {
      // 3-1. aggregate MG results

      auto h_mg_offsets = cugraph::test::to_host(*handle_, d_mg_offsets);
      std::vector<size_t> h_mg_counts(d_mg_start_vertices.size());
      for (size_t i = 0; i < h_mg_counts.size(); ++i) {
        h_mg_counts[i] = h_mg_offsets[i + 1] - h_mg_offsets[i];
      }
      rmm::device_uvector<size_t> d_mg_counts(h_mg_counts.size(), handle_->get_stream());
      raft::update_device(
        d_mg_counts.data(), h_mg_counts.data(), h_mg_counts.size(), handle_->get_stream());

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>((*d_mg_renumber_map_labels).data(),
                                          (*d_mg_renumber_map_labels).size()));
      auto d_mg_aggregate_start_vertices = cugraph::test::device_gatherv(
        *handle_,
        raft::device_span<vertex_t const>(d_mg_start_vertices.data(), d_mg_start_vertices.size()));

      auto d_mg_aggregate_counts = cugraph::test::device_gatherv(
        *handle_, raft::device_span<size_t const>(d_mg_counts.data(), d_mg_counts.size()));
      auto d_mg_aggregate_nbrs = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(d_mg_nbrs.data(), d_mg_nbrs.size()));

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. unrenumbr MG start vertices & neighbors

        cugraph::unrenumber_int_vertices<vertex_t, false>(
          *handle_,
          d_mg_aggregate_start_vertices.data(),
          d_mg_aggregate_start_vertices.size(),
          d_mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});

        cugraph::unrenumber_int_vertices<vertex_t, false>(
          *handle_,
          d_mg_aggregate_nbrs.data(),
          d_mg_aggregate_nbrs.size(),
          d_mg_aggregate_renumber_map_labels.data(),
          std::vector<vertex_t>{mg_graph_view.number_of_vertices()});

        // 3-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
        std::tie(sg_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            *handle_, input_usecase, false, false);

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        // 3-4. run SG K-hop neighbors

        auto [d_sg_offsets, d_sg_nbrs] = cugraph::k_hop_nbrs(
          *handle_,
          sg_graph_view,
          raft::device_span<vertex_t const>(d_mg_aggregate_start_vertices.data(),
                                            d_mg_aggregate_start_vertices.size()),
          k_hop_nbrs_usecase.k);

        // 3-5. compare

        auto h_sg_offsets = cugraph::test::to_host(*handle_, d_sg_offsets);
        auto h_sg_nbrs    = cugraph::test::to_host(*handle_, d_sg_nbrs);

        auto h_mg_aggregate_counts = cugraph::test::to_host(*handle_, d_mg_aggregate_counts);
        std::vector<size_t> h_mg_aggregate_offsets(h_mg_aggregate_counts.size() + 1, 0);
        std::inclusive_scan(h_mg_aggregate_counts.begin(),
                            h_mg_aggregate_counts.end(),
                            h_mg_aggregate_offsets.begin() + 1);
        auto h_mg_aggregate_nbrs = cugraph::test::to_host(*handle_, d_mg_aggregate_nbrs);

        ASSERT_TRUE(std::equal(
          h_mg_aggregate_offsets.begin(), h_mg_aggregate_offsets.end(), h_sg_offsets.begin()))
          << "MG & SG offsets do not match.";

        for (size_t i = 0; i < d_mg_aggregate_start_vertices.size(); ++i) {
          std::sort(h_sg_nbrs.begin() + h_sg_offsets[i], h_sg_nbrs.begin() + h_sg_offsets[i + 1]);
          std::sort(h_mg_aggregate_nbrs.begin() + h_mg_aggregate_offsets[i],
                    h_mg_aggregate_nbrs.begin() + h_mg_aggregate_offsets[i + 1]);
        }

        ASSERT_TRUE(
          std::equal(h_mg_aggregate_nbrs.begin(), h_mg_aggregate_nbrs.end(), h_sg_nbrs.begin()))
          << "MG & SG neighbors do not match.";
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGKHopNbrs<input_usecase_t>::handle_ = nullptr;

using Tests_MGKHopNbrs_File = Tests_MGKHopNbrs<cugraph::test::File_Usecase>;
using Tests_MGKHopNbrs_Rmat = Tests_MGKHopNbrs<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGKHopNbrs_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGKHopNbrs_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGKHopNbrs_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGKHopNbrs_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGKHopNbrs_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KHopNbrs_Usecase{1024, 2}, KHopNbrs_Usecase{1024, 1}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGKHopNbrs_Rmat,
                         ::testing::Values(
                           // enable correctness checks
                           std::make_tuple(KHopNbrs_Usecase{1024, 2},
                                           cugraph::test::Rmat_Usecase(
                                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGKHopNbrs_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(
      KHopNbrs_Usecase{4, 2, false},
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
