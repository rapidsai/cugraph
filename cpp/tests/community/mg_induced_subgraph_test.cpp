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

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

typedef struct InducedSubgraph_Usecase_t {
  std::string graph_file_full_path{};
  std::vector<size_t> subgraph_sizes{};
  bool test_weighted{false};

  InducedSubgraph_Usecase_t(std::string const& graph_file_path,
                            std::vector<size_t> const& subgraph_sizes,
                            bool test_weighted)
    : subgraph_sizes(subgraph_sizes), test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} InducedSubgraph_Usecase;

class Tests_MGInducedSubgraph: public ::testing::TestWithParam<InducedSubgraph_Usecase> { 
 public:
  Tests_MGInducedSubgraph() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running TriangleCount on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(InducedSubgraph_Usecase const& configuration)
  {
    using weight_t = float;

    HighResClock hr_clock{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, true> mg_graph(*handle_);
    std::tie(mg_graph, std::ignore) = cugraph::test::
      read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, configuration.graph_file_full_path, configuration.test_weighted, false);
    auto mg_graph_view = mg_graph.view();
    /*auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true, false, true);*/

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }


    // 2. sort vertices of each subgraph
    std::vector<edge_t> h_offsets(mg_graph_view.number_of_vertices() + 1);
    std::vector<vertex_t> h_indices(graph_view.number_of_edges());
    auto h_weights = graph_view.is_weighted() ? std::make_optional<std::vector<weight_t>>(
                                                  graph_view.number_of_edges(), weight_t{0.0})
                                              : std::nullopt;
    raft::update_host(h_offsets.data(),
                      graph_view.local_edge_partition_view().offsets(),
                      graph_view.number_of_vertices() + 1,
                      handle_.get_stream());
    raft::update_host(h_indices.data(),
                      graph_view.local_edge_partition_view().indices(),
                      graph_view.number_of_edges(),
                      handle_.get_stream());
    if (h_weights) {
      raft::update_host((*h_weights).data(),
                        *(graph_view.local_edge_partition_view().weights()),
                        graph_view.number_of_edges(),
                        handle_.get_stream());
    }
    handle_.sync_stream();

    std::vector<size_t> h_subgraph_offsets(configuration.subgraph_sizes.size() + 1, 0);
    std::partial_sum(configuration.subgraph_sizes.begin(),
                     configuration.subgraph_sizes.end(),
                     h_subgraph_offsets.begin() + 1);
    std::vector<vertex_t> h_subgraph_vertices(h_subgraph_offsets.back(),
                                              cugraph::invalid_vertex_id<vertex_t>::value);
    std::default_random_engine generator{};
    std::uniform_int_distribution<vertex_t> distribution{0, graph_view.number_of_vertices() - 1};

    for (size_t i = 0; i < configuration.subgraph_sizes.size(); ++i) {
      auto start = h_subgraph_offsets[i];
      auto last  = h_subgraph_offsets[i + 1];
      ASSERT_TRUE(last - start <= graph_view.number_of_vertices()) << "Invalid subgraph size.";
      // this is inefficient if last - start << graph_view.number_of_vertices() but this is for
      // the test puspose only and the time & memory cost is only linear to
      // graph_view.number_of_vertices(), so this may not matter.
      std::vector<vertex_t> vertices(graph_view.number_of_vertices());
      std::iota(vertices.begin(), vertices.end(), vertex_t{0});
      std::random_shuffle(vertices.begin(), vertices.end());
      std::copy(
        vertices.begin(), vertices.begin() + (last - start), h_subgraph_vertices.begin() + start);
      std::sort(h_subgraph_vertices.begin() + start, h_subgraph_vertices.begin() + last);
    }


    // 3. run MG InducedSubgraph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }
/*
    cugraph::triangle_count<vertex_t, edge_t, weight_t, true>(
      *handle_,
      mg_graph_view,
      d_mg_vertices ? std::make_optional<raft::device_span<vertex_t const>>(
                        (*d_mg_vertices).begin(), (*d_mg_vertices).end())
                    : std::nullopt,
      raft::device_span<edge_t>(d_mg_triangle_counts.begin(), d_mg_triangle_counts.end()),
      false);*/
    rmm::device_uvector<size_t> d_subgraph_offsets(h_subgraph_offsets.size(), handle_.get_stream());
    rmm::device_uvector<vertex_t> d_subgraph_vertices(h_subgraph_vertices.size(),
                                                      handle_.get_stream());
    raft::update_device(d_subgraph_offsets.data(),
                        h_subgraph_offsets.data(),
                        h_subgraph_offsets.size(),
                        handle_.get_stream());
    raft::update_device(d_subgraph_vertices.data(),
                        h_subgraph_vertices.data(),
                        h_subgraph_vertices.size(),
                        handle_.get_stream());

    auto [h_reference_subgraph_edgelist_majors,
          h_reference_subgraph_edgelist_minors,
          h_reference_subgraph_edgelist_weights,
          h_reference_subgraph_edge_offsets] =
    cugraph::extract_induced_subgraphs(
      *handle_,
      graph_view,
      raft::device_span<size_t const>(d_subgraph_offsets.data(), d_subgraph_offsets.size()),
      raft::device_span<vertex_t const>(d_subgraph_vertices.data(), d_subgraph_vertices.size()),
      configuration.subgraph_sizes.size(),
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG InducedSubgraph took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 4. copmare SG & MG results

    if (triangle_count_usecase.check_correctness) {
      // 4-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_vertices =
        d_mg_vertices ? std::optional<rmm::device_uvector<vertex_t>>{cugraph::test::device_gatherv(
                          *handle_, (*d_mg_vertices).data(), (*d_mg_vertices).size())}
                      : std::nullopt;
      auto d_mg_aggregate_triangle_counts = cugraph::test::device_gatherv(
        *handle_, d_mg_triangle_counts.data(), d_mg_triangle_counts.size());

      if (handle_->get_comms().get_rank() == int{0}) {
        // 4-2. unrenumbr MG results

        if (d_mg_aggregate_vertices) {
          cugraph::unrenumber_int_vertices<vertex_t, false>(
            *handle_,
            (*d_mg_aggregate_vertices).data(),
            (*d_mg_aggregate_vertices).size(),
            d_mg_aggregate_renumber_map_labels.data(),
            std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
          std::tie(d_mg_aggregate_vertices, d_mg_aggregate_triangle_counts) =
            cugraph::test::sort_by_key(
              *handle_, *d_mg_aggregate_vertices, d_mg_aggregate_triangle_counts);
        } else {
          std::tie(std::ignore, d_mg_aggregate_triangle_counts) = cugraph::test::sort_by_key(
            *handle_, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_triangle_counts);
        }

        // 4-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> sg_graph(*handle_);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            *handle_, input_usecase, false, false, false, true);

        auto sg_graph_view = sg_graph.view();

        ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

        // 4-4. run SG TriangleCount

        rmm::device_uvector<edge_t> d_sg_triangle_counts(d_mg_aggregate_vertices
                                                           ? (*d_mg_aggregate_vertices).size()
                                                           : sg_graph_view.number_of_vertices(),
                                                         handle_->get_stream());

        cugraph::triangle_count<vertex_t, edge_t, weight_t>(
          *handle_,
          sg_graph_view,
          d_mg_aggregate_vertices
            ? std::make_optional<raft::device_span<vertex_t const>>(
                (*d_mg_aggregate_vertices).begin(), (*d_mg_aggregate_vertices).end())
            : std::nullopt,
          raft::device_span<edge_t>(d_sg_triangle_counts.begin(), d_sg_triangle_counts.end()),
          false);

        // 4-5. compare

        std::vector<edge_t> h_mg_aggregate_triangle_counts(d_mg_aggregate_triangle_counts.size());
        raft::update_host(h_mg_aggregate_triangle_counts.data(),
                          d_mg_aggregate_triangle_counts.data(),
                          d_mg_aggregate_triangle_counts.size(),
                          handle_->get_stream());

        std::vector<edge_t> h_sg_triangle_counts(d_sg_triangle_counts.size());
        raft::update_host(h_sg_triangle_counts.data(),
                          d_sg_triangle_counts.data(),
                          d_sg_triangle_counts.size(),
                          handle_->get_stream());

        handle_->sync_stream();

        ASSERT_TRUE(std::equal(h_mg_aggregate_triangle_counts.begin(),
                               h_mg_aggregate_triangle_counts.end(),
                               h_sg_triangle_counts.begin()));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTriangleCount<input_usecase_t>::handle_ = nullptr;

using Tests_MGTriangleCount_File = Tests_MGTriangleCount<cugraph::test::File_Usecase>;
using Tests_MGTriangleCount_Rmat = Tests_MGTriangleCount<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTriangleCount_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTriangleCount_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTriangleCount_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTriangleCount_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGTriangleCount_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(TriangleCount_Usecase{0.1}, TriangleCount_Usecase{1.0}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_tests,
                         Tests_MGTriangleCount_Rmat,
                         ::testing::Combine(::testing::Values(TriangleCount_Usecase{0.1},
                                                              TriangleCount_Usecase{1.0}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTriangleCount_Rmat,
  ::testing::Combine(::testing::Values(TriangleCount_Usecase{0.1, false},
                                       TriangleCount_Usecase{1.0, false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
