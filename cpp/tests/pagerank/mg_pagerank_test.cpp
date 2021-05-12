/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/experimental/graph.hpp>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/experimental/graph_view.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 0;

struct PageRank_Usecase {
  double personalization_ratio{0.0};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPageRank
  : public ::testing::TestWithParam<std::tuple<PageRank_Usecase, input_usecase_t>> {
 public:
  Tests_MGPageRank() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running PageRank on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(PageRank_Usecase const& pagerank_usecase,
                        input_usecase_t const& input_usecase)
  {
    // 1. initialize handle
    raft::handle_t handle{};
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) { --row_comm_size; }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create MG graph

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, true> mg_graph(handle);
    rmm::device_uvector<vertex_t> d_mg_renumber_map_labels(0, handle.get_stream());
    std::tie(mg_graph, d_mg_renumber_map_labels) =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, true, true>(handle, true);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. generate personalization vertex/value pairs

    std::vector<vertex_t> h_mg_personalization_vertices{};
    std::vector<result_t> h_mg_personalization_values{};
    if (pagerank_usecase.personalization_ratio > 0.0) {
      std::default_random_engine generator{
        static_cast<long unsigned int>(comm.get_rank()) /* seed */};
      std::uniform_real_distribution<double> distribution{0.0, 1.0};
      h_mg_personalization_vertices.resize(mg_graph_view.get_number_of_local_vertices());
      std::iota(h_mg_personalization_vertices.begin(),
                h_mg_personalization_vertices.end(),
                mg_graph_view.get_local_vertex_first());
      h_mg_personalization_vertices.erase(
        std::remove_if(h_mg_personalization_vertices.begin(),
                       h_mg_personalization_vertices.end(),
                       [&generator, &distribution, pagerank_usecase](auto v) {
                         return distribution(generator) >= pagerank_usecase.personalization_ratio;
                       }),
        h_mg_personalization_vertices.end());
      h_mg_personalization_values.resize(h_mg_personalization_vertices.size());
      std::for_each(h_mg_personalization_values.begin(),
                    h_mg_personalization_values.end(),
                    [&distribution, &generator](auto& val) { val = distribution(generator); });
    }

    rmm::device_uvector<vertex_t> d_mg_personalization_vertices(
      h_mg_personalization_vertices.size(), handle.get_stream());
    rmm::device_uvector<result_t> d_mg_personalization_values(d_mg_personalization_vertices.size(),
                                                              handle.get_stream());
    if (d_mg_personalization_vertices.size() > 0) {
      raft::update_device(d_mg_personalization_vertices.data(),
                          h_mg_personalization_vertices.data(),
                          h_mg_personalization_vertices.size(),
                          handle.get_stream());
      raft::update_device(d_mg_personalization_values.data(),
                          h_mg_personalization_values.data(),
                          h_mg_personalization_values.size(),
                          handle.get_stream());
    }

    // 4. run MG PageRank

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    rmm::device_uvector<result_t> d_mg_pageranks(mg_graph_view.get_number_of_local_vertices(),
                                                 handle.get_stream());

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    cugraph::experimental::pagerank(handle,
                                    mg_graph_view,
                                    static_cast<weight_t*>(nullptr),
                                    d_mg_personalization_vertices.data(),
                                    d_mg_personalization_values.data(),
                                    static_cast<vertex_t>(d_mg_personalization_vertices.size()),
                                    d_mg_pageranks.data(),
                                    alpha,
                                    epsilon,
                                    std::numeric_limits<size_t>::max(),
                                    false);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG PageRank took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 5. copmare SG & MG results

    if (pagerank_usecase.check_correctness) {
      // 5-1. create SG graph

      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, false> sg_graph(handle);
      std::tie(sg_graph, std::ignore) =
        input_usecase.template construct_graph<vertex_t, edge_t, weight_t, true, false>(
          handle, true, false);

      auto sg_graph_view = sg_graph.view();

      // 5-2. collect personalization vertex/value pairs

      rmm::device_uvector<vertex_t> d_sg_personalization_vertices(0, handle.get_stream());
      rmm::device_uvector<result_t> d_sg_personalization_values(0, handle.get_stream());
      if (pagerank_usecase.personalization_ratio > 0.0) {
        rmm::device_uvector<vertex_t> d_unrenumbered_personalization_vertices(
          d_mg_personalization_vertices.size(), handle.get_stream());
        rmm::device_uvector<result_t> d_unrenumbered_personalization_values(
          d_unrenumbered_personalization_vertices.size(), handle.get_stream());
        raft::copy_async(d_unrenumbered_personalization_vertices.data(),
                         d_mg_personalization_vertices.data(),
                         d_mg_personalization_vertices.size(),
                         handle.get_stream());
        raft::copy_async(d_unrenumbered_personalization_values.data(),
                         d_mg_personalization_values.data(),
                         d_mg_personalization_values.size(),
                         handle.get_stream());

        std::vector<vertex_t> vertex_partition_lasts(comm_size);
        for (size_t i = 0; i < vertex_partition_lasts.size(); ++i) {
          vertex_partition_lasts[i] = mg_graph_view.get_vertex_partition_last(i);
        }
        cugraph::experimental::unrenumber_int_vertices<vertex_t, true>(
          handle,
          d_unrenumbered_personalization_vertices.data(),
          d_unrenumbered_personalization_vertices.size(),
          d_mg_renumber_map_labels.data(),
          mg_graph_view.get_local_vertex_first(),
          mg_graph_view.get_local_vertex_last(),
          vertex_partition_lasts,
          handle.get_stream());

        rmm::device_scalar<size_t> d_local_personalization_vector_size(
          d_unrenumbered_personalization_vertices.size(), handle.get_stream());
        rmm::device_uvector<size_t> d_recvcounts(comm_size, handle.get_stream());
        comm.allgather(
          d_local_personalization_vector_size.data(), d_recvcounts.data(), 1, handle.get_stream());
        std::vector<size_t> recvcounts(d_recvcounts.size());
        raft::update_host(
          recvcounts.data(), d_recvcounts.data(), d_recvcounts.size(), handle.get_stream());
        auto status = comm.sync_stream(handle.get_stream());
        ASSERT_EQ(status, raft::comms::status_t::SUCCESS);

        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);

        d_sg_personalization_vertices.resize(displacements.back() + recvcounts.back(),
                                             handle.get_stream());
        d_sg_personalization_values.resize(d_sg_personalization_vertices.size(),
                                           handle.get_stream());

        comm.allgatherv(d_unrenumbered_personalization_vertices.data(),
                        d_sg_personalization_vertices.data(),
                        recvcounts.data(),
                        displacements.data(),
                        handle.get_stream());
        comm.allgatherv(d_unrenumbered_personalization_values.data(),
                        d_sg_personalization_values.data(),
                        recvcounts.data(),
                        displacements.data(),
                        handle.get_stream());

        cugraph::test::sort_by_key(handle,
                                   d_unrenumbered_personalization_vertices.data(),
                                   d_unrenumbered_personalization_values.data(),
                                   d_unrenumbered_personalization_vertices.size());
      }

      // 5-3. run SG PageRank

      rmm::device_uvector<result_t> d_sg_pageranks(sg_graph_view.get_number_of_vertices(),
                                                   handle.get_stream());

      cugraph::experimental::pagerank(handle,
                                      sg_graph_view,
                                      static_cast<weight_t*>(nullptr),
                                      d_sg_personalization_vertices.data(),
                                      d_sg_personalization_values.data(),
                                      static_cast<vertex_t>(d_sg_personalization_vertices.size()),
                                      d_sg_pageranks.data(),
                                      alpha,
                                      epsilon,
                                      std::numeric_limits<size_t>::max(),  // max_iterations
                                      false);

      // 5-4. compare

      std::vector<result_t> h_sg_pageranks(sg_graph_view.get_number_of_vertices());
      raft::update_host(
        h_sg_pageranks.data(), d_sg_pageranks.data(), d_sg_pageranks.size(), handle.get_stream());

      std::vector<result_t> h_mg_pageranks(mg_graph_view.get_number_of_local_vertices());
      raft::update_host(
        h_mg_pageranks.data(), d_mg_pageranks.data(), d_mg_pageranks.size(), handle.get_stream());

      std::vector<vertex_t> h_mg_renumber_map_labels(d_mg_renumber_map_labels.size());
      raft::update_host(h_mg_renumber_map_labels.data(),
                        d_mg_renumber_map_labels.data(),
                        d_mg_renumber_map_labels.size(),
                        handle.get_stream());

      handle.get_stream_view().synchronize();

      auto threshold_ratio = 1e-3;
      auto threshold_magnitude =
        (1.0 / static_cast<result_t>(mg_graph_view.get_number_of_vertices())) *
        threshold_ratio;  // skip comparison for low PageRank verties (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        return std::abs(lhs - rhs) <
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
      };

      for (vertex_t i = 0; i < mg_graph_view.get_number_of_local_vertices(); ++i) {
        auto mapped_vertex = h_mg_renumber_map_labels[i];
        ASSERT_TRUE(nearly_equal(h_mg_pageranks[i], h_sg_pageranks[mapped_vertex]))
          << "MG PageRank value for vertex: " << mapped_vertex << " in rank: " << comm_rank
          << " has value: " << h_mg_pageranks[i]
          << " which exceeds the error margin for comparing to SG value: "
          << h_sg_pageranks[mapped_vertex];
      }
    }
  }
};

using Tests_MGPageRank_File = Tests_MGPageRank<cugraph::test::File_Usecase>;
using Tests_MGPageRank_Rmat = Tests_MGPageRank<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPageRank_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPageRank_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGPageRank_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(PageRank_Usecase{0.0, false},
                      PageRank_Usecase{0.5, false},
                      PageRank_Usecase{0.0, true},
                      PageRank_Usecase{0.5, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_tests,
                         Tests_MGPageRank_Rmat,
                         ::testing::Combine(::testing::Values(PageRank_Usecase{0.0, false},
                                                              PageRank_Usecase{0.5, false},
                                                              PageRank_Usecase{0.0, true},
                                                              PageRank_Usecase{0.5, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false, true))));

INSTANTIATE_TEST_SUITE_P(rmat_large_tests,
                         Tests_MGPageRank_Rmat,
                         ::testing::Combine(::testing::Values(PageRank_Usecase{0.0, false, false},
                                                              PageRank_Usecase{0.5, false, false},
                                                              PageRank_Usecase{0.0, true, false},
                                                              PageRank_Usecase{0.5, true, false}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              20, 32, 0.57, 0.19, 0.19, 0, false, false, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
