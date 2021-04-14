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

#include <algorithms.hpp>
#include <partition_manager.hpp>

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

typedef struct KatzCentrality_Usecase_t {
  cugraph::test::input_graph_specifier_t input_graph_specifier{};

  bool test_weighted{false};
  bool check_correctness{false};

  KatzCentrality_Usecase_t(std::string const& graph_file_path,
                           bool test_weighted,
                           bool check_correctness = true)
    : test_weighted(test_weighted), check_correctness(check_correctness)
  {
    std::string graph_file_full_path{};
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
    input_graph_specifier.tag = cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH;
    input_graph_specifier.graph_file_full_path = graph_file_full_path;
  };

  KatzCentrality_Usecase_t(cugraph::test::rmat_params_t rmat_params,
                           bool test_weighted,
                           bool check_correctness = true)
    : test_weighted(test_weighted), check_correctness(check_correctness)
  {
    input_graph_specifier.tag         = cugraph::test::input_graph_specifier_t::RMAT_PARAMS;
    input_graph_specifier.rmat_params = rmat_params;
  }
} KatzCentrality_Usecase;

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, multi_gpu>,
           rmm::device_uvector<vertex_t>>
read_graph(raft::handle_t const& handle, KatzCentrality_Usecase const& configuration, bool renumber)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();
  auto const comm_rank = comm.get_rank();

  std::vector<size_t> partition_ids(multi_gpu ? size_t{1} : static_cast<size_t>(comm_size));
  std::iota(partition_ids.begin(),
            partition_ids.end(),
            multi_gpu ? static_cast<size_t>(comm_rank) : size_t{0});

  return configuration.input_graph_specifier.tag ==
             cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH
           ? cugraph::test::
               read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, true, multi_gpu>(
                 handle,
                 configuration.input_graph_specifier.graph_file_full_path,
                 configuration.test_weighted,
                 renumber)
           : cugraph::test::
               generate_graph_from_rmat_params<vertex_t, edge_t, weight_t, true, multi_gpu>(
                 handle,
                 configuration.input_graph_specifier.rmat_params.scale,
                 configuration.input_graph_specifier.rmat_params.edge_factor,
                 configuration.input_graph_specifier.rmat_params.a,
                 configuration.input_graph_specifier.rmat_params.b,
                 configuration.input_graph_specifier.rmat_params.c,
                 configuration.input_graph_specifier.rmat_params.seed,
                 configuration.input_graph_specifier.rmat_params.undirected,
                 configuration.input_graph_specifier.rmat_params.scramble_vertex_ids,
                 configuration.test_weighted,
                 renumber,
                 partition_ids,
                 static_cast<size_t>(comm_size));
}

class Tests_MGKatzCentrality : public ::testing::TestWithParam<KatzCentrality_Usecase> {
 public:
  Tests_MGKatzCentrality() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running Katz Centrality on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(KatzCentrality_Usecase const& configuration)
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
      read_graph<vertex_t, edge_t, weight_t, true>(handle, configuration, true);
    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG read_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. compute max in-degree

    auto max_in_degree = mg_graph_view.compute_max_in_degree(handle);

    // 4. run MG Katz Centrality

    result_t const alpha = result_t{1.0} / static_cast<result_t>(max_in_degree + 1);
    result_t constexpr beta{1.0};
    result_t constexpr epsilon{1e-6};

    rmm::device_uvector<result_t> d_mg_katz_centralities(
      mg_graph_view.get_number_of_local_vertices(), handle.get_stream());

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    cugraph::experimental::katz_centrality(handle,
                                           mg_graph_view,
                                           static_cast<result_t*>(nullptr),
                                           d_mg_katz_centralities.data(),
                                           alpha,
                                           beta,
                                           epsilon,
                                           std::numeric_limits<size_t>::max(),
                                           false);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG Katz Centrality took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 5. copmare SG & MG results

    if (configuration.check_correctness) {
      // 5-1. create SG graph

      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, false> sg_graph(handle);
      std::tie(sg_graph, std::ignore) =
        read_graph<vertex_t, edge_t, weight_t, false>(handle, configuration, false);

      auto sg_graph_view = sg_graph.view();

      // 5-3. run SG Katz Centrality

      rmm::device_uvector<result_t> d_sg_katz_centralities(sg_graph_view.get_number_of_vertices(),
                                                           handle.get_stream());

      cugraph::experimental::katz_centrality(handle,
                                             sg_graph_view,
                                             static_cast<result_t*>(nullptr),
                                             d_sg_katz_centralities.data(),
                                             alpha,
                                             beta,
                                             epsilon,
                                             std::numeric_limits<size_t>::max(),  // max_iterations
                                             false);

      // 5-4. compare

      std::vector<result_t> h_sg_katz_centralities(sg_graph_view.get_number_of_vertices());
      raft::update_host(h_sg_katz_centralities.data(),
                        d_sg_katz_centralities.data(),
                        d_sg_katz_centralities.size(),
                        handle.get_stream());

      std::vector<result_t> h_mg_katz_centralities(mg_graph_view.get_number_of_local_vertices());
      raft::update_host(h_mg_katz_centralities.data(),
                        d_mg_katz_centralities.data(),
                        d_mg_katz_centralities.size(),
                        handle.get_stream());

      std::vector<vertex_t> h_mg_renumber_map_labels(d_mg_renumber_map_labels.size());
      raft::update_host(h_mg_renumber_map_labels.data(),
                        d_mg_renumber_map_labels.data(),
                        d_mg_renumber_map_labels.size(),
                        handle.get_stream());

      handle.get_stream_view().synchronize();

      auto threshold_ratio = 1e-3;
      auto threshold_magnitude =
        (1.0 / static_cast<result_t>(mg_graph_view.get_number_of_vertices())) *
        threshold_ratio;  // skip comparison for low KatzCentrality verties (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
        return std::abs(lhs - rhs) <
               std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
      };

      for (vertex_t i = 0; i < mg_graph_view.get_number_of_local_vertices(); ++i) {
        auto mapped_vertex = h_mg_renumber_map_labels[i];
        ASSERT_TRUE(nearly_equal(h_mg_katz_centralities[i], h_sg_katz_centralities[mapped_vertex]))
          << "MG KatzCentrality value for vertex: " << mapped_vertex << " in rank: " << comm_rank
          << " has value: " << h_mg_katz_centralities[i]
          << " which exceeds the error margin for comparing to SG value: "
          << h_sg_katz_centralities[mapped_vertex];
      }
    }
  }
};

TEST_P(Tests_MGKatzCentrality, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_MGKatzCentrality,
  ::testing::Values(
    // enable correctness checks
    KatzCentrality_Usecase("test/datasets/karate.mtx", false),
    KatzCentrality_Usecase("test/datasets/karate.mtx", true),
    KatzCentrality_Usecase("test/datasets/web-Google.mtx", false),
    KatzCentrality_Usecase("test/datasets/web-Google.mtx", true),
    KatzCentrality_Usecase("test/datasets/ljournal-2008.mtx", false),
    KatzCentrality_Usecase("test/datasets/ljournal-2008.mtx", true),
    KatzCentrality_Usecase("test/datasets/webbase-1M.mtx", false),
    KatzCentrality_Usecase("test/datasets/webbase-1M.mtx", true),
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                           false),
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                           true),
    // disable correctness checks for large graphs
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{20, 32, 0.57, 0.19, 0.19, 0, false, false},
                           false,
                           false),
    KatzCentrality_Usecase(cugraph::test::rmat_params_t{20, 32, 0.57, 0.19, 0.19, 0, false, false},
                           true,
                           false)));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
