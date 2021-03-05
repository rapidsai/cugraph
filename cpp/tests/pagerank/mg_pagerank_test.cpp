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

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>

#include <gtest/gtest.h>

#include <random>

typedef struct PageRank_Usecase_t {
  cugraph::test::input_graph_specifier_t input_graph_specifier{};

  double personalization_ratio{0.0};
  bool test_weighted{false};

  PageRank_Usecase_t(std::string const& graph_file_path,
                     double personalization_ratio,
                     bool test_weighted)
    : personalization_ratio(personalization_ratio), test_weighted(test_weighted)
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

  PageRank_Usecase_t(cugraph::test::rmat_params_t rmat_params,
                     double personalization_ratio,
                     bool test_weighted)
    : personalization_ratio(personalization_ratio), test_weighted(test_weighted)
  {
    input_graph_specifier.tag         = cugraph::test::input_graph_specifier_t::RMAT_PARAMS;
    input_graph_specifier.rmat_params = rmat_params;
  }
} PageRank_Usecase;

class Tests_MGPageRank : public ::testing::TestWithParam<PageRank_Usecase> {
 public:
  Tests_MGPageRank() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running pagerank on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(PageRank_Usecase const& configuration)
  {
    // 1. initialize handle

    raft::handle_t handle{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) { --row_comm_size; }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create SG & MG graphs

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, false> sg_graph(handle);
    rmm::device_uvector<vertex_t> d_sg_renumber_map_labels(0, handle.get_stream());
    std::tie(sg_graph, d_sg_renumber_map_labels) =
      configuration.input_graph_specifier.tag ==
          cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH
        ? cugraph::test::
            read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, true, false>(
              handle,
              configuration.input_graph_specifier.graph_file_full_path,
              configuration.test_weighted,
              true)
        : cugraph::test::generate_graph_from_rmat_params<vertex_t, edge_t, weight_t, true, false>(
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
            true);

    auto sg_graph_view = sg_graph.view();

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, true> mg_graph(handle);
    rmm::device_uvector<vertex_t> d_mg_renumber_map_labels(0, handle.get_stream());
    std::tie(mg_graph, d_mg_renumber_map_labels) =
      configuration.input_graph_specifier.tag ==
          cugraph::test::input_graph_specifier_t::MATRIX_MARKET_FILE_PATH
        ? cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, true, true>(
            handle,
            configuration.input_graph_specifier.graph_file_full_path,
            configuration.test_weighted,
            true)
        : cugraph::test::generate_graph_from_rmat_params<vertex_t, edge_t, weight_t, true, true>(
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
            true);

    auto mg_graph_view = mg_graph.view();

    std::vector<vertex_t> h_sg_renumber_map_labels(d_sg_renumber_map_labels.size());
    raft::update_host(h_sg_renumber_map_labels.data(),
                      d_sg_renumber_map_labels.data(),
                      d_sg_renumber_map_labels.size(),
                      handle.get_stream());

    std::vector<vertex_t> h_mg_renumber_map_labels(mg_graph_view.get_number_of_local_vertices());
    raft::update_host(h_mg_renumber_map_labels.data(),
                      d_mg_renumber_map_labels.data(),
                      d_mg_renumber_map_labels.size(),
                      handle.get_stream());

    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    // 2. generate personalization vertex/value pairs

    std::vector<vertex_t> h_personalization_vertices{};
    std::vector<result_t> h_personalization_values{};
    if (configuration.personalization_ratio > 0.0) {
      std::default_random_engine generator{};
      std::uniform_real_distribution<double> distribution{0.0, 1.0};
      h_personalization_vertices.resize(sg_graph_view.get_number_of_vertices());
      std::iota(h_personalization_vertices.begin(), h_personalization_vertices.end(), vertex_t{0});
      h_personalization_vertices.erase(
        std::remove_if(h_personalization_vertices.begin(),
                       h_personalization_vertices.end(),
                       [&generator, &distribution, configuration](auto v) {
                         return distribution(generator) >= configuration.personalization_ratio;
                       }),
        h_personalization_vertices.end());
      h_personalization_values.resize(h_personalization_vertices.size());
      std::for_each(h_personalization_values.begin(),
                    h_personalization_values.end(),
                    [&distribution, &generator](auto& val) { val = distribution(generator); });
    }

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    // 3. run SG pagerank

    std::vector<vertex_t> h_sg_personalization_vertices{};
    std::vector<result_t> h_sg_personalization_values{};
    if (h_personalization_vertices.size() > 0) {
      for (vertex_t i = 0; i < sg_graph_view.get_number_of_vertices(); ++i) {
        auto it = std::lower_bound(h_personalization_vertices.begin(),
                                   h_personalization_vertices.end(),
                                   h_sg_renumber_map_labels[i]);
        if (*it == h_sg_renumber_map_labels[i]) {
          h_sg_personalization_vertices.push_back(i);
          h_sg_personalization_values.push_back(
            h_personalization_values[std::distance(h_personalization_vertices.begin(), it)]);
        }
      }
    }

    rmm::device_uvector<vertex_t> d_sg_personalization_vertices(
      h_sg_personalization_vertices.size(), handle.get_stream());
    rmm::device_uvector<result_t> d_sg_personalization_values(d_sg_personalization_vertices.size(),
                                                              handle.get_stream());
    if (d_sg_personalization_vertices.size() > 0) {
      raft::update_device(d_sg_personalization_vertices.data(),
                          h_sg_personalization_vertices.data(),
                          h_sg_personalization_vertices.size(),
                          handle.get_stream());
      raft::update_device(d_sg_personalization_values.data(),
                          h_sg_personalization_values.data(),
                          h_sg_personalization_values.size(),
                          handle.get_stream());
    }

    rmm::device_uvector<result_t> d_sg_pageranks(sg_graph_view.get_number_of_vertices(),
                                                 handle.get_stream());

    cugraph::experimental::pagerank(handle,
                                    sg_graph_view,
                                    static_cast<weight_t*>(nullptr),
                                    d_sg_personalization_vertices.data(),
                                    d_sg_personalization_values.data(),
                                    static_cast<vertex_t>(d_sg_personalization_vertices.size()),
                                    d_sg_pageranks.begin(),
                                    alpha,
                                    epsilon,
                                    std::numeric_limits<size_t>::max(),  // max_iterations
                                    false,
                                    false);

    std::vector<result_t> h_sg_pageranks(sg_graph_view.get_number_of_vertices());
    raft::update_host(
      h_sg_pageranks.data(), d_sg_pageranks.data(), d_sg_pageranks.size(), handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    // 4. run MG pagerank

    std::vector<vertex_t> h_mg_personalization_vertices{};
    std::vector<result_t> h_mg_personalization_values{};
    if (h_personalization_vertices.size() > 0) {
      for (vertex_t i = 0; i < mg_graph_view.get_number_of_local_vertices(); ++i) {
        auto it = std::lower_bound(h_personalization_vertices.begin(),
                                   h_personalization_vertices.end(),
                                   h_mg_renumber_map_labels[i]);
        if (*it == h_mg_renumber_map_labels[i]) {
          h_mg_personalization_vertices.push_back(mg_graph_view.get_local_vertex_first() + i);
          h_mg_personalization_values.push_back(
            h_personalization_values[std::distance(h_personalization_vertices.begin(), it)]);
        }
      }
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

    rmm::device_uvector<result_t> d_mg_pageranks(mg_graph_view.get_number_of_local_vertices(),
                                                 handle.get_stream());

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    cugraph::experimental::pagerank(handle,
                                    mg_graph_view,
                                    static_cast<weight_t*>(nullptr),
                                    d_mg_personalization_vertices.data(),
                                    d_mg_personalization_values.data(),
                                    static_cast<vertex_t>(d_mg_personalization_vertices.size()),
                                    d_mg_pageranks.begin(),
                                    alpha,
                                    epsilon,
                                    std::numeric_limits<size_t>::max(),
                                    false,
                                    false);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<result_t> h_mg_pageranks(mg_graph_view.get_number_of_local_vertices());
    raft::update_host(
      h_mg_pageranks.data(), d_mg_pageranks.data(), d_mg_pageranks.size(), handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    // 5. copmare SG & MG results

    std::vector<result_t> h_sg_shuffled_pageranks(sg_graph_view.get_number_of_vertices(),
                                                  result_t{0.0});
    for (size_t i = 0; i < h_sg_pageranks.size(); ++i) {
      h_sg_shuffled_pageranks[h_sg_renumber_map_labels[i]] = h_sg_pageranks[i];
    }

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
      ASSERT_TRUE(nearly_equal(h_mg_pageranks[i], h_sg_shuffled_pageranks[mapped_vertex]))
        << "MG PageRank value for vertex: " << i << " in rank: " << comm_rank
        << " has value: " << h_mg_pageranks[i]
        << " which exceeds the error margin for comparing to SG value: "
        << h_sg_shuffled_pageranks[mapped_vertex];
    }
  }
};

TEST_P(Tests_MGPageRank, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_MGPageRank,
  ::testing::Values(
    PageRank_Usecase("test/datasets/karate.mtx", 0.0, false),
    PageRank_Usecase("test/datasets/karate.mtx", 0.5, false),
    PageRank_Usecase("test/datasets/karate.mtx", 0.0, true),
    PageRank_Usecase("test/datasets/karate.mtx", 0.5, true),
    PageRank_Usecase("test/datasets/web-Google.mtx", 0.0, false),
    PageRank_Usecase("test/datasets/web-Google.mtx", 0.5, false),
    PageRank_Usecase("test/datasets/web-Google.mtx", 0.0, true),
    PageRank_Usecase("test/datasets/web-Google.mtx", 0.5, true),
    PageRank_Usecase("test/datasets/ljournal-2008.mtx", 0.0, false),
    PageRank_Usecase("test/datasets/ljournal-2008.mtx", 0.5, false),
    PageRank_Usecase("test/datasets/ljournal-2008.mtx", 0.0, true),
    PageRank_Usecase("test/datasets/ljournal-2008.mtx", 0.5, true),
    PageRank_Usecase("test/datasets/webbase-1M.mtx", 0.0, false),
    PageRank_Usecase("test/datasets/webbase-1M.mtx", 0.5, false),
    PageRank_Usecase("test/datasets/webbase-1M.mtx", 0.0, true),
    PageRank_Usecase("test/datasets/webbase-1M.mtx", 0.5, true),
    PageRank_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                     0.0,
                     false),
    PageRank_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                     0.5,
                     false),
    PageRank_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                     0.0,
                     true),
    PageRank_Usecase(cugraph::test::rmat_params_t{10, 16, 0.57, 0.19, 0.19, 0, false, false},
                     0.5,
                     true)));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
