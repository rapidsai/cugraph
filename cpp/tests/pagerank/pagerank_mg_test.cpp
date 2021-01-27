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

#include <random>

#include <gtest/gtest.h>

#include <raft/comms/mpi_comms.hpp>

#include <algorithms.hpp>
#include <partition_manager.hpp>

#include <utilities/test_utilities.hpp>
#include <utilities/mg_test_utilities.hpp>
#include <utilities/base_fixture.hpp>


////////////////////////////////////////////////////////////////////////////////
// Pagerank reference implementation
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void pagerank_reference(edge_t const* offsets,
                        vertex_t const* indices,
                        weight_t const* weights,
                        vertex_t const* personalization_vertices,
                        result_t const* personalization_values,
                        result_t* pageranks,
                        vertex_t num_vertices,
                        vertex_t personalization_vector_size,
                        result_t alpha,
                        result_t epsilon,
                        size_t max_iterations,
                        bool has_initial_guess)
{
  if (num_vertices == 0) { return; }

  if (has_initial_guess) {
    // use a double type counter (instead of result_t) to accumulate as std::accumulate is
    // inaccurate in adding a large number of comparably sized numbers. In C++17 or later,
    // std::reduce may be a better option.
    auto sum =
      static_cast<result_t>(std::accumulate(pageranks, pageranks + num_vertices, double{0.0}));
    ASSERT_TRUE(sum > 0.0);
    std::for_each(pageranks, pageranks + num_vertices, [sum](auto& val) { val /= sum; });
  } else {
    std::for_each(pageranks, pageranks + num_vertices, [num_vertices](auto& val) {
      val = result_t{1.0} / static_cast<result_t>(num_vertices);
    });
  }

  result_t personalization_sum{0.0};
  if (personalization_vertices != nullptr) {
    // use a double type counter (instead of result_t) to accumulate as std::accumulate is
    // inaccurate in adding a large number of comparably sized numbers. In C++17 or later,
    // std::reduce may be a better option.
    personalization_sum = static_cast<result_t>(std::accumulate(
      personalization_values, personalization_values + personalization_vector_size, double{0.0}));
    ASSERT_TRUE(personalization_sum > 0.0);
  }

  std::vector<weight_t> out_weight_sums(num_vertices, result_t{0.0});
  for (vertex_t i = 0; i < num_vertices; ++i) {
    for (auto j = *(offsets + i); j < *(offsets + i + 1); ++j) {
      auto nbr = indices[j];
      auto w   = weights != nullptr ? weights[j] : 1.0;
      out_weight_sums[nbr] += w;
    }
  }

  std::vector<result_t> old_pageranks(num_vertices, result_t{0.0});
  size_t iter{0};
  while (true) {
    std::copy(pageranks, pageranks + num_vertices, old_pageranks.begin());
    result_t dangling_sum{0.0};
    for (vertex_t i = 0; i < num_vertices; ++i) {
      if (out_weight_sums[i] == result_t{0.0}) { dangling_sum += old_pageranks[i]; }
    }
    for (vertex_t i = 0; i < num_vertices; ++i) {
      pageranks[i] = result_t{0.0};
      for (auto j = *(offsets + i); j < *(offsets + i + 1); ++j) {
        auto nbr = indices[j];
        auto w   = weights != nullptr ? weights[j] : result_t{1.0};
        pageranks[i] += alpha * old_pageranks[nbr] * (w / out_weight_sums[nbr]);
      }
      if (personalization_vertices == nullptr) {
        pageranks[i] +=
          (dangling_sum * alpha + (1.0 - alpha)) / static_cast<result_t>(num_vertices);
      }
    }
    if (personalization_vertices != nullptr) {
      for (vertex_t i = 0; i < personalization_vector_size; ++i) {
        auto v = personalization_vertices[i];
        pageranks[v] += (dangling_sum * alpha + (1.0 - alpha)) *
                        (personalization_values[i] / personalization_sum);
      }
    }
    result_t diff_sum{0.0};
    for (vertex_t i = 0; i < num_vertices; ++i) {
      diff_sum += std::abs(pageranks[i] - old_pageranks[i]);
    }
    if (diff_sum < epsilon) { break; }
    iter++;
    ASSERT_TRUE(iter < max_iterations);
  }

  return;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_CASE_P()
//
typedef struct Pagerank_Testparams_t {
  std::string graph_file_full_path{};
  double personalization_ratio{0.0};
  bool test_weighted{false};

  Pagerank_Testparams_t(std::string const& graph_file_path,
                        double personalization_ratio,
                        bool test_weighted)
    : personalization_ratio(personalization_ratio), test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} Pagerank_Testparams_t;


//
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
class Pagerank_E2E_MG_Testfixture_t : public ::testing::TestWithParam<Pagerank_Testparams_t> {
public:
   Pagerank_E2E_MG_Testfixture_t() {}

   // Run once for the entire fixture
   //
   // FIXME: consider a ::testing::Environment gtest instance instead, as done
   // by cuML in test_opg_utils.h
   static void SetUpTestCase() {
      MPI_Init(NULL, NULL);

      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      int nGpus;
      CUDA_CHECK(cudaGetDeviceCount(&nGpus));

      ASSERT(nGpus >= size,
             "Number of GPUs are lesser than MPI ranks! ngpus=%d, nranks=%d",
             nGpus, size);

      CUDA_CHECK(cudaSetDevice(rank));
   }

   static void TearDownTestCase() {
      MPI_Finalize();
   }

   // Run once for each test instance
   virtual void SetUp() {}
   virtual void TearDown() {}


   template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
   void run_test(const Pagerank_Testparams_t& param)
   {
    raft::handle_t handle;
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto &comm = handle.get_comms();

    cudaStream_t stream = handle.get_stream();

    // Assuming 2 GPUs which means 1 row, 2 cols. 2 cols = row_comm_size of 2.
    // FIXME: DO NOT ASSUME 2 GPUs, add code to compute prows, pcols
    size_t row_comm_size{2};
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, int>
       subcomm_factory(handle, row_comm_size);

    auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

    int my_rank = comm.get_rank();

    auto edgelist_from_mm = ::cugraph::test::read_edgelist_from_matrix_market_file<vertex_t, edge_t, weight_t>(param.graph_file_full_path);
   std::cout<<"READ DATAFILE IN RANK: "<<my_rank<<std::endl;

    auto graph = cugraph::test::create_graph_for_gpu<vertex_t, edge_t, weight_t, true> // store_transposed=true
       (handle, edgelist_from_mm, param.test_weighted);

    auto graph_view = graph.view();

    std::vector<edge_t> h_offsets(graph_view.get_number_of_vertices() + 1);
    std::vector<vertex_t> h_indices(graph_view.get_number_of_edges());
    std::vector<weight_t> h_weights{};
    raft::update_host(h_offsets.data(),
                      graph_view.offsets(),
                      graph_view.get_number_of_vertices() + 1,
                      stream);
    raft::update_host(h_indices.data(),
                      graph_view.indices(),
                      graph_view.get_number_of_edges(),
                      stream);
    if (graph_view.is_weighted()) {
      h_weights.assign(graph_view.get_number_of_edges(), weight_t{0.0});
      raft::update_host(h_weights.data(),
                        graph_view.weights(),
                        graph_view.get_number_of_edges(),
                        stream);
    }

    std::vector<vertex_t> h_personalization_vertices{};
    std::vector<result_t> h_personalization_values{};
    if (param.personalization_ratio > 0.0) {
      std::default_random_engine generator{};
      std::uniform_real_distribution<double> distribution{0.0, 1.0};
      h_personalization_vertices.resize(graph_view.get_number_of_local_vertices());
      std::iota(h_personalization_vertices.begin(),
                h_personalization_vertices.end(),
                graph_view.get_local_vertex_first());
      h_personalization_vertices.erase(
        std::remove_if(h_personalization_vertices.begin(),
                       h_personalization_vertices.end(),
                       [&generator, &distribution, param](auto v) {
                         return distribution(generator) >= param.personalization_ratio;
                       }),
        h_personalization_vertices.end());
      h_personalization_values.resize(h_personalization_vertices.size());
      std::for_each(h_personalization_values.begin(),
                    h_personalization_values.end(),
                    [&distribution, &generator](auto& val) { val = distribution(generator); });
      // use a double type counter (instead of result_t) to accumulate as std::accumulate is
      // inaccurate in adding a large number of comparably sized numbers. In C++17 or later,
      // std::reduce may be a better option.
      auto sum = static_cast<result_t>(std::accumulate(
        h_personalization_values.begin(), h_personalization_values.end(), double{0.0}));
      std::for_each(h_personalization_values.begin(),
                    h_personalization_values.end(),
                    [sum](auto& val) { val /= sum; });
    }

    rmm::device_uvector<vertex_t> d_personalization_vertices(h_personalization_vertices.size(),
                                                             stream);
    rmm::device_uvector<result_t> d_personalization_values(d_personalization_vertices.size(),
                                                           stream);
    if (d_personalization_vertices.size() > 0) {
      raft::update_device(d_personalization_vertices.data(),
                          h_personalization_vertices.data(),
                          h_personalization_vertices.size(),
                          stream);
      raft::update_device(d_personalization_values.data(),
                          h_personalization_values.data(),
                          h_personalization_values.size(),
                          stream);
    }

    rmm::device_uvector<result_t> d_pageranks(graph_view.get_number_of_vertices(),
                                              stream);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    cugraph::experimental::pagerank(handle,
                                    graph_view,
                                    static_cast<weight_t*>(nullptr),
                                    d_personalization_vertices.data(),
                                    d_personalization_values.data(),
                                    static_cast<vertex_t>(d_personalization_vertices.size()),
                                    d_pageranks.begin(),
                                    alpha,
                                    epsilon,
                                    std::numeric_limits<size_t>::max(),
                                    false,
                                    false);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<result_t> h_cugraph_pageranks(graph_view.get_number_of_vertices());

    raft::update_host(
      h_cugraph_pageranks.data(), d_pageranks.data(), d_pageranks.size(), stream);
    CUDA_TRY(cudaStreamSynchronize(stream));

    // Check for correctness against the reference implementation only on the 0th PE.
    /*
    if (my_rank == 0) {
       // Gather pagerank values - each GPU will have pagerank values for their range.
       // graph.get_local_vertex_first()
       // graph.get_local_vertex_last()

       std::vector<result_t> h_reference_pageranks(graph_view.get_number_of_vertices());

       pagerank_reference(h_offsets.data(),
                          h_indices.data(),
                          h_weights.size() > 0 ? h_weights.data() : static_cast<weight_t*>(nullptr),
                          h_personalization_vertices.data(),
                          h_personalization_values.data(),
                          h_reference_pageranks.data(),
                          graph_view.get_number_of_vertices(),
                          static_cast<vertex_t>(h_personalization_vertices.size()),
                          alpha,
                          epsilon,
                          std::numeric_limits<size_t>::max(),
                          false);

       auto threshold_ratio = 1e-3;
       auto threshold_magnitude =
          (1.0 / static_cast<result_t>(graph_view.get_number_of_vertices())) *
          threshold_ratio;  // skip comparison for low PageRank verties (lowly ranked vertices)
       auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
          return std::abs(lhs - rhs) <
          std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
       };

       ASSERT_TRUE(std::equal(h_reference_pageranks.begin(),
                              h_reference_pageranks.end(),
                              h_cugraph_pageranks.begin(),
                              nearly_equal))
          << "PageRank values do not match with the reference values.";
    }
    */
    std::cout<<"RANK: "<<my_rank<<" DONE."<<std::endl;
   }
};


TEST_P(Pagerank_E2E_MG_Testfixture_t, CheckInt32Int32FloatFloat) {
   run_test<int32_t, int32_t, float, float>(GetParam());
}

// FIXME: Enable additional test params once the first one is passing.
INSTANTIATE_TEST_CASE_P(
  e2e,
  Pagerank_E2E_MG_Testfixture_t,
  ::testing::Values(Pagerank_Testparams_t("test/datasets/karate.mtx", 0.0, false)
                    /*
                    Pagerank_Testparams_t("test/datasets/karate.mtx", 0.5, false),
                    Pagerank_Testparams_t("test/datasets/karate.mtx", 0.0, true),
                    Pagerank_Testparams_t("test/datasets/karate.mtx", 0.5, true),
                    Pagerank_Testparams_t("test/datasets/web-Google.mtx", 0.0, false),
                    Pagerank_Testparams_t("test/datasets/web-Google.mtx", 0.5, false),
                    Pagerank_Testparams_t("test/datasets/web-Google.mtx", 0.0, true),
                    Pagerank_Testparams_t("test/datasets/web-Google.mtx", 0.5, true),
                    Pagerank_Testparams_t("test/datasets/ljournal-2008.mtx", 0.0, false),
                    Pagerank_Testparams_t("test/datasets/ljournal-2008.mtx", 0.5, false),
                    Pagerank_Testparams_t("test/datasets/ljournal-2008.mtx", 0.0, true),
                    Pagerank_Testparams_t("test/datasets/ljournal-2008.mtx", 0.5, true),
                    Pagerank_Testparams_t("test/datasets/webbase-1M.mtx", 0.0, false),
                    Pagerank_Testparams_t("test/datasets/webbase-1M.mtx", 0.5, false),
                    Pagerank_Testparams_t("test/datasets/webbase-1M.mtx", 0.0, true),
                    Pagerank_Testparams_t("test/datasets/webbase-1M.mtx", 0.5, true))
                    */
                    )
);

// FIXME: Enable proper RMM configuration by using CUGRAPH_TEST_PROGRAM_MAIN().
//        Currently seeing a RMM failure during init, need to investigate.
//CUGRAPH_TEST_PROGRAM_MAIN()
