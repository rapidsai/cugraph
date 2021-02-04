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
      MPI_TRY(MPI_Init(NULL, NULL));

      int rank, size;
      MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
      MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &size));

      int nGpus;
      CUDA_CHECK(cudaGetDeviceCount(&nGpus));

      ASSERT(nGpus >= size,
             "Number of GPUs are lesser than MPI ranks! ngpus=%d, nranks=%d",
             nGpus, size);

      CUDA_CHECK(cudaSetDevice(rank));
   }

   static void TearDownTestCase() {
      MPI_TRY(MPI_Finalize());
   }

   // Run once for each test instance
   virtual void SetUp() {}
   virtual void TearDown() {}

   //
   //
   template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
   void run_test(const Pagerank_Testparams_t& param)
   {
      result_t constexpr alpha{0.85};
      result_t constexpr epsilon{1e-6};

      raft::handle_t handle;
      raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
      const auto &comm = handle.get_comms();

      cudaStream_t stream = handle.get_stream();

      // Assuming 2 GPUs which means 1 row, 2 cols. 2 cols = row_comm_size of 2.
      // FIXME: DO NOT ASSUME 2 GPUs, add code to compute prows, pcols
      size_t row_comm_size{2};
      cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
         subcomm_factory(handle, row_comm_size);

      int my_rank = comm.get_rank();

      auto edgelist_from_mm = ::cugraph::test::read_edgelist_from_matrix_market_file<vertex_t, edge_t, weight_t>(param.graph_file_full_path);
      std::cout<<"READ DATAFILE IN RANK: "<<my_rank<<std::endl;

      //cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, true>& mg_graph;  // store_transposed=true, multi_gpu=true
      //std::tuple<rmm::device_uvector<vertex_t>, cugraph::experimental::partition_t<vertex_t>, vertex_t, edge_t>& mg_renumber_info;
      // FIXME: edgelist_from_mm must have weights!
      std::unique_ptr<
         cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, true>> // store_transposed=true, multi_gpu=true
         mg_graph_ptr{};
      rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());

      std::tie(mg_graph_ptr, d_renumber_map_labels) =
         cugraph::test::create_graph_for_gpu<vertex_t, edge_t, weight_t, true> // store_transposed=true
            (handle, edgelist_from_mm);

      std::cout<<"GRAPH CTOR IN RANK: "<<my_rank<<" NUM VERTS: "<<mg_graph_ptr->get_number_of_vertices()<<std::endl;

      auto mg_graph_view = mg_graph_ptr->view();

      ////////
      rmm::device_uvector<result_t> d_mg_pageranks(mg_graph_view.get_number_of_vertices(),
                                                   stream);
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

      cugraph::experimental::pagerank(handle,
                                      mg_graph_view,
                                      static_cast<weight_t*>(nullptr),     // adj_matrix_row_out_weight_sums
                                      static_cast<vertex_t*>(nullptr),     // personalization_vertices
                                      static_cast<result_t*>(nullptr),     // personalization_values
                                      static_cast<vertex_t>(0),            // personalization_vector_size
                                      d_mg_pageranks.begin(),              // pageranks
                                      alpha,                               // alpha (damping factor)
                                      epsilon,                             // error tolerance for convergence
                                      std::numeric_limits<size_t>::max(),  // max_iterations
                                      false,                               // has_initial_guess
                                      true);                               // do_expensive_check

      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

      // FIXME: un-renumber somewhere?
      // manually unrenumber using the mapping to properly index the values
      std::vector<result_t> h_mg_pageranks(mg_graph_view.get_number_of_vertices());

      raft::update_host(h_mg_pageranks.data(),
                        d_mg_pageranks.data(),
                        d_mg_pageranks.size(),
                        stream);

      CUDA_TRY(cudaStreamSynchronize(stream));

      std::vector<vertex_t> h_renumber_map_labels(mg_graph_view.get_number_of_vertices());
      raft::update_host(h_renumber_map_labels.data(),
                        d_renumber_map_labels.data(),
                        d_renumber_map_labels.size(),
                        stream);

      ////////
      // single-GPU
      auto sg_graph =
         cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, true>(
            handle, param.graph_file_full_path, param.test_weighted);
      auto sg_graph_view = sg_graph.view();

      rmm::device_uvector<result_t> d_sg_pageranks(sg_graph_view.get_number_of_vertices(), stream);
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

      cugraph::experimental::pagerank(handle,
                                      sg_graph_view,
                                      static_cast<weight_t*>(nullptr),     // adj_matrix_row_out_weight_sums
                                      static_cast<vertex_t*>(nullptr),     // personalization_vertices
                                      static_cast<result_t*>(nullptr),     // personalization_values
                                      static_cast<vertex_t>(0),            // personalization_vector_size
                                      d_sg_pageranks.begin(),                 // pageranks
                                      alpha,                               // alpha (damping factor)
                                      epsilon,                             // error tolerance for convergence
                                      std::numeric_limits<size_t>::max(),  // max_iterations
                                      false,                               // has_initial_guess
                                      true);                               // do_expensive_check

      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

      std::vector<result_t> h_sg_pageranks(sg_graph_view.get_number_of_vertices());

      raft::update_host(h_sg_pageranks.data(),
                        d_sg_pageranks.data(),
                        d_sg_pageranks.size(),
                        stream);

      CUDA_TRY(cudaStreamSynchronize(stream));


      ////////
      // Compare MG to SG
      // Each GPU will have pagerank values for their range, so ech GPU must
      // compare to specific SG results for their respective range.

      // For this test, each GPU will have the full set of vertices and
      // therefore the pageranks vectors should be equal in size.
      // NOTE: Each GPU will only have valid pagerank values for a subset of the
      // vertices, since each is computing a different subset in parallel.
      ASSERT_EQ(h_sg_pageranks.size(), h_mg_pageranks.size());

      auto threshold_ratio = 1e-3;
      auto threshold_magnitude =
         (1.0 / static_cast<result_t>(sg_graph_view.get_number_of_vertices())) *
         threshold_ratio;  // skip comparison for low PageRank verties (lowly ranked vertices)
      auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
         return std::abs(lhs - rhs) <
                std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
      };

      vertex_t mapped_vertex{0};
      for(vertex_t i=0; i+mg_graph_view.get_local_vertex_first() < mg_graph_view.get_local_vertex_last(); ++i){
         mapped_vertex = h_renumber_map_labels[i];

         std::cout<<"RANK: "<<my_rank<<" i: "<<i<<" SG: "<<h_sg_pageranks[mapped_vertex]<<" MG: "<<h_mg_pageranks[i]<<" RNL: "<<mapped_vertex<<std::endl;
      }

      for(vertex_t i=0; i+mg_graph_view.get_local_vertex_first() < mg_graph_view.get_local_vertex_last(); ++i){
         mapped_vertex = h_renumber_map_labels[i];
         ASSERT_TRUE(nearly_equal(h_mg_pageranks[i], h_sg_pageranks[mapped_vertex])) << "MG PageRank value for vertex: " << i << " in rank: " << my_rank << " has value: " << h_mg_pageranks[i] << " which exceeds the error margin for comparing to SG value: " << h_sg_pageranks[i];
      }

      //ASSERT_TRUE(std::equal(h_sg_pageranks.begin(),
      //                       h_sg_pageranks.end(),
      //                       h_mg_pageranks.begin(),
      //                       nearly_equal))
      //   << "MG PageRank values do not match with the SG reference values.";

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
