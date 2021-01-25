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

// Read in data, distributed?
// Shuffle?
// Call Pagerank
// Compare to ref

#include <gtest/gtest.h>

#include <raft/comms/mpi_comms.hpp>

#include <algorithms.hpp>

#include <utilities/test_utilities.hpp>
#include <utilities/base_fixture.hpp>

//
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_CASE_P()
//
typedef struct Pagerank_Testparams_t {
  std::string matrix_file;
  std::string result_file;
  Pagerank_Testparams_t(const std::string& a, const std::string& b)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
    if ((b != "") && (b[0] != '/')) {
      result_file = rapidsDatasetRootDir + "/" + b;
    } else {
      result_file = b;
    }
  }
  Pagerank_Testparams_t& operator=(const Pagerank_Testparams_t& rhs)
  {
    matrix_file = rhs.matrix_file;
    result_file = rhs.result_file;
    return *this;
  }
} Pagerank_Testparams;


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
   // by cuML test_opg_utils.h
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

   template <typename weight_t>
   void run_test(const Pagerank_Testparams_t& param)
   {
    int m, k, nnz;
    MM_typecode mc;

    float tol = 1E-5f;

    raft::handle_t handle;
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto &comm = handle.get_comms();
    const auto allocator = handle.get_device_allocator();

    cudaStream_t stream = handle.get_stream();

    int my_rank = comm.get_rank();
    int size = comm.get_size();

    ////

    FILE* fpin = fopen(param.matrix_file.c_str(), "r");
    ASSERT_NE(fpin, nullptr) << "fopen (" << param.matrix_file << ") failure.";

    ASSERT_EQ(cugraph::test::mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz), 0)
      << "could not read Matrix Market file properties"
      << "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    // Allocate memory on host
    std::vector<int> cooRowInd(nnz), cooColInd(nnz);
    std::vector<weight_t> cooVal(nnz), pagerank(m);

    // device alloc
    rmm::device_uvector<weight_t> pagerank_vector(static_cast<size_t>(m), nullptr);
    weight_t* d_pagerank = pagerank_vector.data();

    // Read
    ASSERT_EQ((cugraph::test::mm_to_coo<int, weight_t>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)),
              0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    //  Pagerank runs on CSC, so feed COOtoCSR the row/col backwards.
    cugraph::GraphCOOView<int, int, weight_t> G_coo(&cooColInd[0], &cooRowInd[0], &cooVal[0], m, nnz);
    auto G_unique = cugraph::coo_to_csr(G_coo);
    cugraph::GraphCSCView<int, int, weight_t> G(G_unique->view().offsets,
                                         G_unique->view().indices,
                                         G_unique->view().edge_data,
                                         G_unique->view().number_of_vertices,
                                         G_unique->view().number_of_edges);

    cudaDeviceSynchronize();

    //cudaProfilerStart();
    cugraph::pagerank<int, int, weight_t>(handle, G, d_pagerank);
    //cudaProfilerStop();
    cudaDeviceSynchronize();

    // Check vs golden data
    if (param.result_file.length() > 0) {
      std::vector<weight_t> calculated_res(m);

      CUDA_TRY(cudaMemcpy(&calculated_res[0], d_pagerank, sizeof(weight_t) * m, cudaMemcpyDeviceToHost));
      std::sort(calculated_res.begin(), calculated_res.end());
      fpin = fopen(param.result_file.c_str(), "rb");
      ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file
                                << std::endl;
      std::vector<weight_t> expected_res(m);
      ASSERT_EQ(cugraph::test::read_binary_vector(fpin, m, expected_res), 0);
      fclose(fpin);
      weight_t err;
      int n_err = 0;
      for (int i = 0; i < m; i++) {
        err = fabs(expected_res[i] - calculated_res[i]);
        if (err > tol * 1.1) {
          n_err++;  // count the number of mismatches
        }
      }
      if (n_err) {
        EXPECT_LE(n_err, 0.001 * m);  // we tolerate 0.1% of values with a litte difference
      }
    }
  }
};


// Create tests using the test fixture
//
TEST_P(Pagerank_E2E_MG_Testfixture_t, CheckFP32_T) { run_test<float>(GetParam()); }
TEST_P(Pagerank_E2E_MG_Testfixture_t, CheckFP64_T) { run_test<double>(GetParam()); }


// Create test instances - this defines the individual tests in the resulting gtest binary.
//
// INSTANTIATE_TEST_CASE_P() creates a test for each input param, for each test
// that uses the Pagerank_E2E_MG_Testfixture_t fixture. The resulting instances
// for these input params will be part of the suite named "e2e".
//
INSTANTIATE_TEST_CASE_P(
  e2e,
  Pagerank_E2E_MG_Testfixture_t,
  ::testing::Values(Pagerank_Testparams_t("test/datasets/karate.mtx", ""),
                    Pagerank_Testparams_t("test/datasets/web-Google.mtx",
                                          "test/ref/pagerank/web-Google.pagerank_val_0.85.bin"),
                    Pagerank_Testparams_t("test/datasets/ljournal-2008.mtx",
                                          "test/ref/pagerank/ljournal-2008.pagerank_val_0.85.bin"),
                    Pagerank_Testparams_t("test/datasets/webbase-1M.mtx",
                                          "test/ref/pagerank/webbase-1M.pagerank_val_0.85.bin"))
                        );

CUGRAPH_TEST_PROGRAM_MAIN()
