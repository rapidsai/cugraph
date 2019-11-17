/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"
#include <cugraph.h>
#include <omp.h>
#include "test_utils.h"
#include "snmg_test_utils.h"

struct MGcoo2csr_Usecase {
  std::string matrix_file;
  MGcoo2csr_Usecase(const std::string& a) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // if RAPIDS_DATASET_ROOT_DIR not set, default to "/datasets"
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
  }
  MGcoo2csr_Usecase& operator=(const MGcoo2csr_Usecase& rhs) {
    matrix_file = rhs.matrix_file;
    return *this;
  }
};

class Tests_MGcoo2csr: public ::testing::TestWithParam<MGcoo2csr_Usecase> {
public:
  Tests_MGcoo2csr() {
  }
  static void SetupTestCase() {
  }
  static void TearDownTestCase() {
  }
  virtual void SetUp() {
  }
  virtual void TearDown() {
  }

  static std::vector<double> mgspmv_time;

  template<typename idx_t, typename val_t>
  void run_current_test(const MGcoo2csr_Usecase& param) {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".")
        + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)
        + std::string("_") + ss.str().c_str();
    std::cout << test_id << "\n";
    int m, k, nnz, n_gpus;
    MM_typecode mc;
    

    double t;

    FILE* fpin = fopen(param.matrix_file.c_str(), "r");

    if (!fpin) {
      std::cout << "Could not open file: " << param.matrix_file << "\n";
      FAIL();
    }

    ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0)<< "could not read Matrix Market file properties"<< "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    // Allocate memory on host
    std::vector<idx_t> cooRowInd(nnz), cooColInd(nnz), csrColInd(nnz), csrRowPtr(m + 1);
    std::vector<idx_t> degree_h(m, 0.0), degree_ref(m, 0.0);
    std::vector<val_t> csrVal(nnz, 0.0);

    // Read
    ASSERT_EQ( (mm_to_coo<int,int>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], NULL, NULL)) , 0)<< "could not read matrix data"<< "\n";
    ASSERT_EQ(fclose(fpin), 0);
    //ASSERT_EQ( (coo_to_csr<int,val_t> (m, m, nnz, &cooRowInd[0],  &cooColInd[0], NULL, NULL, &csrRowPtr[0], NULL, NULL, NULL)), 0) << "could not covert COO to CSR "<< "\n";
    std::vector<idx_t> cooRowInd_tmp(cooRowInd);
    std::vector<idx_t> cooColInd_tmp(cooColInd);
    coo2csr(cooRowInd_tmp, cooColInd_tmp, csrRowPtr, csrColInd);

    CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
    std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus + 1), part_offset_r(n_gpus
        + 1);
    void* comm1;

    if (nnz < 1200000000) {
#pragma omp parallel num_threads(1)
      {
        //omp_set_num_threads(n_gpus);
        auto i = omp_get_thread_num();
        auto p = omp_get_num_threads();
        CUDA_RT_CALL(cudaSetDevice(i));

#ifdef SNMG_VERBOSE
#pragma omp master
        {
          std::cout << "Number of GPUs : "<< n_gpus <<std::endl;
          std::cout << "Number of threads : "<< p <<std::endl;
        }
#endif

        gdf_column *csr_off = new gdf_column;
        gdf_column *csr_ind = new gdf_column;
        gdf_column *csr_val = new gdf_column;
        gdf_column *col_off = new gdf_column;
        gdf_column *col_ind = new gdf_column;
        gdf_column *col_val = new gdf_column;
        gdf_column *coo_row = new gdf_column;
        gdf_column *coo_col = new gdf_column;
        gdf_column *coo_val = new gdf_column;

#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        //load a chunk of the graph on each GPU COO
        load_coo_loc(cooRowInd, cooColInd, csrVal, coo_row, coo_col, coo_val);

        t = omp_get_wtime();
        cugraph::snmg_coo2csr(&part_offset_r[0],
                                  false,
                                  &comm1,
                                  coo_row,
                                  coo_col,
                                  coo_val,
                                  csr_off,
                                  csr_ind,
                                  csr_val);
        
#pragma omp master
        {
          std::cout << "GPU time: " << omp_get_wtime() - t << "\n";
        }

        // Compare the results with those generated on the host

        EXPECT_EQ(part_offset[0], part_offset_r[0]);
        EXPECT_EQ(part_offset[1], part_offset_r[1]);
        EXPECT_TRUE(gdf_csr_equal<idx_t>(csr_off, csr_ind, col_off, col_ind));


        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(csr_off);
        gdf_col_delete(csr_ind);
        gdf_col_delete(csr_val);
        gdf_col_delete(coo_row);
        gdf_col_delete(coo_col);
        gdf_col_delete(coo_val);
      }
    }
    if (n_gpus > 1)
        {
      // Only using the 4 fully connected GPUs on DGX1
      if (n_gpus == 8)
        n_gpus = 4;

#pragma omp parallel num_threads(n_gpus)
      {
        //omp_set_num_threads(n_gpus);
        auto i = omp_get_thread_num();
        auto p = omp_get_num_threads();
        CUDA_RT_CALL(cudaSetDevice(i));

#ifdef SNMG_VERBOSE
#pragma omp master
        {
          std::cout << "Number of GPUs : "<< n_gpus <<std::endl;
          std::cout << "Number of threads : "<< p <<std::endl;
        }
#endif

        gdf_column *csr_off = new gdf_column;
        gdf_column *csr_ind = new gdf_column;
        gdf_column *csr_val = new gdf_column;
        gdf_column *col_off = new gdf_column;
        gdf_column *col_ind = new gdf_column;
        gdf_column *col_val = new gdf_column;
        gdf_column *coo_row = new gdf_column;
        gdf_column *coo_col = new gdf_column;
        gdf_column *coo_val = new gdf_column;
#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        //load a chunk of the graph on each GPU COO
        load_coo_loc(cooRowInd, cooColInd, csrVal, coo_row, coo_col, coo_val);

        t = omp_get_wtime();
        cugraph::snmg_coo2csr(&part_offset_r[0],
                                  false,
                                  &comm1,
                                  coo_row,
                                  coo_col,
                                  coo_val,
                                  csr_off,
                                  csr_ind,
                                  csr_val);

#pragma omp master
        {
          std::cout << "multi-GPU time: " << omp_get_wtime() - t << "\n";
        }

        // Compare the results with those generated on the host
        for (int j = 0; j < n_gpus + 1; j++)
          EXPECT_EQ(part_offset[j], part_offset_r[j]);
        EXPECT_TRUE(gdf_csr_equal<idx_t>(csr_off, csr_ind, col_off, col_ind));

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(csr_off);
        gdf_col_delete(csr_ind);
        gdf_col_delete(csr_val);
        gdf_col_delete(coo_row);
        gdf_col_delete(coo_col);
        gdf_col_delete(coo_val);
      }
    }
    std::cout << std::endl;
  }
};

TEST_P(Tests_MGcoo2csr, CheckInt32_floatmtx) {
  run_current_test<int, float>(GetParam());
}

TEST_P(Tests_MGcoo2csr, CheckInt32_doublemtx) {
  run_current_test<int, double>(GetParam());
}

INSTANTIATE_TEST_CASE_P(mtx_test, Tests_MGcoo2csr,
                        ::testing::Values(MGcoo2csr_Usecase("test/datasets/karate.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/netscience.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/cit-Patents.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/webbase-1M.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/web-Google.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/wiki-Talk.mtx")));

class Tests_MGcoo2csrTrans: public ::testing::TestWithParam<MGcoo2csr_Usecase> {
public:
  Tests_MGcoo2csrTrans() {
  }
  static void SetupTestCase() {
  }
  static void TearDownTestCase() {
  }
  virtual void SetUp() {
  }
  virtual void TearDown() {
  }

  static std::vector<double> mgspmv_time;

  template<typename idx_t, typename val_t>
  void run_current_test(const MGcoo2csr_Usecase& param) {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".")
        + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)
        + std::string("_") + ss.str().c_str();
    std::cout << test_id << "\n";
    int m, k, nnz, n_gpus;
    MM_typecode mc;
    

    double t;

    FILE* fpin = fopen(param.matrix_file.c_str(), "r");

    if (!fpin) {
      std::cout << "Could not open file: " << param.matrix_file << "\n";
      FAIL();
    }

    ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0)<< "could not read Matrix Market file properties"<< "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    // Allocate memory on host
    std::vector<idx_t> cooRowInd(nnz), cooColInd(nnz), csrColInd(nnz), csrRowPtr(m + 1);
    std::vector<idx_t> degree_h(m, 0.0), degree_ref(m, 0.0);
    std::vector<val_t> csrVal(nnz, 0.0);

    // Read
    ASSERT_EQ( (mm_to_coo<int,int>(fpin, 1, nnz, &cooColInd[0], &cooRowInd[0], NULL, NULL)) , 0)<< "could not read matrix data"<< "\n";
    ASSERT_EQ(fclose(fpin), 0);
    //ASSERT_EQ( (coo_to_csr<int,val_t> (m, m, nnz, &cooRowInd[0],  &cooColInd[0], NULL, NULL, &csrRowPtr[0], NULL, NULL, NULL)), 0) << "could not covert COO to CSR "<< "\n";
    std::vector<idx_t> cooRowInd_tmp(cooRowInd);
    std::vector<idx_t> cooColInd_tmp(cooColInd);
    coo2csr(cooRowInd_tmp, cooColInd_tmp, csrRowPtr, csrColInd);

    CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
    std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus + 1), part_offset_r(n_gpus
        + 1);
    void* comm1;

    if (nnz < 1200000000) {
#pragma omp parallel num_threads(1)
      {
        //omp_set_num_threads(n_gpus);
        auto i = omp_get_thread_num();
        auto p = omp_get_num_threads();
        CUDA_RT_CALL(cudaSetDevice(i));

#ifdef SNMG_VERBOSE
#pragma omp master
        {
          std::cout << "Number of GPUs : "<< n_gpus <<std::endl;
          std::cout << "Number of threads : "<< p <<std::endl;
        }
#endif

        gdf_column *csr_off = new gdf_column;
        gdf_column *csr_ind = new gdf_column;
        gdf_column *csr_val = new gdf_column;
        gdf_column *col_off = new gdf_column;
        gdf_column *col_ind = new gdf_column;
        gdf_column *col_val = new gdf_column;
        gdf_column *coo_row = new gdf_column;
        gdf_column *coo_col = new gdf_column;
        gdf_column *coo_val = new gdf_column;

#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        //load a chunk of the graph on each GPU COO
        load_coo_loc(cooRowInd, cooColInd, csrVal, coo_row, coo_col, coo_val);

        t = omp_get_wtime();
        cugraph::snmg_coo2csr(&part_offset_r[0],
                                  false,
                                  &comm1,
                                  coo_row,
                                  coo_col,
                                  coo_val,
                                  csr_off,
                                  csr_ind,
                                  csr_val);
        
#pragma omp master
        {
          std::cout << "GPU time: " << omp_get_wtime() - t << "\n";
        }

        // Compare the results with those generated on the host
      
        EXPECT_EQ(part_offset[0], part_offset_r[0]);
        EXPECT_EQ(part_offset[1], part_offset_r[1]);
        EXPECT_TRUE(gdf_csr_equal<idx_t>(csr_off, csr_ind, col_off, col_ind));


        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(csr_off);
        gdf_col_delete(csr_ind);
        gdf_col_delete(csr_val);
        gdf_col_delete(coo_row);
        gdf_col_delete(coo_col);
        gdf_col_delete(coo_val);
      }
    }
    if (n_gpus > 1)
        {
      // Only using the 4 fully connected GPUs on DGX1
      if (n_gpus == 8)
        n_gpus = 4;

#pragma omp parallel num_threads(n_gpus)
      {
        //omp_set_num_threads(n_gpus);
        auto i = omp_get_thread_num();
        auto p = omp_get_num_threads();
        CUDA_RT_CALL(cudaSetDevice(i));

#ifdef SNMG_VERBOSE
#pragma omp master
        {
          std::cout << "Number of GPUs : "<< n_gpus <<std::endl;
          std::cout << "Number of threads : "<< p <<std::endl;
        }
#endif

        gdf_column *csr_off = new gdf_column;
        gdf_column *csr_ind = new gdf_column;
        gdf_column *csr_val = new gdf_column;
        gdf_column *col_off = new gdf_column;
        gdf_column *col_ind = new gdf_column;
        gdf_column *col_val = new gdf_column;
        gdf_column *coo_row = new gdf_column;
        gdf_column *coo_col = new gdf_column;
        gdf_column *coo_val = new gdf_column;
#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        //load a chunk of the graph on each GPU COO
        load_coo_loc(cooRowInd, cooColInd, csrVal, coo_row, coo_col, coo_val);

        t = omp_get_wtime();
        cugraph::snmg_coo2csr(&part_offset_r[0],
                                  false,
                                  &comm1,
                                  coo_row,
                                  coo_col,
                                  coo_val,
                                  csr_off,
                                  csr_ind,
                                  csr_val);

#pragma omp master
        {
          std::cout << "multi-GPU time: " << omp_get_wtime() - t << "\n";
        }

        // Compare the results with those generated on the host
        
        for (int j = 0; j < n_gpus + 1; j++)
          EXPECT_EQ(part_offset[j], part_offset_r[j]);
        EXPECT_TRUE(gdf_csr_equal<idx_t>(csr_off, csr_ind, col_off, col_ind));

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(csr_off);
        gdf_col_delete(csr_ind);
        gdf_col_delete(csr_val);
        gdf_col_delete(coo_row);
        gdf_col_delete(coo_col);
        gdf_col_delete(coo_val);
      }
    }
    std::cout << std::endl;
  }
};

TEST_P(Tests_MGcoo2csrTrans, CheckInt32_floatmtx) {
  run_current_test<int, float>(GetParam());
}

TEST_P(Tests_MGcoo2csrTrans, CheckInt32_doublemtx) {
  run_current_test<int, double>(GetParam());
}

INSTANTIATE_TEST_CASE_P(mtx_test, Tests_MGcoo2csrTrans,
                        ::testing::Values(MGcoo2csr_Usecase("test/datasets/karate.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/netscience.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/cit-Patents.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/webbase-1M.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/web-Google.mtx"),
                                          MGcoo2csr_Usecase("test/datasets/wiki-Talk.mtx")));

class Tests_MGcoo2csr_hibench: public ::testing::TestWithParam<MGcoo2csr_Usecase> {
public:
  Tests_MGcoo2csr_hibench() {
  }
  static void SetupTestCase() {
  }
  static void TearDownTestCase() {
  }
  virtual void SetUp() {
  }
  virtual void TearDown() {
  }

  static std::vector<double> mgspmv_time;

  template<typename idx_t, typename val_t>
  void run_current_test(const MGcoo2csr_Usecase& param) {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".")
        + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)
        + std::string("_") + ss.str().c_str();
    std::cout << "Filename: " << param.matrix_file << "\n";
    int m, nnz, n_gpus;
    
    std::vector<idx_t> cooRowInd, cooColInd;
    double t;

    ASSERT_EQ(read_single_file(param.matrix_file.c_str(), cooRowInd, cooColInd), 0);
    nnz = cooRowInd.size();
    m = std::max(*(std::max_element(cooRowInd.begin(), cooRowInd.end())),
                 *(std::max_element(cooColInd.begin(), cooColInd.end())));
    m += 1;

    // Allocate memory on host
    std::vector<idx_t> csrColInd(nnz), csrRowPtr(m + 1), degree_ref(m), degree_h(m);
    std::vector<val_t> csrVal(nnz, 0);
    std::vector<idx_t> cooRowInd_tmp(cooRowInd);
    std::vector<idx_t> cooColInd_tmp(cooColInd);
    coo2csr(cooRowInd_tmp, cooColInd_tmp, csrRowPtr, csrColInd);
    CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
    std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus + 1), part_offset_r(n_gpus + 1);
    void* comm1;

    if (nnz < 1200000000) {
#pragma omp parallel num_threads(1)
      {
        //omp_set_num_threads(n_gpus);
        auto i = omp_get_thread_num();
        auto p = omp_get_num_threads();
        CUDA_RT_CALL(cudaSetDevice(i));

#ifdef SNMG_VERBOSE
#pragma omp master
        {
          std::cout << "Number of GPUs : "<< n_gpus <<std::endl;
          std::cout << "Number of threads : "<< p <<std::endl;
        }
#endif

        gdf_column *csr_off = new gdf_column;
        gdf_column *csr_ind = new gdf_column;
        gdf_column *csr_val = new gdf_column;
        gdf_column *col_off = new gdf_column;
        gdf_column *col_ind = new gdf_column;
        gdf_column *col_val = new gdf_column;
        gdf_column *coo_row = new gdf_column;
        gdf_column *coo_col = new gdf_column;
        gdf_column *coo_val = new gdf_column;
#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        //load a chunk of the graph on each GPU COO
        load_coo_loc(cooRowInd, cooColInd, csrVal, coo_row, coo_col, coo_val);

        t = omp_get_wtime();
        cugraph::snmg_coo2csr(&part_offset_r[0],
                                  false,
                                  &comm1,
                                  coo_row,
                                  coo_col,
                                  coo_val,
                                  csr_off,
                                  csr_ind,
                                  csr_val);
        
#pragma omp master
        {
          std::cout << "GPU time: " << omp_get_wtime() - t << "\n";
        }

        // Compare the results with those generated on the host
        
        EXPECT_EQ(part_offset[0], part_offset_r[0]);
        EXPECT_EQ(part_offset[1], part_offset_r[1]);
        EXPECT_TRUE(gdf_csr_equal<idx_t>(csr_off, csr_ind, col_off, col_ind));

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(csr_off);
        gdf_col_delete(csr_ind);
        gdf_col_delete(csr_val);
        gdf_col_delete(coo_row);
        gdf_col_delete(coo_col);
        gdf_col_delete(coo_val);
      }
    }
    if (n_gpus > 1) {
      // Only using the 4 fully connected GPUs on DGX1
      if (n_gpus == 8)
        n_gpus = 4;

#pragma omp parallel num_threads(n_gpus)
      {
        auto i = omp_get_thread_num();
        auto p = omp_get_num_threads();
        CUDA_RT_CALL(cudaSetDevice(i));

#ifdef SNMG_VERBOSE
#pragma omp master
        {
          std::cout << "Number of GPUs : "<< n_gpus <<std::endl;
          std::cout << "Number of threads : "<< p <<std::endl;
        }
#endif

        gdf_column *csr_off = new gdf_column;
        gdf_column *csr_ind = new gdf_column;
        gdf_column *csr_val = new gdf_column;
        gdf_column *col_off = new gdf_column;
        gdf_column *col_ind = new gdf_column;
        gdf_column *col_val = new gdf_column;
        gdf_column *coo_row = new gdf_column;
        gdf_column *coo_col = new gdf_column;
        gdf_column *coo_val = new gdf_column;
#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        //load a chunk of the graph on each GPU COO
        load_coo_loc(cooRowInd, cooColInd, csrVal, coo_row, coo_col, coo_val);

        t = omp_get_wtime();
        cugraph::snmg_coo2csr(&part_offset_r[0],
                                  false,
                                  &comm1,
                                  coo_row,
                                  coo_col,
                                  coo_val,
                                  csr_off,
                                  csr_ind,
                                  csr_val);
        
#pragma omp master
        {
          std::cout << "multi-GPU time: " << omp_get_wtime() - t << "\n";
        }

        // Compare the results with those generated on the host
        for (int j = 0; j < n_gpus + 1; j++)
          EXPECT_EQ(part_offset[j], part_offset_r[j]);
        EXPECT_TRUE(gdf_csr_equal<idx_t>(csr_off, csr_ind, col_off, col_ind));

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(csr_off);
        gdf_col_delete(csr_ind);
        gdf_col_delete(csr_val);
        gdf_col_delete(coo_row);
        gdf_col_delete(coo_col);
        gdf_col_delete(coo_val);
      }
    }
    std::cout << std::endl;
  }
};

TEST_P(Tests_MGcoo2csr_hibench, CheckFP32_hibench) {
  run_current_test<int, float>(GetParam());
}

TEST_P(Tests_MGcoo2csr_hibench, CheckFP64_hibench) {
  run_current_test<int, double>(GetParam());
}

INSTANTIATE_TEST_CASE_P(hibench_test,
                        Tests_MGcoo2csr_hibench,
                        ::testing::Values(MGcoo2csr_Usecase("benchmark/hibench/1/Input-small/edges/part-00000"),
                                          MGcoo2csr_Usecase("benchmark/hibench/1/Input-large/edges/part-00000"),
                                          MGcoo2csr_Usecase("benchmark/hibench/1/Input-huge/edges/part-00000")));

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}
