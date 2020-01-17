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

//#define SNMG_VERBOSE

// ref Degree on the host
template<typename idx_t>
void ref_degree_h(int x,
                  std::vector<idx_t> & off_h,
                  std::vector<idx_t> & ind_h,
                  std::vector<idx_t> & degree) {
  for (size_t i = 0; i < degree.size(); i++)
    degree[i] = 0;
  if (x == 0 || x == 2) {
    for (size_t i = 0; i < degree.size(); ++i) {
      degree[i] += off_h[i + 1] - off_h[i];
    }
  }
  if (x == 0 || x == 1) {
    for (size_t i = 0; i < ind_h.size(); i++)
      degree[ind_h[i]] += 1;
  }
}

struct MGDegree_Usecase {
  std::string matrix_file;
  int x;
  MGDegree_Usecase(const std::string& a, int _x) {
    x = _x;
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // if RAPIDS_DATASET_ROOT_DIR not set, default to "/datasets"
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
  }
  MGDegree_Usecase& operator=(const MGDegree_Usecase& rhs) {
    matrix_file = rhs.matrix_file;
    return *this;
  }
};

class Tests_MGDegree: public ::testing::TestWithParam<MGDegree_Usecase> {
public:
  Tests_MGDegree() {
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

  template<typename idx_t>
  void run_current_test(const MGDegree_Usecase& param) {
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
    std::vector<idx_t> degree_h(m, 0.0), degree_ref(m, 0.0), csrVal(nnz);

    // Read
    ASSERT_EQ( (mm_to_coo<int,int>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], NULL, NULL)) , 0)<< "could not read matrix data"<< "\n";
    ASSERT_EQ(fclose(fpin), 0);
    //ASSERT_EQ( (coo_to_csr<int,val_t> (m, m, nnz, &cooRowInd[0],  &cooColInd[0], NULL, NULL, &csrRowPtr[0], NULL, NULL, NULL)), 0) << "could not covert COO to CSR "<< "\n";
    coo2csr(cooRowInd, cooColInd, csrRowPtr, csrColInd);

    CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
    std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus + 1);
    gdf_column *col_x[n_gpus];
    //reference result
    t = omp_get_wtime();
    ref_degree_h(param.x, csrRowPtr, csrColInd, degree_ref);
    std::cout << "CPU time: " << omp_get_wtime() - t << "\n";
    if (nnz < 1200000000)
        {
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

        gdf_column *col_off = new gdf_column,
            *col_ind = new gdf_column,
            *col_val = new gdf_column;
        col_x[i] = new gdf_column;
        create_gdf_column(degree_h, col_x[i]);
#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        t = omp_get_wtime();
        cugraph::snmg_degree(param.x, &part_offset[0], col_off, col_ind, col_x);
        
#pragma omp master
        {
          std::cout << "GPU time: " << omp_get_wtime() - t << "\n";
        }

#pragma omp master
        {
          //printv(m, (val_t *)col_x[0]->data, 0);
          CUDA_RT_CALL(cudaMemcpy(&degree_h[0],
                                  col_x[0]->data,
                                  sizeof(idx_t) * m,
                                  cudaMemcpyDeviceToHost));

          for (size_t j = 0; j < degree_h.size(); ++j)
            EXPECT_EQ(degree_ref[j], degree_h[j]);
        }

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(col_x[i]);
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

        gdf_column *col_off = new gdf_column,
            *col_ind = new gdf_column,
            *col_val = new gdf_column;
        col_x[i] = new gdf_column;
        create_gdf_column(degree_h, col_x[i]);
#pragma omp barrier

        //load a chunck of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);

        t = omp_get_wtime();
        cugraph::snmg_degree(param.x, &part_offset[0], col_off, col_ind, col_x);
        
#pragma omp master
        {
          std::cout << "multi-GPU time: " << omp_get_wtime() - t << "\n";
        }

#pragma omp master
        {
          //printv(m, (val_t *)col_x[0]->data, 0);
          CUDA_RT_CALL(cudaMemcpy(&degree_h[0],
                                  col_x[0]->data,
                                  sizeof(idx_t) * m,
                                  cudaMemcpyDeviceToHost));

          for (size_t j = 0; j < degree_h.size(); ++j)
            EXPECT_EQ(degree_ref[j], degree_h[j]);
        }

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(col_x[i]);
      }
    }
    std::cout << std::endl;
  }
};

TEST_P(Tests_MGDegree, CheckInt32_mtx) {
  run_current_test<int>(GetParam());
}

INSTANTIATE_TEST_CASE_P(mtx_test, Tests_MGDegree,
                        ::testing::Values(MGDegree_Usecase("test/datasets/karate.mtx", 0)
                                                           ,
                                          MGDegree_Usecase("test/datasets/karate.mtx", 1)
                                                           ,
                                          MGDegree_Usecase("test/datasets/karate.mtx", 2)
                                                           ,
                                          MGDegree_Usecase("test/datasets/netscience.mtx", 0)
                                                           ,
                                          MGDegree_Usecase("test/datasets/netscience.mtx", 1)
                                                           ,
                                          MGDegree_Usecase("test/datasets/netscience.mtx", 2)
                                                           ,
                                          MGDegree_Usecase("test/datasets/cit-Patents.mtx", 0)
                                                           ,
                                          MGDegree_Usecase("test/datasets/cit-Patents.mtx", 1)
                                                           ,
                                          MGDegree_Usecase("test/datasets/cit-Patents.mtx", 2)
                                                           ,
                                          MGDegree_Usecase("test/datasets/webbase-1M.mtx", 0)
                                                           ,
                                          MGDegree_Usecase("test/datasets/webbase-1M.mtx", 1)
                                                           ,
                                          MGDegree_Usecase("test/datasets/webbase-1M.mtx", 2)
                                                           ,
                                          MGDegree_Usecase("test/datasets/web-Google.mtx", 0)
                                                           ,
                                          MGDegree_Usecase("test/datasets/web-Google.mtx", 1)
                                                           ,
                                          MGDegree_Usecase("test/datasets/web-Google.mtx", 2)
                                                           ,
                                          MGDegree_Usecase("test/datasets/wiki-Talk.mtx", 0)
                                                           ,
                                          MGDegree_Usecase("test/datasets/wiki-Talk.mtx", 1)
                                                           ,
                                          MGDegree_Usecase("test/datasets/wiki-Talk.mtx", 2)
                                                           )
                                                           );

class Tests_MGDegree_hibench: public ::testing::TestWithParam<MGDegree_Usecase> {
public:
  Tests_MGDegree_hibench() {
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

  template<typename idx_t>
  void run_current_test(const MGDegree_Usecase& param) {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".")
        + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)
        + std::string("_") + ss.str().c_str();
    std::cout << "Filename: " << param.matrix_file << ", x=" << param.x << "\n";
    int m, nnz, n_gpus;
    
    std::vector<idx_t> cooRowInd, cooColInd;
    double t;

    ASSERT_EQ(read_single_file(param.matrix_file.c_str(), cooRowInd, cooColInd), 0);
    nnz = cooRowInd.size();
    m = std::max(*(std::max_element(cooRowInd.begin(), cooRowInd.end())),
                 *(std::max_element(cooColInd.begin(), cooColInd.end())));
    m += 1;

    // Allocate memory on host
    std::vector<idx_t> csrColInd(nnz), csrRowPtr(m + 1), degree_ref(m), degree_h(m), csrVal(nnz);
    coo2csr(cooRowInd, cooColInd, csrRowPtr, csrColInd);
    CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
    std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus + 1);
    gdf_column *col_x[n_gpus];
    //reference result
    t = omp_get_wtime();
    ref_degree_h(param.x, csrRowPtr, csrColInd, degree_ref);
    std::cout << "CPU time: " << omp_get_wtime() - t << "\n";

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

        gdf_column *col_off = new gdf_column,
            *col_ind = new gdf_column,
            *col_val = new gdf_column;
        col_x[i] = new gdf_column;
        create_gdf_column(degree_h, col_x[i]);
#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);
        //printv(col_val->size,(float*)col_val->data,0);
        t = omp_get_wtime();
        cugraph::snmg_degree(param.x, &part_offset[0], col_off, col_ind, col_x);
        
#pragma omp master
        {
          std::cout << "GPU time: " << omp_get_wtime() - t << "\n";
        }

#pragma omp master
        {
          //printv(m, (val_t *)col_x[0]->data, 0);
          CUDA_RT_CALL(cudaMemcpy(&degree_h[0],
                                  col_x[0]->data,
                                  sizeof(idx_t) * m,
                                  cudaMemcpyDeviceToHost));

          for (size_t j = 0; j < degree_ref.size(); ++j)
            EXPECT_EQ(degree_ref[j], degree_h[j]);
        }

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(col_x[i]);
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

        gdf_column *col_off = new gdf_column,
            *col_ind = new gdf_column,
            *col_val = new gdf_column;
        col_x[i] = new gdf_column;
        create_gdf_column(degree_h, col_x[i]);
#pragma omp barrier

        //load a chunk of the graph on each GPU
        load_csr_loc(csrRowPtr, csrColInd, csrVal,
                     v_loc,
                     e_loc, part_offset,
                     col_off,
                     col_ind, col_val);
        //printv(col_val->size,(float*)col_val->data,0);
        t = omp_get_wtime();
        cugraph::snmg_degree(param.x, &part_offset[0], col_off, col_ind, col_x);
        
#pragma omp master
        {
          std::cout << "multi-GPU time: " << omp_get_wtime() - t << "\n";
        }

#pragma omp master
        {
          //printv(m, (val_t *)col_x[0]->data, 0);
          CUDA_RT_CALL(cudaMemcpy(&degree_h[0],
                                  col_x[0]->data,
                                  sizeof(idx_t) * m,
                                  cudaMemcpyDeviceToHost));

          for (size_t j = 0; j < degree_h.size(); ++j)
            EXPECT_EQ(degree_ref[j], degree_h[j]);
        }

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(col_x[i]);
      }
    }
    std::cout << std::endl;
  }
};

TEST_P(Tests_MGDegree_hibench, CheckFP32_hibench) {
  run_current_test<int>(GetParam());
}

INSTANTIATE_TEST_CASE_P(hibench_test,
                        Tests_MGDegree_hibench,
                        ::testing::Values(MGDegree_Usecase("benchmark/hibench/1/Input-small/edges/part-00000",
                                                           0)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-small/edges/part-00000",
                                                           1)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-small/edges/part-00000",
                                                           2)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-large/edges/part-00000",
                                                           0)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-large/edges/part-00000",
                                                           1)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-large/edges/part-00000",
                                                           2)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-huge/edges/part-00000",
                                                           0)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-huge/edges/part-00000",
                                                           1)
                                                           ,
                                          MGDegree_Usecase("benchmark/hibench/1/Input-huge/edges/part-00000",
                                                           2)
                                                           )
                                                           );

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}
