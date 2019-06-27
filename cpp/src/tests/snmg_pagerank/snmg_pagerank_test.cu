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
#include <math.h>
#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"
#include <cugraph.h>
#include <omp.h>
#include "test_utils.h"
#include "snmg_test_utils.h"
#include "snmg/link_analysis/pagerank.cuh"

//#define SNMG_VERBOSE

typedef struct MGPagerank_Usecase_t {
  std::string matrix_file;
  std::string result_file;

  MGPagerank_Usecase_t(const std::string& a, const std::string& b) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // if RAPIDS_DATASET_ROOT_DIR not set, default to "/datasets"
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
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
  MGPagerank_Usecase_t& operator=(const MGPagerank_Usecase_t& rhs) {
    matrix_file = rhs.matrix_file;
    result_file = rhs.result_file;
    return *this;
  }
} MGPagerank_Usecase;

template <typename val_t>
void verify_pr(gdf_column* col_pagerank, const MGPagerank_Usecase& param){
  // Check vs golden data
  if (param.result_file.length()>0)
  {
    int m = col_pagerank->size;
    std::vector<val_t> calculated_res(m);
    CUDA_RT_CALL(cudaMemcpy(&calculated_res[0],   col_pagerank->data, sizeof(val_t) * m, cudaMemcpyDeviceToHost));
    std::sort(calculated_res.begin(), calculated_res.end());
    FILE* fpin = fopen(param.result_file.c_str(),"rb");
    ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file << std::endl;
    std::vector<val_t> expected_res(m);
    ASSERT_EQ(read_binary_vector(fpin, m, expected_res), 0);
    fclose(fpin);
    val_t err;
    int n_err = 0;
    for (int i = 0; i < m; i++) {
        //check for invalid values
        ASSERT_FALSE(isnan(calculated_res[i])); 
        ASSERT_LE(calculated_res[i], 1.0);
        ASSERT_GE(calculated_res[i], 0.0);
        err = fabs(expected_res[i] - calculated_res[i]);
        if (err> 1e-5) {
            n_err++; // count the number of mismatches 
        }
    }
    if (n_err) {
        ASSERT_LE(n_err, 0.001*m); // tolerate 0.1% of values with a litte difference
    }
  }
}

class Tests_MGPagerank : public ::testing::TestWithParam<MGPagerank_Usecase> {
  public:
  Tests_MGPagerank() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> mgpr_time;   
  
  template <typename idx_t,typename val_t>
  void run_current_test(const MGPagerank_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, k, nnz, n_gpus, max_iter=50;
     val_t alpha = 0.85;
     MM_typecode mc;

     double t;

     FILE* fpin = fopen(param.matrix_file.c_str(),"r");
     ASSERT_NE(fpin, nullptr) << "fopen (" << param.matrix_file << ") failure.";
     
     ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
     ASSERT_TRUE(mm_is_matrix(mc));
     ASSERT_TRUE(mm_is_coordinate(mc));
     ASSERT_FALSE(mm_is_complex(mc));
     ASSERT_FALSE(mm_is_skew(mc));
     
     // Allocate memory on host
     std::vector<idx_t> cooRowInd(nnz), cooColInd(nnz), csrColInd(nnz), csrRowPtr(m+1);
     std::vector<val_t> cooVal(nnz), csrVal(nnz), pagerank_h(m, 1.0/m);

     // Read
     ASSERT_EQ( (mm_to_coo<int,val_t>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], NULL, NULL)) , 0)<< "could not read matrix data"<< "\n";
     ASSERT_EQ(fclose(fpin),0);
     
     // WARNING transpose happening here
     coo2csr(cooColInd, cooRowInd, csrRowPtr, csrColInd);

     CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));  
     std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus+1);
     random_vals(csrVal);
     gdf_column *col_pagerank[n_gpus];
     idx_t *degree[n_gpus];

     if (nnz<1200000000)
     {
       #pragma omp parallel num_threads(1)
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
        col_pagerank[i] = new gdf_column;
        create_gdf_column(pagerank_h, col_pagerank[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        
        t = omp_get_wtime();
        cugraph::SNMGinfo env;
        cugraph::SNMGpagerank<idx_t,val_t> pr_solver(env, &part_offset[0], static_cast<idx_t*>(col_off->data), static_cast<idx_t*>(col_ind->data));
        pr_solver.setup(alpha,degree);

        val_t* pagerank[p];
        for (auto i = 0; i < p; ++i)
          pagerank[i]= static_cast<val_t*>(col_pagerank[i]->data);

        pr_solver.solve(max_iter, pagerank);
        #pragma omp master 
        {std::cout <<  omp_get_wtime() - t << " ";}

        verify_pr<val_t>(col_pagerank[i], param);

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(col_pagerank[i]);
      }
    }

    if (n_gpus > 1)
    {
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
          col_pagerank[i] = new gdf_column;
          create_gdf_column(pagerank_h, col_pagerank[i]);
          #pragma omp barrier

          //load a chunck of the graph on each GPU 
          load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                       v_loc, e_loc, part_offset,
                       col_off, col_ind, col_val);
          t = omp_get_wtime();
          cugraph::SNMGinfo env;
          cugraph::SNMGpagerank<idx_t,val_t> pr_solver(env, &part_offset[0], static_cast<idx_t*>(col_off->data), static_cast<idx_t*>(col_ind->data));
          pr_solver.setup(alpha,degree);

          val_t* pagerank[p];
          for (auto i = 0; i < p; ++i)
            pagerank[i]= static_cast<val_t*>(col_pagerank[i]->data);

          pr_solver.solve(max_iter, pagerank);
          #pragma omp master 
          {std::cout <<  omp_get_wtime() - t << " ";}

          verify_pr<val_t>(col_pagerank[i], param);
          gdf_col_delete(col_off);
          gdf_col_delete(col_ind);
          gdf_col_delete(col_val);
          gdf_col_delete(col_pagerank[i]);


       }
    }
    std::cout << std::endl;
  }

};

class Tests_MGPR_hibench : public ::testing::TestWithParam<MGPagerank_Usecase> {
  public:
  Tests_MGPR_hibench() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> mgspmv_time;   

  template <typename idx_t,typename val_t>
  void run_current_test(const MGPagerank_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, nnz, n_gpus, max_iter=500;
     val_t alpha = 0.85;
     std::vector<idx_t> cooRowInd, cooColInd;
     double t;

     ASSERT_EQ(read_single_file(param.matrix_file.c_str(),cooRowInd,cooColInd),0) << "read_single_file(" << param.matrix_file << ", ...) failure.";
     nnz = cooRowInd.size();
     m = 1 + std::max( *(std::max_element(cooRowInd.begin(), cooRowInd.end())),
                   *(std::max_element(cooColInd.begin(), cooColInd.end())));

     // Allocate memory on host
     std::vector<idx_t> csrColInd(nnz), csrRowPtr(m+1);
     std::vector<val_t> cooVal(nnz), csrVal(nnz), pagerank_h(m, 1.0/m);

     // transpose here
     coo2csr(cooColInd, cooRowInd, csrRowPtr, csrColInd);
     CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));  
     std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus+1);
     random_vals(csrVal);
     gdf_column *col_pagerank[n_gpus];
     idx_t *degree[n_gpus];

     if (nnz<1200000000)
     {
       #pragma omp parallel num_threads(1)
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
        col_pagerank[i] = new gdf_column;
        create_gdf_column(pagerank_h, col_pagerank[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        
        t = omp_get_wtime();
        cugraph::SNMGinfo env;
        cugraph::SNMGpagerank<idx_t,val_t> pr_solver(env, &part_offset[0], static_cast<idx_t*>(col_off->data), static_cast<idx_t*>(col_ind->data));
        pr_solver.setup(alpha,degree);

        val_t* pagerank[p];
        for (auto i = 0; i < p; ++i)
          pagerank[i]= static_cast<val_t*>(col_pagerank[i]->data);

        pr_solver.solve(max_iter, pagerank);
        #pragma omp master 
        {std::cout <<  omp_get_wtime() - t << " ";}

        verify_pr<val_t>(col_pagerank[i], param);

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(col_pagerank[i]);
      }
    }
     if (n_gpus > 1)
     {
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
        col_pagerank[i] = new gdf_column;
        create_gdf_column(pagerank_h, col_pagerank[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        
        t = omp_get_wtime();
        cugraph::SNMGinfo env;
        cugraph::SNMGpagerank<idx_t,val_t> pr_solver(env, &part_offset[0], static_cast<idx_t*>(col_off->data), static_cast<idx_t*>(col_ind->data));
        pr_solver.setup(alpha,degree);

        val_t* pagerank[p];
        for (auto i = 0; i < p; ++i)
          pagerank[i]= static_cast<val_t*>(col_pagerank[i]->data);

        pr_solver.solve(max_iter, pagerank);
        #pragma omp master 
        {std::cout <<  omp_get_wtime() - t << " ";}

        verify_pr<val_t>(col_pagerank[i], param);

        gdf_col_delete(col_off);
        gdf_col_delete(col_ind);
        gdf_col_delete(col_val);
        gdf_col_delete(col_pagerank[i]);
      }
    }
    std::cout << std::endl;
  }
};


TEST_P(Tests_MGPagerank, CheckFP32_mtx) {
    run_current_test<int, float>(GetParam());
}
TEST_P(Tests_MGPagerank, CheckFP64) {
    run_current_test<int,double>(GetParam());
}
TEST_P(Tests_MGPR_hibench, CheckFP32_hibench) {
    run_current_test<int, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(mtx_test, Tests_MGPagerank, 
                        ::testing::Values(   MGPagerank_Usecase("test/datasets/karate.mtx", "")
                                            ,MGPagerank_Usecase("test/datasets/netscience.mtx", "")
                                            ,MGPagerank_Usecase("test/datasets/web-BerkStan.mtx", "test/ref/pagerank/web-BerkStan.pagerank_val_0.85.bin")
                                            ,MGPagerank_Usecase("test/datasets/web-Google.mtx",   "test/ref/pagerank/web-Google.pagerank_val_0.85.bin")
                                            ,MGPagerank_Usecase("test/datasets/wiki-Talk.mtx",    "test/ref/pagerank/wiki-Talk.pagerank_val_0.85.bin")
                                            ,MGPagerank_Usecase("test/datasets/cit-Patents.mtx",  "test/ref/pagerank/cit-Patents.pagerank_val_0.85.bin")
                                            ,MGPagerank_Usecase("test/datasets/ljournal-2008.mtx","test/ref/pagerank/ljournal-2008.pagerank_val_0.85.bin")
                                            ,MGPagerank_Usecase("test/datasets/webbase-1M.mtx",   "test/ref/pagerank/webbase-1M.pagerank_val_0.85.bin")
                                         )
                       );

INSTANTIATE_TEST_CASE_P(hibench_test, Tests_MGPR_hibench,  
                        ::testing::Values(   MGPagerank_Usecase("benchmark/hibench/1/Input-small/edges/part-00000", "")
                                            ,MGPagerank_Usecase("benchmark/hibench/1/Input-large/edges/part-00000", "")
                                            ,MGPagerank_Usecase("benchmark/hibench/1/Input-huge/edges/part-00000", "")
                                         )
                       );


int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
        
  return RUN_ALL_TESTS();
}


