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

// ref SPMV on the host
template <typename idx_t,typename val_t>
void csrmv_h (std::vector<idx_t> & off_h, 
                      std::vector<idx_t> & ind_h, 
                      std::vector<val_t> & val_h,  
                      std::vector<val_t> & x,  
                      std::vector<val_t> & y) {
  #pragma omp parallel for
  for (auto i = size_t{0}; i < y.size(); ++i) 
  {
      //std::cout<< omp_get_num_threads()<<std::endl;
      for (auto j = off_h[i]; j <  off_h[i+1]; ++j) 
        y[i] += val_h[j]*x[ind_h[j]];
  }
}

typedef struct MGSpmv_Usecase_t {
  std::string matrix_file;
  MGSpmv_Usecase_t(const std::string& a) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // if RAPIDS_DATASET_ROOT_DIR not set, default to "/datasets"
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
  }
  MGSpmv_Usecase_t& operator=(const MGSpmv_Usecase_t& rhs) {
    matrix_file = rhs.matrix_file;
    return *this;
  }
} MGSpmv_Usecase;

class Tests_MGSpmv : public ::testing::TestWithParam<MGSpmv_Usecase> {
  public:
  Tests_MGSpmv() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> mgspmv_time;   


  template <typename idx_t,typename val_t>
  void run_current_test(const MGSpmv_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, k, nnz, n_gpus;
     MM_typecode mc;
     gdf_error status;

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
     std::vector<val_t> cooVal(nnz), csrVal(nnz), x_h(m, 1.0), y_h(m, 0.0), y_ref(m, 0.0);

     // Read
     ASSERT_EQ( (mm_to_coo<int,val_t>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], NULL, NULL)) , 0)<< "could not read matrix data"<< "\n";
     ASSERT_EQ(fclose(fpin),0);
     coo2csr(cooRowInd, cooColInd, csrRowPtr, csrColInd);

     CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));  
     std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus+1);
     random_vals(csrVal);
     random_vals(x_h);
     gdf_column *col_x[n_gpus];
     //reference result
     t = omp_get_wtime();
     csrmv_h< idx_t, val_t>(csrRowPtr, csrColInd, csrVal, x_h, y_ref);
     std::cout <<  omp_get_wtime() - t << " ";
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
        col_x[i] = new gdf_column;
        create_gdf_column(x_h, col_x[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        t = omp_get_wtime();
        status = gdf_snmg_csrmv(&part_offset[0], col_off, col_ind, col_val, col_x);
        EXPECT_EQ(status,0);
        #pragma omp master 
          {std::cout <<  omp_get_wtime() - t << " ";}


        #pragma omp master 
        { 
          CUDA_RT_CALL(cudaMemcpy(&y_h[0], col_x[0]->data,   sizeof(val_t) * m, cudaMemcpyDeviceToHost));

          for (auto j = size_t{0}; j < y_h.size(); ++j)
            EXPECT_LE(fabs(y_ref[j] - y_h[j]), 0.0001);
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
        create_gdf_column(x_h, col_x[i]);
        #pragma omp barrier

        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        t = omp_get_wtime();
        status = gdf_snmg_csrmv(&part_offset[0], col_off, col_ind, col_val, col_x);
        EXPECT_EQ(status,0);
        #pragma omp master 
          {std::cout <<  omp_get_wtime() - t << " ";}


        #pragma omp master 
        { 
          CUDA_RT_CALL(cudaMemcpy(&y_h[0], col_x[0]->data,   sizeof(val_t) * m, cudaMemcpyDeviceToHost));

          for (auto j = size_t{0}; j < y_h.size(); ++j)
            EXPECT_LE(fabs(y_ref[j] - y_h[j]), 0.0001);
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
 

TEST_P(Tests_MGSpmv, CheckFP32_mtx) {
    run_current_test<int, float>(GetParam());
}
TEST_P(Tests_MGSpmv, CheckFP64) {
    run_current_test<int,double>(GetParam());
}

class Tests_MGSpmv_hibench : public ::testing::TestWithParam<MGSpmv_Usecase> {
  public:
  Tests_MGSpmv_hibench() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> mgspmv_time;   

  template <typename idx_t,typename val_t>
  void run_current_test(const MGSpmv_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, nnz, n_gpus;
     gdf_error status;
     std::vector<idx_t> cooRowInd, cooColInd;
     double t;

     ASSERT_EQ(read_single_file(param.matrix_file.c_str(),cooRowInd,cooColInd),0) << "read_single_file(" << param.matrix_file << ", ...) failure.";
     nnz = cooRowInd.size();
     m = 1 + std::max( *(std::max_element(cooRowInd.begin(), cooRowInd.end())),
                   *(std::max_element(cooColInd.begin(), cooColInd.end())));

     // Allocate memory on host
     std::vector<idx_t> csrColInd(nnz), csrRowPtr(m+1);
     std::vector<val_t> cooVal(nnz), csrVal(nnz), x_h(m, 1.0), y_h(m, 0.0), y_ref(m, 0.0);
     coo2csr(cooRowInd, cooColInd, csrRowPtr, csrColInd);
     CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));  
     std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus+1);
     random_vals(csrVal);
     random_vals(x_h);
     gdf_column *col_x[n_gpus];
     //reference result
     t = omp_get_wtime();
     csrmv_h (csrRowPtr, csrColInd, csrVal, x_h, y_ref);
     std::cout <<  omp_get_wtime() - t << " ";

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
        col_x[i] = new gdf_column;
        create_gdf_column(x_h, col_x[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        t = omp_get_wtime();
        status = gdf_snmg_csrmv(&part_offset[0], col_off, col_ind, col_val, col_x);
        EXPECT_EQ(status,0);
        #pragma omp master 
          {std::cout <<  omp_get_wtime() - t << " ";}


        #pragma omp master 
        { 
          CUDA_RT_CALL(cudaMemcpy(&y_h[0], col_x[0]->data,   sizeof(val_t) * m, cudaMemcpyDeviceToHost));

          for (auto j = size_t{0}; j < y_h.size(); ++j)
            EXPECT_LE(fabs(y_ref[j] - y_h[j]), 0.0001);
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
        create_gdf_column(x_h, col_x[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        t = omp_get_wtime();
        status = gdf_snmg_csrmv(&part_offset[0], col_off, col_ind, col_val, col_x);
        EXPECT_EQ(status,0);
        #pragma omp master 
          {std::cout <<  omp_get_wtime() - t << " ";}


        #pragma omp master 
        { 
          CUDA_RT_CALL(cudaMemcpy(&y_h[0], col_x[0]->data,   sizeof(val_t) * m, cudaMemcpyDeviceToHost));

          for (auto j = size_t{0}; j < y_h.size(); ++j)
            EXPECT_LE(fabs(y_ref[j] - y_h[j]), 0.0001);
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

TEST_P(Tests_MGSpmv_hibench, CheckFP32_hibench) {
    run_current_test<int, float>(GetParam());
}

class Tests_MGSpmv_unsorted : public ::testing::TestWithParam<MGSpmv_Usecase> {
  public:
  Tests_MGSpmv_unsorted() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> mgspmv_time;   


  template <typename idx_t,typename val_t>
  void run_current_test(const MGSpmv_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, k, nnz, n_gpus;
     MM_typecode mc;
     gdf_error status;

     double t;

     FILE* fpin = fopen(param.matrix_file.c_str(),"r");
     
     ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
     ASSERT_TRUE(mm_is_matrix(mc));
     ASSERT_TRUE(mm_is_coordinate(mc));
     ASSERT_FALSE(mm_is_complex(mc));
     ASSERT_FALSE(mm_is_skew(mc));
     
     // Allocate memory on host
     std::vector<idx_t> cooRowInd(nnz), cooColInd(nnz), csrColInd(nnz), csrRowPtr(m+1);
     std::vector<val_t> cooVal(nnz), csrVal(nnz), x_h(m, 1.0), y_h(m, 0.0), y_ref(m, 0.0);

     // Read
     ASSERT_EQ( (mm_to_coo<int,val_t>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], NULL, NULL)) , 0)<< "could not read matrix data"<< "\n";
     ASSERT_EQ(fclose(fpin),0);
     coo2csr(cooRowInd, cooColInd, csrRowPtr, csrColInd);

     //unsorted random indices
     for (auto i = 0; i < csrColInd.size(); i++)
      csrColInd[i]=static_cast<idx_t>(std::rand()%m);

     CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));  
     std::vector<size_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus+1);
     random_vals(csrVal);
     random_vals(x_h);
     gdf_column *col_x[n_gpus];
     //reference result
     t = omp_get_wtime();
     csrmv_h (csrRowPtr, csrColInd, csrVal, x_h, y_ref);
     std::cout <<  omp_get_wtime() - t << " ";
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
        col_x[i] = new gdf_column;
        create_gdf_column(x_h, col_x[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        t = omp_get_wtime();
        status = gdf_snmg_csrmv(&part_offset[0], col_off, col_ind, col_val, col_x);
        EXPECT_EQ(status,0);
        #pragma omp master 
          {std::cout <<  omp_get_wtime() - t << " ";}


        #pragma omp master 
        { 
          CUDA_RT_CALL(cudaMemcpy(&y_h[0], col_x[0]->data,   sizeof(val_t) * m, cudaMemcpyDeviceToHost));

          for (auto j = 0; j < y_h.size(); ++j)
            EXPECT_LE(fabs(y_ref[j] - y_h[j]), 0.0001);
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
        create_gdf_column(x_h, col_x[i]);
        #pragma omp barrier

        //load a chunck of the graph on each GPU 
        load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                     v_loc, e_loc, part_offset,
                     col_off, col_ind, col_val);
        t = omp_get_wtime();
        status = gdf_snmg_csrmv(&part_offset[0], col_off, col_ind, col_val, col_x);
        EXPECT_EQ(status,0);
        #pragma omp master 
          {std::cout <<  omp_get_wtime() - t << " ";}


        #pragma omp master 
        { 
          CUDA_RT_CALL(cudaMemcpy(&y_h[0], col_x[0]->data,   sizeof(val_t) * m, cudaMemcpyDeviceToHost));

          for (auto j = 0; j < y_h.size(); ++j)
            EXPECT_LE(fabs(y_ref[j] - y_h[j]), 0.0001);
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
 

TEST_P(Tests_MGSpmv_unsorted, CheckFP32_mtx) {
    run_current_test<int, float>(GetParam());
}
TEST_P(Tests_MGSpmv_unsorted, CheckFP64) {
    run_current_test<int,double>(GetParam());
}

INSTANTIATE_TEST_CASE_P(mtx_test, Tests_MGSpmv, 
                        ::testing::Values(   MGSpmv_Usecase("test/datasets/karate.mtx")
                                            ,MGSpmv_Usecase("test/datasets/netscience.mtx")
                                            ,MGSpmv_Usecase("test/datasets/cit-Patents.mtx")
                                            ,MGSpmv_Usecase("test/datasets/webbase-1M.mtx")
                                            ,MGSpmv_Usecase("test/datasets/web-Google.mtx")
                                            ,MGSpmv_Usecase("test/datasets/wiki-Talk.mtx")
                                         )
                       );

INSTANTIATE_TEST_CASE_P(mtx_test, Tests_MGSpmv_unsorted, 
                        ::testing::Values(   MGSpmv_Usecase("test/datasets/karate.mtx")
                                            ,MGSpmv_Usecase("test/datasets/netscience.mtx")
                                            ,MGSpmv_Usecase("test/datasets/cit-Patents.mtx")
                                            ,MGSpmv_Usecase("test/datasets/webbase-1M.mtx")
                                            ,MGSpmv_Usecase("test/datasets/web-Google.mtx")
                                            ,MGSpmv_Usecase("test/datasets/wiki-Talk.mtx")
                                         )
                       );
INSTANTIATE_TEST_CASE_P(hibench_test, Tests_MGSpmv_hibench,  
                        ::testing::Values(   MGSpmv_Usecase("benchmark/hibench/1/Input-small/edges/part-00000")
                                             ,MGSpmv_Usecase("benchmark/hibench/1/Input-large/edges/part-00000")
                                             ,MGSpmv_Usecase("benchmark/hibench/1/Input-huge/edges/part-00000")
                                         )
                       );
                      
int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
        
  return RUN_ALL_TESTS();
}


