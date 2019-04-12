/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"
#include <cugraph.h>
#include <omp.h>
#include "test_utils.h"
//#include "snmg_test_utils.h"

#define SNMG_VERBOSE

template <typename idx_t,typename val_t>
void csrmv_h (std::vector<idx_t> & off_h, 
                      std::vector<idx_t> & ind_h, 
                      std::vector<val_t> & val_h,  
                      std::vector<val_t> & x,  
                      std::vector<val_t> & y) {
#pragma omp for
for (auto i = 0; i < off_h.size(); ++i) 
  for (auto j = off_h[i]; j <  off_h[i+1]; ++j) 
    y[i] += val_h[j]*x[ind_h[j]];
}

// global to local offsets by shifting all offsets by the first offset value
template <typename T>
void shift_offsets(std::vector<T> & off_loc) {
  auto start = off_loc.front();
  for (auto i = 0; i < off_loc.size(); ++i)
    off_loc[i] -= start;
}

// 1D partitioning such as each GPU has about the same number of edges
template <typename T>
void edge_partioning(std::vector<T> & off_h, std::vector<T> & part_offset, std::vector<T> & v_loc, std::vector<T> & e_loc) {
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads();

  //set first and last partition offsets
  part_offset[0] = 0;
  part_offset[p] = off_h.size()-1;
  
  if (i>0) {
    //get the first vertex ID of each partition
    auto loc_nnz = off_h.back()/p;
    auto start_nnz = i*loc_nnz;
    auto start_v = 0;
    for (auto j = 0; j < off_h.size(); ++j) {
      if (off_h[j] > start_nnz) {
        start_v = j;
        break;
      }
    }
    part_offset[i] = start_v;
  }
  // all threads must know their partition offset 
  #pragma omp barrier 

  // Store the local number of V and E for convinience
  v_loc[i] = part_offset[i+1] - part_offset[i];
  e_loc[i] = off_h[part_offset[i+1]] - off_h[part_offset[i]];
}

template <typename idx_t,typename val_t>
void load_csr_loc(std::vector<idx_t> & off_h, std::vector<idx_t> & ind_h, std::vector<val_t> & val_h, 
                  std::vector<idx_t> & v_loc, std::vector<idx_t> & e_loc, std::vector<idx_t> & part_offset,
                  gdf_column* col_off, gdf_column* col_ind, gdf_column* col_val)
{
 
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads(); 
  edge_partioning(off_h, part_offset, v_loc, e_loc);

  std::vector<idx_t> off_loc(off_h.begin()+part_offset[i],off_h.begin()+part_offset[i+1]), 
                     ind_loc(ind_h.begin()+part_offset[i],ind_h.begin()+part_offset[i+1]), 
                     val_loc(val_h.begin()+part_offset[i],val_h.begin()+part_offset[i+1]);

  #ifdef SNMG_VERBOSE
  #pragma omp barrier 
  #pragma omp master 
  { 
    for (auto j = part_offset.begin(); j != part_offset.end(); ++j)
      std::cout << *j << ' ';
    std::cout << std::endl;
    for (auto j = v_loc.begin(); j != v_loc.end(); ++j)
      std::cout << *j << ' ';
    std::cout << std::endl;  
    for (auto j = e_loc.begin(); j != e_loc.end(); ++j)
      std::cout << *j << ' ';
    std::cout << std::endl;
  }
  #pragma omp barrier 
  #endif


  shift_offsets(off_loc);

  create_gdf_column(off_loc, col_off);
  create_gdf_column(ind_loc, col_ind);
  create_gdf_column(val_loc, col_val);
}


typedef struct MGSpmv_Usecase_t {
  std::string matrix_file;
  MGSpmv_Usecase_t(const std::string& a) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // if RAPIDS_DATASET_ROOT_DIR not set, default to "/datasets"
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir("/datasets");
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
     ASSERT_EQ( (mm_to_coo<int,val_t>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
     ASSERT_EQ(fclose(fpin),0);
     ASSERT_EQ( (coo_to_csr<int,val_t> (m, m, nnz, &cooRowInd[0],  &csrColInd[0], &csrVal[0], NULL, &csrRowPtr[0], NULL, NULL, NULL)), 0) << "could not covert COO to CSR "<< "\n";

     CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));  
     std::vector<idx_t> v_loc(n_gpus), e_loc(n_gpus), part_offset(n_gpus+1);
     random_vals(csrVal);

     //reference result
     csrmv_h (csrRowPtr, csrColInd, csrVal, x_h, y_ref);

     #pragma omp parallel num_threads(n_gpus)
     {
      //omp_set_num_threads(n_gpus);
      auto i = omp_get_thread_num();
      auto p = omp_get_num_threads(); 
      #ifdef SNMG_VERBOSE 
        #pragma omp master 
        { 
          std::cout << "Number of GPUs : "<< n_gpus <<std::endl;
          std::cout << "Number of threads : "<< p <<std::endl;
        }
      #endif

      gdf_column *col_off = new gdf_column, 
                 *col_ind = new gdf_column, 
                 *col_val = new gdf_column,
                 *col_x = new gdf_column,
                 *col_y = new gdf_column;

      CUDA_RT_CALL(cudaSetDevice(i));

      //load a chunck of the graph on each GPU 
      load_csr_loc(csrRowPtr, csrColInd, csrVal, 
                   v_loc, e_loc, part_offset,
                   col_off, col_ind, col_val);

      create_gdf_column(x_h, col_x);
      create_gdf_column(y_h, col_y);

      status = gdf_snmg_csrmv(col_off, col_ind, col_val, col_x, col_y);
      EXPECT_EQ(status,0);

      CUDA_RT_CALL(cudaMemcpy(&y_h[0], col_y->data,   sizeof(val_t) * m, cudaMemcpyDeviceToHost));

      // for (auto j = 0; j < y_h.size(); ++j)
      //  EXPECT_LE(fabs(y_ref[j] - y_h[j]), 0.0001);

      gdf_col_delete(col_off);
      gdf_col_delete(col_ind);
      gdf_col_delete(col_val);
      gdf_col_delete(col_x);
      gdf_col_delete(col_y);
    }
  }
};
 

TEST_P(Tests_MGSpmv, CheckFP32) {
    run_current_test<int, float>(GetParam());
}

//TEST_P(Tests_MGSpmv, CheckFP64) {
//    run_current_test<int,double>(GetParam());
//}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_MGSpmv, 
                        ::testing::Values(  MGSpmv_Usecase("networks/karate.mtx")
                                            ,MGSpmv_Usecase("golden_data/graphs/cit-Patents.mtx")
                                            ,MGSpmv_Usecase("golden_data/graphs/ljournal-2008.mtx")
                                            //,MGSpmv_Usecase("golden_data/graphs/webbase-1M.mtx")
                                            //,MGSpmv_Usecase("golden_data/graphs/web-BerkStan.mtx")
                                            //,MGSpmv_Usecase("golden_data/graphs/web-Google.mtx")
                                            //,MGSpmv_Usecase("golden_data/graphs/wiki-Talk.mtx")
                                         )
                       );


int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
        
  return RUN_ALL_TESTS();
}


