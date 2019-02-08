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

// Pagerank solver tests
// Author: Alex Fender afender@nvidia.com

#include "gtest/gtest.h"
#include <cugraph.h>
#include "cuda_profiler_api.h"
#include "test_utils.h"
#include <mpi.h>
typedef struct Pagerank_Usecase_t {
  std::string matrix_file;
  std::string result_file;
  Pagerank_Usecase_t(const std::string& a, const std::string& b) : matrix_file(a), result_file(b){};
  Pagerank_Usecase_t& operator=(const Pagerank_Usecase_t& rhs) {
    matrix_file = rhs.matrix_file;
    result_file = rhs.result_file;
    return *this;
  }
} Pagerank_Usecase;

class Tests_Pagerank : public ::testing::TestWithParam<Pagerank_Usecase> {
  public:
  Tests_Pagerank() {  }
  virtual void SetUp() {  
    char **argv = nullptr;
    int argc = 0;
    MPI_Init(&argc, &argv);
  }
  virtual void TearDown() { MPI_Finalize(); }

  template <typename idx_T>
  void run_current_test(const Pagerank_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, k, nnz;
     MM_typecode mc;
     
     gdf_error status;
     float damping_factor=0.85;
     float tol = 1E-5f;
     int max_iter=20;
     gdf_column *col_src = new gdf_column, 
                *col_dest = new gdf_column, 
                *col_pagerank = new gdf_column, 
                *col_vidx = new gdf_column;
  

     FILE* fpin = fopen(param.matrix_file.c_str(),"r");
     
     ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
     ASSERT_TRUE(mm_is_matrix(mc));
     ASSERT_TRUE(mm_is_coordinate(mc));
     ASSERT_FALSE(mm_is_complex(mc));
     ASSERT_FALSE(mm_is_skew(mc));
     
     // Allocate memory on host
     std::vector<int> cooRowInd(nnz), cooColInd(nnz);
     std::vector<float> cooVal(nnz), pagerank(m);

     // Read
     ASSERT_EQ( (mm_to_coo<int,float>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
     ASSERT_EQ(fclose(fpin),0);
    
    // gdf columns (transposing here)
    create_gdf_column(cooColInd, col_src);
    create_gdf_column(cooRowInd, col_dest);

    // solve
    cudaProfilerStart();
    status =  gdf_multi_pagerank (m, col_src, col_dest, col_vidx, col_pagerank, damping_factor, max_iter);
    cudaProfilerStop();
    cudaDeviceSynchronize();
    EXPECT_EQ(status,0);

    // Print
    // std::vector<float> calculated_res(m);
    // CUDA_RT_CALL(cudaMemcpy(&calculated_res[0],   col_pagerank->data,   sizeof(float) * m, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < m; i++)
    //       std::cout << i << " "<< calculated_res[i] <<std::endl;
    
    // Check vs golden data
    if (param.result_file.length()>0)
    {
      std::vector<float> calculated_res(m);
      CUDA_RT_CALL(cudaMemcpy(&calculated_res[0],   col_pagerank->data,   sizeof(float) * m, cudaMemcpyDeviceToHost));
      std::sort(calculated_res.begin(), calculated_res.end());
      fpin = fopen(param.result_file.c_str(),"rb");
      ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file << std::endl;
      std::vector<float> expected_res(m);
      ASSERT_EQ(read_binary_vector(fpin, m, expected_res), 0);
      fclose(fpin);
      float err;
      int n_err = 0;
      for (int i = 0; i < m; i++)
      {
          if(i > (m-20)) std::cout << expected_res[i] << " " << calculated_res[i] <<std::endl;
          err = fabs(expected_res[i] - calculated_res[i]);
          if (err> tol*1.1)
          {
              n_err++;
          }
      }
      if (n_err)
      {
          EXPECT_LE(n_err, 0.001*m); // we tolerate 0.1% of values with a litte difference
      }
    }

    gdf_col_delete(col_src);
    gdf_col_delete(col_dest);
    gdf_col_delete(col_pagerank);
    gdf_col_delete(col_vidx);
  }
};
 
TEST_P(Tests_Pagerank, Check32) {
    run_current_test<int>(GetParam());
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Pagerank, 
                        ::testing::Values(  //Pagerank_Usecase("/datasets/networks/karate.mtx", "")
                                            //,Pagerank_Usecase("/datasets/golden_data/graphs/cit-Patents.mtx", "/datasets/golden_data/results/pagerank/cit-Patents.pagerank_val_0.85.bin")
                                            //,Pagerank_Usecase("/datasets/golden_data/graphs/ljournal-2008.mtx", "/datasets/golden_data/results/pagerank/ljournal-2008.pagerank_val_0.85.bin")
                                            //,Pagerank_Usecase("/datasets/golden_data/graphs/webbase-1M.mtx", "/datasets/golden_data/results/pagerank/webbase-1M.pagerank_val_0.85.bin")
                                            //Pagerank_Usecase("/datasets/golden_data/graphs/web-BerkStan.mtx", "/datasets/golden_data/results/pagerank/web-BerkStan.pagerank_val_0.85.bin")
                                            //Pagerank_Usecase("/datasets/golden_data/graphs/web-Google.mtx", "/datasets/golden_data/results/pagerank/web-Google.pagerank_val_0.85.bin")
                                            //Pagerank_Usecase("/datasets/golden_data/graphs/ibm32.mtx", "")
                                            Pagerank_Usecase("/datasets/golden_data/graphs/wiki-Talk.mtx", "/datasets/golden_data/results/pagerank/wiki-Talk.pagerank_val_0.85.bin")
                                         )
                       );

int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


