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

// Force_Atlas2 solver tests
// Author: Alex Fender afender@nvidia.com

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"
#include "test_utils.h"
#include <rmm/thrust_rmm_allocator.h>
#include <converters/COOtoCSR.cuh>
#include <graph.hpp>
#include <algorithms.hpp>
#include <iostream>
#include <fstream>

// do the perf measurements
// enabled by command line parameter s'--perf'
static int PERF = 0;

// iterations for perf tests
// enabled by command line parameter '--perf-iters"
static int PERF_MULTIPLIER = 5;

typedef struct Force_Atlas2_Usecase_t {
  std::string matrix_file;
  std::string result_file;
  Force_Atlas2_Usecase_t(const std::string& a, const std::string& b) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
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
  Force_Atlas2_Usecase_t& operator=(const Force_Atlas2_Usecase_t& rhs) {
    matrix_file = rhs.matrix_file;
    result_file = rhs.result_file;
    return *this;
  }
} Force_Atlas2_Usecase;

class Tests_Force_Atlas2 : public ::testing::TestWithParam<Force_Atlas2_Usecase> {
  public:
  Tests_Force_Atlas2() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { 
    if (PERF) {
     for (unsigned int i = 0; i < force_atlas2_time.size(); ++i) {
      std::cout <<  force_atlas2_time[i]/PERF_MULTIPLIER << std::endl;
     }
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> force_atlas2_time;   


  template <typename T>
  void run_current_test(const Force_Atlas2_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, k, nnz;
     MM_typecode mc;
     
     HighResClock hr_clock;
     double time_tmp;

     FILE* fpin = fopen(param.matrix_file.c_str(),"r");
     ASSERT_NE(fpin, nullptr) << "fopen (" << param.matrix_file << ") failure.";
     
     ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
     ASSERT_TRUE(mm_is_matrix(mc));
     ASSERT_TRUE(mm_is_coordinate(mc));
     ASSERT_FALSE(mm_is_complex(mc));
     ASSERT_FALSE(mm_is_skew(mc));
     
     // Allocate memory on host
     std::vector<int> cooRowInd(nnz), cooColInd(nnz);
     std::vector<T> cooVal(nnz);
     std::vector<float> force_atlas2(m * 2);
     
     //device alloc
     rmm::device_vector<float> force_atlas2_vector(m * 2);
     float* d_force_atlas2 = force_atlas2_vector.data().get();

     // Read
     ASSERT_EQ( (mm_to_coo<int,T>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
     ASSERT_EQ(fclose(fpin),0);


     cugraph::experimental::GraphCOO<int,int,T> G(&cooRowInd[0], &cooColInd[0], &cooVal[0], m, nnz);

     std::cout << m << " "<< nnz << "\n";
    
     cudaDeviceSynchronize();

     const int max_iter=1;
     float *x_start = nullptr;
     float *y_start = nullptr;
     bool outbound_attraction_distribution = true;
     bool lin_log_mode = false;
     bool prevent_overlapping = false;
     const float edge_weight_influence = 1.0;
     const float jitter_tolerance = 1.0;
     bool optimize = true;
     const float theta = 0.5;
     const float scaling_ratio = 2.0;
     bool strong_gravity_mode = false;
     const float gravity = 1.0;
     bool verbose = false;
     
     if (PERF) {
       hr_clock.start();
       for (int i = 0; i < PERF_MULTIPLIER; ++i) {
           cugraph::force_atlas2<int,int,T>(G, d_force_atlas2, max_iter,
            x_start, y_start, outbound_attraction_distribution, lin_log_mode,
            prevent_overlapping, edge_weight_influence, jitter_tolerance,
            optimize, theta, scaling_ratio, strong_gravity_mode, gravity,
            verbose);
           cudaDeviceSynchronize();
       }
       hr_clock.stop(&time_tmp);
       force_atlas2_time.push_back(time_tmp);
     } else {
         std::cout << "Start algo\n";
         cudaProfilerStart();
         cugraph::force_atlas2<int,int,T>(G, d_force_atlas2, max_iter,
            x_start, y_start, outbound_attraction_distribution, lin_log_mode,
            prevent_overlapping, edge_weight_influence, jitter_tolerance,
            optimize, theta, scaling_ratio, strong_gravity_mode, gravity,
            verbose);
         cudaProfilerStop();
         cudaDeviceSynchronize();
    }
  }
};
 
std::vector<double> Tests_Force_Atlas2::force_atlas2_time;

TEST_P(Tests_Force_Atlas2, CheckFP32_T) {
    run_current_test<float>(GetParam());
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Force_Atlas2, 
        ::testing::Values(Force_Atlas2_Usecase("test/datasets/karate.mtx", "")//,
           // Force_Atlas2_Usecase("test/datasets/citationCiteseer.mtx", ""),
           // Force_Atlas2_Usecase("test/datasets/dblp.mtx", ""),
           // Force_Atlas2_Usecase("test/datasets/web-Google.mtx", ""),
           // Force_Atlas2_Usecase("test/datasets/webbase-1M.mtx", ""),
           // Force_Atlas2_Usecase("test/datasets/ljournal-2008.mtx", "")
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
