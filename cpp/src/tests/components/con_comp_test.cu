// -*-c++-*-

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// connected components tests
// Author: Andrei Schaffer aschaffer@nvidia.com

//#define DEBUG

#define _DEBUG_C2C

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include <cudf.h>
#include "cuda_profiler_api.h"

#include <cugraph.h>
#include "test_utils.h"
#include <algorithm>
#include <iterator>

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 0;

namespace{ //un-nammed
  struct Usecase
  {
    explicit Usecase(const std::string& a) {
      // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
      const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
      if ((a != "") && (a[0] != '/')) {
	matrix_file = rapidsDatasetRootDir + "/" + a;
      } else {
	matrix_file = a;
      }
    }

    Usecase(const Usecase& rhs)
    {
      matrix_file = rhs.matrix_file;
    }
    
    Usecase& operator = (const Usecase& rhs)
    {
      matrix_file = rhs.matrix_file;

      return *this;
    }
    const std::string& get_matrix_file(void) const
    {
      return matrix_file;
    }
  private:
    std::string matrix_file;
  };
  
}//end un-nammed namespace

struct Tests_Weakly_CC : ::testing::TestWithParam<Usecase>
{
  Tests_Weakly_CC() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { 
    if (PERF) {
     for (unsigned int i = 0; i < weakly_cc_time.size(); ++i) {
      std::cout <<  weakly_cc_time[i] << std::endl;
     }
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> weakly_cc_time;

  void run_current_test(const Usecase& param) {
    const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss; 
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.get_matrix_file())+ std::string("_") + ss.str().c_str();
    cudaStream_t stream{nullptr};

    int m, k, nnz; //
    MM_typecode mc;
     
    HighResClock hr_clock;
    double time_tmp;

    FILE* fpin = fopen(param.get_matrix_file().c_str(),"r");
    ASSERT_TRUE( fpin != nullptr );

    ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_TRUE(mm_is_symmetric(mc));//weakly cc only works w/ undirected graphs, for now;

    //rmmInitialize(nullptr);

#ifdef _DEBUG_WEAK_CC 
    std::cout<<"matrix nrows: "<<m<<"\n";
    std::cout<<"matrix nnz: "<<nnz<<"\n";
#endif
    
    // Allocate memory on host
    std::vector<int> cooRowInd(nnz);
    std::vector<int> cooColInd(nnz);
    std::vector<int> cooVal(nnz);
    std::vector<int> labels(m);//for G(V, E), m := |V|

    // Read: COO Format
    //
    ASSERT_EQ( (mm_to_coo<int,int>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
    ASSERT_EQ(fclose(fpin),0);

    gdf_graph_ptr G{new gdf_graph, gdf_graph_deleter};
    gdf_column_ptr col_src, col_dest, col_labels;

    col_src = create_gdf_column(cooRowInd);
    col_dest = create_gdf_column(cooColInd);
    col_labels = create_gdf_column(labels);

    //Get the COO format 1st:
    //
    ASSERT_EQ(gdf_edge_list_view(G.get(), col_src.get(), col_dest.get(), nullptr),0);

    //Then convert to CSR:
    //
    ASSERT_EQ(gdf_add_adj_list(G.get()),0);

    gdf_error status;
    if (PERF)
      {
        hr_clock.start();
        status = gdf_connected_components(G.get(),
                                          CUGRAPH_WEAK,
                                          col_labels.get());

        cudaDeviceSynchronize();
        hr_clock.stop(&time_tmp);
        weakly_cc_time.push_back(time_tmp);
      }
    else
      {
        cudaProfilerStart();
        status = gdf_connected_components(G.get(),
                                          CUGRAPH_WEAK,
                                          col_labels.get());
        cudaProfilerStop();
        cudaDeviceSynchronize();
      }
    EXPECT_EQ(status,GDF_SUCCESS);
    
    //rmmFinalize();
  }
};
 
std::vector<double> Tests_Weakly_CC::weakly_cc_time;

TEST_P(Tests_Weakly_CC, Weakly_CC) {
    run_current_test(GetParam());
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Weakly_CC, 
                        ::testing::Values(   Usecase("networks/dolphins.mtx")
                                           , Usecase("networks/coPapersDBLP.mtx")
                                           , Usecase("networks/coPapersCiteseer.mtx")
                                           , Usecase("networks/hollywood.mtx")
					  ));


int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--perf") == 0)
            PERF = 1;
    }

  return RUN_ALL_TESTS();
}


