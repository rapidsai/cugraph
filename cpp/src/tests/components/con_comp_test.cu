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
    std::vector<int> verts(m);

    // Read: COO Format
    //
    ASSERT_EQ( (mm_to_coo<int,int>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
    ASSERT_EQ(fclose(fpin),0);

    Graph_ptr G{new cugraph::Graph, Graph_deleter};
    gdf_column_ptr col_src;
    gdf_column_ptr col_dest;
    gdf_column_ptr col_labels;
    gdf_column_ptr col_verts;

    col_src = create_gdf_column(cooRowInd);
    col_dest = create_gdf_column(cooColInd);
    col_labels = create_gdf_column(labels);
    col_verts = create_gdf_column(verts);

    std::vector<gdf_column*> vcols{col_labels.get(), col_verts.get()};
    cudf::table table(vcols);

    //Get the COO format 1st:
    //
    cugraph::edge_list_view(G.get(), col_src.get(), col_dest.get(), nullptr);

    //Then convert to CSR:
    //
    cugraph::add_adj_list(G.get());

    
    if (PERF)
      {
        hr_clock.start();
        cugraph::connected_components(G.get(),
                                          cugraph::CUGRAPH_WEAK,
                                          &table);

        cudaDeviceSynchronize();
        hr_clock.stop(&time_tmp);
        weakly_cc_time.push_back(time_tmp);
      }
    else
      {
        cudaProfilerStart();
        cugraph::connected_components(G.get(),
                                          cugraph::CUGRAPH_WEAK,
                                          &table);
        cudaProfilerStop();
        cudaDeviceSynchronize();
      }
    
    
    //rmmFinalize();
  }
};
 
std::vector<double> Tests_Weakly_CC::weakly_cc_time;

TEST_P(Tests_Weakly_CC, Weakly_CC) {
    run_current_test(GetParam());
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Weakly_CC, 
                        ::testing::Values(   Usecase("test/datasets/dolphins.mtx")
                                           , Usecase("test/datasets/coPapersDBLP.mtx")
                                           , Usecase("test/datasets/coPapersCiteseer.mtx")
                                           , Usecase("test/datasets/hollywood.mtx")
					  ));


int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}


