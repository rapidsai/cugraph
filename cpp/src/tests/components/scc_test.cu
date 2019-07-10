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

// strongly connected components tests
// Author: Andrei Schaffer aschaffer@nvidia.com

//#define DEBUG

#define _DEBUG_CC

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"

#include <cugraph.h>
#include "test_utils.h"
#include <algorithm>
#include <iterator>

#include "components/scc_matrix.cuh"
#include "topology/topology.cuh"

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

struct Tests_Strongly_CC : ::testing::TestWithParam<Usecase>
{
  Tests_Strongly_CC() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { 
    if (PERF) {
     for (unsigned int i = 0; i < strongly_cc_time.size(); ++i) {
      std::cout <<  strongly_cc_time[i] << std::endl;
     }
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> strongly_cc_time;

  void run_current_test(const Usecase& param) {
    const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss; 
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.get_matrix_file())+ std::string("_") + ss.str().c_str();
    ///cudaStream_t stream{nullptr};

    using ByteT = unsigned char;
    using IndexT = int;

    IndexT m, k, nnz; //
    MM_typecode mc;
     
    HighResClock hr_clock;
    double time_tmp;

    FILE* fpin = fopen(param.get_matrix_file().c_str(),"r");
    ASSERT_TRUE( fpin != nullptr );

    ASSERT_EQ(mm_properties<IndexT>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_TRUE(mm_is_symmetric(mc));//strongly cc only works w/ undirected graphs, for now;

    //rmmInitialize(nullptr);

#ifdef _DEBUG_CC 
    std::cout<<"matrix nrows: "<<m<<"\n";
    std::cout<<"matrix nnz: "<<nnz<<"\n";
#endif

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    size_t nrows = static_cast<size_t>(m);
    size_t n2 = 2*nrows * nrows;

    //bail out if not enough memory:
    //
    ASSERT_TRUE( n2 < prop.totalGlobalMem );
   
    
    // Allocate memory on host
    std::vector<IndexT> cooRowInd(nnz);
    std::vector<IndexT> cooColInd(nnz);
    std::vector<IndexT> cooVal(nnz);
    std::vector<IndexT> labels(m);//for G(V, E), m := |V|

    // Read: COO Format
    //
    ASSERT_EQ( (mm_to_coo<IndexT,IndexT>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
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

    static auto row_offsets_ = [](const gdf_graph* G){
      return static_cast<const IndexT*>(G->adjList->offsets->data);
    };

    static auto col_indices_ = [](const gdf_graph* G){
      return static_cast<const IndexT*>(G->adjList->indices->data);
    };

    static auto nrows_ = [](const gdf_graph* G){
      return G->adjList->offsets->size - 1;
    };

    // static auto nnz_ = [](const gdf_graph* G){
    //   return G->adjList->indices->size;
    // };


    SCC_Data<ByteT> sccd(nrows_(G.get()), row_offsets_(G.get()), col_indices_(G.get()));
    IndexT* p_d_labels = static_cast<IndexT*>(col_labels->data);
    size_t count = 0;

    gdf_error status{GDF_SUCCESS};
    if (PERF)
      {
        hr_clock.start();
        //call strongly connected components
        //
        count = sccd.run_scc(p_d_labels);
        
        cudaDeviceSynchronize();
        hr_clock.stop(&time_tmp);
        strongly_cc_time.push_back(time_tmp);
      }
    else
      {
        cudaProfilerStart();
        //call strongly connected components
        //
        count = sccd.run_scc(p_d_labels);
        
        cudaProfilerStop();
        cudaDeviceSynchronize();
      }
    EXPECT_EQ(status,GDF_SUCCESS);

    std::cout << "#iterations: " << count << "\n";
    std::cout <<"labels:\n";
    cudaMemcpy(&labels[0], p_d_labels, m*sizeof(IndexT), cudaMemcpyDeviceToHost);
    print_v(labels, std::cout);

    std::vector<IndexT> l_check(m);//for G(V, E), m := |V|
    gdf_column_ptr check_labels = create_gdf_column(l_check);

    status = gdf_connected_components(G.get(),
                                      CUGRAPH_WEAK,
                                      check_labels.get());

    EXPECT_EQ(status,GDF_SUCCESS);

    IndexT* p_d_l_check = static_cast<IndexT*>(check_labels->data);
    std::cout <<"check labels:\n";
    cudaMemcpy(&l_check[0], p_d_l_check, m*sizeof(IndexT), cudaMemcpyDeviceToHost);
    print_v(l_check, std::cout);
    
    //rmmFinalize();
  }
};
 
std::vector<double> Tests_Strongly_CC::strongly_cc_time;

TEST_P(Tests_Strongly_CC, Strongly_CC) {
    run_current_test(GetParam());
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Strongly_CC, 
                        ::testing::Values(Usecase("test/datasets/dolphins.mtx")//, //okay
                                          //Usecase("test/datasets/coPapersDBLP.mtx")//, //fails (not enough memory)
                                          //Usecase("test/datasets/coPapersCiteseer.mtx")//,fails (not enough memory)
                                          //Usecase("test/datasets/hollywood.mtx")//fails (not enough memory)
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


