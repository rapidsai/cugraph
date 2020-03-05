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

//#define _DEBUG_CC

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"

#include <thrust/sequence.h>
#include <thrust/unique.h>
//
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

template<typename T>
using DVector = thrust::device_vector<T>;

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

  //checker of counts of labels for each component
  //expensive, for testing purposes only;
  //
  //params:
  //p_d_labels: device array of labels of size nrows;
  //nrows: |V| for graph G(V, E);
  //d_v_counts: #labels for each component; (_not_ pre-allocated!)
  //
  template<typename IndexT>
  size_t get_component_sizes(const IndexT* p_d_labels,
                             size_t nrows,
                             DVector<size_t>& d_v_counts)
  {
    DVector<IndexT> d_sorted_l(p_d_labels, p_d_labels+nrows);
    thrust::sort(d_sorted_l.begin(), d_sorted_l.end());

    size_t counts = thrust::distance(d_sorted_l.begin(),
                                     thrust::unique(d_sorted_l.begin(), d_sorted_l.end()));

    IndexT* p_d_srt_l = d_sorted_l.data().get();

    d_v_counts.resize(counts);
    thrust::transform(thrust::device,
                      d_sorted_l.begin(), d_sorted_l.begin() + counts,  
                      d_v_counts.begin(),
                      [p_d_srt_l, counts] __device__ (IndexT indx){
                        return thrust::count_if(thrust::seq,
                                                p_d_srt_l, p_d_srt_l+counts,
                                                [indx] (IndexT label){
                                                  return label == indx;
                                                });
                      });

    //sort the counts:
    thrust::sort(d_v_counts.begin(), d_v_counts.end());
    
    return counts;
  }
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
     
     std::cout<<"#iterations:\n";
     for(auto&& count: strongly_cc_counts)
       std::cout << count << std::endl;
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> strongly_cc_time;
  static std::vector<int> strongly_cc_counts;

  void run_current_test(const Usecase& param) {
    const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss; 
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.get_matrix_file())+ std::string("_") + ss.str().c_str();

    using ByteT = unsigned char;
    using IndexT = int;

    IndexT m, k, nnz;
    MM_typecode mc;
     
    HighResClock hr_clock;
    double time_tmp;

    FILE* fpin = fopen(param.get_matrix_file().c_str(),"r");
    ASSERT_NE(fpin, nullptr) << "fopen (" << param.get_matrix_file().c_str() << ") failure.";

    ASSERT_EQ(mm_properties<IndexT>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    //ASSERT_TRUE(mm_is_symmetric(mc));//strongly cc works with both DG and UDG;

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
    std::vector<IndexT> verts(m);

    // Read: COO Format
    //
    ASSERT_EQ( (mm_to_coo<IndexT,IndexT>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
    ASSERT_EQ(fclose(fpin),0);

    Graph_ptr G{new cugraph::Graph, Graph_deleter};
    gdf_column_ptr col_src;
    gdf_column_ptr col_dest;
    gdf_column_ptr col_labels;
    gdf_column_ptr col_verts;// = thrust::sequence(0..m-1);

    col_src = create_gdf_column(cooRowInd);
    col_dest = create_gdf_column(cooColInd);
    col_labels = create_gdf_column(labels);
    col_verts = create_gdf_column(verts);

    IndexT* begin = static_cast<IndexT*>(col_verts->data);
    thrust::sequence(thrust::device, begin, begin + m);
    
    std::vector<gdf_column*> vcols{col_labels.get(), col_verts.get()};
    cudf::table table(vcols);//to be passed to the API to be filled

    //Get the COO format 1st:
    //
    cugraph::edge_list_view(G.get(), col_src.get(), col_dest.get(), nullptr);

    //Then convert to CSR:
    //
    cugraph::add_adj_list(G.get());

    static auto row_offsets_ = [](const cugraph::Graph* G){
      return static_cast<const IndexT*>(G->adjList->offsets->data);
    };

    static auto col_indices_ = [](const cugraph::Graph* G){
      return static_cast<const IndexT*>(G->adjList->indices->data);
    };

    static auto nrows_ = [](const cugraph::Graph* G){
      return G->adjList->offsets->size - 1;
    };

    SCC_Data<ByteT> sccd(nrows_(G.get()), row_offsets_(G.get()), col_indices_(G.get()));
    IndexT* p_d_labels = static_cast<IndexT*>(col_labels->data);
    size_t count = 0;

    if (PERF)
      {
        hr_clock.start();
        //call strongly connected components
        //
        ///count = sccd.run_scc(p_d_labels);
        cugraph::connected_components(G.get(),
                                          cugraph::CUGRAPH_STRONG,
                                          &table);
        
        cudaDeviceSynchronize();
        hr_clock.stop(&time_tmp);
        strongly_cc_time.push_back(time_tmp);    
      }
    else
      {
        cudaProfilerStart();
        //call strongly connected components
        //
        ///count = sccd.run_scc(p_d_labels);
        cugraph::connected_components(G.get(),
                                          cugraph::CUGRAPH_STRONG,
                                          &table);
        
        cudaProfilerStop();
        cudaDeviceSynchronize();
      }
    strongly_cc_counts.push_back(count);

    DVector<size_t> d_counts;
    auto count_labels = get_component_sizes(p_d_labels, nrows, d_counts);

    std::cout<<"label count: " << count_labels << "\n";
    
    

#ifdef DEBUG_SCC
    std::cout << "#iterations: " << count << "\n";
    std::cout <<"labels:\n";
    cudaMemcpy(&labels[0], p_d_labels, m*sizeof(IndexT), cudaMemcpyDeviceToHost);
    print_v(labels, std::cout);

    DVector<IndexT> d_cct(nrows*nrows);//{sccd.get_C()};//copy to int array for printing:
    thrust::transform(sccd.get_C().begin(), sccd.get_C().end(),
                      d_cct.begin(),
                      [] __device__ (ByteT b){
                        return (b == ByteT{1}? 1:0);
                      });
    
    std::cout<<"C & Ct:\n";
    print_v(d_cct, std::cout);
#endif
    
    //if graph is undirected, check against
    //WeaklyCC:
    //
    if( mm_is_symmetric(mc) )
      {
        std::vector<IndexT> l_check(m);//for G(V, E), m := |V|
        gdf_column_ptr check_labels = create_gdf_column(l_check);
        
        cugraph::connected_components(G.get(),
                                          cugraph::CUGRAPH_WEAK,
                                          &table);

        

        IndexT* p_d_l_check = static_cast<IndexT*>(check_labels->data);
        std::cout <<"check labels:\n";
        cudaMemcpy(&l_check[0], p_d_l_check, m*sizeof(IndexT), cudaMemcpyDeviceToHost);
        print_v(l_check, std::cout);
      }
    //rmmFinalize();
  }
};
 
std::vector<double> Tests_Strongly_CC::strongly_cc_time;
std::vector<int> Tests_Strongly_CC::strongly_cc_counts;

TEST_P(Tests_Strongly_CC, Strongly_CC) {
    run_current_test(GetParam());
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Strongly_CC, 
                        ::testing::Values(Usecase("test/datasets/cage6.mtx") //DG "small" enough to meet SCC GPU memory requirements
					  ));


int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}


