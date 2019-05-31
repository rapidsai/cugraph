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
#include <gtest/gtest.h>
#include <nvgraph/nvgraph.h>
#include <cugraph.h>
#include <nvgraph_gdf.h>
#include "test_utils.h"

// do the perf measurements
// enabled by command line parameter s'--perf'
static int PERF = 0;

// iterations for perf tests
// enabled by command line parameter '--perf-iters"
static int PERF_MULTIPLIER = 5;

typedef struct Sssp_Usecase_t {
  std::string grmat_param;
  int src;
  Sssp_Usecase_t(const std::string& a, int b) : grmat_param(a), src(b){};
  Sssp_Usecase_t& operator=(const Sssp_Usecase_t& rhs) {
    grmat_param = rhs.grmat_param;
    src = rhs.src;
    return *this;
  }
} Sssp_Usecase;

class Tests_Sssp : public ::testing::TestWithParam<Sssp_Usecase> {
  public:
  Tests_Sssp() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { 
    if (PERF) {
     for (unsigned int i = 0; i < sssp_time.size(); ++i) {
      std::cout <<  sssp_time[i]/PERF_MULTIPLIER << std::endl;
     }
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> sssp_time;   

  template <typename T, bool manual_tanspose>
  void run_current_test(const Sssp_Usecase& param) {
    gdf_column col_src, col_dest, col_w;
    size_t v,e;
    col_src.data = nullptr;
    col_src.dtype = GDF_INT32;
    col_src.valid = nullptr;
    col_dest.dtype = GDF_INT32;
    col_dest.valid = nullptr;

    col_src.null_count = 0;
    col_dest.null_count = 0;
    ASSERT_EQ(gdf_grmat_gen (param.grmat_param.c_str(), v, e, &col_src, &col_dest, nullptr),GDF_SUCCESS);
    //col_w.fill(1.0)
    gdf_graph G;
    ASSERT_EQ(gdf_edge_list_view(&G, &col_src, &col_dest, nullptr),GDF_SUCCESS);

    std::vector<T> sssp_distances(v,0.0);
    gdf_column col_sssp_distances;
    create_gdf_column(sssp_distances,&col_sssp_distances);
    
    if (manual_tanspose)
          ASSERT_EQ(gdf_add_transposed_adj_list(&G),0);

    ASSERT_EQ(gdf_sssp_nvgraph(&G, &param.src, &col_sssp_distances),GDF_SUCCESS);

    cudaMemcpy((void*)&sssp_distances[0], (void*)col_sssp_distances.data, sizeof(T)*v, cudaMemcpyDeviceToHost);
    //for (auto i = sssp_distances.begin(); i != sssp_distances.end(); ++i) std::cout << *i << ' ';
  }
};
 
std::vector<double> Tests_Sssp::sssp_time;

TEST_P(Tests_Sssp, CheckFP32) {
    run_current_test<float, true>(GetParam());
}
TEST_P(Tests_Sssp, CheckFP32_auto) {
    run_current_test<float, false>(GetParam());
}

// --gtest_filter=*simple_test*

INSTANTIATE_TEST_CASE_P(simple_test, Tests_Sssp, 
                        ::testing::Values(  Sssp_Usecase("grmat --rmat_scale=10 --rmat_edgefactor=16 --device=0  --normalized --quiet", 0)
                                            ,Sssp_Usecase("grmat --rmat_scale=12 --rmat_edgefactor=8 --device=0  --normalized --quiet", 10)
                                            ,Sssp_Usecase("grmat --rmat_scale=20 --rmat_edgefactor=32 --device=0  --normalized --quiet", 0)
                                         )
                       );


typedef struct Sssp2_Usecase_t {
  std::string matrix_file;
  std::string result_file;
  int src;
  //Sssp2_Usecase_t(const std::string& a, const std::string& b, int c ) : matrix_file(a), result_file(b), src(c){};
  Sssp2_Usecase_t(const std::string& a, const std::string& b, int c ) {
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
    src = c;
  }

  Sssp2_Usecase_t& operator=(const Sssp2_Usecase_t& rhs) {
    matrix_file = rhs.matrix_file;
    result_file = rhs.result_file;
    src = rhs.src;
    return *this;
  }
} Sssp2_Usecase;

class Tests_Sssp2 : public ::testing::TestWithParam<Sssp2_Usecase> {
  public:
  Tests_Sssp2() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { 
    if (PERF) {
     for (unsigned int i = 0; i < sssp_time.size(); ++i) {
      std::cout <<  sssp_time[i]/PERF_MULTIPLIER << std::endl;
     }
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> sssp_time;   


  template <typename T, bool manual_tanspose>
  void run_current_test(const Sssp2_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     std::stringstream ss; 
     std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.matrix_file)+ std::string("_") + ss.str().c_str();

     int m, k, nnz;
     MM_typecode mc;
     
     gdf_graph G;
     gdf_column col_src, col_dest, col_sssp;
     gdf_error status;
     bool has_guess = false;

     FILE* fpin = fopen(param.matrix_file.c_str(),"r");
     ASSERT_NE(fpin, nullptr) << "fopen (" << param.matrix_file << ") failure.";
     
     ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
     ASSERT_TRUE(mm_is_matrix(mc));
     ASSERT_TRUE(mm_is_coordinate(mc));
     ASSERT_FALSE(mm_is_complex(mc));
     ASSERT_FALSE(mm_is_skew(mc));
     
     // Allocate memory on host
     std::vector<int> cooRowInd(nnz), cooColInd(nnz);
     std::vector<T> cooVal(nnz), sssp(m,0.0);

     // Read
     ASSERT_EQ( (mm_to_coo<int,T>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
     ASSERT_EQ(fclose(fpin),0);

     //for (auto i = cooRowInd.begin(); i != cooRowInd.begin()+10; ++i) std::cout << *i << ' ';

     //for (auto i = cooColInd.begin(); i != cooColInd.begin()+10; ++i) std::cout << *i << ' ';


     //std::cout<< *std::min_element(cooRowInd.begin(), cooRowInd.end()) <<std::endl;
     //std::cout<< *std::max_element(cooRowInd.begin(), cooRowInd.end()) <<std::endl <<std::endl;
     //std::cout<< *std::min_element(cooColInd.begin(), cooColInd.end()) <<std::endl;
     //std::cout<< *std::max_element(cooColInd.begin(), cooColInd.end()) <<std::endl <<std::endl; 
     //std::cout<< cooColInd.size() <<std::endl;
    
    // gdf columns
    create_gdf_column(cooRowInd, &col_src);
    create_gdf_column(cooColInd, &col_dest);
    create_gdf_column(sssp     , &col_sssp);

    ASSERT_EQ(gdf_edge_list_view(&G, &col_src, &col_dest, nullptr),0);
    
    if (manual_tanspose)
      ASSERT_EQ(gdf_add_transposed_adj_list(&G),0);


    ASSERT_EQ(gdf_sssp_nvgraph(&G, &param.src, &col_sssp),0);

    // Check vs golden data
    if (param.result_file.length()>0)
    {
      std::vector<T> calculated_res(m);
      cudaMemcpy(&calculated_res[0],   col_sssp.data,   sizeof(T) * m, cudaMemcpyDeviceToHost);
      fpin = fopen(param.result_file.c_str(),"rb");
      ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file << std::endl;
      std::vector<T> expected_res(m);
      ASSERT_EQ(read_binary_vector(fpin, m, expected_res), 0);
      fclose(fpin);
    }
  }
};
 
std::vector<double> Tests_Sssp2::sssp_time;

TEST_P(Tests_Sssp2, CheckFP32_manualT) {
    run_current_test<float, true>(GetParam());
}

TEST_P(Tests_Sssp2, CheckFP32) {
    run_current_test<float, false>(GetParam());
}

// --gtest_filter=*golden_test*
INSTANTIATE_TEST_CASE_P(golden_test, Tests_Sssp2, 
                        ::testing::Values(  Sssp2_Usecase("test/datasets/karate.mtx" , "", 1)
                                           ,Sssp2_Usecase("test/datasets/dblp.mtx" ,     "test/ref/sssp/dblp_T.sssp_100000.bin", 100000)
                                           ,Sssp2_Usecase("test/datasets/dblp.mtx" ,     "test/ref/sssp/dblp_T.sssp_100.bin", 100)
                                           ,Sssp2_Usecase("test/datasets/wiki2003.mtx" , "test/ref/sssp/wiki2003_T.sssp_100000.bin",100000 )
                                           ,Sssp2_Usecase("test/datasets/wiki2003.mtx" , "test/ref/sssp/wiki2003_T.sssp_100.bin", 100)
                                         )
                       );
int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
    //for (int i = 0; i < argc; i++) {
    //    if (strcmp(argv[i], "--perf") == 0)
    //        PERF = 1;
    //    if (strcmp(argv[i], "--perf-iters") == 0)
    //        PERF_MULTIPLIER = atoi(argv[i+1]);
    //}
  return RUN_ALL_TESTS();
}


