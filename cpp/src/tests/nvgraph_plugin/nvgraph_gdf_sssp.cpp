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
          ASSERT_EQ(gdf_add_transpose(&G),0);

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
  Sssp2_Usecase_t(const std::string& a, const std::string& b, int c ) : matrix_file(a), result_file(b), src(c){};
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
      ASSERT_EQ(gdf_add_transpose(&G),0);


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
                        ::testing::Values(  Sssp2_Usecase("/datasets/networks/karate.mtx" , "", 1)
                                           ,Sssp2_Usecase("/datasets/golden_data/graphs/dblp.mtx" , "/datasets/golden_data/results/sssp/dblp_T.sssp_100000.bin", 100000)
                                           ,Sssp2_Usecase("/datasets/golden_data/graphs/dblp.mtx" , "/datasets/golden_data/results/sssp/dblp_T.sssp_100.bin", 100)
                                           ,Sssp2_Usecase("/datasets/golden_data/graphs/wiki2003.mtx" , "/datasets/golden_data/results/sssp/wiki2003_T.sssp_100000.bin",100000 )
                                           ,Sssp2_Usecase("/datasets/golden_data/graphs/wiki2003.mtx" , "/datasets/golden_data/results/sssp/wiki2003_T.sssp_100.bin", 100)
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

/*
typedef struct SSSP_Usecase_t
{
    std::string graph_file;
    int source_vert;
    std::string result_file;
    double tolerance_mul;
    SSSP_Usecase_t(const std::string& a, int b, const std::string& c, double tolerance_multiplier = 1.0) : source_vert(b), tolerance_mul(tolerance_multiplier) { graph_file = convert_to_local_path(a); result_file = convert_to_local_path_refdata(c);};
    SSSP_Usecase_t& operator=(const SSSP_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        source_vert = rhs.source_vert; 
        result_file = rhs.result_file;
        return *this;
    } 
} SSSP_Usecase;



class NVGraphCAPITests_SSSP : public ::testing::TestWithParam<SSSP_Usecase> {
  public:
    NVGraphCAPITests_SSSP() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        if (handle == NULL) {
            status = nvgraphCreate(&handle);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
    }
    virtual void TearDown() {
        if (handle != NULL) {
            status = nvgraphDestroy(handle);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            handle = NULL;
        }
    }
    nvgraphStatus_t status;
    nvgraphHandle_t handle;

    template <typename T>
    void run_current_test(const SSSP_Usecase& param)
    {
        double test_start, test_end, read_start, read_end;
        test_start = second();
        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss; 
        ss << param.source_vert;
        std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file) + std::string("_") + ss.str().c_str();

        nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

        nvgraphStatus_t status;

        read_start = second();
        FILE* fpin = fopen(param.graph_file.c_str(),"rb");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        int n, nnz;
        //Read a transposed network in amgx binary format and the bookmark of dangling nodes
        ASSERT_EQ(read_header_amgx_csr_bin (fpin, n, nnz), 0);
        std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
        std::vector<T> read_val(nnz);
        ASSERT_EQ(read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, read_val), 0);
        fclose(fpin);
        read_end = second();

        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(read_row_ptr.size() + read_col_ind.size())) || 
            (PERF && n < PERF_ROWS_LIMIT))
        {
            std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            return;
        }

        nvgraphGraphDescr_t g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        nvgraphCSCTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo);

        // set up graph data
        size_t numsets = 1;
        std::vector<T> calculated_res(n);
        //void*  vertexptr[1] = {(void*)&calculated_res[0]};
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, numsets, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        //status = nvgraphSetVertexData(handle, descrG, vertexptr[0], 0 );
        //ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, numsets, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        int weight_index = 0;
        int source_vert = param.source_vert;
        int sssp_index = 0;

        // run
        status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
        cudaDeviceSynchronize();
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        if (PERF)
        {
            double start, stop;
            start = second();
            start = second();
            int repeat = simple_repeats;
            for (int i = 0; i < repeat; i++)
            {
                status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            cudaDeviceSynchronize();
            stop = second();
            printf("&&&& PERF Time_%s %10.8f -ms\n", test_id.c_str(), 1000.0*(stop-start)/repeat);
        }

        // get result
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // check with reference
        if (param.result_file.length() > 0)
        {
            fpin = fopen(param.result_file.c_str(),"rb");
            ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file << std::endl;
            std::vector<T> expected_res(n);
            ASSERT_EQ(read_binary_vector(fpin, n, expected_res), 0);
            fclose(fpin);
            for (int i = 0; i < n; i++)
            {
                ASSERT_NEAR(expected_res[i], calculated_res[i], nvgraph_Const<T>::tol) << "In row #" << i << " graph " << param.graph_file << " source_vert=" << source_vert<< "\n" ;
            }
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        test_end = second();
        if (print_test_timings) printf("Test took: %f seconds from which %f seconds were spent on data reading\n", test_end - test_start, read_end - read_start);
    }
};
 
TEST_P(NVGraphCAPITests_SSSP, CheckResultDouble)
{
    run_current_test<double>(GetParam());   
}

TEST_P(NVGraphCAPITests_SSSP, CheckResultFloat)
{
    run_current_test<float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(CorrectnessCheck,
                        NVGraphCAPITests_SSSP,
                        //                                  graph FILE                                                  source vert #    file with expected result (in binary?)
//                                            // we read matrix stored in CSR and pass it as CSC - so matrix is in fact transposed, that's why we compare it to the results calculated on a transposed matrix
                        ::testing::Values(      SSSP_Usecase("graphs/dblp/dblp.mtx", 100,    "graphs/dblp/dblp_T.sssp_100.bin")
                                              , SSSP_Usecase("graphs/dblp/dblp.mtx", 100000, "graphs/dblp/dblp_T.sssp_100000.bin")
                                              , SSSP_Usecase("graphs/Wikipedia/2003/wiki2003.mtx", 100,    "graphs/Wikipedia/2003/wiki2003_T.sssp_100.bin")
                                              , SSSP_Usecase("graphs/Wikipedia/2003/wiki2003.mtx", 100000, "graphs/Wikipedia/2003/wiki2003_T.sssp_100000.bin")
                                              , SSSP_Usecase("graphs/citPatents/cit-Patents_T.mtx.mtx", 6543, "")
                                              //, SSSP_Usecase("dimacs10/kron_g500-logn20_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/hugetrace-00020_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/delaunay_n24_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/road_usa_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/hugebubbles-00020_T.mtx.bin", 100000, "")
                                            ///// more instances
                                         )
                        );

int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



*/
