// This is gtest application that contains all of the C API tests. Parameters:
// nvgraph_capi_tests [--perf] [--stress-iters N] [--gtest_filter=NameFilterPatter]
// It also accepts any other gtest (1.7.0) default parameters.
// Right now this application contains:
// 1) Sanity Check tests - tests on simple examples with known answer (or known behaviour)
// 2) Correctness checks tests - tests on real graph data, uses reference algorithm 
//    (CPU code for SrSPMV and python scripts for other algorithms, see 
//    python scripts here: //sw/gpgpu/nvgraph/test/ref/) with reference results, compares those two.
//    It also measures performance of single algorithm C API call, enf enabled (see below)
// 3) Corner cases tests - tests with some bad inputs, bad parameters, expects library to handle 
//    it gracefully
// 4) Stress tests - makes sure that library result is persistent throughout the library usage
//    (a lot of C API calls). Also makes some assumptions and checks on memory usage during 
//    this test.
//
// We can control what tests to launch by using gtest filters. For example:
// Only sanity tests:
//    ./nvgraph_capi_tests --gtest_filter=*Sanity*
// And, correspondingly:
//    ./nvgraph_capi_tests --gtest_filter=*Correctness*
//    ./nvgraph_capi_tests --gtest_filter=*Corner*
//    ./nvgraph_capi_tests --gtest_filter=*Stress*
// Or, combination:
//    ./nvgraph_capi_tests --gtest_filter=*Sanity*:*Correctness*
//
// Performance reports are provided in the ERIS format and disabled by default. 
// Could be enabled by adding '--perf' to the command line. I added this parameter to vlct
//
// Parameter '--stress-iters N', which gives multiplier (not an absolute value) for the number of launches for stress tests
//

#include <utility>

#include "gtest/gtest.h"

#include "nvgraph_test_common.h"

#include "valued_csr_graph.hxx"
#include "readMatrix.hxx"
#include "nvgraphP.h"
#include "nvgraph.h"
#include <nvgraph_experimental.h>  // experimental header, contains hidden API entries, can be shared only under special circumstances without reveling internal things

#include "stdlib.h"
#include "stdint.h"
#include <algorithm>

// do the perf measurements, enabled by command line parameter '--perf'
static int PERF = 0;

// minimum vertices in the graph to perform perf measurements
#define PERF_ROWS_LIMIT 10000
static int complex_repeats = 20;
static std::string ref_data_prefix = "";
static std::string graph_data_prefix = "";

template <typename T>
struct comparison
{
    bool operator() (T* lhs, T* rhs) {return (*lhs) < (*rhs);}
};


template <typename T>
bool enough_device_memory(int n, int nnz, size_t add)
{
    size_t mtotal, mfree;
    cudaMemGetInfo(&mfree, &mtotal);
    if (mfree > add + sizeof(T)*3*(n + nnz)) 
        return true;
    return false;
}

std::string convert_to_local_path(const std::string& in_file)
{
    std::string wstr = in_file;
    if ((wstr != "dummy") & (wstr != ""))
    {
        std::string prefix;
        if (graph_data_prefix.length() > 0)
        {
            prefix = graph_data_prefix;
        }
        else 
        {
#ifdef _WIN32
            //prefix = "C:\\mnt\\eris\\test\\matrices_collection\\";
            prefix = "\\\\cuda-vnetapp\\eris_matrices\\";
            std::replace(wstr.begin(), wstr.end(), '/', '\\');
#else
            prefix = "/mnt/nvgraph_test_data/";
#endif
        }
        wstr = prefix + wstr;
    }
    return wstr;
}

class NVGraphCAPITests_Triangles_Sanity : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_Triangles_Sanity() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        nvgraphStatus_t status;
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
    

    void prepare_and_run(const void* topo_st, bool lower_triangular, uint64_t expected )
    {
        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        status = nvgraphSetGraphStructure(handle, g1, (void*)topo_st, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        uint64_t res = 0;

        status = nvgraphTriangleCount(handle, g1, &res);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        //printf("Expected triangles: %" PRIu64 ", got triangles: %" PRIu64 "\n", expected, res);

        // get result
        ASSERT_EQ(expected, res);

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }

    void run_star_test_csr()
    {
        int N = 1024; // min is 5
        int n = N - 1; 
        int nnz = 2 * (N - 1) ;
        std::vector<int> offsets(N+1), neighborhood(nnz);
        offsets[0] = 0; offsets[1] = 0;
        int cur_nnz = 0;
        for (int i = 1; i < N; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (j == 0 || j == i - 1 || (j == 1 && i == (N-1)))
                {
                    neighborhood[cur_nnz] = j;
                    cur_nnz++;
                }
            }
            offsets[i+1] = cur_nnz;
        }
        //offsets[n] = cur_nnz;
        /*printf("N = %d, n = %d, nnz = %d\n", N, n, nnz);
        for (int i = 0; i < N+1; i++)
            printf("RO [%d] == %d\n", i, offsets[i]);

        for (int i = 0; i < nnz; i++)
            printf("CI [%d] == %d\n", i, neighborhood[i]);*/

        topo = NVGRAPH_CSR_32;

        nvgraphCSRTopology32I_st topology = {N, nnz, &offsets[0], &neighborhood[0]};
        
        prepare_and_run((void*)&topology, true, n);
    }

    void run_seq_test_csr()
    {
        int N = 1024; // min is 3
        int n = N - 2; // actual number of triangles
        int nnz = 2 * (N - 3) + 3;
        std::vector<int> offsets(N+1), neighborhood(nnz);
        offsets[0] = 0;
        int cur_nnz = 0;
        for (int i = 0; i < N; i++)
        {
            if (i > 1)
            {
                neighborhood[cur_nnz] = i - 2;
                cur_nnz++;
            }
            if (i > 0)
            {
                neighborhood[cur_nnz] = i - 1;
                cur_nnz++;
            }
            offsets[i+1] = cur_nnz;
        }
        //offsets[n] = cur_nnz;
        /*printf("N = %d, n = %d, nnz = %d\n", N, n, nnz);
        for (int i = 0; i < N+1; i++)
            printf("RO [%d] == %d\n", i, offsets[i]);

        for (int i = 0; i < nnz; i++)
            printf("CI [%d] == %d\n", i, neighborhood[i]);*/

        topo = NVGRAPH_CSR_32;

        nvgraphCSRTopology32I_st topology = {N, nnz, &offsets[0], &neighborhood[0]};
        
        prepare_and_run((void*)&topology, true, n);
    }
};

typedef struct TriCount_Usecase_t
{
    std::string graph_file;
    uint64_t ref_tricount;
    TriCount_Usecase_t(const std::string& a, uint64_t b) : ref_tricount(b) { graph_file = convert_to_local_path(a); };
    TriCount_Usecase_t& operator=(const TriCount_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        ref_tricount = rhs.ref_tricount;
        return *this;
    } 
} TriCount_Usecase_t;

class TriCountRefGraphCheck : public ::testing::TestWithParam<TriCount_Usecase_t> {
  public:
    TriCountRefGraphCheck() : handle(NULL) {}

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

    void run_current_test(const TriCount_Usecase_t& param)
    {
        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file);

        nvgraphTopologyType_t topo = NVGRAPH_CSR_32;

        nvgraphStatus_t status;

        FILE* fpin = fopen(param.graph_file.c_str(),"rb");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        int n, nnz;
        //Read a transposed network in amgx binary format and the bookmark of dangling nodes
        std::vector<int> read_row_ptr, read_col_ind;
        ASSERT_EQ(read_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind), 0);
        fclose(fpin);

        if (!enough_device_memory<char>(n, nnz, sizeof(int)*(read_row_ptr.size() + read_col_ind.size())))
        {
            std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            return;
        }

        nvgraphGraphDescr_t g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        nvgraphCSRTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo);

        uint64_t res = 0;

        status = nvgraphTriangleCount(handle, g1, &res);
        cudaDeviceSynchronize();
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // run
        if (PERF && n > PERF_ROWS_LIMIT)
        {
            double start, stop;
            start = second();
            start = second();
            int repeat = complex_repeats;
            for (int i = 0; i < repeat; i++)
            {
                status = nvgraphTriangleCount(handle, g1, &res);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            cudaDeviceSynchronize();
            stop = second();
            printf("&&&& PERF Time_%s %10.8f -ms\n", test_id.c_str(), 1000.0*(stop-start)/repeat);
        }

        //printf("Expected triangles: %" PRIu64 ", got triangles: %" PRIu64 "\n", expected, res);
        ASSERT_EQ(param.ref_tricount, res);

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
};
 
TEST_P(TriCountRefGraphCheck, CorrectnessCheck)
{
    run_current_test(GetParam());   
}


INSTANTIATE_TEST_CASE_P(NVGraphCAPITests_TriCount,
                        TriCountRefGraphCheck,
                        //                                          graph FILE                              reference number of triangles
//                                            // we read matrix stored in CSR and pass it as CSC - so matrix is in fact transposed, that's why we compare it to the results calculated on a transposed matrix
                        ::testing::Values(
                                                TriCount_Usecase_t("graphs/triangles_counting/as-skitter_internet_topo.csr.bin"       , 28769868)
                                              , TriCount_Usecase_t("graphs/triangles_counting/cage15_N_5154859.csr.bin"               , 36106416             )
                                              , TriCount_Usecase_t("graphs/triangles_counting/cit-Patents_N_3774768.csr.bin"          , 7515023)
                                              , TriCount_Usecase_t("graphs/triangles_counting/coAuthorsCiteseer_N_227320.csr.bin"     , 2713298)
                                              , TriCount_Usecase_t("graphs/triangles_counting/com-orkut_N_3072441.csr.bin"            , 627584181)
                                              , TriCount_Usecase_t("graphs/triangles_counting/coPapersCiteseer.csr.bin"               , 872040567)
                                              , TriCount_Usecase_t("graphs/triangles_counting/coPapersDBLP_N_540486.csr.bin"          , 444095058)
                                              , TriCount_Usecase_t("graphs/triangles_counting/europe_osm_N_50912018.csr.bin"          , 61710)
                                              , TriCount_Usecase_t("graphs/triangles_counting/hollywood-2009_N_1139905.csr.bin"       , 4916374555)
                                              , TriCount_Usecase_t("graphs/triangles_counting/kron_g500-simple-logn16.csr.bin"        , 118811321)
                                              , TriCount_Usecase_t("graphs/triangles_counting/kron_g500-simple-logn18.csr.bin"        , 687677667)
                                              , TriCount_Usecase_t("graphs/triangles_counting/kron_g500-simple-logn21.csr.bin"        , 8815649682)
                                              , TriCount_Usecase_t("graphs/triangles_counting/mouse_gene_N_45101.csr.bin"             , 3619097862)
                                              , TriCount_Usecase_t("graphs/triangles_counting/road_central_N_14081816.csr.bin"        , 228918)
                                              , TriCount_Usecase_t("graphs/triangles_counting/soc-LiveJournal1_N_4847571.csr.bin"     , 285730264)
                                              , TriCount_Usecase_t("graphs/triangles_counting/wb-edu_N_9845725.csr.bin"               , 254718147)
                                            ///// more instances
                                         )
                        );
 
TEST_F(NVGraphCAPITests_Triangles_Sanity, SanityStarCSR)
{
    run_star_test_csr();
}

TEST_F(NVGraphCAPITests_Triangles_Sanity, SanitySeqCSR)
{
    run_seq_test_csr();
}

int main(int argc, char **argv) 
{

    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--perf") == 0)
            PERF = 1;
        if (strcmp(argv[i], "--ref-data-dir") == 0)
            ref_data_prefix = std::string(argv[i+1]);
        if (strcmp(argv[i], "--graph-data-dir") == 0)
            graph_data_prefix = std::string(argv[i+1]);
    }
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
        
  return RUN_ALL_TESTS();
}
