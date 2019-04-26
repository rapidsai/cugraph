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
#include <algorithm>

// do the perf measurements, enabled by command line parameter '--perf'
static int PERF = 0;

// minimum vertices in the graph to perform perf measurements
#define PERF_ROWS_LIMIT 10000

// number of repeats = multiplier/num_vertices
#define SRSPMV_ITER_MULTIPLIER   1000000000
#define SSSP_ITER_MULTIPLIER     30000000
#define WIDEST_ITER_MULTIPLIER   30000000
#define PAGERANK_ITER_MULTIPLIER 300000000

static std::string ref_data_prefix = "";
static std::string graph_data_prefix = "";

// iterations for stress tests = this multiplier * iterations for perf tests
static int STRESS_MULTIPLIER = 1;
static int simple_repeats = 50;
static int complex_repeats = 20;
static int print_test_timings = 1;

// utility

template <typename T>
struct nvgraph_Const;

template <>
struct nvgraph_Const<double>
{ 
    static const cudaDataType_t Type = CUDA_R_64F;
    static const double inf;
    static const double tol;
    typedef union fpint 
    {
        double f;
        unsigned long u;
    } fpint_st;
};

const double nvgraph_Const<double>::inf = DBL_MAX;
const double nvgraph_Const<double>::tol = 1e-6; // this is what we use as a tolerance in the algorithms, more precision than this is useless for CPU reference comparison

template <>
struct nvgraph_Const<float>
{ 
    static const cudaDataType_t Type = CUDA_R_32F;
    static const float inf;
    static const float tol;

    typedef union fpint 
    {
        float f;
        unsigned u;
    } fpint_st;

};

const float nvgraph_Const<float>::inf = FLT_MAX;
const float nvgraph_Const<float>::tol = 1e-4;

template <typename T>
struct comparison
{
    bool operator() (T* lhs, T* rhs) {return (*lhs) < (*rhs);}
};

struct SR_OP
{

    const char*
    get_name(nvgraphSemiring_t sr)
    { 
        const char* ret = "Unknown_SR";
        switch (sr)
        {
            case NVGRAPH_PLUS_TIMES_SR:
                ret = "PLUS_TIMES_SR";
                break;
            case NVGRAPH_MIN_PLUS_SR:
                ret = "MIN_PLUS_SR";
                break;
            case NVGRAPH_MAX_MIN_SR:
                ret = "MAX_MIN_SR";
                break;
            case NVGRAPH_OR_AND_SR:
                ret = "OR_AND_SR";
                break;
        }
        return ret;
    };


    template <typename T>
    T plus(const T& a, const T& b, nvgraphSemiring_t sr)
    { 
        T ret = (T)0;
        switch (sr)
        {
            case NVGRAPH_PLUS_TIMES_SR:
                ret = a + b;
                break;
            case NVGRAPH_MIN_PLUS_SR:
                ret = std::min(a, b);
                break;
            case NVGRAPH_MAX_MIN_SR:
                ret = std::max(a, b);
                break;
            case NVGRAPH_OR_AND_SR:
                ret = (T)((bool)(a) | (bool)(b));
                break;
            default:
                printf("Semiring %d is not supported, check line %d\n", (int)sr, __LINE__);
                //FAIL() << "Semiring #" << (int)sr << " is not supported.";
        }
        return ret;
    };

    template <typename T>
    T mul(const T& a, const T& b, nvgraphSemiring_t sr)
    { 
        T ret = (T)0;
        switch (sr)
        {
            case NVGRAPH_PLUS_TIMES_SR:
                ret = a * b;
                break;
            case NVGRAPH_MIN_PLUS_SR:
                ret = a + b;
                break;
            case NVGRAPH_MAX_MIN_SR:
                ret = std::min(a, b);;
                break;
            case NVGRAPH_OR_AND_SR:
                ret = (T)((bool)(a) & (bool)(b));
                break;
            default:
                printf("Semiring %d is not supported, check line %d\n", (int)sr, __LINE__);
                //FAIL() << "Semiring #" << (int)sr << " is not supported.";
        }
        return ret;
    };

    template <typename T>
    T get_ini(const nvgraphSemiring_t& sr) 
    {
        T ret = (T)0;
        switch (sr)
        {
            case NVGRAPH_PLUS_TIMES_SR:
                ret = (T)0;
                break;
            case NVGRAPH_MIN_PLUS_SR:
                ret = nvgraph_Const<T>::inf;
                break;
            case NVGRAPH_MAX_MIN_SR:
                ret = -(nvgraph_Const<T>::inf);
                break;
            case NVGRAPH_OR_AND_SR:
                ret = (T)0;
                break;
            default:
                printf("Semiring %d is not supported, check line %d\n", (int)sr, __LINE__);
                //FAIL() << "Semiring #" << (int)sr << " is not supported.";
        }
        return ret;
    };

} SR_OPS;


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
            prefix = "Z:\\matrices_collection\\";
            std::replace(wstr.begin(), wstr.end(), '/', '\\');
#else
            prefix = "/mnt/nvgraph_test_data/";
#endif
        }
        wstr = prefix + wstr;
    }
    return wstr;
}

std::string convert_to_local_path_refdata(const std::string& in_file)
{
    std::string wstr = in_file;
    if ((wstr != "dummy") & (wstr != ""))
    {
        std::string prefix;
        if (ref_data_prefix.length() > 0)
        {
            prefix = ref_data_prefix;
        }
        else
        {
#ifdef _WIN32
            //prefix = "C:\\mnt\\eris\\test\\ref_data\\";
            prefix = "Z:\\ref_data\\";
            std::replace(wstr.begin(), wstr.end(), '/', '\\');
#else
            prefix = "/mnt/nvgraph_test_data/ref_data/";
#endif
        }
        wstr = prefix + wstr;
    }
    return wstr;
}

// SrSPMV tests

typedef struct SrSPMV_Usecase_t
{
    std::string graph_file;
    nvgraphSemiring_t sr;
    double alpha;
    double beta;
    double tolerance_mul;
    SrSPMV_Usecase_t(const std::string& a, nvgraphSemiring_t b, const double c, const double d, double tolerance_multiplier = 1.0) : sr(b), alpha(c), beta(d), tolerance_mul(tolerance_multiplier) { graph_file = convert_to_local_path(a);};
    SrSPMV_Usecase_t& operator=(const SrSPMV_Usecase_t& rhs) 
    {
        graph_file = rhs.graph_file;
        sr = rhs.sr;
        alpha = rhs.alpha;
        beta = rhs.beta;
        return *this;
    };
} SrSPMV_Usecase;

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

typedef struct WidestPath_Usecase_t
{
    std::string graph_file;
    int source_vert;
    std::string result_file;
    double tolerance_mul;
    WidestPath_Usecase_t(const std::string& a, int b, const std::string& c, double tolerance_multiplier = 1.0) : source_vert(b), tolerance_mul(tolerance_multiplier) { graph_file = convert_to_local_path(a); result_file = convert_to_local_path_refdata(c);};
    WidestPath_Usecase_t& operator=(const WidestPath_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        source_vert = rhs.source_vert;
        result_file = rhs.result_file;
        return *this;
    }
} WidestPath_Usecase;

typedef struct Pagerank_Usecase_t
{
    std::string graph_file;
    float alpha;
    std::string result_file;
    double tolerance_mul;
    Pagerank_Usecase_t(const std::string& a, float b, const std::string& c, double tolerance_multiplier = 1.0) : alpha(b), tolerance_mul(tolerance_multiplier) { graph_file = convert_to_local_path(a); result_file = convert_to_local_path_refdata(c);};
    Pagerank_Usecase_t& operator=(const Pagerank_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        alpha = rhs.alpha; 
        result_file = rhs.result_file;
        return *this;  
    } 
} Pagerank_Usecase;


class NVGraphCAPITests_SrSPMV : public ::testing::TestWithParam<SrSPMV_Usecase> {
  public:
    NVGraphCAPITests_SrSPMV() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        //const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        //printf("We are in test %s of test case %s.\n", test_info->name(), test_info->test_case_name());
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
    void run_current_test(const SrSPMV_Usecase& param)
    {
        double test_start, test_end, read_start, read_end;
        test_start = second();
        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss;
        ss << "_alpha_" << (int)param.alpha << "_beta_" << (int)param.beta;
        std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file) + ss.str();

        nvgraphTopologyType_t topo = NVGRAPH_CSR_32;
        int weight_index = 0;
        int x_index = 0;
        int y_index = 1;
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
            (PERF && (n < PERF_ROWS_LIMIT || param.alpha + param.beta < 2)))
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
        //@TODO: random fill?
        std::vector<T> calculated_res(n);
        std::vector<T> data1(n), data2(n);
        for (int i = 0; i < n; i++)
        {
            data1[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
            data2[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
            //printf ("data1[%d]==%f, data2[%d]==%f\n", i, data1[i], i, data2[i]);
        }
        void*  vertexptr[2] = {(void*)&data1[0], (void*)&data2[0]};
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[0], x_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[1], y_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], weight_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        T alphaT = (T)param.alpha;
        T betaT = (T)param.beta;

        // run
        if (PERF)
        {
            double start, stop;
            // warmup
            status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alphaT, x_index, (void*)&betaT, y_index, param.sr);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            cudaDeviceSynchronize();

            int repeat = simple_repeats;
            start = second();
            start = second();
            // perf loop
            for (int i = 0; i < repeat; i++)
            {
                status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alphaT, x_index, (void*)&betaT, y_index, param.sr);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            cudaDeviceSynchronize();
            stop = second();
            printf("&&&& PERF Time_%s_%s %10.8f -ms\n", test_id.c_str(), SR_OPS.get_name(param.sr), 1000.0*(stop-start)/((double)repeat));
        }

        // reinit data
        status = nvgraphSetVertexData(handle, g1, (void*)&data2[0], y_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alphaT, x_index, (void*)&betaT, y_index, param.sr);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // get result
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], y_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // check correctness 
        std::vector<T> expected_res(n, SR_OPS.get_ini<T>(param.sr));
        for (int row = 0; row < n; row++)
        {
            for (int nz = read_row_ptr[row]; nz < read_row_ptr[row+1]; nz++)
            {
                expected_res[row] = SR_OPS.plus<T>(expected_res[row], SR_OPS.mul<T>(SR_OPS.mul<T>(param.alpha, read_val[nz], param.sr), data1[read_col_ind[nz]], param.sr), param.sr);
            }
            expected_res[row] = SR_OPS.plus<T>(expected_res[row], SR_OPS.mul<T>(data2[row], param.beta, param.sr), param.sr);
            double reference_res = (double)expected_res[row];
            double nvgraph_res = (double)calculated_res[row];
            ASSERT_NEAR(reference_res, nvgraph_res, nvgraph_Const<T>::tol) << "In row #" << row << " graph " << param.graph_file << " semiring " << SR_OPS.get_name(param.sr) << " alpha=" << param.alpha << " beta=" << param.beta << "\n";
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        test_end = second();
        if (print_test_timings) printf("Test took: %f seconds from which %f seconds were spent on data reading\n", test_end - test_start, read_end - read_start);
    }
};
 
TEST_P(NVGraphCAPITests_SrSPMV, CheckResultDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_SrSPMV, CheckResultFloat)
{
    run_current_test<float>(GetParam());
}


/// WidestPath tests

class NVGraphCAPITests_WidestPath : public ::testing::TestWithParam<WidestPath_Usecase> {
  public:
    NVGraphCAPITests_WidestPath() : handle(NULL) {}

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
    void run_current_test(const WidestPath_Usecase& param)
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
        cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, numsets, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        //status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0 );
        //ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, numsets, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int weight_index = 0;
        int source_vert = param.source_vert;
        int widest_path_index = 0;

        status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
        cudaDeviceSynchronize();
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        // run
        if (PERF)
        {
            double start, stop;
            start = second();
            start = second();
            int repeat = simple_repeats;
            for (int i = 0; i < repeat; i++)
            {
                status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            cudaDeviceSynchronize();
            stop = second();
            printf("&&&& PERF Time_%s %10.8f -ms\n", test_id.c_str(), 1000.0*(stop-start)/repeat);
        }

        // get result
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // check correctness 
        if (param.result_file.length()>0)
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
 
TEST_P(NVGraphCAPITests_WidestPath, CheckResultDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_WidestPath, CheckResultFloat)
{
    run_current_test<float>(GetParam());
}



//// SSSP tests

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

class NVGraphCAPITests_Pagerank : public ::testing::TestWithParam<Pagerank_Usecase> {
  public:
    NVGraphCAPITests_Pagerank() : handle(NULL) {}

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
    void run_current_test(const Pagerank_Usecase& param)
    {
        double test_start, test_end, read_start, read_end;
        test_start = second();
        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss; 
        ss << param.alpha;
        std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file) + std::string("_") + ss.str().c_str();

        if (param.graph_file == "dummy")
        {
            std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            return;
        }

        // Waive hugebubbles test, http://nvbugs/200189611
        /*{
            cudaDeviceProp prop;
            cudaGetDeviceProperties ( &prop, 0 );
            std::string gpu(prop.name);
            if (param.graph_file.find("hugebubbles-00020") != std::string::npos &&
                (gpu.find("M40") != npos ||
                 gpu.find("GTX 980 Ti") != npos ||
                 gpu.find("GTX TITAN X") != npos ||
                 gpu.find("M6000") != npos ||
                 gpu.find("GTX 680") != npos)
                )
            std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            return;   
        }*/

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
        std::vector<T> dangling(n);
        ASSERT_EQ(read_data_amgx_csr_bin_rhs (fpin, n, nnz, read_row_ptr, read_col_ind, read_val, dangling), 0);
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
        std::vector<T> calculated_res(n, (T)1.0/n);
        void*  vertexptr[2] = {(void*)&dangling[0], (void*)&calculated_res[0]};
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, 2, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[1], 1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        
        int bookmark_index = 0;
        int weight_index = 0;
        T alpha = param.alpha;
        int pagerank_index = 1;
        int has_guess = 0;
        float tolerance = (sizeof(T) > 4 ?  1e-8f :  1e-6f) * param.tolerance_mul;
        int max_iter = 1000;

        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        cudaDeviceSynchronize();
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // run
        if (PERF)
        {
            double start, stop;
            start = second();
            start = second();
            int repeat = complex_repeats;
            for (int i = 0; i < repeat; i++)
            {
                status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            cudaDeviceSynchronize();
            stop = second();
            printf("&&&& PERF Time_%s %10.8f -ms\n", test_id.c_str(), 1000.0*(stop-start)/repeat);
        }

        // get result
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], pagerank_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        std::sort(calculated_res.begin(), calculated_res.end());
    
        // check with reference
        if (param.result_file.length()>0)
        {
            fpin = fopen(param.result_file.c_str(),"rb");
            ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file << std::endl;
            std::vector<T> expected_res(n);
            ASSERT_EQ(read_binary_vector(fpin, n, expected_res), 0);
            fclose(fpin);
            T tot_err = 0.0, err;
            int n_err = 0;
            for (int i = 0; i < n; i++)
            {
                err = fabs(expected_res[i] - calculated_res[i]);
                if (err> nvgraph_Const<T>::tol)
                {
                    tot_err+=err;
                    n_err++;
                }
            }
            if (n_err)
            {
                EXPECT_NEAR(tot_err/n_err, nvgraph_Const<T>::tol, nvgraph_Const<T>::tol*9.99); // Network x used n*1e-10 for precision
                ASSERT_LE(n_err, 0.001*n); // we tolerate 0.1% of values with a litte difference
                //printf("number of incorrect entries: %d\n", n_err);
            }
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        test_end = second();
        if (print_test_timings) printf("Test took: %f seconds from which %f seconds were spent on data reading\n", test_end - test_start, read_end - read_start);
    }
};
 
TEST_P(NVGraphCAPITests_Pagerank, CheckResultDouble)
{
    run_current_test<double>(GetParam());   
}

TEST_P(NVGraphCAPITests_Pagerank, CheckResultFloat)
{
    run_current_test<float>(GetParam());
}

class NVGraphCAPITests_KrylovPagerank : public ::testing::TestWithParam<Pagerank_Usecase> {
  public:
    NVGraphCAPITests_KrylovPagerank() : handle(NULL) {}

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
    void run_current_test(const Pagerank_Usecase& param)
    {
        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss; 
        ss << param.alpha;
        std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file) + std::string("_") + ss.str().c_str();

        if (param.graph_file == "dummy")
        {
            std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            return;
        }

        nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

        nvgraphStatus_t status;

        FILE* fpin = fopen(param.graph_file.c_str(),"rb");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        int n, nnz;
        //Read a transposed network in amgx binary format and the bookmark of dangling nodes
        ASSERT_EQ(read_header_amgx_csr_bin (fpin, n, nnz), 0);
        std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
        std::vector<T> read_val(nnz);
        std::vector<T> dangling(n);
        ASSERT_EQ(read_data_amgx_csr_bin_rhs (fpin, n, nnz, read_row_ptr, read_col_ind, read_val, dangling), 0);
        fclose(fpin);

        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(read_row_ptr.size() + read_col_ind.size())))
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
        std::vector<T> calculated_res(n, (T)1.0/n);
        void*  vertexptr[2] = {(void*)&dangling[0], (void*)&calculated_res[0]};
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, 2, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[1], 1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        
        int bookmark_index = 0;
        int weight_index = 0;
        T alpha = param.alpha;
        int pagerank_index = 1;
        int has_guess = 0;
        float tolerance = (sizeof(T) > 4 ?  1e-8f :  1e-6f) * param.tolerance_mul;
        int max_iter = 150;
        int ss_sz = 7;


        // run
        if (PERF && n > PERF_ROWS_LIMIT)
        {
            double start, stop;
            start = second();
            start = second();
            int repeat = 10;
            for (int i = 0; i < repeat; i++)
                status = nvgraphKrylovPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, tolerance, max_iter, ss_sz, has_guess, pagerank_index);
            stop = second();
            printf("&&&& PERF Time_%s %10.8f -ms\n", test_id.c_str(), 1000.0*(stop-start)/repeat);
        }
        else
            status = nvgraphKrylovPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, tolerance, max_iter, ss_sz, has_guess, pagerank_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // get result
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], pagerank_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        std::sort(calculated_res.begin(), calculated_res.end());
    
        // check with reference
        if (param.result_file.length()>0)
        {
            fpin = fopen(param.result_file.c_str(),"rb");
            ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file << std::endl;
            std::vector<T> expected_res(n);
            ASSERT_EQ(read_binary_vector(fpin, n, expected_res), 0);
            fclose(fpin);
            T tot_err = 0.0, err;
            int n_err = 0;
            for (int i = 0; i < n; i++)
            {
                err = fabs(expected_res[i] - calculated_res[i]);
                if (err> nvgraph_Const<T>::tol)
                {
                    tot_err+=err;
                    n_err++;
                }
            }
            if (n_err)
            {
                EXPECT_NEAR(tot_err/n_err, nvgraph_Const<T>::tol, nvgraph_Const<T>::tol*9.99); // Network x used n*1e-10 for precision
                ASSERT_LE(n_err, 0.001*n); // we tolerate 0.1% of values with a litte difference
                //printf("number of incorrect entries: %d\n", n_err);
            }
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }

};
 
TEST_P(NVGraphCAPITests_KrylovPagerank, CheckResultDouble)
{
    run_current_test<double>(GetParam());   
}

TEST_P(NVGraphCAPITests_KrylovPagerank, CheckResultFloat)
{
    run_current_test<float>(GetParam());
}

/// Few sanity checks. 

class NVGraphCAPITests_SrSPMV_Sanity : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_SrSPMV_Sanity() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSR_32;
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
    
    template <typename T>
    void prepare_and_run(const nvgraphCSRTopology32I_st& topo_st, T* edgedata, T* data1, T* data2, T alpha, T beta, T* expected )
    {
        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        n = topo_st.nvertices;
        nnz = topo_st.nedges;
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topo_st, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
        status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        void*  vertexptr[2] = {(void*)data1, (void*)data2};
        void*  edgeptr[1] = {(void*)edgedata};
        int weight_index = 0;
        int x_index = 0;
        int y_index = 1;

        status = nvgraphSetVertexData(handle, g1, vertexptr[0], x_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[1], y_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, edgeptr[0], weight_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // get result
        std::vector<T> calculated_res(n);
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], y_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        for (int row = 0; row < n; row++)
        {
            double reference_res = (double)expected[row];
            double nvgraph_res = (double)calculated_res[row];
            ASSERT_NEAR(reference_res, nvgraph_res, nvgraph_Const<T>::tol) << "row=" << row << " alpha=" << alpha << " beta=" << beta << "\n";
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }

    // Trivial matrix with trivial answers, checks plus_times sr only (but that is good enough) and some set of alfa and beta
    template <typename T>
    void run_simple_test()
    {
        n = 1024;
        nnz = 1024;
        std::vector<int> offsets(n+1), neighborhood(nnz);
        std::vector<T> data1(n), data2(n);
        for (int i = 0; i < n; i++)
        {
            data1[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
            data2[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
            offsets[i] = neighborhood[i] = i;
        }
        offsets[n] = n;
        std::vector<T> edge_data(nnz, (T)(-2.0));
        std::vector<T> expected_res(n, SR_OPS.get_ini<T>(NVGRAPH_PLUS_TIMES_SR));

        nvgraphCSRTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};

        T pa[] = {-1.0, 0.0, 0.5, 1.0};
        T pb[] = {-1.0, 0.0, 0.5, 1.0};
        for (int ia = 0; ia < sizeof(pa)/sizeof(T); ia++)
            for (int ib = 0; ib < sizeof(pb)/sizeof(T); ib++)
            {
                for (int i = 0; i < n; i++)
                {
                    expected_res[i] = SR_OPS.get_ini<T>(NVGRAPH_PLUS_TIMES_SR);
                }
                for (int i = 0; i < n; i++)
                {
                    T tv1 = SR_OPS.mul<T>(data1[i], edge_data[i], NVGRAPH_PLUS_TIMES_SR);
                    tv1 = SR_OPS.mul<T>(tv1, pa[ia], NVGRAPH_PLUS_TIMES_SR);
                    T tv2 = SR_OPS.mul<T>(data2[i], pb[ib], NVGRAPH_PLUS_TIMES_SR);
                    tv2 = SR_OPS.plus<T>(tv1, tv2, NVGRAPH_PLUS_TIMES_SR);
                    expected_res[i] = SR_OPS.plus<T>(expected_res[i], tv2, NVGRAPH_PLUS_TIMES_SR);
                }
                prepare_and_run<T>(topology, &edge_data[0], &data1[0], &data2[0], pa[ia], pb[ib], &expected_res[0]);
            }
    }
};
 
TEST_F(NVGraphCAPITests_SrSPMV_Sanity, SanityDouble)
{
    run_simple_test<double>();
    
}

TEST_F(NVGraphCAPITests_SrSPMV_Sanity, SanityFloat)
{
    run_simple_test<float>();
}

class NVGraphCAPITests_SSSP_Sanity : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_SSSP_Sanity() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSC_32;
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
    
    template <typename T>
    void prepare_and_run(const nvgraphCSCTopology32I_st& topo_st, T* edgedata, T* expected )
    {
        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        n = topo_st.nvertices;
        nnz = topo_st.nedges;
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topo_st, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
        status = nvgraphAllocateVertexData(handle, g1, 1, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        void*  edgeptr[1] = {(void*)edgedata};
        status = nvgraphSetEdgeData(handle, g1, edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int source_vert = 0;
        int sssp_index = 0;
        int weight_index = 0;

        status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status) << ", n=" << n << std::endl;

        // get result
        std::vector<T> calculated_res(n);
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        for (int row = 0; row < n; row++)
        {
            double reference_res = (double)expected[row];
            double nvgraph_res = (double)calculated_res[row];
            ASSERT_NEAR(reference_res, nvgraph_res, nvgraph_Const<T>::tol) << "row=" << row << ", n=" << n << std::endl;
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }

// cycle graph, all weights = 1, shortest path = vertex number
    template <typename T>
    void run_cycle_test()
    {
        n = 1050;
        nnz = n;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[i] = i;
            neighborhood[i] = (n - 1 + i) % n;
        }
        offsets[n] = n;
        std::vector<T> edge_data(nnz, (T)1.0);
        std::vector<T> expected_res(n, nvgraph_Const<T>::inf);
        for (int i = 0; i < n; i++)
        {
            expected_res[i] = i;
        }

        // extensive run for small N's
        for (int i = 3; i < 200; i++)
        {
            neighborhood[0] = i - 1;
            nvgraphCSCTopology32I_st topology = {i, i, &offsets[0], &neighborhood[0]};
            prepare_and_run<T>(topology, &edge_data[0], &expected_res[0]);
        }
        // also trying larger N's 
        for (int i = 1020; i < 1030; i++)
        {
            neighborhood[0] = i - 1;
            nvgraphCSCTopology32I_st topology = {i, i, &offsets[0], &neighborhood[0]};
            prepare_and_run<T>(topology, &edge_data[0], &expected_res[0]);
        }
    }

// full binary tree, all weights = 1, shortest path length = level of the node
    template <typename T>
    void run_tree_test()
    {
        int k = 3;
        n = (1 << k) - 1;
        nnz = (1 << k) - 2;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[i+1] = i;
        }
        offsets[0] = 0;
        for (int i = 0; i < nnz; i++)
        {
            neighborhood[i] = i / 2;
        }
        std::vector<T> edge_data(nnz, (T)1.0);
        std::vector<T> expected_res(n, nvgraph_Const<T>::inf);
        expected_res[0] = 0;
        for (int i = 1; i < k; i++)
        {
            for (int v = 0; v < (1 << i); v++)
                expected_res[(1 << i) - 1 + v] = i;
        }

        nvgraphCSCTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
        prepare_and_run<T>(topology, &edge_data[0], &expected_res[0]);
    }
};
 
TEST_F(NVGraphCAPITests_SSSP_Sanity, SanityCycleDouble)
{
    run_cycle_test<double>();
}

TEST_F(NVGraphCAPITests_SSSP_Sanity, SanityCycleFloat)
{
    run_cycle_test<float>();
}

TEST_F(NVGraphCAPITests_SSSP_Sanity, SanityTreeDouble)
{
    run_tree_test<double>();
}

TEST_F(NVGraphCAPITests_SSSP_Sanity, SanityTreeFloat)
{
    run_tree_test<float>();
}


class NVGraphCAPITests_WidestPath_Sanity : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_WidestPath_Sanity() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSC_32;
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
    
    template <typename T>
    void prepare_and_run(const nvgraphCSCTopology32I_st& topo_st, T* edgedata, T* expected )
    {
        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        n = topo_st.nvertices;
        nnz = topo_st.nedges;
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topo_st, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
        status = nvgraphAllocateVertexData(handle, g1, 1, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        void*  edgeptr[1] = {(void*)edgedata};
        status = nvgraphSetEdgeData(handle, g1, edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int source_vert = 0;
        int widest_path_index = 0;
        int weight_index = 0;

        status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // get result
        std::vector<T> calculated_res(n);
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], widest_path_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        for (int row = 0; row < n; row++)
        {
            double reference_res = (double)expected[row];
            double nvgraph_res = (double)calculated_res[row];
            ASSERT_NEAR(reference_res, nvgraph_res, nvgraph_Const<T>::tol);
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }

// cycle graph, weigths are from n-1 to 0 starting with vertex = 0. widest path = [inf, n-1, n-2, ..., 1]
    template <typename T>
    void run_cycle_test()
    {
        n = 1024;
        nnz = n;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[i] = i;
            neighborhood[i] = (n - 1 + i) % n;
        }
        offsets[n] = n;
        std::vector<T> edge_data(nnz, 0);
        std::vector<T> expected_res(n, nvgraph_Const<T>::inf);
        for (int i = 1; i < n; i++)
        {
            edge_data[i] = (T)(n - i);
        }
        for (int i = 1; i < n; i++)
        {
            expected_res[i] = (T)(n - i);
        }

        nvgraphCSCTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
        prepare_and_run<T>(topology, &edge_data[0], &expected_res[0]);
    }

// cycle graph, edge weigths are equal to the (max_tree_lvl - edge_lvl). widest path to vertex is = (max_lvl - vertex_lvl)
    template <typename T>
    void run_tree_test()
    {
        int k = 10;
        n = (1 << k) - 1;
        nnz = (1 << k) - 2;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[i+1] = i;
        }
        offsets[0] = 0;
        for (int i = 0; i < nnz; i++)
        {
            neighborhood[i] = i / 2;
        }
        // fill edge data and expected res accordingly
        std::vector<T> edge_data(nnz);
        std::vector<T> expected_res(n, nvgraph_Const<T>::inf);
        for (int i = 1; i < k; i++)
        {
            for (int v = 0; v < (1 << i); v++)
            {
                edge_data[(1 << i) - 2 + v] = (k - i);
                expected_res[(1 << i) - 1 + v] = (k - i);
            }
        }

        nvgraphCSCTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
        prepare_and_run<T>(topology, &edge_data[0], &expected_res[0]);
    }
};
 
TEST_F(NVGraphCAPITests_WidestPath_Sanity, SanityCycleDouble)
{
    run_cycle_test<double>();
}

TEST_F(NVGraphCAPITests_WidestPath_Sanity, SanityCycleFloat)
{
    run_cycle_test<float>();
}

TEST_F(NVGraphCAPITests_WidestPath_Sanity, SanityTreeDouble)
{
    run_tree_test<double>();
}

TEST_F(NVGraphCAPITests_WidestPath_Sanity, SanityTreeFloat)
{
    run_tree_test<float>();
}


class NVGraphCAPITests_Pagerank_Sanity : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_Pagerank_Sanity() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSC_32;
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
    
    template <typename T>
    void prepare_and_run(const nvgraphCSCTopology32I_st& topo_st, T* bookmark, T* edge_data )
    {
        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        n = topo_st.nvertices;
        nnz = topo_st.nedges;
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topo_st, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
        status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int bookmark_index = 0;
        int weight_index = 0;
        T alpha = 0.85;
        int pagerank_index = 1;
        int has_guess = 0;
        float tolerance = 1e-6;//sizeof(T) > 4 ?  1e-8f :  1e-6f;
        int max_iter = 1000;

        status = nvgraphSetVertexData(handle, g1, (void*)bookmark, bookmark_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        void*  edgeptr[1] = {(void*)edge_data};
        status = nvgraphSetEdgeData(handle, g1, edgeptr[0], weight_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // run
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // get result
        std::vector<T> calculated_res(n);
        status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res[0], pagerank_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        for (int row = 1; row < n; row++)
        {
            //printf("PR[%d] == %10.7g, PR[%d] == %10.7g\n", row-1, calculated_res[row-1], row, calculated_res[row]);
            double res1 = (double)calculated_res[row-1];
            double res2 = (double)calculated_res[row];
            ASSERT_LE(res1, res2) << "In row: " << row << "\n";
        }

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }

// path graph, weigths are = 1, last node is dangling, pagerank should be in ascending order
    template <typename T>
    void run_path_test()
    {
        n = 1024;
        nnz = n - 1;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[1+i] = i;
            neighborhood[i] = i;
        }
        offsets[0] = 0;
        std::vector<T> edge_data(nnz, 1);
        std::vector<T> dangling(n, 0);
        dangling[n-1] = (T)(1);

        nvgraphCSCTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
        prepare_and_run<T>(topology, &dangling[0], &edge_data[0]);
    }
};
 
TEST_F(NVGraphCAPITests_Pagerank_Sanity, SanityPathDouble)
{
    run_path_test<double>();
}

TEST_F(NVGraphCAPITests_Pagerank_Sanity, SanitypathFloat)
{
    run_path_test<float>();
}



/// Corner cases for the C API

class NVGraphCAPITests_SrSPMV_CornerCases : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_SrSPMV_CornerCases() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSR_32;
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
    
    // Trivial matrix with trivial answers, checks plus_times sr only (but that is good enough) and sets of alfa and beta from {0.0, 1.0}
    template <typename T>
    void run_simple_test()
    {
        n = 1024;
        nnz = 1024;
        std::vector<int> offsets(n+1), neighborhood(nnz);
        std::vector<T> data1(n), data2(n);
        for (int i = 0; i < n; i++)
        {
            data1[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
            data2[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
            offsets[i] = neighborhood[i] = i;
        }
        offsets[n] = n;
        std::vector<T> edge_data(nnz, (T)1.0);

        nvgraphCSRTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
        T alpha = (T)(1.0);
        T beta = (T)(1.0);
        int weight_index = 0;
        int x_index = 0;
        int y_index = 1;

        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        n = topology.nvertices;
        nnz = topology.nedges;
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        // not multivalued CSR
        status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

        status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        void*  vertexptr[2] = {(void*)&data1[0], (void*)&data2[0]};
        void*  edgeptr[1] = {(void*)(&edge_data[0])};
        status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[1], 1 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // different bad values
        status = nvgraphSrSpmv(NULL, g1, weight_index, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSrSpmv(handle, NULL, weight_index, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSrSpmv(handle, g1, 10, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alpha, 10, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alpha, x_index, (void*)&beta, 10, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // only CSR is supported
        {
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSC_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
            ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }

        // only 32F and 64F real are supported
        // but we cannot check SrSPMV for that because AllocateData will throw an error first
        /*for (int i = 0; i < 10; i++)
        {
            if (i == CUDA_R_32F || i == CUDA_R_64F)
                continue;
            cudaDataType_t t_type_v[2] = {(cudaDataType_t)i, (cudaDataType_t)i};
            cudaDataType_t t_type_e[1] = {(cudaDataType_t)i};
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 2, t_type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, t_type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alpha, x_index, (void*)&beta, y_index, NVGRAPH_PLUS_TIMES_SR);
            ASSERT_EQ(NVGRAPH_STATUS_TYPE_NOT_SUPPORTED, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        */
    }
};
 
TEST_F(NVGraphCAPITests_SrSPMV_CornerCases, CornerCasesDouble)
{
    run_simple_test<double>();
    
}

TEST_F(NVGraphCAPITests_SrSPMV_CornerCases, CornerCasesFloat)
{
    run_simple_test<float>();
}


class NVGraphCAPITests_SSSP_CornerCases : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_SSSP_CornerCases() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSC_32;
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

    template <typename T>
    void run_cycle_test()
    {
        n = 1024;
        nnz = n;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[i] = i;
            neighborhood[i] = (n - 1 + i) % n;
        }
        offsets[n] = n;
        std::vector<T> edge_data(nnz, (T)1.0);

        nvgraphCSCTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
        int source_vert = 0;
        int sssp_index = 0;
        int weight_index = 0;
        
        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // only multivaluedCSR are supported
        status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
        ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);

        cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
        status = nvgraphAllocateVertexData(handle, g1, 1, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        void*  edgeptr[1] = {(void*)&edge_data[0]};
        status = nvgraphSetEdgeData(handle, g1, edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);


        status = nvgraphSssp(NULL, g1, weight_index, &source_vert, sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSssp(handle, NULL, weight_index, &source_vert, sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSssp(handle, g1, 500, &source_vert, sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSssp(handle, g1, weight_index, NULL, sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSssp(handle, g1, weight_index, &source_vert, 500);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // only CSC is supported
        {
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 1, type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
            ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }

        // only 32F and 64F real are supported
        // but we cannot check SSSP for that because AllocateData will throw an error first
        /*for (int i = 0; i < 10; i++)
        {
            if (i == CUDA_R_32F || i == CUDA_R_64F)
                continue;
            cudaDataType_t t_type_v[2] = {(cudaDataType_t)i, (cudaDataType_t)i};
            cudaDataType_t t_type_e[1] = {(cudaDataType_t)i};
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSC_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 1, t_type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, t_type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
            ASSERT_EQ(NVGRAPH_STATUS_TYPE_NOT_SUPPORTED, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        */
    }
};
 
TEST_F(NVGraphCAPITests_SSSP_CornerCases, CornerCasesDouble)
{
    run_cycle_test<double>();
}

TEST_F(NVGraphCAPITests_SSSP_CornerCases, CornerCasesFloat)
{
    run_cycle_test<float>();
}



class NVGraphCAPITests_WidestPath_CornerCases : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_WidestPath_CornerCases() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSC_32;
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

    template <typename T>
    void run_test()
    {
        n = 1024;
        nnz = n;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[i] = i;
            neighborhood[i] = (n - 1 + i) % n;
        }
        offsets[n] = n;
        std::vector<T> edge_data(nnz, (T)1.0);
        std::vector<T> expected_res(n, nvgraph_Const<T>::inf);
        for (int i = 0; i < n; i++)
        {
            expected_res[i] = i;
        }

        nvgraphCSCTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
                g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        int source_vert = 0;
        int widest_path_index = 0;
        int weight_index = 0;

        status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
        ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);


        cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
        status = nvgraphAllocateVertexData(handle, g1, 1, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        void*  edgeptr[1] = {(void*)&edge_data[0]};
        status = nvgraphSetEdgeData(handle, g1, edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);


        status = nvgraphWidestPath(NULL, g1, weight_index, &source_vert, widest_path_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphWidestPath(handle, NULL, weight_index, &source_vert, widest_path_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphWidestPath(handle, g1, 500, &source_vert, widest_path_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphWidestPath(handle, g1, weight_index, NULL, widest_path_index);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, 500);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // only CSC is supported
        {
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 1, type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
            ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }

        // only 32F and 64F real are supported
        // but we cannot check WidestPath for that because AllocateData will throw an error first
        /*for (int i = 0; i < 10; i++)
        {
            if (i == CUDA_R_32F || i == CUDA_R_64F)
                continue;
            cudaDataType_t t_type_v[2] = {(cudaDataType_t)i, (cudaDataType_t)i};
            cudaDataType_t t_type_e[1] = {(cudaDataType_t)i};
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSC_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 1, t_type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, t_type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
            ASSERT_EQ(NVGRAPH_STATUS_TYPE_NOT_SUPPORTED, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        */
    }
};
 
TEST_F(NVGraphCAPITests_WidestPath_CornerCases, CornerCasesDouble)
{
    run_test<double>();
}

TEST_F(NVGraphCAPITests_WidestPath_CornerCases, CornerCasesFloat)
{
    run_test<float>();
}


class NVGraphCAPITests_Pagerank_CornerCases : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphTopologyType_t topo;
    int n;
    int nnz;
    nvgraphGraphDescr_t g1;

    NVGraphCAPITests_Pagerank_CornerCases() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        topo = NVGRAPH_CSC_32;
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


    template <typename T>
    void run_test()
    {
        n = 1024;
        nnz = n - 1;
        std::vector<int> offsets(n+1), neighborhood(n);
        for (int i = 0; i < n; i++)
        {
            offsets[1+i] = i;
            neighborhood[i] = i;
        }
        offsets[0] = 0;
        std::vector<T> edge_data(nnz, 1.0);
        std::vector<T> dangling(n, 0);
        dangling[n-1] = (T)(1);

        nvgraphCSCTopology32I_st topology = {n, nnz, &offsets[0], &neighborhood[0]};
        
        g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        n = topology.nvertices;
        nnz = topology.nedges;
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        int bookmark_index = 0;
        int weight_index = 0;
        T alpha = 0.85;
        T alpha_bad = -10.0;
        int pagerank_index = 1;
        int has_guess = 0;
        float tolerance = 1e-6;//sizeof(T) > 4 ?  1e-8f :  1e-6f;
        int max_iter = 1000;

        // should be multivalued
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);


        status = nvgraphSetVertexData(handle, g1, (void*)&dangling[0], bookmark_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void*)&edge_data[0], weight_index );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // different invalid values
        status = nvgraphPagerank(NULL, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, NULL, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, 500, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, NULL, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha_bad, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, 500, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, 500, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, 500, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, -10.0f, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, 10.0f, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        {
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
            ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }

        // only 32F and 64F real are supported
        // but we cannot check Pagerank for that because AllocateData will throw an error first
        /*for (int i = 0; i < 10; i++)
        {
            if (i == CUDA_R_32F || i == CUDA_R_64F)
                continue;
            cudaDataType_t t_type_v[2] = {(cudaDataType_t)i, (cudaDataType_t)i};
            cudaDataType_t t_type_e[1] = {(cudaDataType_t)i};
            status = nvgraphCreateGraphDescr(handle, &g1);  
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSC_32);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateVertexData(handle, g1, 2, t_type_v );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphAllocateEdgeData(handle, g1, 1, t_type_e);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
            ASSERT_EQ(NVGRAPH_STATUS_TYPE_NOT_SUPPORTED, status);
            status = nvgraphDestroyGraphDescr(handle, g1);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        */
    }
};
 
TEST_F(NVGraphCAPITests_Pagerank_CornerCases, CornerCasesDouble)
{
    run_test<double>();
}

TEST_F(NVGraphCAPITests_Pagerank_CornerCases, CornerCasesFloat)
{
    run_test<float>();
}


class NVGraphCAPITests_SrSPMV_Stress : public ::testing::TestWithParam<SrSPMV_Usecase> {
  public:
    NVGraphCAPITests_SrSPMV_Stress() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        //const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        //printf("We are in test %s of test case %s.\n", test_info->name(), test_info->test_case_name());
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
    void run_current_test(const SrSPMV_Usecase& param)
    {
        nvgraphTopologyType_t topo = NVGRAPH_CSR_32;

        nvgraphStatus_t status;

        FILE* fpin = fopen(param.graph_file.c_str(),"rb");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        int n, nnz;
        //Read a transposed network in amgx binary format and the bookmark of dangling nodes
        ASSERT_EQ(read_header_amgx_csr_bin (fpin, n, nnz), 0);
        std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
        std::vector<T> read_val(nnz);
        ASSERT_EQ(read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, read_val), 0);
        fclose(fpin);

        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();

        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(read_row_ptr.size() + read_col_ind.size())))
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
        //@TODO: random fill?
        std::vector<T> calculated_res(n);
        std::vector<T> data1(n), data2(n);
        for (int i = 0; i < n; i++)
        {
            data1[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
            data2[i] = (T)(1.0*rand()/RAND_MAX - 0.5);
        }
        void*  vertexptr[2] = {(void*)&data1[0], (void*)&data2[0]};
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, 2, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[1], 1 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int weight_index = 0;
        int x_index = 0;
        int y_index = 1;

        // reinit data
        status = nvgraphSetVertexData(handle, g1, (void*)&data2[0], y_index);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);


        T alphaT = (T)param.alpha;
        T betaT = (T)param.beta;

        // run
        int repeat = std::max((int)(((float)(SRSPMV_ITER_MULTIPLIER)*STRESS_MULTIPLIER)/n), 1);
        //printf ("Repeating C API call for %d times\n", repeat);
        std::vector<T> calculated_res1(n), calculated_res_mid(n);
        size_t free_mid = 0, free_last = 0, total = 0;      
        for (int i = 0; i < repeat; i++)
        {
//            cudaMemGetInfo(&t, &total);
//            printf("Iteration: %d, freemem: %zu\n", i, t);

            status = nvgraphSrSpmv(handle, g1, weight_index, (void*)&alphaT, x_index, (void*)&betaT, y_index, param.sr);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

            // all of those should be equal
            if (i == 0)
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res1[0], y_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            else
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res_mid[0], y_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

                for (int row = 0; row < n; row++)
                {
                    // stronger condition - bit by bit equality
                    /*
                    if (calculated_res1[row] != calculated_res_mid[row])
                    {
                        typename nvgraph_Const<T>::fpint_st comp1, comp2;
                        comp1.f = calculated_res1[row];
                        comp2.f = calculated_res_mid[row];
                        ASSERT_EQ(comp1.u, comp2.u) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" << i;
                    }
                    */
                    ASSERT_NEAR(calculated_res1[row], calculated_res_mid[row], nvgraph_Const<T>::tol) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" <<  i;
                }
            }
            if (i == std::min(50, (int)(repeat/2)))
            {
                cudaMemGetInfo(&free_mid, &total);
            }
            if (i == repeat-1)
            {
                cudaMemGetInfo(&free_last, &total);
            }

            // reset vectors
            status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0 );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            status = nvgraphSetVertexData(handle, g1, vertexptr[1], 1 );
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }

        ASSERT_LE(free_mid, free_last) << "Memory difference between iteration #" << std::min(50, (int)(repeat/2)) << " and last iteration is " << (double)(free_last-free_mid)/1e+6 << "MB";

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
};
 
TEST_P(NVGraphCAPITests_SrSPMV_Stress, StressDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_SrSPMV_Stress, StressFloat)
{
    run_current_test<float>(GetParam());
}



class NVGraphCAPITests_Widest_Stress : public ::testing::TestWithParam<WidestPath_Usecase> {
  public:
    NVGraphCAPITests_Widest_Stress() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        //const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        //printf("We are in test %s of test case %s.\n", test_info->name(), test_info->test_case_name());
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
    void run_current_test(const WidestPath_Usecase& param)
    {
        nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

        nvgraphStatus_t status;

        FILE* fpin = fopen(param.graph_file.c_str(),"rb");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        int n, nnz;
        //Read a transposed network in amgx binary format and the bookmark of dangling nodes
        ASSERT_EQ(read_header_amgx_csr_bin (fpin, n, nnz), 0);
        std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
        std::vector<T> read_val(nnz);
        ASSERT_EQ(read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, read_val), 0);
        fclose(fpin);

        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();

        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(read_row_ptr.size() + read_col_ind.size())))
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
        cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, numsets, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, numsets, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int weight_index = 0;
        int source_vert = param.source_vert;
        int widest_path_index = 0;

        // run
        int repeat = std::max((int)(((float)(WIDEST_ITER_MULTIPLIER)*STRESS_MULTIPLIER)/(3*n)), 1);
        //printf ("Repeating C API call for %d times\n", repeat);
        std::vector<T> calculated_res1(n), calculated_res_mid(n);
        size_t free_mid = 0, free_last = 0, total = 0;      
        for (int i = 0; i < repeat; i++)
        {
            //cudaMemGetInfo(&t, &total);
            //printf("Iteration: %d, freemem: %zu\n", i, t);

            status = nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

            // all of those should be equal
            if (i == 0)
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res1[0], widest_path_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            else
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res_mid[0], widest_path_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

                for (int row = 0; row < n; row++)
                {
                    // stronger condition - bit by bit equality
                    /*
                    if (calculated_res1[row] != calculated_res_mid[row])
                    {
                        typename nvgraph_Const<T>::fpint_st comp1, comp2;
                        comp1.f = calculated_res1[row];
                        comp2.f = calculated_res_mid[row];
                        ASSERT_EQ(comp1.u, comp2.u) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" << i;
                    }
                    */
                    ASSERT_NEAR(calculated_res1[row], calculated_res_mid[row], nvgraph_Const<T>::tol) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" <<  i;
                }
            }

            if (i == std::min(50, (int)(repeat/2)))
            {
                cudaMemGetInfo(&free_mid, &total);
            }
            if (i == repeat-1)
            {
                cudaMemGetInfo(&free_last, &total);
            }
        }

        ASSERT_LE(free_mid, free_last) << "Memory difference between iteration #" << std::min(50, (int)(repeat/2)) << " and last iteration is " << (double)(free_last-free_mid)/1e+6 << "MB";

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
};
 
TEST_P(NVGraphCAPITests_Widest_Stress, StressDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_Widest_Stress, StressFloat)
{
    run_current_test<float>(GetParam());
}




class NVGraphCAPITests_SSSP_Stress : public ::testing::TestWithParam<SSSP_Usecase> {
  public:
    NVGraphCAPITests_SSSP_Stress() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        //const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        //printf("We are in test %s of test case %s.\n", test_info->name(), test_info->test_case_name());
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
        nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

        nvgraphStatus_t status;

        FILE* fpin = fopen(param.graph_file.c_str(),"rb");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        int n, nnz;
        //Read a transposed network in amgx binary format and the bookmark of dangling nodes
        ASSERT_EQ(read_header_amgx_csr_bin (fpin, n, nnz), 0);
        std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
        std::vector<T> read_val(nnz);
        ASSERT_EQ(read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, read_val), 0);
        fclose(fpin);

        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();

        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(read_row_ptr.size() + read_col_ind.size())))
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
        cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, numsets, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, numsets, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int weight_index = 0;
        int source_vert = param.source_vert;
        int sssp_index = 0;

        // run
        int repeat = std::max((int)(((float)(SSSP_ITER_MULTIPLIER)*STRESS_MULTIPLIER)/(3*n)), 1);
        //printf ("Repeating C API call for %d times\n", repeat);
        std::vector<T> calculated_res1(n), calculated_res_mid(n), calculated_res_last(n);
        size_t free_mid = 0, free_last = 0, total = 0;      
        for (int i = 0; i < repeat; i++)
        {
//            cudaMemGetInfo(&t, &total);
//            printf("Iteration: %d, freemem: %zu\n", i, t);

            status = nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

            // all of those should be equal
            if (i == 0)
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res1[0], sssp_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            else
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res_mid[0], sssp_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

                for (int row = 0; row < n; row++)
                {
                    // stronger condition - bit by bit equality
                    /*
                    if (calculated_res1[row] != calculated_res_mid[row])
                    {
                        typename nvgraph_Const<T>::fpint_st comp1, comp2;
                        comp1.f = calculated_res1[row];
                        comp2.f = calculated_res_mid[row];
                        ASSERT_EQ(comp1.u, comp2.u) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" << i;
                    }
                    */
                    ASSERT_NEAR(calculated_res1[row], calculated_res_mid[row], nvgraph_Const<T>::tol) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" <<  i;
                }
            }

            if (i == std::min(50, (int)(repeat/2)))
            {
                cudaMemGetInfo(&free_mid, &total);
            }
            if (i == repeat-1)
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res_last[0], sssp_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
                cudaMemGetInfo(&free_last, &total);
            }
        }

        ASSERT_LE(free_mid, free_last) << "Memory difference between iteration #" << std::min(50, (int)(repeat/2)) << " and last iteration is " << (double)(free_last-free_mid)/1e+6 << "MB";

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
};
 
TEST_P(NVGraphCAPITests_SSSP_Stress, StressDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_SSSP_Stress, StressFloat)
{
    run_current_test<float>(GetParam());
}




class NVGraphCAPITests_Pagerank_Stress : public ::testing::TestWithParam<Pagerank_Usecase> {
  public:
    NVGraphCAPITests_Pagerank_Stress() : handle(NULL) {}

    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        //const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        //printf("We are in test %s of test case %s.\n", test_info->name(), test_info->test_case_name());
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
    void run_current_test(const Pagerank_Usecase& param)
    {
        nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

        nvgraphStatus_t status;

        FILE* fpin = fopen(param.graph_file.c_str(),"rb");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        int n, nnz;
        //Read a transposed network in amgx binary format and the bookmark of dangling nodes
        ASSERT_EQ(read_header_amgx_csr_bin (fpin, n, nnz), 0);
        std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
        std::vector<T> read_val(nnz);
        std::vector<T> dangling(n);
        ASSERT_EQ(read_data_amgx_csr_bin_rhs (fpin, n, nnz, read_row_ptr, read_col_ind, read_val, dangling), 0);
        fclose(fpin);

        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();

        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(read_row_ptr.size() + read_col_ind.size())))
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
        std::vector<T> calculated_res(n, (T)1.0/n);
        void*  vertexptr[2] = {(void*)&dangling[0], (void*)&calculated_res[0]};
        cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&read_val[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateVertexData(handle, g1, 2, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, g1, vertexptr[1], 1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, 1, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        
        int bookmark_index = 0;
        int weight_index = 0;
        T alpha = param.alpha;
        int pagerank_index = 1;
        int has_guess = 1;
        float tolerance = {sizeof(T) > 4 ?  1e-8f :  1e-6f};
        int max_iter = 1000;


        // run
        int repeat = std::max((int)(((float)(PAGERANK_ITER_MULTIPLIER)*STRESS_MULTIPLIER)/n), 1);
        //printf ("Repeating C API call for %d times\n", repeat);
        std::vector<T> calculated_res1(n), calculated_res_mid(n);
        size_t free_mid = 0, free_last = 0, total = 0;      
        for (int i = 0; i < repeat; i++)
        {
            //cudaMemGetInfo(&t, &total);
            //printf("Iteration: %d, freemem: %zu\n", i, t);

            status = nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

            // all of those should be equal
            if (i == 0)
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res1[0], pagerank_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            }
            else
            {
                status = nvgraphGetVertexData(handle, g1, (void *)&calculated_res_mid[0], pagerank_index);
                ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

                for (int row = 0; row < n; row++)
                {
                    // stronger condition - bit by bit equality
                    /*
                    if (calculated_res1[row] != calculated_res_mid[row])
                    {
                        typename nvgraph_Const<T>::fpint_st comp1, comp2;
                        comp1.f = calculated_res1[row];
                        comp2.f = calculated_res_mid[row];
                        ASSERT_EQ(comp1.u, comp2.u) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" << i;
                    }
                    */
                    ASSERT_NEAR(calculated_res1[row], calculated_res_mid[row], nvgraph_Const<T>::tol) << "Difference in result in row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" <<  i;
                }
            }

            if (i == std::min(50, (int)(repeat/2)))
            {
                cudaMemGetInfo(&free_mid, &total);
            }
            if (i == repeat-1)
            {
                cudaMemGetInfo(&free_last, &total);
            }
        }

        ASSERT_LE(free_mid, free_last) << "Memory difference between iteration #" << std::min(50, (int)(repeat/2)) << " and last iteration is " << (double)(free_last-free_mid)/1e+6 << "MB";

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
};
 
TEST_P(NVGraphCAPITests_Pagerank_Stress, StressDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_Pagerank_Stress, StressFloat)
{
    run_current_test<float>(GetParam());
}



// instatiation of the performance/correctness checks 

INSTANTIATE_TEST_CASE_P(CorrectnessCheck1,
                        NVGraphCAPITests_SrSPMV,
                        ::testing::Values(    // maybe check NVGRAPH_OR_AND_SR on some special bool matrices?
                                              SrSPMV_Usecase("graphs/small/small.bin", NVGRAPH_PLUS_TIMES_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/small/small.bin", NVGRAPH_MIN_PLUS_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/small/small.bin", NVGRAPH_MAX_MIN_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/small/small.bin", NVGRAPH_OR_AND_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_PLUS_TIMES_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MIN_PLUS_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MAX_MIN_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_OR_AND_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_PLUS_TIMES_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MIN_PLUS_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MAX_MIN_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_OR_AND_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_PLUS_TIMES_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MIN_PLUS_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MAX_MIN_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_OR_AND_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_PLUS_TIMES_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MIN_PLUS_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_MAX_MIN_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/dblp/dblp.bin", NVGRAPH_OR_AND_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_PLUS_TIMES_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MIN_PLUS_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MAX_MIN_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_OR_AND_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_PLUS_TIMES_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MIN_PLUS_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MAX_MIN_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_OR_AND_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_PLUS_TIMES_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MIN_PLUS_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MAX_MIN_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_OR_AND_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_PLUS_TIMES_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MIN_PLUS_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_MAX_MIN_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2003/wiki2003.bin", NVGRAPH_OR_AND_SR, 1, 1)
                                            ///// more instances
                                            )
                        );


INSTANTIATE_TEST_CASE_P(CorrectnessCheck2,
                        NVGraphCAPITests_SrSPMV,
                        ::testing::Values(
                                              SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_PLUS_TIMES_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MIN_PLUS_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MAX_MIN_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_OR_AND_SR, 0, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_PLUS_TIMES_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MIN_PLUS_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MAX_MIN_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_OR_AND_SR, 0, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MIN_PLUS_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MAX_MIN_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_OR_AND_SR, 1, 0)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MIN_PLUS_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MAX_MIN_SR, 1, 1)
                                            , SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_OR_AND_SR, 1, 1)
                                            // these tests fails because of exceeding tolerance: diff = 0.00012826919555664062 vs tol = 9.9999997473787516e-05
                                            //, SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_PLUS_TIMES_SR, 1, 1)
                                            //, SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_PLUS_TIMES_SR, 1, 0)
                                            ///// more instances
                                            )
                        );


INSTANTIATE_TEST_CASE_P(CorrectnessCheck,
                       NVGraphCAPITests_WidestPath,
                          //                                  graph FILE                                                 source vert #     file with expected result (in binary?)
//                                            // we read matrix stored in CSR and pass it as CSC - so matrix is in fact transposed, that's why we compare it to the results calculated on a transposed matrix
                       ::testing::Values(    
                                                WidestPath_Usecase("graphs/cage/cage13_T.mtx.bin", 0,   "graphs/cage/cage13.widest_0.bin")
                                              , WidestPath_Usecase("graphs/cage/cage13_T.mtx.bin", 101, "graphs/cage/cage13.widest_101.bin")
                                              , WidestPath_Usecase("graphs/cage/cage14_T.mtx.bin", 0,   "graphs/cage/cage14.widest_0.bin")
                                              , WidestPath_Usecase("graphs/cage/cage14_T.mtx.bin", 101, "graphs/cage/cage14.widest_101.bin")
                                              // file might be missing on eris
                                              //, WidestPath_Usecase("graphs/small/small_T.bin", 2,  "graphs/small/small_T.widest_2.bin")
                                              , WidestPath_Usecase("graphs/dblp/dblp.bin", 100, "graphs/dblp/dblp_T.widest_100.bin")
                                              , WidestPath_Usecase("graphs/dblp/dblp.bin", 100000, "graphs/dblp/dblp_T.widest_100000.bin")
                                              , WidestPath_Usecase("graphs/Wikipedia/2003/wiki2003_T.bin", 100,  "graphs/Wikipedia/2003/wiki2003_T.widest_100.bin")
                                              , WidestPath_Usecase("graphs/Wikipedia/2003/wiki2003_T.bin", 100000, "graphs/Wikipedia/2003/wiki2003_T.widest_100000.bin")
                                              , WidestPath_Usecase("graphs/citPatents/cit-Patents_T.mtx.bin", 6543, "")
                                              //, WidestPath_Usecase("dimacs10/kron_g500-logn20_T.mtx.bin", 100000, "")
                                              //, WidestPath_Usecase("dimacs10/hugetrace-00020_T.mtx.bin", 100000, "")
                                              //, WidestPath_Usecase("dimacs10/delaunay_n24_T.mtx.bin", 100000, "")
                                              //, WidestPath_Usecase("dimacs10/road_usa_T.mtx.bin", 100000, "")
                                              //, WidestPath_Usecase("dimacs10/hugebubbles-00020_T.mtx.bin", 100000, "")
                                           ///// more instances
                                           )
                       );


INSTANTIATE_TEST_CASE_P(CorrectnessCheck,
                        NVGraphCAPITests_SSSP,
                        //                                  graph FILE                                                  source vert #    file with expected result (in binary?)
//                                            // we read matrix stored in CSR and pass it as CSC - so matrix is in fact transposed, that's why we compare it to the results calculated on a transposed matrix
                        ::testing::Values(    
                                                SSSP_Usecase("graphs/cage/cage13_T.mtx.bin", 0,   "graphs/cage/cage13.sssp_0.bin")
                                              , SSSP_Usecase("graphs/cage/cage13_T.mtx.bin", 101, "graphs/cage/cage13.sssp_101.bin")
                                              , SSSP_Usecase("graphs/cage/cage14_T.mtx.bin", 0,   "graphs/cage/cage14.sssp_0.bin")
                                              , SSSP_Usecase("graphs/cage/cage14_T.mtx.bin", 101, "graphs/cage/cage14.sssp_101.bin")
                                              , SSSP_Usecase("graphs/small/small.bin", 2, "graphs/small/small.sssp_2.bin")
                                              , SSSP_Usecase("graphs/dblp/dblp.bin", 100,    "graphs/dblp/dblp_T.sssp_100.bin")
                                              , SSSP_Usecase("graphs/dblp/dblp.bin", 100000, "graphs/dblp/dblp_T.sssp_100000.bin")
                                              , SSSP_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 100,    "graphs/Wikipedia/2003/wiki2003_T.sssp_100.bin")
                                              , SSSP_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 100000, "graphs/Wikipedia/2003/wiki2003_T.sssp_100000.bin")
                                              , SSSP_Usecase("graphs/citPatents/cit-Patents_T.mtx.bin", 6543, "")
                                              //, SSSP_Usecase("dimacs10/kron_g500-logn20_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/hugetrace-00020_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/delaunay_n24_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/road_usa_T.mtx.bin", 100000, "")
                                              //, SSSP_Usecase("dimacs10/hugebubbles-00020_T.mtx.bin", 100000, "")
                                            ///// more instances
                                         )
                        );
INSTANTIATE_TEST_CASE_P(CorrectnessCheck,
                        NVGraphCAPITests_Pagerank,
                        //                                        graph FILE                                                  alpha                file with expected result                                            
                        ::testing::Values(    
                                           // Pagerank_Usecase("graphs/small/small_T.bin", 0.85, "graphs/small/small.pagerank_val_0.85.bin"),
                                            Pagerank_Usecase("graphs/webbase1M/webbase-1M_T.mtx.bin", 0.85, "graphs/webbase1M/webbase-1M.pagerank_val_0.85.bin"),
                                            Pagerank_Usecase("graphs/webBerkStan/web-BerkStan_T.mtx.bin", 0.85, "graphs/webBerkStan/web-BerkStan.pagerank_val_0.85.bin"),
                                            Pagerank_Usecase("graphs/webGoogle/web-Google_T.mtx.bin", 0.85, "graphs/webGoogle/web-Google.pagerank_val_0.85.bin"),
                                            Pagerank_Usecase("graphs/WikiTalk/wiki-Talk_T.mtx.bin", 0.85, "graphs/WikiTalk/wiki-Talk.pagerank_val_0.85.bin"),
                                            Pagerank_Usecase("graphs/citPatents/cit-Patents_T.mtx.bin", 0.85, "graphs/citPatents/cit-Patents.pagerank_val_0.85.bin"),
                                            Pagerank_Usecase("graphs/liveJournal/ljournal-2008_T.mtx.bin", 0.85, "graphs/liveJournal/ljournal-2008.pagerank_val_0.85.bin"),
                                            Pagerank_Usecase("dummy", 0.85, ""),
                                            Pagerank_Usecase("dimacs10/delaunay_n24_T.mtx.bin", 0.85, ""),
                                            Pagerank_Usecase("dummy", 0.85, ""), // waived until cublas change, see http://nvbugs/200189611, was: Pagerank_Usecase("dimacs10/hugebubbles-00020_T.mtx.bin", 0.85, ""),
                                            Pagerank_Usecase("dimacs10/hugetrace-00020_T.mtx.bin", 0.85, "", 10.0),
                                            Pagerank_Usecase("dimacs10/kron_g500-logn20_T.mtx.bin", 0.85, ""),
                                            Pagerank_Usecase("dimacs10/road_usa_T.mtx.bin", 0.85, "")
                                            //Pagerank_Usecase("dimacs10/channel-500x100x100-b050_T.mtx.bin", 0.85, ""),
                                            //Pagerank_Usecase("dimacs10/coPapersCiteseer_T.mtx.bin", 0.85, "")
                                            ///// more instances
                                            )
                        );


//INSTANTIATE_TEST_CASE_P(CorrectnessCheck,
//                        NVGraphCAPITests_KrylovPagerank,
//                        //                                        graph FILE                                                  alpha                file with expected result                                            
//                        ::testing::Values(    
//                                            //Pagerank_Usecase("graphs/small/small_T.bin", 0.85, "graphs/small/small.pagerank_val_0.85.bin"),
//                                            Pagerank_Usecase("graphs/webbase1M/webbase-1M_T.mtx.bin", 0.85, "graphs/webbase1M/webbase-1M.pagerank_val_0.85.bin"),
//                                            Pagerank_Usecase("graphs/webBerkStan/web-BerkStan_T.mtx.bin", 0.85, "graphs/webBerkStan/web-BerkStan.pagerank_val_0.85.bin"),
//                                            Pagerank_Usecase("graphs/webGoogle/web-Google_T.mtx.bin", 0.85, "graphs/webGoogle/web-Google.pagerank_val_0.85.bin"),
//                                            Pagerank_Usecase("graphs/WikiTalk/wiki-Talk_T.mtx.bin", 0.85, "graphs/WikiTalk/wiki-Talk.pagerank_val_0.85.bin"),
//                                            Pagerank_Usecase("graphs/citPatents/cit-Patents_T.mtx.bin", 0.85, "graphs/citPatents/cit-Patents.pagerank_val_0.85.bin"),
//                                            Pagerank_Usecase("graphs/liveJournal/ljournal-2008_T.mtx.bin", 0.85, "graphs/liveJournal/ljournal-2008.pagerank_val_0.85.bin"),
//                                            Pagerank_Usecase("dummy", 0.85, ""),
//                                            Pagerank_Usecase("dimacs10/delaunay_n24_T.mtx.bin", 0.85, ""),
//                                            Pagerank_Usecase("dimacs10/hugebubbles-00020_T.mtx.bin", 0.85, ""),
//                                            Pagerank_Usecase("dimacs10/hugetrace-00020_T.mtx.bin", 0.85, "", 10.0),
//                                            Pagerank_Usecase("dimacs10/kron_g500-logn20_T.mtx.bin", 0.85, ""),
//                                            Pagerank_Usecase("dimacs10/road_usa_T.mtx.bin", 0.85, "")
//                                            //Pagerank_Usecase("dimacs10/channel-500x100x100-b050_T.mtx.bin", 0.85, ""),
//                                            //Pagerank_Usecase("dimacs10/coPapersCiteseer_T.mtx.bin", 0.85, "")
//                                            ///// more instances
//                                            )
//                        );

INSTANTIATE_TEST_CASE_P(StressTest,
                        NVGraphCAPITests_SrSPMV_Stress,
                        ::testing::Values(
                                              SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_PLUS_TIMES_SR, 1, 1),
                                              SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MIN_PLUS_SR, 1, 1),
                                              SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_MAX_MIN_SR, 1, 1),
                                              SrSPMV_Usecase("graphs/Wikipedia/2011/wiki2011.bin", NVGRAPH_OR_AND_SR, 1, 1)
                                              )
                        );


INSTANTIATE_TEST_CASE_P(StressTest,
                        NVGraphCAPITests_Widest_Stress,
                        ::testing::Values(
                                              WidestPath_Usecase("graphs/citPatents/cit-Patents_T.mtx.bin", 6543, "")
                                            )
                        );


INSTANTIATE_TEST_CASE_P(StressTest,
                        NVGraphCAPITests_SSSP_Stress,
                        ::testing::Values(
                                              SSSP_Usecase("graphs/citPatents/cit-Patents_T.mtx.bin", 6543, "")
                                            )
                        );


INSTANTIATE_TEST_CASE_P(StressTest,
                        NVGraphCAPITests_Pagerank_Stress,
                        ::testing::Values(
                                              Pagerank_Usecase("graphs/citPatents/cit-Patents_T.mtx.bin", 0.7, "")
                                            )
                        );


int main(int argc, char **argv) 
{

    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--perf") == 0)
            PERF = 1;
        if (strcmp(argv[i], "--stress-iters") == 0)
            STRESS_MULTIPLIER = atoi(argv[i+1]);
        if (strcmp(argv[i], "--ref-data-dir") == 0)
            ref_data_prefix = std::string(argv[i+1]);
        if (strcmp(argv[i], "--graph-data-dir") == 0)
            graph_data_prefix = std::string(argv[i+1]);
    }
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
        
  return RUN_ALL_TESTS();
}
