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
#include "nvgraph_test_common.h"
#include "valued_csr_graph.hxx"
#include "readMatrix.hxx"
#include "nvgraphP.h"
#include "nvgraph.h"
#include "nvgraph_experimental.h"
#include "stdlib.h"
#include "stdint.h"
#include <algorithm>
extern "C" {
#include "mmio.h"
}
#include "mm.hxx"
// minimum vertices in the graph to perform perf measurements
#define PERF_ROWS_LIMIT 10000

// number of repeats = multiplier/num_vertices
#define SRSPMV_ITER_MULTIPLIER   1000000000
#define SSSP_ITER_MULTIPLIER     30000000
#define WIDEST_ITER_MULTIPLIER   30000000
#define PAGERANK_ITER_MULTIPLIER 300000000

// utility

#define NVGRAPH_SAFE_CALL(call) \
{\
    nvgraphStatus_t status = (call) ;\
    if ( NVGRAPH_STATUS_SUCCESS != status )\
    {\
        std::cout << "Error #" << status << " in " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);\
    }\
} 

#define CUDA_SAFE_CALL(call) \
{\
    cudaError_t status = (call) ;\
    if ( cudaSuccess != status )\
    {\
        std::cout << "Error #" << status << " in " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);\
    }\
} 

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

template <>
struct nvgraph_Const<int>
{ 
    static const cudaDataType_t Type = CUDA_R_32I;
    static const int inf;
    static const int tol;

};

const int nvgraph_Const<int>::inf = INT_MAX;
const int nvgraph_Const<int>::tol = 0;

typedef struct SrSPMV_Usecase_t
{
    std::string graph_file;
    int repeats;
    SrSPMV_Usecase_t(const std::string& a, const int b) : graph_file(a), repeats(b){};
    SrSPMV_Usecase_t& operator=(const SrSPMV_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        repeats = rhs.repeats;
        return *this;
    }
} SrSPMV_Usecase;

template <typename T>
void run_srspmv_bench(const SrSPMV_Usecase& param)
{
    std::cout << "Initializing nvGRAPH library..." << std::endl; 

    nvgraphHandle_t handle = NULL;

    if (handle == NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));
    }

    std::cout << "Reading input data..." << std::endl;  

    FILE* fpin = fopen(param.graph_file.c_str(),"r");
    if (fpin == NULL)
    {
        std::cout << "Cannot open input graph file: " << param.graph_file << std::endl;  
        exit(1);
    } 

    int m, n, nnz;
    MM_typecode mc;
 
    if(mm_properties<int>(fpin, 1, &mc, &m, &n, &nnz) != 0) 
    {
        std::cout <<  "could not read Matrix Market file properties"<< "\n";
        exit(1);
    }

    std::vector<int> read_row_ptr(n+1), read_col_ind(nnz), coo_row_ind(nnz);
    std::vector<T> csr_read_val(nnz);
        
    if(mm_to_coo<int,T>(fpin, 1, nnz, &coo_row_ind[0], &read_col_ind[0], &csr_read_val[0], NULL)) 
    {
        std::cout << "could not read matrix data"<< "\n";
        exit(1);
    }

    if(coo_to_csr<int,T> (n, n, nnz, &coo_row_ind[0],  &read_col_ind[0], &csr_read_val[0], NULL, &read_row_ptr[0], NULL, NULL, NULL)) 
    {
        std::cout << "could not covert COO to CSR "<< "\n";
        exit(1);
    }

    //Read a transposed network in amgx binary format and the bookmark of dangling nodes
    /*if (read_header_amgx_csr_bin (fpin, n, nnz) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
    std::vector<T> read_val(nnz);
    if (read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, read_val) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }*/
    fclose(fpin);

    std::cout << "Initializing data structures ..." << std::endl;  

    nvgraphGraphDescr_t g1 = NULL;
    NVGRAPH_SAFE_CALL(nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphTopologyType_t topo = NVGRAPH_CSR_32;
    nvgraphCSRTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
    NVGRAPH_SAFE_CALL(nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo));

    // set up graph data
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
    
    void*  edgeptr[1] = {(void*)&csr_read_val[0]};
    cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

    int weight_index = 0;
    int x_index = 0;
    int y_index = 1;
    NVGRAPH_SAFE_CALL(nvgraphAllocateVertexData(handle, g1, 2, type_v ));
    NVGRAPH_SAFE_CALL(nvgraphSetVertexData(handle, g1, vertexptr[0], x_index ));
    NVGRAPH_SAFE_CALL(nvgraphSetVertexData(handle, g1, vertexptr[1], y_index ));
    NVGRAPH_SAFE_CALL(nvgraphAllocateEdgeData(handle, g1, 1, type_e));
    NVGRAPH_SAFE_CALL(nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], weight_index ));

    // run
    double start, stop, total = 0.;
    T alphaT = 1., betaT = 0.;
    nvgraphSemiring_t sr = NVGRAPH_PLUS_TIMES_SR;
    int repeat = std::max(param.repeats, 1);
    NVGRAPH_SAFE_CALL(nvgraphSrSpmv(handle, g1, weight_index, (void*)&alphaT, x_index, (void*)&betaT, y_index, sr));
    NVGRAPH_SAFE_CALL(nvgraphSrSpmv(handle, g1, weight_index, (void*)&alphaT, x_index, (void*)&betaT, y_index, sr));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout << "Running spmv for " << repeat << " times..." << std::endl;
    std::cout << "n = " << n << ", nnz = " << nnz << std::endl;
    for (int i = 0; i < repeat; i++)
    {
        start = second();
        start = second();
        NVGRAPH_SAFE_CALL(nvgraphSrSpmv(handle, g1, weight_index, (void*)&alphaT, x_index, (void*)&betaT, y_index, sr));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        stop = second();
        total += stop - start;
    }
    std::cout << "nvgraph time = " << 1000.*total/((double)repeat) << std::endl;

    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));

    if (handle != NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphDestroy(handle));
        handle = NULL;
    }
}

typedef struct WidestPath_Usecase_t
{
    std::string graph_file;
    int source_vert;
    int repeats;
    WidestPath_Usecase_t(const std::string& a, int b, const int c) : graph_file(a), source_vert(b), repeats(c){};
    WidestPath_Usecase_t& operator=(const WidestPath_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        source_vert = rhs.source_vert;
        repeats = rhs.repeats;
        return *this;
    }
} WidestPath_Usecase;

// ref functions taken from cuSparse
template <typename T_ELEM>
void ref_csr2csc (int m, int n, int nnz, const T_ELEM *csrVals, const int *csrRowptr, const int *csrColInd, T_ELEM *cscVals, int *cscRowind, int *cscColptr, int base=0){
    int i,j, row, col, index;
    int * counters;
    T_ELEM val;

    /* early return */
    if ((m <= 0) || (n <= 0) || (nnz <= 0)){
        return;
    }

    /* build compressed column pointers */
    memset(cscColptr, 0, (n+1)*sizeof(cscColptr[0]));
    cscColptr[0]=base;
    for (i=0; i<nnz; i++){
        cscColptr[1+csrColInd[i]-base]++;
    }
    for(i=0; i<n; i++){
        cscColptr[i+1]+=cscColptr[i];
    }

    /* expand row indecis and copy them and values into csc arrays according to permutation */
    counters = (int *)malloc(n*sizeof(counters[0]));
    memset(counters, 0, n*sizeof(counters[0]));
    for (i=0; i<m; i++){
        for (j=csrRowptr[i]; j<csrRowptr[i+1]; j++){
            row = i+base;
            col = csrColInd[j-base];

            index=cscColptr[col-base]-base+counters[col-base];
            counters[col-base]++;

            cscRowind[index]=row;

            if(csrVals!=NULL || cscVals!=NULL){
                val = csrVals[j-base];
                cscVals[index]  = val;
            }
        }
    }
    free(counters);
}

template <typename T>
void run_widest_bench(const WidestPath_Usecase& param)
{
    std::cout << "Initializing nvGRAPH library..." << std::endl; 

    nvgraphHandle_t handle = NULL;

    if (handle == NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));
    }

    nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

    std::cout << "Reading input data..." << std::endl;  

    FILE* fpin = fopen(param.graph_file.c_str(),"r");
    if (fpin == NULL)
    {
        std::cout << "Cannot open input graph file: " << param.graph_file << std::endl;  
        exit(1);
    } 

    int n, nnz;
    //Read a transposed network in amgx binary format and the bookmark of dangling nodes
    if (read_header_amgx_csr_bin (fpin, n, nnz) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
    std::vector<T> read_val(nnz);
    if (read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, read_val) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    fclose(fpin);

    std::cout << "Initializing data structures ..." << std::endl;  

    nvgraphGraphDescr_t g1 = NULL;
    NVGRAPH_SAFE_CALL(nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphCSCTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
    NVGRAPH_SAFE_CALL(nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo));

    // set up graph data
    size_t numsets = 1;
    std::vector<T> calculated_res(n);
    //void*  vertexptr[1] = {(void*)&calculated_res[0]};
    cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
    
    void*  edgeptr[1] = {(void*)&read_val[0]};
    cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

    NVGRAPH_SAFE_CALL(nvgraphAllocateVertexData(handle, g1, numsets, type_v));
    NVGRAPH_SAFE_CALL(nvgraphAllocateEdgeData(handle, g1, numsets, type_e ));
    NVGRAPH_SAFE_CALL(nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 ));

    int weight_index = 0;
    int source_vert = param.source_vert;
    int widest_path_index = 0;


    // run
    std::cout << "Running algorithm..." << std::endl;
    double start, stop;
    start = second();
    start = second();
    int repeat = std::max(param.repeats, 1);
    for (int i = 0; i < repeat; i++)
        NVGRAPH_SAFE_CALL(nvgraphWidestPath(handle, g1, weight_index, &source_vert, widest_path_index));
    stop = second();
    printf("Time of single WidestPath call is %10.8fsecs\n", (stop-start)/repeat);
    
    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));

    if (handle != NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphDestroy(handle));
        handle = NULL;
    }
}

typedef struct SSSP_Usecase_t
{
    std::string graph_file;
    int source_vert;
    int repeats;
    SSSP_Usecase_t(const std::string& a, int b, int c) : graph_file(a), source_vert(b), repeats(c){};
    SSSP_Usecase_t& operator=(const SSSP_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        source_vert = rhs.source_vert; 
        repeats = rhs.repeats;
        return *this;
    } 
} SSSP_Usecase;

template <typename T>
void run_sssp_bench(const SSSP_Usecase& param)
{
    std::cout << "Initializing nvGRAPH library..." << std::endl;  

    nvgraphHandle_t handle = NULL;

    if (handle == NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));
    }

    nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

    std::cout << "Reading input data..." << std::endl; 

    FILE* fpin = fopen(param.graph_file.c_str(),"r");
    if (fpin == NULL)
    {
        std::cout << "Cannot read input graph file: " << param.graph_file << std::endl;  
        exit(1);
    } 

    int n, nnz;
    //Read a transposed network in amgx binary format and the bookmark of dangling nodes
    if (read_header_amgx_csr_bin (fpin, n, nnz) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
    std::vector<T> read_val(nnz);
    if (read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, read_val) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    fclose(fpin);

    std::cout << "Initializing data structures ..." << std::endl;  

    nvgraphGraphDescr_t g1 = NULL;
    NVGRAPH_SAFE_CALL(nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphCSCTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
    NVGRAPH_SAFE_CALL(nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo));

    // set up graph data
    size_t numsets = 1;
    cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
    void*  edgeptr[1] = {(void*)&read_val[0]};
    cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

    NVGRAPH_SAFE_CALL(nvgraphAllocateVertexData(handle, g1, numsets, type_v));
    NVGRAPH_SAFE_CALL(nvgraphAllocateEdgeData(handle, g1, numsets, type_e ));
    NVGRAPH_SAFE_CALL(nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0));

    int weight_index = 0;
    int source_vert = param.source_vert;
    int sssp_index = 0;

    // run
    std::cout << "Running algorithm ..." << std::endl;
    double start, stop;
    start = second();
    start = second();
    int repeat = std::max(param.repeats, 1);
    for (int i = 0; i < repeat; i++)
        NVGRAPH_SAFE_CALL(nvgraphSssp(handle, g1, weight_index, &source_vert, sssp_index));
    stop = second();
    printf("Time of single SSSP call is %10.8fsecs\n", (stop-start)/repeat);
    
    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));

    if (handle != NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphDestroy(handle));
        handle = NULL;
    }
}

typedef struct Traversal_Usecase_t
{
    std::string graph_file;
    int source_vert;
    int repeats;
    Traversal_Usecase_t(const std::string& a, int b, int c) : graph_file(a), source_vert(b), repeats(c){};
    Traversal_Usecase_t& operator=(const Traversal_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        source_vert = rhs.source_vert; 
        repeats = rhs.repeats;
        return *this;
    } 
} Traversal_Usecase;


template <typename T>
void run_traversal_bench(const Traversal_Usecase& param)
{
    std::cout << "Initializing nvGRAPH library..." << std::endl;  

    nvgraphHandle_t handle = NULL;

    if (handle == NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));
    }

    nvgraphTopologyType_t topo = NVGRAPH_CSR_32;

    std::cout << "Reading input data..." << std::endl; 

    FILE* fpin = fopen(param.graph_file.c_str(),"r");
    if (fpin == NULL)
    {
        std::cout << "Cannot read input graph file: " << param.graph_file << std::endl;  
        exit(1);
    } 


    //Read a transposed network in amgx binary format and the bookmark of dangling nodes
    /*
    if (read_header_amgx_csr_bin (fpin, n, nnz) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
   if (read_data_amgx_csr_bin (fpin, n, nnz, read_row_ptr, read_col_ind, csr_read_val) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    fclose(fpin);
    */
    int m, n, nnz;
    MM_typecode mc;
 
    if(mm_properties<int>(fpin, 1, &mc, &m, &n, &nnz) != 0) {
	std::cout <<  "could not read Matrix Market file properties"<< "\n";
	exit(1);
    }

    std::vector<int> read_row_ptr(n+1), read_col_ind(nnz), coo_row_ind(nnz);
    std::vector<T> csr_read_val(nnz);
        
       if(mm_to_coo<int,T>(fpin, 1, nnz, &coo_row_ind[0], &read_col_ind[0], &csr_read_val[0], NULL)) {
	std::cout << "could not read matrix data"<< "\n";
	exit(1);
    }

    if(coo_to_csr<int,T> (n, n, nnz, &coo_row_ind[0],  &read_col_ind[0], &csr_read_val[0], NULL, &read_row_ptr[0], NULL, NULL, NULL)) {
	std::cout << "could not covert COO to CSR "<< "\n";
	exit(1);
    }

      
    std::cout << "Initializing data structures ..." << std::endl;  

    nvgraphGraphDescr_t g1 = NULL;
    NVGRAPH_SAFE_CALL(nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphCSRTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
    NVGRAPH_SAFE_CALL(nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo));

    // set up graph data
    size_t numsets = 1;
    cudaDataType_t type_v[1] = {nvgraph_Const<int>::Type};

    NVGRAPH_SAFE_CALL(nvgraphAllocateVertexData(handle, g1, numsets, type_v));

    int source_vert = param.source_vert;
    nvgraphTraversalParameter_t traversal_param;
    nvgraphTraversalParameterInit(&traversal_param);
    nvgraphTraversalSetDistancesIndex(&traversal_param, 0);


    // run
    std::cout << "Running algorithm ..." << std::endl;
    double start, stop;
    start = second();
    start = second();
    int repeat = std::max(param.repeats, 1);
    for (int i = 0; i < repeat; i++)
        NVGRAPH_SAFE_CALL(nvgraphTraversal(handle, g1, NVGRAPH_TRAVERSAL_BFS, &source_vert, traversal_param));
    stop = second();
    printf("Time of single Traversal call is %10.8fsecs\n", (stop-start)/repeat);
    
    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));

    if (handle != NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphDestroy(handle));
        handle = NULL;
    }
}

typedef struct Pagerank_Usecase_t
{
    std::string graph_file;
    float alpha;
    int repeats;
    int max_iters;
    double tolerance;
    Pagerank_Usecase_t(const std::string& a, float b, const int c, const int d, const double e) : graph_file(a), alpha(b), repeats(c), max_iters(d), tolerance(e) {};
    Pagerank_Usecase_t& operator=(const Pagerank_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        alpha = rhs.alpha; 
        repeats = rhs.repeats;
        max_iters = rhs.max_iters;
        tolerance = rhs.tolerance;
        return *this;  
    } 
} Pagerank_Usecase;

template <typename T>
void run_pagerank_bench(const Pagerank_Usecase& param)
{
    std::cout << "Initializing nvGRAPH library..." << std::endl;  
    nvgraphHandle_t handle = NULL;

    if (handle == NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));
    }

    nvgraphTopologyType_t topo = NVGRAPH_CSC_32;

    std::cout << "Reading input data..." << std::endl;  

    FILE* fpin = fopen(param.graph_file.c_str(),"r");
    if (fpin == NULL)
    {
        std::cout << "Cannot open input graph file: " << param.graph_file << std::endl;  
        exit(1);
    } 
    int n, nnz;
    //Read a transposed network in amgx binary format and the bookmark of dangling nodes
    if (read_header_amgx_csr_bin (fpin, n, nnz) != 0)
    {
        std::cout << "Cannot read input graph file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    std::vector<int> read_row_ptr(n+1), read_col_ind(nnz);
    std::vector<T> read_val(nnz);
    std::vector<T> dangling(n);
    if (read_data_amgx_csr_bin_rhs (fpin, n, nnz, read_row_ptr, read_col_ind, read_val, dangling) != 0)
    {
        std::cout << "Cannot read input graph file: " << param.graph_file << std::endl;  
        exit(1);
    }
    fclose(fpin);

    std::cout << "Initializing data structures ..." << std::endl;  

    nvgraphGraphDescr_t g1 = NULL;
    NVGRAPH_SAFE_CALL(nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphCSCTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
    NVGRAPH_SAFE_CALL(nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo));

    // set up graph data
    std::vector<T> calculated_res(n, (T)1.0/n);
    void*  vertexptr[2] = {(void*)&dangling[0], (void*)&calculated_res[0]};
    cudaDataType_t type_v[2] = {nvgraph_Const<T>::Type, nvgraph_Const<T>::Type};
    
    void*  edgeptr[1] = {(void*)&read_val[0]};
    cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

    NVGRAPH_SAFE_CALL(nvgraphAllocateVertexData(handle, g1, 2, type_v));
    NVGRAPH_SAFE_CALL(nvgraphSetVertexData(handle, g1, vertexptr[0], 0 ));
    NVGRAPH_SAFE_CALL(nvgraphSetVertexData(handle, g1, vertexptr[1], 1 ));
    NVGRAPH_SAFE_CALL(nvgraphAllocateEdgeData(handle, g1, 1, type_e ));
    NVGRAPH_SAFE_CALL(nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 ));

    int bookmark_index = 0;
    int weight_index = 0;
    T alpha = param.alpha;
    int pagerank_index = 1;
    int has_guess = 0;
    float tolerance = (T)param.tolerance;
    int max_iter = param.max_iters;

    std::cout << "Running algorithm ..." << std::endl;  
    // run
    double start, stop;
    start = second();
    start = second();
    int repeat = std::max(param.repeats, 1);
    for (int i = 0; i < repeat; i++)
        NVGRAPH_SAFE_CALL(nvgraphPagerank(handle, g1, weight_index, (void*)&alpha, bookmark_index, has_guess, pagerank_index, tolerance, max_iter));
    stop = second();
    printf("Time of single Pargerank call is %10.8fsecs\n", (stop-start)/repeat);
    
    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));

    if (handle != NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphDestroy(handle));
        handle = NULL;
    }
}

typedef struct ModMax_Usecase_t
{
    std::string graph_file;
    int clusters;
    int evals;
    int repeats;
    ModMax_Usecase_t(const std::string& a, int b, int c, int d) : graph_file(a), clusters(b), evals(c), repeats(d){};
    ModMax_Usecase_t& operator=(const ModMax_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        clusters = rhs.clusters;
        evals = rhs.evals;
        repeats = rhs.repeats;
        return *this;
    }
} ModMax_Usecase;

template <typename T> 
void run_modularity_bench(const ModMax_Usecase& param)
{
     // this function prints :
     // #clusters,time in ms,modularity

     nvgraphHandle_t handle = NULL;
     NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));

    int m, n, nnz;
    MM_typecode mc;

    FILE* fpin = fopen(param.graph_file.c_str(),"r");
    
    mm_properties<int>(fpin, 1, &mc, &m, &n, &nnz) ;

    // Allocate memory on host
    std::vector<int> cooRowIndA(nnz);
    std::vector<int> csrColIndA(nnz);
    std::vector<int> csrRowPtrA(n+1);
    std::vector<T> csrValA(nnz);

    mm_to_coo<int,T>(fpin, 1, nnz, &cooRowIndA[0], &csrColIndA[0], &csrValA[0],NULL) ;
    coo_to_csr<int,T> (n, n, nnz, &cooRowIndA[0],  &csrColIndA[0], &csrValA[0], NULL, &csrRowPtrA[0], NULL, NULL, NULL);
    fclose(fpin);        

     //remove diagonal
     for (int i = 0; i < n; i++)
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            if (csrColIndA[j]==i)
                csrValA[j] = 0.0;

     nvgraphGraphDescr_t g1 = NULL;

     struct SpectralClusteringParameter clustering_params;
     clustering_params.n_clusters = param.clusters; 
     clustering_params.n_eig_vects = param.evals; 
     clustering_params.algorithm = NVGRAPH_MODULARITY_MAXIMIZATION; 
     clustering_params.evs_tolerance = 0.0f ;
     clustering_params.evs_max_iter = 0;
     clustering_params.kmean_tolerance = 0.0f; 
     clustering_params.kmean_max_iter = 0;

    int weight_index = 0; 
   
    //std::vector<T> clustering_h(n);
    //std::vector<T> eigVals_h(clustering_params.n_clusters);
    //std::vector<T> eigVecs_h(n*clustering_params.n_clusters);

    //could also be on device
    int *clustering_d; cudaMalloc((void**)&clustering_d , n*sizeof(int));
    T* eigVals_d; cudaMalloc((void**)&eigVals_d, clustering_params.n_clusters*sizeof(T));
    T* eigVecs_d; cudaMalloc((void**)&eigVecs_d, n*clustering_params.n_clusters*sizeof(T));
    
    NVGRAPH_SAFE_CALL( nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphCSRTopology32I_st topology = {n, nnz, &csrRowPtrA[0], &csrColIndA[0]};
    NVGRAPH_SAFE_CALL( nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32));

    // set up graph data
    size_t numsets = 1;
    void*  edgeptr[1] = {(void*)&csrValA[0]};
    cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
    NVGRAPH_SAFE_CALL( nvgraphAllocateEdgeData(handle, g1, numsets, type_e ));
    NVGRAPH_SAFE_CALL( nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 ));   
    
    printf("%d,", clustering_params.n_clusters);

    double start, stop;
    start = second();
    int repeat = std::max(param.repeats, 1);
    for (int i = 0; i < repeat; i++)
     // NVGRAPH_SAFE_CALL(nvgraphSpectralClustering(handle, g1, weight_index, &clustering_params, (int*)&clustering_h[0], (void*)&eigVals_h[0], (void*)&eigVecs_h[0])); 
      NVGRAPH_SAFE_CALL(nvgraphSpectralClustering(handle, g1, weight_index, &clustering_params, clustering_d, eigVals_d, eigVecs_d));
    //for (int i = 0; i < repeat; i++)
       // NVGRAPH_SAFE_CALL( nvgraphSpectralModularityMaximization(handle, g1, weight_index, clustering_params.n_clusters, clustering_params.n_eig_vects, 0.0f, 0, 0.0f, 0, clustering_d, (void*)&eigVals_h[0], (void*)&eigVecs_h[0])); 
    //for (int i = 0; i < repeat; i++)
        // NVGRAPH_SAFE_CALL( nvgraphBalancedCutClustering(handle, g1, weight_index, clustering_params.n_clusters, clustering_params.n_eig_vects, 0, 0.0f, 0, 0.0f, 0, clustering_d, (void*)&eigVals_h[0], (void*)&eigVecs_h[0])); 
    stop = second();
    printf("%10.8f,", 1000.0*(stop-start)/repeat);

    //Print
    //std::vector<int> clust_h(n);
    //cudaMemcpy(&clust_h[0], clustering_d,n*sizeof(int),cudaMemcpyDeviceToHost);
    //printf("\n ");
    //for (int i = 0; i < n; ++i)
    //   printf("%d ", clust_h [i]);
    //printf("\n ");
    //for (int i = 0; i < clustering_params.n_clusters; ++i)
    //    std::cout << eigVals_h[i]<< ' ' ;
    //printf("\n ");
    //std::cout<< std::endl;
    //std::cout << std::endl;
    //for (int i = 0; i < clustering_params.n_clusters; ++i)
    //{
    //    for (int j = 0; j < 10; ++j)
    //        std::cout << eigVecs_h[i*n+j] << ' '; 
    //    std::cout<< std::endl;
    //}

    // Analyse quality
    float score =0.0;
    nvgraphAnalyzeClustering(handle, g1, weight_index, clustering_params.n_clusters, clustering_d, NVGRAPH_MODULARITY, &score);  
    printf("%f\n", score);

    // ratio cut
    // float ec =0.0, rc =0.0;
    // NVGRAPH_SAFE_CALL(nvgraphAnalyzeBalancedCut(handle, g1, weight_index, clustering_params.n_clusters, clustering_d, &ec, &rc));  
    // printf("%f,", rc);
    
    // // Synthetic random 
    // for (int i=0; i<n; i++)
    // {
    //     parts_h[i] = rand() % clustering_params.n_clusters;
    //     //printf("%d ", parts_h[i]);
    // }
    // // Analyse quality
    // cudaMemcpy(clustering_d,&parts_h[0],n*sizeof(int),cudaMemcpyHostToDevice);
    // //NVGRAPH_SAFE_CALL( nvgraphAnalyzeModularityClustering(handle, g1, weight_index, clustering_params.n_clusters, clustering_d, &modularity1));  
    // //printf("%f\n", modularity1);
    // NVGRAPH_SAFE_CALL(nvgraphAnalyzeBalancedCut(handle, g1, weight_index, clustering_params.n_clusters, clustering_d, &ec, &rc));  
    // printf("%f\n", rc);

    //exit
    cudaFree(clustering_d);
    cudaFree(eigVals_d);
    cudaFree(eigVecs_d);
    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));
}

typedef struct BalancedCut_Usecase_t
{
    std::string graph_file;
    int clusters;
    int evals;
    int repeats;
    BalancedCut_Usecase_t(const std::string& a, int b, int c, int d) : graph_file(a), clusters(b), evals(c), repeats(d){};
    BalancedCut_Usecase_t& operator=(const BalancedCut_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        clusters = rhs.clusters;
        evals = rhs.evals;
        repeats = rhs.repeats;
        return *this;
    }
} BalancedCut_Usecase;

template <typename T> 
void run_balancedCut_bench(const BalancedCut_Usecase& param)
{
     // this function prints :
     // #clusters,time in ms,rc

     nvgraphHandle_t handle = NULL;
     NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));

    int m, n, nnz;
    MM_typecode mc;

    FILE* fpin = fopen(param.graph_file.c_str(),"r");
    
    mm_properties<int>(fpin, 1, &mc, &m, &n, &nnz) ;

    // Allocate memory on host
    std::vector<int> cooRowIndA(nnz);
    std::vector<int> csrColIndA(nnz);
    std::vector<int> csrRowPtrA(n+1);
    std::vector<T> csrValA(nnz);

    mm_to_coo<int,T>(fpin, 1, nnz, &cooRowIndA[0], &csrColIndA[0], &csrValA[0],NULL) ;
    coo_to_csr<int,T> (n, n, nnz, &cooRowIndA[0],  &csrColIndA[0], &csrValA[0], NULL, &csrRowPtrA[0], NULL, NULL, NULL);
    fclose(fpin);        

     //remove diagonal
     for (int i = 0; i < n; i++)
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            if (csrColIndA[j]==i)
                csrValA[j] = 0.0;

     nvgraphGraphDescr_t g1 = NULL;

     struct SpectralClusteringParameter clustering_params;
     clustering_params.n_clusters = param.clusters; 
     clustering_params.n_eig_vects = param.evals; 
     clustering_params.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS; 
     clustering_params.evs_tolerance = 0.0f ;
     clustering_params.evs_max_iter = 0;
     clustering_params.kmean_tolerance = 0.0f; 
     clustering_params.kmean_max_iter = 0;

    int weight_index = 0; 
   
    //std::vector<T> clustering_h(n);
    //std::vector<T> eigVals_h(clustering_params.n_clusters);
    //std::vector<T> eigVecs_h(n*clustering_params.n_clusters);

    //could also be on device
    int *clustering_d; cudaMalloc((void**)&clustering_d , n*sizeof(int));
    T* eigVals_d; cudaMalloc((void**)&eigVals_d, clustering_params.n_clusters*sizeof(T));
    T* eigVecs_d; cudaMalloc((void**)&eigVecs_d, n*clustering_params.n_clusters*sizeof(T));
    
    NVGRAPH_SAFE_CALL( nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphCSRTopology32I_st topology = {n, nnz, &csrRowPtrA[0], &csrColIndA[0]};
    NVGRAPH_SAFE_CALL( nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32));

    // set up graph data
    size_t numsets = 1;
    void*  edgeptr[1] = {(void*)&csrValA[0]};
    cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};
    NVGRAPH_SAFE_CALL( nvgraphAllocateEdgeData(handle, g1, numsets, type_e ));
    NVGRAPH_SAFE_CALL( nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 ));   
    
    printf("%d,", clustering_params.n_clusters);

    double start, stop;
    start = second();
    int repeat = std::max(param.repeats, 1);
    for (int i = 0; i < repeat; i++)
     // NVGRAPH_SAFE_CALL(nvgraphSpectralClustering(handle, g1, weight_index, &clustering_params, (int*)&clustering_h[0], (void*)&eigVals_h[0], (void*)&eigVecs_h[0])); 
      NVGRAPH_SAFE_CALL(nvgraphSpectralClustering(handle, g1, weight_index, &clustering_params, clustering_d, eigVals_d, eigVecs_d));
    stop = second();
    printf("%10.8f,", 1000.0*(stop-start)/repeat);

    // Analyse quality
    float score =0.0;
    nvgraphAnalyzeClustering(handle, g1, weight_index, clustering_params.n_clusters, clustering_d, NVGRAPH_RATIO_CUT, &score);  
    printf("%f\n", score);

    //exit
    cudaFree(clustering_d);
    cudaFree(eigVals_d);
    cudaFree(eigVecs_d);

    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));
}

typedef struct TriCount_Usecase_t
{
    std::string graph_file;
    int repeats;
    TriCount_Usecase_t(const std::string& a, const int b) : graph_file(a), repeats(b){};
    TriCount_Usecase_t& operator=(const TriCount_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        repeats = rhs.repeats;
        return *this;
    }
} TriCount_Usecase;

void run_tricount_bench(const TriCount_Usecase& param)
{
    std::cout << "Initializing nvGRAPH library..." << std::endl; 

    nvgraphHandle_t handle = NULL;

    if (handle == NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphCreate(&handle));
    }

    nvgraphTopologyType_t topo = NVGRAPH_CSR_32;

    std::cout << "Reading input data..." << std::endl;  

    FILE* fpin = fopen(param.graph_file.c_str(),"rb");
    if (fpin == NULL)
    {
        std::cout << "Cannot open input graph file: " << param.graph_file << std::endl;  
        exit(1);
    } 

    int n, nnz;
    std::vector<int> read_row_ptr, read_col_ind;
    //Read CSR of lower triangular of undirected graph
    if (read_csr_bin<int> (fpin, n, nnz, read_row_ptr, read_col_ind) != 0)
    {
        std::cout << "Error reading input file: " << param.graph_file << std::endl;  
        exit(1);  
    }
    fclose(fpin);

    std::cout << "Initializing data structures ..." << std::endl;  

    nvgraphGraphDescr_t g1 = NULL;
    NVGRAPH_SAFE_CALL(nvgraphCreateGraphDescr(handle, &g1));  

    // set up graph
    nvgraphCSRTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0]};
    NVGRAPH_SAFE_CALL(nvgraphSetGraphStructure(handle, g1, (void*)&topology, topo));

    // set up graph data
    uint64_t res = 0;
    // run
    std::cout << "Running algorithm..." << std::endl;
    double start, stop;
    start = second();
    start = second();
    int repeat = std::max(param.repeats, 1);
    for (int i = 0; i < repeat; i++)
        NVGRAPH_SAFE_CALL(nvgraphTriangleCount(handle, g1, &res));
    stop = second();
    printf("Number of triangles counted: %lli\n", (long long int)res);
    printf("Time of single TriangleCount call is %10.8fsecs\n", (stop-start)/repeat);
    
    NVGRAPH_SAFE_CALL(nvgraphDestroyGraphDescr(handle, g1));

    if (handle != NULL) 
    {
        NVGRAPH_SAFE_CALL(nvgraphDestroy(handle));
        handle = NULL;
    }
}


int findParamIndex(const char** argv, int argc, const char* parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++) 
    {
        if (strncmp(argv[i], parm, 100)==0)
        {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1) 
    {
        return index;
    }
    else 
    {
        printf("Error, parameter %s has been specified more than once, exiting\n",parm);
        exit(1);
    }

    return -1;
}

int main(int argc, const char **argv) 
{
    int pidx = 0;
    int repeats = 100;

    if (argc < 2 || findParamIndex(argv, argc, "--help") != -1)
    {
        printf("Usage:                                                                                                  \n");
        printf("    nvgraph_benchmark [--double|--float] [--repeats N] --spmv     graph_file                            \n");
        printf("    nvgraph_benchmark [--double|--float] [--repeats N] --widest   graph_file start_vertex               \n");
        printf("    nvgraph_benchmark [--double|--float] [--repeats N] --sssp     graph_file start_vertex               \n");
        printf("    nvgraph_benchmark [--double|--float] [--repeats N] --pagerank graph_file alpha max_iters tolerance  \n");
        printf("    nvgraph_benchmark [--double|--float] [--repeats N] --modularity  graph_file nb_clusters nb_eigvals  \n");
        printf("    nvgraph_benchmark [--double|--float] [--repeats N] --traversal   graph_file start_vertex            \n");
        printf("    nvgraph_benchmark [--double|--float] [--repeats N] --balancedCut  graph_file nb_clusters nb_eigvals \n");
        printf("    nvgraph_benchmark                    [--repeats N] --tricount   graph_file                          \n");
        exit(0);
    }

    if ( (pidx = findParamIndex(argv, argc, "--repeats")) != -1)
    {
        repeats = atoi(argv[pidx+1]);
    }

    if (findParamIndex(argv, argc, "--double") != -1 || findParamIndex(argv, argc, "--float") == -1)
    {
        if ((pidx = findParamIndex(argv, argc, "--widest")) != -1)
        {
            run_widest_bench<double>(WidestPath_Usecase(argv[pidx+1], atoi(argv[pidx+2]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--spmv")) != -1)
        {
            run_srspmv_bench<double>(SrSPMV_Usecase(argv[pidx+1], repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--sssp")) != -1)
        {
            run_sssp_bench<double>(SSSP_Usecase(argv[pidx+1], atoi(argv[pidx+2]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--pagerank")) != -1)
        {
            run_pagerank_bench<double>(Pagerank_Usecase(argv[pidx+1], atof(argv[pidx+2]), repeats, atoi(argv[pidx+3]), atof(argv[pidx+4])));
        }
         else if ((pidx = findParamIndex(argv, argc, "--modularity")) != -1)
        {
            run_modularity_bench<double>(ModMax_Usecase(argv[pidx+1], atoi(argv[pidx+2]), atoi(argv[pidx+3]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--traversal")) != -1)
        {
            run_traversal_bench<double>(Traversal_Usecase(argv[pidx+1], atoi(argv[pidx+2]), repeats));
        } 
        else if ((pidx = findParamIndex(argv, argc, "--balancedCut")) != -1)
        {
            run_balancedCut_bench<double>(BalancedCut_Usecase(argv[pidx+1], atoi(argv[pidx+2]), atoi(argv[pidx+3]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--tricount")) != -1)
        {
            run_tricount_bench(TriCount_Usecase(argv[pidx+1], repeats));
        }
	else
        {
            printf("Specify one of the algorithms: '--widest', '--sssp', '--pagerank', '--modularity', '--balancedCut', '--traversal', or 'tricount'\n");
        }
    }
    else
    {
        if ((pidx = findParamIndex(argv, argc, "--widest")) != -1)
        {
            run_widest_bench<float>(WidestPath_Usecase(argv[pidx+1], atoi(argv[pidx+2]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--spmv")) != -1)
        {
            run_srspmv_bench<float>(SrSPMV_Usecase(argv[pidx+1], repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--sssp")) != -1)
        {
            run_sssp_bench<float>(SSSP_Usecase(argv[pidx+1], atoi(argv[pidx+2]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--pagerank")) != -1)
        {
            run_pagerank_bench<float>(Pagerank_Usecase(argv[pidx+1], atof(argv[pidx+2]), repeats, atoi(argv[pidx+3]), atof(argv[pidx+4])));
        }
        else if ((pidx = findParamIndex(argv, argc, "--modularity")) != -1)
        {
            run_modularity_bench<float>(ModMax_Usecase(argv[pidx+1], atoi(argv[pidx+2]), atoi(argv[pidx+3]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--traversal")) != -1)
        {
            run_traversal_bench<float>(Traversal_Usecase(argv[pidx+1], atoi(argv[pidx+2]), repeats));
        }
        else if ((pidx = findParamIndex(argv, argc, "--balancedCut")) != -1)
        {
            run_balancedCut_bench<float>(BalancedCut_Usecase(argv[pidx+1], atoi(argv[pidx+2]), atoi(argv[pidx+3]), repeats));
        }
	else
        {
            printf("Specify one of the algorithms: '--widest', '--sssp' , '--pagerank', '--modularity', '--balancedCut' or '--traversal'\n");
        }
    }

    return 0;
}

