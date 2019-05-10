#include <utility>
#include "gtest/gtest.h"
#include "nvgraph_test_common.h"
#include "valued_csr_graph.hxx"
#include "readMatrix.hxx"
#include "nvgraphP.h"
#include "nvgraph.h"
#include "nvgraph_experimental.h"
#include "stdlib.h"
#include <algorithm>
extern "C" {
#include "mmio.h"
}
#include "mm.hxx"

// do the perf measurements, enabled by command line parameter '--perf'
static int PERF = 0;

// minimum vertices in the graph to perform perf measurements
#define PERF_ROWS_LIMIT 1000

// number of repeats = multiplier/num_vertices
#define PARTITIONER_ITER_MULTIPLIER 1
#define SELECTOR_ITER_MULTIPLIER 1

// iterations for stress tests = this multiplier * iterations for perf tests
static int STRESS_MULTIPLIER = 10;

static std::string ref_data_prefix = "";
static std::string graph_data_prefix = "";

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


/****************************
* SPECTRAL CLUSTERING
*****************************/

typedef struct SpectralClustering_Usecase_t
{
    std::string graph_file;
    int clusters;
    int eigenvalues;
    nvgraphSpectralClusteringType_t algorithm;
    nvgraphClusteringMetric_t metric;
    SpectralClustering_Usecase_t(const std::string& a, int b, int c, nvgraphSpectralClusteringType_t d, nvgraphClusteringMetric_t e) : clusters(b), eigenvalues(c), algorithm(d), metric(e){ graph_file = convert_to_local_path(a);};
    SpectralClustering_Usecase_t& operator=(const SpectralClustering_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        clusters = rhs.clusters;
        eigenvalues = rhs.eigenvalues;
        algorithm = rhs.algorithm;
        metric = rhs.metric;
        return *this;
    }
} SpectralClustering_Usecase;


class NVGraphCAPITests_SpectralClustering : public ::testing::TestWithParam<SpectralClustering_Usecase> {
  public:
    NVGraphCAPITests_SpectralClustering() : handle(NULL) {}

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
    void run_current_test(const  SpectralClustering_Usecase& param)
    {
        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss; 
        ss << param.clusters;
        ss << param.eigenvalues;
        ss << param.algorithm;
        ss << param.metric;
        std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file) + std::string("_") + ss.str().c_str();

        nvgraphStatus_t status;
        int m, n, nnz;
        MM_typecode mc;

        FILE* fpin = fopen(param.graph_file.c_str(),"r");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        
        ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &n, &nnz),0) << "could not read Matrix Market file properties"<< "\n";

        ASSERT_TRUE(mm_is_matrix(mc));
        ASSERT_TRUE(mm_is_coordinate(mc));
        ASSERT_TRUE(m==n);
        ASSERT_FALSE(mm_is_complex(mc));
        ASSERT_FALSE(mm_is_skew(mc));

        // Allocate memory on host
        std::vector<int> cooRowIndA(nnz);
        std::vector<int> csrColIndA(nnz);
        std::vector<int> csrRowPtrA(n+1);
        std::vector<T> csrValA(nnz);

        ASSERT_EQ( (mm_to_coo<int,T>(fpin, 1, nnz, &cooRowIndA[0], &csrColIndA[0], &csrValA[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
        ASSERT_EQ( (coo_to_csr<int,T> (n, n, nnz, &cooRowIndA[0],  &csrColIndA[0], &csrValA[0], NULL, &csrRowPtrA[0], NULL, NULL, NULL)), 0) << "could not covert COO to CSR "<< "\n";

        ASSERT_EQ(fclose(fpin),0);
        //ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        
        int *clustering_d;

         
        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(csrRowPtrA.size() + csrColIndA.size())))
        {
            std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            return;
        }

        cudaMalloc((void**)&clustering_d , n*sizeof(int));

        nvgraphGraphDescr_t g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        nvgraphCSRTopology32I_st topology = {n, nnz, &csrRowPtrA[0], &csrColIndA[0]};
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32);

        // set up graph data
        size_t numsets = 1;
        
        void*  edgeptr[1] = {(void*)&csrValA[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        status = nvgraphAllocateEdgeData(handle, g1, numsets, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int weight_index = 0;
        struct SpectralClusteringParameter clustering_params;
        clustering_params.n_clusters = param.clusters; 
        clustering_params.n_eig_vects = param.eigenvalues; 
        clustering_params.algorithm = param.algorithm;
        clustering_params.evs_tolerance = 0.0f ;
        clustering_params.evs_max_iter = 0;
        clustering_params.kmean_tolerance = 0.0f; 
        clustering_params.kmean_max_iter = 0;

        std::vector<int> random_assignments_h(n);
        std::vector<T> eigVals_h(param.eigenvalues);
        std::vector<T> eigVecs_h(n*param.eigenvalues);
        float score = 0.0, random_score = 0.0;     

        if (PERF && n > PERF_ROWS_LIMIT)
        {
            double start, stop;
            start = second();
            int repeat = std::max((int)((float)(PARTITIONER_ITER_MULTIPLIER)/n), 1);
            for (int i = 0; i < repeat; i++)
                status =nvgraphSpectralClustering(handle, g1, weight_index, &clustering_params, clustering_d, &eigVals_h[0], &eigVecs_h[0]);
            stop = second();
            printf("&&&& PERF Time_%s %10.8f -ms\n", test_id.c_str(), 1000.0*(stop-start)/repeat);
        }
        else
           status =nvgraphSpectralClustering(handle, g1, weight_index, &clustering_params, clustering_d, &eigVals_h[0], &eigVecs_h[0]);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
       
        // Analyse quality
        status = nvgraphAnalyzeClustering(handle, g1, weight_index,  param.clusters, clustering_d,  param.metric, &score);  

        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        //printf("Score = %f\n", score);

        // ===
        // Synthetic random 
        
        for (int i=0; i<n; i++)
        {
            random_assignments_h[i] = rand() % param.clusters;
            //printf("%d ", random_assignments_h[i]);
        }

        status = nvgraphAnalyzeClustering(handle, g1, weight_index,  param.clusters, &random_assignments_h[0],  param.metric, &random_score);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        //printf("Random modularity = %f\n", modularity2);
        if (param.metric == NVGRAPH_MODULARITY)
            EXPECT_GE(score, random_score); // we want higher modularity
        else
             EXPECT_GE(random_score, score); //we want less edge cut

        cudaFree(clustering_d);
        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
};
 
TEST_P(NVGraphCAPITests_SpectralClustering, CheckResultDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_SpectralClustering, CheckResultFloat)
{
    run_current_test<float>(GetParam());
}

// --gtest_filter=*ModularityCorrectness*
INSTANTIATE_TEST_CASE_P(SpectralModularityCorrectnessCheck,
                       NVGraphCAPITests_SpectralClustering,
                          //                                  graph FILE        number of clusters #     number of eigenvalues #     
                       ::testing::Values(    
                                            SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 2, 2, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 3, 3, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/uk.mtx", 2, 2, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/uk.mtx", 3, 3, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/data.mtx", 3, 3, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/data.mtx", 5, 5, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/data.mtx", 7, 7, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/cti.mtx", 3, 3,NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/cti.mtx", 5, 5,NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                            SpectralClustering_Usecase("dimacs10/cti.mtx", 7, 7,NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY)
                                           ///// more instances
                                         )
                       );

// --gtest_filter=*ModularityCorner*
INSTANTIATE_TEST_CASE_P(SpectralModularityCornerCheck,
                       NVGraphCAPITests_SpectralClustering,
                          //                                  graph FILE        number of clusters #     number of eigenvalues #     
                       ::testing::Values(  
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_MODULARITY),
                                           SpectralClustering_Usecase("dimacs10/uk.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_MODULARITY),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n12.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_MODULARITY),
                                           SpectralClustering_Usecase("dimacs10/data.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_MODULARITY),
                                           SpectralClustering_Usecase("dimacs10/cti.mtx", 7, 4,NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_MODULARITY),  
                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 7, 4, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 17, 7, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY)
                                           // tests cases on coAuthorsDBLP may diverge on some cards (likely due to different normalization operation)
                                           //SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 7, 4,NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
                                           //SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 17, 7,NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY)
                                           ///// more instances
                                         )
                       );
// --gtest_filter=*LanczosBlancedCutCorrectness*
INSTANTIATE_TEST_CASE_P(SpectralLanczosBlancedCutCorrectnessCheck,
                       NVGraphCAPITests_SpectralClustering,
                          //                                  graph FILE       number of clusters #     number of eigenvalues #   
                       ::testing::Values( 
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 2, 2,NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 4, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/uk.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n12.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/data.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/cti.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/cti.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT)
                                           ///// more instances
                                         )
                       );
// --gtest_filter=*LanczosBlancedCutCorner*
INSTANTIATE_TEST_CASE_P(SpectralLanczosBlancedCutCornerCheck,
                       NVGraphCAPITests_SpectralClustering,
                          //                                  graph FILE        number of clusters #     number of eigenvalues #     
                       ::testing::Values(    
                                           
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/uk.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n12.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/data.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/cti.mtx", 7, 4,NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 17, 7, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT)
                                           // tests cases on coAuthorsDBLP may diverge on some cards (likely due to different normalization operation)
                                           //SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 7, 4,NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT),
                                           //SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 17, 7,NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_EDGE_CUT)                       
                                         )
                       );

// --gtest_filter=*LobpcgBlancedCutCorrectness*
INSTANTIATE_TEST_CASE_P(SpectralLobpcgBlancedCutCorrectnessCheck,
                       NVGraphCAPITests_SpectralClustering,
                          //                                  graph FILE       number of clusters #     number of eigenvalues #   
                       ::testing::Values(   
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 2, 2,NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 4, 4, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/uk.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n12.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/cti.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
                                           SpectralClustering_Usecase("dimacs10/cti.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT)
                                           ///// more instances
                                         )
                       );
// --gtest_filter=*LobpcgBlancedCutCorner*
INSTANTIATE_TEST_CASE_P(SpectralLobpcgBlancedCutCornerCheck,
                       NVGraphCAPITests_SpectralClustering,
                          //                                  graph FILE        number of clusters #     number of eigenvalues #     
                       ::testing::Values(                                              
                                           SpectralClustering_Usecase("dimacs10/delaunay_n10.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/uk.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/delaunay_n12.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/data.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/cti.mtx", 7, 4,NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 7, 4, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT),
                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 17, 7, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT)
                                           // tests cases on coAuthorsDBLP may diverge on some cards (likely due to different normalization operation)
                                           //SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 7, 4,NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT),
                                           //SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 17, 7,NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_EDGE_CUT)
                                           ///// more instances
                                         )
                       );
//Followinf tests were commented becasue they are a bit redundent and quite long to run
// previous tests already contain dataset with 1 million edges

//// --gtest_filter=*ModularityLargeCorrectness*
//INSTANTIATE_TEST_CASE_P(SpectralModularityLargeCorrectnessCheck,
//                       NVGraphCAPITests_SpectralClustering,
//                          //                                  graph FILE        number of clusters #     number of eigenvalues #     
//                       ::testing::Values(    
//                                            SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 2, 2, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
//                                            SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 3, 3, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
//                                            SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 5, 5, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
//                                            SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 7, 7, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
//                                            SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 2, 2, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
//                                            SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 3, 3, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
//                                            SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 5, 5, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY),
//                                            SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 7, 7, NVGRAPH_MODULARITY_MAXIMIZATION, NVGRAPH_MODULARITY)
//                                          ///// more instances
//                                         )
//                       );
//
//// --gtest_filter=*LanczosBlancedCutLargeCorrectness*
//INSTANTIATE_TEST_CASE_P(SpectralLanczosBlancedCutLargeCorrectnessCheck,
//                       NVGraphCAPITests_SpectralClustering,
//                          //                                  graph FILE       number of clusters #     number of eigenvalues #   
//                       ::testing::Values(    
//                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 2, 2, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 5, 5, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LANCZOS, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 2, 2, NVGRAPH_BALANCED_CUT_LANCZOS,NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LANCZOS,NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 5, 5, NVGRAPH_BALANCED_CUT_LANCZOS,NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LANCZOS,NVGRAPH_RATIO_CUT)
//                                         )
//                       );
//// --gtest_filter=*LobpcgBlancedCutLargeCorrectness*
//INSTANTIATE_TEST_CASE_P(SpectralLobpcgBlancedCutLargeCorrectnessCheck,
//                       NVGraphCAPITests_SpectralClustering,
//                          //                                  graph FILE       number of clusters #     number of eigenvalues #   
//                       ::testing::Values(    
//                                           //SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 2, 2, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 5, 5, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/citationCiteseer.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 2, 2, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 3, 3, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 5, 5, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT),
//                                           SpectralClustering_Usecase("dimacs10/coAuthorsDBLP.mtx", 7, 7, NVGRAPH_BALANCED_CUT_LOBPCG, NVGRAPH_RATIO_CUT)
//                                         )
//                       );
/****************************
* SELECTOR
*****************************/

typedef struct Selector_Usecase_t
{
    std::string graph_file;
    nvgraphEdgeWeightMatching_t metric;
    Selector_Usecase_t(const std::string& a, nvgraphEdgeWeightMatching_t b) : metric(b){ graph_file = convert_to_local_path(a);};
    Selector_Usecase_t& operator=(const Selector_Usecase_t& rhs)
    {
        graph_file = rhs.graph_file;
        metric = rhs.metric;
        return *this;
    }
}Selector_Usecase;

class NVGraphCAPITests_Selector : public ::testing::TestWithParam<Selector_Usecase> {
  public:
    NVGraphCAPITests_Selector() : handle(NULL) {}

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
    void run_current_test(const Selector_Usecase& param)
    {
        const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss; 
        ss << param.metric;
        std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file)+ std::string("_") + ss.str().c_str();

        nvgraphStatus_t status;
        int m, n, nnz;
        MM_typecode mc;

        FILE* fpin = fopen(param.graph_file.c_str(),"r");
        ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
        
        ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &n, &nnz),0) << "could not read Matrix Market file properties"<< "\n";

        ASSERT_TRUE(mm_is_matrix(mc));
        ASSERT_TRUE(mm_is_coordinate(mc));
        ASSERT_TRUE(m==n);
        ASSERT_FALSE(mm_is_complex(mc));
        ASSERT_FALSE(mm_is_skew(mc));

        // Allocate memory on host
        std::vector<int> cooRowIndA(nnz);
        std::vector<int> csrColIndA(nnz);
        std::vector<int> csrRowPtrA(n+1);
        std::vector<T> csrValA(nnz);

        ASSERT_EQ( (mm_to_coo<int,T>(fpin, 1, nnz, &cooRowIndA[0], &csrColIndA[0], &csrValA[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
        ASSERT_EQ( (coo_to_csr<int,T> (n, n, nnz, &cooRowIndA[0],  &csrColIndA[0], &csrValA[0], NULL, &csrRowPtrA[0], NULL, NULL, NULL)), 0) << "could not covert COO to CSR "<< "\n";

        ASSERT_EQ(fclose(fpin),0);
        //ASSERT_TRUE(fpin != NULL) << "Cannot read input graph file: " << param.graph_file << std::endl;
                 
        if (!enough_device_memory<T>(n, nnz, sizeof(int)*(csrRowPtrA.size() + csrColIndA.size())))
        {
            std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            return;
        }
        //int *aggregates_d;
        //cudaMalloc((void**)&aggregates_d , n*sizeof(int));

        nvgraphGraphDescr_t g1 = NULL;
        status = nvgraphCreateGraphDescr(handle, &g1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph
        nvgraphCSRTopology32I_st topology = {n, nnz, &csrRowPtrA[0], &csrColIndA[0]};
        status = nvgraphSetGraphStructure(handle, g1, (void*)&topology, NVGRAPH_CSR_32);

        // set up graph data
        size_t numsets = 1;
        //void*  vertexptr[1] = {(void*)&calculated_res[0]};
        //cudaDataType_t type_v[1] = {nvgraph_Const<T>::Type};
        
        void*  edgeptr[1] = {(void*)&csrValA[0]};
        cudaDataType_t type_e[1] = {nvgraph_Const<T>::Type};

        //status = nvgraphAllocateVertexData(handle, g1, numsets, type_v);
        //ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        //status = nvgraphSetVertexData(handle, g1, vertexptr[0], 0, NVGRAPH_CSR_32 );
        //ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(handle, g1, numsets, type_e );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, g1, (void *)edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        int weight_index = 0;
        std::vector<int> aggregates_h(n);
        //std::vector<int> aggregates_global_h(n);
         size_t num_aggregates;
         size_t *num_aggregates_ptr = &num_aggregates;

        status =  nvgraphHeavyEdgeMatching(handle,  g1,  weight_index, param.metric, &aggregates_h[0],  num_aggregates_ptr);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        std::cout  << "n = " << n << ", num aggregates = " << num_aggregates << std::endl;
        
        if (param.metric == NVGRAPH_SCALED_BY_DIAGONAL)
            EXPECT_EQ(num_aggregates, static_cast<size_t>(166)); // comparing against amgx result on poisson2D.mtx
        else        
            EXPECT_LE(num_aggregates, static_cast<size_t>(n)); // just make sure the output make sense
        
        //for (int i=0; i<n; i++)
        //{
        //    printf("%d\n", aggregates_h[i]);
        //}

        status = nvgraphDestroyGraphDescr(handle, g1);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
};
 
TEST_P(NVGraphCAPITests_Selector, CheckResultDouble)
{
    run_current_test<double>(GetParam());
    
}

TEST_P(NVGraphCAPITests_Selector, CheckResultFloat)
{
    run_current_test<float>(GetParam());
}

// --gtest_filter=*Correctness*
INSTANTIATE_TEST_CASE_P(SmallCorrectnessCheck,
                       NVGraphCAPITests_Selector,
                          //                                  graph FILE     SIMILARITY_METRIC
                       ::testing::Values(    
                                            Selector_Usecase("Florida/poisson2D.mtx", NVGRAPH_SCALED_BY_DIAGONAL),
                                            Selector_Usecase("dimacs10/delaunay_n10.mtx", NVGRAPH_SCALED_BY_ROW_SUM),
                                            Selector_Usecase("dimacs10/delaunay_n10.mtx", NVGRAPH_UNSCALED),
                                            Selector_Usecase("dimacs10/uk.mtx", NVGRAPH_SCALED_BY_ROW_SUM),
                                            Selector_Usecase("dimacs10/uk.mtx", NVGRAPH_UNSCALED),
                                            Selector_Usecase("dimacs10/data.mtx", NVGRAPH_SCALED_BY_ROW_SUM),
                                            Selector_Usecase("dimacs10/data.mtx", NVGRAPH_UNSCALED),
                                            Selector_Usecase("dimacs10/cti.mtx", NVGRAPH_SCALED_BY_ROW_SUM),
                                            Selector_Usecase("dimacs10/cti.mtx", NVGRAPH_UNSCALED)
                                           ///// more instances
                                         )
                       );

int main(int argc, char **argv) 
{
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
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
        
  return RUN_ALL_TESTS();

    return 0;
}


