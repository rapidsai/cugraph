/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Grmat tests
// Author: Ramakrishna Prabhu ramakrishnap@nvidia.com

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"
#include <cugraph.h>
#include "test_utils.h"
#include <string.h>

#include <rmm_utils.h>

//#include "functions.h"
// do the perf measurements
// enabled by command line parameter s'--perf'
static int PERF = 0;

// iterations for perf tests
// enabled by command line parameter '--perf-iters"
static int PERF_MULTIPLIER = 5;

void dumy(void* in, void* out ) {

}


void get_array_of_strings (char** argv, char* args, int& argc)
{
    char* tmp = nullptr;
    tmp = strtok(args, " ");
    for (int i = 0; (tmp != nullptr); i++)
    {
        argv[i] = (char *)malloc (sizeof(char)*(strlen(tmp)+1));
        strcpy (argv[i], tmp);
        argc += 1;
        tmp = strtok(nullptr, " ");
    }
}

void release_array (int argc, char** argv)
{
    if (argv != nullptr)
    {
        for (int i = 0; i < argc; i++)
        {
           if (argv[i] != nullptr)
           {
               free (argv[i]);
           }
        }
    }
}

typedef struct Grmat_Usecase_t {
  std::string argv;
  Grmat_Usecase_t(){
  }
  Grmat_Usecase_t(std::string args){
      argv = args;
  }
  ~Grmat_Usecase_t(){
  } 
} Grmat_Usecase;

class Tests_Grmat : public ::testing::TestWithParam<Grmat_Usecase> {
  public:
  Tests_Grmat() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { 
    if (PERF) {
     for (unsigned int i = 0; i < grmat_time.size(); ++i) {
      std::cout <<  grmat_time[i]/PERF_MULTIPLIER << std::endl;
     }
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> grmat_time;   

  // Check the coulmns of src and destination after the graph has been formed

  template <typename T>
  void run_check_configuration (const Grmat_Usecase& param) { 
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
     gdf_column col_sources, col_destinations;

     
     gdf_dtype gdf_vertexId_type;

     if (sizeof (T) == 4)
         gdf_vertexId_type = GDF_INT32;
     else
         gdf_vertexId_type = GDF_INT64;

     col_sources.dtype = gdf_vertexId_type;
     col_sources.valid = nullptr;
     col_destinations.dtype = gdf_vertexId_type;
     col_destinations.valid = nullptr;
     col_sources.null_count = 0;
     col_destinations.null_count = 0;
     col_sources.null_count = 0;
     col_destinations.null_count = 0;

     int rmat_scale = 0, edge_factor = 0, undirected = false;
     char* argv[32] = {0};
     int argc = 0;
     std::string tmp_argv(param.argv.c_str());
     get_array_of_strings (argv, (char *)tmp_argv.c_str(), argc);
     rmat_scale = atoi(strrchr(argv[1], '=')+1);
     edge_factor = atoi(strrchr(argv[2], '=')+1);
     for (int i = 0; i < argc; i++)
     {
         if (strcmp(argv[i], "--rmat_undirected") == 0)
         {
             undirected = true;
             break;
         }
     }
     release_array(argc, argv);

     size_t vertices = 1 << rmat_scale;
     size_t edges = vertices * edge_factor * ((undirected == true)? 2 : 1);
     size_t vertices1 = 0, edges1 = 0;
     if ((vertices < 1000) || (edge_factor < 8))
     { 
         return;
     }

     size_t free_before, total_before;
     cudaMemGetInfo (&free_before, &total_before);

    cugraph::grmat_gen ((char *)param.argv.c_str(), vertices1, edges1, &col_sources, &col_destinations, nullptr);
     
     size_t free_after, total_after;
     cudaMemGetInfo (&free_after, &total_after);

     ASSERT_EQ((0.99*(1<<vertices) >= vertices1), 0);
     ASSERT_EQ((0.99*(1<<edges) >= edges1), 0);
     size_t memory_occupied_before = total_before - free_before;
     size_t memory_occupied_after = total_after - free_after;
     size_t expected_amount_of_memory = (edges1 * sizeof (T) * (2) ); // 2 - sources and destination
     
     if (expected_amount_of_memory < total_after)
     {
         ASSERT_EQ((expected_amount_of_memory <= (memory_occupied_after-memory_occupied_before)), 1);
     }

    cudaStream_t stream{nullptr};
    ALLOC_FREE_TRY(col_sources.data, stream);
    ALLOC_FREE_TRY(col_destinations.data, stream);

     //size_t free_release, total_release;
     //cudaMemGetInfo (&free_release, &total_release);
     //ASSERT_EQ(((total_release - free_release) < expected_amount_of_memory) ,1);
  }

  template <typename VertexId>
  void run_check_max(const Grmat_Usecase& param) {
    int rmat_scale = 0, edge_factor = 0, undirected = false;;
    char* argv[32] = {0};
    int argc = 0;
    std::string tmp_argv(param.argv.c_str());

    get_array_of_strings (argv, (char *)tmp_argv.c_str(), argc);
    
    rmat_scale = atoi(strrchr(argv[1], '=')+1);
    edge_factor = atoi(strrchr(argv[2], '=')+1);
    for (int i = 0; i < argc; i++)
    {
         if (strcmp(argv[i], "--rmat_undirected") == 0)
         {
             undirected = true;
             break;
         }
    }
    release_array(argc, argv);
    edge_factor = edge_factor * ((undirected == true)? 2 :1);
    size_t max_vertices = (1<<26);
    size_t max_size =  max_vertices * 23 * 4;
    size_t current_size = (sizeof(VertexId) * (1<<rmat_scale) * edge_factor);
    if (max_size < current_size)
    {
        return;
    }
    const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
    Graph_ptr G{new cugraph::Graph, Graph_deleter};
    gdf_column col_sources, col_destinations;

    gdf_dtype gdf_vertexId_type;

    if (sizeof (VertexId) == 4)
         gdf_vertexId_type = GDF_INT32;
     else
         gdf_vertexId_type = GDF_INT64;

    col_sources.dtype = gdf_vertexId_type;
    col_sources.valid = nullptr;
    col_destinations.dtype = gdf_vertexId_type;
    col_destinations.valid = nullptr;

    col_sources.null_count = 0;
    col_destinations.null_count = 0;

    size_t vertices = 0, edges = 0;

    cugraph::grmat_gen ((char *)param.argv.c_str(), vertices, edges, &col_sources, &col_destinations, nullptr);

    ASSERT_EQ((vertices < (1 << 30)), 1);
    cudaStream_t stream{nullptr};
    ALLOC_FREE_TRY(col_sources.data, stream);
    ALLOC_FREE_TRY(col_destinations.data, stream);

  }

  template <typename T>
  void run_check_intergrity(const Grmat_Usecase& param) {
    const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
    Graph_ptr G{new cugraph::Graph, Graph_deleter};
    gdf_column col_sources, col_destinations;

    gdf_dtype gdf_vertexId_type;

    gdf_vertexId_type = GDF_INT32;

    col_sources.dtype = gdf_vertexId_type;
    col_sources.valid = nullptr;
    col_destinations.dtype = gdf_vertexId_type;
    col_destinations.valid = nullptr;

    col_sources.null_count = 0;
    col_destinations.null_count = 0;

    size_t vertices = 0, edges = 0;

    cugraph::grmat_gen ((char *)param.argv.c_str(), vertices, edges, &col_sources, &col_destinations, nullptr);
    std::vector<int> src1_h(edges), dest1_h(edges);

    (cudaMemcpy(&src1_h[0], col_sources.data, sizeof(int) * edges, cudaMemcpyDeviceToHost));
    (cudaMemcpy(&dest1_h[0], col_destinations.data, sizeof(int) * edges, cudaMemcpyDeviceToHost));

    col_sources.valid = nullptr;
    col_destinations.valid = nullptr;
    col_sources.null_count = 0;
    col_destinations.null_count = 0;

    cugraph::edge_list_view(G.get(), &col_sources, &col_destinations, nullptr);
    std::vector<int> src2_h(edges), dest2_h(edges);

    (cudaMemcpy(&src2_h[0],  G.get()->edgeList->src_indices->data, sizeof(int) * edges, cudaMemcpyDeviceToHost));
    (cudaMemcpy(&dest2_h[0], G.get()->edgeList->dest_indices->data, sizeof(int) * edges, cudaMemcpyDeviceToHost));

    ASSERT_EQ( eq(src1_h,src2_h), 0);
    ASSERT_EQ( eq(dest1_h,dest2_h), 0);

    cudaStream_t stream{nullptr};
    ALLOC_FREE_TRY(col_sources.data, stream);
    ALLOC_FREE_TRY(col_destinations.data, stream);
    }

  template <typename T1, typename T2>
  void run_check_with_different_size(const Grmat_Usecase& param) {
    const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
    Graph_ptr G{new cugraph::Graph, Graph_deleter};
    gdf_column col_sources, col_destinations;

    gdf_dtype gdf_vertexId_type;

    if (sizeof (T1) == 4)
         gdf_vertexId_type = GDF_INT32;
     else
         gdf_vertexId_type = GDF_INT64;

    col_sources.dtype = gdf_vertexId_type;
    col_sources.valid = nullptr;
    col_destinations.dtype = gdf_vertexId_type;
    col_destinations.valid = nullptr;

    col_sources.null_count = 0;
    col_destinations.null_count = 0;

    size_t vertices1 = 0, edges1 = 0;

    cugraph::grmat_gen ((char *)param.argv.c_str(), vertices1, edges1, &col_sources, &col_destinations, nullptr);
    std::vector<T1> src1_h(edges1), dest1_h(edges1);

    cudaMemcpy(&src1_h[0], col_sources.data, sizeof(T1) * edges1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&dest1_h[0], col_destinations.data, sizeof(T1) * edges1, cudaMemcpyDeviceToHost);
    
    cudaStream_t stream{nullptr};
    ALLOC_FREE_TRY(col_sources.data, stream);
    ALLOC_FREE_TRY(col_destinations.data, stream);

    if (sizeof (T2) == 4)
         gdf_vertexId_type = GDF_INT32;
     else
         gdf_vertexId_type = GDF_INT64;

    col_sources.dtype = gdf_vertexId_type;
    col_destinations.dtype = gdf_vertexId_type;
    col_sources.valid = nullptr;
    col_destinations.valid = nullptr;

    col_sources.null_count = 0;
    col_destinations.null_count = 0;
 
    size_t vertices2 = 0, edges2 = 0;

    cugraph::grmat_gen ((char *)param.argv.c_str(), vertices2, edges2, &col_sources, &col_destinations, nullptr);

    std::vector<T2> src2_h(edges2), dest2_h(edges2);

    (cudaMemcpy(&src2_h[0], col_sources.data, sizeof(T2) * edges2, cudaMemcpyDeviceToHost));
    (cudaMemcpy(&dest2_h[0], col_destinations.data, sizeof(T2) * edges2, cudaMemcpyDeviceToHost));

    ASSERT_EQ( eq(src1_h, src2_h), 0);
    ASSERT_EQ( eq(dest1_h, dest2_h), 0);

    ALLOC_FREE_TRY(col_sources.data, stream);
    ALLOC_FREE_TRY(col_destinations.data, stream);
    }

  template <typename VertexId, typename T, bool manual_tanspose>
  void run_current_test(const Grmat_Usecase& param) {
     const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();

     Graph_ptr G{new cugraph::Graph, Graph_deleter};
     gdf_column col_sources, col_destinations;
     gdf_error GDF_CUDA_ERROR;
     float alpha = 0.85;
     float tol = 1E-5f;
     int max_iter = 500;
     bool has_guess = false;

     HighResClock hr_clock;
     double time_tmp;
     gdf_column_ptr col_grmat;
     gdf_dtype gdf_vertexId_type;

     if (sizeof (VertexId) == 4) 
         gdf_vertexId_type = GDF_INT32;
     else
         gdf_vertexId_type = GDF_INT64;
 
     // Currently, the page rank supports only int32 and doesn't support long
     gdf_vertexId_type = GDF_INT32;
     col_sources.dtype = gdf_vertexId_type;
     col_sources.valid = nullptr;
     col_destinations.dtype = gdf_vertexId_type;
     col_destinations.valid = nullptr;

    col_sources.null_count = 0;
    col_destinations.null_count = 0;

    size_t vertices = 0, edges = 0;

    cugraph::grmat_gen ((char *)param.argv.c_str(), vertices, edges, &col_sources, &col_destinations, nullptr);

    gdf_dtype_extra_info extra_info;
    extra_info.time_unit = TIME_UNIT_NONE;
    col_sources.dtype_info = extra_info;
    col_sources.valid = nullptr;
    col_destinations.dtype_info = extra_info;
    col_destinations.valid = nullptr;
    col_sources.null_count = 0;
    col_destinations.null_count = 0;
    std::vector<T> grmat(vertices);
    col_grmat = create_gdf_column(grmat);

    cugraph::edge_list_view(G.get(), &col_sources, &col_destinations, nullptr);
    if (manual_tanspose)
      cugraph::add_transposed_adj_list(G.get());

    int device = 0;
    (cudaGetDevice (&device));  
 
    (cudaDeviceSynchronize());
    if (PERF) {
      hr_clock.start();
      for (int i = 0; i < PERF_MULTIPLIER; ++i) {
       cugraph::pagerank(G.get(), col_grmat.get(), nullptr, nullptr, alpha, tol, max_iter, has_guess);
       (cudaDeviceSynchronize());
      }
      hr_clock.stop(&time_tmp);
      grmat_time.push_back(time_tmp);
    }
    else {
      cudaProfilerStart();
      cugraph::pagerank(G.get(), col_grmat.get(), nullptr, nullptr, alpha, tol, max_iter, has_guess);
      cudaProfilerStop();
      (cudaDeviceSynchronize());
    }
    cudaStream_t stream{nullptr};
    ALLOC_FREE_TRY (col_sources.data, stream);
    ALLOC_FREE_TRY (col_destinations.data, stream);

    col_sources.data = nullptr;
    col_destinations.data = nullptr;
    
  }
};

std::vector<double> Tests_Grmat::grmat_time;

TEST_P(Tests_Grmat, CheckFP32) {
    run_current_test<int, float, true>(GetParam());
    run_current_test<int, float, false>(GetParam());
}

TEST_P(Tests_Grmat, CheckFP64) {
    run_current_test<int, double, true>(GetParam());
    run_current_test<int, double,false>(GetParam());
}

TEST_P(Tests_Grmat, CheckInt32)
{
    run_check_max<int> (GetParam());
}

TEST_P(Tests_Grmat, CheckInt64)
{
    run_check_max<long long> (GetParam());
}

TEST_P (Tests_Grmat, misc)
{
    run_check_configuration<int> (GetParam());
    run_check_configuration<long> (GetParam());
    run_check_intergrity<float> (GetParam());
    run_check_with_different_size<int, int> (GetParam());
    run_check_with_different_size<long long, long long> (GetParam());
}

//--gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Grmat, 
                        ::testing::Values( Grmat_Usecase("grmat --rmat_scale=16 --rmat_edgefactor=14  --device=0 --normalized --quiet")
                        				  ,Grmat_Usecase("grmat --rmat_scale=16 --rmat_edgefactor=16 --device=0 --rmat_undirected --quiet")
                                          ,Grmat_Usecase("grmat --rmat_scale=17 --rmat_edgefactor=22 --device=0 --normalized --quiet")
                                         )
                       );


int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}


