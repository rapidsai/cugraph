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
#include <algorithm>
#include "test_utils.h"

#include <rmm_utils.h>

TEST(nvgraph_louvain, success)
{
  cugraph::Graph G;

  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0,
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33,
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15,
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
 
  gdf_column col_off, col_ind, col_w;


  create_gdf_column(off_h,&col_off);
  create_gdf_column(ind_h,&col_ind);
  create_gdf_column(w_h  ,&col_w);

  cugraph::adj_list_view(&G, &col_off, &col_ind, &col_w);

  if (!(G.adjList))
    cugraph::add_adj_list(&G);

  int no_vertex = off_h.size()-1;
  int weighted = 0; //false
  int has_init_cluster = 0; //false
  float modularity = 0.0;
  int num_level = 40;
  int* best_cluster_vec = NULL;

  cudaStream_t stream{nullptr};
  ALLOC_TRY((void**)&best_cluster_vec, sizeof(int) * no_vertex, stream);
  
  ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, nvgraphLouvain (CUDA_R_32I, CUDA_R_32F, no_vertex, ind_h.size(),
                            G.adjList->offsets->data, G.adjList->indices->data, G.adjList->edge_data->data, weighted, has_init_cluster, nullptr,
                            (void*) &modularity, (void*) best_cluster_vec, (void *)(&num_level)));
  
  std::vector<int> cluster_id (34, -1);
  cudaMemcpy ((void*) &(cluster_id[0]), best_cluster_vec, sizeof(int)*34, cudaMemcpyDeviceToHost);
  int max = *max_element (cluster_id.begin(), cluster_id.end()); 
  int min = *min_element (cluster_id.begin(), cluster_id.end()); 
  ASSERT_EQ((min >= 0), 1);
  ASSERT_EQ((modularity >= 0.402777), 1);

  //printf ("max is %d and min is %d \n", max, min);

  //printf ("Modularity is %f \n", modularity);

  ALLOC_FREE_TRY (best_cluster_vec, stream);
}
/*
//TODO: revive the test(s) below, once
//      Gunrock GRMAT is back and stable again;
//
TEST(nvgraph_louvain_grmat, success)
{
  cugraph::Graph G;
  gdf_column col_src, col_dest, col_weights;
  size_t vertices = 0, edges = 0;
  char argv[1024] = "grmat --rmat_scale=23 --rmat_edgefactor=16 --device=0 --normalized --quiet ";

  col_src.data = nullptr;
  col_src.dtype = GDF_INT32;
  col_src.valid = nullptr;
  col_dest.data = nullptr;
  col_dest.dtype = GDF_INT32;
  col_dest.valid = nullptr;
  col_weights.data = nullptr;
  col_weights.dtype = GDF_FLOAT32;
  col_weights.valid = nullptr;

  col_src.null_count = 0;
  col_dest.null_count = 0;
  col_weights.null_count = 0;

  cugraph::grmat_gen(argv, vertices, edges, &col_src, &col_dest, nullptr);
  cudaStream_t stream{nullptr};
  ALLOC_TRY ((void**)&col_weights.data, sizeof(int) * edges, stream);
  col_weights.size = edges;
  std::vector<float> w_h (edges, (float)1.0);
  cudaMemcpy (col_weights.data, (void*) &(w_h[0]), sizeof(float)*edges, cudaMemcpyHostToDevice);
  cugraph::edge_list_view(&G, &col_src, &col_dest, &col_weights);

  if (!(G.adjList))
  {
    cugraph::add_adj_list(&G);
  }
  int weighted = 1; //false
  int has_init_cluster = 0; //false
  float modularity = 0.0;
  int num_level = 0;
  int* best_cluster_vec = NULL;

  ALLOC_TRY ((void**)&best_cluster_vec, sizeof(int) * vertices, stream);

  ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, nvgraphLouvain (CUDA_R_32I, CUDA_R_32F, vertices, edges, G.adjList->offsets->data, G.adjList->indices->data, G.adjList->edge_data->data, weighted, has_init_cluster, nullptr, (void*) &modularity, (void*) best_cluster_vec, (void *)(&num_level)));

  
  std::vector<int> cluster_id (vertices, -1);
  cudaMemcpy ((void*) &(cluster_id[0]), best_cluster_vec, sizeof(int)*vertices, cudaMemcpyDeviceToHost);
  int max = *max_element (cluster_id.begin(), cluster_id.end()); 
  int min = *min_element (cluster_id.begin(), cluster_id.end()); 

  ASSERT_EQ((min >= 0), 1);
  ASSERT_EQ((modularity >= 0.002875), 1);
   
  ALLOC_FREE_TRY (best_cluster_vec, stream);
  ALLOC_FREE_TRY(col_src.data, stream);
  ALLOC_FREE_TRY(col_dest.data, stream);
  ALLOC_FREE_TRY(col_weights.data, stream);

}
*/
int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}



