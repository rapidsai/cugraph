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
#include <stdio.h>
#include <gtest/gtest.h>
#include <nvgraph/nvgraph.h>
#include <cugraph.h>
#include <algorithm>
#include "test_utils.h"

#include <rmm_utils.h>

template <bool weighted, typename T> 
int jaccard_ref(int n, int e, int *csrPtr, int *csrInd, T * csrVal, T *v, T *work, T gamma, T *weight) {
   /* ASSUMPTION: std::set_intersection assumes the arrays are sorted/ordered */
   // intersect (Vi, Vj) and store the result in a vector using a standard intersection routine
   int start,end,length,col,cstart,cend;
   T Wi,Ws,Wu,last;
   std::vector<int> ind(n);
   std::vector<int>::iterator ind_it;
  for (int row=0; row<n; row++) {
      start = csrPtr[row];
      end   = csrPtr[row+1];
      length= end-start;
      //compute row sums
      if (weighted) {
          last =0.0;
          for (int j=start; j<end; j++) {
              col = csrInd[j];
              last += v[col];
          }
          work[row] = last;
      }
      else {
          work[row] = (T)length;
      }
  }
  for (int row=0; row<n; row++) { 
    start = csrPtr[row];
    end   = csrPtr[row+1];
    
    for (int j=start; j<end; j++) { 
        col = csrInd[j];
        cstart = csrPtr[col];
        cend   = csrPtr[col+1];

      ind_it = std::set_intersection(csrInd + start, csrInd + end, 
                       csrInd + cstart,csrInd + cend, 
                       ind.begin());
      
      length = ind_it - ind.begin();

      if (weighted) { //&& !use_edge_weight
        last =0.0;
        for (int i=0; i<length; i++) {
          last += v[ind[i]];
        }
        Wi =  last;          
      }
      else {
        Wi =  length;
      }
      // weight[j] =  |intersect(Vi,Vj)|/|union(Vi,Vj)| -- which is the jacard weight and |union(Vi,Vj)| = |Vi|+|Vj|-|intersect(Vi,Vj)| 
      Ws =  work[row] + work[col];
      Wu =  Ws - Wi;
      weight[j] = (gamma*csrVal[j])* (Wi/Wu); //Wi; //Ws;
    }
  }
  return 0;
}  

TEST(nvgraph_jaccard, success)
{
  cugraph::Graph G;
  gdf_column col_off, col_ind;
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0,
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33,
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15,
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
 
  create_gdf_column(off_h, &col_off);
  create_gdf_column(ind_h, &col_ind);

  cugraph::adj_list_view(&G, &col_off, &col_ind, nullptr);

  int no_vertex = off_h.size()-1;
  size_t edges = ind_h.size();
  int weighted = 0; // false, it assumes weight of size 1.0 for all the edges
  float* weight_j = NULL;
  float gamma = 1.0;

  cudaStream_t stream{nullptr};
  ALLOC_TRY((void**)&weight_j, sizeof(float)*edges, stream);
  
  ASSERT_EQ(nvgraphJaccard (CUDA_R_32I, CUDA_R_32F, no_vertex, edges,
                            (void*)G.adjList->offsets->data, 
                            (void *)G.adjList->indices->data, 
                            nullptr, 
                            weighted, nullptr, (void*)&gamma, (void*)weight_j), NVGRAPH_STATUS_SUCCESS);

  std::vector<float> val_h (edges, 1.0);
  std::vector<float> jw_h (edges, -1.0);
  std::vector<float> v (no_vertex, 1.0);
  std::vector<float> work (no_vertex, 0.0);
 
  std::vector<float> jaccard_w (edges, 0.0);
  cudaMemcpy((void*)&jaccard_w[0], (void*)weight_j, sizeof(float)*edges, cudaMemcpyDeviceToHost);

  jaccard_ref <true, float>(no_vertex, edges, &off_h[0], &ind_h[0], &val_h[0], &v[0], &work[0], gamma, &jw_h[0]);

  EXPECT_EQ(eq (jaccard_w, jw_h), 0);  

  ALLOC_FREE_TRY (weight_j, stream);
  ALLOC_FREE_TRY (col_off.data, stream);
  ALLOC_FREE_TRY (col_ind.data, stream);
}

/*
//TODO: revive the test(s) below, once
//      Gunrock GRMAT is back and stable again;
//
TEST(nvgraph_jaccard_grmat, success)
{
  cugraph::Graph G;
  gdf_column col_src, col_dest;

  size_t vertices = 0, edges = 0;
  char argv[1024] = "grmat --rmat_scale=16 --rmat_edgefactor=10 --device=0 --normalized --quiet";

  col_src.data = nullptr;
  col_src.dtype = GDF_INT32;
  col_src.valid = nullptr;
  col_dest.data = nullptr;
  col_dest.dtype = GDF_INT32;
  col_dest.valid = nullptr;

  col_src.null_count = 0;
  col_dest.null_count = 0;

  cugraph::grmat_gen(argv, vertices, edges, &col_src, &col_dest, nullptr);
  std::vector<int> src_h (col_src.size, 0);
  std::vector<int> dest_h (col_dest.size, 0);
  cudaMemcpy((void*)&src_h[0], (void*)col_src.data, sizeof(float)*edges, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)&dest_h[0], (void*)col_dest.data, sizeof(float)*edges, cudaMemcpyDeviceToHost);


  cugraph::edge_list_view(&G, &col_src, &col_dest, nullptr);

  if (!G.adjList)
    cugraph::add_adj_list(&G);

  
  int weighted = 0; //false, it assumes weight of size 1.0 for all the edges
  float* weight_j = NULL;
  float gamma = 1.0;
   
  std::vector<int> off_h ((vertices+1), 0.0);
  std::vector<int> ind_h (edges, 0.0);
  cudaMemcpy ((void*) &off_h[0], G.adjList->offsets->data, sizeof(int)*(vertices+1), cudaMemcpyDeviceToHost);
  cudaMemcpy ((void*) &ind_h[0], G.adjList->indices->data, sizeof(int)*edges, cudaMemcpyDeviceToHost);

  cudaStream_t stream{nullptr};
  ALLOC_TRY((void**)&weight_j, sizeof(float)*edges, stream);

  ASSERT_EQ(nvgraphJaccard (CUDA_R_32I, CUDA_R_32F, vertices, edges,
                            (void*)G.adjList->offsets->data,
                            (void *)G.adjList->indices->data,
                            nullptr,
                            weighted, nullptr, (void*)&gamma, (void*)weight_j), NVGRAPH_STATUS_SUCCESS);

  std::vector<float> val_h (edges, 1.0);
  std::vector<float> jw_h (edges, -1.0);
  std::vector<float> v (vertices, 1.0);
  std::vector<float> work (vertices, 0.0);
  int max = *max_element (ind_h.begin(), ind_h.end());
  int min = *min_element (ind_h.begin(), ind_h.end());

  std::vector<float> jaccard_w (edges, 0.0);
  cudaMemcpy((void*)&jaccard_w[0], (void*)weight_j, sizeof(float)*edges, cudaMemcpyDeviceToHost);

  jaccard_ref <false, float>(vertices, edges, &off_h[0], &ind_h[0], &val_h[0], &v[0], &work[0], gamma, &jw_h[0]);
  
  EXPECT_EQ(eq (jaccard_w, jw_h), 0);

  ALLOC_FREE_TRY(weight_j, stream);
  ALLOC_FREE_TRY(col_src.data, stream);
  ALLOC_FREE_TRY(col_dest.data, stream);

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



