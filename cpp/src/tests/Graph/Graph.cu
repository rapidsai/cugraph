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

// Graph tests
// Author: Alex Fender afender@nvidia.com

#include "gtest/gtest.h"
#include <cugraph.h>
#include "test_utils.h"
#include <string.h>

#include <rmm_utils.h>

/*
//TODO: revive the test(s) below, once
//      Gunrock GRMAT is back and stable again;
//
TEST(gdf_edge_list, success)
{

  cudaStream_t stream{nullptr};
       
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  gdf_column col_src, col_dest, col_weights;
  
  col_src.dtype = GDF_INT32;
  col_src.valid = nullptr;
  col_src.null_count = 0;
  col_dest.dtype = GDF_INT32; 
  col_dest.valid = nullptr;
  col_dest.null_count = 0;
  col_weights.dtype = GDF_FLOAT32; 
  col_weights.valid = nullptr;
  col_weights.null_count = 0;

  size_t vertices = 0, edges = 0;
  char argv [1024] = "grmat --rmat_scale=20 --rmat_edgefactor=16 --device=0 --normalized --rmat_self_loops --quiet";
  gdf_grmat_gen(argv, vertices, edges, &col_src, &col_dest, &col_weights);
  
  std::vector<int> src_h(edges), dest_h(edges);
  std::vector<float> w_h(edges);

  cudaMemcpy(&src_h[0], col_src.data, sizeof(int) * edges, cudaMemcpyDeviceToHost);
  cudaMemcpy(&dest_h[0], col_dest.data, sizeof(int) * edges, cudaMemcpyDeviceToHost);
  cudaMemcpy(&w_h[0], col_weights.data, sizeof(float) * edges, cudaMemcpyDeviceToHost);

  cugraph::edge_list_view(G.get(), &col_src, &col_dest, &col_weights);

  std::vector<int> src2_h(edges), dest2_h(edges);
  std::vector<float> w2_h(edges);

  cudaMemcpy(&src2_h[0], G.get()->edgeList->src_indices->data, sizeof(int) * edges, cudaMemcpyDeviceToHost);
  cudaMemcpy(&dest2_h[0], G.get()->edgeList->dest_indices->data, sizeof(int) * edges, cudaMemcpyDeviceToHost);
  cudaMemcpy(&w2_h[0], G.get()->edgeList->edge_data->data, sizeof(float) * edges, cudaMemcpyDeviceToHost);
  
  ASSERT_EQ( eq(src_h,src2_h), 0);
  ASSERT_EQ( eq(dest_h,dest2_h), 0);
  ASSERT_EQ( eq(w_h,w2_h), 0);

  ALLOC_FREE_TRY(col_src.data, stream);
  ALLOC_FREE_TRY(col_dest.data, stream);
  ALLOC_FREE_TRY(col_weights.data, stream);

}

//TODO: revive the test(s) below, once
//      Gunrock GRMAT is back and stable again;
//
TEST(gdf_edge_list, success_no_weights)
{

  cudaStream_t stream{nullptr};
       
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  gdf_column col_src, col_dest;
  
  col_src.dtype = GDF_INT32;
  col_src.valid = nullptr;
  col_dest.dtype = GDF_INT32; 
  col_dest.valid = nullptr;
  col_src.null_count = 0;
  col_dest.null_count = 0;

 
  size_t vertices = 0, edges = 0;
  char argv [1024] = "grmat --rmat_scale=20 --rmat_edgefactor=16 --device=0 --normalized --rmat_self_loops --quiet";
  gdf_grmat_gen(argv, vertices, edges, &col_src, &col_dest, nullptr);
 
  cugraph::edge_list_view(G.get(), &col_src, &col_dest, nullptr);

  ALLOC_FREE_TRY(col_src.data, stream);
  ALLOC_FREE_TRY(col_dest.data, stream);
}
*/

TEST(gdf_edge_list, size_mismatch)
{
       
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  gdf_column_ptr col_src, col_dest, col_weights;
  
  std::vector<int> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4};
  std::vector<float> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50};

  col_src = create_gdf_column(src_h);
  col_dest = create_gdf_column(dest_h);
  col_weights = create_gdf_column(w_h);

  ASSERT_THROW(cugraph::edge_list_view(G.get(), col_src.get(), col_dest.get(), col_weights.get()), std::logic_error);
}


TEST(gdf_edge_list, size_mismatch2)
{
       
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  gdf_column_ptr col_src, col_dest, col_weights;
  
  std::vector<int> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<float> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50};
  
  col_src = create_gdf_column(src_h);
  col_dest = create_gdf_column(dest_h);
  col_weights = create_gdf_column(w_h);

  ASSERT_THROW(cugraph::edge_list_view(G.get(), col_src.get(), col_dest.get(), col_weights.get()), std::logic_error);

}

TEST(gdf_edge_list, wrong_type)
{
       
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  gdf_column_ptr col_src, col_dest;
  
  std::vector<float> src_h={0.0, 0.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0}, dest_h={1.0, 2.0, 0.0, 1.0, 4.0, 4.0, 5.0, 3.0, 5.0, 3.0};

  col_src = create_gdf_column(src_h);
  col_dest = create_gdf_column(dest_h);

  ASSERT_THROW(cugraph::edge_list_view(G.get(), col_src.get(), col_dest.get(), nullptr), std::logic_error);
}

TEST(gdf_adj_list, success)
{
  
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
      
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  gdf_column_ptr col_off, col_ind, col_w;
  
  col_off = create_gdf_column(off_h);
  col_ind = create_gdf_column(ind_h);
  col_w = create_gdf_column(w_h);

  cugraph::adj_list_view(G.get(), col_off.get(), col_ind.get(), col_w.get());

  std::vector<int> off2_h(off_h.size()), ind2_h(ind_h.size());
  std::vector<float> w2_h(w_h.size());

  cudaMemcpy(&off2_h[0], G.get()->adjList->offsets->data, sizeof(int) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind2_h[0], G.get()->adjList->indices->data, sizeof(int) * ind_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&w2_h[0], G.get()->adjList->edge_data->data, sizeof(float) * w_h.size(), cudaMemcpyDeviceToHost);
  
  ASSERT_EQ( eq(off_h,off2_h), 0);
  ASSERT_EQ( eq(ind_h,ind2_h), 0);
  ASSERT_EQ( eq(w_h,w2_h), 0);
}

TEST(gdf_adj_list, success_no_weights)
{
  
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
      
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  gdf_column_ptr col_off, col_ind;
  
  col_off = create_gdf_column(off_h);
  col_ind = create_gdf_column(ind_h);

  cugraph::adj_list_view(G.get(), col_off.get(), col_ind.get(), nullptr);

  std::vector<int> off2_h(off_h.size()), ind2_h(ind_h.size());

  cudaMemcpy(&off2_h[0], G.get()->adjList->offsets->data, sizeof(int) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind2_h[0], G.get()->adjList->indices->data, sizeof(int) * ind_h.size(), cudaMemcpyDeviceToHost);
  
  ASSERT_EQ( eq(off_h,off2_h), 0);
  ASSERT_EQ( eq(ind_h,ind2_h), 0);
}

TEST(Graph_properties, success)
{
     
  Graph_ptr G{new cugraph::Graph, Graph_deleter};
  cugraph::Graph_properties *prop = new cugraph::Graph_properties;
  ASSERT_FALSE(prop->directed);
  ASSERT_FALSE(prop->weighted);
  ASSERT_FALSE(prop->multigraph);
  ASSERT_FALSE(prop->bipartite);
  ASSERT_FALSE(prop->tree);
  prop->directed = true;
  prop->weighted = true;
  prop->tree = false;
  ASSERT_TRUE(prop->directed);
  ASSERT_TRUE(prop->weighted);
  ASSERT_FALSE(prop->multigraph);
  ASSERT_FALSE(prop->bipartite);
  ASSERT_FALSE(prop->tree);
}

TEST(number_of_vertices, success1)
{
  std::vector<int> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5};
  std::vector<int> dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<float> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50, 0.5};

  cugraph::Graph G;
  gdf_column col_src, col_dest, col_w;
  create_gdf_column(src_h, &col_src);
  create_gdf_column(dest_h, &col_dest);
  create_gdf_column(w_h, &col_w);

  cugraph::edge_list_view(&G, &col_src, &col_dest, &col_w);
  ASSERT_EQ(G.numberOfVertices, 0);

  cugraph::number_of_vertices(&G);

  ASSERT_EQ(G.numberOfVertices, 6);
}

TEST(gdf_delete_adjacency_list, success1)
{
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
      
  cugraph::Graph G;
  gdf_column col_off, col_ind, col_w;
  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  create_gdf_column(off_h, &col_off);
  create_gdf_column(ind_h, &col_ind);
  create_gdf_column(w_h, &col_w);

  cugraph::adj_list_view(&G, &col_off, &col_ind, &col_w);
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_adj_list(&G);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);
}

TEST(gdf_delete_adjacency_list, success2)
{
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
      
  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_off = new gdf_column, *col_ind = new gdf_column, *col_w = new gdf_column;
  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  create_gdf_column(off_h, col_off);
  create_gdf_column(ind_h, col_ind);
  create_gdf_column(w_h, col_w);

  cugraph::adj_list_view(G, col_off, col_ind, col_w);
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_adj_list(G);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);

  delete G;
  delete col_off;
  delete col_ind;
  delete col_w;
}


TEST(delete_edge_list, success1)
{
  std::vector<int> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<float> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50, 1.00};

  cugraph::Graph G ;
  gdf_column col_src, col_dest, col_w;
  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  create_gdf_column(src_h, &col_src);
  create_gdf_column(dest_h, &col_dest);
  create_gdf_column(w_h, &col_w);

  cugraph::edge_list_view(&G, &col_src, &col_dest, &col_w);
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_edge_list(&G);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);
}

TEST(delete_edge_list, success2)
{
  std::vector<int> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<float> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50, 1.00};

  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_src = new gdf_column, *col_dest = new gdf_column, *col_w = new gdf_column;
  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  create_gdf_column(src_h, col_src);
  create_gdf_column(dest_h, col_dest);
  create_gdf_column(w_h, col_w);

  cugraph::edge_list_view(G, col_src, col_dest, col_w);
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_edge_list(G);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);

  delete G;
  delete col_src;
  delete col_dest;
  delete col_w;
}

TEST(Graph, add_transposed_adj_list)
{
  std::vector<int> src_h={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
  std::vector<int> dest_h={1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
  
  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_src = new gdf_column, *col_dest = new gdf_column;
  //size_t free, free2, free3, free4, total;  
  
  //cudaMemGetInfo(&free, &total);
  
  create_gdf_column(src_h, col_src);
  create_gdf_column(dest_h, col_dest);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);

  cugraph::edge_list_view(G, col_src, col_dest, nullptr);
  
  //cudaMemGetInfo(&free3, &total);
  //EXPECT_EQ(free2,free3);
  //EXPECT_NE(free,free3);

  cugraph::add_transposed_adj_list(G);

  //this check doen't work on small case (false positive)
  //cudaMemGetInfo(&free3, &total);
  //EXPECT_NE(free3,free2);

  std::vector<int> off_h(G->transposedAdjList->offsets->size ), ind_h(G->transposedAdjList->indices->size);

  cudaMemcpy(&off_h[0], G->transposedAdjList->offsets->data, sizeof(int) * G->transposedAdjList->offsets->size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind_h[0], G->transposedAdjList->indices->data, sizeof(int) * G->transposedAdjList->indices->size, cudaMemcpyDeviceToHost);
  size_t zero = 0;
  EXPECT_GT(off_h.size(), zero);
  EXPECT_GT(ind_h.size(), zero);
  EXPECT_EQ(off_h.size()-2, (size_t)(*(std::max_element(ind_h.begin(), ind_h.end()))));
  EXPECT_EQ(ind_h.size(), (size_t)off_h.back());

  std::sort (ind_h.begin(), ind_h.end());
  std::sort (src_h.begin(), src_h.end());

  EXPECT_EQ( eq(ind_h,src_h), 0);

  delete G;

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free2);
  //EXPECT_NE(free4,free);

  gdf_col_delete(col_src);
  gdf_col_delete(col_dest);

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free);
}

TEST(Graph, gdf_add_adjList)
{
  std::vector<int> src_h={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
  std::vector<int> dest_h={1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
  std::vector<int> off_ref_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156};

  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_src = new gdf_column, *col_dest = new gdf_column;

  //size_t free, free2, free3, free4, total;  
  
  //cudaMemGetInfo(&free, &total);
  
  create_gdf_column(src_h, col_src);
  create_gdf_column(dest_h, col_dest);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);

  cugraph::edge_list_view(G, col_src, col_dest, nullptr);
  
  //cudaMemGetInfo(&free3, &total);
  //EXPECT_EQ(free2,free3);
  //EXPECT_NE(free,free3);

  cugraph::add_adj_list(G);

  //this check doen't work on small case (false positive)
  //cudaMemGetInfo(&free3, &total);
  //EXPECT_NE(free3,free2);

  std::vector<int> off_h(G->adjList->offsets->size ), ind_h(G->adjList->indices->size);

  cudaMemcpy(&off_h[0], G->adjList->offsets->data, sizeof(int) * G->adjList->offsets->size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind_h[0], G->adjList->indices->data, sizeof(int) * G->adjList->indices->size, cudaMemcpyDeviceToHost);
  size_t zero = 0;
  EXPECT_GT(off_h.size(), zero);
  EXPECT_GT(ind_h.size(), zero);
  EXPECT_EQ(off_h.size()-2, (size_t)(*(std::max_element(ind_h.begin(), ind_h.end()))));
  EXPECT_EQ(ind_h.size(), (size_t)off_h.back());

  std::sort (ind_h.begin(), ind_h.end());
  std::sort (dest_h.begin(), dest_h.end()); 

  EXPECT_EQ( eq(ind_h,dest_h), 0);
  EXPECT_EQ( eq(off_h,off_ref_h), 0);

  delete G;

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free2);
  //EXPECT_NE(free4,free);

  gdf_col_delete(col_src);
  gdf_col_delete(col_dest);

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free);
}
void offsets2indices(std::vector<int> &offsets, std::vector<int> &indices) {
  for (int i = 0; i < (int)offsets.size()-1; ++i) 
    for (int j = offsets[i]; j < offsets[i+1]; ++j) 
      indices[j] = i;
}
TEST(Graph, add_edge_list)
{
  
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
      
  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_off = new gdf_column, *col_ind = new gdf_column, *col_w = new gdf_column;
  
  create_gdf_column(off_h, col_off);
  create_gdf_column(ind_h, col_ind);
  create_gdf_column(w_h, col_w);

  cugraph::adj_list_view(G, col_off, col_ind, col_w);

  cugraph::add_edge_list(G);

  std::vector<int> src_h(ind_h.size()), src2_h(ind_h.size()), dest2_h(ind_h.size());
  std::vector<float> w2_h(w_h.size());

  cudaMemcpy(&src2_h[0], G->edgeList->src_indices->data, sizeof(int) * ind_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&dest2_h[0], G->edgeList->dest_indices->data, sizeof(int) * ind_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&w2_h[0], G->edgeList->edge_data->data, sizeof(float) * w_h.size(), cudaMemcpyDeviceToHost);
  
  offsets2indices(off_h, src_h);

  ASSERT_LE(*(std::max_element(src2_h.begin(), src2_h.end())),(int)off_h.size()-1);
  ASSERT_GE(*(std::min_element(src2_h.begin(), src2_h.end())),off_h.front());

  ASSERT_EQ( eq(src_h,src2_h), 0);
  ASSERT_EQ( eq(ind_h,dest2_h), 0);
  ASSERT_EQ( eq(w_h,w2_h), 0);

  delete G;
  gdf_col_delete(col_off);
  gdf_col_delete(col_ind);
  gdf_col_delete(col_w);
}

TEST(Graph, get_vertex_identifiers)
{
  
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};

  std::vector<int> idx_h(off_h.size()-1), idx2_h(off_h.size()-1);

      
  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_off = new gdf_column, *col_ind = new gdf_column, *col_idx = new gdf_column;
  
  create_gdf_column(off_h, col_off);
  create_gdf_column(ind_h, col_ind);
  create_gdf_column(idx2_h, col_idx);

  cugraph::adj_list_view(G, col_off, col_ind, nullptr);
  G->adjList->get_vertex_identifiers(col_idx);

  cudaMemcpy(&idx2_h[0], col_idx->data, sizeof(int) * col_idx->size, cudaMemcpyDeviceToHost);
  
  std::generate(idx_h.begin(), idx_h.end(), [n = 0]() mutable {return n++;});
  
  ASSERT_EQ( eq(idx_h,idx2_h), 0);

  delete G;
  gdf_col_delete(col_off);
  gdf_col_delete(col_ind);
  gdf_col_delete(col_idx);
}

TEST(Graph, get_source_indices)
{
  
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};

  std::vector<int> src_h(ind_h.size()), src2_h(ind_h.size());
      
  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_off = new gdf_column, *col_ind = new gdf_column, *col_src = new gdf_column;
  
  create_gdf_column(off_h, col_off);
  create_gdf_column(ind_h, col_ind);
  create_gdf_column(src2_h, col_src);

  cugraph::adj_list_view(G, col_off, col_ind, nullptr);
  G->adjList->get_source_indices(col_src);
  cudaMemcpy(&src2_h[0], col_src->data, sizeof(int) * col_src->size, cudaMemcpyDeviceToHost);
  
  offsets2indices(off_h, src_h);

  ASSERT_EQ( eq(src_h,src2_h), 0);

  delete G;
  gdf_col_delete(col_off);
  gdf_col_delete(col_ind);
  gdf_col_delete(col_src);
}

/*
//TODO: revive the test(s) below, once
//      Gunrock GRMAT is back and stable again;
//
TEST(Graph, memory)
{

  Graph *G = new cugraph::Graph;
  gdf_column col_src, col_dest;

  col_src.dtype = GDF_INT32;
  col_src.valid = nullptr;
  col_dest.dtype = GDF_INT32;
  col_dest.valid = nullptr;

  col_src.null_count = 0;
  col_dest.null_count = 0;

  //size_t free, free2, free3, free4_, free4, total;  
  
  //cudaMemGetInfo(&free, &total);

  size_t vertices = 0, edges = 0;
  char argv[1024] = "grmat --rmat_scale=23 --rmat_edgefactor=16 --device=0 --normalized --rmat_self_loops --quiet";

  gdf_grmat_gen(argv, vertices, edges, &col_src, &col_dest, nullptr);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);

  cugraph::edge_list_view(G, &col_src, &col_dest, nullptr);
  
  //cudaMemGetInfo(&free3, &total);
  //EXPECT_EQ(free2,free3);
  //EXPECT_NE(free,free3);

  cugraph::add_transposed_adj_list(G);
  //this check doen't work on small case (false positive)
  //cudaMemGetInfo(&free4_, &total);
  //EXPECT_NE(free4_,free2);

  cugraph::add_adj_list(G);
  cugraph::delete_adj_list(G);

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free4_);
  //EXPECT_NE(free4,free2);

  delete G;

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free3);
  //EXPECT_NE(free4,free);

  cudaStream_t stream{nullptr};
  ALLOC_FREE_TRY(col_src.data, stream);
  ALLOC_FREE_TRY(col_dest.data, stream);
  
  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free);
}
*/

TEST(Graph, gdf_column_overhead)
{
  size_t sz = 100000000;
  std::vector<int> src_h(sz,1);
  std::vector<int> dest_h(sz,1);

  //size_t free, free2, free3, total;  
  //cudaMemGetInfo(&free, &total);

  cugraph::Graph *G = new cugraph::Graph;
  gdf_column *col_src = new gdf_column, *col_dest = new gdf_column;

  create_gdf_column(src_h, col_src);
  create_gdf_column(dest_h, col_dest);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);

  // check that gdf_column_overhead < 5 per cent
  //EXPECT_LT(free-free2, 2*sz*sizeof(int)*1.05);

  cugraph::edge_list_view(G, col_src, col_dest, nullptr);

  //cudaMemGetInfo(&free3, &total);
  //EXPECT_EQ(free2,free3);
  //EXPECT_NE(free,free3);

  delete G;
  gdf_col_delete(col_src);
  gdf_col_delete(col_dest);
}

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}
