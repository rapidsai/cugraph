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
#include "types.h"
#include "functions.h"
#include "test_utils.h"
#include <string.h>
#include <rmm_utils.h>

TEST(edge_list, size_mismatch)
{
       
  typedef int VT;
  typedef float WT;

  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  
  std::vector<VT> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4};
  std::vector<WT> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50};
  size_t num_e = src_h.size();

  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  ASSERT_THROW(cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get(), d_w.get()), std::logic_error);
}


TEST(edge_list, size_mismatch2)
{
  typedef int VT;
  typedef float WT;
       
  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  
  std::vector<VT> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<WT> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50};
  size_t num_e = src_h.size();
  
  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  ASSERT_THROW(cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get(), d_w.get()), std::logic_error);

}

TEST(adj_list, success)
{
  typedef int VT;
  typedef float WT;
  // Hard-coded Zachary Karate Club network input
  std::vector<VT> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<VT> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<WT> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  size_t num_v = off_h.size()-1, num_e = ind_h.size();
      
  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  
  d_ptr<VT> d_off = create_d_ptr(off_h);
  d_ptr<VT> d_ind = create_d_ptr(ind_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  cugraph::adj_list_view(G.get(), num_v, num_e, d_off.get(), d_ind.get(), d_w.get());

  std::vector<VT> off2_h(off_h.size()), ind2_h(ind_h.size());
  std::vector<WT> w2_h(w_h.size());

  cudaMemcpy(&off2_h[0], G.get()->adjList->offsets, sizeof(VT) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind2_h[0], G.get()->adjList->indices, sizeof(VT) * ind_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&w2_h[0], G.get()->adjList->edge_data, sizeof(WT) * w_h.size(), cudaMemcpyDeviceToHost);
  
  ASSERT_EQ( eq(off_h,off2_h), 0);
  ASSERT_EQ( eq(ind_h,ind2_h), 0);
  ASSERT_EQ( eq(w_h,w2_h), 0);
}

TEST(adj_list, success_no_weights)
{
  typedef int VT;
  // Hard-coded Zachary Karate Club network input
  std::vector<VT> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<VT> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  size_t num_v = off_h.size()-1, num_e = ind_h.size();

  Graph_ptr<VT> G{new cugraph::Graph<VT>, Graph_deleter<VT>};
  
  d_ptr<VT> d_off = create_d_ptr(off_h);
  d_ptr<VT> d_ind = create_d_ptr(ind_h);

  cugraph::adj_list_view(G.get(), num_v, num_e, d_off.get(), d_ind.get());

  std::vector<VT> off2_h(off_h.size()), ind2_h(ind_h.size());

  cudaMemcpy(&off2_h[0], G.get()->adjList->offsets, sizeof(VT) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind2_h[0], G.get()->adjList->indices, sizeof(VT) * ind_h.size(), cudaMemcpyDeviceToHost);
  
  ASSERT_EQ( eq(off_h,off2_h), 0);
  ASSERT_EQ( eq(ind_h,ind2_h), 0);
}

TEST(Graph_properties, success)
{
  typedef int VT;
  typedef float WT;

  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
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
  typedef int VT;
  typedef float WT;

  std::vector<VT> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5};
  std::vector<VT> dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<WT> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50, 0.5};
  size_t num_e = src_h.size();

  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get(), d_w.get());
  ASSERT_EQ(G.get()->v, 0);

  cugraph::number_of_vertices(G.get());
  ASSERT_EQ(G.get()->v, 6);
}

TEST(delete_adjacency_list, success1)
{
  typedef int VT;
  typedef float WT;
  // Hard-coded Zachary Karate Club network input
  std::vector<VT> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<VT> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<WT> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  
  size_t num_v = off_h.size()-1, num_e = ind_h.size();

  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  d_ptr<VT> d_off = create_d_ptr(off_h);
  d_ptr<VT> d_ind = create_d_ptr(ind_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  cugraph::adj_list_view(G.get(), num_v, num_e, d_off.get(), d_ind.get(), d_w.get());
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_adj_list(G.get());

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);
}

TEST(delete_adjacency_list, success2)
{
  typedef int VT;
  typedef float WT;
  // Hard-coded Zachary Karate Club network input
  std::vector<VT> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<VT> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<WT> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  
  size_t num_v = off_h.size()-1, num_e = ind_h.size();
    
  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  d_ptr<VT> d_off = create_d_ptr(off_h);
  d_ptr<VT> d_ind = create_d_ptr(ind_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  cugraph::adj_list_view(G.get(), num_v, num_e, d_off.get(), d_ind.get(), d_w.get());
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_adj_list(G.get());

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);
}


TEST(delete_edge_list, success1)
{
  typedef int VT;
  typedef float WT;

  std::vector<VT> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<WT> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50, 1.00};
  size_t num_e = src_h.size();

  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get(), d_w.get());
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_edge_list(G.get());

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);
}

TEST(delete_edge_list, success2)
{
  typedef int VT;
  typedef float WT;

  std::vector<VT> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<WT> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50, 1.00};
  size_t num_e = src_h.size();

  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};

  //size_t free, free2, total;  
  //cudaMemGetInfo(&free, &total);
  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get(), d_w.get());
  
  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);
  
  cugraph::delete_edge_list(G.get());

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_EQ(free,free2);

}

TEST(Graph, add_transposed_adj_list)
{
  typedef int VT;

  std::vector<VT> src_h={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
  std::vector<VT> dest_h={1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
  
  size_t num_e = src_h.size();

  Graph_ptr<VT> G{new cugraph::Graph<VT>, Graph_deleter<VT>};
  //size_t free, free2, free3, free4, total;  
  
  //cudaMemGetInfo(&free, &total);
  
  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);

  //cudaMemGetInfo(&free2, &total);
  //EXPECT_NE(free,free2);

  cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get());
  
  //cudaMemGetInfo(&free3, &total);
  //EXPECT_EQ(free2,free3);
  //EXPECT_NE(free,free3);

  cugraph::add_transposed_adj_list(G.get());

  //this check doen't work on small case (false positive)
  //cudaMemGetInfo(&free3, &total);
  //EXPECT_NE(free3,free2);

  std::vector<VT> off_h(G.get()->v+1 ), ind_h(G.get()->e);

  cudaMemcpy(&off_h[0], G.get()->transposedAdjList->offsets, sizeof(VT) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind_h[0], G.get()->transposedAdjList->indices, sizeof(VT) * ind_h.size(), cudaMemcpyDeviceToHost);
  size_t zero = 0;
  EXPECT_GT(off_h.size(), zero);
  EXPECT_GT(ind_h.size(), zero);
  EXPECT_EQ(off_h.size()-2, (size_t)(*(std::max_element(ind_h.begin(), ind_h.end()))));
  EXPECT_EQ(ind_h.size(), (size_t)off_h.back());

  std::sort (ind_h.begin(), ind_h.end());
  std::sort (src_h.begin(), src_h.end());

  EXPECT_EQ( eq(ind_h,src_h), 0);

  

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free2);
  //EXPECT_NE(free4,free);


  

  //cudaMemGetInfo(&free4, &total);
  //EXPECT_EQ(free4,free);
}

TEST(Graph, add_adjList)
{
  typedef int VT;

  std::vector<VT> src_h={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
  std::vector<VT> dest_h={1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2, 3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12, 13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
  std::vector<VT> off_ref_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156};

  size_t num_e = src_h.size();

  Graph_ptr<VT> G{new cugraph::Graph<VT>, Graph_deleter<VT>};
  
  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);

  cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get());
  
  cugraph::add_adj_list(G.get());

  std::vector<VT> off_h(G.get()->v+1 ), ind_h(G.get()->e);

  cudaMemcpy(&off_h[0], G.get()->adjList->offsets, sizeof(VT) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind_h[0], G.get()->adjList->indices, sizeof(VT) * ind_h.size(), cudaMemcpyDeviceToHost);
  size_t zero = 0;
  EXPECT_GT(off_h.size(), zero);
  EXPECT_GT(ind_h.size(), zero);
  EXPECT_EQ(off_h.size()-2, (size_t)(*(std::max_element(ind_h.begin(), ind_h.end()))));
  EXPECT_EQ(ind_h.size(), (size_t)off_h.back());

  std::sort (ind_h.begin(), ind_h.end());
  std::sort (dest_h.begin(), dest_h.end()); 

  EXPECT_EQ( eq(ind_h,dest_h), 0);
  EXPECT_EQ( eq(off_h,off_ref_h), 0);

}

void offsets2indices(std::vector<int> &offsets, std::vector<int> &indices) {
  for (auto i = 0; i < offsets.size()-1; ++i) 
    for (auto j = offsets[i]; j < offsets[i+1]; ++j) 
      indices[j] = i;
}
TEST(Graph, add_edge_list)
{
  typedef int VT;
  typedef float WT;
  
  // Hard-coded Zachary Karate Club network input
  std::vector<VT> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<VT> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<WT> w_h = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  size_t num_v = off_h.size()-1, num_e = ind_h.size();
      
  Graph_ptr<VT,WT> G{new cugraph::Graph<VT,WT>, Graph_deleter<VT,WT>};
  
  d_ptr<VT> d_off = create_d_ptr(off_h);
  d_ptr<VT> d_ind = create_d_ptr(ind_h);
  d_ptr<WT> d_w = create_d_ptr(w_h);

  cugraph::adj_list_view(G.get(), num_v, num_e, d_off.get(), d_ind.get(), d_w.get());

  cugraph::add_edge_list(G.get());

  std::vector<VT> src_h(num_e), src2_h(num_e), dest2_h(num_e);
  std::vector<WT> w2_h(num_e);

  cudaMemcpy(&src2_h[0], G.get()->edgeList->src_indices, sizeof(VT) * num_e, cudaMemcpyDeviceToHost);
  cudaMemcpy(&dest2_h[0], G.get()->edgeList->dest_indices, sizeof(VT) * num_e, cudaMemcpyDeviceToHost);
  cudaMemcpy(&w2_h[0], G.get()->edgeList->edge_data, sizeof(WT) * num_e, cudaMemcpyDeviceToHost);
  
  offsets2indices(off_h, src_h);

  ASSERT_LE(*(std::max_element(src2_h.begin(), src2_h.end())),(VT)num_v);
  ASSERT_GE(*(std::min_element(src2_h.begin(), src2_h.end())),off_h.front());

  ASSERT_EQ( eq(src_h,src2_h), 0);
  ASSERT_EQ( eq(ind_h,dest2_h), 0);
  ASSERT_EQ( eq(w_h,w2_h), 0);

}

TEST(Graph, get_vertex_identifiers)
{
  typedef int VT;
  
  // Hard-coded Zachary Karate Club network input
  std::vector<VT> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<VT> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};

  size_t num_v = off_h.size()-1, num_e = ind_h.size();

  std::vector<VT> idx_h(num_v), idx2_h(num_v);

  Graph_ptr<VT> G{new cugraph::Graph<VT>, Graph_deleter<VT>};
  d_ptr<VT> d_off = create_d_ptr(off_h);
  d_ptr<VT> d_ind = create_d_ptr(ind_h);
  d_ptr<VT> col_idx = create_d_ptr(idx2_h);

  cugraph::adj_list_view(G.get(), num_v, num_e, d_off.get(), d_ind.get());
  G->adjList->get_vertex_identifiers(num_v, col_idx.get());

  cudaMemcpy(&idx2_h[0], col_idx.get(), sizeof(VT) * idx2_h.size(), cudaMemcpyDeviceToHost);
  
  std::generate(idx_h.begin(), idx_h.end(), [n = 0]() mutable {return n++;});
  
  ASSERT_EQ( eq(idx_h,idx2_h), 0);

}

TEST(Graph, get_source_indices)
{
  typedef int VT;
  // Hard-coded Zachary Karate Club network input
  std::vector<VT> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<VT> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  
  size_t num_v = off_h.size()-1, num_e = ind_h.size();
  std::vector<VT> src_h(num_e), src2_h(num_e);
      
  Graph_ptr<VT> G{new cugraph::Graph<VT>, Graph_deleter<VT>};
  d_ptr<VT> d_off = create_d_ptr(off_h);
  d_ptr<VT> d_ind = create_d_ptr(ind_h);
  d_ptr<VT> d_src = create_d_ptr(src2_h);

  cugraph::adj_list_view(G.get(), num_v, num_e, d_off.get(), d_ind.get());
  G->adjList->get_source_indices(num_v, d_src.get());
  cudaMemcpy(&src2_h[0], d_src.get(), sizeof(VT) * G->e, cudaMemcpyDeviceToHost);
  offsets2indices(off_h, src_h);
  ASSERT_EQ( eq(src_h,src2_h), 0);
}

TEST(Graph, column_overhead)
{
  typedef int VT;
  
  size_t sz = 100000000;
  std::vector<VT> src_h(sz,1);
  std::vector<VT> dest_h(sz,1);
  size_t num_e = src_h.size();

  Graph_ptr<VT> G{new cugraph::Graph<VT>, Graph_deleter<VT>};

  d_ptr<VT> d_src = create_d_ptr(src_h);
  d_ptr<VT> d_dst = create_d_ptr(dest_h);

  cugraph::edge_list_view(G.get(), num_e, d_src.get(), d_dst.get());
}

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;

  }
