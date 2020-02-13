/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <string.h>

#include "test_utils.hpp"
#include "gtest/gtest.h"

#include <graph.hpp>
#include <cudf_conversion.hpp>
#include <rmm_utils.h>

using cudf::column;
using cugraph::experimental::Graph;

TEST(edge_list, size_mismatch)
{
  std::vector<int> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4};
  std::vector<float> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50, 0.50};

  auto col_src = detail::create_column<int>(src_h.begin(), src_h.end());
  auto col_dest = detail::create_column<int>(dest_h.begin(), dest_h.end());
  auto col_weights = detail::create_column<float>(w_h.begin(), w_h.end());

  ASSERT_THROW(cugraph::experimental::from_edge_list(*col_src, *col_dest, *col_weights, 6, 10),
               std::logic_error
               );
}


TEST(edge_list, size_mismatch2)
{
  std::vector<int> src_h={0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<float> w_h={0.50, 0.50, 0.33, 0.33, 0.33, 0.50, 0.50, 0.50};
  
  auto col_src = detail::create_column<int>(src_h.begin(), src_h.end());
  auto col_dest = detail::create_column<int>(dest_h.begin(), dest_h.end());
  auto col_weights = detail::create_column<float>(w_h.begin(), w_h.end());

  ASSERT_THROW(
               {
                 auto g = cugraph::experimental::from_edge_list(*col_src, *col_dest, *col_weights, 6, 10);
               },
               std::logic_error
               );
}

TEST(edge_list, wrong_type)
{
  std::vector<float> src_h={0.0, 0.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0}, dest_h={1.0, 2.0, 0.0, 1.0, 4.0, 4.0, 5.0, 3.0, 5.0, 3.0};

  auto col_src = detail::create_column<float>(src_h.begin(), src_h.end());
  auto col_dest = detail::create_column<float>(dest_h.begin(), dest_h.end());

  ASSERT_THROW(cugraph::experimental::from_edge_list(*col_src, *col_dest, 6, 10), std::logic_error);
}

TEST(adj_list, success)
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
      
  auto col_off = detail::create_column<int>(off_h.begin(), off_h.end());
  auto col_ind = detail::create_column<int>(ind_h.begin(), ind_h.end());
  auto col_w   = detail::create_column<float>(w_h.begin(), w_h.end());

  auto G = cugraph::experimental::from_adj_list<int,float>(*col_off, *col_ind, *col_w, off_h.size()-1, off_h[off_h.size()-1]);

  std::vector<int> off2_h(off_h.size()), ind2_h(ind_h.size());
  std::vector<float> w2_h(w_h.size());

  cudaMemcpy(&off2_h[0], G->adjList.offsets, sizeof(int) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind2_h[0], G->adjList.indices, sizeof(int) * ind_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&w2_h[0], G->adjList.edge_data, sizeof(float) * w_h.size(), cudaMemcpyDeviceToHost);
  
  ASSERT_EQ(off_h,off2_h);
  ASSERT_EQ(ind_h,ind2_h);
  ASSERT_EQ(w_h,w2_h);
}

TEST(adj_list, success_no_weights)
{
  // Hard-coded Zachary Karate Club network input
  std::vector<int> off_h = {0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66, 67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 
      139, 156};
  std::vector<int> ind_h = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0, 
      6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 
      25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 
      18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};

  auto col_off = detail::create_column<int>(off_h.begin(), off_h.end());
  auto col_ind = detail::create_column<int>(ind_h.begin(), ind_h.end());

  auto G = cugraph::experimental::from_adj_list<int,float>(*col_off, *col_ind, off_h.size()-1, off_h[off_h.size()-1]);
  
  std::vector<int> off2_h(off_h.size()), ind2_h(ind_h.size());

  cudaMemcpy(&off2_h[0], G->adjList.offsets, sizeof(int) * off_h.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ind2_h[0], G->adjList.indices, sizeof(int) * ind_h.size(), cudaMemcpyDeviceToHost);
  
  ASSERT_EQ(off_h,off2_h);
  ASSERT_EQ(ind_h,ind2_h);
}

TEST(GraphProperties, success)
{
  cugraph::experimental::GraphProperties prop;
  ASSERT_FALSE(prop.directed);
  ASSERT_FALSE(prop.weighted);
  ASSERT_FALSE(prop.multigraph);
  ASSERT_FALSE(prop.bipartite);
  ASSERT_FALSE(prop.tree);
  prop.directed = true;
  prop.weighted = true;
  prop.tree = false;
  ASSERT_TRUE(prop.directed);
  ASSERT_TRUE(prop.weighted);
  ASSERT_FALSE(prop.multigraph);
  ASSERT_FALSE(prop.bipartite);
  ASSERT_FALSE(prop.tree);
}

void offsets2indices(std::vector<int> &offsets, std::vector<int> &indices) {
  for (int i = 0; i < (int)offsets.size()-1; ++i) 
    for (int j = offsets[i]; j < offsets[i+1]; ++j) 
      indices[j] = i;
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


  auto col_off = detail::create_column<int>(off_h.begin(), off_h.end());
  auto col_ind = detail::create_column<int>(ind_h.begin(), ind_h.end());

  auto G = cugraph::experimental::from_adj_list<int,float>(*col_off, *col_ind, off_h.size()-1, off_h[off_h.size()-1]);
      
  // TODO:  col_idx does NOT need to be a column, it could be a device_vector
  // TODO:  implement get_vertex_identifiers and get_source_indices
  thrust::device_vector<int>   idx2(off_h.size()-1);

  G->adjList.get_vertex_identifiers(idx2.data().get());

  cudaMemcpy(&idx2_h[0], idx2.data().get(), sizeof(int) * idx2.size(), cudaMemcpyDeviceToHost);
  
  std::generate(idx_h.begin(), idx_h.end(), [n = 0]() mutable {return n++;});
  
  ASSERT_EQ(idx_h, idx2_h);
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

  auto col_off = detail::create_column<int>(off_h.begin(), off_h.end());
  auto col_ind = detail::create_column<int>(ind_h.begin(), ind_h.end());

  auto G = cugraph::experimental::from_adj_list<int,float>(*col_off, *col_ind, off_h.size()-1, off_h[off_h.size()-1]);
  
  thrust::device_vector<int>   src2(ind_h.size());

  G->adjList.get_source_indices(src2.data().get());
  cudaMemcpy(&src2_h[0], src2.data().get(), sizeof(int) * src2.size(), cudaMemcpyDeviceToHost);
  
  offsets2indices(off_h, src_h);

  ASSERT_EQ(src_h,src2_h);
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

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}
