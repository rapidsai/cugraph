/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include <gtest/gtest.h>

#include <algorithms.hpp>
#include <graph.hpp>

#include <thrust/extrema.h>

#include <rmm/thrust_rmm_allocator.h>

#include <rmm/mr/device/cuda_memory_resource.hpp>

TEST(nvgraph_louvain, success)
{
  std::vector<int> off_h = {0,  16,  25,  35,  41,  44,  48,  52,  56,  61,  63, 66,
                            67, 69,  74,  76,  78,  80,  82,  84,  87,  89,  91, 93,
                            98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156};
  std::vector<int> ind_h = {
    1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 0,  2,  3,  7,  13, 17, 19,
    21, 30, 0,  1,  3,  7,  8,  9,  13, 27, 28, 32, 0,  1,  2,  7,  12, 13, 0,  6,  10, 0,  6,
    10, 16, 0,  4,  5,  16, 0,  1,  2,  3,  0,  2,  30, 32, 33, 2,  33, 0,  4,  5,  0,  0,  3,
    0,  1,  2,  3,  33, 32, 33, 32, 33, 5,  6,  0,  1,  32, 33, 0,  1,  33, 32, 33, 0,  1,  32,
    33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31, 29, 33, 2,  23, 24, 33, 2,  31, 33, 23, 26,
    32, 33, 1,  8,  32, 33, 0,  24, 25, 28, 32, 33, 2,  8,  14, 15, 18, 20, 22, 23, 29, 30, 31,
    33, 8,  9,  13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  int num_verts = off_h.size() - 1;
  int num_edges = ind_h.size();

  std::vector<int> cluster_id(num_verts, -1);

  rmm::device_vector<int> offsets_v(off_h);
  rmm::device_vector<int> indices_v(ind_h);
  rmm::device_vector<float> weights_v(w_h);
  rmm::device_vector<int> result_v(cluster_id);

  cugraph::experimental::GraphCSRView<int, int, float> G(
    offsets_v.data().get(), indices_v.data().get(), weights_v.data().get(), num_verts, num_edges);

  float modularity{0.0};
  int num_level = 40;

  cugraph::nvgraph::louvain(G, &modularity, &num_level, result_v.data().get());

  cudaMemcpy((void*)&(cluster_id[0]),
             result_v.data().get(),
             sizeof(int) * num_verts,
             cudaMemcpyDeviceToHost);
  int min = *min_element(cluster_id.begin(), cluster_id.end());

  ASSERT_TRUE(min >= 0);
  ASSERT_TRUE(modularity >= 0.402777);
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

  ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, nvgraphLouvain (CUDA_R_32I, CUDA_R_32F, vertices, edges,
G.adjList->offsets->data, G.adjList->indices->data, G.adjList->edge_data->data, weighted,
has_init_cluster, nullptr, (void*) &modularity, (void*) best_cluster_vec, (void *)(&num_level)));


  std::vector<int> cluster_id (vertices, -1);
  cudaMemcpy ((void*) &(cluster_id[0]), best_cluster_vec, sizeof(int)*vertices,
cudaMemcpyDeviceToHost); int max = *max_element (cluster_id.begin(), cluster_id.end()); int min =
*min_element (cluster_id.begin(), cluster_id.end());

  ASSERT_EQ((min >= 0), 1);
  ASSERT_EQ((modularity >= 0.002875), 1);

  ALLOC_FREE_TRY (best_cluster_vec, stream);
  ALLOC_FREE_TRY(col_src.data, stream);
  ALLOC_FREE_TRY(col_dest.data, stream);
  ALLOC_FREE_TRY(col_weights.data, stream);

}
*/
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}
