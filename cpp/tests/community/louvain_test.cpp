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

#include <community/louvain_kernels.hpp>

#include <thrust/extrema.h>

#include <rmm/thrust_rmm_allocator.h>

#include <rmm/mr/device/cnmem_memory_resource.hpp>

TEST(louvain, success)
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

  cugraph::GraphCSRView<int, int, float> G(
    offsets_v.data().get(), indices_v.data().get(), weights_v.data().get(), num_verts, num_edges);

  float modularity{0.0};
  int num_level = 40;

  cugraph::louvain(G, &modularity, &num_level, result_v.data().get());

  cudaMemcpy((void*)&(cluster_id[0]),
             result_v.data().get(),
             sizeof(int) * num_verts,
             cudaMemcpyDeviceToHost);

  int min = *min_element(cluster_id.begin(), cluster_id.end());

  ASSERT_GE(min, 0);
  ASSERT_GE(modularity, 0.402777 * 0.95);
}

TEST(louvain_modularity, simple) {
  std::vector<int> off_h     = {0, 1, 4, 7, 10, 11, 12};
  std::vector<int> src_ind_h = { 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5 };
  std::vector<int> ind_h     = { 1, 0, 2, 3, 1, 3, 4, 1, 2, 5, 2, 3 };
  std::vector<float> w_h     = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  std::vector<float> v_weights_h = { 1.0, 3.0, 3.0, 3.0, 1.0, 1.0 };

  //
  //  Initial cluster, everything on its own
  //
  std::vector<int> cluster_h = { 0, 1, 2, 3, 4, 5 };
  std::vector<float> cluster_weights_h = { 1.0, 3.0, 3.0, 3.0, 1.0, 1.0 };

  std::vector<int> cluster_hash_h = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  std::vector<float> delta_Q_h = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0 };
  std::vector<float> tmp_size_V_h = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };


  int num_verts = off_h.size() - 1;
  int num_edges = ind_h.size();

  float q{0.0};

  rmm::device_vector<int> offsets_v(off_h);
  rmm::device_vector<int> src_indices_v(src_ind_h);
  rmm::device_vector<int> indices_v(ind_h);
  rmm::device_vector<float> weights_v(w_h);
  rmm::device_vector<float> vertex_weights_v(v_weights_h);
  rmm::device_vector<int> cluster_v(cluster_h);
  rmm::device_vector<float> cluster_weights_v(cluster_weights_h);
  rmm::device_vector<int> cluster_hash_v(cluster_hash_h);
  rmm::device_vector<float> delta_Q_v(delta_Q_h);
  rmm::device_vector<float> tmp_size_V_v(tmp_size_V_h);


  cudaStream_t stream{0};


  //
  // Create graph
  //
  cugraph::GraphCSRView<int, int, float> G(
    offsets_v.data().get(), indices_v.data().get(), weights_v.data().get(), num_verts, num_edges);

  q = cugraph::detail::modularity(float{12}, float{1}, G, cluster_v.data().get());

  ASSERT_FLOAT_EQ(q, float{-30.0 / 144.0});

  cugraph::detail::compute_delta_modularity(float{12}, 
                                            float{1},
                                            G,
                                            src_indices_v,
                                            vertex_weights_v,
                                            cluster_weights_v,
                                            cluster_v,
                                            cluster_hash_v,
                                            delta_Q_v,
                                            tmp_size_V_v);


  CUDA_TRY(cudaMemcpy(cluster_hash_h.data(), cluster_hash_v.data().get(), sizeof(int) * num_edges, cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(delta_Q_h.data(), delta_Q_v.data().get(), sizeof(float) * num_edges, cudaMemcpyDeviceToHost));

  ASSERT_EQ(cluster_hash_h[0], 1);
  ASSERT_EQ(cluster_hash_h[10], 2);
  ASSERT_EQ(cluster_hash_h[11], 3);
  ASSERT_FLOAT_EQ(delta_Q_h[0], float{1.0 / 8.0});
  ASSERT_FLOAT_EQ(delta_Q_h[10], float{1.0 / 8.0});
  ASSERT_FLOAT_EQ(delta_Q_h[11], float{1.0 / 8.0});

  //
  //  Move vertex 0 into cluster 1
  //
  cluster_h[0] = 1;
  cluster_weights_h[0] = 0.0;
  cluster_weights_h[1] = 4.0;

  CUDA_TRY(cudaMemcpy(cluster_v.data().get(), cluster_h.data(), sizeof(int) * num_verts, cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(cluster_weights_v.data().get(), cluster_weights_h.data(), sizeof(float) * num_verts, cudaMemcpyHostToDevice));

  q = cugraph::detail::modularity(float{12}, float{1}, G, cluster_v.data().get());

  ASSERT_FLOAT_EQ(q, float{-12.0 / 144.0});
  
  cugraph::detail::compute_delta_modularity(float{12}, 
                                            float{1},
                                            G,
                                            src_indices_v,
                                            vertex_weights_v,
                                            cluster_weights_v,
                                            cluster_v,
                                            cluster_hash_v,
                                            delta_Q_v,
                                            tmp_size_V_v);

  CUDA_TRY(cudaMemcpy(cluster_hash_h.data(), cluster_hash_v.data().get(), sizeof(int) * num_edges, cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(delta_Q_h.data(), delta_Q_v.data().get(), sizeof(float) * num_edges, cudaMemcpyDeviceToHost));

  ASSERT_EQ(cluster_hash_h[10], 2);
  ASSERT_EQ(cluster_hash_h[11], 3);
  ASSERT_FLOAT_EQ(delta_Q_h[10], float{1.0 / 8.0});
  ASSERT_FLOAT_EQ(delta_Q_h[11], float{1.0 / 8.0});

  //
  //  Move vertex 1 into cluster 2.  Not the optimal, in fact it will reduce
  //  modularity (so Louvain would never do this), but let's see if it reduces
  //  by the expected amount (-12/144).
  //
  ASSERT_EQ(cluster_hash_h[3], 2);
  ASSERT_FLOAT_EQ(delta_Q_h[3], float{-12.0 / 144.0});
  
  cluster_h[1] = 2;
  cluster_weights_h[1] = 1.0;
  cluster_weights_h[2] = 6.0;

  CUDA_TRY(cudaMemcpy(cluster_v.data().get(), cluster_h.data(), sizeof(int) * num_verts, cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(cluster_weights_v.data().get(), cluster_weights_h.data(), sizeof(float) * num_verts, cudaMemcpyHostToDevice));

  q = cugraph::detail::modularity(float{12}, float{1}, G, cluster_v.data().get());

  ASSERT_FLOAT_EQ(q, float{-24.0 / 144.0});
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cnmem_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}
