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
#include <utilities/base_fixture.hpp>

#include <algorithms.hpp>
#include <graph.hpp>

#include <thrust/extrema.h>

#include <rmm/thrust_rmm_allocator.h>

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

  std::vector<int> result_h = {1, 1, 1, 1, 0, 0, 0, 1, 3, 1, 0, 1, 1, 1, 3, 3, 0,
                               1, 3, 1, 3, 1, 3, 2, 2, 2, 3, 2, 1, 3, 3, 2, 3, 3};

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
  size_t num_level = 40;

  raft::handle_t handle;

  std::tie(num_level, modularity) = cugraph::louvain(handle, G, result_v.data().get());

  cudaMemcpy((void*)&(cluster_id[0]),
             result_v.data().get(),
             sizeof(int) * num_verts,
             cudaMemcpyDeviceToHost);

  int min = *min_element(cluster_id.begin(), cluster_id.end());

  std::cout << "modularity = " << modularity << std::endl;

  ASSERT_GE(min, 0);
  ASSERT_GE(modularity, 0.402777 * 0.95);
  ASSERT_EQ(result_v, result_h);
}

TEST(louvain_renumbered, success)
{
  std::vector<int> off_h = {0,   16,  25,  30,  34,  38,  42,  44,  46,  48,  50,  52,
                            54,  56,  73,  85,  95,  101, 107, 112, 117, 121, 125, 129,
                            132, 135, 138, 141, 144, 147, 149, 151, 153, 155, 156

  };
  std::vector<int> ind_h = {
    1,  3,  7,  11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 30, 33, 0,  5,  11, 15, 16, 19, 21,
    25, 30, 4,  13, 14, 22, 27, 0,  9,  20, 24, 2,  13, 15, 26, 1,  13, 14, 18, 13, 15, 0,  16,
    13, 14, 3,  20, 13, 14, 0,  1,  13, 22, 2,  4,  5,  6,  8,  10, 12, 14, 17, 18, 19, 22, 25,
    28, 29, 31, 32, 2,  5,  8,  10, 13, 15, 17, 18, 22, 29, 31, 32, 0,  1,  4,  6,  14, 16, 18,
    19, 21, 28, 0,  1,  7,  15, 19, 21, 0,  13, 14, 26, 27, 28, 0,  5,  13, 14, 15, 0,  1,  13,
    16, 16, 0,  3,  9,  23, 0,  1,  15, 16, 2,  12, 13, 14, 0,  20, 24, 0,  3,  23, 0,  1,  13,
    4,  17, 27, 2,  17, 26, 13, 15, 17, 13, 14, 0,  1,  13, 14, 13, 14, 0};

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
  size_t num_level = 40;

  raft::handle_t handle;

  std::tie(num_level, modularity) = cugraph::louvain(handle, G, result_v.data().get());

  cudaMemcpy((void*)&(cluster_id[0]),
             result_v.data().get(),
             sizeof(int) * num_verts,
             cudaMemcpyDeviceToHost);

  int min = *min_element(cluster_id.begin(), cluster_id.end());

  std::cout << "modularity = " << modularity << std::endl;

  ASSERT_GE(min, 0);
  ASSERT_GE(modularity, 0.402777 * 0.95);
}

CUGRAPH_TEST_PROGRAM_MAIN()
