/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "cuda_profiler_api.h"

#include "renumber.cuh"

#include <chrono>


struct RenumberingTest : public ::testing::Test
{
};

__global__ void display_list(const char *label, uint32_t *verts, size_t length) {

  printf("%s\n", label);

  for (size_t i = 0 ; i < length ; ++i) {
    printf("  %lu\n", verts[i]);
  }
}

TEST_F(RenumberingTest, SmallFixedVertexList)
{
  uint32_t src_data[] = { 4U,  6U,  8U, 20U,  1U };
  uint32_t dst_data[] = { 1U, 29U, 35U,  0U, 77U };

  uint32_t src_expected[] = { 2U, 3U, 4U, 5U, 1U };
  uint32_t dst_expected[] = { 1U, 6U, 7U, 0U, 8U };

  size_t length = sizeof(src_data) / sizeof(src_data[0]);

  uint32_t *src_d;
  uint32_t *dst_d;
  uint32_t *number_map_d;

  uint32_t tmp_results[length];
  uint32_t tmp_map[2 * length];

  cudaMalloc(&src_d, sizeof(uint32_t) * length);
  cudaMalloc(&dst_d, sizeof(uint32_t) * length);
  cudaMalloc(&number_map_d, 2 * sizeof(uint32_t) * length);

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint32_t) * length, cudaMemcpyHostToDevice), GDF_SUCCESS);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint32_t) * length, cudaMemcpyHostToDevice), GDF_SUCCESS);

  cugraph::renumber_vertices(length, src_d, dst_d, src_d, dst_d, number_map_d);

  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, 2 * sizeof(uint32_t) * length, cudaMemcpyDeviceToHost), GDF_SUCCESS);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_d, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost), GDF_SUCCESS);

  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], src_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_d, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost), GDF_SUCCESS);
  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], dst_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], dst_data[i]);
  }

  cudaFree(src_d);
  cudaFree(dst_d);
  cudaFree(number_map_d);
}

TEST_F(RenumberingTest, SmallFixedVertexList64Bit)
{
  uint64_t src_data[] = { 4U,  6U,  8U, 20U,  1U };
  uint64_t dst_data[] = { 1U, 29U, 35U,  0U, 77U };

  uint64_t src_expected[] = { 2U, 3U, 4U, 5U, 1U };
  uint64_t dst_expected[] = { 1U, 6U, 7U, 0U, 8U };

  size_t length = sizeof(src_data) / sizeof(src_data[0]);

  uint64_t *src_d;
  uint64_t *dst_d;
  uint64_t *number_map_d;

  uint64_t tmp_results[length];
  uint64_t tmp_map[2 * length];

  cudaMalloc(&src_d, sizeof(uint64_t) * length);
  cudaMalloc(&dst_d, sizeof(uint64_t) * length);
  cudaMalloc(&number_map_d, 2 * sizeof(uint64_t) * length);

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint64_t) * length, cudaMemcpyHostToDevice), GDF_SUCCESS);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint64_t) * length, cudaMemcpyHostToDevice), GDF_SUCCESS);

  cugraph::renumber_vertices(length, src_d, dst_d, src_d, dst_d, number_map_d);

  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, 2 * sizeof(uint64_t) * length, cudaMemcpyDeviceToHost), GDF_SUCCESS);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_d, sizeof(uint64_t) * length, cudaMemcpyDeviceToHost), GDF_SUCCESS);

  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], src_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_d, sizeof(uint64_t) * length, cudaMemcpyDeviceToHost), GDF_SUCCESS);
  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], dst_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], dst_data[i]);
  }

  cudaFree(src_d);
  cudaFree(dst_d);
  cudaFree(number_map_d);
}

TEST_F(RenumberingTest, Random100KVertexSet)
{
  const int num_verts = 100000;

  uint64_t *src_d;
  uint64_t *dst_d;
  uint64_t *number_map_d;

  uint64_t *src_data    = (uint64_t *) malloc(num_verts * sizeof(uint64_t));
  uint64_t *dst_data    = (uint64_t *) malloc(num_verts * sizeof(uint64_t));
  uint64_t *tmp_results = (uint64_t *) malloc(num_verts * sizeof(uint64_t));
  uint64_t *tmp_map     = (uint64_t *) malloc(2 * num_verts * sizeof(uint64_t));

  EXPECT_EQ(cudaMalloc(&src_d, sizeof(uint64_t) * num_verts), GDF_SUCCESS);
  EXPECT_EQ(cudaMalloc(&dst_d, sizeof(uint64_t) * num_verts), GDF_SUCCESS);
  EXPECT_EQ(cudaMalloc(&number_map_d, 2 * sizeof(uint64_t) * num_verts), GDF_SUCCESS);

  //
  //  Generate random source and vertex values
  //
  srand(43);

  for (int i = 0 ; i < num_verts ; ++i) {
    src_data[i] = (uint64_t) rand();
  }

  for (int i = 0 ; i < num_verts ; ++i) {
    dst_data[i] = (uint64_t) rand();
  }

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint64_t) * num_verts, cudaMemcpyHostToDevice), GDF_SUCCESS);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint64_t) * num_verts, cudaMemcpyHostToDevice), GDF_SUCCESS);

  //
  //  Renumber everything
  //
  auto start = std::chrono::system_clock::now();
  cugraph::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, number_map_d);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;


  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, 2 * sizeof(uint64_t) * num_verts, cudaMemcpyDeviceToHost), GDF_SUCCESS);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_d, sizeof(uint64_t) * num_verts, cudaMemcpyDeviceToHost), GDF_SUCCESS);

  for (size_t i = 0 ; i < num_verts ; ++i) {
    EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_d, sizeof(uint64_t) * num_verts, cudaMemcpyDeviceToHost), GDF_SUCCESS);
  for (size_t i = 0 ; i < num_verts ; ++i) {
    EXPECT_EQ(tmp_map[tmp_results[i]], dst_data[i]);
  }

  cudaFree(src_d);
  cudaFree(dst_d);
  cudaFree(number_map_d);
}

TEST_F(RenumberingTest, Random2MVertexSet)
{
  const int num_verts = 2000000;
  //  A sampling of performance on aschaffer-DGX-Station
  //const int hash_size =  33554467;  // 6192 ms
  //const int hash_size =  3355453;   // 4867 ms
  //const int hash_size =  335557;    // 3966 ms
  const int hash_size =  32767;     // 3943 ms
  //const int hash_size =  8191;      // 4211 ms

  uint64_t *src_d;
  uint64_t *dst_d;
  uint64_t *number_map_d;

  uint64_t *src_data    = (uint64_t *) malloc(num_verts * sizeof(uint64_t));
  uint64_t *dst_data    = (uint64_t *) malloc(num_verts * sizeof(uint64_t));

  EXPECT_EQ(cudaMalloc(&src_d, sizeof(uint64_t) * num_verts), GDF_SUCCESS);
  EXPECT_EQ(cudaMalloc(&dst_d, sizeof(uint64_t) * num_verts), GDF_SUCCESS);
  EXPECT_EQ(cudaMalloc(&number_map_d, 2 * sizeof(uint64_t) * num_verts), GDF_SUCCESS);

  //
  //  Generate random source and vertex values
  //
  srand(43);

  for (int i = 0 ; i < num_verts ; ++i) {
    src_data[i] = (uint64_t) rand();
  }

  for (int i = 0 ; i < num_verts ; ++i) {
    dst_data[i] = (uint64_t) rand();
  }

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint64_t) * num_verts, cudaMemcpyHostToDevice), GDF_SUCCESS);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint64_t) * num_verts, cudaMemcpyHostToDevice), GDF_SUCCESS);

  //
  //  Renumber everything
  //
  auto start = std::chrono::system_clock::now();
  cugraph::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, number_map_d, hash_size);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;

  cudaFree(src_d);
  cudaFree(dst_d);
  cudaFree(number_map_d);
}
