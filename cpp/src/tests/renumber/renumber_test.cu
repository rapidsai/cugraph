// -*-c++-*-

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

#include "converters/renumber.cuh"
#include "rmm_utils.h"

#include <chrono>

#include <curand_kernel.h>


struct RenumberingTest : public ::testing::Test
{
};

__global__ void display_list(const char *label, uint32_t *verts, size_t length) {

  printf("%s\n", label);

  for (size_t i = 0 ; i < length ; ++i) {
    printf("  %lu\n", verts[i]);
  }
}

__global__ void setup_generator(curandState *state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(43, id, 0, &state[id]);
}

__global__ void generate_sources(curandState *state, int n, uint32_t *verts) {
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  curandState local_state = state[first];
  for (int id = first ; id < n ; id += stride) {
    verts[id] = curand(&local_state);
  }

  state[first] = local_state;
}
  
__global__ void generate_destinations(curandState *state, int n, const uint32_t *sources, uint32_t *destinations) {
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  curandState local_state = state[first];
  for (int id = first ; id < n ; id += stride) {
    destinations[id] = sources[curand(&local_state) % n];
  }

  state[first] = local_state;
}

cudaError_t test_free(void *ptr) {
  ALLOC_FREE_TRY(ptr, nullptr);
  return cudaSuccess;
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

  cudaStream_t stream{nullptr};

  EXPECT_EQ(RMM_ALLOC(&src_d, sizeof(uint32_t) * length, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_d, sizeof(uint32_t) * length, stream), RMM_SUCCESS);

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint32_t) * length, cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint32_t) * length, cudaMemcpyHostToDevice), cudaSuccess);

  size_t unique_verts = 0;
  EXPECT_EQ(cugraph::renumber_vertices(length, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d), GDF_SUCCESS);

  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, sizeof(uint32_t) * unique_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_d, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);

  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], src_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_d, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], dst_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], dst_data[i]);
  }

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
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

  cudaStream_t stream{nullptr};

  EXPECT_EQ(RMM_ALLOC(&src_d, sizeof(uint64_t) * length, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_d, sizeof(uint64_t) * length, stream), RMM_SUCCESS);

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint64_t) * length, cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint64_t) * length, cudaMemcpyHostToDevice), cudaSuccess);

  size_t unique_verts = 0;
  EXPECT_EQ(cugraph::renumber_vertices(length, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d), GDF_SUCCESS);

  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, sizeof(uint64_t) * unique_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_d, sizeof(uint64_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);

  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], src_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_d, sizeof(uint64_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], dst_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], dst_data[i]);
  }

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
}

TEST_F(RenumberingTest, SmallFixedVertexList64BitTo32Bit)
{
  uint64_t src_data[] = { 4U,  6U,  8U, 20U,  1U };
  uint64_t dst_data[] = { 1U, 29U, 35U,  0U, 77U };

  uint32_t src_expected[] = { 2U, 3U, 4U, 5U, 1U };
  uint32_t dst_expected[] = { 1U, 6U, 7U, 0U, 8U };

  size_t length = sizeof(src_data) / sizeof(src_data[0]);

  uint64_t *src_d;
  uint64_t *dst_d;
  uint32_t *src_renumbered_d;
  uint32_t *dst_renumbered_d;
  uint64_t *number_map_d;

  uint32_t tmp_results[length];
  uint64_t tmp_map[2 * length];

  cudaStream_t stream{nullptr};

  EXPECT_EQ(RMM_ALLOC(&src_d, sizeof(uint64_t) * length, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_d, sizeof(uint64_t) * length, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&src_renumbered_d, sizeof(uint32_t) * length, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_renumbered_d, sizeof(uint32_t) * length, stream), RMM_SUCCESS);

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint64_t) * length, cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint64_t) * length, cudaMemcpyHostToDevice), cudaSuccess);

  size_t unique_verts = 0;
  EXPECT_EQ(cugraph::renumber_vertices(length, src_d, dst_d, src_renumbered_d, dst_renumbered_d, &unique_verts, &number_map_d), GDF_SUCCESS);

  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, sizeof(uint64_t) * unique_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_renumbered_d, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);

  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], src_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_renumbered_d, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], dst_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], dst_data[i]);
  }

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
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

  cudaStream_t stream{nullptr};

  EXPECT_EQ(RMM_ALLOC(&src_d, sizeof(uint64_t) * num_verts, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_d, sizeof(uint64_t) * num_verts, stream), RMM_SUCCESS);

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

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(uint64_t) * num_verts, cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(uint64_t) * num_verts, cudaMemcpyHostToDevice), cudaSuccess);

  //
  //  Renumber everything
  //
  size_t unique_verts = 0;

  auto start = std::chrono::system_clock::now();
  EXPECT_EQ(cugraph::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d), GDF_SUCCESS);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;


  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, sizeof(uint64_t) * unique_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_d, sizeof(uint64_t) * num_verts, cudaMemcpyDeviceToHost), cudaSuccess);

  size_t min_id = unique_verts;
  size_t max_id = 0;

  size_t cnt = 0;
  for (size_t i = 0 ; i < num_verts ; ++i) {
    min_id = min(min_id, tmp_results[i]);
    max_id = max(max_id, tmp_results[i]);
    if (tmp_map[tmp_results[i]] != src_data[i])
      ++cnt;

    if (cnt < 20)
      EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  if (cnt > 0)
    printf("  src error count = %ld out of %d\n", cnt, num_verts);

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_d, sizeof(uint64_t) * num_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0 ; i < num_verts ; ++i) {
    min_id = min(min_id, tmp_results[i]);
    max_id = max(max_id, tmp_results[i]);
    if (tmp_map[tmp_results[i]] != dst_data[i])
      ++cnt;

    if (cnt < 20)
      EXPECT_EQ(tmp_map[tmp_results[i]], dst_data[i]);
  }

  if (cnt > 0)
    printf("  src error count = %ld out of %d\n", cnt, num_verts);

  EXPECT_EQ(min_id, 0u);
  EXPECT_EQ(max_id, (unique_verts - 1));
  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
  free(src_data);
  free(dst_data);
  free(tmp_results);
  free(tmp_map);
}

TEST_F(RenumberingTest, Random10MVertexSet)
{
  const int num_verts = 10000000;

  //  A sampling of performance on single Quadro GV100
  //const int hash_size =  32767;       // 238 ms
  const int hash_size =  8191;      // 224 ms

  uint32_t *src_d;
  uint32_t *dst_d;
  uint32_t *number_map_d;

  cudaStream_t stream{nullptr};

  EXPECT_EQ(RMM_ALLOC(&src_d, sizeof(uint32_t) * num_verts, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_d, sizeof(uint32_t) * num_verts, stream), RMM_SUCCESS);

  //
  //  Init the random number generate
  //
  const int num_threads{64};
  curandState *state;

  EXPECT_EQ(RMM_ALLOC(&state, sizeof(curandState) * num_threads, stream), RMM_SUCCESS);
  setup_generator<<<num_threads,1>>>(state);
  generate_sources<<<num_threads,1>>>(state, num_verts, src_d);
  generate_destinations<<<num_threads,1>>>(state, num_verts, src_d, dst_d);

  std::cout << "done with initialization" << std::endl;

  //
  //  Renumber everything
  //
  size_t unique_verts = 0;
  auto start = std::chrono::system_clock::now();
  EXPECT_EQ(cugraph::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, CUDA_MAX_KERNEL_THREADS, CUDA_MAX_BLOCKS, hash_size), GDF_SUCCESS);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
  std::cout << "  unique verts = " << unique_verts << std::endl;
  std::cout << "  hash size = " << hash_size << std::endl;

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
}

TEST_F(RenumberingTest, Random100MVertexSet)
{
  const int num_verts = 100000000;

  //  A sampling of performance on single Quadro GV100
  const int hash_size =  8192;      // 2833 ms
  //const int hash_size =  16384;      // 2796 ms
  //const int hash_size =  32768;      // 3255 ms

  uint32_t *src_d;
  uint32_t *dst_d;
  uint32_t *number_map_d;

  cudaStream_t stream{nullptr};

  EXPECT_EQ(RMM_ALLOC(&src_d, sizeof(uint32_t) * num_verts, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_d, sizeof(uint32_t) * num_verts, stream), RMM_SUCCESS);

  //
  //  Init the random number generate
  //
  const int num_threads{64};
  curandState *state;

  EXPECT_EQ(RMM_ALLOC(&state, sizeof(curandState) * num_threads, stream), RMM_SUCCESS);
  setup_generator<<<num_threads,1>>>(state);
  generate_sources<<<num_threads,1>>>(state, num_verts, src_d);
  generate_destinations<<<num_threads,1>>>(state, num_verts, src_d, dst_d);

  std::cout << "done with initialization" << std::endl;

  //
  //  Renumber everything
  //
  size_t unique_verts = 0;
  auto start = std::chrono::system_clock::now();
  EXPECT_EQ(cugraph::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, CUDA_MAX_KERNEL_THREADS, CUDA_MAX_BLOCKS, hash_size), GDF_SUCCESS);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
  std::cout << "  unique verts = " << unique_verts << std::endl;
  std::cout << "  hash size = " << hash_size << std::endl;

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
}
