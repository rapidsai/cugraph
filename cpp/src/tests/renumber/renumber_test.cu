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

  //cugraph::detail::renumber_vertices(length, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(8191), thrust::less<uint32_t>());
  cugraph::detail::renumber_vertices(length, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(511), thrust::less<uint32_t>());

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

TEST_F(RenumberingTest, SmallFixedVertexListNegative)
{
  int64_t src_data[] = { 4,  6,  8, -20,  1 };
  int64_t dst_data[] = { 1, 29, 35,   0, 77 };

  int64_t src_expected[] = { 2, 3, 4, 8, 1 };
  int64_t dst_expected[] = { 1, 5, 6, 0, 7 };

  size_t length = sizeof(src_data) / sizeof(src_data[0]);

  int64_t *src_d;
  int64_t *dst_d;
  int64_t *number_map_d;

  int64_t tmp_results[length];
  int64_t tmp_map[2 * length];

  cudaStream_t stream{nullptr};

  EXPECT_EQ(RMM_ALLOC(&src_d, sizeof(int64_t) * length, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_ALLOC(&dst_d, sizeof(int64_t) * length, stream), RMM_SUCCESS);

  EXPECT_EQ(cudaMemcpy(src_d, src_data, sizeof(int64_t) * length, cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dst_d, dst_data, sizeof(int64_t) * length, cudaMemcpyHostToDevice), cudaSuccess);

  size_t unique_verts = 0;

  cugraph::detail::renumber_vertices(length, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(511), thrust::less<int64_t>());


  EXPECT_EQ(cudaMemcpy(tmp_map, number_map_d, sizeof(int64_t) * unique_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(tmp_results, src_d, sizeof(int64_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);

  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], src_expected[i]);
    EXPECT_EQ(tmp_map[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_d, sizeof(int64_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);
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

  //cugraph::detail::renumber_vertices(length, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(8191), thrust::less<uint64_t>());
  cugraph::detail::renumber_vertices(length, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(511), thrust::less<uint64_t>());

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

TEST_F(RenumberingTest, SmallFixedVertexListString)
{
  const char * src_data[] = { "4U",  "6U",  "8U", "20U",  "1U" };
  const char * dst_data[] = { "1U", "29U", "35U",  "0U", "77U" };

  int32_t src_expected[] = { 5, 3, 2, 0, 1 };
  int32_t dst_expected[] = { 1, 8, 4, 7, 6 };

  size_t length = sizeof(src_data) / sizeof(src_data[0]);

  NVStrings *srcs = NVStrings::create_from_array(src_data, length);
  NVStrings *dsts = NVStrings::create_from_array(dst_data, length);

  cudaStream_t stream{nullptr};

  thrust::pair<const char *, size_t> *src_d;
  thrust::pair<const char *, size_t> *dst_d;
  thrust::pair<const char *, size_t> *output_map;
  int32_t *src_output_d;
  int32_t *dst_output_d;
  size_t unique_verts = 0;
  int32_t tmp_results[length];
  thrust::pair<const char *, size_t> tmp_map[2 * length];
  thrust::pair<const char *, size_t> tmp_compare[length];

  ALLOC_TRY((void**) &src_d, sizeof(thrust::pair<const char *, size_t>) * length, stream);
  ALLOC_TRY((void**) &dst_d, sizeof(thrust::pair<const char *, size_t>) * length, stream);
  ALLOC_TRY((void**) &src_output_d, sizeof(int32_t) * length, stream);
  ALLOC_TRY((void**) &dst_output_d, sizeof(int32_t) * length, stream);

  srcs->create_index((std::pair<const char *, size_t> *) src_d, true);
  dsts->create_index((std::pair<const char *, size_t> *) dst_d, true);

  cugraph::detail::renumber_vertices(length,
				       src_d,
				       dst_d,
				       src_output_d,
				       dst_output_d,
				       &unique_verts,
				       &output_map,
				       cugraph::detail::HashFunctionObjectString(7),
				       cugraph::detail::CompareString());

  //
  //  Bring output_map back as local_strings so we can do comparisons
  //
  NVStrings *omap = NVStrings::create_from_index((std::pair<const char *, size_t> *) output_map, unique_verts);

  int maxStringLen = 4;
  char local_buffer[unique_verts * maxStringLen];
  char *local_strings[unique_verts];
  memset(local_buffer, 0, unique_verts * maxStringLen);

  local_strings[0] = local_buffer;
  for (size_t i = 1 ; i < unique_verts ; ++i)
    local_strings[i] = local_strings[i-1] + maxStringLen;

  EXPECT_EQ(omap->to_host(local_strings, 0, unique_verts), 0);


  //
  //  Now, bring back results and compare them
  //
  EXPECT_EQ(cudaMemcpy(tmp_map, output_map, sizeof(thrust::pair<const char *, size_t>) * unique_verts, cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(cudaMemcpy(tmp_results, src_output_d, sizeof(int32_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(tmp_compare, src_d, sizeof(thrust::pair<const char *, size_t>) * length, cudaMemcpyDeviceToHost), cudaSuccess);

  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], src_expected[i]);
    EXPECT_STREQ(local_strings[tmp_results[i]], src_data[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_output_d, sizeof(int32_t) * length, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(tmp_compare, dst_d, sizeof(thrust::pair<const char *, size_t>) * length, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0 ; i < length ; ++i) {
    EXPECT_EQ(tmp_results[i], dst_expected[i]);
    EXPECT_STREQ(local_strings[tmp_results[i]], dst_data[i]);
  }

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(src_output_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_output_d, stream), RMM_SUCCESS);

  NVStrings::destroy(omap);
  NVStrings::destroy(srcs);
  NVStrings::destroy(dsts);
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

  //cugraph::detail::renumber_vertices(length, src_d, dst_d, src_renumbered_d, dst_renumbered_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(8191), thrust::less<uint64_t>());
  cugraph::detail::renumber_vertices(length, src_d, dst_d, src_renumbered_d, dst_renumbered_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(511), thrust::less<uint64_t>());

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

  //cugraph::detail::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(8191), thrust::less<uint64_t>());
  cugraph::detail::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(511), thrust::less<uint64_t>());

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

  EXPECT_EQ(min_id, 0);
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
  //const int hash_size =  8191;      // 224 ms
  const int hash_size =  511;      // 224 ms

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
  cugraph::detail::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(hash_size), thrust::less<uint64_t>());
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
  std::cout << "  unique verts = " << unique_verts << std::endl;
  std::cout << "  hash size = " << hash_size << std::endl;

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
}

TEST_F(RenumberingTest, Random10MVertexListString)
{
  const int num_verts = 10000000;
  //const int hash_size = 32768;
  const int hash_size = 65536;

  uint32_t *src_d;
  uint32_t *dst_d;

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

  uint32_t *src = new uint32_t[num_verts];
  uint32_t *dst = new uint32_t[num_verts];

  EXPECT_EQ(cudaMemcpy(src, src_d, sizeof(uint32_t) * num_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dst, dst_d, sizeof(uint32_t) * num_verts, cudaMemcpyDeviceToHost), cudaSuccess);

  //
  //  Now we want to convert integers to strings
  //
  NVStrings *srcs = NVStrings::itos((int *) src_d, num_verts, nullptr, true);
  NVStrings *dsts = NVStrings::itos((int *) dst_d, num_verts, nullptr, true);

  thrust::pair<const char *, size_t> *src_pair_d;
  thrust::pair<const char *, size_t> *dst_pair_d;
  thrust::pair<const char *, size_t> *output_map;
  int32_t *src_output_d;
  int32_t *dst_output_d;
  size_t unique_verts = 0;

  std::cout << "done with initialization" << std::endl;

  int32_t *tmp_results = new int32_t[num_verts];
  thrust::pair<const char *, size_t> *tmp_map = new thrust::pair<const char *, size_t>[2 * num_verts];
  thrust::pair<const char *, size_t> *tmp_compare = new thrust::pair<const char *, size_t>[num_verts];

  ALLOC_TRY((void**) &src_pair_d, sizeof(thrust::pair<const char *, size_t>) * num_verts, stream);
  ALLOC_TRY((void**) &dst_pair_d, sizeof(thrust::pair<const char *, size_t>) * num_verts, stream);
  ALLOC_TRY((void**) &src_output_d, sizeof(int32_t) * num_verts, stream);
  ALLOC_TRY((void**) &dst_output_d, sizeof(int32_t) * num_verts, stream);

  srcs->create_index((std::pair<const char *, size_t> *) src_pair_d, true);
  dsts->create_index((std::pair<const char *, size_t> *) dst_pair_d, true);

  auto start = std::chrono::system_clock::now();


  cugraph::detail::renumber_vertices(num_verts,
				       src_pair_d,
				       dst_pair_d,
				       src_output_d,
				       dst_output_d,
				       &unique_verts,
				       &output_map,
				       cugraph::detail::HashFunctionObjectString(hash_size),
				       cugraph::detail::CompareString());

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
  std::cout << "  unique verts = " << unique_verts << std::endl;
  std::cout << "  hash size = " << hash_size << std::endl;

  //
  //  Bring output_map back as local_strings so we can do comparisons
  //
  NVStrings *omap = NVStrings::create_from_index((std::pair<const char *, size_t> *) output_map, unique_verts);

  //  12 bytes (minimum int32 is -2147483648, need room for a null byte)
  //
  //  Create a local string buffer and then populate it.  There ought to
  //  be a good way for NVStrings library to do this exactly rather than
  //  approximating and wasting space like this.
  //
  int maxStringLen = 12;
  char *local_buffer = new char[unique_verts * maxStringLen];
  char **local_strings = new char *[unique_verts];

  memset(local_buffer, 0, unique_verts * maxStringLen);

  local_strings[0] = local_buffer;
  for (size_t i = 1 ; i < unique_verts ; ++i)
    local_strings[i] = local_strings[i-1] + maxStringLen;

  EXPECT_EQ(omap->to_host(local_strings, 0, unique_verts), 0);

  cudaDeviceSynchronize();
  CUDA_CHECK_LAST();

  printf("checking results\n");

  //
  //  Now, bring back results and compare them
  //
  EXPECT_EQ(cudaMemcpy(tmp_map, output_map, sizeof(thrust::pair<const char *, size_t>) * unique_verts, cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(cudaMemcpy(tmp_results, src_output_d, sizeof(int32_t) * num_verts, cudaMemcpyDeviceToHost), cudaSuccess);

  for (size_t i = 0 ; i < num_verts ; ++i) {
    uint32_t vid = 0;
    sscanf(local_strings[tmp_results[i]], "%u", &vid);
    EXPECT_EQ(vid, src[i]);
  }

  EXPECT_EQ(cudaMemcpy(tmp_results, dst_output_d, sizeof(int32_t) * num_verts, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0 ; i < num_verts ; ++i) {
    uint32_t vid = 0;
    sscanf(local_strings[tmp_results[i]], "%u", &vid);
    EXPECT_EQ(vid, dst[i]);
  }

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(state, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(src_pair_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_pair_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(src_output_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_output_d, stream), RMM_SUCCESS);

  NVStrings::destroy(omap);
  NVStrings::destroy(srcs);
  NVStrings::destroy(dsts);

  delete [] local_strings;
  delete [] local_buffer;
  delete [] tmp_results;
  delete [] tmp_map;
  delete [] tmp_compare;
  delete [] src;
  delete [] dst;
}

TEST_F(RenumberingTest, Random100MVertexSet)
{
  const int num_verts = 100000000;

  //  A sampling of performance on single Quadro GV100
  //const int hash_size =  8192;        // 1811 ms
  //const int hash_size =  16384;       // 1746 ms
  //const int hash_size =  32768;       // 1662 ms
  //const int hash_size =  65536;       // 1569 ms
  //const int hash_size =  16777216;      // 1328 ms
  const int hash_size = 511;

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
  cugraph::detail::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(hash_size), thrust::less<uint64_t>());
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
  std::cout << "  unique verts = " << unique_verts << std::endl;
  std::cout << "  hash size = " << hash_size << std::endl;

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
}

TEST_F(RenumberingTest, Random500MVertexSet)
{
  const int num_verts = 500000000;

  //  A sampling of performance on single Quadro GV100
  //const int hash_size =  8192;      // 9918 ms
  //const int hash_size =  16384;      // 9550 ms
  //const int hash_size =  32768;      // 9146 ms
  //const int hash_size =  131072;      // 8537 ms
  const int hash_size =  1048576;      // 7335 ms
  //const int hash_size =  511;      // 7335 ms

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
  cugraph::detail::renumber_vertices(num_verts, src_d, dst_d, src_d, dst_d, &unique_verts, &number_map_d, cugraph::detail::HashFunctionObjectInt(hash_size), thrust::less<uint64_t>());
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Renumber kernel elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
  std::cout << "  unique verts = " << unique_verts << std::endl;
  std::cout << "  hash size = " << hash_size << std::endl;

  EXPECT_EQ(RMM_FREE(src_d, stream), RMM_SUCCESS);
  EXPECT_EQ(RMM_FREE(dst_d, stream), RMM_SUCCESS);
  EXPECT_EQ(test_free(number_map_d), cudaSuccess);
}

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}