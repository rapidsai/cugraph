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

#include "sort/sort.cuh"
#include "rmm_utils.h"
#include "test_utils.h"

#include <chrono>

#include <curand_kernel.h>

#define MAX_NUM_GPUS 16

struct SortTest : public ::testing::Test
{
};

__global__ void setup_generator(curandState *state, unsigned long long seed = 43) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

template <typename Key_t, int size>
struct RandomKey {
  __inline__ __device__ Key_t operator()(curandState *state) {
    return curand(state);
  }
};

template <typename Key_t>
struct RandomKey<Key_t, 8> {
  __inline__ __device__ Key_t operator()(curandState *state) {
    return (static_cast<Key_t>(curand(state)) << 32) | curand(state);
  }
};

template <typename Key_t>
__global__ void generate_array(curandState *state, int n, Key_t *array) {
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  curandState local_state = state[first];
  RandomKey<Key_t, sizeof(Key_t)> random_key;
  for (int id = first ; id < n ; id += stride) {
    array[id] = random_key(&local_state);
  }

  state[first] = local_state;
}

template <typename Value_t, typename Length_t>
void initialize_values(Value_t *vals, Length_t num_elements, cudaStream_t stream) {
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<int>(0),
                     thrust::make_counting_iterator<int>(num_elements),
                     [vals] __device__ (int idx) {
                       vals[idx] = idx;
                     });
}

template <typename Key_t, typename Value_t, typename Length_t>
void generate_random(Key_t **d_key, Value_t **d_value,
                     Length_t *h_offsets, int num_gpus,
                     int seed, cudaStream_t stream) {

#pragma omp parallel
  {
    int cpu_tid = omp_get_thread_num();
    cudaSetDevice(cpu_tid);

    Length_t num_elements = h_offsets[cpu_tid+1] - h_offsets[cpu_tid];

    EXPECT_EQ(RMM_ALLOC(d_key + cpu_tid, sizeof(Key_t) * num_elements, stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_ALLOC(d_value + cpu_tid, sizeof(Value_t) * num_elements, stream), RMM_SUCCESS);

    //
    //  Init the random number generator
    //
    const int num_threads{64};
    curandState *state;

    EXPECT_EQ(RMM_ALLOC(&state, sizeof(curandState) * num_threads, stream), RMM_SUCCESS);
    setup_generator<<<num_threads,1>>>(state, seed + cpu_tid);

    //
    //  Now generate random data
    //
    generate_array<<<num_threads,1>>>(state, num_elements, d_key[cpu_tid]);

    initialize_values(d_value[cpu_tid], num_elements, stream);
    
    //
    //  Free the state
    //
    EXPECT_EQ(RMM_FREE(state, stream), RMM_SUCCESS);
  }
}

template <typename Key_t, typename Length_t>
void generate_random(Key_t **d_key, Length_t *h_offsets, int num_gpus,
                     int seed, cudaStream_t stream) {

#pragma omp parallel
  {
    int cpu_tid = omp_get_thread_num();
    cudaSetDevice(cpu_tid);

    Length_t num_elements = h_offsets[cpu_tid+1] - h_offsets[cpu_tid];

    EXPECT_EQ(RMM_ALLOC(d_key + cpu_tid, sizeof(Key_t) * num_elements, stream), RMM_SUCCESS);

    //
    //  Init the random number generator
    //
    const int num_threads{64};
    curandState *state;

    EXPECT_EQ(RMM_ALLOC(&state, sizeof(curandState) * num_threads, stream), RMM_SUCCESS);
    setup_generator<<<num_threads,1>>>(state, seed + cpu_tid);

    //
    //  Now generate random data
    //
    generate_array<<<num_threads,1>>>(state, num_elements, d_key[cpu_tid]);

    //
    //  Free the state
    //
    EXPECT_EQ(RMM_FREE(state, stream), RMM_SUCCESS);
  }
}

template <typename Key_t, typename Value_t, typename Length_t>
void verify_sorted_order(Key_t **d_key, Value_t **d_value,
                         Length_t *h_offsets, int num_gpus,
                         cudaStream_t stream, bool verbose = false) {

  Key_t keys_0[num_gpus] = { Key_t{0} };
  Key_t keys_n[num_gpus] = { Key_t{0} };
  
#pragma omp parallel
  {
    int cpu_tid = omp_get_thread_num();
    cudaSetDevice(cpu_tid);

    Length_t length = h_offsets[cpu_tid+1] - h_offsets[cpu_tid];

    if (length > 0) {
      int* diffCounter;
      EXPECT_EQ(RMM_ALLOC(&diffCounter, sizeof(int) * length, stream), RMM_SUCCESS);

      int cpu_tid = omp_get_thread_num();

      Key_t *key = d_key[cpu_tid];

      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        thrust::make_counting_iterator(Length_t{0}),
                        thrust::make_counting_iterator(length),
                        diffCounter,
                        [key, cpu_tid, verbose] __device__ (Length_t v) {
                          if (v > 0) {
                            if (key[v-1] > key[v]) {
                              if (verbose)
                                printf("key[%d] (%016llx) > key[%d] (%016llx)\n",
                                       v-1, (uint64_t) key[v-1], v, (uint64_t) key[v]);

                              return 1;
                            }
                          }
                          return 0;
                        });

      cudaDeviceSynchronize();
      CUDA_CHECK_LAST();

      int result = thrust::reduce(rmm::exec_policy(stream)->on(stream), diffCounter, diffCounter + length, 0);

      EXPECT_EQ(result, 0);
      EXPECT_EQ(RMM_FREE(diffCounter, stream), RMM_SUCCESS);

      cudaMemcpy(keys_0 + cpu_tid, d_key[cpu_tid], sizeof(Key_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(keys_n + cpu_tid, d_key[cpu_tid] + length - 1, sizeof(Key_t), cudaMemcpyDeviceToHost);
    }
  }

  int edge_errors = 0;
  for (int i = 1 ; i < num_gpus ; ++i)
    if (keys_0[i] < keys_n[i-1]) {
      ++edge_errors;
    }

  EXPECT_EQ(edge_errors, 0);
}

template <typename Key_t, typename Length_t>
void verify_sorted_order(Key_t **d_key, Length_t *h_offsets,
                         int num_gpus, cudaStream_t stream,
                         bool verbose = false) {

  Key_t keys_0[num_gpus] = { Key_t{0} };
  Key_t keys_n[num_gpus] = { Key_t{0} };

#pragma omp parallel
  {
    int cpu_tid = omp_get_thread_num();
    cudaSetDevice(cpu_tid);

    Length_t length = h_offsets[cpu_tid+1] - h_offsets[cpu_tid];

    if (length > 0) {
      int* diffCounter;
      EXPECT_EQ(RMM_ALLOC(&diffCounter, sizeof(int) * length, stream), RMM_SUCCESS);

      int cpu_tid = omp_get_thread_num();

      Key_t *key = d_key[cpu_tid];

      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        thrust::make_counting_iterator(Length_t{0}),
                        thrust::make_counting_iterator(length),
                        diffCounter,
                        [key, cpu_tid, verbose] __device__ (Length_t v) {
                          if (v > 0) {
                            if (key[v-1] > key[v]) {
                              if (verbose)
                                printf("key[%d] (%016llx) > key[%d] (%016llx)\n",
                                       v-1, (uint64_t) key[v-1], v, (uint64_t) key[v]);

                              return 1;
                            }
                          }
                          return 0;
                        });

      cudaDeviceSynchronize();
      CUDA_CHECK_LAST();

      int result = thrust::reduce(rmm::exec_policy(stream)->on(stream), diffCounter, diffCounter + length, 0);

      EXPECT_EQ(result, 0);
      EXPECT_EQ(RMM_FREE(diffCounter, stream), RMM_SUCCESS);

      cudaMemcpy(keys_0 + cpu_tid, d_key[cpu_tid], sizeof(Key_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(keys_n + cpu_tid, d_key[cpu_tid] + length - 1, sizeof(Key_t), cudaMemcpyDeviceToHost);
    }
  }

  int edge_errors = 0;
  for (int i = 1 ; i < num_gpus ; ++i)
    if (keys_0[i] < keys_n[i-1]) {
      std::cout << "keys_0[" << i << "] = " << keys_0[i] << std::endl;
      std::cout << "  keys_n[" << (i-1) << "] = " << keys_n[i-1] << std::endl;
      ++edge_errors;
    }

  EXPECT_EQ(edge_errors, 0);
}

TEST_F(SortTest, Random10MPerDevice_uint64_t)
{
  cudaStream_t stream{nullptr};
  
  uint64_t           *d_input[MAX_NUM_GPUS];
  uint64_t           *d_input_values[MAX_NUM_GPUS];
  uint64_t           *d_output[MAX_NUM_GPUS];
  uint64_t           *d_output_values[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 10000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, d_input_values, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key_value(d_input, d_input_values, h_input_offsets,
                         d_output, d_output_values, h_output_offsets,
                         n_gpus);

  verify_sorted_order(d_output, d_output_values, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_input_values[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output_values[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random10MPerDevice_uint32_t)
{
  cudaStream_t stream{nullptr};
  
  uint32_t           *d_input[MAX_NUM_GPUS];
  uint32_t           *d_input_values[MAX_NUM_GPUS];
  uint32_t           *d_output[MAX_NUM_GPUS];
  uint32_t           *d_output_values[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 10000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, d_input_values, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key_value(d_input, d_input_values, h_input_offsets,
                         d_output, d_output_values, h_output_offsets,
                         n_gpus);

  verify_sorted_order(d_output, d_output_values, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_input_values[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output_values[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random100MPerDevice_uint64_t)
{
  cudaStream_t stream{nullptr};
  
  uint64_t           *d_input[MAX_NUM_GPUS];
  uint64_t           *d_input_values[MAX_NUM_GPUS];
  uint64_t           *d_output[MAX_NUM_GPUS];
  uint64_t           *d_output_values[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 100000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, d_input_values, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key_value(d_input, d_input_values, h_input_offsets,
                         d_output, d_output_values, h_output_offsets,
                         n_gpus);

  verify_sorted_order(d_output, d_output_values, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_input_values[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output_values[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random100MPerDevice_uint32_t)
{
  cudaStream_t stream{nullptr};
  
  uint32_t           *d_input[MAX_NUM_GPUS];
  uint32_t           *d_input_values[MAX_NUM_GPUS];
  uint32_t           *d_output[MAX_NUM_GPUS];
  uint32_t           *d_output_values[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 100000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, d_input_values, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key_value(d_input, d_input_values, h_input_offsets,
                         d_output, d_output_values, h_output_offsets,
                         n_gpus);

  verify_sorted_order(d_output, d_output_values, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_input_values[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output_values[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, DISABLED_Random256MPerDevice_uint64_t)
{
  cudaStream_t stream{nullptr};
  
  uint64_t           *d_input[MAX_NUM_GPUS];
  uint64_t           *d_input_values[MAX_NUM_GPUS];
  uint64_t           *d_output[MAX_NUM_GPUS];
  uint64_t           *d_output_values[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 256000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, d_input_values, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key_value(d_input, d_input_values, h_input_offsets,
                         d_output, d_output_values, h_output_offsets,
                         n_gpus);

  verify_sorted_order(d_output, d_output_values, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_input_values[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output_values[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random256MPerDevice_uint32_t)
{
  cudaStream_t stream{nullptr};
  
  uint32_t           *d_input[MAX_NUM_GPUS];
  uint32_t           *d_input_values[MAX_NUM_GPUS];
  uint32_t           *d_output[MAX_NUM_GPUS];
  uint32_t           *d_output_values[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 256000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, d_input_values, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key_value(d_input, d_input_values, h_input_offsets,
                         d_output, d_output_values, h_output_offsets,
                         n_gpus);

  verify_sorted_order(d_output, d_output_values, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_input_values[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output_values[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random10MKeysPerDevice_uint64_t)
{
  cudaStream_t stream{nullptr};
  
  uint64_t           *d_input[MAX_NUM_GPUS];
  uint64_t           *d_output[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 10000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key(d_input, h_input_offsets,
                   d_output, h_output_offsets,
                   n_gpus);

  verify_sorted_order(d_output, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random10MKeysPerDevice_uint32_t)
{
  cudaStream_t stream{nullptr};
  
  uint32_t           *d_input[MAX_NUM_GPUS];
  uint32_t           *d_output[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 10000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key(d_input, h_input_offsets,
                   d_output, h_output_offsets,
                   n_gpus);

  verify_sorted_order(d_output, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random100MKeysPerDevice_uint64_t)
{
  cudaStream_t stream{nullptr};
  
  uint64_t           *d_input[MAX_NUM_GPUS];
  uint64_t           *d_output[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 100000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key(d_input, h_input_offsets,
                   d_output, h_output_offsets,
                   n_gpus);

  verify_sorted_order(d_output, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random100MKeysPerDevice_uint32_t)
{
  cudaStream_t stream{nullptr};
  
  uint32_t           *d_input[MAX_NUM_GPUS];
  uint32_t           *d_output[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 100000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key(d_input, h_input_offsets,
                   d_output, h_output_offsets,
                   n_gpus);

  verify_sorted_order(d_output, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random256MKeysPerDevice_uint64_t)
{
  cudaStream_t stream{nullptr};
  
  uint64_t           *d_input[MAX_NUM_GPUS];
  uint64_t           *d_output[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 256000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key(d_input, h_input_offsets,
                   d_output, h_output_offsets,
                   n_gpus);

  verify_sorted_order(d_output, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
  }
}

TEST_F(SortTest, Random256MKeysPerDevice_uint32_t)
{
  cudaStream_t stream{nullptr};
  
  uint32_t           *d_input[MAX_NUM_GPUS];
  uint32_t           *d_output[MAX_NUM_GPUS];
  unsigned long long  h_input_offsets[MAX_NUM_GPUS+1];
  unsigned long long  h_output_offsets[MAX_NUM_GPUS+1];

  const long long num_elements = 256000000;
  const int seed = 43;
  int n_gpus = 0;

  CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
  ASSERT_LE(n_gpus, MAX_NUM_GPUS);

  for (int i = 0 ; i < (n_gpus + 1) ; ++i)
    h_input_offsets[i] = i * num_elements;

  omp_set_num_threads(n_gpus);

  generate_random(d_input, h_input_offsets, n_gpus, seed, stream);

  //
  //  Initialize Peer-to-peer communications
  //
  cusort::initialize_snmg_communication(n_gpus);

  // NOTE:  could vary numBins, binScale, useThrust
  cusort::sort_key(d_input, h_input_offsets,
                   d_output, h_output_offsets,
                   n_gpus);

  verify_sorted_order(d_output, h_output_offsets, n_gpus, stream, true);

  for (int i = 0 ; i < n_gpus ; ++i) {
    cudaSetDevice(i);

    EXPECT_EQ(RMM_FREE(d_input[i], stream), RMM_SUCCESS);
    EXPECT_EQ(RMM_FREE(d_output[i], stream), RMM_SUCCESS);
  }
}

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}
