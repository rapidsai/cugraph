/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cugraph_etl/functions.hpp>

#include <cugraph/utilities/error.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/host/new_delete_resource.hpp>

#include <hash/concurrent_unordered_map.cuh>

#include <cub/device/device_radix_sort.cuh>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thrust/pair.h>
#include <thrust/sort.h>

#include <tuple>

namespace cugraph {
namespace etl {
using size_type  = cudf::size_type;  // size_type is int32
using accum_type = uint32_t;

constexpr uint32_t hash_inc_constant = 9999;

typedef struct str_hash_value {
  __host__ __device__ str_hash_value(){};

  __host__ __device__ str_hash_value(size_type row, accum_type count, int32_t col)
  {
    row_   = row;
    count_ = count;
    col_   = col;
  };

  size_type row_{std::numeric_limits<size_type>::max()};
  accum_type count_{std::numeric_limits<accum_type>::max()};
  int32_t col_{std::numeric_limits<int32_t>::max()};  // 0 or 1 based on src or dst vertex

} str_hash_value;

// key is uint32 hash value
using cudf_map_type = concurrent_unordered_map<uint32_t, str_hash_value>;

__device__ uint32_t rotl32(uint32_t x, int8_t r) { return (x << r) | (x >> (32 - r)); }

__device__ inline uint32_t fmix32(uint32_t h)
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

// get 4 bytes of char
__device__ uint32_t MurmurHash3_32_head(uint32_t val, uint32_t hash_begin = 32435354)
{
  uint32_t h1           = hash_begin;  // seeds
  constexpr uint32_t c1 = 0xcc9e2d51;
  constexpr uint32_t c2 = 0x1b873593;

  uint32_t k1 = val;
  k1 *= c1;
  k1 = rotl32(k1, 15);
  k1 *= c2;
  h1 ^= k1;
  h1 = rotl32(h1, 13);
  h1 = h1 * 5 + 0xe6546b64;

  return h1;
}

__device__ uint32_t MurmurHash3_32_tail(uint8_t* data,
                                        int len_byte_tail,
                                        int len_total,
                                        uint32_t hash_begin = 32435354)
{
  uint32_t h1           = hash_begin;
  uint32_t k1           = 0;
  constexpr uint32_t c1 = 0xcc9e2d51;
  constexpr uint32_t c2 = 0x1b873593;
  switch (len_total & 3) {
    case 3: k1 ^= data[2] << 16;
    case 2: k1 ^= data[1] << 8;
    case 1:
      k1 ^= data[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  };
  //----------
  // finalization
  h1 ^= len_total;
  h1 = fmix32(h1);
  return h1;
}

__device__ uint32_t calc_murmur32_hash(int8_t* loc_1,
                                       int8_t* loc_2,
                                       int32_t len_str1,
                                       int32_t len_str2)
{
  uint32_t hashed_str_val = 32435354;  // seed

  for (int i = 0; i < len_str1 / 4; i++) {
    int8_t* loc    = &(loc_1[i * 4]);
    uint32_t data  = loc[0] | (loc[1] << 8) | (loc[2] << 16) | (loc[3] << 24);
    hashed_str_val = MurmurHash3_32_head(data, hashed_str_val);
  }
  uint32_t trailing_chars = (len_str1 & 3);
  uint32_t data           = 0;
  switch (len_str1 & 3) {
    case 3: data |= loc_1[len_str1 - 3] << 16;
    case 2: data |= loc_1[len_str1 - 2] << 8;
    case 1: data |= loc_1[len_str1 - 1];
  }

  // this assumes no empty data in col2
  // either have read misalignment or dtype read align problem
  for (int i = 0; i < len_str2; i++) {
    data |= loc_2[i] << (trailing_chars * 8);
    trailing_chars++;

    if (trailing_chars == 4) {
      hashed_str_val = MurmurHash3_32_head(data, hashed_str_val);
      trailing_chars = 0;
      data           = 0;
    }
  }
  if (trailing_chars != 0) {
    hashed_str_val = MurmurHash3_32_tail(
      reinterpret_cast<uint8_t*>(&data), trailing_chars, len_str1 + len_str2, hashed_str_val);
  }
  return hashed_str_val;
}

__device__ __inline__ bool compare_string(size_type src_idx,
                                          size_type match_idx,
                                          const int8_t* col_1,
                                          const int32_t* offset_1,
                                          const int8_t* col_2,
                                          const int32_t* offset_2)
{
  // match length
  int32_t start_a_0  = offset_1[src_idx];
  int32_t length_a_0 = offset_1[src_idx + 1] - start_a_0;
  int32_t start_a_1  = offset_2[src_idx];
  int32_t length_a_1 = offset_2[src_idx + 1] - start_a_1;

  int32_t start_b_0  = offset_1[match_idx];
  int32_t length_b_0 = offset_1[match_idx + 1] - start_b_0;
  int32_t start_b_1  = offset_2[match_idx];
  int32_t length_b_1 = offset_2[match_idx + 1] - start_b_1;
  if ((length_a_0 == length_b_0) && (length_a_1 == length_b_1)) {
    // match first part
    while (length_a_0 > 0) {
      if (col_1[start_a_0++] != col_1[start_b_0++]) { return false; }
      length_a_0--;
    }

    // match second part
    while (length_a_1 > 0) {
      if (col_2[start_a_1++] != col_2[start_b_1++]) { return false; }
      length_a_1--;
    }

  } else {
    return false;
  }

  // match chars
  return true;
}

__device__ __inline__ bool compare_string_2(size_type src_idx,
                                            size_type match_idx,
                                            int8_t* base_col_1,
                                            int32_t* base_offset_1,
                                            int8_t* base_col_2,
                                            int32_t* base_offset_2,
                                            int8_t* col_1,
                                            int32_t* offset_1,
                                            int8_t* col_2,
                                            int32_t* offset_2)
{
  // match length
  int32_t start_a_0  = base_offset_1[src_idx];
  int32_t length_a_0 = base_offset_1[src_idx + 1] - start_a_0;
  int32_t start_a_1  = base_offset_2[src_idx];
  int32_t length_a_1 = base_offset_2[src_idx + 1] - start_a_1;

  int32_t start_b_0  = offset_1[match_idx];
  int32_t length_b_0 = offset_1[match_idx + 1] - start_b_0;
  int32_t start_b_1  = offset_2[match_idx];
  int32_t length_b_1 = offset_2[match_idx + 1] - start_b_1;
  if ((length_a_0 == length_b_0) && (length_a_1 == length_b_1)) {
    // match first part
    while (length_a_0 > 0) {
      if (base_col_1[start_a_0++] != col_1[start_b_0++]) { return false; }
      length_a_0--;
    }

    // match second part
    while (length_a_1 > 0) {
      if (base_col_2[start_a_1++] != col_2[start_b_1++]) { return false; }
      length_a_1--;
    }

  } else {
    // printf("not equal length\n");
    return false;
  }

  // match chars
  // printf("%d matched\n", threadIdx.x);
  return true;
}

__device__ __inline__ size_type validate_ht_row_insert(volatile size_type* ptr)
{
  size_type row = ptr[0];
  int32_t sleep = 133;
  while (row == std::numeric_limits<size_type>::max()) {
#if (__CUDA_ARCH__ >= 700)
    __nanosleep(sleep);
#endif
    sleep = sleep * 2;
    row   = ptr[0];
  }
  return row;
}

__device__ __inline__ int32_t validate_ht_col_insert(volatile int32_t* ptr_col)
{
  volatile int32_t col = ptr_col[0];
  int32_t sleep        = 133;

  while (col == std::numeric_limits<int32_t>::max()) {
#if (__CUDA_ARCH__ >= 700)
    __nanosleep(sleep);
#endif
    sleep = sleep * 2;
    col   = ptr_col[0];
  }
  return col;
}

__global__ void concat_and_create_histogram(int8_t* col_1,
                                            int32_t* offset_1,
                                            int8_t* col_2,
                                            int32_t* offset_2,
                                            size_type num_rows,
                                            cudf_map_type hash_map,
                                            accum_type* sysmem_insert_counter)
{
  extern __shared__ int8_t smem_[];
  int32_t* smem_col_1_offsets = reinterpret_cast<int32_t*>(smem_);
  int32_t* smem_col_2_offsets =
    reinterpret_cast<int32_t*>(smem_ + ((blockDim.x + 1) * sizeof(int32_t)));
  accum_type* insert_counter =
    reinterpret_cast<uint32_t*>(smem_ + ((blockDim.x + 1) * 2 * sizeof(int32_t)));

  int warp_accum_idx = threadIdx.x / warpSize;

  if ((threadIdx.x % warpSize) == 0) insert_counter[warp_accum_idx] = 0;
  __syncwarp();

  size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;  // size_type is int32_t

  if (start_idx < num_rows) {
    smem_col_1_offsets[threadIdx.x] = offset_1[start_idx];
    smem_col_2_offsets[threadIdx.x] = offset_2[start_idx];
  }

  if (threadIdx.x == 0) {
    if ((start_idx + blockDim.x) <= num_rows) {
      smem_col_1_offsets[blockDim.x] = offset_1[start_idx + blockDim.x];
      smem_col_2_offsets[blockDim.x] = offset_2[start_idx + blockDim.x];
    } else {
      int32_t last_offset_idx             = num_rows - start_idx;
      smem_col_1_offsets[last_offset_idx] = offset_1[num_rows];
      smem_col_2_offsets[last_offset_idx] = offset_2[num_rows];
    }
  }
  __syncthreads();

  if (start_idx < num_rows) {
    int32_t len_str1 = smem_col_1_offsets[threadIdx.x + 1] - smem_col_1_offsets[threadIdx.x];
    int32_t len_str2 = smem_col_2_offsets[threadIdx.x + 1] - smem_col_2_offsets[threadIdx.x];

    int8_t* loc_1           = &(col_1[smem_col_1_offsets[threadIdx.x]]);
    int8_t* loc_2           = &(col_2[smem_col_2_offsets[threadIdx.x]]);
    uint32_t hashed_str_val = calc_murmur32_hash(loc_1, loc_2, len_str1, len_str2);

    // concurrent_unordered_map
    // key : hashed_val, val: {idx, count}
    auto insert_pair =
      hash_map.insert(thrust::make_pair(hashed_str_val, str_hash_value{start_idx, 0, 0}));

    if (!insert_pair.second) {
      size_type row__ = validate_ht_row_insert(&(insert_pair.first->second.row_));

      while (!compare_string(row__, start_idx, col_1, offset_1, col_2, offset_2)) {
        // else loop over +1 count of hash value and insert again
        hashed_str_val += hash_inc_constant;
        insert_pair =
          hash_map.insert(thrust::make_pair(hashed_str_val, str_hash_value{start_idx, 0, 0}));
        if (insert_pair.second) {
          atomicAdd(&(insert_counter[warp_accum_idx]), 1);
          break;
        }
        row__ = validate_ht_row_insert(&(insert_pair.first->second.row_));
      }
      atomicAdd((accum_type*)&(insert_pair.first->second.count_), 1);
    } else {
      atomicAdd((accum_type*)&(insert_pair.first->second.count_), 1);
      // // smem atomic counter before global aggregation
      atomicAdd(&(insert_counter[warp_accum_idx]), 1);
    }
  }
  __syncwarp();
  if ((threadIdx.x % warpSize) == 0) {
    atomicAdd(sysmem_insert_counter, insert_counter[warp_accum_idx]);
  }
}

__global__ void concat_and_create_histogram_2(int8_t* col_1,
                                              int32_t* offset_1,
                                              int8_t* col_2,
                                              int32_t* offset_2,
                                              int8_t* match_col_1,
                                              int32_t* match_offset_1,
                                              int8_t* match_col_2,
                                              int32_t* match_offset_2,
                                              size_type num_rows,
                                              cudf_map_type hash_map,
                                              accum_type* sysmem_insert_counter)
{
  extern __shared__ int8_t smem_[];
  int32_t* smem_col_1_offsets = reinterpret_cast<int32_t*>(smem_);
  int32_t* smem_col_2_offsets =
    reinterpret_cast<int32_t*>(smem_ + ((blockDim.x + 1) * sizeof(int32_t)));
  accum_type* insert_counter =
    reinterpret_cast<uint32_t*>(smem_ + ((blockDim.x + 1) * 2 * sizeof(int32_t)));

  int warp_accum_idx = threadIdx.x / warpSize;

  if ((threadIdx.x % warpSize) == 0) insert_counter[warp_accum_idx] = 0;
  __syncwarp();

  size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;  // size_type is int32_t

  if (start_idx < num_rows) {
    smem_col_1_offsets[threadIdx.x] = offset_1[start_idx];
    smem_col_2_offsets[threadIdx.x] = offset_2[start_idx];
  }

  if (threadIdx.x == 0) {
    if ((start_idx + blockDim.x) <= num_rows) {
      smem_col_1_offsets[blockDim.x] = offset_1[start_idx + blockDim.x];
      smem_col_2_offsets[blockDim.x] = offset_2[start_idx + blockDim.x];
    } else {
      int32_t last_offset_idx             = num_rows - start_idx;
      smem_col_1_offsets[last_offset_idx] = offset_1[num_rows];
      smem_col_2_offsets[last_offset_idx] = offset_2[num_rows];
    }
  }
  __syncthreads();

  if (start_idx < num_rows) {
    int32_t len_str1 = smem_col_1_offsets[threadIdx.x + 1] - smem_col_1_offsets[threadIdx.x];
    int32_t len_str2 = smem_col_2_offsets[threadIdx.x + 1] - smem_col_2_offsets[threadIdx.x];

    int8_t* loc_1           = &(col_1[smem_col_1_offsets[threadIdx.x]]);
    int8_t* loc_2           = &(col_2[smem_col_2_offsets[threadIdx.x]]);
    uint32_t hashed_str_val = calc_murmur32_hash(loc_1, loc_2, len_str1, len_str2);

    // concurrent_unordered_map
    // key : hashed_val, val: {idx, count}

    auto insert_pair =
      hash_map.insert(thrust::make_pair(hashed_str_val, str_hash_value{start_idx, 0, 1}));

    if (!insert_pair.second) {
      size_type row__ = validate_ht_row_insert(&(insert_pair.first->second.row_));
      int32_t col__   = validate_ht_col_insert(&(insert_pair.first->second.col_));

      while (1) {
        if (col__ == 0) {
          if (compare_string_2(row__,
                               start_idx,
                               match_col_1,
                               match_offset_1,
                               match_col_2,
                               match_offset_2,
                               col_1,
                               offset_1,
                               col_2,
                               offset_2))
            break;
        } else if (col__ == 1) {
          if (compare_string(row__, start_idx, col_1, offset_1, col_2, offset_2)) break;
        }
        // else loop over +1 count of hash value and insert again
        hashed_str_val += hash_inc_constant;
        // printf("new insert\n");
        insert_pair =
          hash_map.insert(thrust::make_pair(hashed_str_val, str_hash_value{start_idx, 0, 1}));
        if (insert_pair.second) {
          atomicAdd(&(insert_counter[warp_accum_idx]), 1);
          break;
        }
        row__ = validate_ht_row_insert(&(insert_pair.first->second.row_));
        col__ = validate_ht_col_insert(&(insert_pair.first->second.col_));
      }
      // atomicAdd((unsigned int *)&(insert_pair.first->second.count_), 1);
    } else {
      // atomicAdd((unsigned int *)&(insert_pair.first->second.count_), 1);
      // smem atomic counter before global aggregation
      atomicAdd(&(insert_counter[warp_accum_idx]), 1);
    }
  }
  __syncwarp();
  if ((threadIdx.x % warpSize) == 0) {
    atomicAdd(sysmem_insert_counter, insert_counter[warp_accum_idx]);
  }
}

template <typename T>
__global__ void set_src_vertex_idx(int8_t* col_1,
                                   int32_t* offset_1,
                                   int8_t* col_2,
                                   int32_t* offset_2,
                                   size_type num_rows,
                                   cudf_map_type lookup_table,
                                   T* out_vertex_mapping)
{
  extern __shared__ int8_t smem_[];
  int32_t* smem_col_1_offsets = reinterpret_cast<int32_t*>(smem_);
  int32_t* smem_col_2_offsets =
    reinterpret_cast<int32_t*>(smem_ + ((blockDim.x + 1) * sizeof(int32_t)));

  size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;  // size_type is int32_t

  if (start_idx < num_rows) {
    smem_col_1_offsets[threadIdx.x] = offset_1[start_idx];
    smem_col_2_offsets[threadIdx.x] = offset_2[start_idx];
  }

  if (threadIdx.x == 0) {
    if ((start_idx + blockDim.x) <= num_rows) {
      smem_col_1_offsets[blockDim.x] = offset_1[start_idx + blockDim.x];
      smem_col_2_offsets[blockDim.x] = offset_2[start_idx + blockDim.x];
    } else {
      int32_t last_offset_idx             = num_rows - start_idx;
      smem_col_1_offsets[last_offset_idx] = offset_1[num_rows];
      smem_col_2_offsets[last_offset_idx] = offset_2[num_rows];
    }
  }
  __syncthreads();

  if (start_idx < num_rows) {
    int32_t len_str1 = smem_col_1_offsets[threadIdx.x + 1] - smem_col_1_offsets[threadIdx.x];
    int32_t len_str2 = smem_col_2_offsets[threadIdx.x + 1] - smem_col_2_offsets[threadIdx.x];

    int8_t* loc_1           = &(col_1[smem_col_1_offsets[threadIdx.x]]);
    int8_t* loc_2           = &(col_2[smem_col_2_offsets[threadIdx.x]]);
    uint32_t hashed_str_val = calc_murmur32_hash(loc_1, loc_2, len_str1, len_str2);

    // concurrent_unordered_map
    // key : hashed_val, val: {idx, count}

    auto it = lookup_table.find(hashed_str_val);
    // match string, if not match hash+1 find again
    while (it != lookup_table.end()) {
      if (compare_string(it->second.row_, start_idx, col_1, offset_1, col_2, offset_2)) {
        out_vertex_mapping[start_idx] = (T)it->second.count_;
        break;
      }
      hashed_str_val += hash_inc_constant;
      it = lookup_table.find(hashed_str_val);
    }
  }
}

template <typename T>
__global__ void set_dst_vertex_idx(int8_t* col_1,
                                   int32_t* offset_1,
                                   int8_t* col_2,
                                   int32_t* offset_2,
                                   int8_t* match_col_1,
                                   int32_t* match_offset_1,
                                   int8_t* match_col_2,
                                   int32_t* match_offset_2,
                                   size_type num_rows,
                                   cudf_map_type lookup_table,
                                   T* out_vertex_mapping)
{
  extern __shared__ int8_t smem_[];
  int32_t* smem_col_1_offsets = reinterpret_cast<int32_t*>(smem_);
  int32_t* smem_col_2_offsets =
    reinterpret_cast<int32_t*>(smem_ + ((blockDim.x + 1) * sizeof(int32_t)));

  size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;  // size_type is int32_t

  if (start_idx < num_rows) {
    smem_col_1_offsets[threadIdx.x] = offset_1[start_idx];
    smem_col_2_offsets[threadIdx.x] = offset_2[start_idx];
  }

  if (threadIdx.x == 0) {
    if ((start_idx + blockDim.x) <= num_rows) {
      smem_col_1_offsets[blockDim.x] = offset_1[start_idx + blockDim.x];
      smem_col_2_offsets[blockDim.x] = offset_2[start_idx + blockDim.x];
    } else {
      int32_t last_offset_idx             = num_rows - start_idx;
      smem_col_1_offsets[last_offset_idx] = offset_1[num_rows];
      smem_col_2_offsets[last_offset_idx] = offset_2[num_rows];
    }
  }
  __syncthreads();

  if (start_idx < num_rows) {
    int32_t len_str1 = smem_col_1_offsets[threadIdx.x + 1] - smem_col_1_offsets[threadIdx.x];
    int32_t len_str2 = smem_col_2_offsets[threadIdx.x + 1] - smem_col_2_offsets[threadIdx.x];

    int8_t* loc_1           = &(col_1[smem_col_1_offsets[threadIdx.x]]);
    int8_t* loc_2           = &(col_2[smem_col_2_offsets[threadIdx.x]]);
    uint32_t hashed_str_val = calc_murmur32_hash(loc_1, loc_2, len_str1, len_str2);

    // concurrent_unordered_map
    // key : hashed_val, val: {idx, count}

    auto it = lookup_table.find(hashed_str_val);
    // match string, if not match hash+1 find again
    while (it != lookup_table.end()) {
      if (it->second.col_ == 0) {
        if (compare_string_2(it->second.row_,
                             start_idx,
                             match_col_1,
                             match_offset_1,
                             match_col_2,
                             match_offset_2,
                             col_1,
                             offset_1,
                             col_2,
                             offset_2)) {
          out_vertex_mapping[start_idx] = (T)it->second.count_;
          break;
        }
      } else if (it->second.col_ == 1) {
        if (compare_string(it->second.row_, start_idx, col_1, offset_1, col_2, offset_2)) {
          out_vertex_mapping[start_idx] = (T)it->second.count_;
          break;
        }
      }
      hashed_str_val += hash_inc_constant;
      it = lookup_table.find(hashed_str_val);
    }
  }
}

__global__ void create_mapping_histogram(uint32_t* hash_value,
                                         str_hash_value* payload,
                                         cudf_map_type hash_map,
                                         accum_type count)
{
  accum_type idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < count) { auto it = hash_map.insert(thrust::make_pair(hash_value[idx], payload[idx])); }
}

__global__ void assign_histogram_idx(cudf_map_type cuda_map_obj,
                                     size_t slot_count,
                                     str_hash_value* key,
                                     uint32_t* value,
                                     size_type* counter)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { counter[0] = 0; }
  __threadfence();
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto it = cuda_map_obj.data();
  for (size_t idx = tid; idx < slot_count; idx += (gridDim.x * blockDim.x)) {
    auto hash_itr = it + idx;

    if ((hash_itr->second.row_ != cuda_map_obj.get_unused_element().row_) &&
        (hash_itr->second.count_ != cuda_map_obj.get_unused_element().count_) &&
        (hash_itr->first != cuda_map_obj.get_unused_key())) {
      size_type count   = atomicAdd((size_type*)counter, 1);
      value[count]      = hash_itr->first;
      key[count].row_   = hash_itr->second.row_;
      key[count].count_ = hash_itr->second.count_;
      key[count].col_   = hash_itr->second.col_;
    }
  }
}

__global__ void set_vertex_indices(str_hash_value* ht_value_payload, accum_type count)
{
  accum_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  // change count_ to renumber_idx
  for (accum_type idx = tid; idx < count; idx += (gridDim.x * blockDim.x)) {
    ht_value_payload[idx].count_ = idx;
  }
}

__global__ void set_output_col_offsets(str_hash_value* row_col_pair,
                                       int32_t* out_col1_offset,
                                       int32_t* out_col2_offset,
                                       int dst_pair_match,
                                       int32_t* in_col1_offset,
                                       int32_t* in_col2_offset,
                                       accum_type total_elements)
{
  int32_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;

  for (int32_t idx = start_idx; idx < total_elements; idx += (gridDim.x * blockDim.x)) {
    if (dst_pair_match == row_col_pair[idx].col_ && idx == row_col_pair[idx].count_) {
      // get row
      int32_t row          = row_col_pair[idx].row_;
      out_col1_offset[idx] = in_col1_offset[row + 1] - in_col1_offset[row];
      out_col2_offset[idx] = in_col2_offset[row + 1] - in_col2_offset[row];
    } else {
      out_col1_offset[idx] = 0;
      out_col2_offset[idx] = 0;
    }
  }
}

__global__ void offset_buffer_size_comp(int32_t* out_col1_length,
                                        int32_t* out_col2_length,
                                        int32_t* out_col1_offsets,
                                        int32_t* out_col2_offsets,
                                        accum_type total_elem,
                                        accum_type* out_sum)
{
  int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx == 0) {
    accum_type sum = accum_type(out_col1_offsets[total_elem - 1] + out_col1_length[total_elem - 1]);
    out_col1_offsets[total_elem] = sum;
    out_sum[0]                   = sum;
  } else if (idx == 1) {
    accum_type sum = accum_type(out_col2_offsets[total_elem - 1] + out_col2_length[total_elem - 1]);
    out_col2_offsets[total_elem] = sum;
    out_sum[1]                   = sum;
  }
}

__global__ void select_unrenumber_string(str_hash_value* idx_to_col_row,
                                         int32_t total_elements,
                                         int8_t* src_col1,
                                         int8_t* src_col2,
                                         int32_t* src_col1_offsets,
                                         int32_t* src_col2_offsets,
                                         int8_t* dst_col1,
                                         int8_t* dst_col2,
                                         int32_t* dst_col1_offsets,
                                         int32_t* dst_col2_offsets,
                                         int8_t* col1_out,
                                         int8_t* col2_out,
                                         int32_t* col1_out_offsets,
                                         int32_t* col2_out_offsets)
{
  size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;

  for (size_type idx = start_idx; idx < total_elements; idx += (gridDim.x * blockDim.x)) {
    int32_t row = idx_to_col_row[idx].row_;
    int32_t col = idx_to_col_row[idx].col_;

    if (col == 0) {
      int32_t col1_src_str_start  = src_col1_offsets[row];
      int32_t col1_src_str_length = src_col1_offsets[row + 1] - col1_src_str_start;
      int32_t col1_out_offset     = col1_out_offsets[idx];

      for (int32_t i = 0; i < col1_src_str_length; i++) {
        col1_out[col1_out_offset + i] = src_col1[col1_src_str_start + i];
      }

      int32_t col2_src_str_start  = src_col2_offsets[row];
      int32_t col2_src_str_length = src_col2_offsets[row + 1] - col2_src_str_start;
      int32_t col2_out_offset     = col2_out_offsets[idx];

      for (int32_t i = 0; i < col2_src_str_length; i++) {
        col2_out[col2_out_offset + i] = src_col2[col2_src_str_start + i];
      }

    } else if (col == 1) {
      int32_t col1_dst_str_start  = dst_col1_offsets[row];
      int32_t col1_dst_str_length = dst_col1_offsets[row + 1] - col1_dst_str_start;
      int32_t col1_out_offset     = col1_out_offsets[idx];

      for (int32_t i = 0; i < col1_dst_str_length; i++) {
        col1_out[col1_out_offset + i] = src_col1[col1_dst_str_start + i];
      }

      int32_t col2_dst_str_start  = dst_col2_offsets[row];
      int32_t col2_dst_str_length = dst_col2_offsets[row + 1] - col2_dst_str_start;
      int32_t col2_out_offset     = col2_out_offsets[idx];

      for (int32_t i = 0; i < col2_dst_str_length; i++) {
        col2_out[col2_out_offset + i] = src_col2[col2_dst_str_start + i];
      }
    }
  }
}

struct struct_sort_descending {
  __host__ __device__ bool operator()(str_hash_value& a, str_hash_value& b)
  {
    return (a.count_ > b.count_);
  }
};

struct renumber_functor {
  template <typename Dtype, std::enable_if_t<not std::is_integral<Dtype>::value>* = nullptr>
  std::tuple<std::unique_ptr<cudf::column>,
             std::unique_ptr<cudf::column>,
             std::unique_ptr<cudf::table>>
  operator()(raft::handle_t const& handle,
             cudf::table_view const& src_view,
             cudf::table_view const& dst_view)
  {
    return std::make_tuple(
      std::unique_ptr<cudf::column>(new cudf::column(
        cudf::data_type(cudf::type_id::INT32), 0, rmm::device_buffer{0, cudaStream_t{0}})),
      std::unique_ptr<cudf::column>(new cudf::column(
        cudf::data_type(cudf::type_id::INT32), 0, rmm::device_buffer{0, cudaStream_t{0}})),
      std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{}));
  }

  template <typename Dtype, std::enable_if_t<std::is_integral<Dtype>::value>* = nullptr>
  std::tuple<std::unique_ptr<cudf::column>,
             std::unique_ptr<cudf::column>,
             std::unique_ptr<cudf::table>>
  operator()(raft::handle_t const& handle,
             cudf::table_view const& src_view,
             cudf::table_view const& dst_view)
  {
    assert(src_view.num_columns() == 2);
    assert(dst_view.num_columns() == 2);

    size_type num_rows    = src_view.num_rows();
    using char_type       = int8_t;
    using str_offset_type = int32_t;  // kernels init'd int32 only

    std::vector<char_type*> src_vertex_chars_ptrs;
    std::vector<str_offset_type*> src_vertex_offset_ptrs;
    std::vector<char_type*> dst_vertex_chars_ptrs;
    std::vector<str_offset_type*> dst_vertex_offset_ptrs;

    for (int i = 0; i < src_view.num_columns(); i++) {
      auto str_col_view = cudf::strings_column_view(src_view.column(i));
      src_vertex_chars_ptrs.push_back(
        const_cast<char_type*>(str_col_view.chars().data<char_type>()));
      src_vertex_offset_ptrs.push_back(
        const_cast<str_offset_type*>(str_col_view.offsets().data<str_offset_type>()));
    }

    for (int i = 0; i < dst_view.num_columns(); i++) {
      auto str_col_view = cudf::strings_column_view(dst_view.column(i));
      dst_vertex_chars_ptrs.push_back(
        const_cast<char_type*>(str_col_view.chars().data<char_type>()));
      dst_vertex_offset_ptrs.push_back(
        const_cast<str_offset_type*>(str_col_view.offsets().data<str_offset_type>()));
    }

    cudaStream_t exec_strm = handle.get_stream();

    auto mr = rmm::mr::new_delete_resource();
    size_t hist_size = sizeof(accum_type) * 32;
    accum_type* hist_insert_counter = static_cast<accum_type*>(mr.allocate(hist_size));
    *hist_insert_counter            = 0;

    float load_factor = 0.7;

    rmm::device_uvector<accum_type> atomic_agg(32, exec_strm);  // just padded to 32
    CHECK_CUDA(cudaMemsetAsync(atomic_agg.data(), 0, sizeof(accum_type), exec_strm));

    auto cuda_map_obj = cudf_map_type::create(
                          std::max(static_cast<size_t>(static_cast<double>(num_rows) / load_factor),
                                   (size_t)num_rows + 1),
                          exec_strm,
                          str_hash_value{})
                          .release();
    dim3 block(512, 1, 1);
    dim3 grid((num_rows - 1) / block.x + 1, 1, 1);

    int32_t num_multiprocessors = 80;  // get from cuda properties

    // assumes warp_size is 32
    size_t warp_size = 32;
    size_t smem_size =
      (block.x + 1) * 2 * sizeof(int32_t) + (block.x / warp_size) * sizeof(accum_type);

    concat_and_create_histogram<<<grid, block, smem_size, exec_strm>>>(src_vertex_chars_ptrs[0],
                                                                       src_vertex_offset_ptrs[0],
                                                                       src_vertex_chars_ptrs[1],
                                                                       src_vertex_offset_ptrs[1],
                                                                       num_rows,
                                                                       *cuda_map_obj,
                                                                       atomic_agg.data());

    concat_and_create_histogram_2<<<grid, block, smem_size, exec_strm>>>(dst_vertex_chars_ptrs[0],
                                                                         dst_vertex_offset_ptrs[0],
                                                                         dst_vertex_chars_ptrs[1],
                                                                         dst_vertex_offset_ptrs[1],
                                                                         src_vertex_chars_ptrs[0],
                                                                         src_vertex_offset_ptrs[0],
                                                                         src_vertex_chars_ptrs[1],
                                                                         src_vertex_offset_ptrs[1],
                                                                         num_rows,
                                                                         *cuda_map_obj,
                                                                         atomic_agg.data());

    CHECK_CUDA(cudaMemcpy(
      hist_insert_counter, atomic_agg.data(), sizeof(accum_type), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(exec_strm));

    accum_type key_value_count = hist_insert_counter[0];
    // {row, count} pairs, sortDesecending on count w/ custom comparator
    rmm::device_uvector<str_hash_value> sort_key(key_value_count, exec_strm);
    rmm::device_uvector<uint32_t> sort_value(key_value_count, exec_strm);  // string hash values
    rmm::device_uvector<size_type> atomic_idx(32, exec_strm);              // just padded to 32

    int32_t num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, assign_histogram_idx, block.x, 0);
    grid.x = num_multiprocessors * num_blocks;
    assign_histogram_idx<<<grid, block, 0, exec_strm>>>(*cuda_map_obj,
                                                        cuda_map_obj->capacity(),
                                                        sort_key.data(),
                                                        sort_value.data(),
                                                        atomic_idx.data());

    // can release original histogram memory here

    // FIXME: cub doesnt have custom comparator sort
    // new cub release will have sort with custom comparator
    thrust::sort_by_key(rmm::exec_policy(exec_strm),
                        sort_key.begin(),
                        sort_key.end(),
                        sort_value.begin(),
                        struct_sort_descending());

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, set_vertex_indices, block.x, 0);
    grid.x = num_multiprocessors * num_blocks;
    set_vertex_indices<<<grid, block, 0, exec_strm>>>(sort_key.data(), hist_insert_counter[0]);

    // can extract unrenumber table here
    // get separate src and dst idxs.
    rmm::device_uvector<str_offset_type> out_col1_length(key_value_count, exec_strm);
    rmm::device_uvector<str_offset_type> out_col2_length(key_value_count, exec_strm);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, set_output_col_offsets, block.x, 0);
    grid.x = num_multiprocessors * num_blocks;
    // k-v count pair, out_offset_ptr, inputcol offset ptrs (to measure length)
    set_output_col_offsets<<<grid, block, 0, exec_strm>>>(sort_key.data(),
                                                          out_col1_length.data(),
                                                          out_col2_length.data(),
                                                          0,
                                                          src_vertex_offset_ptrs[0],
                                                          src_vertex_offset_ptrs[1],
                                                          key_value_count);

    set_output_col_offsets<<<grid, block, 0, exec_strm>>>(sort_key.data(),
                                                          out_col1_length.data(),
                                                          out_col2_length.data(),
                                                          1,
                                                          dst_vertex_offset_ptrs[0],
                                                          dst_vertex_offset_ptrs[1],
                                                          key_value_count);

    // prefix sum to extract column offsets
    rmm::device_uvector<str_offset_type> out_col1_offsets(key_value_count + 1, exec_strm);
    rmm::device_uvector<str_offset_type> out_col2_offsets(key_value_count + 1, exec_strm);

    size_t tmp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  tmp_storage_bytes,
                                  out_col1_length.data(),
                                  out_col1_offsets.data(),
                                  key_value_count,
                                  exec_strm);
    rmm::device_buffer tmp_storage(tmp_storage_bytes, exec_strm);
    cub::DeviceScan::ExclusiveSum(tmp_storage.data(),
                                  tmp_storage_bytes,
                                  out_col1_length.data(),
                                  out_col1_offsets.data(),
                                  key_value_count,
                                  exec_strm);
    cub::DeviceScan::ExclusiveSum(tmp_storage.data(),
                                  tmp_storage_bytes,
                                  out_col2_length.data(),
                                  out_col2_offsets.data(),
                                  key_value_count,
                                  exec_strm);

    // reduce to get size of column allocations
    // just reusing exscan output instead of using cub::Reduce::Sum() again
    // also sets last value of offset buffer that exscan didnt set
    offset_buffer_size_comp<<<1, 32, 0, exec_strm>>>(out_col1_length.data(),
                                                     out_col2_length.data(),
                                                     out_col1_offsets.data(),
                                                     out_col2_offsets.data(),
                                                     key_value_count,
                                                     hist_insert_counter);

    CHECK_CUDA(cudaStreamSynchronize(exec_strm));
    // allocate output columns buffers
    rmm::device_buffer unrenumber_col1_chars(hist_insert_counter[0], exec_strm);
    rmm::device_buffer unrenumber_col2_chars(hist_insert_counter[1], exec_strm);

    // select string kernel
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, select_unrenumber_string, block.x, 0);
    grid.x = num_multiprocessors * num_blocks;
    select_unrenumber_string<<<grid, block, 0, exec_strm>>>(
      sort_key.data(),
      key_value_count,
      src_vertex_chars_ptrs[0],
      src_vertex_chars_ptrs[1],
      src_vertex_offset_ptrs[0],
      src_vertex_offset_ptrs[1],
      dst_vertex_chars_ptrs[0],
      dst_vertex_chars_ptrs[1],
      dst_vertex_offset_ptrs[0],
      dst_vertex_offset_ptrs[1],
      reinterpret_cast<char_type*>(unrenumber_col1_chars.data()),
      reinterpret_cast<char_type*>(unrenumber_col2_chars.data()),
      out_col1_offsets.data(),
      out_col2_offsets.data());
    CHECK_CUDA(cudaStreamSynchronize(exec_strm));  // do we need sync here??

    std::vector<std::unique_ptr<cudf::column>> renumber_table_vectors;

    auto offset_col_1 =
      std::unique_ptr<cudf::column>(new cudf::column(cudf::data_type(cudf::type_id::INT32),
                                                     key_value_count + 1,
                                                     std::move(out_col1_offsets.release())));

    auto str_col_1 =
      std::unique_ptr<cudf::column>(new cudf::column(cudf::data_type(cudf::type_id::INT8),
                                                     hist_insert_counter[0],
                                                     std::move(unrenumber_col1_chars)));

    renumber_table_vectors.push_back(
      cudf::make_strings_column(size_type(key_value_count),
                                std::move(offset_col_1),
                                std::move(str_col_1),
                                0,
                                rmm::device_buffer(size_type(0), exec_strm)));

    auto offset_col_2 =
      std::unique_ptr<cudf::column>(new cudf::column(cudf::data_type(cudf::type_id::INT32),
                                                     key_value_count + 1,
                                                     std::move(out_col2_offsets.release())));

    auto str_col_2 =
      std::unique_ptr<cudf::column>(new cudf::column(cudf::data_type(cudf::type_id::INT8),
                                                     hist_insert_counter[1],
                                                     std::move(unrenumber_col2_chars)));

    renumber_table_vectors.push_back(
      cudf::make_strings_column(size_type(key_value_count),
                                std::move(offset_col_2),
                                std::move(str_col_2),
                                0,
                                rmm::device_buffer(size_type(0), exec_strm)));

    // make table from string columns - did at the end

    // net HT just insert K-V pairs
    auto cuda_map_obj_mapping =
      cudf_map_type::create(static_cast<size_t>(static_cast<double>(key_value_count) / load_factor),
                            exec_strm,
                            str_hash_value{})
        .release();

    grid.x = (key_value_count - 1) / block.x + 1;
    create_mapping_histogram<<<grid, block, 0, exec_strm>>>(
      sort_value.data(), sort_key.data(), *cuda_map_obj_mapping, key_value_count);
    CHECK_CUDA(cudaStreamSynchronize(exec_strm));

    rmm::device_buffer src_buffer(sizeof(Dtype) * num_rows, exec_strm);
    rmm::device_buffer dst_buffer(sizeof(Dtype) * num_rows, exec_strm);

    // iterate input, check hash-map, match string, set vertex idx in buffer
    grid.x    = (num_rows - 1) / block.x + 1;
    smem_size = (block.x + 1) * 2 * sizeof(str_offset_type);
    set_src_vertex_idx<<<grid, block, smem_size, exec_strm>>>(
      src_vertex_chars_ptrs[0],
      src_vertex_offset_ptrs[0],
      src_vertex_chars_ptrs[1],
      src_vertex_offset_ptrs[1],
      num_rows,
      *cuda_map_obj_mapping,
      reinterpret_cast<Dtype*>(src_buffer.data()));
    CHECK_CUDA(cudaStreamSynchronize(exec_strm));
    set_dst_vertex_idx<<<grid, block, smem_size, exec_strm>>>(
      dst_vertex_chars_ptrs[0],
      dst_vertex_offset_ptrs[0],
      dst_vertex_chars_ptrs[1],
      dst_vertex_offset_ptrs[1],
      src_vertex_chars_ptrs[0],
      src_vertex_offset_ptrs[0],
      src_vertex_chars_ptrs[1],
      src_vertex_offset_ptrs[1],
      num_rows,
      *cuda_map_obj_mapping,
      reinterpret_cast<Dtype*>(dst_buffer.data()));

    std::vector<std::unique_ptr<cudf::column>> cols_vector;
    cols_vector.push_back(std::unique_ptr<cudf::column>(
      new cudf::column(cudf::data_type(cudf::type_id::INT32), num_rows, std::move(src_buffer))));

    cols_vector.push_back(std::unique_ptr<cudf::column>(
      new cudf::column(cudf::data_type(cudf::type_id::INT32), num_rows, std::move(dst_buffer))));

    CHECK_CUDA(cudaDeviceSynchronize());

    mr.deallocate(hist_insert_counter, hist_size);

    return std::make_tuple(
      std::move(cols_vector[0]),
      std::move(cols_vector[1]),
      std::move(std::make_unique<cudf::table>(std::move(renumber_table_vectors))));
  }
};

std::
  tuple<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>>
  renumber_cudf_tables(raft::handle_t const& handle,
                       cudf::table_view const& src_table,
                       cudf::table_view const& dst_table,
                       cudf::type_id dtype)
{
  CUGRAPH_EXPECTS(src_table.num_columns() == 2,
                  "Src col: only two string column vertex are supported");
  CUGRAPH_EXPECTS(dst_table.num_columns() == 2,
                  "Dst col: only two string column vertex are supported");

  size_type num_rows_ = src_table.num_rows();

  auto x =
    cudf::type_dispatcher(cudf::data_type{dtype}, renumber_functor{}, handle, src_table, dst_table);
  return x;
}

}  // namespace etl
}  // namespace cugraph
