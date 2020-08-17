/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#pragma once

#define CUB_STDERR

#include <chrono>

#include <cub/cub.cuh>

#include <cuda_runtime_api.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <raft/cudart_utils.h>
#include <rmm/device_buffer.hpp>

#include <utilities/error.hpp>
#include "sort/bitonic.cuh"
#include "utilities/graph_utils.cuh"

namespace cugraph {
namespace detail {

namespace renumber {
typedef uint32_t hash_type;
typedef uint32_t index_type;
}  // namespace renumber

class HashFunctionObjectInt {
 public:
  HashFunctionObjectInt(renumber::hash_type hash_size) : hash_size_(hash_size) {}

  template <typename VertexIdType>
  __device__ __inline__ renumber::hash_type operator()(const VertexIdType &vertex_id) const
  {
    return ((vertex_id % hash_size_) + hash_size_) % hash_size_;
  }

  renumber::hash_type getHashSize() const { return hash_size_; }

 private:
  renumber::hash_type hash_size_;
};

/**
 * @brief Renumber vertices to a dense numbering (0..vertex_size-1)
 *
 *    This is a templated function so it can take 32 or 64 bit integers.  The
 *    intention is to take source and destination vertex ids that might be
 *    sparsely scattered across the range and push things down to a dense
 *    numbering.
 *
 *    Arrays src, dst, src_renumbered, dst_renumbered and numbering_map are
 *    assumed to be pre-allocated.  numbering_map is best safely allocated
 *    to store 2 * size vertices.
 *
 * @param[in]  size                 Number of edges
 * @param[in]  src                  List of source vertices
 * @param[in]  dst                  List of dest vertices
 * @param[out] src_renumbered       List of source vertices, renumbered
 * @param[out] dst_renumbered       List of dest vertices, renumbered
 * @param[out] vertex_size          Number of unique vertices
 * @param[out] numbering_map        Map of new vertex id to original vertex id. numbering_map[newId]
 * = oldId
 *
 */
template <typename T_size, typename T_in, typename T_out, typename Hash_t, typename Compare_t>
std::unique_ptr<rmm::device_buffer> renumber_vertices(T_size size,
                                                      const T_in *src,
                                                      const T_in *dst,
                                                      T_out *src_renumbered,
                                                      T_out *dst_renumbered,
                                                      T_size *map_size,
                                                      Hash_t hash,
                                                      Compare_t compare,
                                                      rmm::mr::device_memory_resource *mr)
{
  //
  // This function will allocate numbering_map to be the exact size needed
  // (user doesn't know a priori how many unique vertices there are.
  //
  // Here's the idea: Create a hash table. Since we're dealing with integers,
  // we can take the integer modulo some prime p to create hash buckets.  Then
  // we dedupe the hash buckets to create a deduped set of entries.  This hash
  // table can then be used to renumber everything.
  //
  // We need 2 arrays for hash indexes, and one array for data
  //
  cudaStream_t stream = nullptr;

  renumber::hash_type hash_size = hash.getHashSize();

  rmm::device_vector<T_in> hash_data_v(2 * size);
  rmm::device_vector<renumber::index_type> hash_bins_start_v(1 + hash_size,
                                                             renumber::index_type{0});
  rmm::device_vector<renumber::index_type> hash_bins_end_v(1 + hash_size);

  T_in *hash_data                       = hash_data_v.data().get();
  renumber::index_type *hash_bins_start = hash_bins_start_v.data().get();
  renumber::index_type *hash_bins_end   = hash_bins_end_v.data().get();

  //
  //  Pass 1: count how many vertex ids end up in each hash bin
  //
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   src,
                   src + size,
                   [hash_bins_start, hash] __device__(T_in vid) {
                     atomicAdd(hash_bins_start + hash(vid), renumber::index_type{1});
                   });

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   dst,
                   dst + size,
                   [hash_bins_start, hash] __device__(T_in vid) {
                     atomicAdd(hash_bins_start + hash(vid), renumber::index_type{1});
                   });

  //
  //  Compute exclusive sum and copy it into both hash_bins_start and
  //  hash_bins_end.  hash_bins_end will be used to populate the
  //  hash_data array and at the end will identify the end of
  //  each range.
  //
  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         hash_bins_start,
                         hash_bins_start + hash_size + 1,
                         hash_bins_end);

  CUDA_TRY(cudaMemcpy(hash_bins_start,
                      hash_bins_end,
                      (hash_size + 1) * sizeof(renumber::hash_type),
                      cudaMemcpyDeviceToDevice));

  //
  //  Pass 2: Populate hash_data with data from the hash bins.
  //
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   src,
                   src + size,
                   [hash_bins_end, hash_data, hash] __device__(T_in vid) {
                     uint32_t hash_index              = hash(vid);
                     renumber::index_type hash_offset = atomicAdd(&hash_bins_end[hash_index], 1);
                     hash_data[hash_offset]           = vid;
                   });

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   dst,
                   dst + size,
                   [hash_bins_end, hash_data, hash] __device__(T_in vid) {
                     uint32_t hash_index              = hash(vid);
                     renumber::index_type hash_offset = atomicAdd(&hash_bins_end[hash_index], 1);
                     hash_data[hash_offset]           = vid;
                   });

  //
  //  Now that we have data in hash bins, we'll do a segmented sort of the has bins
  //  to sort each bin.  This will allow us to identify duplicates (all duplicates
  //  are in the same hash bin so they will end up sorted consecutively).
  //
  renumber::index_type size_as_int = size;
  cugraph::sort::bitonic::segmented_sort(
    hash_size, size_as_int, hash_bins_start, hash_bins_end, hash_data, compare, stream);

  //
  //  Now we rinse and repeat.  hash_data contains the data organized into sorted
  //  hash bins.  This allows us to identify duplicates.  We'll start over but
  //  we'll skip the duplicates when we repopulate the hash table.
  //

  //
  //  Pass 3: count how many vertex ids end up in each hash bin after deduping
  //
  CUDA_TRY(cudaMemset(hash_bins_start, 0, (1 + hash_size) * sizeof(renumber::index_type)));

  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<renumber::index_type>(0),
    thrust::make_counting_iterator<renumber::index_type>(2 * size),
    [hash_data, hash_bins_start, hash, compare, size] __device__(renumber::index_type idx) {
      //
      //     Two items (a and b) are equal if
      //   compare(a,b) is false and compare(b,a)
      //   is also false.  If either is true then
      //   a and b are not equal.
      //
      //     Note that if there are k duplicate
      //   instances of an entry, only the LAST
      //   entry will be counted
      //
      bool unique = ((idx + 1) == (2 * size)) || compare(hash_data[idx], hash_data[idx + 1]) ||
                    compare(hash_data[idx + 1], hash_data[idx]);

      if (unique) atomicAdd(hash_bins_start + hash(hash_data[idx]), renumber::index_type{1});
    });

  //
  //  Compute exclusive sum and copy it into both hash_bins_start and
  //  hash bins end.
  //
  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         hash_bins_start,
                         hash_bins_start + hash_size + 1,
                         hash_bins_end);

  CUDA_TRY(cudaMemcpy(hash_bins_start,
                      hash_bins_end,
                      (hash_size + 1) * sizeof(renumber::hash_type),
                      cudaMemcpyDeviceToDevice));

  //
  //    The last entry in the array (hash_bins_end[hash_size]) is the
  //  total number of unique vertices
  //
  renumber::index_type temp = 0;
  CUDA_TRY(cudaMemcpy(
    &temp, hash_bins_end + hash_size, sizeof(renumber::index_type), cudaMemcpyDeviceToHost));
  *map_size = temp;

  rmm::device_buffer numbering_map(temp * sizeof(T_in), stream, mr);
  T_in *local_numbering_map = static_cast<T_in *>(numbering_map.data());

  //
  //  Pass 4: Populate hash_data with data from the hash bins after deduping
  //
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<renumber::index_type>(0),
                   thrust::make_counting_iterator<renumber::index_type>(2 * size),
                   [hash_bins_end, hash_data, local_numbering_map, hash, compare, size] __device__(
                     renumber::index_type idx) {
                     bool unique = ((idx + 1) == (2 * size)) ||
                                   compare(hash_data[idx], hash_data[idx + 1]) ||
                                   compare(hash_data[idx + 1], hash_data[idx]);

                     if (unique) {
                       uint32_t hash_index              = hash(hash_data[idx]);
                       renumber::index_type hash_offset = atomicAdd(&hash_bins_end[hash_index], 1);
                       local_numbering_map[hash_offset] = hash_data[idx];
                     }
                   });

  //
  //  At this point, hash_bins_start and numbering_map partition the
  //  unique data into a hash table.
  //

  //
  //  If we do a segmented sort now, we can do the final lookups.
  //
  size_as_int = size;
  cugraph::sort::bitonic::segmented_sort(
    hash_size, size_as_int, hash_bins_start, hash_bins_end, local_numbering_map, compare, stream);

  //
  //     Renumber the input.  For each vertex, identify the
  //   hash bin, and then search the hash bin for the
  //   record that matches, the relative offset between that
  //   element and the beginning of the array is the vertex
  //   id in the renumbered map.
  //
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<renumber::index_type>(0),
                   thrust::make_counting_iterator<renumber::index_type>(size),
                   [local_numbering_map,
                    hash_bins_start,
                    hash_bins_end,
                    hash,
                    src,
                    src_renumbered,
                    compare] __device__(renumber::index_type idx) {
                     renumber::hash_type tmp = hash(src[idx]);
                     const T_in *id =
                       thrust::lower_bound(thrust::seq,
                                           local_numbering_map + hash_bins_start[tmp],
                                           local_numbering_map + hash_bins_end[tmp],
                                           src[idx],
                                           compare);
                     src_renumbered[idx] = id - local_numbering_map;
                   });

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<renumber::index_type>(0),
                   thrust::make_counting_iterator<renumber::index_type>(size),
                   [local_numbering_map,
                    hash_bins_start,
                    hash_bins_end,
                    hash,
                    dst,
                    dst_renumbered,
                    compare] __device__(renumber::index_type idx) {
                     renumber::hash_type tmp = hash(dst[idx]);
                     const T_in *id =
                       thrust::lower_bound(thrust::seq,
                                           local_numbering_map + hash_bins_start[tmp],
                                           local_numbering_map + hash_bins_end[tmp],
                                           dst[idx],
                                           compare);
                     dst_renumbered[idx] = id - local_numbering_map;
                   });

  return std::make_unique<rmm::device_buffer>(std::move(numbering_map));
}

}  // namespace detail
}  // namespace cugraph
