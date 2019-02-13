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

// Renumber vertices
// Author: Chuck Hastings charlesh@nvidia.com

#ifndef RENUMBER_H
#define RENUMBER_H

#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <cudf.h>
#include <cuda_runtime_api.h>

#include "utilities/error_utils.h"
#include "graph_utils.cuh"
#include "rmm_utils.h"

namespace cugraph {

  namespace detail {
    typedef uint32_t   hash_type;
    
    template <typename VertexIdType>
    class HashFunctionObject {
    public:
      HashFunctionObject(hash_type hash_size): hash_size_(hash_size) {}

      __device__ __host__
      hash_type operator()(const VertexIdType &vertex_id) {
	return (vertex_id % hash_size_);
      }

    private:
      hash_type hash_size_;
    };

    template<typename T, typename F>
    __global__ void CountHash(const T *vertex_ids, size_t size, hash_type *counts, F hashing_function) {

      int first = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;
 
      for (int i = first ; i < size ; i += stride) {
        atomicAdd(&counts[hashing_function(vertex_ids[i])], hash_type{1});
      }
    }

    template<typename T, typename F>
    __global__ void PopulateHash(const T *vertex_ids, size_t size, T *hash_data,
                                 hash_type *hash_bins_start,
				 hash_type *hash_bins_end,
                                 F hashing_function ) {

      uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
      uint32_t stride = blockDim.x * gridDim.x;

      for (uint32_t i = first ; i < size ; i += stride) {
        uint32_t hash_index = hashing_function(vertex_ids[i]);

        //
        //  We're populating the hash table in parallel.  To limit
        //  the large numbers of duplicates being inserted we'll
        //  scan the hash bin to see if this address is already
        //  there.  Note that there's a race condition here, we will
        //  insert some number of duplicate entries.  We'll dedupe
        //  the rest later.
        //
        uint32_t hash_begin = hash_bins_start[hash_index];
        uint32_t hash_end = hash_bins_end[hash_index];
        bool found = false;

        for (uint32_t j = hash_begin ; (j < hash_end) && (!found) ; ++j) {
          if (vertex_ids[i] == hash_data[j])
            found = true;
        }

        if (!found) {
          hash_type hash_offset = atomicAdd(&hash_bins_end[hash_index], 1);
          hash_data[hash_offset] = vertex_ids[i];
        }
      }
    }

    template<typename T>
    __global__ void DedupeHash(hash_type hash_size, T *hash_data, hash_type *hash_bins_start,
                               hash_type *hash_bins_end, hash_type *hash_bins_base) {

      int first = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;

      for (int i = first ; i < hash_size ; i += stride) {
        //
        //  For each hash bin, we want to dedupe the
        //  elements.  For now, let's just sort and dedupe
        //  using the STL.  At the end we want:
        //
        //    hash_bins_start[i] to be unchanged
        //    hash_bins_end[i] to point to the end of the deduped bin
        //    hash_bins_base[i] to identify the number of unique elements in the bin
        //
        if (hash_bins_end[i] > hash_bins_start[i]) {
          thrust::sort(thrust::device, hash_data + hash_bins_start[i], hash_data + hash_bins_end[i]);
          T *new_end = thrust::unique(thrust::device, hash_data + hash_bins_start[i], hash_data + hash_bins_end[i]);

          hash_bins_base[i] = (new_end - (hash_data + hash_bins_start[i]));
          hash_bins_end[i] = hash_bins_start[i] + hash_bins_base[i];
        } else {
          hash_bins_base[i] = 0;
        }
      }
    }

    template<typename T_in, typename T_out, typename F>
    __global__ void Renumber(const T_in *vertex_ids, size_t size,
                             T_out *renumbered_ids, const T_in *hash_data,
                             const hash_type *hash_bins_start,
			     const hash_type *hash_bins_end,
                             const hash_type *hash_bins_base,
			     F hashing_function ) {
      //
      //  For each vertex, look up in the hash table the vertex id
      //
      int first = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;

      for (int i = first ; i < size ; i += stride) {
        hash_type hash = hashing_function(vertex_ids[i]);
        const T_in *id = thrust::lower_bound(thrust::device, hash_data + hash_bins_start[hash], hash_data + hash_bins_end[hash], vertex_ids[i]);
        renumbered_ids[i] = hash_bins_base[hash] + (id - (hash_data + hash_bins_start[hash]));
      }
    }

    template<typename T>
    __global__ void CompactNumbers(T *numbering_map, hash_type hash_size, const T *hash_data, const hash_type *hash_bins_start,
                                   const hash_type *hash_bins_end, const hash_type *hash_bins_base) {

      int first = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;

      for (int i = first ; i < hash_size ; i += stride) {
        for (int j = hash_bins_start[i] ; j < hash_bins_end[i] ; ++j) {
          numbering_map[hash_bins_base[i] + (j - hash_bins_start[i])] = hash_data[j];
        }
      }
    }

    __global__ void SetupHash(hash_type hash_size, hash_type *hash_bins_start, hash_type *hash_bins_end) {
      hash_bins_end[0] = 0;
      for (hash_type i = 0 ; i < hash_size ; ++i) {
        hash_bins_end[i+1] = hash_bins_end[i] + hash_bins_start[i];
      }

      for (hash_type i = 0 ; i < (hash_size + 1) ; ++i) {
        hash_bins_start[i] = hash_bins_end[i];
      }
   }

    __global__ void ComputeBase(hash_type hash_size, hash_type *hash_bins_base) {
      hash_type sum = 0;
      for (hash_type i = 0 ; i < hash_size ; ++i) {
        sum += hash_bins_base[i];
      }

      hash_bins_base[hash_size] = sum;
      for (hash_type i = hash_size ; i > 0 ; --i) {
        hash_bins_base[i-1] = hash_bins_base[i] - hash_bins_base[i-1];
      }
    }
  }


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
   * @param[out] numbering_map        Map of new vertex id to original vertex id.  numbering_map[newId] = oldId
   *
   * @return  SOME SORT OF ERROR CODE
   */
  template <typename T_in, typename T_out>
  gdf_error renumber_vertices(size_t size,
			      const T_in *src,
			      const T_in *dst,
			      T_out *src_renumbered,
			      T_out *dst_renumbered,
                              size_t *new_size,
			      T_in ** numbering_map,
			      int max_threads_per_block = CUDA_MAX_KERNEL_THREADS,
			      int max_blocks = CUDA_MAX_BLOCKS,
			      //detail::hash_type hash_size = 8191) {
			      detail::hash_type hash_size = 79) {

    //
    // Assume - src/dst/src_renumbered/dst_renumbered/numbering_map are all pre-allocated.
    //
    // Here's the idea: Create a hash table. Since we're dealing with integers,
    // we can take the integer modulo some prime p to create hash buckets.  Then
    // we dedupe the hash buckets to create a deduped set of entries.  This hash
    // table can then be used to renumber everything.
    //
    // We need 3 for hashing, and one array for data
    //
    T_in *hash_data;

    detail::HashFunctionObject<T_in>  hash(hash_size);

    detail::hash_type  *hash_bins_start;
    detail::hash_type  *hash_bins_end;
    detail::hash_type  *hash_bins_base;

    int threads_per_block = min((int) size, max_threads_per_block);
    int thread_blocks = min(((int) size + threads_per_block - 1) / threads_per_block, max_blocks);

    ALLOC_TRY(&hash_data,       2 * size * sizeof(T_in), nullptr);
    ALLOC_TRY(&hash_bins_start, (1 + hash_size) * sizeof(detail::hash_type), nullptr);
    ALLOC_TRY(&hash_bins_end,   (1 + hash_size) * sizeof(detail::hash_type), nullptr);
    ALLOC_TRY(&hash_bins_base,  (1 + hash_size) * sizeof(detail::hash_type), nullptr);

    //
    //  Pass 1: count how many vertex ids end up in each hash bin
    //
    CUDA_TRY(cudaMemset(hash_bins_start, 0, (1 + hash_size) * sizeof(detail::hash_type)));
    detail::CountHash<<<thread_blocks, threads_per_block>>>(src, size, hash_bins_start, hash);
    detail::CountHash<<<thread_blocks, threads_per_block>>>(dst, size, hash_bins_start, hash);

    //
    //  Need to compute the partial sums and copy them into hash_bins_end
    //
    detail::SetupHash<<<1,1>>>(hash_size, hash_bins_start, hash_bins_end);

    //
    //  Pass 2: Populate hash_data with data from the hash bins.  This implementation
    //    will do some partial deduplication, but we'll need to fully dedupe later.
    //
    detail::PopulateHash<<<thread_blocks, threads_per_block>>>(src, size, hash_data, hash_bins_start, hash_bins_end, hash);
    detail::PopulateHash<<<thread_blocks, threads_per_block>>>(dst, size, hash_data, hash_bins_start, hash_bins_end, hash);

    //
    //  Now we need to dedupe the hash bins
    //
    detail::DedupeHash<<<thread_blocks, threads_per_block>>>(hash_size, hash_data, hash_bins_start, hash_bins_end, hash_bins_base);

    //
    //  Now we can compute densly packed indices
    //
    detail::ComputeBase<<<1,1>>>(hash_size, hash_bins_base);

    //
    //  Finally, we'll iterate over src and dst and populate src_renumbered
    //  and dst_renumbered.
    //
    detail::Renumber<<<thread_blocks, threads_per_block>>>(src, size, src_renumbered, hash_data, hash_bins_start, hash_bins_end, hash_bins_base, hash);
    detail::Renumber<<<thread_blocks, threads_per_block>>>(dst, size, dst_renumbered, hash_data, hash_bins_start, hash_bins_end, hash_bins_base, hash);

    detail::hash_type temp{0};
    CUDA_TRY(cudaMemcpy(&temp, hash_bins_base + hash_size, sizeof(detail::hash_type), cudaMemcpyDeviceToHost));
    *new_size = temp;

    ALLOC_TRY(numbering_map, (*new_size) * sizeof(T_in), nullptr);

    detail::CompactNumbers<<<thread_blocks, threads_per_block>>>(*numbering_map, hash_size, hash_data, hash_bins_start, hash_bins_end, hash_bins_base);

    ALLOC_FREE_TRY(hash_data, nullptr);
    ALLOC_FREE_TRY(hash_bins_start, nullptr);
    ALLOC_FREE_TRY(hash_bins_end, nullptr);
    ALLOC_FREE_TRY(hash_bins_base, nullptr);

    return GDF_SUCCESS;
  }
  
}

#endif
