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

#pragma once

#include <cstdint>
#include <cub/cub.cuh>
#include <limits>
#include <raft/core/handle.hpp>

#include <cugraph/graph_mask.hpp>

namespace cugraph {
namespace detail {

/**
 * Find the kth masked bit in a mask using divide and conquer approach
 */

template<typename t = int>
__device__ int level_to_bits(t l) {
    t l2 = 5-(l+1);
    return pow(2, l2);
}

__host__ __device__ __forceinline__ void print_ulong_bin(const uint32_t * const var, int bits) {
        int i;

    #if defined(__LP64__) || defined(_LP64)
            if( (bits > 64) || (bits <= 0) )
    #else
            if( (bits > 32) || (bits <= 0) )
    #endif
            return;

        for(i = 0; i < bits; i++) {
            printf("%d", (*var >> (bits - 1 - i)) & 0x01);
        }
        printf("\n");
    }


/**
 * Performs a bit-level binary search to find the index of the
 * kth bit in the mask which corresponds with the edge offset
 * in the original adjacency list.
 * @tparam edge_t
 * @tparam mask_type
 * @param mask
 * @param start_offset
 * @param stop_offset
 * @param n_bits
 * @param k
 * @return
 */
template<typename edge_t, typename mask_type>
__device__ edge_t kth_bit(mask_type mask_elm,
                          edge_t k, bool debug) {

    constexpr mask_type FMASK = 0xffffffff;
    constexpr mask_type n_shifts = std::numeric_limits<mask_type>::digits;

    /**
     *
     * 1. For hypersparse regions, we should be able to iterate through the mask bits, using the algorithm
     * below.
     *
     * 2. For regions of medium sparsity, we should be able to bin the vertices, compute their degrees, and then
     * cumsum them so that we can narrow down a small(-ish) range of mask indices we need to visit. After we've
     * narrowed down a few potential indices (and their starting degrees), we can run the algorithm below over them.
     *
     * 3. For regions with high density, we might be able to adjust the bin sizes so that the algorithm in 2 will work.
     */

    mask_type tmp_mask_elm = mask_elm;
    int idx = level_to_bits(0); // this is capped at n_shifts
    int k_tmp = k+1;

    // Iterate for first 4 levels
    for(int i = 0; i < 4; ++i) {
        int popl = __popc(tmp_mask_elm & FMASK >> (n_shifts - idx)); // count of left branch
        mask_type n_mask = (FMASK >> (n_shifts - idx));
        n_mask <<= (k_tmp > popl) * idx;
        k_tmp -= (k_tmp > popl) * popl;
        idx += level_to_bits(i + 1) * (-1 * (k_tmp <= popl) + (k_tmp > popl));
        tmp_mask_elm &= n_mask;
    }

    // Correct idx for the final level
    return idx - (__popc(mask_elm & FMASK >> (n_shifts - idx)) > k);
}

/**
 * A simple block-level kernel for computing mask vertex degrees by using CUDA intrinsics
 * for counting the number of bits set in a mask and aggregating them across threads.
 *
 * This kernel is not load balanced and so it assumes the same block size
 * will work across all vertices. An optimization could be to use the global
 * (unmasked) vertex degrees to better spread the load of computing the masked
 * vertex degrees, and subtract their complements atomically from the global
 * vertex degrees.
 */
template <typename vertex_t, typename edge_t, typename mask_type, int tpb = 128>
__global__ void masked_degree_kernel(edge_t* degrees_output,
                                     mask_type *edge_mask,
                                     edge_t const* indptr)
{
  /**
   *   1. For each vertex for each block, load the start and end offsets from indptr
   *   2. Compute start and end indices in the mask, along w/ the start and end masks
   *   3. For each element in the mask array, apply start or end mask if needed,
   *      compute popc (or popcll) and perform summed reduce of the result for each vertex
   */

  typedef cub::BlockReduce<vertex_t, tpb> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  int vertex            = blockIdx.x;

  vertex_t start_offset = indptr[vertex];
  vertex_t stop_offset  = indptr[vertex + 1];

  mask_type start_mask_offset = start_offset / std::numeric_limits<mask_type>::digits;
  mask_type stop_mask_offset  = stop_offset / std::numeric_limits<mask_type>::digits;

  mask_type start_bit = start_offset & (std::numeric_limits<mask_type>::digits - 1);

  mask_type stop_bit  = (std::numeric_limits<mask_type>::digits) - (stop_offset & (std::numeric_limits<mask_type>::digits - 1));

  // TODO: Check vertex mask for vertex
//  mask_type* vertex_mask = mask.get_vertex_mask();
//  mask_type* edge_mask = mask.get_edge_mask();

  vertex_t degree = 0;
  for (vertex_t i = threadIdx.x; i <= (stop_mask_offset - start_mask_offset); i += tpb) {


      mask_type mask_elm = edge_mask[i + start_mask_offset];

      // Apply start / stop masks to the first and last elements
      if(i == 0) {
          mask_elm = mask_elm & 0xffffffff << start_bit;
      }

      if(i == stop_mask_offset - start_mask_offset) {
          mask_elm = mask_elm & 0xffffffff >> stop_bit;
      }

      degree += __popc(mask_elm);
  }

  degree = BlockReduce(temp_storage).Sum(degree);

  if(degree > 0)
      printf("bid=%d, degree=%ld\n", blockIdx.x, degree);
  if (threadIdx.x == 0) { degrees_output[vertex] = degree; }
}

template <typename vertex_t, typename edge_t, typename mask_type, int tpb = 128>
__global__ void masked_degree_kernel(edge_t* degrees_output,
                                     mask_type *edge_mask,
                                     int major_range_first,
                                     vertex_t const* vertex_ids,
                                     edge_t const* indptr)
{
  /**
   *   1. For each vertex for each block, load the start and end offsets from indptr
   *   2. Compute start and end indices in the mask, along w/ the start and end masks
   *   3. For each element in the mask array, apply start or end mask if needed,
   *      compute popc (or popcll) and perform summed reduce of the result for each vertex
   */

  typedef cub::BlockReduce<vertex_t, tpb> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

    int vertex            = blockIdx.x;

    vertex_t start_offset = indptr[vertex];
    vertex_t stop_offset  = indptr[vertex + 1];

    mask_type start_mask_offset = start_offset / std::numeric_limits<mask_type>::digits;
    mask_type stop_mask_offset  = stop_offset / std::numeric_limits<mask_type>::digits;

    mask_type start_bit = start_offset & (std::numeric_limits<mask_type>::digits - 1);
    mask_type stop_bit  = (std::numeric_limits<mask_type>::digits) -
            (stop_offset & (std::numeric_limits<mask_type>::digits - 1));

    vertex_t degree = 0;
    for (vertex_t i = threadIdx.x; i <= (stop_mask_offset - start_mask_offset); i += tpb) {

        mask_type mask_elm = edge_mask[i + start_mask_offset];

        // Apply start / stop masks to the first and last elements
        if(i == 0) {
            mask_elm = mask_elm & 0xffffffff << start_bit;
        }

        if(i == stop_mask_offset - start_mask_offset) {
            mask_elm = mask_elm & 0xffffffff >> stop_bit;
        }

        degree += __popc(mask_elm);
    }

  degree = BlockReduce(temp_storage).Sum(degree);

  if (threadIdx.x == 0) { degrees_output[vertex_ids[vertex] - major_range_first] = degree; }
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename mask_type = std::uint32_t>
void masked_degrees(raft::handle_t const& handle,
                    edge_t* degrees_output,
                    vertex_t size,
                    graph_mask_view_t<vertex_t, edge_t, mask_type> const& mask,
                    edge_t const* indptr)
{
  detail::masked_degree_kernel<vertex_t, edge_t, mask_type, 128>
    <<<size, 128, 0, handle.get_stream()>>>(degrees_output, mask.get_edge_mask(), indptr);
}

template <typename vertex_t, typename edge_t, typename mask_type = std::uint32_t>
void masked_degrees(raft::handle_t const& handle,
                    edge_t* degrees_output,
                    vertex_t size,
                    graph_mask_view_t<vertex_t, edge_t, mask_type> const& mask,
                    int major_range_first,
                    vertex_t const* vertex_ids,
                    edge_t const* indptr)
{
  detail::masked_degree_kernel<vertex_t, edge_t, mask_type, 128>
    <<<size, 128, 0, handle.get_stream()>>>(
      degrees_output, mask.get_edge_mask(), major_range_first, vertex_ids, indptr);
}

};  // end namespace cugraph
