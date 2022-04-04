/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

// Author: Prasun Gera pgera@nvidia.com

#include <iostream>

#include "traversal_common.cuh"
#include <cub/cub.cuh>
#include <cugraph/utilities/error.hpp>
namespace cugraph {
namespace detail {
namespace sssp_kernels {

// This is the second pass after relax_edges that sets the active frontier
// nodes and predecessors
template <typename IndexType, typename DistType>
__global__ void populate_frontier_and_preds(
  const IndexType* row_ptr,
  const IndexType* col_ind,
  const DistType* edge_weights,
  const IndexType* frontier,
  const IndexType frontier_size,
  const IndexType totaldegree,
  const IndexType max_items_per_thread,
  IndexType* new_frontier,
  IndexType* new_frontier_cnt,
  const IndexType* frontier_degrees_exclusive_sum,
  const IndexType* frontier_degrees_exclusive_sum_buckets_offsets,
  int* next_frontier_bmap,
  const int* relaxed_edges_bmap,
  const int* isolated_bmap,
  DistType* distances,
  DistType* next_distances,
  IndexType* predecessors,
  const int* edge_mask)
{
  // BlockScan
  typedef cub::BlockScan<IndexType, TOP_DOWN_EXPAND_DIMX> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_storage;

  // We will do a scan to know where to write in frontier
  // This will contain the common offset of the block
  __shared__ IndexType frontier_common_block_offset;

  __shared__ IndexType shared_buckets_offsets[TOP_DOWN_EXPAND_DIMX - NBUCKETS_PER_BLOCK + 1];
  __shared__ IndexType shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX + 1];

  IndexType block_offset = (blockDim.x * blockIdx.x) * max_items_per_thread;
  IndexType n_items_per_thread_left =
    (totaldegree - block_offset + TOP_DOWN_EXPAND_DIMX - 1) / TOP_DOWN_EXPAND_DIMX;

  n_items_per_thread_left = min(max_items_per_thread, n_items_per_thread_left);

  for (; (n_items_per_thread_left > 0) && (block_offset < totaldegree);

       block_offset += MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD * blockDim.x,
       n_items_per_thread_left -= MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD) {
    // In this loop, we will process batch_set_size batches
    IndexType nitems_per_thread =
      min(n_items_per_thread_left, (IndexType)MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD);

    // Loading buckets offset (see compute_bucket_offsets_kernel)

    if (threadIdx.x < (nitems_per_thread * NBUCKETS_PER_BLOCK + 1))
      shared_buckets_offsets[threadIdx.x] =
        frontier_degrees_exclusive_sum_buckets_offsets[block_offset / TOP_DOWN_BUCKET_SIZE +
                                                       threadIdx.x];

    // We will use shared_buckets_offsets
    __syncthreads();

    //
    // shared_buckets_offsets gives us a range of the possible indexes
    // for edge of linear_threadx, we are looking for the value k such as
    // k is the max value such as frontier_degrees_exclusive_sum[k] <=
    // linear_threadx
    //
    // we have 0 <= k < frontier_size
    // but we also have :
    //
    // frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE]
    // <= k
    // <=
    // frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE
    // + 1]
    //
    // To find the exact value in that range, we need a few values from
    // frontier_degrees_exclusive_sum (see below)
    // We will load them here
    // We will load as much as we can - if it doesn't fit we will make multiple
    // iteration of the next loop
    // Because all vertices in frontier have degree > 0, we know it will fits
    // if left + 1 = right (see below)

    // We're going to load values in frontier_degrees_exclusive_sum for batch
    // [left; right[
    // If it doesn't fit, --right until it does, then loop
    // It is excepted to fit on the first try, that's why we start right =
    // nitems_per_thread

    IndexType left  = 0;
    IndexType right = nitems_per_thread;

    while (left < nitems_per_thread) {
      //
      // Values that are necessary to compute the local binary searches
      // We only need those with indexes between extremes indexes of
      // buckets_offsets
      // We need the next val for the binary search, hence the +1
      //

      IndexType nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK] -
                                  shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;

      // If left = right + 1 we are sure to have nvalues_to_load <
      // TOP_DOWN_EXPAND_DIMX+1
      while (nvalues_to_load > (TOP_DOWN_EXPAND_DIMX + 1)) {
        --right;

        nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK] -
                          shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;
      }

      IndexType nitems_per_thread_for_this_load = right - left;

      IndexType frontier_degrees_exclusive_sum_block_offset =
        shared_buckets_offsets[left * NBUCKETS_PER_BLOCK];

      if (threadIdx.x < nvalues_to_load) {
        shared_frontier_degrees_exclusive_sum[threadIdx.x] =
          frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset + threadIdx.x];
      }

      if (nvalues_to_load == (TOP_DOWN_EXPAND_DIMX + 1) && threadIdx.x == 0) {
        shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX] =
          frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset +
                                         TOP_DOWN_EXPAND_DIMX];
      }

      // shared_frontier_degrees_exclusive_sum is in shared mem, we will use
      // it, sync
      __syncthreads();

      // Now we will process the edges
      // Here each thread will process nitems_per_thread_for_this_load
      for (IndexType item_index = 0; item_index < nitems_per_thread_for_this_load;
           item_index += TOP_DOWN_BATCH_SIZE) {
        // We process TOP_DOWN_BATCH_SIZE edge in parallel (instruction
        // parallism)
        // Reduces latency

        IndexType current_max_edge_index =
          min(block_offset + (left + nitems_per_thread_for_this_load) * blockDim.x, totaldegree);

        IndexType naccepted_vertices = 0;
        IndexType vec_frontier_candidate[TOP_DOWN_BATCH_SIZE];

#pragma unroll
        for (IndexType iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType ibatch           = left + item_index + iv;
          IndexType gid              = block_offset + ibatch * blockDim.x + threadIdx.x;
          vec_frontier_candidate[iv] = -1;

          if (gid < current_max_edge_index) {
            IndexType start_off_idx = (ibatch * blockDim.x + threadIdx.x) / TOP_DOWN_BUCKET_SIZE;
            IndexType bucket_start =
              shared_buckets_offsets[start_off_idx] - frontier_degrees_exclusive_sum_block_offset;
            IndexType bucket_end = shared_buckets_offsets[start_off_idx + 1] -
                                   frontier_degrees_exclusive_sum_block_offset;

            IndexType k = traversal::binsearch_maxle(
                            shared_frontier_degrees_exclusive_sum, gid, bucket_start, bucket_end) +
                          frontier_degrees_exclusive_sum_block_offset;

            IndexType src_id = frontier[k];  // origin of this edge
            IndexType edge   = row_ptr[src_id] + gid - frontier_degrees_exclusive_sum[k];

            bool was_edge_relaxed = relaxed_edges_bmap[gid / INT_SIZE] & (1 << (gid % INT_SIZE));
            // Check if this edge was relaxed in relax_edges earlier
            if (was_edge_relaxed) {
              IndexType dst_id      = col_ind[edge];
              DistType dst_val      = next_distances[dst_id];
              DistType expected_val = distances[src_id] + edge_weights[edge];

              if (expected_val == dst_val) {
                // Our relaxation was the last one (not necessarily unique)
                // Try to become the parent in the SSSP tree atomically to
                // break potential ties
                // Set bit in next_frontier_bmap to 1 and check for old value
                // to check for success

                int old_val =
                  atomicOr(&next_frontier_bmap[dst_id / INT_SIZE], 1 << (dst_id % INT_SIZE));

                bool fail = (old_val >> (dst_id % INT_SIZE)) & 1;

                if (!fail) {
                  // Add dst_id to frontier if dst is not isolated
                  // (Can't have zero degree verts in frontier for the
                  // bucket/prefix-sum logic to work)
                  bool is_isolated = (isolated_bmap[dst_id / INT_SIZE] >> (dst_id % INT_SIZE)) & 1;

                  if (!is_isolated) {
                    vec_frontier_candidate[iv] = dst_id;
                    ++naccepted_vertices;
                  }

                  // Add src_id to predecessor in either case if needed
                  if (predecessors) { predecessors[dst_id] = src_id; }
                }
                // else lost the tie
              }
              // else somebody else relaxed it to a lower value after us in the
              // previous kernel
            }
          }
        }

        // We need to have all nfrontier_candidates to be ready before doing
        // the scan
        __syncthreads();

        // Computing block offsets
        IndexType thread_new_frontier_offset = 0;  // offset inside block
        BlockScan(scan_storage).ExclusiveSum(naccepted_vertices, thread_new_frontier_offset);

        if (threadIdx.x == (TOP_DOWN_EXPAND_DIMX - 1)) {
          IndexType inclusive_sum = thread_new_frontier_offset + naccepted_vertices;
          // for this thread, thread_new_frontier_offset + has_successor
          // (exclusive sum)
          if (inclusive_sum)
            frontier_common_block_offset = atomicAdd(new_frontier_cnt, inclusive_sum);
        }

        // Broadcasting frontier_common_block_offset
        __syncthreads();

        // Write to global memory
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType frontier_candidate = vec_frontier_candidate[iv];

          if (frontier_candidate != -1) {
            IndexType off     = frontier_common_block_offset + thread_new_frontier_offset++;
            new_frontier[off] = frontier_candidate;
          }
        }
      }

      // We need to keep shared_frontier_degrees_exclusive_sum coherent
      __syncthreads();

      // Preparing for next load
      left  = right;
      right = nitems_per_thread;
    }

    // we need to keep shared_buckets_offsets coherent
    __syncthreads();
  }
}

template <typename IndexType, typename DistType>
__global__ void relax_edges(const IndexType* row_ptr,
                            const IndexType* col_ind,
                            const DistType* edge_weights,
                            const IndexType* frontier,
                            const IndexType frontier_size,
                            const IndexType totaldegree,
                            const IndexType max_items_per_thread,
                            const IndexType* frontier_degrees_exclusive_sum,
                            const IndexType* frontier_degrees_exclusive_sum_buckets_offsets,
                            int* relaxed_edges_bmap,
                            DistType* distances,
                            DistType* next_distances,
                            const int* edge_mask)
{
  __shared__ IndexType shared_buckets_offsets[TOP_DOWN_EXPAND_DIMX - NBUCKETS_PER_BLOCK + 1];
  __shared__ IndexType shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX + 1];

  IndexType block_offset = (blockDim.x * blockIdx.x) * max_items_per_thread;
  IndexType n_items_per_thread_left =
    (totaldegree - block_offset + TOP_DOWN_EXPAND_DIMX - 1) / TOP_DOWN_EXPAND_DIMX;

  n_items_per_thread_left = min(max_items_per_thread, n_items_per_thread_left);

  for (; (n_items_per_thread_left > 0) && (block_offset < totaldegree);

       block_offset += MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD * blockDim.x,
       n_items_per_thread_left -= MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD) {
    // In this loop, we will process batch_set_size batches
    IndexType nitems_per_thread =
      min(n_items_per_thread_left, (IndexType)MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD);

    // Loading buckets offset (see compute_bucket_offsets_kernel)

    if (threadIdx.x < (nitems_per_thread * NBUCKETS_PER_BLOCK + 1))
      shared_buckets_offsets[threadIdx.x] =
        frontier_degrees_exclusive_sum_buckets_offsets[block_offset / TOP_DOWN_BUCKET_SIZE +
                                                       threadIdx.x];

    // We will use shared_buckets_offsets
    __syncthreads();

    //
    // shared_buckets_offsets gives us a range of the possible indexes
    // for edge of linear_threadx, we are looking for the value k such as
    // k is the max value such as frontier_degrees_exclusive_sum[k] <=
    // linear_threadx
    //
    // we have 0 <= k < frontier_size
    // but we also have :
    //
    // frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE]
    // <= k
    // <=
    // frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE
    // + 1]
    //
    // To find the exact value in that range, we need a few values from
    // frontier_degrees_exclusive_sum (see below)
    // We will load them here
    // We will load as much as we can - if it doesn't fit we will make multiple
    // iteration of the next loop
    // Because all vertices in frontier have degree > 0, we know it will fits
    // if left + 1 = right (see below)

    // We're going to load values in frontier_degrees_exclusive_sum for batch
    // [left; right[
    // If it doesn't fit, --right until it does, then loop
    // It is excepted to fit on the first try, that's why we start right =
    // nitems_per_thread

    IndexType left  = 0;
    IndexType right = nitems_per_thread;

    while (left < nitems_per_thread) {
      //
      // Values that are necessary to compute the local binary searches
      // We only need those with indexes between extremes indexes of
      // buckets_offsets
      // We need the next val for the binary search, hence the +1
      //

      IndexType nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK] -
                                  shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;

      // If left = right + 1 we are sure to have nvalues_to_load <
      // TOP_DOWN_EXPAND_DIMX+1
      while (nvalues_to_load > (TOP_DOWN_EXPAND_DIMX + 1)) {
        --right;

        nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK] -
                          shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;
      }

      IndexType nitems_per_thread_for_this_load = right - left;

      IndexType frontier_degrees_exclusive_sum_block_offset =
        shared_buckets_offsets[left * NBUCKETS_PER_BLOCK];

      if (threadIdx.x < nvalues_to_load) {
        shared_frontier_degrees_exclusive_sum[threadIdx.x] =
          frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset + threadIdx.x];
      }

      if (nvalues_to_load == (TOP_DOWN_EXPAND_DIMX + 1) && threadIdx.x == 0) {
        shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX] =
          frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset +
                                         TOP_DOWN_EXPAND_DIMX];
      }

      // shared_frontier_degrees_exclusive_sum is in shared mem, we will use
      // it, sync
      __syncthreads();

      // Now we will process the edges
      // Here each thread will process nitems_per_thread_for_this_load
      for (IndexType item_index = 0; item_index < nitems_per_thread_for_this_load;
           item_index += TOP_DOWN_BATCH_SIZE) {
        // We process TOP_DOWN_BATCH_SIZE edge in parallel (instruction
        // parallism)
        // Reduces latency

        IndexType current_max_edge_index =
          min(block_offset + (left + nitems_per_thread_for_this_load) * blockDim.x, totaldegree);

#pragma unroll
        for (IndexType iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType ibatch = left + item_index + iv;
          IndexType gid    = block_offset + ibatch * blockDim.x + threadIdx.x;

          if (gid < current_max_edge_index) {
            IndexType start_off_idx = (ibatch * blockDim.x + threadIdx.x) / TOP_DOWN_BUCKET_SIZE;
            IndexType bucket_start =
              shared_buckets_offsets[start_off_idx] - frontier_degrees_exclusive_sum_block_offset;
            IndexType bucket_end = shared_buckets_offsets[start_off_idx + 1] -
                                   frontier_degrees_exclusive_sum_block_offset;

            IndexType k = traversal::binsearch_maxle(
                            shared_frontier_degrees_exclusive_sum, gid, bucket_start, bucket_end) +
                          frontier_degrees_exclusive_sum_block_offset;

            IndexType src_id = frontier[k];
            IndexType edge   = row_ptr[frontier[k]] + gid - frontier_degrees_exclusive_sum[k];
            IndexType dst_id = col_ind[edge];

            // Try to relax non-masked edges
            if (!edge_mask || edge_mask[edge]) {
              DistType* update_addr = &next_distances[dst_id];
              DistType old_val      = distances[dst_id];
              DistType new_val      = distances[src_id] + edge_weights[edge];
              if (new_val < old_val) {
                // This edge can be relaxed

                // OPTION1
                // Add it to local candidates. Create shared candidates and
                // then call atomic on the candidates queue
                // More storage and work, but may have better performance since
                // the atomics will be packed in contiguous lanes
                // Not pursued

                // OPTION2
                // Try to relax with atomicmin directly. Easier, but may have
                // worse performance
                old_val = traversal::atomicMin(update_addr, new_val);

                if (old_val > new_val) {
                  // OPTION1:
                  // Add <src,dst> to frontier candidates
                  // Increment thread_frontier_count
                  // We'll sort/reduce and remove dupes in a different kernel
                  // Needs extra O(E) storage
                  // Not pursued

                  // OPTION2:
                  // Mark the bits in the edge bitmap. Still needs O(E) bitmap,
                  // but smaller constant
                  // We'll do a second pass for frontier and preds

                  int m = 1 << (gid % INT_SIZE);
                  atomicOr(&relaxed_edges_bmap[gid / INT_SIZE], m);

                  // OPTION3:
                  // Mark the ends (i.e., src and dst) in two bitmap. O(V)
                  // bitmap overhead, but more memory accesses in the second
                  // pass due to indexing col_ind
                }
                // else somebody else relaxed the dst distance to a lower value
              }
            }
          }
        }
      }

      // We need to keep shared_frontier_degrees_exclusive_sum coherent
      __syncthreads();

      // Preparing for next load
      left  = right;
      right = nitems_per_thread;
    }

    // we need to keep shared_buckets_offsets coherent
    __syncthreads();
  }
}

template <typename IndexType, typename DistType>
void frontier_expand(const IndexType* row_ptr,
                     const IndexType* col_ind,
                     const DistType* edge_weights,
                     const IndexType* frontier,
                     const IndexType frontier_size,
                     const IndexType totaldegree,
                     IndexType* new_frontier,
                     IndexType* new_frontier_cnt,
                     const IndexType* frontier_degrees_exclusive_sum,
                     const IndexType* frontier_degrees_exclusive_sum_buckets_offsets,
                     DistType* distances,
                     DistType* next_distances,
                     IndexType* predecessors,
                     const int* edge_mask,
                     int* next_frontier_bmap,
                     int* relaxed_edges_bmap,
                     const int* isolated_bmap,
                     cudaStream_t m_stream)
{
  if (!totaldegree) return;

  dim3 block;
  block.x = TOP_DOWN_EXPAND_DIMX;

  IndexType max_items_per_thread = (totaldegree + MAXBLOCKS * block.x - 1) / (MAXBLOCKS * block.x);

  dim3 grid;
  grid.x =
    min((totaldegree + max_items_per_thread * block.x - 1) / (max_items_per_thread * block.x),
        (IndexType)MAXBLOCKS);

  // Relax edges going out from the current frontier
  relax_edges<<<grid, block, 0, m_stream>>>(row_ptr,
                                            col_ind,
                                            edge_weights,
                                            frontier,
                                            frontier_size,
                                            totaldegree,
                                            max_items_per_thread,
                                            frontier_degrees_exclusive_sum,
                                            frontier_degrees_exclusive_sum_buckets_offsets,
                                            relaxed_edges_bmap,
                                            distances,
                                            next_distances,
                                            edge_mask);

  // Revisit relaxed edges and update the next frontier and preds
  populate_frontier_and_preds<<<grid, block, 0, m_stream>>>(
    row_ptr,
    col_ind,
    edge_weights,
    frontier,
    frontier_size,
    totaldegree,
    max_items_per_thread,
    new_frontier,
    new_frontier_cnt,
    frontier_degrees_exclusive_sum,
    frontier_degrees_exclusive_sum_buckets_offsets,
    next_frontier_bmap,
    relaxed_edges_bmap,
    isolated_bmap,
    distances,
    next_distances,
    predecessors,
    edge_mask);

  RAFT_CHECK_CUDA(m_stream);
}
}  // namespace sssp_kernels
}  // namespace detail
}  // namespace cugraph
