/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <iostream>

#include <cub/cub.cuh>
#include <raft/util/cudart_utils.hpp>

#include "traversal_common.cuh"
#include <cugraph/legacy/graph.hpp>

namespace cugraph {
namespace detail {
namespace bfs_kernels {
//
//  -------------------------  Bottom up -------------------------
//

//
// fill_unvisited_queue_kernel
//
// Finding unvisited vertices in the visited_bmap, and putting them in the queue
// Vertices represented by the same int in the bitmap are adjacent in the queue,
// and sorted For instance, the queue can look like this : 34 38 45 58 61 4 18
// 24 29 71 84 85 90 Because they are represented by those ints in the bitmap :
// [34 38 45 58 61] [4 18 24 29] [71 84 85 90]

// visited_bmap_nints = the visited_bmap is made of that number of ints

template <typename IndexType>
__global__ void fill_unvisited_queue_kernel(int* visited_bmap,
                                            IndexType visited_bmap_nints,
                                            IndexType n,
                                            IndexType* unvisited,
                                            IndexType* unvisited_cnt)
{
  typedef cub::BlockScan<int, FILL_UNVISITED_QUEUE_DIMX> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_temp_storage;

  // When filling the "unvisited" queue, we use "unvisited_cnt" to know where to
  // write in the queue (equivalent of int off = atomicAddd(unvisited_cnt, 1) )
  // We will actually do only one atomicAdd per block - we first do a scan, then
  // call one atomicAdd, and store the common offset for the block in
  // unvisited_common_block_offset
  __shared__ IndexType unvisited_common_block_offset;

  // We don't want threads divergence in the loop (we're going to call
  // __syncthreads) Using a block-only dependent in the condition of the loop
  for (IndexType block_v_idx = blockIdx.x * blockDim.x; block_v_idx < visited_bmap_nints;
       block_v_idx += blockDim.x * gridDim.x) {
    // Index of visited_bmap that this thread will compute
    IndexType v_idx = block_v_idx + threadIdx.x;

    int thread_visited_int = (v_idx < visited_bmap_nints)
                               ? visited_bmap[v_idx]
                               : (~0);  // will be neutral in the next lines
                                        // (virtual vertices all visited)

    // The last int can only be partially valid
    // If we are indeed taking care of the last visited int in this thread,
    // We need to first disable (ie set as "visited") the inactive bits
    // (vertices >= n)
    if (v_idx == (visited_bmap_nints - 1)) {
      int active_bits   = n - (INT_SIZE * v_idx);
      int inactive_bits = INT_SIZE - active_bits;
      int mask          = traversal::getMaskNLeftmostBitSet(inactive_bits);
      thread_visited_int |= mask;  // Setting inactive bits as visited
    }

    // Counting number of unvisited vertices represented by this int
    int n_unvisited_in_int = __popc(~thread_visited_int);
    int unvisited_thread_offset;

    // We will need to write n_unvisited_in_int unvisited vertices to the
    // unvisited queue We ask for that space when computing the block scan, that
    // will tell where to write those vertices in the queue, using the common
    // offset of the block (see below)
    BlockScan(scan_temp_storage).ExclusiveSum(n_unvisited_in_int, unvisited_thread_offset);

    // Last thread knows how many vertices will be written to the queue by this
    // block Asking for that space in the queue using the global count, and
    // saving the common offset
    if (threadIdx.x == (FILL_UNVISITED_QUEUE_DIMX - 1)) {
      IndexType total               = unvisited_thread_offset + n_unvisited_in_int;
      unvisited_common_block_offset = traversal::atomicAdd(unvisited_cnt, total);
    }

    // syncthreads for two reasons :
    // - we need to broadcast unvisited_common_block_offset
    // - we will reuse scan_temp_storage (cf CUB doc)
    __syncthreads();

    IndexType current_unvisited_index = unvisited_common_block_offset + unvisited_thread_offset;
    int nvertices_to_write            = n_unvisited_in_int;

    // getNextZeroBit uses __ffs, which gives least significant bit set
    // which means that as long as n_unvisited_in_int is valid,
    // we will use valid bits

    while (nvertices_to_write > 0) {
      if (nvertices_to_write >= 4 && (current_unvisited_index % 4) == 0) {
        typename traversal::vec_t<IndexType>::vec4 vec_v;

        vec_v.x = v_idx * INT_SIZE + traversal::getNextZeroBit(thread_visited_int);
        vec_v.y = v_idx * INT_SIZE + traversal::getNextZeroBit(thread_visited_int);
        vec_v.z = v_idx * INT_SIZE + traversal::getNextZeroBit(thread_visited_int);
        vec_v.w = v_idx * INT_SIZE + traversal::getNextZeroBit(thread_visited_int);

        typename traversal::vec_t<IndexType>::vec4* unvisited_i4 =
          reinterpret_cast<typename traversal::vec_t<IndexType>::vec4*>(
            &unvisited[current_unvisited_index]);
        *unvisited_i4 = vec_v;

        current_unvisited_index += 4;
        nvertices_to_write -= 4;
      } else if (nvertices_to_write >= 2 && (current_unvisited_index % 2) == 0) {
        typename traversal::vec_t<IndexType>::vec2 vec_v;

        vec_v.x = v_idx * INT_SIZE + traversal::getNextZeroBit(thread_visited_int);
        vec_v.y = v_idx * INT_SIZE + traversal::getNextZeroBit(thread_visited_int);

        typename traversal::vec_t<IndexType>::vec2* unvisited_i2 =
          reinterpret_cast<typename traversal::vec_t<IndexType>::vec2*>(
            &unvisited[current_unvisited_index]);
        *unvisited_i2 = vec_v;

        current_unvisited_index += 2;
        nvertices_to_write -= 2;
      } else {
        IndexType v = v_idx * INT_SIZE + traversal::getNextZeroBit(thread_visited_int);

        unvisited[current_unvisited_index] = v;

        current_unvisited_index += 1;
        nvertices_to_write -= 1;
      }
    }
  }
}

// Wrapper
template <typename IndexType>
void fill_unvisited_queue(int* visited_bmap,
                          IndexType visited_bmap_nints,
                          IndexType n,
                          IndexType* unvisited,
                          IndexType* unvisited_cnt,
                          cudaStream_t m_stream,
                          bool deterministic)
{
  dim3 grid, block;
  block.x = FILL_UNVISITED_QUEUE_DIMX;

  grid.x = std::min(static_cast<size_t>(MAXBLOCKS),
                    (static_cast<size_t>(visited_bmap_nints) + block.x - 1) / block.x);

  fill_unvisited_queue_kernel<<<grid, block, 0, m_stream>>>(
    visited_bmap, visited_bmap_nints, n, unvisited, unvisited_cnt);
  RAFT_CHECK_CUDA(m_stream);
}

//
// count_unvisited_edges_kernel
// Couting the total number of unvisited edges in the graph - using an
// potentially unvisited queue We need the current unvisited vertices to be in
// the unvisited queue But visited vertices can be in the potentially_unvisited
// queue We first check if the vertex is still unvisited before using it Useful
// when switching from "Bottom up" to "Top down"
//

template <typename IndexType>
__global__ void count_unvisited_edges_kernel(const IndexType* potentially_unvisited,
                                             const IndexType potentially_unvisited_size,
                                             const int* visited_bmap,
                                             IndexType* degree_vertices,
                                             IndexType* mu)
{
  typedef cub::BlockReduce<IndexType, COUNT_UNVISITED_EDGES_DIMX> BlockReduce;
  __shared__ typename BlockReduce::TempStorage reduce_temp_storage;

  // number of undiscovered edges counted by this thread
  IndexType thread_unvisited_edges_count = 0;

  for (IndexType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < potentially_unvisited_size;
       idx += blockDim.x * gridDim.x) {
    IndexType u        = potentially_unvisited[idx];
    int u_visited_bmap = visited_bmap[u / INT_SIZE];
    int is_visited     = u_visited_bmap & (1 << (u % INT_SIZE));

    if (!is_visited) thread_unvisited_edges_count += degree_vertices[u];
  }

  // We need all thread_unvisited_edges_count to be ready before reducing
  __syncthreads();

  IndexType block_unvisited_edges_count =
    BlockReduce(reduce_temp_storage).Sum(thread_unvisited_edges_count);

  // block_unvisited_edges_count is only defined is th.x == 0
  if (threadIdx.x == 0) traversal::atomicAdd(mu, block_unvisited_edges_count);
}

// Wrapper
template <typename IndexType>
void count_unvisited_edges(const IndexType* potentially_unvisited,
                           const IndexType potentially_unvisited_size,
                           const int* visited_bmap,
                           IndexType* node_degree,
                           IndexType* mu,
                           cudaStream_t m_stream)
{
  dim3 grid, block;
  block.x = COUNT_UNVISITED_EDGES_DIMX;
  grid.x  = std::min(static_cast<size_t>(MAXBLOCKS),
                    (static_cast<size_t>(potentially_unvisited_size) + block.x - 1) / block.x);

  count_unvisited_edges_kernel<<<grid, block, 0, m_stream>>>(
    potentially_unvisited, potentially_unvisited_size, visited_bmap, node_degree, mu);
  RAFT_CHECK_CUDA(m_stream);
}

//
// Main Bottom Up kernel
// Here we will start to process unvisited vertices in the unvisited queue
// We will only consider the first MAIN_BOTTOMUP_MAX_EDGES edges
// If it's not possible to define a valid parent using only those edges,
// add it to the "left_unvisited_queue"
//

//
// We will use the "vertices represented by the same int in the visited bmap are
// adjacents and sorted in the unvisited queue" property It is used to do a
// reduction locally and fully build the new visited_bmap
//

template <typename IndexType>
__global__ void main_bottomup_kernel(const IndexType* unvisited,
                                     const IndexType unvisited_size,
                                     IndexType* left_unvisited,
                                     IndexType* left_unvisited_cnt,
                                     int* visited_bmap,
                                     const IndexType* row_ptr,
                                     const IndexType* col_ind,
                                     IndexType lvl,
                                     IndexType* new_frontier,
                                     IndexType* new_frontier_cnt,
                                     IndexType* distances,
                                     IndexType* predecessors,
                                     int* edge_mask)
{
  typedef cub::BlockDiscontinuity<IndexType, MAIN_BOTTOMUP_DIMX> BlockDiscontinuity;
  typedef cub::WarpReduce<int> WarpReduce;
  typedef cub::BlockScan<int, MAIN_BOTTOMUP_DIMX> BlockScan;

  __shared__ typename BlockDiscontinuity::TempStorage discontinuity_temp_storage;
  __shared__ typename WarpReduce::TempStorage reduce_temp_storage;
  __shared__ typename BlockScan::TempStorage scan_temp_storage;

  // To write vertices in the frontier,
  // We will use a block scan to locally compute the offsets
  // frontier_common_block_offset contains the common offset for the block
  __shared__ IndexType frontier_common_block_offset;

  // When building the new visited_bmap, we reduce (using a bitwise and) the
  // visited_bmap ints from the vertices represented by the same int (for
  // instance vertices 1, 5, 9, 13, 23) vertices represented by the same int
  // will be designed as part of the same "group" To detect the deliminations
  // between those groups, we use BlockDiscontinuity Then we need to create the
  // new "visited_bmap" within those group. We use a warp reduction that takes
  // into account limits between groups to do it But a group can be cut in two
  // different warps : in that case, the second warp put the result of its local
  // reduction in local_visited_bmap_warp_head the first warp will then read it
  // and finish the reduction

  __shared__ int local_visited_bmap_warp_head[MAIN_BOTTOMUP_NWARPS];

  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;

  // When this kernel is converted to support different VT and ET, this
  // will likely split into invalid_vid and invalid_eid
  // This is equivalent to ~IndexType(0) (i.e., all bits set to 1)
  constexpr IndexType invalid_idx = cugraph::legacy::invalid_idx<IndexType>::value;

  // we will call __syncthreads inside the loop
  // we need to keep complete block active
  for (IndexType block_off = blockIdx.x * blockDim.x; block_off < unvisited_size;
       block_off += blockDim.x * gridDim.x) {
    IndexType idx = block_off + threadIdx.x;

    // This thread will take care of unvisited_vertex
    // in the visited_bmap, it is represented by the int at index
    // visited_bmap_index = unvisited_vertex/INT_SIZE
    // it will be used by BlockDiscontinuity
    // to flag the separation between groups of vertices (vertices represented
    // by different in in visited_bmap)
    IndexType visited_bmap_index[1];  // this is an array of size 1 because CUB
                                      // needs one

    visited_bmap_index[0]      = invalid_idx;
    IndexType unvisited_vertex = invalid_idx;

    // local_visited_bmap gives info on the visited bit of unvisited_vertex
    //
    // By default, everything is visited
    // This is because we only take care of unvisited vertices here,
    // The other are by default unvisited
    // If a vertex remain unvisited, we will notice it here
    // That's why by default we consider everything visited ( ie ~0 )
    // If we fail to assign one parent to an unvisited vertex, we will
    // explicitly unset the bit
    int local_visited_bmap = (~0);
    int found              = 0;
    int more_to_visit      = 0;
    IndexType valid_parent;
    IndexType left_unvisited_off;

    if (idx < unvisited_size) {
      // Processing first STPV edges of unvisited v
      // If bigger than that, push to left_unvisited queue
      unvisited_vertex = unvisited[idx];

      IndexType edge_begin = row_ptr[unvisited_vertex];
      IndexType edge_end   = row_ptr[unvisited_vertex + 1];

      visited_bmap_index[0] = unvisited_vertex / INT_SIZE;

      IndexType degree = edge_end - edge_begin;

      for (IndexType edge = edge_begin;
           edge < min(static_cast<size_t>(edge_end),
                      static_cast<size_t>(edge_begin) + MAIN_BOTTOMUP_MAX_EDGES);
           ++edge) {
        if (edge_mask && !edge_mask[edge]) continue;

        IndexType parent_candidate = col_ind[edge];

        if (distances[parent_candidate] == (lvl - 1)) {
          found        = 1;
          valid_parent = parent_candidate;
          break;
        }
      }

      // This vertex will remain unvisited at the end of this kernel
      // Explicitly say it
      if (!found)
        local_visited_bmap &= ~(1 << (unvisited_vertex % INT_SIZE));  // let this one unvisited
      else {
        if (distances) distances[unvisited_vertex] = lvl;
        if (predecessors) predecessors[unvisited_vertex] = valid_parent;
      }

      // If we haven't found a parent and there's more edge to check
      if (!found && degree > MAIN_BOTTOMUP_MAX_EDGES) {
        left_unvisited_off = traversal::atomicAdd(left_unvisited_cnt, static_cast<IndexType>(1));
        more_to_visit      = 1;
      }
    }

    //
    // We will separate vertices in group
    // Two vertices are in the same group if represented by same int in
    // visited_bmap ie u and v in same group <=> u/32 == v/32
    //
    // We will now flag the head of those group (first element of each group)
    //
    // 1) All vertices within the same group are adjacent in the queue (cf
    // fill_unvisited_queue) 2) A group is of size <= 32, so a warp will contain
    // at least one head, and a group will be contained at most by two warps

    int is_head_a[1];  // CUB need an array
    BlockDiscontinuity(discontinuity_temp_storage)
      .FlagHeads(is_head_a, visited_bmap_index, cub::Inequality());
    int is_head = is_head_a[0];

    // Computing the warp reduce within group
    // This primitive uses the is_head flags to know where the limits of the
    // groups are We use bitwise and as operator, because of the fact that 1 is
    // the default value If a vertex is unvisited, we have to explicitly ask for
    // it
    int local_bmap_agg =
      WarpReduce(reduce_temp_storage)
        .HeadSegmentedReduce(local_visited_bmap, is_head, traversal::BitwiseAnd());

    // We need to take care of the groups cut in two in two different warps
    // Saving second part of the reduce here, then applying it on the first part
    // bellow Corner case : if the first thread of the warp is a head, then this
    // group is not cut in two and then we have to be neutral (for an bitwise
    // and, it's an ~0)
    if (laneid == 0) { local_visited_bmap_warp_head[warpid] = (is_head) ? (~0) : local_bmap_agg; }

    // broadcasting local_visited_bmap_warp_head
    __syncthreads();

    int head_ballot = __ballot_sync(raft::warp_full_mask(), is_head);

    // As long as idx < unvisited_size, we know there's at least one head per
    // warp
    int laneid_last_head_in_warp = INT_SIZE - 1 - __clz(head_ballot);

    int is_last_head_in_warp = (laneid == laneid_last_head_in_warp);

    // if laneid == 0 && is_last_head_in_warp, it's a special case where
    // a group of size 32 starts exactly at lane 0
    // in that case, nothing to do (this group is not cut by a warp
    // delimitation) we also have to make sure that a warp actually exists after
    // this one (this corner case is handled after)
    if (laneid != 0 && (is_last_head_in_warp & (warpid + 1) < MAIN_BOTTOMUP_NWARPS)) {
      local_bmap_agg &= local_visited_bmap_warp_head[warpid + 1];
    }

    // Three cases :
    // -> This is the first group of the block - it may be cut in two (with
    // previous block)
    // -> This is the last group of the block - same thing
    // -> This group is completely contained in this block

    if (warpid == 0 && laneid == 0) {
      // The first elt of this group considered in this block is
      // unvisited_vertex We know that's the case because elts are sorted in a
      // group, and we are at laneid == 0 We will do an atomicOr - we have to be
      // neutral about elts < unvisited_vertex
      int iv   = unvisited_vertex % INT_SIZE;  // we know that this unvisited_vertex is valid
      int mask = traversal::getMaskNLeftmostBitSet(INT_SIZE - iv);
      local_bmap_agg &= mask;  // we have to be neutral for elts < unvisited_vertex
      atomicOr(&visited_bmap[unvisited_vertex / INT_SIZE], local_bmap_agg);
    } else if (warpid == (MAIN_BOTTOMUP_NWARPS - 1) &&
               laneid >= laneid_last_head_in_warp &&  // We need the other ones
                                                      // to go in else case
               idx < unvisited_size                   // we could be out
    ) {
      // Last head of the block
      // We don't know if this group is complete

      // last_v is the last unvisited_vertex of the group IN THIS block
      // we dont know about the rest - we have to be neutral about elts > last_v

      // the destination thread of the __shfl is active
      int laneid_max =
        min(static_cast<IndexType>(WARP_SIZE - 1), (unvisited_size - (block_off + 32 * warpid)));
      IndexType last_v = __shfl_sync(__activemask(), unvisited_vertex, laneid_max, WARP_SIZE);

      if (is_last_head_in_warp) {
        int ilast_v = last_v % INT_SIZE + 1;
        int mask    = traversal::getMaskNRightmostBitSet(ilast_v);
        local_bmap_agg &= mask;  // we have to be neutral for elts > last_unvisited_vertex
        atomicOr(&visited_bmap[unvisited_vertex / INT_SIZE], local_bmap_agg);
      }
    } else {
      // group completely in block
      if (is_head && idx < unvisited_size) {
        visited_bmap[unvisited_vertex / INT_SIZE] = local_bmap_agg;  // no atomics needed, we know
                                                                     // everything about this int
      }
    }

    // Saving in frontier

    int thread_frontier_offset;
    BlockScan(scan_temp_storage).ExclusiveSum(found, thread_frontier_offset);
    IndexType inclusive_sum = thread_frontier_offset + found;
    if (threadIdx.x == (MAIN_BOTTOMUP_DIMX - 1) && inclusive_sum) {
      frontier_common_block_offset = traversal::atomicAdd(new_frontier_cnt, inclusive_sum);
    }

    // 1) Broadcasting frontier_common_block_offset
    // 2) we want to reuse the *_temp_storage
    __syncthreads();

    if (found)
      new_frontier[frontier_common_block_offset + thread_frontier_offset] = unvisited_vertex;
    if (more_to_visit) left_unvisited[left_unvisited_off] = unvisited_vertex;
  }
}

template <typename IndexType>
void bottom_up_main(IndexType* unvisited,
                    IndexType unvisited_size,
                    IndexType* left_unvisited,
                    IndexType* d_left_unvisited_idx,
                    int* visited,
                    const IndexType* row_ptr,
                    const IndexType* col_ind,
                    IndexType lvl,
                    IndexType* new_frontier,
                    IndexType* new_frontier_idx,
                    IndexType* distances,
                    IndexType* predecessors,
                    int* edge_mask,
                    cudaStream_t m_stream,
                    bool deterministic)
{
  dim3 grid, block;
  block.x = MAIN_BOTTOMUP_DIMX;

  grid.x = std::min(static_cast<size_t>(MAXBLOCKS),
                    (static_cast<size_t>(unvisited_size) + block.x) / block.x);

  main_bottomup_kernel<<<grid, block, 0, m_stream>>>(unvisited,
                                                     unvisited_size,
                                                     left_unvisited,
                                                     d_left_unvisited_idx,
                                                     visited,
                                                     row_ptr,
                                                     col_ind,
                                                     lvl,
                                                     new_frontier,
                                                     new_frontier_idx,
                                                     distances,
                                                     predecessors,
                                                     edge_mask);
  RAFT_CHECK_CUDA(m_stream);
}

//
// bottom_up_large_degree_kernel
// finishing the work started in main_bottomup_kernel for vertex with degree >
// MAIN_BOTTOMUP_MAX_EDGES && no parent found
//
template <typename IndexType>
__global__ void bottom_up_large_degree_kernel(IndexType* left_unvisited,
                                              IndexType left_unvisited_size,
                                              int* visited,
                                              const IndexType* row_ptr,
                                              const IndexType* col_ind,
                                              IndexType lvl,
                                              IndexType* new_frontier,
                                              IndexType* new_frontier_cnt,
                                              IndexType* distances,
                                              IndexType* predecessors,
                                              int* edge_mask)
{
  int logical_lane_id         = threadIdx.x % BOTTOM_UP_LOGICAL_WARP_SIZE;
  int logical_warp_id         = threadIdx.x / BOTTOM_UP_LOGICAL_WARP_SIZE;
  int logical_warps_per_block = blockDim.x / BOTTOM_UP_LOGICAL_WARP_SIZE;

  // When this kernel is converted to support different VT and ET, this
  // will likely split into invalid_vid and invalid_eid
  // This is equivalent to ~IndexType(0) (i.e., all bits set to 1)
  constexpr IndexType invalid_idx = cugraph::legacy::invalid_idx<IndexType>::value;

  // Inactive threads are not a pb for __ballot (known behaviour)
  for (IndexType idx = logical_warps_per_block * blockIdx.x + logical_warp_id;
       idx < left_unvisited_size;
       idx += gridDim.x * logical_warps_per_block) {
    // Unvisited vertices - potentially in the next frontier
    IndexType v = left_unvisited[idx];

    // Used only with symmetric graphs
    // Parents are included in v's neighbors
    IndexType first_i_edge = row_ptr[v] + MAIN_BOTTOMUP_MAX_EDGES;  // we already have checked the
                                                                    // first MAIN_BOTTOMUP_MAX_EDGES
                                                                    // edges in find_unvisited

    IndexType end_i_edge = row_ptr[v + 1];

    // We can have warp divergence in the next loop
    // It's not a pb because the behaviour of __ballot
    // is know with inactive threads
    for (IndexType i_edge = first_i_edge + logical_lane_id; i_edge < end_i_edge;
         i_edge += BOTTOM_UP_LOGICAL_WARP_SIZE) {
      IndexType valid_parent = invalid_idx;

      if (!edge_mask || edge_mask[i_edge]) {
        IndexType u     = col_ind[i_edge];
        IndexType lvl_u = distances[u];

        if (lvl_u == (lvl - 1)) { valid_parent = u; }
      }

      unsigned int warp_valid_p_ballot =
        __ballot_sync(raft::warp_full_mask(), valid_parent != invalid_idx);

      int logical_warp_id_in_warp = (threadIdx.x % WARP_SIZE) / BOTTOM_UP_LOGICAL_WARP_SIZE;
      unsigned int mask           = (1 << BOTTOM_UP_LOGICAL_WARP_SIZE) - 1;
      unsigned int logical_warp_valid_p_ballot =
        warp_valid_p_ballot >> (BOTTOM_UP_LOGICAL_WARP_SIZE * logical_warp_id_in_warp);
      logical_warp_valid_p_ballot &= mask;

      int chosen_thread = __ffs(logical_warp_valid_p_ballot) - 1;

      if (chosen_thread == logical_lane_id) {
        // Using only one valid parent (reduce bw)
        IndexType off = traversal::atomicAdd(new_frontier_cnt, static_cast<IndexType>(1));
        int m         = 1 << (v % INT_SIZE);
        atomicOr(&visited[v / INT_SIZE], m);
        distances[v] = lvl;

        if (predecessors) predecessors[v] = valid_parent;

        new_frontier[off] = v;
      }

      if (logical_warp_valid_p_ballot) { break; }
    }
  }
}

template <typename IndexType>
void bottom_up_large(IndexType* left_unvisited,
                     IndexType left_unvisited_size,
                     int* visited,
                     const IndexType* row_ptr,
                     const IndexType* col_ind,
                     IndexType lvl,
                     IndexType* new_frontier,
                     IndexType* new_frontier_idx,
                     IndexType* distances,
                     IndexType* predecessors,
                     int* edge_mask,
                     cudaStream_t m_stream,
                     bool deterministic)
{
  dim3 grid, block;
  block.x = LARGE_BOTTOMUP_DIMX;
  grid.x  = std::min(
    static_cast<size_t>(MAXBLOCKS),
    ((static_cast<size_t>(left_unvisited_size) + block.x - 1) * BOTTOM_UP_LOGICAL_WARP_SIZE) /
      block.x);

  bottom_up_large_degree_kernel<<<grid, block, 0, m_stream>>>(left_unvisited,
                                                              left_unvisited_size,
                                                              visited,
                                                              row_ptr,
                                                              col_ind,
                                                              lvl,
                                                              new_frontier,
                                                              new_frontier_idx,
                                                              distances,
                                                              predecessors,
                                                              edge_mask);
  RAFT_CHECK_CUDA(m_stream);
}

//
// topdown_expand_kernel
// Read current frontier and compute new one with top down paradigm
// One thread = One edge
// To know origin of edge, we have to find where is index_edge in the values of
// frontier_degrees_exclusive_sum (using a binary search, max less or equal
// than) This index k will give us the origin of this edge, which is frontier[k]
// This thread will then process the (linear_idx_thread -
// frontier_degrees_exclusive_sum[k])-ith edge of vertex frontier[k]
//
// To process blockDim.x = TOP_DOWN_EXPAND_DIMX edges, we need to first load
// NBUCKETS_PER_BLOCK bucket offsets - those will help us do the binary searches
// We can load up to TOP_DOWN_EXPAND_DIMX of those bucket offsets - that way we
// prepare for the next MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD
// * blockDim.x edges
//
// Once we have those offsets, we may still need a few values from
// frontier_degrees_exclusive_sum to compute exact index k To be able to do it,
// we will load the values that we need from frontier_degrees_exclusive_sum in
// shared memory We know that it will fit because we never add node with degree
// == 0 in the frontier, so we have an upper bound on the number of value to
// load (see below)
//
// We will then look which vertices are not visited yet :
// 1) if the unvisited vertex is isolated (=> degree == 0), we mark it as
// visited, update distances and predecessors, and move on 2) if the unvisited
// vertex has degree > 0, we add it to the "frontier_candidates" queue
//
// We then treat the candidates queue using the threadIdx.x < ncandidates
// If we are indeed the first thread to discover that vertex (result of
// atomicOr(visited)) We add it to the new frontier
//

template <typename IndexType>
__global__ void topdown_expand_kernel(
  const IndexType* row_ptr,
  const IndexType* col_ind,
  const IndexType* frontier,
  const IndexType frontier_size,
  const IndexType totaldegree,
  const IndexType max_items_per_thread,
  const IndexType lvl,
  IndexType* new_frontier,
  IndexType* new_frontier_cnt,
  const IndexType* frontier_degrees_exclusive_sum,
  const IndexType* frontier_degrees_exclusive_sum_buckets_offsets,
  int* previous_bmap,
  int* bmap,
  IndexType* distances,
  IndexType* predecessors,
  double* sp_counters,
  const int* edge_mask,
  const int* isolated_bmap,
  bool directed)
{
  // BlockScan
  typedef cub::BlockScan<IndexType, TOP_DOWN_EXPAND_DIMX> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_storage;

  // We will do a scan to know where to write in frontier
  // This will contain the common offset of the block
  __shared__ IndexType frontier_common_block_offset;

  __shared__ IndexType shared_buckets_offsets[TOP_DOWN_EXPAND_DIMX - NBUCKETS_PER_BLOCK + 1];
  __shared__ IndexType shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX + 1];

  //
  // Frontier candidates local queue
  // We process TOP_DOWN_BATCH_SIZE vertices in parallel, so we need to be able
  // to store everything We also save the predecessors here, because we will not
  // be able to retrieve it after
  //
  __shared__ IndexType
    shared_local_new_frontier_candidates[TOP_DOWN_BATCH_SIZE * TOP_DOWN_EXPAND_DIMX];
  __shared__ IndexType
    shared_local_new_frontier_predecessors[TOP_DOWN_BATCH_SIZE * TOP_DOWN_EXPAND_DIMX];
  __shared__ IndexType block_n_frontier_candidates;

  IndexType block_offset = (blockDim.x * blockIdx.x) * max_items_per_thread;

  // When this kernel is converted to support different VT and ET, this
  // will likely split into invalid_vid and invalid_eid
  // This is equivalent to ~IndexType(0) (i.e., all bits set to 1)
  constexpr IndexType invalid_idx = cugraph::legacy::invalid_idx<IndexType>::value;

  IndexType n_items_per_thread_left =
    (totaldegree > block_offset)
      ? (totaldegree - block_offset + TOP_DOWN_EXPAND_DIMX - 1) / TOP_DOWN_EXPAND_DIMX
      : 0;

  n_items_per_thread_left = min(max_items_per_thread, n_items_per_thread_left);

  for (; (n_items_per_thread_left > 0) && (block_offset < totaldegree);

       block_offset += MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD * blockDim.x,
       n_items_per_thread_left -= min(
         n_items_per_thread_left, static_cast<IndexType>(MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD))) {
    // In this loop, we will process batch_set_size batches
    IndexType nitems_per_thread =
      min(n_items_per_thread_left, static_cast<IndexType>(MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD));

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
    // frontier_degrees_exclusive_sum (see below) We will load them here We will
    // load as much as we can - if it doesn't fit we will make multiple
    // iteration of the next loop Because all vertices in frontier have degree >
    // 0, we know it will fits if left + 1 = right (see below)

    // We're going to load values in frontier_degrees_exclusive_sum for batch
    // [left; right[ If it doesn't fit, --right until it does, then loop It is
    // excepted to fit on the first try, that's why we start right =
    // nitems_per_thread

    IndexType left  = 0;
    IndexType right = nitems_per_thread;

    while (left < nitems_per_thread) {
      //
      // Values that are necessary to compute the local binary searches
      // We only need those with indexes between extremes indexes of
      // buckets_offsets We need the next val for the binary search, hence the
      // +1
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

      // shared_frontier_degrees_exclusive_sum is in shared mem, we will use it,
      // sync
      __syncthreads();

      // Now we will process the edges
      // Here each thread will process nitems_per_thread_for_this_load
      for (IndexType item_index = 0; item_index < nitems_per_thread_for_this_load;
           item_index += TOP_DOWN_BATCH_SIZE) {
        // We process TOP_DOWN_BATCH_SIZE edge in parallel (instruction
        // parallism) Reduces latency

        IndexType current_max_edge_index = min(
          static_cast<size_t>(block_offset) + (left + nitems_per_thread_for_this_load) * blockDim.x,
          static_cast<size_t>(totaldegree));

        // We will need vec_u (source of the edge) until the end if we need to
        // save the predecessors For others informations, we will reuse pointers
        // on the go (nvcc does not color well the registers in that case)

        IndexType vec_u[TOP_DOWN_BATCH_SIZE];
        IndexType local_buf1[TOP_DOWN_BATCH_SIZE];
        IndexType local_buf2[TOP_DOWN_BATCH_SIZE];

        IndexType* vec_frontier_degrees_exclusive_sum_index = &local_buf2[0];

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
            vec_u[iv]                                    = frontier[k];  // origin of this edge
            vec_frontier_degrees_exclusive_sum_index[iv] = frontier_degrees_exclusive_sum[k];
          } else {
            vec_u[iv]                                    = invalid_idx;
            vec_frontier_degrees_exclusive_sum_index[iv] = invalid_idx;
          }
        }

        IndexType* vec_row_ptr_u = &local_buf1[0];
#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType u = vec_u[iv];
          // row_ptr for this vertex origin u
          vec_row_ptr_u[iv] = (u != invalid_idx) ? row_ptr[u] : invalid_idx;
        }

        // We won't need row_ptr after that, reusing pointer
        IndexType* vec_dest_v = vec_row_ptr_u;

#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType thread_item_index = left + item_index + iv;
          IndexType gid               = block_offset + thread_item_index * blockDim.x + threadIdx.x;

          IndexType row_ptr_u = vec_row_ptr_u[iv];
          // Need this check so that we don't use invalid values of edge to index
          if (row_ptr_u != invalid_idx) {
            IndexType edge = row_ptr_u + gid - vec_frontier_degrees_exclusive_sum_index[iv];

            if (edge_mask && !edge_mask[edge]) {
              // Disabling edge
              row_ptr_u = invalid_idx;
            } else {
              // Destination of this edge
              vec_dest_v[iv] = col_ind[edge];
            }
          }
        }

        // We don't need vec_frontier_degrees_exclusive_sum_index anymore
        IndexType* vec_v_visited_bmap = vec_frontier_degrees_exclusive_sum_index;

        // Visited bmap need to contain information about the previous
        // frontier if we actually process every edge (shortest path counting)
        // otherwise we can read and update from the same bmap
#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType v = vec_dest_v[iv];
          vec_v_visited_bmap[iv] =
            (v != invalid_idx) ? previous_bmap[v / INT_SIZE] : (~int(0));  // will look visited
        }

        // From now on we will consider v as a frontier candidate
        // If for some reason vec_candidate[iv] should be put in the
        // new_frontier Then set vec_candidate[iv] = -1
        IndexType* vec_frontier_candidate = vec_dest_v;

#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType v = vec_frontier_candidate[iv];
          int m       = 1 << (v % INT_SIZE);

          int is_visited = vec_v_visited_bmap[iv] & m;

          if (is_visited) vec_frontier_candidate[iv] = invalid_idx;
        }

        // Each source should update the destination shortest path counter
        // if the destination has not been visited in the *previous* frontier
        if (sp_counters) {
#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType dst = vec_frontier_candidate[iv];
            if (dst != invalid_idx) {
              IndexType src = vec_u[iv];
              atomicAdd(&sp_counters[dst], sp_counters[src]);
            }
          }
        }

        if (directed) {
          // vec_v_visited_bmap is available
          IndexType* vec_is_isolated_bmap = vec_v_visited_bmap;

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType v              = vec_frontier_candidate[iv];
            vec_is_isolated_bmap[iv] = (v != invalid_idx) ? isolated_bmap[v / INT_SIZE] : ~int(0);
          }

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType v     = vec_frontier_candidate[iv];
            int m           = 1 << (v % INT_SIZE);
            int is_isolated = vec_is_isolated_bmap[iv] & m;

            // If v is isolated, we will not add it to the frontier (it's not a
            // frontier candidate) 1st reason : it's useless 2nd reason : it
            // will make top down algo fail we need each node in frontier to
            // have a degree > 0 If it is isolated, we just need to mark it as
            // visited, and save distance and predecessor here. Not need to
            // check return value of atomicOr

            if (is_isolated && v != invalid_idx) {
              int m = 1 << (v % INT_SIZE);
              atomicOr(&bmap[v / INT_SIZE], m);
              if (distances) distances[v] = lvl;

              if (predecessors) predecessors[v] = vec_u[iv];

              // This is no longer a candidate, neutralize it
              vec_frontier_candidate[iv] = invalid_idx;
            }
          }
        }

        // Number of successor candidate hold by this thread
        IndexType thread_n_frontier_candidates = 0;

#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          IndexType v = vec_frontier_candidate[iv];
          if (v != invalid_idx) ++thread_n_frontier_candidates;
        }

        // We need to have all nfrontier_candidates to be ready before doing the
        // scan
        __syncthreads();

        // We will put the frontier candidates in a local queue
        // Computing offsets
        IndexType thread_frontier_candidate_offset = 0;  // offset inside block
        BlockScan(scan_storage)
          .ExclusiveSum(thread_n_frontier_candidates, thread_frontier_candidate_offset);

#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          // May have bank conflicts
          IndexType frontier_candidate = vec_frontier_candidate[iv];

          if (frontier_candidate != invalid_idx) {
            shared_local_new_frontier_candidates[thread_frontier_candidate_offset] =
              frontier_candidate;
            shared_local_new_frontier_predecessors[thread_frontier_candidate_offset] = vec_u[iv];
            ++thread_frontier_candidate_offset;
          }
        }

        if (threadIdx.x == (TOP_DOWN_EXPAND_DIMX - 1)) {
          // No need to add nsuccessor_candidate, even if its an
          // exclusive sum
          // We incremented the thread_frontier_candidate_offset
          block_n_frontier_candidates = thread_frontier_candidate_offset;
        }

        // broadcast block_n_frontier_candidates
        __syncthreads();

        IndexType naccepted_vertices = 0;
        // We won't need vec_frontier_candidate after that
        IndexType* vec_frontier_accepted_vertex = vec_frontier_candidate;

#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          const int idx_shared             = iv * blockDim.x + threadIdx.x;
          vec_frontier_accepted_vertex[iv] = invalid_idx;

          if (idx_shared < block_n_frontier_candidates) {
            IndexType v = shared_local_new_frontier_candidates[idx_shared];  // popping
                                                                             // queue
            int m = 1 << (v % INT_SIZE);
            int q = atomicOr(&bmap[v / INT_SIZE], m);  // atomicOr returns old

            if (!(m & q)) {  // if this thread was the first to discover this node
              if (distances) distances[v] = lvl;

              if (predecessors) {
                IndexType pred  = shared_local_new_frontier_predecessors[idx_shared];
                predecessors[v] = pred;
              }

              vec_frontier_accepted_vertex[iv] = v;
              ++naccepted_vertices;
            }
          }
        }

        // We need naccepted_vertices to be ready
        __syncthreads();

        IndexType thread_new_frontier_offset;

        BlockScan(scan_storage).ExclusiveSum(naccepted_vertices, thread_new_frontier_offset);

        if (threadIdx.x == (TOP_DOWN_EXPAND_DIMX - 1)) {
          IndexType inclusive_sum = thread_new_frontier_offset + naccepted_vertices;
          // for this thread, thread_new_frontier_offset + has_successor
          // (exclusive sum)
          if (inclusive_sum)
            frontier_common_block_offset = traversal::atomicAdd(new_frontier_cnt, inclusive_sum);
        }

        // Broadcasting frontier_common_block_offset
        __syncthreads();

#pragma unroll
        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
          const int idx_shared = iv * blockDim.x + threadIdx.x;
          if (idx_shared < block_n_frontier_candidates) {
            IndexType new_frontier_vertex = vec_frontier_accepted_vertex[iv];

            if (new_frontier_vertex != invalid_idx) {
              IndexType off     = frontier_common_block_offset + thread_new_frontier_offset++;
              new_frontier[off] = new_frontier_vertex;
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

template <typename IndexType>
void frontier_expand(const IndexType* row_ptr,
                     const IndexType* col_ind,
                     const IndexType* frontier,
                     const IndexType frontier_size,
                     const IndexType totaldegree,
                     const IndexType lvl,
                     IndexType* new_frontier,
                     IndexType* new_frontier_cnt,
                     const IndexType* frontier_degrees_exclusive_sum,
                     const IndexType* frontier_degrees_exclusive_sum_buckets_offsets,
                     int* previous_visited_bmap,
                     int* visited_bmap,
                     IndexType* distances,
                     IndexType* predecessors,
                     double* sp_counters,
                     const int* edge_mask,
                     const int* isolated_bmap,
                     bool directed,
                     cudaStream_t m_stream,
                     bool deterministic)
{
  if (!totaldegree) return;

  dim3 block;
  block.x = TOP_DOWN_EXPAND_DIMX;

  IndexType max_items_per_thread =
    (static_cast<size_t>(totaldegree) + MAXBLOCKS * block.x - 1) / (MAXBLOCKS * block.x);

  dim3 grid;
  grid.x = std::min((static_cast<size_t>(totaldegree) + max_items_per_thread * block.x - 1) /
                      (max_items_per_thread * block.x),
                    static_cast<size_t>(MAXBLOCKS));

  // Shortest Path counting (Betweenness Centrality)
  // We need to keep track of the previously visited bmap

  // If the coutner of shortest path is nullptr
  // The previous_visited_bmap is no longer needed (and should be nullptr on
  // the first access), so it can be the same as the current visitedbmap
  if (!sp_counters) { previous_visited_bmap = visited_bmap; }
  topdown_expand_kernel<<<grid, block, 0, m_stream>>>(
    row_ptr,
    col_ind,
    frontier,
    frontier_size,
    totaldegree,
    max_items_per_thread,
    lvl,
    new_frontier,
    new_frontier_cnt,
    frontier_degrees_exclusive_sum,
    frontier_degrees_exclusive_sum_buckets_offsets,
    previous_visited_bmap,
    visited_bmap,
    distances,
    predecessors,
    sp_counters,
    edge_mask,
    isolated_bmap,
    directed);
  RAFT_CHECK_CUDA(m_stream);
}

}  // namespace bfs_kernels
}  // namespace detail
}  // namespace cugraph
