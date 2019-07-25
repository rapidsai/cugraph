
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
#include <iostream>

#include "include/sm_utils.h"
#include <cub/cub.cuh>

#include <rmm/rmm.h>

#include "include/nvgraph_error.hxx"

#define MAXBLOCKS 65535
#define WARP_SIZE 32
#define INT_SIZE 32

//
// Bottom up macros
//

#define FILL_UNVISITED_QUEUE_DIMX 256

#define COUNT_UNVISITED_EDGES_DIMX 256

#define MAIN_BOTTOMUP_DIMX 256
#define MAIN_BOTTOMUP_NWARPS (MAIN_BOTTOMUP_DIMX/WARP_SIZE)

#define LARGE_BOTTOMUP_DIMX 256

//Number of edges processed in the main bottom up kernel
#define MAIN_BOTTOMUP_MAX_EDGES 6

//Power of 2 < 32 (strict <)
#define BOTTOM_UP_LOGICAL_WARP_SIZE 4

//
// Top down macros
//

// We will precompute the results the binsearch_maxle every TOP_DOWN_BUCKET_SIZE edges
#define TOP_DOWN_BUCKET_SIZE 32

// DimX of the kernel
#define TOP_DOWN_EXPAND_DIMX 256

// TOP_DOWN_EXPAND_DIMX edges -> NBUCKETS_PER_BLOCK buckets
#define NBUCKETS_PER_BLOCK (TOP_DOWN_EXPAND_DIMX/TOP_DOWN_BUCKET_SIZE)

// How many items_per_thread we can process with one bucket_offset loading
// the -1 is here because we need the +1 offset
#define MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD (TOP_DOWN_BUCKET_SIZE - 1)

// instruction parallelism
// for how many edges will we create instruction parallelism
#define TOP_DOWN_BATCH_SIZE 2

#define COMPUTE_BUCKET_OFFSETS_DIMX 512

//Other macros

#define FLAG_ISOLATED_VERTICES_DIMX 128

//Number of vertices handled by one thread
//Must be power of 2, lower than 32
#define FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD 4 

//Number of threads involved in the "construction" of one int in the bitset
#define FLAG_ISOLATED_VERTICES_THREADS_PER_INT (INT_SIZE/FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD)

//
// Parameters of the heuristic to switch between bottomup/topdown
//Finite machine described in http://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf 
//

using namespace nvgraph;

namespace bfs_kernels {
  //
  // gives the equivalent vectors from a type
  // for the max val, would be better to use numeric_limits<>::max() once
  // cpp11 is allowed in nvgraph
  //

  template<typename >
  struct vec_t {
    typedef int4 vec4;
    typedef int2 vec2;
  };

  template<>
  struct vec_t<int> {
    typedef int4 vec4;
    typedef int2 vec2;
    static const int max = INT_MAX;
  };

  template<>
  struct vec_t<long long int> {
    typedef longlong4 vec4;
    typedef longlong2 vec2;
    static const long long int max = LLONG_MAX;
  };

  //
  // ------------------------- Helper device functions -------------------
  //

  __forceinline__ __device__ int getMaskNRightmostBitSet(int n) {
    if (n == INT_SIZE)
      return (~0);
    int mask = (1 << n) - 1;
    return mask;
  }

  __forceinline__ __device__ int getMaskNLeftmostBitSet(int n) {
    if (n == 0)
      return 0;
    int mask = ~((1 << (INT_SIZE - n)) - 1);
    return mask;
  }

  __forceinline__ __device__ int getNextZeroBit(int& val) {
    int ibit = __ffs(~val) - 1;
    val |= (1 << ibit);

    return ibit;
  }

  struct BitwiseAnd
  {
    template<typename T>
    __host__  __device__  __forceinline__ T operator()(const T &a, const T &b) const
                                      {
      return (a & b);
    }
  };

  struct BitwiseOr
  {
    template<typename T>
    __host__  __device__  __forceinline__ T operator()(const T &a, const T &b) const
                                      {
      return (a | b);
    }
  };

  template<typename IndexType>
  __device__ IndexType binsearch_maxle(  const IndexType *vec,
                            const IndexType val,
                            IndexType low,
                            IndexType high) {
    while (true) {
      if (low == high)
        return low; //we know it exists
      if ((low + 1) == high)
        return (vec[high] <= val) ? high : low;

      IndexType mid = low + (high - low) / 2;

      if (vec[mid] > val)
        high = mid - 1;
      else
        low = mid;

    }
  }

  //
  //  -------------------------  Bottom up -------------------------
  //

  //
  // fill_unvisited_queue_kernel
  //
  // Finding unvisited vertices in the visited_bmap, and putting them in the queue
  // Vertices represented by the same int in the bitmap are adjacent in the queue, and sorted
  // For instance, the queue can look like this :
  // 34 38 45 58 61 4 18 24 29 71 84 85 90
  // Because they are represented by those ints in the bitmap :
  // [34 38 45 58 61] [4 18 24 29] [71 84 85 90]

  //visited_bmap_nints = the visited_bmap is made of that number of ints

  template<typename IndexType>
  __global__ void fill_unvisited_queue_kernel(  int *visited_bmap,
                                IndexType visited_bmap_nints,
                                IndexType n,
                                IndexType *unvisited,
                                IndexType *unvisited_cnt) {
    typedef cub::BlockScan<int, FILL_UNVISITED_QUEUE_DIMX> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_temp_storage;

    //When filling the "unvisited" queue, we use "unvisited_cnt" to know where to write in the queue (equivalent of int off = atomicAddd(unvisited_cnt, 1) )
    //We will actually do only one atomicAdd per block - we first do a scan, then call one atomicAdd, and store the common offset for the block in
    //unvisited_common_block_offset
    __shared__ IndexType unvisited_common_block_offset;

    //We don't want threads divergence in the loop (we're going to call __syncthreads)
    //Using a block-only dependent in the condition of the loop
    for (IndexType block_v_idx = blockIdx.x * blockDim.x;
        block_v_idx < visited_bmap_nints;
        block_v_idx += blockDim.x * gridDim.x) {

      //Index of visited_bmap that this thread will compute
      IndexType v_idx = block_v_idx + threadIdx.x;

      int thread_visited_int = (v_idx < visited_bmap_nints)
                        ? visited_bmap[v_idx]
                          :
                          (~0); //will be neutral in the next lines (virtual vertices all visited)

      //The last int can only be partially valid
      //If we are indeed taking care of the last visited int in this thread,
      //We need to first disable (ie set as "visited") the inactive bits (vertices >= n)
      if (v_idx == (visited_bmap_nints - 1)) {
        int active_bits = n - (INT_SIZE * v_idx);
        int inactive_bits = INT_SIZE - active_bits;
        int mask = getMaskNLeftmostBitSet(inactive_bits);
        thread_visited_int |= mask; //Setting inactive bits as visited
      }

      //Counting number of unvisited vertices represented by this int
      int n_unvisited_in_int = __popc(~thread_visited_int);
      int unvisited_thread_offset;

      //We will need to write n_unvisited_in_int unvisited vertices to the unvisited queue
      //We ask for that space when computing the block scan, that will tell where to write those
      //vertices in the queue, using the common offset of the block (see below)
      BlockScan(scan_temp_storage).ExclusiveSum(n_unvisited_in_int, unvisited_thread_offset);

      //Last thread knows how many vertices will be written to the queue by this block
      //Asking for that space in the queue using the global count, and saving the common offset
      if (threadIdx.x == (FILL_UNVISITED_QUEUE_DIMX - 1)) {
        IndexType total = unvisited_thread_offset + n_unvisited_in_int;
        unvisited_common_block_offset = atomicAdd(unvisited_cnt, total);
      }

      //syncthreads for two reasons : 
      // - we need to broadcast unvisited_common_block_offset
      // - we will reuse scan_temp_storage (cf CUB doc)
      __syncthreads();

      IndexType current_unvisited_index = unvisited_common_block_offset
          + unvisited_thread_offset;
      int nvertices_to_write = n_unvisited_in_int;

      // getNextZeroBit uses __ffs, which gives least significant bit set
      // which means that as long as n_unvisited_in_int is valid,
      // we will use valid bits

      while (nvertices_to_write > 0) {
        if (nvertices_to_write >= 4 && (current_unvisited_index % 4) == 0) {
          typename vec_t<IndexType>::vec4 vec_v;

          vec_v.x = v_idx * INT_SIZE + getNextZeroBit(thread_visited_int);
          vec_v.y = v_idx * INT_SIZE + getNextZeroBit(thread_visited_int);
          vec_v.z = v_idx * INT_SIZE + getNextZeroBit(thread_visited_int);
          vec_v.w = v_idx * INT_SIZE + getNextZeroBit(thread_visited_int);

          typename vec_t<IndexType>::vec4 *unvisited_i4 = reinterpret_cast<typename vec_t<
              IndexType>::vec4*>(&unvisited[current_unvisited_index]);
          *unvisited_i4 = vec_v;

          current_unvisited_index += 4;
          nvertices_to_write -= 4;
        }
        else if (nvertices_to_write >= 2 && (current_unvisited_index % 2) == 0) {
          typename vec_t<IndexType>::vec2 vec_v;

          vec_v.x = v_idx * INT_SIZE + getNextZeroBit(thread_visited_int);
          vec_v.y = v_idx * INT_SIZE + getNextZeroBit(thread_visited_int);

          typename vec_t<IndexType>::vec2 *unvisited_i2 = reinterpret_cast<typename vec_t<
              IndexType>::vec2*>(&unvisited[current_unvisited_index]);
          *unvisited_i2 = vec_v;

          current_unvisited_index += 2;
          nvertices_to_write -= 2;
        } else {
          IndexType v = v_idx * INT_SIZE + getNextZeroBit(thread_visited_int);

          unvisited[current_unvisited_index] = v;

          current_unvisited_index += 1;
          nvertices_to_write -= 1;
        }

      }
    }
  }

  //Wrapper
  template<typename IndexType>
  void fill_unvisited_queue(  int *visited_bmap,
                    IndexType visited_bmap_nints,
                    IndexType n,
                    IndexType *unvisited,
                    IndexType *unvisited_cnt,
                    cudaStream_t m_stream,
                    bool deterministic) {
    dim3 grid, block;
    block.x = FILL_UNVISITED_QUEUE_DIMX;

    grid.x = min((IndexType) MAXBLOCKS, (visited_bmap_nints + block.x - 1) / block.x);

    fill_unvisited_queue_kernel<<<grid, block, 0, m_stream>>>(  visited_bmap,
                                            visited_bmap_nints,
                                            n,
                                            unvisited,
                                            unvisited_cnt);
    cudaCheckError()
    ;
  }

  //
  // count_unvisited_edges_kernel
  // Couting the total number of unvisited edges in the graph - using an potentially unvisited queue
  // We need the current unvisited vertices to be in the unvisited queue
  // But visited vertices can be in the potentially_unvisited queue
  // We first check if the vertex is still unvisited before using it
  // Useful when switching from "Bottom up" to "Top down"
  //

  template<typename IndexType>
  __global__ void count_unvisited_edges_kernel(const IndexType *potentially_unvisited,
                                const IndexType potentially_unvisited_size,
                                const int *visited_bmap,
                                IndexType *degree_vertices,
                                IndexType *mu) {
    typedef cub::BlockReduce<IndexType, COUNT_UNVISITED_EDGES_DIMX> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_temp_storage;

    //number of undiscovered edges counted by this thread
    IndexType thread_unvisited_edges_count = 0;

    for (IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < potentially_unvisited_size;
        idx += blockDim.x * gridDim.x) {

      IndexType u = potentially_unvisited[idx];
      int u_visited_bmap = visited_bmap[u / INT_SIZE];
      int is_visited = u_visited_bmap & (1 << (u % INT_SIZE));

      if (!is_visited)
        thread_unvisited_edges_count += degree_vertices[u];

    }

    //We need all thread_unvisited_edges_count to be ready before reducing
    __syncthreads();

    IndexType block_unvisited_edges_count =
        BlockReduce(reduce_temp_storage).Sum(thread_unvisited_edges_count);

    //block_unvisited_edges_count is only defined is th.x == 0
    if (threadIdx.x == 0)
      atomicAdd(mu, block_unvisited_edges_count);
  }

  //Wrapper
  template<typename IndexType>
  void count_unvisited_edges(const IndexType *potentially_unvisited,
                    const IndexType potentially_unvisited_size,
                    const int *visited_bmap,
                    IndexType *node_degree,
                    IndexType *mu,
                    cudaStream_t m_stream) {
    dim3 grid, block;
    block.x = COUNT_UNVISITED_EDGES_DIMX;
    grid.x = min((IndexType) MAXBLOCKS, (potentially_unvisited_size + block.x - 1) / block.x);

    count_unvisited_edges_kernel<<<grid, block, 0, m_stream>>>(  potentially_unvisited,
                                            potentially_unvisited_size,
                                            visited_bmap,
                                            node_degree,
                                            mu);
    cudaCheckError()
    ;
  }

  //
  // Main Bottom Up kernel
  // Here we will start to process unvisited vertices in the unvisited queue
  // We will only consider the first MAIN_BOTTOMUP_MAX_EDGES edges
  // If it's not possible to define a valid parent using only those edges,
  // add it to the "left_unvisited_queue"
  //

  //
  // We will use the "vertices represented by the same int in the visited bmap are adjacents and sorted in the unvisited queue" property
  // It is used to do a reduction locally and fully build the new visited_bmap
  //

  template<typename IndexType>
  __global__ void main_bottomup_kernel(  const IndexType *unvisited,
                            const IndexType unvisited_size,
                            IndexType *left_unvisited,
                            IndexType *left_unvisited_cnt,
                            int *visited_bmap,
                            const IndexType *row_ptr,
                            const IndexType *col_ind,
                            IndexType lvl,
                            IndexType *new_frontier,
                            IndexType *new_frontier_cnt,
                            IndexType *distances,
                            IndexType *predecessors,
                            int *edge_mask) {
    typedef cub::BlockDiscontinuity<IndexType, MAIN_BOTTOMUP_DIMX> BlockDiscontinuity;
    typedef cub::WarpReduce<int> WarpReduce;
    typedef cub::BlockScan<int, MAIN_BOTTOMUP_DIMX> BlockScan;

    __shared__ typename BlockDiscontinuity::TempStorage discontinuity_temp_storage;
    __shared__ typename WarpReduce::TempStorage reduce_temp_storage;
    __shared__ typename BlockScan::TempStorage scan_temp_storage;

    //To write vertices in the frontier,
    //We will use a block scan to locally compute the offsets
    //frontier_common_block_offset contains the common offset for the block
    __shared__ IndexType frontier_common_block_offset;

    // When building the new visited_bmap, we reduce (using a bitwise and) the visited_bmap ints
    // from the vertices represented by the same int (for instance vertices 1, 5, 9, 13, 23)
    // vertices represented by the same int will be designed as part of the same "group"
    // To detect the deliminations between those groups, we use BlockDiscontinuity
    // Then we need to create the new "visited_bmap" within those group.
    // We use a warp reduction that takes into account limits between groups to do it
    // But a group can be cut in two different warps : in that case, the second warp
    // put the result of its local reduction in local_visited_bmap_warp_head
    // the first warp will then read it and finish the reduction

    __shared__ int local_visited_bmap_warp_head[MAIN_BOTTOMUP_NWARPS];

    const int warpid = threadIdx.x / WARP_SIZE;
    const int laneid = threadIdx.x % WARP_SIZE;

    // we will call __syncthreads inside the loop
    // we need to keep complete block active
    for (IndexType block_off = blockIdx.x * blockDim.x;
        block_off < unvisited_size;
        block_off += blockDim.x * gridDim.x)
            {
      IndexType idx = block_off + threadIdx.x;

      // This thread will take care of unvisited_vertex
      // in the visited_bmap, it is represented by the int at index
      // visited_bmap_index = unvisited_vertex/INT_SIZE
      // it will be used by BlockDiscontinuity
      // to flag the separation between groups of vertices (vertices represented by different in in visited_bmap)
      IndexType visited_bmap_index[1]; //this is an array of size 1 because CUB needs one
      visited_bmap_index[0] = -1;
      IndexType unvisited_vertex = -1;

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
      int found = 0;
      int more_to_visit = 0;
      IndexType valid_parent;
      IndexType left_unvisited_off;

      if (idx < unvisited_size)
          {
        //Processing first STPV edges of unvisited v
        //If bigger than that, push to left_unvisited queue
        unvisited_vertex = unvisited[idx];

        IndexType edge_begin = row_ptr[unvisited_vertex];
        IndexType edge_end = row_ptr[unvisited_vertex + 1];

        visited_bmap_index[0] = unvisited_vertex / INT_SIZE;

        IndexType degree = edge_end - edge_begin;

        for (IndexType edge = edge_begin;
            edge < min(edge_end, edge_begin + MAIN_BOTTOMUP_MAX_EDGES); ++edge)
            {
          if (edge_mask && !edge_mask[edge])
            continue;

          IndexType parent_candidate = col_ind[edge];

          if (distances[parent_candidate] == (lvl - 1))
              {
            found = 1;
            valid_parent = parent_candidate;
            break;
          }
        }

        // This vertex will remain unvisited at the end of this kernel
        // Explicitly say it
        if (!found)
          local_visited_bmap &= ~(1 << (unvisited_vertex % INT_SIZE)); //let this one unvisited
        else
        {
          if (distances)
            distances[unvisited_vertex] = lvl;
          if (predecessors)
            predecessors[unvisited_vertex] = valid_parent;
        }

        //If we haven't found a parent and there's more edge to check
        if (!found && degree > MAIN_BOTTOMUP_MAX_EDGES)
        {
          left_unvisited_off = atomicAdd(left_unvisited_cnt, (IndexType) 1); //TODO scan
          more_to_visit = 1;
        }

      }

      //
      // We will separate vertices in group
      // Two vertices are in the same group if represented by same int in visited_bmap
      // ie u and v in same group <=> u/32 == v/32
      //
      // We will now flag the head of those group (first element of each group)
      //
      // 1) All vertices within the same group are adjacent in the queue (cf fill_unvisited_queue)
      // 2) A group is of size <= 32, so a warp will contain at least one head, and a group will be contained
      // at most by two warps

      int is_head_a[1]; //CUB need an array
      BlockDiscontinuity(discontinuity_temp_storage).FlagHeads(is_head_a,
                                            visited_bmap_index,
                                            cub::Inequality());
      int is_head = is_head_a[0];

      // Computing the warp reduce within group
      // This primitive uses the is_head flags to know where the limits of the groups are
      // We use bitwise and as operator, because of the fact that 1 is the default value
      // If a vertex is unvisited, we have to explicitly ask for it
      int local_bmap_agg =
          WarpReduce(reduce_temp_storage).HeadSegmentedReduce(  local_visited_bmap,
                                              is_head,
                                              BitwiseAnd());

      // We need to take care of the groups cut in two in two different warps
      // Saving second part of the reduce here, then applying it on the first part bellow
      // Corner case : if the first thread of the warp is a head, then this group is not cut in two
      // and then we have to be neutral (for an bitwise and, it's an ~0)
      if (laneid == 0)
          {
        local_visited_bmap_warp_head[warpid] = (is_head) ? (~0) : local_bmap_agg;
      }

      //broadcasting local_visited_bmap_warp_head
      __syncthreads();

      int head_ballot = nvgraph::utils::ballot(is_head);

      //As long as idx < unvisited_size, we know there's at least one head per warp
      int laneid_last_head_in_warp = INT_SIZE - 1 - __clz(head_ballot);

      int is_last_head_in_warp = (laneid == laneid_last_head_in_warp);

      // if laneid == 0 && is_last_head_in_warp, it's a special case where
      // a group of size 32 starts exactly at lane 0
      // in that case, nothing to do (this group is not cut by a warp delimitation)
      // we also have to make sure that a warp actually exists after this one (this corner case is handled after)
      if (laneid != 0 && is_last_head_in_warp & (warpid + 1) < MAIN_BOTTOMUP_NWARPS)
      {
        local_bmap_agg &= local_visited_bmap_warp_head[warpid + 1];
      }

      //Three cases :
      // -> This is the first group of the block - it may be cut in two (with previous block)
      // -> This is the last group of the block - same thing
      // -> This group is completely contained in this block

      if (warpid == 0 && laneid == 0)
          {
        //The first elt of this group considered in this block is unvisited_vertex
        //We know that's the case because elts are sorted in a group, and we are at laneid == 0
        //We will do an atomicOr - we have to be neutral about elts < unvisited_vertex
        int iv = unvisited_vertex % INT_SIZE; // we know that this unvisited_vertex is valid
        int mask = getMaskNLeftmostBitSet(INT_SIZE - iv);
        local_bmap_agg &= mask; //we have to be neutral for elts < unvisited_vertex
        atomicOr(&visited_bmap[unvisited_vertex / INT_SIZE], local_bmap_agg);
      }
      else if (warpid == (MAIN_BOTTOMUP_NWARPS - 1) &&
          laneid >= laneid_last_head_in_warp && // We need the other ones to go in else case
          idx < unvisited_size //we could be out
              )
              {
        //Last head of the block
        //We don't know if this group is complete

        //last_v is the last unvisited_vertex of the group IN THIS block
        //we dont know about the rest - we have to be neutral about elts > last_v

        //the destination thread of the __shfl is active
        int laneid_max = min((IndexType) (WARP_SIZE - 1),
                      (unvisited_size - (block_off + 32 * warpid)));
        IndexType last_v = nvgraph::utils::shfl(  unvisited_vertex,
                                    laneid_max,
                                    WARP_SIZE,
                                    __activemask());

        if (is_last_head_in_warp)
        {
          int ilast_v = last_v % INT_SIZE + 1;
          int mask = getMaskNRightmostBitSet(ilast_v);
          local_bmap_agg &= mask; //we have to be neutral for elts > last_unvisited_vertex
          atomicOr(&visited_bmap[unvisited_vertex / INT_SIZE], local_bmap_agg);
        }
      }
      else
      {
        //group completely in block
        if (is_head && idx < unvisited_size) {
          visited_bmap[unvisited_vertex / INT_SIZE] = local_bmap_agg; //no atomics needed, we know everything about this int
        }
      }

      //Saving in frontier

      int thread_frontier_offset;
      BlockScan(scan_temp_storage).ExclusiveSum(found, thread_frontier_offset);
      IndexType inclusive_sum = thread_frontier_offset + found;
      if (threadIdx.x == (MAIN_BOTTOMUP_DIMX - 1) && inclusive_sum)
          {
        frontier_common_block_offset = atomicAdd(new_frontier_cnt, inclusive_sum);
      }

      //1) Broadcasting frontier_common_block_offset
      //2) we want to reuse the *_temp_storage
      __syncthreads();

      if (found)
        new_frontier[frontier_common_block_offset + thread_frontier_offset] = unvisited_vertex;
      if (more_to_visit)
        left_unvisited[left_unvisited_off] = unvisited_vertex;

    }
  }

  template<typename IndexType>
  void bottom_up_main(  IndexType *unvisited,
                IndexType unvisited_size,
                IndexType *left_unvisited,
                IndexType *d_left_unvisited_idx,
                int *visited,
                const IndexType *row_ptr,
                const IndexType *col_ind,
                IndexType lvl,
                IndexType *new_frontier,
                IndexType *new_frontier_idx,
                IndexType *distances,
                IndexType *predecessors,
                int *edge_mask,
                cudaStream_t m_stream,
                bool deterministic) {
    dim3 grid, block;
    block.x = MAIN_BOTTOMUP_DIMX;

    grid.x = min((IndexType) MAXBLOCKS, ((unvisited_size + block.x - 1)) / block.x);

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
    cudaCheckError()
    ;
  }

  //
  // bottom_up_large_degree_kernel
  // finishing the work started in main_bottomup_kernel for vertex with degree > MAIN_BOTTOMUP_MAX_EDGES && no parent found
  //
  template<typename IndexType>
  __global__ void bottom_up_large_degree_kernel(  IndexType *left_unvisited,
                                  IndexType left_unvisited_size,
                                  int *visited,
                                  const IndexType *row_ptr,
                                  const IndexType *col_ind,
                                  IndexType lvl,
                                  IndexType *new_frontier,
                                  IndexType *new_frontier_cnt,
                                  IndexType *distances,
                                  IndexType *predecessors,
                                  int *edge_mask) {

    int logical_lane_id = threadIdx.x % BOTTOM_UP_LOGICAL_WARP_SIZE;
    int logical_warp_id = threadIdx.x / BOTTOM_UP_LOGICAL_WARP_SIZE;
    int logical_warps_per_block = blockDim.x / BOTTOM_UP_LOGICAL_WARP_SIZE;

    //Inactive threads are not a pb for __ballot (known behaviour)
    for (IndexType idx = logical_warps_per_block * blockIdx.x + logical_warp_id;
        idx < left_unvisited_size;
        idx += gridDim.x * logical_warps_per_block) {

      //Unvisited vertices - potentially in the next frontier
      IndexType v = left_unvisited[idx];

      //Used only with symmetric graphs
      //Parents are included in v's neighbors
      IndexType first_i_edge = row_ptr[v] + MAIN_BOTTOMUP_MAX_EDGES; //we already have checked the first MAIN_BOTTOMUP_MAX_EDGES edges in find_unvisited

      IndexType end_i_edge = row_ptr[v + 1];

      //We can have warp divergence in the next loop
      //It's not a pb because the behaviour of __ballot
      //is know with inactive threads
      for (IndexType i_edge = first_i_edge + logical_lane_id;
          i_edge < end_i_edge;
          i_edge += BOTTOM_UP_LOGICAL_WARP_SIZE) {

        IndexType valid_parent = -1;

        if (!edge_mask || edge_mask[i_edge]) {
          IndexType u = col_ind[i_edge];
          IndexType lvl_u = distances[u];

          if (lvl_u == (lvl - 1)) {
            valid_parent = u;
          }
        }

        unsigned int warp_valid_p_ballot = nvgraph::utils::ballot((valid_parent != -1));

        int logical_warp_id_in_warp = (threadIdx.x % WARP_SIZE) / BOTTOM_UP_LOGICAL_WARP_SIZE;
        unsigned int mask = (1 << BOTTOM_UP_LOGICAL_WARP_SIZE) - 1;
        unsigned int logical_warp_valid_p_ballot = warp_valid_p_ballot
            >> (BOTTOM_UP_LOGICAL_WARP_SIZE * logical_warp_id_in_warp);
        logical_warp_valid_p_ballot &= mask;

        int chosen_thread = __ffs(logical_warp_valid_p_ballot) - 1;

        if (chosen_thread == logical_lane_id) {
          //Using only one valid parent (reduce bw)
          IndexType off = atomicAdd(new_frontier_cnt, (IndexType) 1);
          int m = 1 << (v % INT_SIZE);
          atomicOr(&visited[v / INT_SIZE], m);
          distances[v] = lvl;

          if (predecessors)
            predecessors[v] = valid_parent;

          new_frontier[off] = v;
        }

        if (logical_warp_valid_p_ballot) {
          break;
        }
      }

    }
  }

  template<typename IndexType>
  void bottom_up_large(IndexType *left_unvisited,
                IndexType left_unvisited_size,
                int *visited,
                const IndexType *row_ptr,
                const IndexType *col_ind,
                IndexType lvl,
                IndexType *new_frontier,
                IndexType *new_frontier_idx,
                IndexType *distances,
                IndexType *predecessors,
                int *edge_mask,
                cudaStream_t m_stream,
                bool deterministic) {
    dim3 grid, block;
    block.x = LARGE_BOTTOMUP_DIMX;
    grid.x = min(  (IndexType) MAXBLOCKS,
              ((left_unvisited_size + block.x - 1) * BOTTOM_UP_LOGICAL_WARP_SIZE) / block.x);

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
    cudaCheckError()
    ;
  }

  //
  //
  //  ------------------------------ Top down ------------------------------
  //
  //

  //
  // compute_bucket_offsets_kernel
  // simply compute the position in the frontier corresponding all valid edges with index=TOP_DOWN_BUCKET_SIZE * k, k integer
  //

  template<typename IndexType>
  __global__ void compute_bucket_offsets_kernel(  const IndexType *frontier_degrees_exclusive_sum,
                                  IndexType *bucket_offsets,
                                  const IndexType frontier_size,
                                  IndexType total_degree) {
    IndexType end = ((total_degree - 1 + TOP_DOWN_EXPAND_DIMX) / TOP_DOWN_EXPAND_DIMX
        * NBUCKETS_PER_BLOCK + 1);

    for (IndexType bid = blockIdx.x * blockDim.x + threadIdx.x;
        bid <= end;
        bid += gridDim.x * blockDim.x) {

      IndexType eid = min(bid * TOP_DOWN_BUCKET_SIZE, total_degree - 1);

      bucket_offsets[bid] = binsearch_maxle(  frontier_degrees_exclusive_sum,
                                eid,
                                (IndexType) 0,
                                frontier_size - 1);

    }
  }

  template<typename IndexType>
  void compute_bucket_offsets(  IndexType *cumul,
                      IndexType *bucket_offsets,
                      IndexType frontier_size,
                      IndexType total_degree,
                      cudaStream_t m_stream) {
    dim3 grid, block;
    block.x = COMPUTE_BUCKET_OFFSETS_DIMX;

    grid.x = min(  (IndexType) MAXBLOCKS,
              ((total_degree - 1 + TOP_DOWN_EXPAND_DIMX) / TOP_DOWN_EXPAND_DIMX
                  * NBUCKETS_PER_BLOCK + 1 + block.x - 1) / block.x);

    compute_bucket_offsets_kernel<<<grid, block, 0, m_stream>>>(cumul,
                                            bucket_offsets,
                                            frontier_size,
                                            total_degree);
    cudaCheckError()
    ;
  }

  //
  // topdown_expand_kernel
  // Read current frontier and compute new one with top down paradigm
  // One thread = One edge
  // To know origin of edge, we have to find where is index_edge in the values of frontier_degrees_exclusive_sum (using a binary search, max less or equal than)
  // This index k will give us the origin of this edge, which is frontier[k]
  // This thread will then process the (linear_idx_thread - frontier_degrees_exclusive_sum[k])-ith edge of vertex frontier[k]
  //
  // To process blockDim.x = TOP_DOWN_EXPAND_DIMX edges, we need to first load NBUCKETS_PER_BLOCK bucket offsets - those will help us do the binary searches
  // We can load up to TOP_DOWN_EXPAND_DIMX of those bucket offsets - that way we prepare for the next MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD * blockDim.x edges
  //
  // Once we have those offsets, we may still need a few values from frontier_degrees_exclusive_sum to compute exact index k
  // To be able to do it, we will load the values that we need from frontier_degrees_exclusive_sum in shared memory
  // We know that it will fit because we never add node with degree == 0 in the frontier, so we have an upper bound on the number of value to load (see below)
  //
  // We will then look which vertices are not visited yet :
  // 1) if the unvisited vertex is isolated (=> degree == 0), we mark it as visited, update distances and predecessors, and move on
  // 2) if the unvisited vertex has degree > 0, we add it to the "frontier_candidates" queue
  //
  // We then treat the candidates queue using the threadIdx.x < ncandidates
  // If we are indeed the first thread to discover that vertex (result of atomicOr(visited))
  // We add it to the new frontier
  //

  template<typename IndexType>
  __global__ void topdown_expand_kernel(  const IndexType *row_ptr,
                            const IndexType *col_ind,
                            const IndexType *frontier,
                            const IndexType frontier_size,
                            const IndexType totaldegree,
                            const IndexType max_items_per_thread,
                            const IndexType lvl,
                            IndexType *new_frontier,
                            IndexType *new_frontier_cnt,
                            const IndexType *frontier_degrees_exclusive_sum,
                            const IndexType *frontier_degrees_exclusive_sum_buckets_offsets,
                            int *bmap,
                            IndexType *distances,
                            IndexType *predecessors,
                            const int *edge_mask,
                            const int *isolated_bmap,
                            bool directed) {
    //BlockScan
    typedef cub::BlockScan<IndexType, TOP_DOWN_EXPAND_DIMX> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;

    // We will do a scan to know where to write in frontier
    // This will contain the common offset of the block
    __shared__ IndexType frontier_common_block_offset;

    __shared__ IndexType shared_buckets_offsets[TOP_DOWN_EXPAND_DIMX - NBUCKETS_PER_BLOCK + 1];
    __shared__ IndexType shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX + 1];

    //
    // Frontier candidates local queue
    // We process TOP_DOWN_BATCH_SIZE vertices in parallel, so we need to be able to store everything
    // We also save the predecessors here, because we will not be able to retrieve it after
    //
    __shared__ IndexType shared_local_new_frontier_candidates[TOP_DOWN_BATCH_SIZE
        * TOP_DOWN_EXPAND_DIMX];
    __shared__ IndexType shared_local_new_frontier_predecessors[TOP_DOWN_BATCH_SIZE
        * TOP_DOWN_EXPAND_DIMX];
    __shared__ IndexType block_n_frontier_candidates;

    IndexType block_offset = (blockDim.x * blockIdx.x) * max_items_per_thread;
    IndexType n_items_per_thread_left = (totaldegree - block_offset + TOP_DOWN_EXPAND_DIMX - 1)
        / TOP_DOWN_EXPAND_DIMX;

    n_items_per_thread_left = min(max_items_per_thread, n_items_per_thread_left);

    for (;
        (n_items_per_thread_left > 0) && (block_offset < totaldegree);

        block_offset += MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD * blockDim.x,
            n_items_per_thread_left -= MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD) {

      // In this loop, we will process batch_set_size batches
      IndexType nitems_per_thread = min(  n_items_per_thread_left,
                              (IndexType) MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD);

      // Loading buckets offset (see compute_bucket_offsets_kernel)

      if (threadIdx.x < (nitems_per_thread * NBUCKETS_PER_BLOCK + 1))
        shared_buckets_offsets[threadIdx.x] =
            frontier_degrees_exclusive_sum_buckets_offsets[block_offset / TOP_DOWN_BUCKET_SIZE
                + threadIdx.x];

      // We will use shared_buckets_offsets
      __syncthreads();

      //
      // shared_buckets_offsets gives us a range of the possible indexes
      // for edge of linear_threadx, we are looking for the value k such as
      // k is the max value such as frontier_degrees_exclusive_sum[k] <= linear_threadx
      //
      // we have 0 <= k < frontier_size
      // but we also have :
      //
      // frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE]
      // <= k
      // <= frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE + 1]
      //
      // To find the exact value in that range, we need a few values from frontier_degrees_exclusive_sum (see below)
      // We will load them here
      // We will load as much as we can - if it doesn't fit we will make multiple iteration of the next loop
      // Because all vertices in frontier have degree > 0, we know it will fits if left + 1 = right (see below)

      //We're going to load values in frontier_degrees_exclusive_sum for batch [left; right[
      //If it doesn't fit, --right until it does, then loop
      //It is excepted to fit on the first try, that's why we start right = nitems_per_thread

      IndexType left = 0;
      IndexType right = nitems_per_thread;

      while (left < nitems_per_thread) {
        //
        // Values that are necessary to compute the local binary searches
        // We only need those with indexes between extremes indexes of buckets_offsets
        // We need the next val for the binary search, hence the +1
        //

        IndexType nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK]
            - shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;

        //If left = right + 1 we are sure to have nvalues_to_load < TOP_DOWN_EXPAND_DIMX+1
        while (nvalues_to_load > (TOP_DOWN_EXPAND_DIMX + 1)) {
          --right;

          nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK]
              - shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;
        }

        IndexType nitems_per_thread_for_this_load = right - left;

        IndexType frontier_degrees_exclusive_sum_block_offset = shared_buckets_offsets[left
            * NBUCKETS_PER_BLOCK];

        //TODO put again the nvalues_to_load == 1
        if (threadIdx.x < nvalues_to_load) {
          shared_frontier_degrees_exclusive_sum[threadIdx.x] =
              frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset
                  + threadIdx.x];
        }

        if (nvalues_to_load == (TOP_DOWN_EXPAND_DIMX + 1) && threadIdx.x == 0) {
          shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX] =
              frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset
                  + TOP_DOWN_EXPAND_DIMX];
        }

        //shared_frontier_degrees_exclusive_sum is in shared mem, we will use it, sync
        //TODO we don't use it if nvalues_to_load == 1
        __syncthreads();

        // Now we will process the edges
        // Here each thread will process nitems_per_thread_for_this_load
        for (IndexType item_index = 0;
            item_index < nitems_per_thread_for_this_load;
            item_index += TOP_DOWN_BATCH_SIZE) {

          // We process TOP_DOWN_BATCH_SIZE edge in parallel (instruction parallism)
          // Reduces latency

          IndexType current_max_edge_index = min(block_offset
                                        + (left
                                            + nitems_per_thread_for_this_load)
                                            * blockDim.x,
                                    totaldegree);

          //We will need vec_u (source of the edge) until the end if we need to save the predecessors
          //For others informations, we will reuse pointers on the go (nvcc does not color well the registers in that case)

          IndexType vec_u[TOP_DOWN_BATCH_SIZE];
          IndexType local_buf1[TOP_DOWN_BATCH_SIZE];
          IndexType local_buf2[TOP_DOWN_BATCH_SIZE];

          IndexType *vec_frontier_degrees_exclusive_sum_index = &local_buf2[0];

#pragma unroll
          for (IndexType iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {

            IndexType ibatch = left + item_index + iv;
            IndexType gid = block_offset + ibatch * blockDim.x + threadIdx.x;

            if (gid < current_max_edge_index) {
              IndexType start_off_idx = (ibatch * blockDim.x + threadIdx.x)
                  / TOP_DOWN_BUCKET_SIZE;
              IndexType bucket_start = shared_buckets_offsets[start_off_idx]
                  - frontier_degrees_exclusive_sum_block_offset;
              IndexType bucket_end = shared_buckets_offsets[start_off_idx + 1]
                  - frontier_degrees_exclusive_sum_block_offset;

              IndexType k = binsearch_maxle(shared_frontier_degrees_exclusive_sum,
                                  gid,
                                  bucket_start,
                                  bucket_end)
                  + frontier_degrees_exclusive_sum_block_offset;
              vec_u[iv] = frontier[k]; // origin of this edge
              vec_frontier_degrees_exclusive_sum_index[iv] =
                  frontier_degrees_exclusive_sum[k];
            } else {
              vec_u[iv] = -1;
              vec_frontier_degrees_exclusive_sum_index[iv] = -1;
            }

          }

          IndexType *vec_row_ptr_u = &local_buf1[0];
#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType u = vec_u[iv];
            //row_ptr for this vertex origin u
            vec_row_ptr_u[iv] = (u != -1)
                          ? row_ptr[u]
                            :
                            -1;
          }

          //We won't need row_ptr after that, reusing pointer
          IndexType *vec_dest_v = vec_row_ptr_u;

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType thread_item_index = left + item_index + iv;
            IndexType gid = block_offset + thread_item_index * blockDim.x + threadIdx.x;

            IndexType row_ptr_u = vec_row_ptr_u[iv];
            IndexType edge = row_ptr_u + gid - vec_frontier_degrees_exclusive_sum_index[iv];

            if (edge_mask && !edge_mask[edge])
              row_ptr_u = -1; //disabling edge

            //Destination of this edge
            vec_dest_v[iv] = (row_ptr_u != -1)
                        ? col_ind[edge]
                          :
                          -1;
          }

          //We don't need vec_frontier_degrees_exclusive_sum_index anymore
          IndexType *vec_v_visited_bmap = vec_frontier_degrees_exclusive_sum_index;
#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType v = vec_dest_v[iv];
            vec_v_visited_bmap[iv] = (v != -1)
                              ? bmap[v / INT_SIZE]
                                :
                                (~0); //will look visited
          }

          // From now on we will consider v as a frontier candidate
          // If for some reason vec_candidate[iv] should be put in the new_frontier
          // Then set vec_candidate[iv] = -1
          IndexType *vec_frontier_candidate = vec_dest_v;

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType v = vec_frontier_candidate[iv];
            int m = 1 << (v % INT_SIZE);

            int is_visited = vec_v_visited_bmap[iv] & m;

            if (is_visited)
              vec_frontier_candidate[iv] = -1;
          }

          if (directed) {
            //vec_v_visited_bmap is available

            IndexType *vec_is_isolated_bmap = vec_v_visited_bmap;

#pragma unroll
            for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
              IndexType v = vec_frontier_candidate[iv];
              vec_is_isolated_bmap[iv] = (v != -1)
                                ? isolated_bmap[v / INT_SIZE]
                                  :
                                  -1;
            }

#pragma unroll
            for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
              IndexType v = vec_frontier_candidate[iv];
              int m = 1 << (v % INT_SIZE);
              int is_isolated = vec_is_isolated_bmap[iv] & m;

              //If v is isolated, we will not add it to the frontier (it's not a frontier candidate)
              // 1st reason : it's useless
              // 2nd reason : it will make top down algo fail
              // we need each node in frontier to have a degree > 0
              // If it is isolated, we just need to mark it as visited, and save distance and predecessor here. Not need to check return value of atomicOr

              if (is_isolated && v != -1) {
                int m = 1 << (v % INT_SIZE);
                atomicOr(&bmap[v / INT_SIZE], m);
                if (distances)
                  distances[v] = lvl;

                if (predecessors)
                  predecessors[v] = vec_u[iv];

                //This is no longer a candidate, neutralize it
                vec_frontier_candidate[iv] = -1;
              }

            }
          }

          //Number of successor candidate hold by this thread
          IndexType thread_n_frontier_candidates = 0;

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            IndexType v = vec_frontier_candidate[iv];
            if (v != -1)
              ++thread_n_frontier_candidates;
          }

          // We need to have all nfrontier_candidates to be ready before doing the scan
          __syncthreads();

          // We will put the frontier candidates in a local queue
          // Computing offsets
          IndexType thread_frontier_candidate_offset = 0; //offset inside block
          BlockScan(scan_storage).ExclusiveSum(  thread_n_frontier_candidates,
                                    thread_frontier_candidate_offset);

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            //May have bank conflicts
            IndexType frontier_candidate = vec_frontier_candidate[iv];

            if (frontier_candidate != -1) {
              shared_local_new_frontier_candidates[thread_frontier_candidate_offset] =
                  frontier_candidate;
              shared_local_new_frontier_predecessors[thread_frontier_candidate_offset] =
                  vec_u[iv];
              ++thread_frontier_candidate_offset;
            }
          }

          if (threadIdx.x == (TOP_DOWN_EXPAND_DIMX - 1)) {
            //No need to add nsuccessor_candidate, even if its an
            //exclusive sum
            //We incremented the thread_frontier_candidate_offset
            block_n_frontier_candidates = thread_frontier_candidate_offset;
          }

          //broadcast block_n_frontier_candidates
          __syncthreads();

          IndexType naccepted_vertices = 0;
          //We won't need vec_frontier_candidate after that
          IndexType *vec_frontier_accepted_vertex = vec_frontier_candidate;

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            const int idx_shared = iv * blockDim.x + threadIdx.x;
            vec_frontier_accepted_vertex[iv] = -1;

            if (idx_shared < block_n_frontier_candidates) {
              IndexType v = shared_local_new_frontier_candidates[idx_shared]; //popping queue
              int m = 1 << (v % INT_SIZE);
              int q = atomicOr(&bmap[v / INT_SIZE], m); //atomicOr returns old

              if (!(m & q)) { //if this thread was the first to discover this node
                if (distances)
                  distances[v] = lvl;

                if (predecessors) {
                  IndexType pred = shared_local_new_frontier_predecessors[idx_shared];
                  predecessors[v] = pred;
                }

                vec_frontier_accepted_vertex[iv] = v;
                ++naccepted_vertices;
              }
            }

          }

          //We need naccepted_vertices to be ready
          __syncthreads();

          IndexType thread_new_frontier_offset;

          BlockScan(scan_storage).ExclusiveSum(naccepted_vertices, thread_new_frontier_offset);

          if (threadIdx.x == (TOP_DOWN_EXPAND_DIMX - 1)) {

            IndexType inclusive_sum = thread_new_frontier_offset + naccepted_vertices;
            //for this thread, thread_new_frontier_offset + has_successor (exclusive sum)
            if (inclusive_sum)
              frontier_common_block_offset = atomicAdd(new_frontier_cnt, inclusive_sum);
          }

          //Broadcasting frontier_common_block_offset
          __syncthreads();

#pragma unroll
          for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
            const int idx_shared = iv * blockDim.x + threadIdx.x;
            if (idx_shared < block_n_frontier_candidates) {

              IndexType new_frontier_vertex = vec_frontier_accepted_vertex[iv];

              if (new_frontier_vertex != -1) {
                IndexType off = frontier_common_block_offset + thread_new_frontier_offset++;
                //TODO Access is not good
                new_frontier[off] = new_frontier_vertex;
              }
            }
          }

        }

        //We need to keep shared_frontier_degrees_exclusive_sum coherent
        __syncthreads();

        //Preparing for next load
        left = right;
        right = nitems_per_thread;
      }

      //we need to keep shared_buckets_offsets coherent
      __syncthreads();
    }

  }

  template<typename IndexType>
  void frontier_expand(const IndexType *row_ptr,
                const IndexType *col_ind,
                const IndexType *frontier,
                const IndexType frontier_size,
                const IndexType totaldegree,
                const IndexType lvl,
                IndexType *new_frontier,
                IndexType *new_frontier_cnt,
                const IndexType *frontier_degrees_exclusive_sum,
                const IndexType *frontier_degrees_exclusive_sum_buckets_offsets,
                int *visited_bmap,
                IndexType *distances,
                IndexType *predecessors,
                const int *edge_mask,
                const int *isolated_bmap,
                bool directed,
                cudaStream_t m_stream,
                bool deterministic) {
    if (!totaldegree)
      return;

    dim3 block;
    block.x = TOP_DOWN_EXPAND_DIMX;

    IndexType max_items_per_thread = (totaldegree + MAXBLOCKS * block.x - 1)
        / (MAXBLOCKS * block.x);

    dim3 grid;
    grid.x = min(  (totaldegree + max_items_per_thread * block.x - 1)
                  / (max_items_per_thread * block.x),
              (IndexType) MAXBLOCKS);

    topdown_expand_kernel<<<grid, block, 0, m_stream>>>(  row_ptr,
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
                                        visited_bmap,
                                        distances,
                                        predecessors,
                                        edge_mask,
                                        isolated_bmap,
                                        directed);
    cudaCheckError()
    ;
  }

  template<typename IndexType>
  __global__ void flag_isolated_vertices_kernel(  IndexType n,
                                  int *isolated_bmap,
                                  const IndexType *row_ptr,
                                  IndexType *degrees,
                                  IndexType *nisolated) {
    typedef cub::BlockLoad<IndexType, FLAG_ISOLATED_VERTICES_DIMX,
        FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;
    typedef cub::BlockStore<IndexType, FLAG_ISOLATED_VERTICES_DIMX,
        FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStore;
    typedef cub::BlockReduce<IndexType, FLAG_ISOLATED_VERTICES_DIMX> BlockReduce;
    typedef cub::WarpReduce<int, FLAG_ISOLATED_VERTICES_THREADS_PER_INT> WarpReduce;

    __shared__ typename BlockLoad::TempStorage load_temp_storage;
    __shared__ typename BlockStore::TempStorage store_temp_storage;
    __shared__ typename BlockReduce::TempStorage block_reduce_temp_storage;

    __shared__ typename WarpReduce::TempStorage warp_reduce_temp_storage[FLAG_ISOLATED_VERTICES_DIMX
        / FLAG_ISOLATED_VERTICES_THREADS_PER_INT];

    __shared__ IndexType row_ptr_tail[FLAG_ISOLATED_VERTICES_DIMX];

    for (IndexType block_off = FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD
        * (blockDim.x * blockIdx.x);
        block_off < n;
        block_off += FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD * (blockDim.x * gridDim.x)) {

      IndexType thread_off = block_off
          + FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD * threadIdx.x;
      IndexType last_node_thread = thread_off + FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1;

      IndexType thread_row_ptr[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD];
      IndexType block_valid_items = n - block_off + 1; //+1, we need row_ptr[last_node+1]

      BlockLoad(load_temp_storage).Load(  row_ptr + block_off,
                              thread_row_ptr,
                              block_valid_items,
                              -1);

      //To compute 4 degrees, we need 5 values of row_ptr
      //Saving the "5th" value in shared memory for previous thread to use
      if (threadIdx.x > 0) {
        row_ptr_tail[threadIdx.x - 1] = thread_row_ptr[0];
      }

      //If this is the last thread, it needs to load its row ptr tail value
      if (threadIdx.x == (FLAG_ISOLATED_VERTICES_DIMX - 1) && last_node_thread < n) {
        row_ptr_tail[threadIdx.x] = row_ptr[last_node_thread + 1];

      }
      __syncthreads(); // we may reuse temp_storage

      int local_isolated_bmap = 0;

      IndexType imax = (n - thread_off);

      IndexType local_degree[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD];

#pragma unroll
      for (int i = 0; i < (FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1); ++i) {
        IndexType degree = local_degree[i] = thread_row_ptr[i + 1] - thread_row_ptr[i];

        if (i < imax)
          local_isolated_bmap |= ((degree == 0) << i);
      }

      if (last_node_thread < n) {
        IndexType degree = local_degree[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1] =
            row_ptr_tail[threadIdx.x]
                - thread_row_ptr[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1];

        local_isolated_bmap |= ((degree == 0)
            << (FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1));

      }

      local_isolated_bmap <<= (thread_off % INT_SIZE);

      IndexType local_nisolated = __popc(local_isolated_bmap);

      //We need local_nisolated and local_isolated_bmap to be ready for next steps
      __syncthreads();

      IndexType total_nisolated = BlockReduce(block_reduce_temp_storage).Sum(local_nisolated);

      if (threadIdx.x == 0 && total_nisolated) {
        atomicAdd(nisolated, total_nisolated);
      }

      int logicalwarpid = threadIdx.x / FLAG_ISOLATED_VERTICES_THREADS_PER_INT;

      //Building int for bmap
      int int_aggregate_isolated_bmap =
          WarpReduce(warp_reduce_temp_storage[logicalwarpid]).Reduce(  local_isolated_bmap,
                                                  BitwiseOr());

      int is_head_of_visited_int =
          ((threadIdx.x % (FLAG_ISOLATED_VERTICES_THREADS_PER_INT)) == 0);
      if (is_head_of_visited_int) {
        isolated_bmap[thread_off / INT_SIZE] = int_aggregate_isolated_bmap;
      }

      BlockStore(store_temp_storage).Store(degrees + block_off, local_degree, block_valid_items);
    }
  }

  template<typename IndexType>
  void flag_isolated_vertices(  IndexType n,
                      int *isolated_bmap,
                      const IndexType *row_ptr,
                      IndexType *degrees,
                      IndexType *nisolated,
                      cudaStream_t m_stream) {
    dim3 grid, block;
    block.x = FLAG_ISOLATED_VERTICES_DIMX;

    grid.x = min(  (IndexType) MAXBLOCKS,
              (n / FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD + 1 + block.x - 1) / block.x);

    flag_isolated_vertices_kernel<<<grid, block, 0, m_stream>>>(n,
                                            isolated_bmap,
                                            row_ptr,
                                            degrees,
                                            nisolated);
    cudaCheckError()
    ;
  }

  //
  //
  //
  // Some utils functions
  //
  //

  //Creates CUB data for graph size n
  template<typename IndexType>
  void cub_exclusive_sum_alloc(IndexType n, void*& d_temp_storage, size_t &temp_storage_bytes) {
    // Determine temporary device storage requirements for exclusive prefix scan
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    IndexType *d_in = NULL, *d_out = NULL;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    // Allocate temporary storage for exclusive prefix scan
    cudaStream_t stream{nullptr};
    RMM_ALLOC(&d_temp_storage, temp_storage_bytes, stream);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
  }

  template<typename IndexType>
  __global__ void fill_kernel(IndexType *vec, IndexType n, IndexType val) {
    for (IndexType u = blockDim.x * blockIdx.x + threadIdx.x;
        u < n;
        u += gridDim.x * blockDim.x)
      vec[u] = val;

  }

  template<typename IndexType>
  void fill(IndexType *vec, IndexType n, IndexType val, cudaStream_t m_stream) {
    dim3 grid, block;
    block.x = 256;
    grid.x = min((n + block.x - 1) / block.x, (IndexType) MAXBLOCKS);
    fill_kernel<<<grid, block, 0, m_stream>>>(vec, n, val);
    cudaCheckError()
    ;
  }

  template<typename IndexType>
  __global__ void set_frontier_degree_kernel(  IndexType *frontier_degree,
                                IndexType *frontier,
                                const IndexType *degree,
                                IndexType n) {
    for (IndexType idx = blockDim.x * blockIdx.x + threadIdx.x;
        idx < n;
        idx += gridDim.x * blockDim.x) {
      IndexType u = frontier[idx];
      frontier_degree[idx] = degree[u];
    }
  }

  template<typename IndexType>
  void set_frontier_degree(  IndexType *frontier_degree,
                    IndexType *frontier,
                    const IndexType *degree,
                    IndexType n,
                    cudaStream_t m_stream) {
    dim3 grid, block;
    block.x = 256;
    grid.x = min((n + block.x - 1) / block.x, (IndexType) MAXBLOCKS);
    set_frontier_degree_kernel<<<grid, block, 0, m_stream>>>(frontier_degree,
                                          frontier,
                                          degree,
                                          n);
    cudaCheckError()
    ;
  }

  template<typename IndexType>
  void exclusive_sum(  void *d_temp_storage,
                size_t temp_storage_bytes,
                IndexType *d_in,
                IndexType *d_out,
                IndexType num_items,
                cudaStream_t m_stream) {
    if (num_items <= 1)
      return; //DeviceScan fails if n==1
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                        temp_storage_bytes,
                        d_in,
                        d_out,
                        num_items,
                        m_stream);
  }

  template<typename T>
  __global__ void fill_vec_kernel(T *vec, T n, T val) {
    for (T idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < n;
        idx += blockDim.x * gridDim.x)
      vec[idx] = val;
  }

  template<typename T>
  void fill_vec(T *vec, T n, T val, cudaStream_t stream) {
    dim3 grid, block;
    block.x = 256;
    grid.x = (n + block.x - 1) / block.x;

    fill_vec_kernel<<<grid, block, 0, stream>>>(vec, n, val);
    cudaCheckError()
    ;
  }
}
//
