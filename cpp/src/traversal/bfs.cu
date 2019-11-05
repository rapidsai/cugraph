/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <cugraph.h>
#include <algorithm>
#include <iomanip>
#include "bfs.cuh"
#include <limits>
#include "rmm_utils.h"

#include "utilities/graph_utils.cuh"
#include "traversal_common.cuh"
#include "bfs_kernels.cuh"

namespace cugraph { 
namespace detail {
  enum BFS_ALGO_STATE {
    TOPDOWN, BOTTOMUP
  };

  template<typename IndexType>
  void Bfs<IndexType>::setup() {

    // Determinism flag, false by default
    deterministic = false;
    //Working data
    //Each vertex can be in the frontier at most once
    ALLOC_TRY(&frontier, n * sizeof(IndexType), nullptr);

    //We will update frontier during the execution
    //We need the orig to reset frontier, or ALLOC_FREE_TRY
    original_frontier = frontier;

    //size of bitmaps for vertices
    vertices_bmap_size = (n / (8 * sizeof(int)) + 1);
    //ith bit of visited_bmap is set <=> ith vertex is visited
    ALLOC_TRY(&visited_bmap, sizeof(int) * vertices_bmap_size, nullptr);

    //ith bit of isolated_bmap is set <=> degree of ith vertex = 0
    ALLOC_TRY(&isolated_bmap, sizeof(int) * vertices_bmap_size, nullptr);

    //vertices_degree[i] = degree of vertex i
    ALLOC_TRY(&vertex_degree, sizeof(IndexType) * n, nullptr);

    //Cub working data
    traversal::cub_exclusive_sum_alloc(n + 1, d_cub_exclusive_sum_storage, cub_exclusive_sum_storage_bytes);

    //We will need (n+1) ints buffer for two differents things (bottom up or top down) - sharing it since those uses are mutually exclusive
    ALLOC_TRY(&buffer_np1_1, (n + 1) * sizeof(IndexType), nullptr);
    ALLOC_TRY(&buffer_np1_2, (n + 1) * sizeof(IndexType), nullptr);

    //Using buffers : top down

    //frontier_vertex_degree[i] is the degree of vertex frontier[i]
    frontier_vertex_degree = buffer_np1_1;
    //exclusive sum of frontier_vertex_degree
    exclusive_sum_frontier_vertex_degree = buffer_np1_2;

    //Using buffers : bottom up
    //contains list of unvisited vertices
    unvisited_queue = buffer_np1_1;
    //size of the "last" unvisited queue : size_last_unvisited_queue
    //refers to the size of unvisited_queue
    //which may not be up to date (the queue may contains vertices that are now visited)

    //We may leave vertices unvisited after bottom up main kernels - storing them here
    left_unvisited_queue = buffer_np1_2;

    //We use buckets of edges (32 edges per bucket for now, see exact macro in bfs_kernels). frontier_vertex_degree_buckets_offsets[i] is the index k such as frontier[k] is the source of the first edge of the bucket
    //See top down kernels for more details
    ALLOC_TRY(&exclusive_sum_frontier_vertex_buckets_offsets,
              ((nnz / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2) * sizeof(IndexType), nullptr);

    //Init device-side counters
    //Those counters must be/can be reset at each bfs iteration
    //Keeping them adjacent in memory allow use call only one cudaMemset - launch latency is the current bottleneck
    ALLOC_TRY(&d_counters_pad, 4 * sizeof(IndexType), nullptr);

    d_new_frontier_cnt = &d_counters_pad[0];
    d_mu = &d_counters_pad[1];
    d_unvisited_cnt = &d_counters_pad[2];
    d_left_unvisited_cnt = &d_counters_pad[3];

    //Lets use this int* for the next 3 lines
    //Its dereferenced value is not initialized - so we dont care about what we put in it
    IndexType * d_nisolated = d_new_frontier_cnt;
    cudaMemsetAsync(d_nisolated, 0, sizeof(IndexType), stream);

    //Computing isolated_bmap
    //Only dependent on graph - not source vertex - done once
    traversal::flag_isolated_vertices(n, isolated_bmap, row_offsets, vertex_degree, d_nisolated, stream);
    cudaMemcpyAsync(&nisolated, d_nisolated, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);

    //We need nisolated to be ready to use
    cudaStreamSynchronize(stream);
  }

  template<typename IndexType>
  void Bfs<IndexType>::configure(IndexType *_distances,
                                 IndexType *_predecessors,
                                 int *_edge_mask)
  {
    distances = _distances;
    predecessors = _predecessors;
    edge_mask = _edge_mask;

    useEdgeMask = (edge_mask != NULL);
    computeDistances = (distances != NULL);
    computePredecessors = (predecessors != NULL);

    //We need distances to use bottom up
    if (directed && !computeDistances)
      ALLOC_TRY(&distances, n * sizeof(IndexType), nullptr);
  }

  template<typename IndexType>
  void Bfs<IndexType>::traverse(IndexType source_vertex) {

    //Init visited_bmap
    //If the graph is undirected, we not that
    //we will never discover isolated vertices (in degree = out degree = 0)
    //we avoid a lot of work by flagging them now
    //in g500 graphs they represent ~25% of total vertices
    //more than that for wiki and twitter graphs

    if (directed) {
      cudaMemsetAsync(visited_bmap, 0, vertices_bmap_size * sizeof(int), stream);
    }
    else {
      cudaMemcpyAsync(visited_bmap,
                      isolated_bmap,
                      vertices_bmap_size * sizeof(int),
                      cudaMemcpyDeviceToDevice,
                      stream);
    }

    //If needed, setting all vertices as undiscovered (inf distance)
    //We dont use computeDistances here
    //if the graph is undirected, we may need distances even if
    //computeDistances is false
    if (distances)
      traversal::fill_vec(distances, n, traversal::vec_t<IndexType>::max, stream);

    //If needed, setting all predecessors to non-existent (-1)
    if (computePredecessors) {
      cudaMemsetAsync(predecessors, -1, n * sizeof(IndexType), stream);
    }

    //
    //Initial frontier
    //

    frontier = original_frontier;

    if (distances) {
      cudaMemsetAsync(&distances[source_vertex], 0, sizeof(IndexType), stream);
    }

    //Setting source_vertex as visited
    //There may be bit already set on that bmap (isolated vertices) - if the graph is undirected
    int current_visited_bmap_source_vert = 0;

    if (!directed) {
      cudaMemcpyAsync(&current_visited_bmap_source_vert,
                      &visited_bmap[source_vertex / INT_SIZE],
                      sizeof(int),
                      cudaMemcpyDeviceToHost);
      //We need current_visited_bmap_source_vert
      cudaStreamSynchronize(stream);
    }

    int m = (1 << (source_vertex % INT_SIZE));

    //In that case, source is isolated, done now
    if (!directed && (m & current_visited_bmap_source_vert)) {
      //Init distances and predecessors are done, (cf Streamsync in previous if)
      return;
    }

    m |= current_visited_bmap_source_vert;

    cudaMemcpyAsync(&visited_bmap[source_vertex / INT_SIZE],
                    &m,
                    sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream);

    //Adding source_vertex to init frontier
    cudaMemcpyAsync(&frontier[0],
                    &source_vertex,
                    sizeof(IndexType),
                    cudaMemcpyHostToDevice,
                    stream);

    //mf : edges in frontier
    //nf : vertices in frontier
    //mu : edges undiscovered
    //nu : nodes undiscovered
    //lvl : current frontier's depth
    IndexType mf, nf, mu, nu;
    bool growing;
    IndexType lvl = 1;

    //Frontier has one vertex
    nf = 1;

    //all edges are undiscovered (by def isolated vertices have 0 edges)
    mu = nnz;

    //all non isolated vertices are undiscovered (excepted source vertex, which is in frontier)
    //That number is wrong if source_vertex is also isolated - but it's not important
    nu = n - nisolated - nf;

    //Last frontier was 0, now it is 1
    growing = true;

    IndexType size_last_left_unvisited_queue = n; //we just need value > 0
    IndexType size_last_unvisited_queue = 0; //queue empty

    //Typical pre-top down workflow. set_frontier_degree + exclusive-scan
    traversal::set_frontier_degree(frontier_vertex_degree, frontier, vertex_degree, nf, stream);
    traversal::exclusive_sum(d_cub_exclusive_sum_storage,
                  cub_exclusive_sum_storage_bytes,
                  frontier_vertex_degree,
                  exclusive_sum_frontier_vertex_degree,
                  nf + 1,
                  stream);

    cudaMemcpyAsync(&mf,
                    &exclusive_sum_frontier_vertex_degree[nf],
                    sizeof(IndexType),
                    cudaMemcpyDeviceToHost,
                    stream);

    //We need mf
    cudaStreamSynchronize(stream);

    //At first we know we have to use top down
    BFS_ALGO_STATE algo_state = TOPDOWN;

    //useDistances : we check if a vertex is a parent using distances in bottom up - distances become working data
    //undirected g : need parents to be in children's neighbors
    bool can_use_bottom_up = !directed && distances;

    while (nf > 0) {
      //Each vertices can appear only once in the frontierer array - we know it will fit
      new_frontier = frontier + nf;
      IndexType old_nf = nf;
      resetDevicePointers();

      if (can_use_bottom_up) {
        //Choosing algo
        //Finite machine described in http://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf

        switch (algo_state) {
          case TOPDOWN:
            if (mf > mu / alpha)
              algo_state = BOTTOMUP;
            break;
          case BOTTOMUP:
            if (!growing && nf < n / beta) {

              //We need to prepare the switch back to top down
              //We couldnt keep track of mu during bottom up - because we dont know what mf is. Computing mu here
	      bfs_kernels::count_unvisited_edges(unvisited_queue,
                                    size_last_unvisited_queue,
                                    visited_bmap,
                                    vertex_degree,
                                    d_mu,
                                    stream);

              //Typical pre-top down workflow. set_frontier_degree + exclusive-scan
              traversal::set_frontier_degree(frontier_vertex_degree,
                                  frontier,
                                  vertex_degree,
                                  nf,
                                  stream);
              traversal::exclusive_sum(d_cub_exclusive_sum_storage,
                            cub_exclusive_sum_storage_bytes,
                            frontier_vertex_degree,
                            exclusive_sum_frontier_vertex_degree,
                            nf + 1,
                            stream);

              cudaMemcpyAsync(&mf,
                              &exclusive_sum_frontier_vertex_degree[nf],
                              sizeof(IndexType),
                              cudaMemcpyDeviceToHost,
                              stream);

              cudaMemcpyAsync(&mu, d_mu, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);

              //We will need mf and mu
              cudaStreamSynchronize(stream);
              algo_state = TOPDOWN;
            }
            break;
        }
      }

      //Executing algo

      switch (algo_state) {
        case TOPDOWN:
          traversal::compute_bucket_offsets(exclusive_sum_frontier_vertex_degree,
                                 exclusive_sum_frontier_vertex_buckets_offsets,
                                 nf,
                                 mf,
                                 stream);
	  bfs_kernels::frontier_expand(row_offsets,
                          col_indices,
                          frontier,
                          nf,
                          mf,
                          lvl,
                          new_frontier,
                          d_new_frontier_cnt,
                          exclusive_sum_frontier_vertex_degree,
                          exclusive_sum_frontier_vertex_buckets_offsets,
                          visited_bmap,
                          distances,
                          predecessors,
                          edge_mask,
                          isolated_bmap,
                          directed,
                          stream,
                          deterministic);

          mu -= mf;

          cudaMemcpyAsync(&nf,
                          d_new_frontier_cnt,
                          sizeof(IndexType),
                          cudaMemcpyDeviceToHost,
                          stream);
          CUDA_CHECK_LAST();

          //We need nf
          cudaStreamSynchronize(stream);

          if (nf) {
            //Typical pre-top down workflow. set_frontier_degree + exclusive-scan
            traversal::set_frontier_degree(frontier_vertex_degree,
                                new_frontier,
                                vertex_degree,
                                nf,
                                stream);
            traversal::exclusive_sum(d_cub_exclusive_sum_storage,
                           cub_exclusive_sum_storage_bytes,
                           frontier_vertex_degree,
                           exclusive_sum_frontier_vertex_degree,
                           nf + 1,
                           stream);
            cudaMemcpyAsync(&mf,
                            &exclusive_sum_frontier_vertex_degree[nf],
                            sizeof(IndexType),
                            cudaMemcpyDeviceToHost,
                            stream);

            //We need mf
            cudaStreamSynchronize(stream);
          }
          break;

        case BOTTOMUP:
	  bfs_kernels::fill_unvisited_queue(visited_bmap,
                               vertices_bmap_size,
                               n,
                               unvisited_queue,
                               d_unvisited_cnt,
                               stream,
                               deterministic);

          size_last_unvisited_queue = nu;

	  bfs_kernels::bottom_up_main(unvisited_queue,
                         size_last_unvisited_queue,
                         left_unvisited_queue,
                         d_left_unvisited_cnt,
                         visited_bmap,
                         row_offsets,
                         col_indices,
                         lvl,
                         new_frontier,
                         d_new_frontier_cnt,
                         distances,
                         predecessors,
                         edge_mask,
                         stream,
                         deterministic);

          //The number of vertices left unvisited decreases
          //If it wasnt necessary last time, it wont be this time
          if (size_last_left_unvisited_queue) {
            cudaMemcpyAsync(&size_last_left_unvisited_queue,
                            d_left_unvisited_cnt,
                            sizeof(IndexType),
                            cudaMemcpyDeviceToHost,
                            stream);
            CUDA_CHECK_LAST()
            //We need last_left_unvisited_size
            cudaStreamSynchronize(stream);
	    bfs_kernels::bottom_up_large(left_unvisited_queue,
                            size_last_left_unvisited_queue,
                            visited_bmap,
                            row_offsets,
                            col_indices,
                            lvl,
                            new_frontier,
                            d_new_frontier_cnt,
                            distances,
                            predecessors,
                            edge_mask,
                            stream,
                            deterministic);
          }
          cudaMemcpyAsync(&nf,
                          d_new_frontier_cnt,
                          sizeof(IndexType),
                          cudaMemcpyDeviceToHost,
                          stream);
          CUDA_CHECK_LAST()

          //We will need nf
          cudaStreamSynchronize(stream);
          break;
      }

      //Updating undiscovered edges count
      nu -= nf;

      //Using new frontier
      frontier = new_frontier;
      growing = (nf > old_nf);

      ++lvl;
    }
  }

  template<typename IndexType>
  void Bfs<IndexType>::resetDevicePointers() {
    cudaMemsetAsync(d_counters_pad, 0, 4 * sizeof(IndexType), stream);
  }

  template<typename IndexType>
  void Bfs<IndexType>::clean() {
    //the vectors have a destructor that takes care of cleaning
    ALLOC_FREE_TRY(original_frontier, nullptr);
    ALLOC_FREE_TRY(visited_bmap, nullptr);
    ALLOC_FREE_TRY(isolated_bmap, nullptr);
    ALLOC_FREE_TRY(vertex_degree, nullptr);
    ALLOC_FREE_TRY(d_cub_exclusive_sum_storage, nullptr);
    ALLOC_FREE_TRY(buffer_np1_1, nullptr);
    ALLOC_FREE_TRY(buffer_np1_2, nullptr);
    ALLOC_FREE_TRY(exclusive_sum_frontier_vertex_buckets_offsets, nullptr);
    ALLOC_FREE_TRY(d_counters_pad, nullptr);

    //In that case, distances is a working data
    if (directed && !computeDistances)
      ALLOC_FREE_TRY(distances, nullptr);
  }

  template class Bfs<int> ;
} //namespace 
void bfs(Graph *graph, gdf_column *distances, gdf_column *predecessors, int start_vertex, bool directed) {

  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList->offsets->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(graph->adjList->indices->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(distances->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(predecessors->dtype == GDF_INT32, "Unsupported data type");


  int n = graph->adjList->offsets->size - 1;
  int e = graph->adjList->indices->size;
  int* offsets_ptr = (int*)graph->adjList->offsets->data;
  int* indices_ptr = (int*)graph->adjList->indices->data;
  int* distances_ptr = (int*)distances->data;
  int* predecessors_ptr = (int*)predecessors->data;
  int alpha = 15;
  int beta = 18;

  cugraph::detail::Bfs<int> bfs(n, e, offsets_ptr, indices_ptr, directed, alpha, beta);
  bfs.configure(distances_ptr, predecessors_ptr, nullptr);
  bfs.traverse(start_vertex);
}

} //namespace 