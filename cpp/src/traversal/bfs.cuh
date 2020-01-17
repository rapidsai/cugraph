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

#pragma once

#include <climits> 

#define TRAVERSAL_DEFAULT_ALPHA 15

#define TRAVERSAL_DEFAULT_BETA 18

namespace cugraph { 
namespace detail {
	template<typename IndexType>
	class Bfs {
	private:
		IndexType n, nnz;
		IndexType* row_offsets;
		IndexType* col_indices;

		bool directed;
		bool deterministic;

		// edgemask, distances, predecessors are set/read by users - using Vectors
		bool useEdgeMask;
		bool computeDistances;
		bool computePredecessors;
		IndexType *distances;
		IndexType *predecessors;
		int *edge_mask;

		//Working data
		//For complete description of each, go to bfs.cu
		IndexType nisolated;
		IndexType *frontier, *new_frontier;
		IndexType * original_frontier;
		IndexType vertices_bmap_size;
		int *visited_bmap, *isolated_bmap;
		IndexType *vertex_degree;
		IndexType *buffer_np1_1, *buffer_np1_2;
		IndexType *frontier_vertex_degree;
		IndexType *exclusive_sum_frontier_vertex_degree;
		IndexType *unvisited_queue;
		IndexType *left_unvisited_queue;
		IndexType *exclusive_sum_frontier_vertex_buckets_offsets;
		IndexType *d_counters_pad;
		IndexType *d_new_frontier_cnt;
		IndexType *d_mu;
		IndexType *d_unvisited_cnt;
		IndexType *d_left_unvisited_cnt;
		void *d_cub_exclusive_sum_storage;
		size_t cub_exclusive_sum_storage_bytes;

		//Parameters for direction optimizing
		IndexType alpha, beta;
		cudaStream_t stream;

		//resets pointers defined by d_counters_pad (see implem)
		void resetDevicePointers();
		void setup();
		void clean();

	public:
		virtual ~Bfs(void) {
			clean();
		}

		Bfs(	IndexType _n,
				IndexType _nnz,
				IndexType *_row_offsets,
				IndexType *_col_indices,
				bool _directed,
				IndexType _alpha,
				IndexType _beta,
				cudaStream_t _stream = 0) :
						n(_n),
						nnz(_nnz),
						row_offsets(_row_offsets),
						col_indices(_col_indices),
						directed(_directed),
						alpha(_alpha),
						beta(_beta),
						stream(_stream) {
			setup();
		}

		void configure(IndexType *distances, IndexType *predecessors, int *edge_mask);

		void traverse(IndexType source_vertex);
	};
} } //namespace

