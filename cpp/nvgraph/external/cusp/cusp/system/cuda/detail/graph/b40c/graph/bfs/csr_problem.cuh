/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * GPU CSR storage management structure for BFS problem data
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/cuda_properties.cuh"
#include "../../util/memset_kernel.cuh"
#include "../../util/cta_work_progress.cuh"
#include "../../util/error_utils.cuh"
#include "../../util/multiple_buffering.cuh"

#include "problem_type.cuh"

#include <vector>

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {


/**
 * Enumeration of global frontier queue configurations
 */
enum FrontierType {
	VERTEX_FRONTIERS,		// O(n) ping-pong global vertex frontiers
	EDGE_FRONTIERS,			// O(m) ping-pong global edge frontiers
	MIXED_FRONTIERS,		// O(n) global vertex frontier, O(m) global edge frontier
	MULTI_GPU_FRONTIERS,	// O(MULTI_GPU_VERTEX_FRONTIER_SCALE * n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier

	MULTI_GPU_VERTEX_FRONTIER_SCALE = 2,
};



/**
 * CSR storage management structure for BFS problems.  
 */
template <
	typename _VertexId,
	typename _SizeT,
	bool MARK_PREDECESSORS>		// Whether to mark predecessors (vs. mark distance from source)
struct CsrProblem
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef ProblemType<
		_VertexId,				// VertexId
		_SizeT,					// SizeT
		unsigned char,			// VisitedMask
		unsigned char, 			// ValidFlag
		MARK_PREDECESSORS>		// MARK_PREDECESSORS
			ProblemType;

	typedef typename ProblemType::VertexId 			VertexId;
	typedef typename ProblemType::SizeT				SizeT;
	typedef typename ProblemType::VisitedMask 		VisitedMask;
	typedef typename ProblemType::ValidFlag 		ValidFlag;


	//---------------------------------------------------------------------
	// Helper structures
	//---------------------------------------------------------------------

	/**
	 * Graph slice per GPU
	 */
	struct GraphSlice
	{
		// GPU index
		int 			gpu;

		// Standard CSR device storage arrays
		VertexId 		*d_column_indices;
		SizeT 			*d_row_offsets;
		VertexId 		*d_labels;				// Can be used for source distance or predecessor pointer

		// Best-effort mask for keeping track of which vertices we've seen so far
		VisitedMask 	*d_visited_mask;

		// Frontier queues.  Keys track work, values optionally track predecessors.  Only
		// multi-gpu uses triple buffers (single-GPU only uses ping-pong buffers).
		util::TripleBuffer<VertexId, VertexId> 		frontier_queues;
		SizeT 										frontier_elements[3];
		SizeT 										predecessor_elements[3];

		// Flags for filtering duplicates from the edge-frontier queue when partitioning during multi-GPU BFS.
		ValidFlag 		*d_filter_mask;

		// Number of nodes and edges in slice
		VertexId		nodes;
		SizeT			edges;

		// CUDA stream to use for processing this slice
		cudaStream_t 	stream;

		/**
		 * Constructor
		 */
		GraphSlice(int gpu, cudaStream_t stream) :
			gpu(gpu),
			d_column_indices(NULL),
			d_row_offsets(NULL),
			d_labels(NULL),
			d_visited_mask(NULL),
			d_filter_mask(NULL),
			nodes(0),
			edges(0),
			stream(stream)
		{
			// Initialize triple-buffer frontier queue lengths
			for (int i = 0; i < 3; i++) {
				frontier_elements[i] = 0;
				predecessor_elements[i] = 0;
			}
		}

		/**
		 * Destructor
		 */
		virtual ~GraphSlice()
		{
			// Set device
			util::B40CPerror<0>(cudaSetDevice(gpu), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

			// Free pointers
			if (d_column_indices) 				util::B40CPerror<0>(cudaFree(d_column_indices), "GpuSlice cudaFree d_column_indices failed", __FILE__, __LINE__);
			if (d_row_offsets) 					util::B40CPerror<0>(cudaFree(d_row_offsets), "GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
			if (d_labels) 						util::B40CPerror<0>(cudaFree(d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
			if (d_visited_mask) 				util::B40CPerror<0>(cudaFree(d_visited_mask), "GpuSlice cudaFree d_visited_mask failed", __FILE__, __LINE__);
			if (d_filter_mask) 						util::B40CPerror<0>(cudaFree(d_filter_mask), "GpuSlice cudaFree d_filter_mask failed", __FILE__, __LINE__);
			for (int i = 0; i < 3; i++) {
				if (frontier_queues.d_keys[i]) 		util::B40CPerror<0>(cudaFree(frontier_queues.d_keys[i]), "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__);
				if (frontier_queues.d_values[i]) 	util::B40CPerror<0>(cudaFree(frontier_queues.d_values[i]), "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__);
			}

			// Destroy stream
			if (stream) {
				util::B40CPerror<0>(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
			}
		}
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Number of GPUS to be sliced over
	int							num_gpus;

	// Size of the graph
	SizeT 						nodes;
	SizeT						edges;

	// Set of graph slices (one for each GPU)
	std::vector<GraphSlice*> 	graph_slices;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	CsrProblem() :
		num_gpus(0),
		nodes(0),
		edges(0)
	{}


	/**
	 * Destructor
	 */
	virtual ~CsrProblem()
	{
		// Cleanup graph slices on the heap
		for (typename std::vector<GraphSlice*>::iterator itr = graph_slices.begin();
			itr != graph_slices.end();
			itr++)
		{
			if (*itr) delete (*itr);
		}
	}


	/**
	 * Returns index of the gpu that owns the neighbor list of
	 * the specified vertex
	 */
	template <typename VertexId>
	int GpuIndex(VertexId vertex)
	{
		if (graph_slices.size() == 1) {

			// Special case for only one GPU, which may be set as with
			// an ordinal other than 0.
			return graph_slices[0]->gpu;

		} else {

			return vertex % num_gpus;
		}
	}


	/**
	 * Returns the row within a gpu's GraphSlice row_offsets vector
	 * for the specified vertex
	 */
	template <typename VertexId>
	VertexId GraphSliceRow(VertexId vertex)
	{
		return vertex / num_gpus;
	}


	/**
	 * Extract into a single host vector the BFS results disseminated across
	 * all GPUs
	 */
	cudaError_t ExtractResults(VertexId *h_label)
	{
		cudaError_t retval = cudaSuccess;

		do {
			if (graph_slices.size() == 1) {

				// Set device
				if (util::B40CPerror<0>(cudaSetDevice(graph_slices[0]->gpu),
					"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;;

				// Special case for only one GPU, which may be set as with
				// an ordinal other than 0.
				if (retval = util::B40CPerror<0>(cudaMemcpy(
						h_label,
						graph_slices[0]->d_labels,
						sizeof(VertexId) * graph_slices[0]->nodes,
						cudaMemcpyDeviceToHost),
					"CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

			} else {

				VertexId **gpu_labels = new VertexId*[num_gpus];

				// Copy out
				for (int gpu = 0; gpu < num_gpus; gpu++) {

					// Set device
					if (util::B40CPerror<0>(cudaSetDevice(graph_slices[gpu]->gpu),
						"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;;

					// Allocate and copy out
					gpu_labels[gpu] = new VertexId[graph_slices[gpu]->nodes];

					if (retval = util::B40CPerror<0>(cudaMemcpy(
							gpu_labels[gpu],
							graph_slices[gpu]->d_labels,
							sizeof(VertexId) * graph_slices[gpu]->nodes,
							cudaMemcpyDeviceToHost),
						"CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
				}
				if (retval) break;

				// Combine
				for (VertexId node = 0; node < nodes; node++) {
					int gpu = GpuIndex(node);
					VertexId slice_row = GraphSliceRow(node);
					h_label[node] = gpu_labels[gpu][slice_row];

					switch (h_label[node]) {
					case -1:
					case -2:
						break;
					default:
						h_label[node] &= ProblemType::VERTEX_ID_MASK;
					};
				}

				// Clean up
				for (int gpu = 0; gpu < num_gpus; gpu++) {
					if (gpu_labels[gpu]) delete gpu_labels[gpu];
				}
				delete gpu_labels;
			}
		} while(0);

		return retval;
	}


	/**
	 * Initialize from host CSR problem
	 */
	cudaError_t FromHostProblem(
		bool		stream_from_host,			// Only meaningful for single-GPU BFS
		SizeT 		nodes,
		SizeT 		edges,
		VertexId 	*h_column_indices,
		SizeT 		*h_row_offsets,
		int 		num_gpus)
	{
		cudaError_t retval 				= cudaSuccess;
		this->nodes						= nodes;
		this->edges 					= edges;
		this->num_gpus 					= num_gpus;

		do {
			if (num_gpus <= 1) {

				// Create a single GPU slice for the currently-set gpu
				int gpu;
				if (retval = util::B40CPerror<0>(cudaGetDevice(&gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
				graph_slices.push_back(new GraphSlice(gpu, 0));
				graph_slices[0]->nodes = nodes;
				graph_slices[0]->edges = edges;

				if (stream_from_host) {

					// Map the pinned graph pointers into device pointers
					if (retval = util::B40CPerror<0>(cudaHostGetDevicePointer(
							(void **)&graph_slices[0]->d_column_indices,
							(void *) h_column_indices, 0),
						"CsrProblem cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror<0>(cudaHostGetDevicePointer(
							(void **)&graph_slices[0]->d_row_offsets,
							(void *) h_row_offsets, 0),
						"CsrProblem cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;

				} else {

					// Allocate and initialize d_column_indices

					printf("GPU %d column_indices: %lld elements (%lld bytes)\n",
						graph_slices[0]->gpu,
						(unsigned long long) (graph_slices[0]->edges),
						(unsigned long long) (graph_slices[0]->edges * sizeof(VertexId) * sizeof(SizeT)));

					if (retval = util::B40CPerror<0>(cudaMalloc(
							(void**) &graph_slices[0]->d_column_indices,
							graph_slices[0]->edges * sizeof(VertexId)),
						"CsrProblem cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror<0>(cudaMemcpy(
							graph_slices[0]->d_column_indices,
							h_column_indices,
							graph_slices[0]->edges * sizeof(VertexId),
							cudaMemcpyHostToDevice),
						"CsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

					// Allocate and initialize d_row_offsets

					printf("GPU %d row_offsets: %lld elements (%lld bytes)\n",
						graph_slices[0]->gpu,
						(unsigned long long) (graph_slices[0]->nodes + 1),
						(unsigned long long) (graph_slices[0]->nodes + 1) * sizeof(SizeT));

					if (retval = util::B40CPerror<0>(cudaMalloc(
							(void**) &graph_slices[0]->d_row_offsets,
							(graph_slices[0]->nodes + 1) * sizeof(SizeT)),
						"CsrProblem cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror<0>(cudaMemcpy(
							graph_slices[0]->d_row_offsets,
							h_row_offsets,
							(graph_slices[0]->nodes + 1) * sizeof(SizeT),
							cudaMemcpyHostToDevice),
						"CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
				}

			} else {

				// Create multiple GPU graph slices
				for (int gpu = 0; gpu < num_gpus; gpu++) {

					// Set device
					if (retval = util::B40CPerror<0>(cudaSetDevice(gpu),
						"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

					// Create stream
					cudaStream_t stream;
					if (retval = util::B40CPerror<0>(cudaStreamCreate(&stream),
						"CsrProblem cudaStreamCreate failed", __FILE__, __LINE__)) break;

					// Create slice
					graph_slices.push_back(new GraphSlice(gpu, stream));
				}
				if (retval) break;

				// Count up nodes and edges for each gpu
				for (VertexId node = 0; node < nodes; node++) {
					int gpu = GpuIndex(node);
					graph_slices[gpu]->nodes++;
					graph_slices[gpu]->edges += h_row_offsets[node + 1] - h_row_offsets[node];
				}

				// Allocate data structures for gpu on host
				SizeT **slice_row_offsets 			= new SizeT*[num_gpus];
				VertexId **slice_column_indices 	= new VertexId*[num_gpus];
				for (int gpu = 0; gpu < num_gpus; gpu++) {

					printf("GPU %d gets %lld vertices and %lld edges\n",
						gpu, (long long) graph_slices[gpu]->nodes, (long long) graph_slices[gpu]->edges);
					fflush(stdout);

					slice_row_offsets[gpu] = new SizeT[graph_slices[gpu]->nodes + 1];
					slice_row_offsets[gpu][0] = 0;

					slice_column_indices[gpu] = new VertexId[graph_slices[gpu]->edges];

					// Reset for construction
					graph_slices[gpu]->edges = 0;
				}

				printf("Done allocating gpu data structures on host\n");
				fflush(stdout);

				// Construct data structures for gpus on host
				for (VertexId node = 0; node < nodes; node++) {

					int gpu 				= GpuIndex(node);
					VertexId slice_row 		= GraphSliceRow(node);
					SizeT row_edges			= h_row_offsets[node + 1] - h_row_offsets[node];

					memcpy(
						slice_column_indices[gpu] + slice_row_offsets[gpu][slice_row],
						h_column_indices + h_row_offsets[node],
						row_edges * sizeof(VertexId));

					graph_slices[gpu]->edges += row_edges;
					slice_row_offsets[gpu][slice_row + 1] = graph_slices[gpu]->edges;

					// Mask in owning gpu
					for (int edge = 0; edge < row_edges; edge++) {
						VertexId *ptr = slice_column_indices[gpu] + slice_row_offsets[gpu][slice_row] + edge;
						VertexId owner = GpuIndex(*ptr);
						(*ptr) |= (owner << ProblemType::GPU_MASK_SHIFT);
					}
				}

				printf("Done constructing gpu data structures on host\n");
				fflush(stdout);

				// Initialize data structures on GPU
				for (int gpu = 0; gpu < num_gpus; gpu++) {

					// Set device
					if (util::B40CPerror<0>(cudaSetDevice(graph_slices[gpu]->gpu),
						"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

					// Allocate and initialize d_row_offsets: copy and adjust by gpu offset

					printf("GPU %d row_offsets: %lld elements (%lld bytes)\n",
						graph_slices[gpu]->gpu,
						(unsigned long long) (graph_slices[gpu]->nodes + 1),
						(unsigned long long) (graph_slices[gpu]->nodes + 1) * sizeof(SizeT));

					if (retval = util::B40CPerror<0>(cudaMalloc(
							(void**) &graph_slices[gpu]->d_row_offsets,
							(graph_slices[gpu]->nodes + 1) * sizeof(SizeT)),
						"CsrProblem cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror<0>(cudaMemcpy(
							graph_slices[gpu]->d_row_offsets,
							slice_row_offsets[gpu],
							(graph_slices[gpu]->nodes + 1) * sizeof(SizeT),
							cudaMemcpyHostToDevice),
						"CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

					// Allocate and initialize d_column_indices

					printf("GPU %d column_indices: %lld elements (%lld bytes)\n",
						graph_slices[gpu]->gpu,
						(unsigned long long) (graph_slices[gpu]->edges),
						(unsigned long long) (graph_slices[gpu]->edges * sizeof(VertexId) * sizeof(SizeT)));

					if (retval = util::B40CPerror<0>(cudaMalloc(
							(void**) &graph_slices[gpu]->d_column_indices,
							graph_slices[gpu]->edges * sizeof(VertexId)),
						"CsrProblem cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror<0>(cudaMemcpy(
							graph_slices[gpu]->d_column_indices,
							slice_column_indices[gpu],
							graph_slices[gpu]->edges * sizeof(VertexId),
							cudaMemcpyHostToDevice),
						"CsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

					// Cleanup host construction arrays
					if (slice_row_offsets[gpu]) delete slice_row_offsets[gpu];
					if (slice_column_indices[gpu]) delete slice_column_indices[gpu];
				}
				if (retval) break;

				printf("Done initializing gpu data structures on gpus\n");
				fflush(stdout);


				if (slice_row_offsets) delete slice_row_offsets;
				if (slice_column_indices) delete slice_column_indices;
			}

		} while (0);

		return retval;
	}


	/**
	 * Performs any initialization work needed for this problem type.  Must be called
	 * prior to each search
	 */
	cudaError_t Reset(
		FrontierType frontier_type,			// The frontier type (i.e., edge/vertex/mixed)
		double queue_sizing)			// Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).  0.0 is unspecified.
	{
		cudaError_t retval = cudaSuccess;

		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (retval = util::B40CPerror<0>(cudaSetDevice(graph_slices[gpu]->gpu),
				"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

			//
			// Allocate output labels if necessary
			//

			if (!graph_slices[gpu]->d_labels) {

				printf("GPU %d labels: %lld elements (%lld bytes)\n",
					graph_slices[gpu]->gpu,
					(unsigned long long) graph_slices[gpu]->nodes,
					(unsigned long long) graph_slices[gpu]->nodes * sizeof(VertexId));

				if (retval = util::B40CPerror<0>(cudaMalloc(
						(void**) &graph_slices[gpu]->d_labels,
						graph_slices[gpu]->nodes * sizeof(VertexId)),
					"CsrProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
			}


			//
			// Allocate visited masks for the entire graph if necessary
			//

			int visited_mask_bytes 			= ((nodes * sizeof(VisitedMask)) + 8 - 1) / 8;					// round up to the nearest VisitedMask
			int visited_mask_elements		= visited_mask_bytes * sizeof(VisitedMask);
			if (!graph_slices[gpu]->d_visited_mask) {

				/* printf("GPU %d visited mask: %lld elements (%lld bytes)\n", */
				/* 	graph_slices[gpu]->gpu, */
				/* 	(unsigned long long) visited_mask_elements, */
				/* 	(unsigned long long) visited_mask_bytes); */

				if (retval = util::B40CPerror<0>(cudaMalloc(
						(void**) &graph_slices[gpu]->d_visited_mask,
						visited_mask_bytes),
					"CsrProblem cudaMalloc d_visited_mask failed", __FILE__, __LINE__)) return retval;
			}


			//
			// Allocate frontier queues if necessary
			//

			// Determine frontier queue sizes
			SizeT new_frontier_elements[3] = {0,0,0};
			SizeT new_predecessor_elements[3] = {0,0,0};

			switch (frontier_type) {
			case VERTEX_FRONTIERS :
				// O(n) ping-pong global vertex frontiers
				new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
				new_frontier_elements[1] = new_frontier_elements[0];
				break;

			case EDGE_FRONTIERS :
				// O(m) ping-pong global edge frontiers
				new_frontier_elements[0] = double(graph_slices[gpu]->edges) * queue_sizing;
				new_frontier_elements[1] = new_frontier_elements[0];
				if (MARK_PREDECESSORS) {
					new_predecessor_elements[0] = new_frontier_elements[0];
					new_predecessor_elements[1] = new_frontier_elements[1];
				}
				break;

			case MIXED_FRONTIERS :
				// O(n) global vertex frontier, O(m) global edge frontier
				new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
				new_frontier_elements[1] = double(graph_slices[gpu]->edges) * queue_sizing;
				if (MARK_PREDECESSORS) {
					new_predecessor_elements[1] = new_frontier_elements[1];
				}
				break;

			case MULTI_GPU_FRONTIERS :
				// O(n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier
				new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * MULTI_GPU_VERTEX_FRONTIER_SCALE * queue_sizing;
				new_frontier_elements[1] = double(graph_slices[gpu]->edges) * queue_sizing;
				new_frontier_elements[2] = new_frontier_elements[1];
				if (MARK_PREDECESSORS) {
					new_predecessor_elements[1] = new_frontier_elements[1];
					new_predecessor_elements[2] = new_frontier_elements[2];
				}
				break;
			}

			// Iterate through global frontier queue setups
			for (int i = 0; i < 3; i++) {

				// Allocate frontier queue if not big enough
				if (graph_slices[gpu]->frontier_elements[i] < new_frontier_elements[i]) {

					// Free if previously allocated
					if (graph_slices[gpu]->frontier_queues.d_keys[i]) {
						if (retval = util::B40CPerror<0>(cudaFree(
							graph_slices[gpu]->frontier_queues.d_keys[i]),
								"GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
					}

					graph_slices[gpu]->frontier_elements[i] = new_frontier_elements[i];

					/* printf("GPU %d frontier queue[%d] (queue-sizing factor %.2fx): %lld elements (%lld bytes)\n", */
					/* 	graph_slices[gpu]->gpu, */
					/* 	i, */
					/* 	queue_sizing, */
					/* 	(unsigned long long) graph_slices[gpu]->frontier_elements[i], */
					/* 	(unsigned long long) graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)); */
					/* fflush(stdout); */

					if (retval = util::B40CPerror<0>(cudaMalloc(
						(void**) &graph_slices[gpu]->frontier_queues.d_keys[i],
						graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)),
							"CsrProblem cudaMalloc frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
				}


				// Allocate predecessor queue if not big enough
				if (graph_slices[gpu]->predecessor_elements[i] < new_predecessor_elements[i]) {

					// Free if previously allocated
					if (graph_slices[gpu]->frontier_queues.d_values[i]) {
						if (retval = util::B40CPerror<0>(cudaFree(
							graph_slices[gpu]->frontier_queues.d_values[i]),
								"GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
					}

					graph_slices[gpu]->predecessor_elements[i] = new_predecessor_elements[i];

					/* printf("GPU %d predecessor queue[%d] (queue-sizing factor %.2fx): %lld elements (%lld bytes)\n", */
					/* 	graph_slices[gpu]->gpu, */
					/* 	i, */
					/* 	queue_sizing, */
					/* 	(unsigned long long) graph_slices[gpu]->predecessor_elements[i], */
					/* 	(unsigned long long) graph_slices[gpu]->predecessor_elements[i] * sizeof(VertexId)); */
					/* fflush(stdout); */

					if (retval = util::B40CPerror<0>(cudaMalloc(
						(void**) &graph_slices[gpu]->frontier_queues.d_values[i],
						graph_slices[gpu]->predecessor_elements[i] * sizeof(VertexId)),
							"CsrProblem cudaMalloc frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
				}
			}


			//
			// Allocate duplicate filter mask if necessary (for multi-gpu)
			//

			if ((frontier_type == MULTI_GPU_FRONTIERS) && (!graph_slices[gpu]->d_filter_mask)) {

				printf("GPU %d_filter_mask flags: %lld elements (%lld bytes)\n",
					graph_slices[gpu]->gpu,
					(unsigned long long) graph_slices[gpu]->frontier_elements[1],
					(unsigned long long) graph_slices[gpu]->frontier_elements[1] * sizeof(ValidFlag));

				if (retval = util::B40CPerror<0>(cudaMalloc(
						(void**) &graph_slices[gpu]->d_filter_mask,
						graph_slices[gpu]->frontier_elements[1] * sizeof(ValidFlag)),
					"CsrProblem cudaMalloc d_filter_mask failed", __FILE__, __LINE__)) return retval;
			}


			//
			// Initialize labels and visited mask
			//

			int memset_block_size 		= 256;
			int memset_grid_size_max 	= 32 * 1024;	// 32K CTAs
			int memset_grid_size;

			// Initialize d_labels elements to -1
			memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VertexId><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
				graph_slices[gpu]->d_labels,
				-1,
				graph_slices[gpu]->nodes);

			if (retval = util::B40CPerror<0>(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) return retval;

			// Initialize d_visited_mask elements to 0
			memset_grid_size = B40C_MIN(memset_grid_size_max, (visited_mask_elements + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VisitedMask><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
				graph_slices[gpu]->d_visited_mask,
				0,
				visited_mask_elements);

			if (retval = util::B40CPerror<0>(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) return retval;

		}

		return retval;
	}
};


} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

