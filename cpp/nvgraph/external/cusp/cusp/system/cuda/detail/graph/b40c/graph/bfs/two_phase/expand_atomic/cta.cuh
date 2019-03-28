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
 * CTA tile-processing abstraction for BFS frontier expansion
 ******************************************************************************/

#pragma once

#include "../../../../util/device_intrinsics.cuh"
#include "../../../../util/cta_work_progress.cuh"
#include "../../../../util/scan/cooperative_scan.cuh"
#include "../../../../util/io/modified_load.cuh"
#include "../../../../util/io/modified_store.cuh"
#include "../../../../util/io/load_tile.cuh"
#include "../../../../util/operators.cuh"

#include "../../../../util/soa_tuple.cuh"
#include "../../../../util/scan/soa/cooperative_soa_scan.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {
namespace two_phase {
namespace expand_atomic {


/**
 * CTA tile-processing abstraction for BFS frontier expansion
 */
template <typename SizeT>
struct RowOffsetTex
{
	static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename SizeT>
texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;

template<typename SizeT, typename VertexId>
struct Tex
{
  static __device__ __forceinline__ VertexId fetch(SizeT* row_offsets, VertexId row_id)
  {
     return row_offsets[row_id];
  }
};

template<typename VertexId>
struct Tex<int, VertexId>
{
  static __device__ __forceinline__ VertexId fetch(int* row_offsets, VertexId row_id)
  {
     return tex1Dfetch(RowOffsetTex<int>::ref, row_id);
  }
};

/**
 * Derivation of KernelPolicy that encapsulates tile-processing routines
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 			VertexId;
	typedef typename KernelPolicy::SizeT 				SizeT;

	typedef typename KernelPolicy::SmemStorage			SmemStorage;

	typedef typename KernelPolicy::SoaScanOp			SoaScanOp;
	typedef typename KernelPolicy::RakingSoaDetails 	RakingSoaDetails;
	typedef typename KernelPolicy::TileTuple 			TileTuple;

	typedef util::Tuple<
		SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
		SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> 		RankSoa;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	VertexId 				*d_in;						// Incoming vertex frontier
	VertexId 				*d_out;						// Outgoing edge frontier
	VertexId 				*d_predecessor_out;			// Outgoing predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
	VertexId				*d_column_indices;			// CSR column-indices array
	SizeT					*d_row_offsets;				// CSR row-offsets array

	// Work progress
	VertexId 				queue_index;				// Current frontier queue counter index
	util::CtaWorkProgress	&work_progress;				// Atomic workstealing and queueing counters
	SizeT					max_edge_frontier;			// Maximum size (in elements) of outgoing edge frontier
	int 					num_gpus;					// Number of GPUs

	// Operational details for raking grid
	RakingSoaDetails 		raking_soa_details;

	// Shared memory for the CTA
	SmemStorage				&smem_storage;



	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile of incoming vertex frontier to process
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
			LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
		};

		typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Edge list details
		SizeT		row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Global scatter offsets.  Coarse for CTA/warp-based scatters, fine for scan-based scatters
		SizeT 		fine_count;
		SizeT		coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Progress for expanding scan-based gather offsets
		SizeT		row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		progress;

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate next vector element
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * Init
			 */
			template <typename Tile>
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				tile->row_length[LOAD][VEC] = 0;
				tile->row_progress[LOAD][VEC] = 0;

				Iterate<LOAD, VEC + 1>::Init(tile);
			}

			/**
			 * Inspect
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Translate vertex-id into local gpu row-id (currently stride of num_gpu)
					VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

					// Load neighbor row range from d_row_offsets
					Vec2SizeT row_range;
					row_range.x = Tex<SizeT, VertexId>::fetch(cta->d_row_offsets, row_id);
					row_range.y = Tex<SizeT, VertexId>::fetch(cta->d_row_offsets, row_id + 1);

					// Node is previously unvisited: compute row offset and length
					tile->row_offset[LOAD][VEC] = row_range.x;
					tile->row_length[LOAD][VEC] = row_range.y - row_range.x;
				}

				tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					tile->row_length[LOAD][VEC] : 0;

				tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					0 : tile->row_length[LOAD][VEC];

				Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
			}


			/**
			 * Expand by CTA
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
			{
				// CTA-based expansion/loading
				while (true) {

					// Vie
					if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD) {
						cta->smem_storage.state.cta_comm = threadIdx.x;
					}

					__syncthreads();

					// Check
					int owner = cta->smem_storage.state.cta_comm;
					if (owner == KernelPolicy::THREADS) {
						// No contenders
						break;
					}

					if (owner == threadIdx.x) {

						// Got control of the CTA: command it
						cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];										// start
						cta->smem_storage.state.warp_comm[0][1] = tile->coarse_row_rank[LOAD][VEC];									// queue rank
						cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
						if (KernelPolicy::MARK_PREDECESSORS) {
							cta->smem_storage.state.warp_comm[0][3] = tile->vertex_id[LOAD][VEC];									// predecessor
						}

						// Unset row length
						tile->row_length[LOAD][VEC] = 0;

						// Unset my command
						cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;	// invalid
					}

					__syncthreads();

					// Read commands
					SizeT coop_offset 	= cta->smem_storage.state.warp_comm[0][0];
					SizeT coop_rank	 	= cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
					SizeT coop_oob 		= cta->smem_storage.state.warp_comm[0][2];

					VertexId predecessor_id;
					if (KernelPolicy::MARK_PREDECESSORS) {
						predecessor_id = cta->smem_storage.state.warp_comm[0][3];
					}

					VertexId neighbor_id;
					while (coop_offset + KernelPolicy::THREADS < coop_oob) {

						// Gather
						util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
							neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);

						// Scatter neighbor
						util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
							neighbor_id,
							cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

						if (KernelPolicy::MARK_PREDECESSORS) {
							// Scatter predecessor
							util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
								predecessor_id,
								cta->d_predecessor_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
						}

						coop_offset += KernelPolicy::THREADS;
						coop_rank += KernelPolicy::THREADS;
					}

                    if (coop_offset + threadIdx.x < coop_oob) {
                        // Gather
                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                            neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);

                        // Scatter neighbor
                        util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                            neighbor_id, cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

                        if (KernelPolicy::MARK_PREDECESSORS) {
                            // Scatter predecessor
                            util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                predecessor_id, cta->d_predecessor_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
                        }
                    }

				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by warp
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
			{
				if (KernelPolicy::WARP_GATHER_THRESHOLD < KernelPolicy::CTA_GATHER_THRESHOLD) {

					// Warp-based expansion/loading
					int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS(KernelPolicy::CUDA_ARCH);
					int lane_id = util::LaneId();

					while (__any(tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)) {

						if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD) {
							// Vie for control of the warp
							cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
						}

						if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0]) {

							// Got control of the warp
							cta->smem_storage.state.warp_comm[warp_id][0] = tile->row_offset[LOAD][VEC];									// start
							cta->smem_storage.state.warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC];								// queue rank
							cta->smem_storage.state.warp_comm[warp_id][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
							if (KernelPolicy::MARK_PREDECESSORS) {
								cta->smem_storage.state.warp_comm[warp_id][3] = tile->vertex_id[LOAD][VEC];								// predecessor
							}

							// Unset row length
							tile->row_length[LOAD][VEC] = 0;
						}

						SizeT coop_offset 	= cta->smem_storage.state.warp_comm[warp_id][0];
						SizeT coop_rank 	= cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
						SizeT coop_oob 		= cta->smem_storage.state.warp_comm[warp_id][2];

						VertexId predecessor_id;
						if (KernelPolicy::MARK_PREDECESSORS) {
							predecessor_id = cta->smem_storage.state.warp_comm[warp_id][3];
						}

						VertexId neighbor_id;
						while (coop_offset  + B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH) < coop_oob) {

							// Gather
							util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
								neighbor_id, cta->d_column_indices + coop_offset + lane_id);

							// Scatter neighbor
							util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
								neighbor_id, cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

							if (KernelPolicy::MARK_PREDECESSORS) {
								// Scatter predecessor
								util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
									predecessor_id, cta->d_predecessor_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
							}

							coop_offset += B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH);
							coop_rank += B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH);
						}

						if (coop_offset + lane_id < coop_oob) {
							// Gather
							util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
								neighbor_id, cta->d_column_indices + coop_offset + lane_id);

							// Scatter neighbor
							util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
								neighbor_id, cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

							if (KernelPolicy::MARK_PREDECESSORS) {
								// Scatter predecessor
								util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
									predecessor_id, cta->d_predecessor_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
							}
						}
					}

					// Next vector element
					Iterate<LOAD, VEC + 1>::ExpandByWarp(cta, tile);
				}
			}


			/**
			 * Expand by scan
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				// Attempt to make further progress on this dequeued item's neighbor
				// list if its current offset into local scratch is in range
				SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;

				while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
					(scratch_offset < SmemStorage::GATHER_ELEMENTS))
				{
					// Put gather offset into scratch space
					cta->smem_storage.gather_offsets[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];

					if (KernelPolicy::MARK_PREDECESSORS) {
						// Put dequeued vertex as the predecessor into scratch space
						cta->smem_storage.gather_predecessors[scratch_offset] = tile->vertex_id[LOAD][VEC];
					}

					tile->row_progress[LOAD][VEC]++;
					scratch_offset++;
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * Init
			 */
			template <typename Tile>
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Init(tile);
			}

			/**
			 * Inspect
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Inspect(cta, tile);
			}

			/**
			 * Expand by CTA
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by warp
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByWarp(cta, tile);
			}

			/**
			 * Expand by scan
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// Init
			template <typename Tile>
			static __device__ __forceinline__ void Init(Tile *tile) {}

			// Inspect
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

			// ExpandByCta
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile) {}

			// ExpandByWarp
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile) {}

			// ExpandByScan
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		__device__ __forceinline__ Tile()
		{
			Iterate<0, 0>::Init(this);
		}

		/**
		 * Inspect dequeued vertices, updating label if necessary and
		 * obtaining edge-list details
		 */
		template <typename Cta>
		__device__ __forceinline__ void Inspect(Cta *cta)
		{
			Iterate<0, 0>::Inspect(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices at CTA-expansion granularity
		 */
		template <typename Cta>
		__device__ __forceinline__ void ExpandByCta(Cta *cta)
		{
			Iterate<0, 0>::ExpandByCta(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices a warp-expansion granularity
		 */
		template <typename Cta>
		__device__ __forceinline__ void ExpandByWarp(Cta *cta)
		{
			Iterate<0, 0>::ExpandByWarp(cta, this);
		}

		/**
		 * Expands neighbor lists by local scan rank
		 */
		template <typename Cta>
		__device__ __forceinline__ void ExpandByScan(Cta *cta)
		{
			Iterate<0, 0>::ExpandByScan(cta, this);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId 				queue_index,
		int						num_gpus,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_predecessor_out,
		VertexId 				*d_column_indices,
		SizeT 					*d_row_offsets,
		util::CtaWorkProgress	&work_progress,
		SizeT					max_edge_frontier) :

			queue_index(queue_index),
			num_gpus(num_gpus),
			smem_storage(smem_storage),
			raking_soa_details(
				typename RakingSoaDetails::GridStorageSoa(
					smem_storage.coarse_raking_elements,
					smem_storage.fine_raking_elements),
				typename RakingSoaDetails::WarpscanSoa(
					smem_storage.state.coarse_warpscan,
					smem_storage.state.fine_warpscan),
				TileTuple(0, 0)),
			d_in(d_in),
			d_out(d_out),
			d_predecessor_out(d_predecessor_out),
			d_column_indices(d_column_indices),
			d_row_offsets(d_row_offsets),
			work_progress(work_progress),
			max_edge_frontier(max_edge_frontier)
	{
		if (threadIdx.x == 0) {
			smem_storage.state.cta_comm = KernelPolicy::THREADS;		// invalid
			smem_storage.state.overflowed = false;						// valid
		}
	}


	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::QUEUE_READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_id,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		// Inspect dequeued vertices, updating label and obtaining
		// edge-list details
		tile.Inspect(this);

		// Scan tile with carry update in raking threads
		SoaScanOp scan_op;
		TileTuple totals;
		util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
			totals,
			raking_soa_details,
			RankSoa(tile.coarse_row_rank, tile.fine_row_rank),
			scan_op);

		SizeT coarse_count = totals.t0;
		tile.fine_count = totals.t1;

		// Use a single atomic add to reserve room in the queue
		if (threadIdx.x == 0) {

			SizeT enqueue_amt = coarse_count + tile.fine_count;
			SizeT enqueue_offset = work_progress.Enqueue(enqueue_amt, queue_index + 1);

			smem_storage.state.coarse_enqueue_offset = enqueue_offset;
			smem_storage.state.fine_enqueue_offset = enqueue_offset + coarse_count;

			// Check for queue overflow due to redundant expansion
			if (enqueue_offset + enqueue_amt >= max_edge_frontier) {
				smem_storage.state.overflowed = true;
				work_progress.SetOverflow<SizeT>();
			}
		}

		// Protect overflowed flag
		__syncthreads();

		// Quit if overflow
		if (smem_storage.state.overflowed) {
			util::ThreadExit();
		}

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByCta(this);

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByWarp(this);

		//
		// Enqueue the adjacency lists of unvisited node-IDs by repeatedly
		// gathering edges into the scratch space, and then
		// having the entire CTA copy the scratch pool into the outgoing
		// frontier queue.
		//

		tile.progress = 0;
		while (tile.progress < tile.fine_count) {

			// Fill the scratch space with gather-offsets for neighbor-lists.
			tile.ExpandByScan(this);

			__syncthreads();

			// Copy scratch space into queue
			int scratch_remainder = B40C_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);

			for (int scratch_offset = threadIdx.x;
				scratch_offset < scratch_remainder;
				scratch_offset += KernelPolicy::THREADS)
			{
				// Gather a neighbor
				VertexId neighbor_id;
				util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
					neighbor_id,
					d_column_indices + smem_storage.gather_offsets[scratch_offset]);

				// Scatter it into queue
				util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
					neighbor_id,
					d_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);

				if (KernelPolicy::MARK_PREDECESSORS) {
					// Scatter predecessor it into queue
					VertexId predecessor_id = smem_storage.gather_predecessors[scratch_offset];
					util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
						predecessor_id,
						d_predecessor_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
				}
			}

			tile.progress += SmemStorage::GATHER_ELEMENTS;

			__syncthreads();
		}
	}
};



} // namespace expand_atomic
} // namespace two_phase
} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

