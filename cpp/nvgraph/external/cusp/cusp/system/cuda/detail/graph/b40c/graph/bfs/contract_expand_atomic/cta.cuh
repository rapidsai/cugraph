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
 * CTA tile-processing abstraction for BFS frontier contraction+expansion
 ******************************************************************************/

#pragma once

#include "../../../util/device_intrinsics.cuh"
#include "../../../util/cta_work_progress.cuh"
#include "../../../util/scan/cooperative_scan.cuh"
#include "../../../util/io/modified_load.cuh"
#include "../../../util/io/modified_store.cuh"
#include "../../../util/io/load_tile.cuh"

#include "../../../util/soa_tuple.cuh"
#include "../../../util/scan/soa/cooperative_soa_scan.cuh"
#include "../../../util/operators.cuh"

#include "../two_phase/expand_atomic/cta.cuh"
#include "../two_phase/contract_atomic/cta.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {
namespace contract_expand_atomic {


/**
 * Templated texture reference for visited mask
 */
template <typename VisitedMask>
struct BitmaskTex
{
	static texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename VisitedMask>
texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> BitmaskTex<VisitedMask>::ref;


/**
 * Templated texture reference for row-offsets
 */
template <typename SizeT>
struct RowOffsetTex
{
	static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename SizeT>
texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;



/**
 * CTA tile-processing abstraction for BFS frontier contraction+expansion
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 			VertexId;
	typedef typename KernelPolicy::VisitedMask 			VisitedMask;
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
	VertexId 				*d_in;						// Incoming edge frontier
	VertexId 				*d_out;						// Outgoing edge frontier
	VertexId 				*d_predecessor_in;			// Incoming predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
	VertexId 				*d_predecessor_out;			// Outgoing predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
	VertexId				*d_column_indices;			// CSR column-indices array
	SizeT					*d_row_offsets;				// CSR row-offsets array
	VertexId				*d_labels;					// BFS labels to set
	VisitedMask 			*d_visited_mask;			// Mask for detecting visited status

	// Work progress
	VertexId 				iteration;					// Current BFS iteration
	VertexId 				queue_index;				// Current frontier queue counter index
	util::CtaWorkProgress	&work_progress;				// Atomic workstealing and queueing counters
	SizeT					max_edge_frontier;			// Maximum size (in elements) of edge frontiers
	int 					num_gpus;					// Number of GPUs

	// Operational details for raking grid
	RakingSoaDetails 		raking_soa_details;

	// Shared memory for the CTA
	SmemStorage				&smem_storage;

	// Whether or not to perform bitmask culling (incurs extra latency on small frontiers)
	bool 					bitmask_cull;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE>
	struct Tile :
		two_phase::expand_atomic::Cta<KernelPolicy>::template Tile<LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE>	// Derive from expand_atomic tile
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

		// Dequeued predecessors
		VertexId 	predecessor_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Temporary state for local culling
		int 		hash[LOADS_PER_TILE][LOAD_VEC_SIZE];			// Hash ids for vertex ids
		bool 		duplicate[LOADS_PER_TILE][LOAD_VEC_SIZE];		// Status as potential duplicate


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
			 * Inspect
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Translate vertex-id into local gpu row-id (currently stride of num_gpu)
					VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

					// Node is previously unvisited: compute row offset and length
					/* tile->row_offset[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id); */
					/* tile->row_length[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1) - tile->row_offset[LOAD][VEC]; */
					tile->row_offset[LOAD][VEC] = cta->d_row_offsets[row_id];
					tile->row_length[LOAD][VEC] = cta->d_row_offsets[row_id + 1] - tile->row_offset[LOAD][VEC];
				}

				tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					tile->row_length[LOAD][VEC] : 0;

				tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					0 : tile->row_length[LOAD][VEC];

				Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
			}


			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Location of mask byte to read
					SizeT mask_byte_offset = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) >> 3;

					// Bit in mask byte corresponding to current vertex id
					VisitedMask mask_bit = 1 << (tile->vertex_id[LOAD][VEC] & 7);

					// Read byte from from visited mask (tex)
					/* VisitedMask mask_byte = tex1Dfetch( */
					/* 	BitmaskTex<VisitedMask>::ref, */
					/* 	mask_byte_offset); */
					VisitedMask mask_byte = cta->d_visited_mask[mask_byte_offset];

					if (mask_bit & mask_byte) {

						// Seen it
						tile->vertex_id[LOAD][VEC] = -1;

					} else {

						util::io::ModifiedLoad<util::io::ld::cg>::Ld(
							mask_byte, cta->d_visited_mask + mask_byte_offset);

						if (mask_bit & mask_byte) {

							// Seen it
							tile->vertex_id[LOAD][VEC] = -1;

						} else {

							// Update with best effort
							mask_byte |= mask_bit;
							util::io::ModifiedStore<util::io::st::cg>::St(
								mask_byte,
								cta->d_visited_mask + mask_byte_offset);
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::BitmaskCull(cta, tile);
			}


			/**
			 * VertexCull
			 */
			static __device__ __forceinline__ void VertexCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

					// Load label of node
					VertexId label;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						label,
						cta->d_labels + row_id);


					if (label != -1) {

						// Seen it
						tile->vertex_id[LOAD][VEC] = -1;

					} else {

						if (KernelPolicy::MARK_PREDECESSORS) {

							// Update label with predecessor vertex
							util::io::ModifiedStore<util::io::st::cg>::St(
								tile->predecessor_id[LOAD][VEC],
								cta->d_labels + row_id);
						} else {

							// Update label with current iteration
							util::io::ModifiedStore<util::io::st::cg>::St(
								cta->iteration,
								cta->d_labels + row_id);
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::VertexCull(cta, tile);
			}


			/**
			 * CtaCull
			 */
			__device__ __forceinline__ void CtaCull(
				Cta *cta,
				Tile *tile)
			{
				// Hash the node-IDs into smem scratch

				int hash = tile->vertex_id[LOAD][VEC] % SmemStorage::HASH_ELEMENTS;
				bool duplicate = false;

				// Hash the node-IDs into smem scratch
				if (tile->vertex_id[LOAD][VEC] != -1) {
					cta->smem_storage.cta_hashtable[hash] = tile->vertex_id[LOAD][VEC];
				}

				__syncthreads();

				// Retrieve what vertices "won" at the hash locations. If a
				// different node beat us to this hash cell; we must assume
				// that we may not be a duplicate.  Otherwise assume that
				// we are a duplicate... for now.

				if (tile->vertex_id[LOAD][VEC] != -1) {
					VertexId hashed_node = cta->smem_storage.cta_hashtable[hash];
					duplicate = (hashed_node == tile->vertex_id[LOAD][VEC]);
				}

				__syncthreads();

				// For the possible-duplicates, hash in thread-IDs to select
				// one of the threads to be the unique one
				if (duplicate) {
					cta->smem_storage.cta_hashtable[hash] = threadIdx.x;
				}

				__syncthreads();

				// See if our thread won out amongst everyone with similar node-IDs
				if (duplicate) {
					// If not equal to our tid, we are not an authoritative thread
					// for this node-ID
					if (cta->smem_storage.cta_hashtable[hash] != threadIdx.x) {
						tile->vertex_id[LOAD][VEC] = -1;
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::CtaCull(cta, tile);
			}


			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					int warp_id 		= threadIdx.x >> 5;
					int hash 			= tile->vertex_id[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

					cta->smem_storage.warp_hashtable[warp_id][hash] = tile->vertex_id[LOAD][VEC];
					VertexId retrieved = cta->smem_storage.warp_hashtable[warp_id][hash];

					if (retrieved == tile->vertex_id[LOAD][VEC]) {

						cta->smem_storage.warp_hashtable[warp_id][hash] = threadIdx.x;
						VertexId tid = cta->smem_storage.warp_hashtable[warp_id][hash];
						if (tid != threadIdx.x) {
							tile->vertex_id[LOAD][VEC] = -1;
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::WarpCull(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Inspect(cta, tile);
			}

			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
			}

			// VertexCull
			static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::VertexCull(cta, tile);
			}

			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
			}

			/**
			 * CtaCull
			 */
			static __device__ __forceinline__ void CtaCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::CtaCull(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// Inspect
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {}

			// VertexCull
			static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile) {}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}

			// CtaCull
			static __device__ __forceinline__ void CtaCull(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Inspect dequeued vertices, updating label if necessary and
		 * obtaining edge-list details
		 */
		__device__ __forceinline__ void Inspect(Cta *cta)
		{
			Iterate<0, 0>::Inspect(cta, this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_visited_mask bitmask
		 */
		__device__ __forceinline__ void BitmaskCull(Cta *cta)
		{
			Iterate<0, 0>::BitmaskCull(cta, this);
		}

		/**
		 * Culls vertices
		 */
		__device__ __forceinline__ void VertexCull(Cta *cta)
		{
			Iterate<0, 0>::VertexCull(cta, this);
		}

		/**
		 * Warp cull
		 */
		__device__ __forceinline__ void WarpCull(Cta *cta)
		{
			Iterate<0, 0>::WarpCull(cta, this);

			__syncthreads();
		}

		/**
		 * CTA cull
		 */
		__device__ __forceinline__ void CtaCull(Cta *cta)
		{
			Iterate<0, 0>::WarpCull(cta, this);

			__syncthreads();
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId 				iteration,
		VertexId				queue_index,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_predecessor_in,
		VertexId 				*d_predecessor_out,
		VertexId 				*d_column_indices,
		SizeT 					*d_row_offsets,
		VertexId 				*d_labels,
		VisitedMask 			*d_visited_mask,
		util::CtaWorkProgress	&work_progress,
		SizeT					max_edge_frontier) :

			raking_soa_details(
				typename RakingSoaDetails::GridStorageSoa(
					smem_storage.coarse_raking_elements,
					smem_storage.fine_raking_elements),
				typename RakingSoaDetails::WarpscanSoa(
					smem_storage.state.coarse_warpscan,
					smem_storage.state.fine_warpscan),
				TileTuple(0, 0)),
			smem_storage(smem_storage),
			iteration(iteration),
			queue_index(queue_index),
			num_gpus(1),
			d_in(d_in),
			d_out(d_out),
			d_predecessor_in(d_predecessor_in),
			d_predecessor_out(d_predecessor_out),
			d_column_indices(d_column_indices),
			d_row_offsets(d_row_offsets),
			d_labels(d_labels),
			d_visited_mask(d_visited_mask),
			work_progress(work_progress),
			max_edge_frontier(max_edge_frontier),
			bitmask_cull(
				(KernelPolicy::END_BITMASK_CULL < 0) ?
					true : 														// always bitmask cull
					(KernelPolicy::END_BITMASK_CULL == 0) ?
						false : 												// never bitmask cull
						(iteration < KernelPolicy::END_BITMASK_CULL))
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

		// Load tile of predecessors
		if (KernelPolicy::MARK_PREDECESSORS) {

			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::QUEUE_READ_MODIFIER,
				false>::LoadValid(
					tile.predecessor_id,
					d_predecessor_in,
					cta_offset,
					guarded_elements);
		}

		// Cull visited vertices and update discovered vertices
		if (bitmask_cull) {
			tile.BitmaskCull(this);		// using global visited mask
		}
		tile.VertexCull(this);			// using vertex visitation status (update discovered vertices)

		// Cull nearby duplicates from the incoming frontier using collision-hashing
//		tile.CtaCull(this);				// doesn't seem to be worthwhile
		tile.WarpCull(this);

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
			return;
		}

		// Enqueue valid edge lists into outgoing queue (includes barrier)
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



} // namespace contract_expand_atomic
} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

