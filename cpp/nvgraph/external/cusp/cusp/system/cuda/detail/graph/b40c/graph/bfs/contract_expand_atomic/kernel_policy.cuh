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
 * Kernel configuration policy for BFS frontier contraction+expansion kernels.
 ******************************************************************************/

#pragma once

#include "../../../util/basic_utils.cuh"
#include "../../../util/cuda_properties.cuh"
#include "../../../util/cta_work_distribution.cuh"
#include "../../../util/soa_tuple.cuh"
#include "../../../util/srts_grid.cuh"
#include "../../../util/srts_soa_details.cuh"
#include "../../../util/io/modified_load.cuh"
#include "../../../util/io/modified_store.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {
namespace contract_expand_atomic {

/**
 * Kernel configuration policy for BFS frontier contraction+expansion kernels.
 *
 * Parameterizations of this type encapsulate our kernel-tuning parameters
 * (i.e., they are reflected via the static fields).
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ProblemType type parameters
	typename _ProblemType,								// BFS problem type (e.g., b40c::graph::bfs::ProblemType)

	// Machine parameters
	int _CUDA_ARCH,										// CUDA SM architecture to generate code for

	// Behavioral control parameters
	bool _INSTRUMENT,									// Whether or not to record per-CTA clock timing statistics (for detecting load imbalance)
	int _SATURATION_QUIT,								// If positive, signal that we're done with fused iterations if frontier size rises above (SATURATION_QUIT * grid_size)

	// Tunable parameters
	int _MIN_CTA_OCCUPANCY,												// Lower bound on number of CTAs to have resident per SM (influences per-CTA smem cache sizes and register allocation/spills)
	int _LOG_THREADS,													// Number of threads per CTA (log)
	int _LOG_LOAD_VEC_SIZE,												// Number of incoming frontier vertex-ids to dequeue in a single load (log)
	int _LOG_LOADS_PER_TILE,											// Number of such loads that constitute a tile of incoming frontier vertex-ids (log)
	int _LOG_RAKING_THREADS,											// Number of raking threads to use for prefix sum (log), range [5, LOG_THREADS]
	util::io::ld::CacheModifier _QUEUE_READ_MODIFIER,					// Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer, where util::io::ld::cg is req'd for fused-iteration implementations incorporating software global barriers.
	util::io::ld::CacheModifier _COLUMN_READ_MODIFIER,					// Load instruction cache-modifier for reading CSR column-indices
	util::io::ld::CacheModifier _ROW_OFFSET_ALIGNED_READ_MODIFIER,		// Load instruction cache-modifier for reading CSR row-offsets (when 8-byte aligned)
	util::io::ld::CacheModifier _ROW_OFFSET_UNALIGNED_READ_MODIFIER,	// Load instruction cache-modifier for reading CSR row-offsets (when 4-byte aligned)
	util::io::st::CacheModifier _QUEUE_WRITE_MODIFIER,					// Store instruction cache-modifier for writing outgoign frontier vertex-ids. Valid on SM2.0 or newer, where util::io::st::cg is req'd for fused-iteration implementations incorporating software global barriers.
	bool _WORK_STEALING,												// Whether or not incoming frontier tiles are distributed via work-stealing or by even-share.
	int _WARP_GATHER_THRESHOLD,											// Adjacency-list length above which we expand an that list using coarser-grained warp-based cooperative expansion (below which we perform fine-grained scan-based expansion)
	int _CTA_GATHER_THRESHOLD,											// Adjacency-list length above which we expand an that list using coarsest-grained CTA-based cooperative expansion (below which we perform warp-based expansion)
	int _END_BITMASK_CULL,												// BFS Iteration after which to skip bitmask filtering (Alternatively 0 to never perform bitmask filtering, -1 to always perform bitmask filtering)
	int _LOG_SCHEDULE_GRANULARITY>										// The scheduling granularity of incoming frontier tiles (for even-share work distribution only) (log)

struct KernelPolicy : _ProblemType
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	typedef _ProblemType 					ProblemType;
	typedef typename ProblemType::VertexId 	VertexId;
	typedef typename ProblemType::SizeT 	SizeT;

	static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER 					= _QUEUE_READ_MODIFIER;
	static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER 					= _COLUMN_READ_MODIFIER;
	static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER 		= _ROW_OFFSET_ALIGNED_READ_MODIFIER;
	static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER 	= _ROW_OFFSET_UNALIGNED_READ_MODIFIER;
	static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER 					= _QUEUE_WRITE_MODIFIER;

	enum {
		CUDA_ARCH						= _CUDA_ARCH,
		INSTRUMENT						= _INSTRUMENT,
		SATURATION_QUIT					= _SATURATION_QUIT,

		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_LOAD_VEC_SIZE  				= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= _LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_LOAD_STRIDE					= LOG_THREADS + LOG_LOAD_VEC_SIZE,
		LOAD_STRIDE						= 1 << LOG_LOAD_STRIDE,

		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY,

		WORK_STEALING					= _WORK_STEALING,
		WARP_GATHER_THRESHOLD			= _WARP_GATHER_THRESHOLD,
		CTA_GATHER_THRESHOLD			= _CTA_GATHER_THRESHOLD,
		END_BITMASK_CULL 			= _END_BITMASK_CULL,
	};


	// Prefix sum raking grid for coarse-grained expansion allocations
	typedef util::RakingGrid<
		CUDA_ARCH,
		SizeT,									// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			CoarseGrid;


	// Prefix sum raking grid for fine-grained expansion allocations
	typedef util::RakingGrid<
		CUDA_ARCH,
		SizeT,									// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			FineGrid;


	// Type for (coarse-partial, fine-partial) tuples
	typedef util::Tuple<SizeT, SizeT> TileTuple;


	// Structure-of-array (SOA) prefix sum raking grid type (CoarseGrid, FineGrid)
	typedef util::Tuple<
		CoarseGrid,
		FineGrid> RakingGridTuple;


	// Operational details type for SOA raking grid
	typedef util::RakingSoaDetails<
		TileTuple,
		RakingGridTuple> RakingSoaDetails;


	// Prefix sum tuple operator for SOA raking grid
	struct SoaScanOp
	{
		enum {
			IDENTITY_STRIDES = true,			// There is an "identity" region of warpscan storage exists for strides to index into
		};

		// SOA scan operator
		__device__ __forceinline__ TileTuple operator()(
			const TileTuple &first,
			const TileTuple &second)
		{
			return TileTuple(first.t0 + second.t0, first.t1 + second.t1);
		}

		// SOA identity operator
		__device__ __forceinline__ TileTuple operator()()
		{
			return TileTuple(0,0);
		}
	};


	/**
	 * Shared memory storage type for the CTA
	 */
	struct SmemStorage
	{
		// Persistent shared state for the CTA
		struct State {
			// Type describing four shared memory channels per warp for intra-warp communication
			typedef SizeT 						WarpComm[WARPS][4];

			// Whether or not we overflowed our outgoing frontier
			bool								overflowed;

			// Shared work-processing limits
			util::CtaWorkDistribution<SizeT>	work_decomposition;

			// Shared memory channels for intra-warp communication
			volatile WarpComm					warp_comm;
			int 								cta_comm;

			// Storage for scanning local contract-expand ranks
			SizeT 								coarse_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];
			SizeT 								fine_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];

			// Enqueue offset for neighbors of the current tile
			SizeT								fine_enqueue_offset;
			SizeT								coarse_enqueue_offset;

		} state;

		enum {
			// Amount of storage we can use for hashing scratch space under target occupancy
			MAX_SCRATCH_BYTES_PER_CTA		= (B40C_SMEM_BYTES(CUDA_ARCH) / _MIN_CTA_OCCUPANCY)
												- sizeof(State)
												- 140,											// Fudge-factor to guarantee occupancy

			SCRATCH_ELEMENT_SIZE 			= (ProblemType::MARK_PREDECESSORS) ?
													sizeof(SizeT) + sizeof(VertexId) :			// Need both gather offset and predecessor
													sizeof(SizeT),								// Only gather offset

			GATHER_ELEMENTS					= MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE,
			PARENT_ELEMENTS					= (ProblemType::MARK_PREDECESSORS) ?  GATHER_ELEMENTS : 0,
			HASH_ELEMENTS					= MAX_SCRATCH_BYTES_PER_CTA / sizeof(VertexId),

			WARP_HASH_ELEMENTS				= 128,												// Collision hash table size (per warp)
		};

		union {
			// Raking elements
			struct {
				SizeT 						coarse_raking_elements[CoarseGrid::TOTAL_RAKING_ELEMENTS];
				SizeT 						fine_raking_elements[FineGrid::TOTAL_RAKING_ELEMENTS];
			};

			// Scratch elements
			struct {
				SizeT 						gather_offsets[GATHER_ELEMENTS];
				VertexId 					gather_predecessors[PARENT_ELEMENTS];
			};
			volatile VertexId 				warp_hashtable[WARPS][WARP_HASH_ELEMENTS];
			VertexId 						cta_hashtable[HASH_ELEMENTS];
		};
	};

	enum {
		// Total number of smem quads needed by this kernel
		SMEM_QUADS						= B40C_QUADS(sizeof(SmemStorage)),

		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY					= B40C_SMEM_BYTES(CUDA_ARCH) / (SMEM_QUADS * sizeof(uint4)),
		CTA_OCCUPANCY  					= B40C_MIN(_MIN_CTA_OCCUPANCY, B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

		VALID							= (CTA_OCCUPANCY > 0),
	};
};


} // namespace contract_expand_atomic
} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

