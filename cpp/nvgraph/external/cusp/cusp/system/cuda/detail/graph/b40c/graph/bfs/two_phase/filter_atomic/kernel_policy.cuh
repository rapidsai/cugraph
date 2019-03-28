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
 * Kernel configuration policy for BFS edge-frontier filtering kernels
 ******************************************************************************/

#pragma once

#include "../../../../util/basic_utils.cuh"
#include "../../../../util/cuda_properties.cuh"
#include "../../../../util/cta_work_distribution.cuh"
#include "../../../../util/srts_grid.cuh"
#include "../../../../util/srts_details.cuh"
#include "../../../../util/io/modified_load.cuh"
#include "../../../../util/io/modified_store.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {
namespace two_phase {
namespace filter_atomic {



/**
 * Kernel configuration policy for BFS frontier contraction kernels.
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
	int _SATURATION_QUIT,								// If positive, signal that we're done with two-phase iterations if frontier size drops below (SATURATION_QUIT * grid_size)

	// Tunable parameters
	int _MIN_CTA_OCCUPANCY,								// Lower bound on number of CTAs to have resident per SM (influences per-CTA smem cache sizes and register allocation/spills)
	int _LOG_THREADS,									// Number of threads per CTA (log)
	int _LOG_LOAD_VEC_SIZE,								// Number of incoming frontier vertex-ids to dequeue in a single load (log)
	int _LOG_LOADS_PER_TILE,							// Number of such loads that constitute a tile of incoming frontier vertex-ids (log)
	int _LOG_RAKING_THREADS,							// Number of raking threads to use for prefix sum (log), range [5, LOG_THREADS]
	util::io::ld::CacheModifier _QUEUE_READ_MODIFIER,	// Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer, where util::io::ld::cg is req'd for fused-iteration implementations incorporating software global barriers.
	util::io::st::CacheModifier _QUEUE_WRITE_MODIFIER,	// Store instruction cache-modifier for writing outgoign frontier vertex-ids. Valid on SM2.0 or newer, where util::io::st::cg is req'd for fused-iteration implementations incorporating software global barriers.
	bool _WORK_STEALING,								// Whether or not incoming frontier tiles are distributed via work-stealing or by even-share.
	int _LOG_SCHEDULE_GRANULARITY>						// The scheduling granularity of incoming frontier tiles (for even-share work distribution only) (log)

struct KernelPolicy : _ProblemType
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	typedef _ProblemType 					ProblemType;
	typedef typename ProblemType::VertexId 	VertexId;
	typedef typename ProblemType::SizeT 	SizeT;

	static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER 	= _QUEUE_READ_MODIFIER;
	static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER 	= _QUEUE_WRITE_MODIFIER;
	static const bool WORK_STEALING									= _WORK_STEALING;

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
	};


	// Prefix sum raking grid for contraction allocations
	typedef util::RakingGrid<
		CUDA_ARCH,
		SizeT,									// Partial type (valid counts)
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			RakingGrid;


	// Operational details type for raking grid type
	typedef util::RakingDetails<RakingGrid> RakingDetails;


	/**
	 * Shared memory storage type for the CTA
	 */
	struct SmemStorage
	{
		// Persistent shared state for the CTA
		struct State {

			// Shared work-processing limits
			util::CtaWorkDistribution<SizeT>	work_decomposition;

			// Storage for scanning local ranks
			SizeT 								warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];

			// Prefix sum raking lanes
			SizeT								raking_elements[RakingGrid::TOTAL_RAKING_ELEMENTS];

		} state;

	};

	enum {
		THREAD_OCCUPANCY	= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY		= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
		CTA_OCCUPANCY  		= B40C_MIN(_MIN_CTA_OCCUPANCY, B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),
		VALID				= (CTA_OCCUPANCY > 0),
	};
};

} // namespace filter_atomic
} // namespace two_phase
} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

