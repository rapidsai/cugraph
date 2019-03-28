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
 * Kernel configuration policy for fused-iteration two-phase BFS kernels.
 ******************************************************************************/

#pragma once

#include <b40c/graph/bfs/two_phase/contract_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/two_phase/expand_atomic/kernel_policy.cuh>

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {
namespace two_phase {

/**
 * Kernel configuration policy for fused-iteration two-phase BFS kernels.
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
	int CUDA_ARCH,										// CUDA SM architecture to generate code for

	// Behavioral control parameters
	bool _INSTRUMENT,									// Whether or not to record per-CTA clock timing statistics (for detecting load imbalance)
	int SATURATION_QUIT,								// Early-exit threshold factor for the incoming frontier, i.e., quit when: frontier_size < SATURATION_QUIT * grid_size * TILE_SIZE.  (Can be -1 to never perform bitmask filtering)

	// Tunable parameters (generic)
	int MIN_CTA_OCCUPANCY,												// Lower bound on number of CTAs to have resident per SM (influences per-CTA smem cache sizes and register allocation/spills)
	int _LOG_THREADS,													// Number of threads per CTA (log)
	util::io::ld::CacheModifier QUEUE_READ_MODIFIER,					// Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer, where util::io::ld::cg is req'd for fused-iteration implementations incorporating software global barriers.
	util::io::ld::CacheModifier COLUMN_READ_MODIFIER,					// Load instruction cache-modifier for reading CSR column-indices
	util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER,		// Load instruction cache-modifier for reading CSR row-offsets (when 8-byte aligned)
	util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER,		// Load instruction cache-modifier for reading CSR row-offsets (when 4-byte aligned)
	util::io::st::CacheModifier QUEUE_WRITE_MODIFIER,					// Store instruction cache-modifier for writing outgoign frontier vertex-ids. Valid on SM2.0 or newer, where util::io::st::cg is req'd for fused-iteration implementations incorporating software global barriers.

	// Tunable parameters (contract)
	int CONTRACT_LOG_LOAD_VEC_SIZE,										// Number of incoming frontier vertex-ids to dequeue in a single load (log)
	int CONTRACT_LOG_LOADS_PER_TILE,									// Number of such loads that constitute a tile of incoming frontier vertex-ids (log)
	int CONTRACT_LOG_RAKING_THREADS,									// Number of raking threads to use for prefix sum (log), range [5, LOG_THREADS]
	bool CONTRACT_WORK_STEALING,										// Whether or not incoming frontier tiles are distributed via work-stealing or by even-share.
	int CONTRACT_END_BITMASK_CULL,								// Threshold factor for incoming frontier queue length above which we perform bitmask-based culling prior to regular label-checking, i.e., bitmask filter when: frontier_size < (SATURATION_QUIT * grid_size * TILE_SIZE).  (Can be -1 to never perform bitmask filtering)
	int CONTRACT_LOG_SCHEDULE_GRANULARITY,								// The scheduling granularity of incoming frontier tiles (for even-share work distribution only) (log)

	// Tunable parameters (expand)
	int EXPAND_LOG_LOAD_VEC_SIZE,										// Number of incoming frontier vertex-ids to dequeue in a single load (log)
	int EXPAND_LOG_LOADS_PER_TILE,										// Number of such loads that constitute a tile of incoming frontier vertex-ids (log)
	int EXPAND_LOG_RAKING_THREADS,										// Number of raking threads to use for prefix sum (log), range [5, LOG_THREADS]
	bool EXPAND_WORK_STEALING,											// Whether or not incoming frontier tiles are distributed via work-stealing or by even-share.
	int EXPAND_WARP_GATHER_THRESHOLD,									// Adjacency-list length above which we expand an that list using coarser-grained warp-based cooperative expansion (below which we perform fine-grained scan-based expansion)
	int EXPAND_CTA_GATHER_THRESHOLD,									// Adjacency-list length above which we expand an that list using coarsest-grained CTA-based cooperative expansion (below which we perform warp-based expansion)
	int EXPAND_LOG_SCHEDULE_GRANULARITY>								// The scheduling granularity of incoming frontier tiles (for even-share work distribution only) (log)

struct KernelPolicy : _ProblemType
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	typedef _ProblemType 							ProblemType;

	typedef typename ProblemType::VertexId 			VertexId;
	typedef typename ProblemType::SizeT 			SizeT;
	typedef typename ProblemType::VisitedMask 		VisitedMask;

	typedef contract_atomic::KernelPolicy<
		_ProblemType,
		CUDA_ARCH,
		_INSTRUMENT,
		SATURATION_QUIT,
		true,										// DEQUEUE_PROBLEM_SIZE
		MIN_CTA_OCCUPANCY,
		_LOG_THREADS,
		CONTRACT_LOG_LOAD_VEC_SIZE,
		CONTRACT_LOG_LOADS_PER_TILE,
		CONTRACT_LOG_RAKING_THREADS,
		QUEUE_READ_MODIFIER,
		QUEUE_WRITE_MODIFIER,
		CONTRACT_WORK_STEALING,
		CONTRACT_END_BITMASK_CULL,
		CONTRACT_LOG_SCHEDULE_GRANULARITY>
			ContractKernelPolicy;

	typedef expand_atomic::KernelPolicy<
		_ProblemType,
		CUDA_ARCH,
		_INSTRUMENT,
		MIN_CTA_OCCUPANCY,
		_LOG_THREADS,
		EXPAND_LOG_LOAD_VEC_SIZE,
		EXPAND_LOG_LOADS_PER_TILE,
		EXPAND_LOG_RAKING_THREADS,
		QUEUE_READ_MODIFIER,
		COLUMN_READ_MODIFIER,
		ROW_OFFSET_ALIGNED_READ_MODIFIER,
		ROW_OFFSET_UNALIGNED_READ_MODIFIER,
		QUEUE_WRITE_MODIFIER,
		EXPAND_WORK_STEALING,
		EXPAND_WARP_GATHER_THRESHOLD,
		EXPAND_CTA_GATHER_THRESHOLD,
		EXPAND_LOG_SCHEDULE_GRANULARITY>
			ExpandKernelPolicy;

	/**
	 * Shared memory storage type for the CTA
	 */
	union SmemStorage
	{
		typename ContractKernelPolicy::SmemStorage 	contract;
		typename ExpandKernelPolicy::SmemStorage 	expand;
	};

	// Constants
	enum {
		INSTRUMENT						= _INSTRUMENT,
		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		// Total number of smem quads needed by this kernel
		SMEM_QUADS						= B40C_QUADS(sizeof(SmemStorage)),

		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY					= B40C_SMEM_BYTES(CUDA_ARCH) / (SMEM_QUADS * sizeof(uint4)),
		CTA_OCCUPANCY  					= B40C_MIN(MIN_CTA_OCCUPANCY, B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

		VALID							= (CTA_OCCUPANCY > 0),
	};
};


} // namespace two_phase
} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

