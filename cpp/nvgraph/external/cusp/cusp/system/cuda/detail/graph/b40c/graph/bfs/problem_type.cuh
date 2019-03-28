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
 * BFS partition-contraction problem type
 ******************************************************************************/

#pragma once

#include "../../partition/problem_type.cuh"
#include "../../util/basic_utils.cuh"
#include "../../radix_sort/sort_utils.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {


/**
 * Type of BFS problem
 */
template <
	typename 	_VertexId,						// Type of signed integer to use as vertex id (e.g., uint32)
	typename 	_SizeT,							// Type of unsigned integer to use for array indexing (e.g., uint32)
	typename 	_VisitedMask,					// Type of unsigned integer to use for visited mask (e.g., uint8)
	typename 	_ValidFlag,						// Type of integer to use for contraction validity (e.g., uint8)
	bool 		_MARK_PREDECESSORS>				// Whether to mark predecessor-vertices (vs. distance-from-source)
struct ProblemType : partition::ProblemType<
	_VertexId, 																	// KeyType
	typename util::If<_MARK_PREDECESSORS, _VertexId, util::NullType>::Type,		// ValueType
	_SizeT>																		// SizeT
{
	typedef _VertexId														VertexId;
	typedef _VisitedMask													VisitedMask;
	typedef _ValidFlag														ValidFlag;
	typedef typename radix_sort::KeyTraits<VertexId>::ConvertedKeyType		UnsignedBits;		// Unsigned type corresponding to VertexId

	static const bool MARK_PREDECESSORS		= _MARK_PREDECESSORS;
	static const _VertexId LOG_MAX_GPUS		= 2;										// The "problem type" currently only reserves space for 4 gpu identities in upper vertex identifier bits
	static const _VertexId MAX_GPUS			= 1 << LOG_MAX_GPUS;

	static const _VertexId GPU_MASK_SHIFT	= (sizeof(_VertexId) * 8) - LOG_MAX_GPUS;
	static const _VertexId GPU_MASK			= (MAX_GPUS - 1) << GPU_MASK_SHIFT;			// Bitmask for masking off the lower vertex id bits to reveal owner gpu id
	static const _VertexId VERTEX_ID_MASK	= ~GPU_MASK;								// Bitmask for masking off the upper control bits in vertex identifiers
};


} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

