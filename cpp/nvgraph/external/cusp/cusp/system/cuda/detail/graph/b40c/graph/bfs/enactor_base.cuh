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
 * Base BFS Search Enactor
 ******************************************************************************/

#pragma once

#include "../../util/cuda_properties.cuh"
#include "../../util/cta_work_progress.cuh"
#include "../../util/error_utils.cuh"

#include "../../graph/bfs/csr_problem.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {



/**
 * Base class for breadth-first-search enactors.
 * 
 * A BFS search iteratively expands outwards from the given source node.  At 
 * each iteration, the algorithm discovers unvisited nodes that are adjacent 
 * to the nodes discovered by the previous iteration.  The first iteration 
 * discovers the source node. 
 */
class EnactorBase
{
protected:	

	//Device properties
	util::CudaProperties cuda_props;
	
	// Queue size counters and accompanying functionality
	util::CtaWorkProgressLifetime work_progress;

	FrontierType frontier_type;

public:

	// Allows display to stdout of search details
	bool DEBUG;

	FrontierType GetFrontierType() { return frontier_type;}

protected: 	

	/**
	 * Constructor.
	 */
	EnactorBase(FrontierType frontier_type, bool DEBUG) :
		frontier_type(frontier_type),
		DEBUG(DEBUG)
	{
		// Setup work progress (only needs doing once since we maintain
		// it in our kernel code)
		work_progress.Setup();
	}


	/**
	 * Utility function: Returns the default maximum number of threadblocks
	 * this enactor class can launch.
	 */
	int MaxGridSize(int cta_occupancy, int max_grid_size = 0)
	{
		if (max_grid_size <= 0) {
			// No override: Fully populate all SMs
			max_grid_size = this->cuda_props.device_props.multiProcessorCount * cta_occupancy;
		}

		return max_grid_size;
	}


	/**
	 * Utility method to display the contents of a device array
	 */
	template <typename T>
	void DisplayDeviceResults(
		T *d_data,
		size_t num_elements)
	{
		// Allocate array on host and copy back
		T *h_data = (T*) malloc(num_elements * sizeof(T));
		cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

		// Display data
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_data[i]);
			printf(", ");
		}
		printf("\n\n");

		// Cleanup
		if (h_data) free(h_data);
	}
};


} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

