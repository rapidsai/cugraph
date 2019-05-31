/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#pragma once
#include "shfl.hxx"
#include "sm_utils.h"

namespace nvgraph
{
	//This file is to do a blockwide reduction by key as specialized for Key-Value Pairs.
//Each thread will call this function. There will be two outputs. One will be the calling thread's
//own output key value pair and the other will be the block-wide aggegrate reduction of the input items
//This is based on Duane Merrills's Exclusive Scan function in Cub

//Implementing key value pair to be called in device functions
template<typename IndexType_, typename ValueType_> //allow for different datatypes
struct KeyValuePair
{
	IndexType_ key;
	ValueType_ value;
};

//binary reduction operator to be applied to the values- we can template on the type on 
//the operator for the general case but only using sum () in our case so can simplify
template<typename SemiRingType_>
struct ReduceByKeySum
{
	SemiRingType_ SR;
	__host__ __device__ __forceinline__ ReduceByKeySum(SemiRingType_ SR) : SR(SR) //pass in semiring 
	{

	}
	template<typename IndexType_, typename ValueType_>
	__host__ __device__ __forceinline__ KeyValuePair<IndexType_, ValueType_> 
								operator() (const KeyValuePair<IndexType_, ValueType_> &first,
											const KeyValuePair<IndexType_, ValueType_> &second)
	{
		KeyValuePair<IndexType_, ValueType_> result = second;
		//check if they have matching keys and if so sum them
		if (first.key == second.key)
			result.value = SR.plus(first.value, result.value);
		return result;
	}
};
//Statically determien log2(N), rounded up
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
	/// Static logarithm value
	enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE }; // Inductive case
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
	enum {VALUE = (1 << (COUNT - 1) < N) ? // Base case
	COUNT :
	COUNT - 1 };
};

template<typename IndexType_, typename ValueType_, typename SemiRingType_, int BLOCK_DIM_X>
struct PrefixSum
{
	int laneId, warpId, linearTid;
	SemiRingType_ SR;
	//list constants
	enum
	{
		//number of threads per warp
		WARP_THREADS = 32, 
		// The number of warp scan steps log2
		STEPS = Log2<WARP_THREADS>::VALUE,
		// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
		SHFL_C = ((-1 << STEPS) & 31) << 8,
		//add in more enums for the warps!
		//calculate the thread block size in threads
		BLOCK_DIM_Y = 1,
		BLOCK_DIM_Z = 1,
		BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
		//calculate the number of active warps
		WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,
	};
	//constructor
	__device__ __forceinline__ PrefixSum(SemiRingType_ SR) : SR(SR)
	{
		laneId = utils::lane_id(); //set lane id
		linearTid = threadIdx.x; //simple for linear 1D block
		warpId = (WARPS == 1) ? 0 : linearTid / WARP_THREADS;
	}
 	
 	//Final function with the exclusive scan outputs one partial sum for the calling thread and the blockwide reduction
	__device__ __forceinline__ void ExclusiveKeyValueScan(
		KeyValuePair<IndexType_, ValueType_> &output, //input/output key value pair from the calling thread
		KeyValuePair<IndexType_,ValueType_> &blockAggegrate) //blockwide reduction output
	{
		KeyValuePair<IndexType_, ValueType_> inclusiveOutput;
		KeyValueScan(inclusiveOutput, output); //to get individual thread res
		CalcBlockAggregate(output, inclusiveOutput, blockAggegrate, (laneId > 0)); //to get blockwide res
	}

	//This function uses the inclusive scan below to calculate the exclusive scan
	__device__ __forceinline__ void KeyValueScan(
		KeyValuePair<IndexType_,ValueType_> &inclusiveOutput, //calling thread's inclusive-scan output item
		KeyValuePair<IndexType_,ValueType_> &exclusiveOutput) //calling thread's exclusive-scan output item
	{	//exclusiveOutput is the initial input as well
		InclusiveKeyValueScan(exclusiveOutput, inclusiveOutput); //inclusive starts at first number and last element is total reduction
		//to get exclusive output shuffle the keys and values both up by 1
		exclusiveOutput.key = utils::shfl_up(inclusiveOutput.key, 1);
		exclusiveOutput.value = utils::shfl_up(inclusiveOutput.value, 1);
	}

	//This function computes an inclusive scan odf key value pairs
	__device__ __forceinline__ void InclusiveKeyValueScan(
		KeyValuePair<IndexType_, ValueType_> input, //calling thread's input item
		KeyValuePair<IndexType_, ValueType_> &output //calling thread's input item
		)
	{
		//__shfl_up and __ballot are intrinsic functions require SM30 or greater-send error message for lower hardwares
		output = input;
		IndexType_ predKey = utils::shfl_up(output.key, 1); //shuffle key to next neighbor
		unsigned int ballot = utils::ballot((predKey != output.key));//intrinsic evaluates a condition for all threads in the warp and returns a 32-bit value 
		//where each bit gives the condition for the corresponding thread in the warp.

		//Mask away all lanes greater than ours
		ballot = ballot & utils::lane_mask_le();

		//Find index of first set bit
		int firstLane = max(0, 31 - __clz(ballot));//Count the number of consecutive leading zero bits, 
		//starting at the most significant bit (bit 31) of x. //Returns a value between 0 and 32 inclusive representing the number of zero bits. 
		//Iterate scan steps
		for (int step = 0; step < STEPS; ++step) //only called on double not key so not specific to key value pairs
		{
			output.value = SR.shflPlus(output.value, firstLane | SHFL_C, 1 << step); //plus defined on class operator
			//if (threadIdx.x + blockDim.x *blockIdx.x < 4)printf("%.1f\n", output.value);
		}
	}

	//This completes the warp-prefix scan.  Now we will use the Warp Aggregates to also calculate a blockwide aggregate
	// Update the calling thread's partial reduction with the warp-wide aggregates from preceding warps.  
	//Also returns block-wide aggregate
    __device__ __forceinline__ void CalcBlockAggregate( //can add in scan operators later
        KeyValuePair<IndexType_, ValueType_>   &partial,   //Calling thread's partial reduction
        KeyValuePair<IndexType_, ValueType_>   warpAggregate,     //Warp-wide aggregate reduction of input items
        KeyValuePair<IndexType_, ValueType_>   &blockAggregate,   //Threadblock-wide aggregate reduction of input items
        bool            laneValid = true)  //Whether or not the partial belonging to the current thread is valid
    {
    	//use shared memory in the block approach
        // Last lane in each warp shares its warp-aggregate
        //use 1D linear linear_tid def
        __shared__ KeyValuePair<IndexType_, ValueType_> warpAggregates[WARPS];
        if (laneId == WARP_THREADS - 1) //number of threads per warp
            warpAggregates[warpId] = warpAggregate; 
        //load into shared memory and wait until all threads are done
        __syncthreads();

        blockAggregate = warpAggregates[0];
        ReduceByKeySum<SemiRingType_> keyValAdd(SR); //call scn operator only add together if keys match
        for (int warp = 1; warp < WARPS; ++warp)
        {
        	KeyValuePair<IndexType_, ValueType_> inclusive = keyValAdd(blockAggregate, partial);
        	if (warpId == warp)
            	partial = (laneValid) ? inclusive : blockAggregate;

        	KeyValuePair<IndexType_, ValueType_> addend = warpAggregates[warp];
        	blockAggregate = keyValAdd(blockAggregate, addend); //only add if matching keys
        }
    }
};

} //end namespace nvgraph

