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

 /* This file contains the nvgraph generalized implementation of the Duane Merrill's CUB CSRMV using MergePath */

#include "include/nvgraph_csrmv.hxx"
#include "include/exclusive_kv_scan.hxx" //atomics are included in semiring
#include "include/semiring.hxx"
#include "include/nvgraph_error.hxx"
 
//IMPORTANT: IndexType_ must be a signed integer, long, long long etc. Unsigned int is not supported, since -1 is
 //used as a flag value

 namespace nvgraph{

 //Calculates SM to be used-add to cpp host file
__forceinline__ cudaError_t SmVersion(int &smVersion, int deviceOrdinal)
{
    cudaError_t error = cudaSuccess; //assume sucess and state otherwise if fails condition
    do
    {
        //Find out SM version
        int major, minor;
        if (error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceOrdinal)) break;
        if (error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceOrdinal)) break;
        smVersion = 100 * major + 10 * minor;
    } while(0);
    return error;
} 

template<
int _BLOCK_THREADS, //number of threads per thread block
int _ITEMS_PER_THREAD> //number of items per individual thread
struct SpmvBlockThread //this is in agent file other template parameters ignoring for now
{
//set constants
	enum
	{
		BLOCK_THREADS = _BLOCK_THREADS, //number of threads per thread block
		ITEMS_PER_THREAD = _ITEMS_PER_THREAD, //number of items per thread per tile(tid) of input
	};
};

//This function calculates the MergePath(load-balancing) for each thread by doing a binary search
//along the diagonal
template<typename IndexType_>
__device__ __forceinline__ void MergePathSearch(
 		IndexType_ diag,
 		IndexType_ *A, //rowoffsets + 1 
 		IndexType_ offset, //counter array 
 		IndexType_ A_length, 
 		IndexType_ B_length, 
 		Coord<IndexType_> &pathCoord) //returned by reference stores the path
 {
 	IndexType_ splitMin = max(diag - B_length, IndexType_(0)); //must be nonnegative
 	IndexType_ splitMax = min(diag, A_length); //stay in bounds
	//do binary search along diagonal
 	while (splitMin < splitMax)
 	{
 		IndexType_ splitPivot = (splitMin + splitMax) / 2; //take average integer division-start in middle so can go up or down diagonal
 		if (A[splitPivot] <= diag - splitPivot - 1 + offset) //i+j = diag -1 along cross diag **ignored B
 			//move up A and down B from (i,j) to (i-1,j+1)
 		{
 			splitMin = splitPivot + 1; //increase a in case that it is less clearly before split_min <= split_pivot less than average
 		}
 		else
 		{
 			//move down A and up B
 			splitMax = splitPivot;
 		}
 	}
 	//transform back to array coordinates from cross diagaonl coordinates
 	pathCoord.x = min(splitMin, A_length); //make sure do not go out of bounds;
 	//constraint i + j  = k
	pathCoord.y = diag - splitMin;
 }

 //Spmv search kernel that calls merge path and identifies the merge path starting coordinates for each tile
 template <typename SpmvBlockThread, typename IndexType_, typename ValueType_>
 __global__ void DeviceSpmvSearchKernel( //calls device function merge path
 	int numMergeTiles, //[input] Number of spmv merge tiles which is the spmv grid size
 	Coord<IndexType_> *dTileCoords, //[output] pointer to a temporary array of tile starting coordinates
 	CsrMvParams<IndexType_, ValueType_> spParams) //[input] spmv input parameter with corrdponding needed arrays
{
	//set the constants for the gpu architecture
	enum
	{
		BLOCK_THREADS = SpmvBlockThread::BLOCK_THREADS,
		ITEMS_PER_THREAD = SpmvBlockThread::ITEMS_PER_THREAD,
		TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
	};
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid <= numMergeTiles) //verify within domain
	{
		IndexType_ diag = tid * TILE_ITEMS; 
		Coord<IndexType_> tileCoord; //each tid will compute its own tile_coordinate
		//the above coordinate will be stored in tile_coordinate passed by reference
		//input row pointer starting at csrRowPtr[1] merge path ignores the 0 entry
		//the first argument to the counting constructor is the size-nnz and the second argument is where to start countings
		
		IndexType_ countStart = 0; //if row pointer is 1 based make sure count starts at 1 instead of 0
		MergePathSearch(diag, spParams.csrRowPtr, countStart, spParams.m, spParams.nnz, tileCoord);
		//store path of thread in array of coordinates
		dTileCoords[tid] = tileCoord; //stores (y,x) = (i.j) coord of thread computed*
	}
}

//Agent sturct with two main inline functions which compute the spmv
template<
typename SpmvPolicyT, // parameterized SpmvBlockThread tuning policy type as listed above
typename IndexType_, //index value of rowOffsets and ColIndices
typename ValueType_, //matrix and vector value type
typename SemiRingType_, //this follows different semiring structs to be passed depending on the enum
bool hasAlpha, //signifies whether the input parameter alpha is 1 in y = alpha*A*x + beta*A*y
bool hasBeta> //signifies whether the input parameter beta is 0
struct AgentSpmv
{
	//set constants
	enum
	{
		BLOCK_THREADS = SpmvPolicyT::BLOCK_THREADS,
		ITEMS_PER_THREAD = SpmvPolicyT::ITEMS_PER_THREAD,
		TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
	};
//we use the return type pair for scanning where the pairs are accumulated segment-value with segemn-index
	__device__ __forceinline__ KeyValuePair<IndexType_,ValueType_> consumeTile(
		Coord<IndexType_> tileStartCoord, //this gives the starting coordinate to be determined from the initial mergepath call
		Coord<IndexType_> tileEndCoord,
		CsrMvParams<IndexType_, ValueType_> &spParams,
		SemiRingType_ SR) //pass struct as a const reference
	{	
		
		IndexType_ tileNumRows = tileEndCoord.x - tileStartCoord.x; //length(rowOffSets) = numRows + 1 in merge path ignore first element for 1 and so length of path in x-direction gives the exact number of rows
		IndexType_ tileNnz = tileEndCoord.y - tileStartCoord.y; //number of nonzero goes down path countingITerator is indexed by columnInd and Val array which are of size nnz
		//load row offsets into shared memory-create shared memory row offset pointer
		__shared__ IndexType_ smemTileRowPtr[ITEMS_PER_THREAD + TILE_ITEMS + 1];
		//copy row offsets into shared memory for accumulating matrix vector dot products in the merge path
		for (int item = threadIdx.x; item <= tileNumRows; item += BLOCK_THREADS) //index by block_threads that is the number of threads per block
			//start with rowoffsets at the strat coordinate and corresponding threadId can modiy wd to do a cache wrapper for efficiency later
		{
			if ((tileStartCoord.x + item) < spParams.m) //memory protection since already at +1 only go up to m
			{
				smemTileRowPtr[item] = spParams.csrRowPtr[tileStartCoord.x + item];
			}
		}

	//after loading into shared memory we must sync the threads to make sure all complete
		__syncthreads();
		Coord<IndexType_> threadStartCoord;
		//call MergePath again on shared memory after using start indices
		IndexType_ diag = threadIdx.x * ITEMS_PER_THREAD; //compute diagonal
		//shared memory row pointer has been indexed down to 0 so count offset can start at 0 too
		//counter iterator starts at current y position
		IndexType_ countIndId = tileStartCoord.y;
		MergePathSearch(diag, 
						smemTileRowPtr, //sort list A = row offsets in shared memort
						countIndId, //sort list B = natural number consecutive counting indices starting index
						tileNumRows,
						tileNnz,
						threadStartCoord); //resulting path is stored in threadStartCoord 
		__syncthreads(); //make sure every thread has completed their diagonal of merge path

		//Compute the thread's merge path segment to perform the dot product foing down the merge path below in the loop
		Coord<IndexType_> threadCurrentCoord = threadStartCoord;
		KeyValuePair<IndexType_, ValueType_> scanSegment[ITEMS_PER_THREAD]; //static array of type key value pairs
		//initialize each dot product contribution to 0
		ValueType_ totalValue;
		SR.setPlus_ident(totalValue);//initialize to semiring identity for plus operation
		#pragma unroll //unroll for loop for efficiency
		for (int item = 0; item < ITEMS_PER_THREAD; ++item) //loop over items belonging to thread along merge path
		{
			//go down merge path and sum. when move to right new component of result vector y
			//countInd is consecutive nonzero natural number array going down the matrix B so 
			//indexed by y whereas rowOffset goes to the move and is A indexed by x
			countIndId = threadCurrentCoord.y + tileStartCoord.y; //line number problem 
			
			IndexType_ nnzId = min(countIndId, spParams.nnz - 1); //make sure stay in bounds
			IndexType_ colIdx = spParams.csrColInd[nnzId];
			
			ValueType_ A_val = spParams.csrVal[nnzId]; //A val
			//we assume A and x are of the same datatype
			//recall standard algorithm : y[row] += val[nz]*x[colInd[nnz]] in traditional sparse matrix vector form
			ValueType_ x_val = spParams.x[colIdx]; //csrColInd[nnzId]
			//wrapper of x vector could change dependent on the architecture
			//counter will tell direction to move either right or down since last entry of rowoffsets is the totla number of nonzeros
			//the counter array keeps track of this
			if (countIndId < smemTileRowPtr[threadCurrentCoord.x]) //this means less than the number of nonzeros in that row
			{ //move down current row accumulating matrix and vector dot product
				totalValue = SR.plus(SR.times(A_val, x_val), totalValue); //add binary operation because may change to minus and min rather than + and *
				//store in key value pair
				scanSegment[item].key = tileNumRows;
				scanSegment[item].value = totalValue;
				++threadCurrentCoord.y;
			}
			else  //move right to new row and reset
			{//added in else if condition
				scanSegment[item].key = threadCurrentCoord.x;
				scanSegment[item].value = totalValue; //store current without adding new and set to 0 for new row
				SR.setPlus_ident(totalValue);//0.0;//SR.times_null;
				++threadCurrentCoord.x;
			}
		}
		__syncthreads(); //now each thread block has their matrix vector multiplication and we must do a blockwide reduction
		//Block-wide reduce-value-by-segment
		KeyValuePair<IndexType_, ValueType_> scanItem, tileCarry; //this is the key value pair that we will be returning

		scanItem.key = threadCurrentCoord.x; //added min in other version had min with num rows
		scanItem.value = totalValue;

		PrefixSum<IndexType_, ValueType_, SemiRingType_, BLOCK_THREADS>(SR).ExclusiveKeyValueScan(scanItem, tileCarry);
		if (tileNumRows > 0)
		{
			if (threadIdx.x == 0)
				scanItem.key = -1; //can be negative imp to be int rather than unsigned int
			//do a direct scatter
			#pragma unroll
			for (int item = 0; item <  ITEMS_PER_THREAD; ++item)
			{
				if (scanSegment[item].key < tileNumRows) //scanSegment is an array of key value pairs
				{
					if (scanItem.key == scanSegment[item].key)
					{	
						scanSegment[item].value = SR.plus(scanItem.value, scanSegment[item].value);
					}

					if (hasAlpha){
					//boolean set to 1 need to multiply Ax by alpha as stored in spParams
						scanSegment[item].value = SR.times(spParams.alpha, scanSegment[item].value);
					}

					//check if has beta then need to alter y the right hand side is multiplied by beta
					if (hasBeta)
					{ //y = alpha*A*x + beta*y
						ValueType_ y_val = spParams.y[tileStartCoord.x + scanSegment[item].key]; //currentxcoord is stored in the key and this will give corresponding and desired row entry in y 
						scanSegment[item].value = SR.plus(SR.times(spParams.beta, y_val), scanSegment[item].value);
					}

					//Set the output vector row element
					spParams.y[tileStartCoord.x + scanSegment[item].key] = scanSegment[item].value; //disjoint keys
				}
			}
		}
		//Return the til'es running carry-out key value pair
		return tileCarry; //will come from exclusive scan
	}

	//overload consumetile function for the one in the interafce which will be called by the dispatch function
	__device__ __forceinline__ void consumeTile (
		Coord<IndexType_> *dTileCoords, //pointer to the temporary array of tile starting cooordinates
		IndexType_ *dTileCarryKeys, //output pointer to temporary array carry-out dot product row-ids, one per block
		ValueType_ *dTileCarryValues, //output pointer to temporary array carry-out dot product row-ids, one per block
		int numMergeTiles, //number of merge tiles
		CsrMvParams<IndexType_, ValueType_> spParams,
		SemiRingType_ SR)
	{
		int tid = (blockIdx.x * gridDim.y) + blockIdx.y; //curent tile index
		//only continue if tid is in proper range
		if (tid >= numMergeTiles) 
			return;
		Coord<IndexType_> tileStartCoord = dTileCoords[tid]; //+0 ignored
		Coord<IndexType_> tileEndCoord = dTileCoords[tid + 1];

		//Consume multi-segment tile by calling above consumeTile overloaded function
		KeyValuePair<IndexType_, ValueType_> tileCarry = consumeTile(
			tileStartCoord,
			tileEndCoord,
			spParams,
			SR); 

		//output the tile's carry out
		if (threadIdx.x == 0)
		{
			if (hasAlpha)
				tileCarry.value = SR.times(spParams.alpha, tileCarry.value);

			tileCarry.key += tileStartCoord.x;
			
			if (tileCarry.key < spParams.m)
			{
				dTileCarryKeys[tid] = tileCarry.key;
				dTileCarryValues[tid] = tileCarry.value;
			}
			else
			{
				// Make sure to reject keys larger than the matrix size directly here.
				// printf("%d %lf\n",tileCarry.key , tileCarry.value);
				// this patch may be obsolete after the changes related to bug#1754610
				dTileCarryKeys[tid] = -1;
			}
		}
	}
};

//this device kernel will call the above agent function-ignoring policies for now
template <
	typename SpmvBlockThread, //parameterized spmvpolicy tunign policy type
	typename IndexType_, //index type either 32 bit or 64 bit integer for rowoffsets of columnindices
	typename ValueType_, //matrix and vector value type
	typename SemiRingType_, //this follows different semiring structs to be passed depending on the enum
	bool hasAlpha, //determines where alpha = 1 as above
	bool hasBeta> //determines whether beta = 0 as above
__global__ void DeviceSpmvKernel( //this will call consume tile
	CsrMvParams<IndexType_, ValueType_> spParams, //pass constant reference to spmv parameters
	const SemiRingType_ &SR,
	Coord<IndexType_> *dTileCoords, //input pointer to temporaray array of the tile starting coordinates of each (y,x) = (i,j) pair on the merge path
	IndexType_ *dTileCarryKeys, //output is a pointer to the temp array that carries out the dot porduct row-ids where it is one per block
	ValueType_ *dTileCarryValues, //output is a pointer to the temp array that carries out the dot porduct row-ids where it is one per block
	int numTiles //input which is the number of merge tiles
	)
{
	//call Spmv agent type specialization- need to fix this call!!
	//now call cosntructor to initialize and consumeTile to calculate the row dot products
	AgentSpmv<SpmvBlockThread, IndexType_, ValueType_, SemiRingType_, hasAlpha, hasBeta>().consumeTile(
		dTileCoords, 
		dTileCarryKeys, 
		dTileCarryValues, 
		numTiles, 
		spParams,
		SR);
}

//Helper functions for the reduction by kernel
//for block loading block_load_vectorize for SM_30 implemenation from cub
//Load linear segment into blocked arrangement across the thread block, guarded by range, 
//with a fall-back assignment of -1 for out of bound
template<int ITEMS_PER_THREAD, typename IndexType_, typename ValueType_>
__device__ __forceinline__ void loadDirectBlocked(
    int linearTid, //input:a  asuitable 1d thread-identifier for calling the thread
    IndexType_ *blockItrKeys, //input: thread block's base input iterator for loading from
    ValueType_ *blockItrValues, //input: thread block's base input iterator for loading from
    KeyValuePair<IndexType_, ValueType_> (&items)[ITEMS_PER_THREAD], // output:data to load
    int validItems, //input:Number of valid items to load
    KeyValuePair<IndexType_, ValueType_> outOfBoundsDefault) //input:Default value to assign to out of bounds items -1 in this case
{
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
        int offset = (linearTid * ITEMS_PER_THREAD) + item;
        // changed validItems to validItems-1 for bug#1754610 since it was causing uninitialized memory accesses here
        items[item].key = (offset < validItems-1) ? blockItrKeys[offset] : outOfBoundsDefault.key;
        items[item].value = (offset < validItems-1) ? blockItrValues[offset] : outOfBoundsDefault.value;
    }
}

//load linear segment of items into a blocked arangement across a thread block
template<int ITEMS_PER_THREAD, typename IndexType_, typename ValueType_>
__device__ __forceinline__ void loadDirectBlocked(
    int linearTid,
    IndexType_ * blockItrKeys,
    ValueType_ * blockItrValues,
    KeyValuePair<IndexType_,ValueType_> (&items)[ITEMS_PER_THREAD])
{
    //Load directly in thread-blocked order
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
        items[item].key = blockItrKeys[(linearTid *ITEMS_PER_THREAD) + item];
        items[item].value = blockItrValues[(linearTid *ITEMS_PER_THREAD) + item];
    }
}

//This part pertains to the fixup kernel which does a device-wide reduce-value-by-key 
//for the thread blocks
template<
typename SpmvPolicyT, // parameterized SpmvBlockThread tuning policy type as listed above
typename IndexType_,
typename ValueType_,
typename SemiRingType_> //matrix and vector value type
struct AgentSegmentReduction
{
	//set constants
	enum
	{
		BLOCK_THREADS = SpmvPolicyT::BLOCK_THREADS,
		ITEMS_PER_THREAD = SpmvPolicyT::ITEMS_PER_THREAD,
		TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
	};
	
	//This function processes an input tile and uses an atomic rewrite strategy
	template<bool isLastTile>
	__device__ __forceinline__ void consumeTilePost(
		IndexType_ *dInKeys, //input array of key value pairs
		ValueType_ *dInValues, //input array of key value pairs
		ValueType_ *dAggregatesOut, //output value aggregates into final array y
		IndexType_ numRemaining, //Number of global input items remaining including this tile
		IndexType_ tileOffset, //Tile offset
		SemiRingType_ SR
		)
	{
		KeyValuePair<IndexType_,ValueType_> pairs[ITEMS_PER_THREAD];
		KeyValuePair<IndexType_, ValueType_> outOfBoundsPair;
		outOfBoundsPair.key = -1; //default value to assign to out of bounds items is set to be -1
		int linearTid = threadIdx.x;
		//load the values into pairs
		if (isLastTile)
		{	
			loadDirectBlocked<ITEMS_PER_THREAD, IndexType_, ValueType_>
							(linearTid, 
							dInKeys + tileOffset,
							dInValues + tileOffset,
							pairs, 
							numRemaining, 
							outOfBoundsPair);
			
		}
		else
		{
			loadDirectBlocked<ITEMS_PER_THREAD, IndexType_, ValueType_>
							(linearTid, 
							dInKeys + tileOffset,
							dInValues + tileOffset,
							pairs);
		}

		#pragma unroll
		for (int item = 1; item < ITEMS_PER_THREAD; ++item)
		{
			ValueType_ *dScatter = dAggregatesOut + pairs[item-1].key; //write to correct row using the key
			if (pairs[item].key != pairs[item-1].key)
			{
				SR.atomicPlus(dScatter, pairs[item -1].value);
			}
			else
				pairs[item].value = SR.plus(pairs[item -1].value, pairs[item].value); //the operation is SUm
		}
		// Write out last item if it is valid by checking last key boolean.
		// pairs[ITEMS_PER_THREAD - 1].key = -1 for out bound elements.
		ValueType_ *dScatter = dAggregatesOut + pairs[ITEMS_PER_THREAD - 1].key;
		if ((!isLastTile || pairs[ITEMS_PER_THREAD - 1].key >= 0))
		{
			//printf("hello %d %lf\n", pairs[ITEMS_PER_THREAD - 1].key , pairs[ITEMS_PER_THREAD -1].value);
			SR.atomicPlus(dScatter, pairs[ITEMS_PER_THREAD -1].value);
		}
	}
	//this function will call consumeTilePost and it scans the tiles of items as a part of a dynamic chained scan
	__device__ __forceinline__ void consumeRange(
		IndexType_ *dKeysIn, //input array of key value pairs
		ValueType_ *dValuesIn, //input array of key value pairs
		ValueType_ *dAggregatesOut, //output value aggregates into final array y
		int numItems, //totall number of input items
		int numTiles, //total number of input tiles
		SemiRingType_ SR)
	{
		//Blocks are launched in increasing order, so we assign one tile per block
		int tileIdx = (blockIdx.x * gridDim.y) + blockIdx.y; //current tile index same as in consumeTile
		IndexType_ tileOffset = tileIdx * TILE_ITEMS; //Global offset for the current tile
		IndexType_ numRemaining = numItems - tileOffset; //Remaining items which includes this tile
		if (numRemaining > TILE_ITEMS) //this is not the last tile so call wit template argument set to be false
			consumeTilePost<false>(dKeysIn, dValuesIn, dAggregatesOut, numRemaining,tileOffset, SR);
		else if (numRemaining > 0) //this is the last tile which could be possibly partially full
			consumeTilePost<true>(dKeysIn, dValuesIn, dAggregatesOut, numRemaining,tileOffset, SR);		
	}
};

//Blockwide reduction by key final kernel
template <
typename SpmvBlockThreadSegment, //parameterized spmvpolicy tuning policy type
typename IndexType_,
typename ValueType_,
typename SemiRingType_>
__global__ void DeviceSegmentReductionByKeyKernel( //this will call consume tile
	IndexType_ *dKeysIn, //input pointer to the arry of dot product carried out by row-ids, one per spmv block
	ValueType_ *dValuesIn, //input pointer to the arry of dot product carried out by row-ids, one per spmv block
	ValueType_ *dAggregatesOut, //output value aggregates - will be y-final output of method
	IndexType_ numItems, // total number of items to select
	int numTiles, //total number of tiles for the entire problem
	SemiRingType_ SR)
{
	//now call cosntructor to initialize and consumeTile to calculate the row dot products
	AgentSegmentReduction<SpmvBlockThreadSegment, IndexType_, ValueType_, SemiRingType_>().consumeRange(
		dKeysIn, 
		dValuesIn,
		dAggregatesOut, 
		numItems,
		numTiles,
		SR);
}

template<typename IndexType_,
		 typename ValueType_,
		 typename SemiRingType_,
		bool hasAlpha,
		bool hasBeta> //matrix and vector value type
	//this is setting all the grid parameters and size
struct DispatchSpmv
{
	//declare constants
	enum
	{
		INIT_KERNEL_THREADS = 128
	};
	//sample tuning polic- can add more later
	//SM30
	struct Policy350 //as a sample there are many other policies to follow
	{
		typedef SpmvBlockThread< (sizeof(ValueType_) > 4) ? 96 : 128, //for double use 96 threads per block otherwise 128
								 (sizeof(ValueType_) > 4) ? 4 : 4 //for double use 4 items per thread otherwise use 7
								> SpmvPolicyT;///use instead of PtxPolicy come backa nd use cusparse to determine the architetcure 
	};

	struct Policy350Reduction //as a sample there are many other policies to follow
	{
		typedef SpmvBlockThread<128,3> SpmvPolicyT; //use instead of PtxPolicy come backa nd use cusparse to determine the architetcure 
	};//for <128,1> 1 item per thread need a reduction by key

	__forceinline__ static cudaError_t Dispatch(CsrMvParams<IndexType_,ValueType_> spParams, const SemiRingType_ &SR, cudaStream_t stream = 0)
	{
		cudaError_t error = cudaSuccess;	
		//could move this block to initkernel fucntion 
		int blockThreads = Policy350::SpmvPolicyT::BLOCK_THREADS;
		int itemsPerThread = Policy350::SpmvPolicyT::ITEMS_PER_THREAD;

		int blockThreadsRed = Policy350Reduction::SpmvPolicyT::BLOCK_THREADS;
		int itemsPerThreadRed = Policy350Reduction::SpmvPolicyT::ITEMS_PER_THREAD;
		//calculate total number of  spmv work items
		do { //do-while loop condition at end of loop
			//Get device ordinal
			int deviceOrdinal, smVersion, smCount, maxDimx;
			if (error = cudaGetDevice(&deviceOrdinal)) break;

			//Get device SM version
			if (error = SmVersion(smVersion, deviceOrdinal)) break;

			//Get SM count-cudaDeviceGetAttribute is built in cuda function
			if (error = cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceOrdinal)) break;

			//Get max dimension of the grid in the x direction
			if (error = cudaDeviceGetAttribute(&maxDimx, cudaDevAttrMaxGridDimX, deviceOrdinal)) break;

			int numMergeItems = spParams.m + spParams.nnz; //total amount of work for one diagonal/thread
			
			//Tile sizes of relevant kernels
			int mergeTileSize = blockThreads * itemsPerThread; //for floats this will be a larger number
			//and since we will be dividing by it less memory allocated for the float case
			int segmentRedTileSize = blockThreadsRed * itemsPerThreadRed;

			//Calculate number of tiles for the kernels
			//need unsigned int to prevent underflow/overflow
			unsigned int numMergeTiles = (numMergeItems + mergeTileSize - 1) / mergeTileSize; //launch thread number
			unsigned int numSegmentRedTiles = (numMergeTiles + segmentRedTileSize - 1) / segmentRedTileSize;
			//int spmv_sm_occupancy ignore maxSmOccupancy function for now and corresponding segmentfixup
			//get grid dimensions use cuda built in dattetype dim3-has constructor with the 3 arguments
			
			dim3 spmvGridSize(min(numMergeTiles, (unsigned int) maxDimx),
							  (numMergeTiles + maxDimx - 1) / maxDimx, //make sure at least 1
							  1); //2D grid
			//grid for second kernel
			dim3 segmentRedGridSize(min(numSegmentRedTiles, (unsigned int) maxDimx),
									(numSegmentRedTiles + maxDimx -1) / maxDimx,
									1);
			Vector<Coord<IndexType_> > dTileCoords(numMergeTiles + 1, stream);
			Vector<IndexType_> dTileCarryKeys(numMergeTiles, stream);
			Vector<ValueType_> dTileCarryValues(numMergeTiles, stream);
			
			//Get search grid dimensions
			int searchBlockSize = INIT_KERNEL_THREADS;
			int searchGridSize = (numMergeTiles + searchBlockSize) / searchBlockSize; //ignored the +1 -1
			//call Search Kernel within the host so need <<>>>
			//call devicesearch kernel to compute starting coordiantes of merge path                
			DeviceSpmvSearchKernel<typename Policy350::SpmvPolicyT, IndexType_, ValueType_> 
				<<<searchGridSize, searchBlockSize, 0, stream >>>(
				numMergeTiles, 
				dTileCoords.raw(),
				spParams);   
			cudaCheckError();             
			//this will give the starting coordaintes to be called in DeviceSPmvKernel
			
			DeviceSpmvKernel<typename Policy350::SpmvPolicyT, IndexType_,ValueType_, SemiRingType_, hasAlpha, hasBeta>  
				<<<spmvGridSize, blockThreads, 0, stream>>>(
				spParams,
				SR,
				dTileCoords.raw(), 
				dTileCarryKeys.raw(), 
				dTileCarryValues.raw(),
				numMergeTiles);                
			cudaCheckError();
			//Run reduce by key kernel if necessary
			//if (error = cudaPeekAtLastError()) break; //check for failure to launch
			if (numMergeTiles > 1)
			{
				DeviceSegmentReductionByKeyKernel<typename Policy350Reduction::SpmvPolicyT, IndexType_, ValueType_, SemiRingType_> 
												<<<segmentRedGridSize, blockThreadsRed, 0>>>
											  (dTileCarryKeys.raw(),
											  dTileCarryValues.raw(),
											   spParams.y, 
											   numMergeTiles,
											   numSegmentRedTiles,
											   SR);
				cudaCheckError();
				//if (error = cudaPeekAtLastError()) break; //check for failure to launch of fixup kernel
			}
		} while(0); //make sure executes exactly once to give chance to break earlier with errors
		cudaCheckError();              
		
		return error;
	}
};

template<typename IndexType_, typename ValueType_, typename SemiRingType_>
cudaError_t callDispatchSpmv(CsrMvParams<IndexType_, ValueType_> &spParams, const SemiRingType_ &SR, cudaStream_t stream = 0)
{
	cudaError_t error;
	//determine semiring type
	if (spParams.beta == SR.times_null)
	{
		if (spParams.alpha == SR.times_ident) //simply y = A*x
			error =  DispatchSpmv<IndexType_, ValueType_, SemiRingType_, false, false>::Dispatch(spParams, SR, stream); //must be on the device
		
		else
			error =  DispatchSpmv<IndexType_, ValueType_,SemiRingType_, true, false>::Dispatch(spParams, SR, stream); //must be passed by reference to some since writing 
	}
	else
	{
		if (spParams.alpha == SR.times_ident)
			error =  DispatchSpmv<IndexType_, ValueType_, SemiRingType_, false, true>::Dispatch(spParams, SR, stream);
		else
			error =  DispatchSpmv<IndexType_, ValueType_, SemiRingType_, true, true>::Dispatch(spParams, SR, stream);
	}
	return error;
}

template<typename IndexType_, typename ValueType_>
cudaError_t callSemiringSpmv(CsrMvParams<IndexType_, ValueType_> &spParams, Semiring SR, cudaStream_t stream = 0)
{
    // This is dangerous but we need to initialize this value, probably it's
    // better to return success than to return some misleading error code
	cudaError_t error = cudaSuccess;
	switch(SR)
	{
		case PlusTimes:
		{
			PlusTimesSemiring<ValueType_> plustimes; //can be float or double for real case
			error =  callDispatchSpmv(spParams, plustimes, stream);
		}
		break;	
		case MinPlus:
		{
			MinPlusSemiring<ValueType_> minplus;
			error =  callDispatchSpmv(spParams, minplus, stream);
		}
		break;
		case MaxMin:
		{
			MaxMinSemiring<ValueType_> maxmin;
			error =  callDispatchSpmv(spParams, maxmin, stream);
		}
		break;
		case OrAndBool:
		{
			OrAndBoolSemiring<ValueType_> orandbool;
			error =  callDispatchSpmv(spParams, orandbool, stream);
		}
		break;
		case LogPlus:
		{
			LogPlusSemiring<ValueType_> logplus;
			error =  callDispatchSpmv(spParams, logplus, stream);
		}
		break;
	}
	return error;
}

//create a device function interface to call the above dispatch function
template <typename IndexType_, typename ValueType_>
cudaError_t csrmv_mp(
	IndexType_ n,
	IndexType_ m, 
	IndexType_ nnz,
	ValueType_ alpha,
	ValueType_ * dValues, //all must be preallocated on the device
	IndexType_ * dRowOffsets,
	IndexType_ * dColIndices,
	ValueType_ *dVectorX,
	ValueType_ beta,
	ValueType_ *dVectorY,
	Semiring SR, 
	cudaStream_t stream)
{ //create user interface
	 //calling device kernel depends on tempalte boolean parameters fro alpha/beta
	//Set parameters for struct
	CsrMvParams<IndexType_, ValueType_> spParams;
	spParams.m = m;
	spParams.n = n;
	spParams.nnz = nnz;
	spParams.alpha = alpha;
	spParams.beta = beta;
	spParams.csrRowPtr = dRowOffsets + 1; //ignore first 0 component in merge path specific for this spmv only
	spParams.csrVal = dValues;
	spParams.csrColInd = dColIndices;
	spParams.x = dVectorX;
	spParams.y = dVectorY;

	return callSemiringSpmv(spParams, SR, stream);
}


template<typename IndexType_, typename ValueType_>
cudaError_t csrmv_mp(
	IndexType_ n,
	IndexType_ m,
	IndexType_ nnz,
	ValueType_ alpha,
	ValuedCsrGraph <IndexType_, ValueType_> network,
	ValueType_ *dVectorX,
	ValueType_ beta,
	ValueType_ *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	)
{
	 //calling device kernel depends on tempalte boolean parameters fro alpha/beta
	//Set parameters for struct

	CsrMvParams<IndexType_, ValueType_> spParams;
	spParams.m = m;
	spParams.n = n;
	spParams.nnz = nnz;
	spParams.alpha = alpha;
	spParams.beta = beta;
	spParams.csrRowPtr = network.get_raw_row_offsets() + 1; //ignore first 0 component in merge path specific for this spmv only
	spParams.csrVal = network.get_raw_values();
	spParams.csrColInd = network.get_raw_column_indices();
	spParams.x = dVectorX;
	spParams.y = dVectorY;

	return callSemiringSpmv(spParams, SR, stream);
}

//declare template types to be called
template cudaError_t csrmv_mp<int, double>(
	int n,
	int m, 
	int nnz,
	double alpha,
	double * dValues, //all must be preallocated on the device
	int * dRowOffsets,
	int * dColIndices,
	double *dVectorX,
	double beta,
	double *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);

template cudaError_t csrmv_mp<long long, double>(
	long long n,
	long long m, 
	long long nnz,
	double alpha,
	double * dValues, //all must be preallocated on the device
	long long * dRowOffsets,
	long long * dColIndices,
	double *dVectorX,
	double beta,
	double *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);

template cudaError_t csrmv_mp<int, float>(
	int n,
	int m, 
	int nnz,
	float alpha,
	float * dValues, //all must be preallocated on the device
	int * dRowOffsets,
	int * dColIndices,
	float *dVectorX,
	float beta,
	float *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);
//for 64 bit support which may not be needed
template cudaError_t csrmv_mp<long long, float>(
	long long n,
	long long m, 
	long long nnz,
	float alpha,
	float * dValues, //all must be preallocated on the device
	long long * dRowOffsets,
	long long * dColIndices,
	float *dVectorX,
	float beta,
	float *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);
//assume embedding booleans in the reals
/*template cudaError_t csrmv_mp<int, bool>(
	int n,
	int m, 
	int nnz,
	bool alpha,
	bool * dValues, //all must be preallocated on the device
	int * dRowOffsets,
	int * dColIndices,
	bool *dVectorX,
	bool beta,
	bool *dVectorY,
	Semiring SR
	);
//for 64 bit support which may not be needed
template cudaError_t csrmv_mp<long long, bool>(
	long long n,
	long long m, 
	long long nnz,
	bool alpha,
	bool * dValues, //all must be preallocated on the device
	long long * dRowOffsets,
	long long * dColIndices,
	bool *dVectorX,
	bool beta,
	bool *dVectorY,
	Semiring SR
	);*/

//declare template types to be called using valued_csr_graph version
template cudaError_t csrmv_mp<int, double>(
	int n,
	int m, 
	int nnz,
	double alpha,
	ValuedCsrGraph <int, double> network,
	double *dVectorX,
	double beta,
	double *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);

template cudaError_t csrmv_mp<long long, double>(
	long long n,
	long long m, 
	long long nnz,
	double alpha,
	ValuedCsrGraph <long long, double> network,
	double *dVectorX,
	double beta,
	double *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);

template cudaError_t csrmv_mp<int, float>(
	int n,
	int m, 
	int nnz,
	float alpha,
	ValuedCsrGraph <int, float> network,
	float *dVectorX,
	float beta,
	float *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);
//for 64 bit support which may not be needed
template cudaError_t csrmv_mp<long long, float>(
	long long n,
	long long m, 
	long long nnz,
	float alpha,
	ValuedCsrGraph <long long, float> network,
	float *dVectorX,
	float beta,
	float *dVectorY,
	Semiring SR, 
	cudaStream_t stream
	);

/*template cudaError_t csrmv_mp<int, bool>(
	int n,
	int m, 
	int nnz,
	bool alpha,
	ValuedCsrGraph <int, bool> network,
	bool *dVectorX,
	bool beta,
	bool *dVectorY,
	Semiring SR
	);
//for 64 bit support which may not be needed
template cudaError_t csrmv_mp<long long, bool>(
	long long n,
	long long m, 
	long long nnz,
	bool alpha,
	ValuedCsrGraph <long long, bool> network,
	bool *dVectorX,
	bool beta,
	bool *dVectorY,
	Semiring SR
	);*/

} //end namespace nvgraph

using namespace nvgraph;

//this is the standard kernel used to test the semiring operations	
template<typename IndexType_, typename ValueType_, typename SemiRingType_>
 __global__ void csrmv(IndexType_ num_rows, IndexType_ *dRowOffsets, IndexType_ *dColIndices, ValueType_ *dValues, 
 	ValueType_ *dVectorX, ValueType_ *dVectorY, SemiRingType_ SR, ValueType_ alpha, ValueType_ beta) 
{
	int row = blockDim.x * blockIdx.x + threadIdx.x ;
	if (row < num_rows) 
	{
		ValueType_ dot;
		SR.setPlus_ident(dot);
		//SR.setPlus_ident(dVectorY[row]); //need to initialize y outside
		IndexType_ row_start = dRowOffsets[row];
		IndexType_ row_end = dRowOffsets[row + 1];
		for (int i = row_start; i < row_end; i++) 
		{
			dot = SR.plus(SR.times(alpha,SR.times(dValues[i], dVectorX[dColIndices[i]])), dot);
		}
		dVectorY[row] = SR.plus(dot, (SR.times(beta, dVectorY[row])));
	}
}	

template<typename IndexType_, typename ValueType_>
void callTestCsrmv(IndexType_ num_rows, IndexType_ *dRowOffsets, IndexType_ *dColIndices, ValueType_ *dValues, 
 	ValueType_ *dVectorX, ValueType_ *dVectorY, nvgraph::Semiring SR, ValueType_ alpha, ValueType_ beta)
{
	const int side = 2048;
	const int numThreads = 256;
	const int numBlocks = (side * side + numThreads - 1) / numThreads;
	switch(SR)
	{
		case nvgraph::PlusTimes:
		{
			nvgraph::PlusTimesSemiring<ValueType_> plustimes; //can be float or double for real case
			csrmv<<<numBlocks, numThreads>>>(num_rows, dRowOffsets, dColIndices, dValues, dVectorX, dVectorY, plustimes, alpha, beta);
		}
		break;	
		case nvgraph::MinPlus:
		{
			nvgraph::MinPlusSemiring<ValueType_> minplus;
			csrmv<<<numBlocks, numThreads>>>(num_rows, dRowOffsets, dColIndices, dValues, dVectorX, dVectorY, minplus, alpha, beta);
		}
		break;
		case nvgraph::MaxMin:
		{
			nvgraph::MaxMinSemiring<ValueType_> maxmin;
			csrmv<<<numBlocks, numThreads>>>(num_rows, dRowOffsets, dColIndices, dValues, dVectorX, dVectorY, maxmin, alpha, beta);
		}
		break;
		case nvgraph::OrAndBool:
		{
			nvgraph::OrAndBoolSemiring<ValueType_> orandbool;
			csrmv<<<numBlocks, numThreads>>>(num_rows, dRowOffsets, dColIndices, dValues, dVectorX, dVectorY, orandbool, alpha, beta);
		}
		break;
		case nvgraph::LogPlus:
		{
			nvgraph::LogPlusSemiring<ValueType_> logplus;
			csrmv<<<numBlocks, numThreads>>>(num_rows, dRowOffsets, dColIndices, dValues, dVectorX, dVectorY, logplus, alpha, beta);
		}
		break;
	}
	cudaCheckError();

}

template void callTestCsrmv<int, float>(int num_rows, int *dRowOffsets, int*dColIndices, float *dValues, 
 	float *dVectorX, float *dVectorY, nvgraph::Semiring SR, float alpha, float beta);

template void callTestCsrmv<int, double>(int num_rows, int *dRowOffsets, int*dColIndices, double *dValues, 
 double *dVectorX, double *dVectorY, nvgraph::Semiring SR, double alpha, double beta);	

