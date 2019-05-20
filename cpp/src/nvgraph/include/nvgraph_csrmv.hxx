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
#include <algorithm>
#include <stdio.h>
#include "valued_csr_graph.hxx"
#include "nvgraph_vector.hxx"

namespace nvgraph{

//this header file defines the various semirings using enum
 enum Semiring
 {//the datatype is assumed to be real unless otherwise specified in the name
 	PlusTimes, //standard matrix vector multiplication
 	MinPlus, //breadth first search-also called tropical
 	MaxMin, //mas flow problems
 	OrAndBool,
 	LogPlus
 };	

//Merge Path Coord array depends on the integere type
template<typename IndexType_>
struct Coord 
{
    IndexType_ x;
    IndexType_ y;
};

//struct which stores the csr matrix format, templated on the index and value
 template <typename IndexType_, typename ValueType_>
 struct CsrMvParams {
 	ValueType_ alpha;
 	ValueType_ beta;
 	ValueType_ *csrVal; //nonzero values from matrix A
 	//row pointer must look at next address to avoid the 0 in merge path
 	IndexType_ *csrRowPtr; //row offsets last entry is number of nonzeros size is m +1
 	IndexType_ *csrColInd; //column indices of nonzeros
 	ValueType_ *x; //vector x in alpha*A*x
 	ValueType_ *y; //output y will be modified and store the output
 	IndexType_ m; //number of rows
 	IndexType_ n; //number of columns
	IndexType_ nnz; 
 };

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
	Semiring SR,  //this parameter is of type enum and gives the semiring name
	cudaStream_t stream = 0 );
//overloaded function that has valued_csr_graph parameter to store the matrix
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
	Semiring SR, //this parameter is of type enum and gives the semiring name
	cudaStream_t stream = 0);	
} //end nvgraph namespace

template<typename IndexType_, typename ValueType_>
void callTestCsrmv(IndexType_ num_rows, IndexType_ *dRowOffsets, IndexType_ *dColIndices, ValueType_ *dValues, 
 	ValueType_ *dVectorX, ValueType_ *dVectorY, nvgraph::Semiring SR, ValueType_ alpha, ValueType_ beta);

