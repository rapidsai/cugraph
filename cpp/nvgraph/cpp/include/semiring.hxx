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
#include <cfloat>
#include <algorithm>
#include <stdio.h>
#include "atomics.hxx"
#include "nvgraph_error.hxx"

namespace nvgraph{
//define nvgraph min and max oprators
template<typename T>
__host__ __device__ __forceinline__ T min(const T&a, const T &b)
{
	return (a < b) ? a : b;
}

template<typename T>
__host__ __device__ __forceinline__ T max(const T&a, const T &b)
{
	return (a > b) ? a : b;
}

//have routines to return these operators
template<typename ValueType_> //ValueType_ is Value_type of the graph
struct PlusTimesSemiring
{
	typedef ValueType_ SR_type;
	SR_type plus_ident, times_ident, times_null;
	PlusTimesSemiring()
	{
		if (typeid(ValueType_) != typeid(float) && typeid(ValueType_) != typeid(double))
			FatalError("Graph value type is not supported by this semiring.", NVGRAPH_ERR_BAD_PARAMETERS);

		//for semiring need multiplicative and additive identity
		plus_ident = SR_type(0);
		times_ident = SR_type(1);
		//also need multiplicative null
		times_null = SR_type(0);
	}
	__host__ __device__ __forceinline__ void setPlus_ident(SR_type &val) 
	{
		val = SR_type(0);
	}

	__host__ __device__ __forceinline__ SR_type plus(const SR_type &arg0, const SR_type &arg1)
	{
		return arg0 + arg1;
	}
	__host__ __device__ __forceinline__ SR_type times(const SR_type &arg0, const SR_type &arg1)
	{
		return arg0 * arg1;
	}
	//potential private member to be used in reduction by key so only need atomic for plus operator
	__device__ __forceinline__ void atomicPlus(SR_type *addr, SR_type val)
	{
		atomicFPAdd(addr, val);
	}
	__device__ __forceinline__ SR_type shflPlus(SR_type input, int firstLane, int offset)
	{
		return shflFPAdd(input, firstLane, offset);
	}
};

template<typename ValueType_>
struct MinPlusSemiring
{
	typedef ValueType_ SR_type; //possibly change for integers to cast to floats
	SR_type plus_ident, times_ident, times_null;
	MinPlusSemiring()
	{
		if (typeid(ValueType_) != typeid(float) && typeid(ValueType_) != typeid(double))
			FatalError("Graph value type is not supported by this semiring.", NVGRAPH_ERR_BAD_PARAMETERS);

		//for semiring need multiplicative and additive identity//put in constructor
		SR_type inf = (typeid(ValueType_) == typeid(float)) ? FLT_MAX : DBL_MAX; //check for cuda add type identifiers
		plus_ident = SR_type(inf);
		times_ident = SR_type(0);
		//also need multiplicative null
		times_null = SR_type(inf);
	}
	__host__ __device__ __forceinline__ void setPlus_ident(float &val) 
	{
		val = FLT_MAX;
	}

	__host__ __device__ __forceinline__ void setPlus_ident(double &val) 
	{
		val = DBL_MAX;
	}

	__host__ __device__ __forceinline__ SR_type plus(const SR_type &arg0, const SR_type &arg1)
	{
		return min(arg0, arg1); //check and change!-using min in csrmv.cu
	}
	__host__ __device__ __forceinline__ SR_type times(const SR_type &arg0, const SR_type &arg1)
	{
		return arg0 + arg1;
	}
	//potential private member to be used in reduction by key so only need atomic for plus operator
	__device__ __forceinline__ void atomicPlus(SR_type *addr, SR_type val)
	{
		atomicFPMin(addr, val);
	}
	__device__ __forceinline__ SR_type shflPlus(SR_type input, int firstLane, int offset)
	{
		return shflFPMin(input, firstLane, offset);
	}
};

template<typename ValueType_>
struct MaxMinSemiring //bottleneck semiring
{
	typedef ValueType_ SR_type;//could be integers template and check that type makes sense
	SR_type plus_ident, times_ident, times_null;
	MaxMinSemiring()
	{
		if (typeid(ValueType_) != typeid(float) && typeid(ValueType_) != typeid(double))
			FatalError("Graph value type is not supported by this semiring.", NVGRAPH_ERR_BAD_PARAMETERS);

		//for semiring need multiplicative and additive identity
		SR_type inf = (typeid(ValueType_) == typeid(float)) ? FLT_MAX : DBL_MAX;
		plus_ident = SR_type(-inf);
		times_ident = SR_type(inf);
		//also need multiplicative null
		times_null = SR_type(-inf);
	}
	__host__ __device__ __forceinline__ void setPlus_ident(float &val) 
	{
		val = -FLT_MAX;
	}

	__host__ __device__ __forceinline__ void setPlus_ident(double &val) 
	{
		val = -DBL_MAX;
	}

	__host__ __device__ __forceinline__ SR_type plus(const SR_type &arg0, const SR_type &arg1)
	{
		return max(arg0, arg1); //check and change!-using min in csrmv.cu can use thrust
	}
	__host__ __device__ __forceinline__ SR_type times(const SR_type &arg0, const SR_type &arg1)
	{
		return min(arg0,arg1);
	}
	//potential private member to be used in reduction by key so only need atomic for plus operator
	__device__ __forceinline__ void atomicPlus(SR_type *addr, SR_type val)
	{
		atomicFPMax(addr, val);
	}
	__device__ __forceinline__ SR_type shflPlus(SR_type input, int firstLane, int offset)
	{
		return shflFPMax(input, firstLane, offset);
	}
};

template<typename ValueType_>
struct OrAndBoolSemiring //bottleneck semiring
{
	typedef ValueType_ SR_type;//could be integers
	SR_type plus_ident, times_ident, times_null;
	OrAndBoolSemiring()
	{
		//embed the bools in the reals just use 0 and 1 in floats
		if (typeid(ValueType_) != typeid(float) && typeid(ValueType_) != typeid(double))
			FatalError("Graph value type is not supported by this semiring.", NVGRAPH_ERR_BAD_PARAMETERS);

		//for semiring need multiplicative and additive identity
		plus_ident = SR_type(0);
		times_ident = SR_type(1);
		//also need multiplicative null
		times_null = SR_type(0);
	}
	__host__ __device__ __forceinline__ void setPlus_ident(SR_type &val) 
	{
		val = SR_type(0);
	}

	__host__ __device__ __forceinline__ SR_type plus(const SR_type &arg0, const SR_type &arg1)
	{
		return (bool) arg0 | (bool) arg1; //check and change!-using min in csrmv.cu can use thrust
	}
	__host__ __device__ __forceinline__ SR_type times(const SR_type &arg0, const SR_type &arg1)
	{
		return (bool) arg0 & (bool) arg1;
	}
	//potential private member to be used in reduction by key so only need atomic for plus operator
	//need to check this atomic since it takes integer parameters instead of boolean
	__device__ __forceinline__ void atomicPlus(SR_type *addr, SR_type val)
	{
		atomicFPOr(addr, val);
	}
	//DOESN"T work returns exclusive or
	__device__ __forceinline__ SR_type shflPlus(SR_type input, int firstLane, int offset)
	{
		return shflFPOr(input, firstLane, offset);
	}
};
//This Semiring does not work. WIll not be supported in first version
template<typename ValueType_>
struct LogPlusSemiring //bottleneck semiring
{
	typedef ValueType_ SR_type;//could be integers
	SR_type plus_ident, times_ident, times_null;
	LogPlusSemiring()
	{
		//for semiring need multiplicative and additive identity
		if (typeid(ValueType_) != typeid(float) && typeid(ValueType_) != typeid(double))
			FatalError("Graph value type is not supported by this semiring.", NVGRAPH_ERR_BAD_PARAMETERS);
		
		SR_type inf = (typeid(ValueType_) == typeid(float)) ? FLT_MAX : DBL_MAX;
		plus_ident = SR_type(inf);
		times_ident = SR_type(0);
		//also need multiplicative null
		times_null = SR_type(inf);
	}

	__host__ __device__ __forceinline__ void setPlus_ident(float &val) 
	{
		val = FLT_MAX;
	}

	__host__ __device__ __forceinline__ void setPlus_ident(double &val) 
	{
		val = DBL_MAX;
	}

	__host__ __device__ __forceinline__ SR_type plus(const SR_type &arg0, const SR_type &arg1)
	{
		return -log(exp(-arg0) + exp(-arg1)); //check calling cuda log and arg0 ok for float not double?
	}
	__host__ __device__ __forceinline__ SR_type times(const SR_type &arg0, const SR_type &arg1)
	{
		return arg0 + arg1;
	}
	//this will not work!
	__device__ __forceinline__ void atomicPlus(SR_type *addr, SR_type val)
	{
		atomicFPLog(addr, val);
	}
	//this DOES NOT work! Need customized shfl isntructions for logPlus
	__device__ __forceinline__ SR_type shflPlus(SR_type input, int firstLane, int offset)
	{
		return shflFPAdd(input, firstLane, offset);
	}
};

}// end namespace nvgraph

