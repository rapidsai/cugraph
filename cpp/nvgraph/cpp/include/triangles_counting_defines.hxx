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

#include <cuda_runtime.h>
#include <limits.h>

#ifdef _MSC_VER
#include <stdint.h>
#else
#include <inttypes.h>
#endif


/*
#ifdef MSVC_VER
#include <intrin.h> 
#pragma intrinsic(_BitScanForward) 
#pragma intrinsic(_BitScanForward64) 
#pragma intrinsic(_BitScanReverse) 
#pragma intrinsic(_BitScanReverse64) 
#endif
*/

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

#define THREADS        (128)
#define DIV_UP(a,b)    (((a)+((b)-1))/(b))
#define BITSOF(x)    (sizeof(*x)*8)

#define BLK_BWL0 (128)
#define WRP_BWL0 (128)

#define HUGE_GRAPH

#define DEG_THR1  (3.5) 
#define DEG_THR2 (38.0) 

namespace nvgraph
{

namespace triangles_counting
{

template <typename T> struct type_utils;

template <>
struct type_utils<int>
{
    typedef int  LOCINT;
    static const LOCINT LOCINT_MAX = INT_MAX;
#ifdef MPI_VERSION
    static const MPI_Datatype LOCINT_MPI = MPI_INT;
#endif
    static __inline__ LOCINT abs(const LOCINT& x)
    {
        return abs(x);
    }
};

template <>
struct type_utils<int64_t>
{
    typedef uint64_t  LOCINT;
    static const LOCINT LOCINT_MAX = LLONG_MAX;
#ifdef MPI_VERSION
    static const MPI_Datatype LOCINT_MPI = MPI_LONG_LONG;
#endif

    static __inline__ LOCINT abs(const LOCINT& x)
    {
        return llabs(x);
    }
};


template <typename T>
struct spmat_t {
    T    N;
    T    nnz;
    T    nrows;
    const T    *roff_d;
    const T    *rows_d;
    const T    *cols_d;
    bool is_lower_triangular;
};

} // namespace triangles_counting

} // namespace nvgraph
