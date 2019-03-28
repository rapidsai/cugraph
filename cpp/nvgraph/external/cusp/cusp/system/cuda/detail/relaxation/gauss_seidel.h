/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/detail/config.h>

#include <cusp/system/cuda/arch.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/device_ptr.h>

#include <algorithm>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(VECTORS_PER_BLOCK * THREADS_PER_VECTOR,1)
__global__ void
gauss_seidel_kernel(const IndexType num_rows,
                    const IndexType * Ap,
                    const IndexType * Aj,
                    const ValueType * Ax,
                    ValueType * x,
                    const ValueType * b,
                    const IndexType * indices)
{
    __shared__ volatile ValueType sdiags[VECTORS_PER_BLOCK];
    __shared__ volatile ValueType sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];

    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType index = vector_id; index < num_rows; index += num_vectors)
    {
        IndexType row = indices[index];

        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        ValueType sum = 0;

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
            // ensure aligned memory access to Aj and Ax
            IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
            {
                IndexType col = Aj[jj];
                bool diag = row == col;
                sum += diag ? 0 : Ax[jj] * x[col];
                if(diag) sdiags[vector_lane] = Ax[jj];
            }

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
            {
                IndexType col = Aj[jj];
                bool diag = row == col;
                sum += diag ? 0 : Ax[jj] * x[col];
                if(diag) sdiags[vector_lane] = Ax[jj];
            }
        }
        else
        {
            // accumulate local sums
            for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
            {
                IndexType col = Aj[jj];
                bool diag = row == col;
                sum += diag ? 0 : Ax[jj] * x[col];
                if(diag) sdiags[vector_lane] = Ax[jj];
            }
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];

        // first thread writes the result
        if (thread_lane == 0)
            x[row] = (b[row] - sdata[threadIdx.x]) / sdiags[vector_lane];
    }
}

template <unsigned int THREADS_PER_VECTOR,
         typename DerivedPolicy,
         typename MatrixType,
         typename ArrayType1,
         typename ArrayType2>
void gauss_seidel_spmv(cuda::execution_policy<DerivedPolicy>& exec,
                       const MatrixType& A,
                       ArrayType1&  x,
                       const ArrayType1&  b,
                       const ArrayType2& indices,
                       const int row_start,
                       const int row_stop,
                       const int row_step)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    const size_t num_rows = row_stop - row_start;
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(gauss_seidel_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(num_rows, VECTORS_PER_BLOCK));

    const IndexType * R = thrust::raw_pointer_cast(&A.row_offsets[0]);
    const IndexType * J = thrust::raw_pointer_cast(&A.column_indices[0]);
    const ValueType * V = thrust::raw_pointer_cast(&A.values[0]);
    ValueType * x_ptr = thrust::raw_pointer_cast(&x[0]);
    const ValueType * b_ptr = thrust::raw_pointer_cast(&b[0]);
    const IndexType * i_ptr = thrust::raw_pointer_cast(&indices[row_start]);

    gauss_seidel_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
    (num_rows, R, J, V, x_ptr, b_ptr, i_ptr);
}

template<typename DerivedPolicy,
         typename MatrixType,
         typename ArrayType1,
         typename ArrayType2>
void gauss_seidel_indexed(cuda::execution_policy<DerivedPolicy>& exec,
                          const MatrixType& A,
                                ArrayType1& x,
                          const ArrayType1& b,
                          const ArrayType2& indices,
                          const int row_start,
                          const int row_stop,
                          const int row_step)
{
    typedef typename MatrixType::index_type IndexType;

    const IndexType nnz_per_row = A.num_entries / A.num_rows;

    if (nnz_per_row <=  2) {
        gauss_seidel_spmv<2>(exec, A, x, b, indices, row_start, row_stop, row_step);
        return;
    }
    if (nnz_per_row <=  4) {
        gauss_seidel_spmv<4>(exec, A, x, b, indices, row_start, row_stop, row_step);
        return;
    }
    if (nnz_per_row <=  8) {
        gauss_seidel_spmv<8>(exec, A, x, b, indices, row_start, row_stop, row_step);
        return;
    }
    if (nnz_per_row <= 16) {
        gauss_seidel_spmv<16>(exec, A, x, b, indices, row_start, row_stop, row_step);
        return;
    }

    gauss_seidel_spmv<32>(exec, A, x, b, indices, row_start, row_stop, row_step);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

