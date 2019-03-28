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

#include <thrust/extrema.h>

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

////////////////////////////////////////////////////////////////////////
// DIA SpMV kernels
///////////////////////////////////////////////////////////////////////
//
// Diagonal matrices arise in grid-based discretizations using stencils.
// For instance, the standard 5-point discretization of the two-dimensional
// Laplacian operator has the stencil:
//      [  0  -1   0 ]
//      [ -1   4  -1 ]
//      [  0  -1   0 ]
// and the resulting DIA format has 5 diagonals.
//
// spmv_dia
//   Each thread computes y[i] += A[i,:] * x
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_dia_tex
//   Same as spmv_dia, except x is accessed via texture cache.
//


template <typename OffsetsIterator, typename ValueIterator1, typename ValueIterator2, typename ValueIterator3, typename UnaryFunction, typename BinaryFunction1, typename BinaryFunction2, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_dia_kernel(const int num_rows,
                const int num_cols,
                const int num_diagonals,
                const int pitch,
                const OffsetsIterator diagonal_offsets,
                const ValueIterator1 values,
                const ValueIterator2 x,
                ValueIterator3 y,
                UnaryFunction initialize,
                BinaryFunction1 combine,
                BinaryFunction2 reduce)
{
    typedef typename thrust::iterator_value<OffsetsIterator>::type IndexType;
    typedef typename thrust::iterator_value<ValueIterator1>::type  ValueType;

    __shared__ IndexType offsets[BLOCK_SIZE];

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const IndexType grid_size = BLOCK_SIZE * gridDim.x;

    for(IndexType base = 0; base < num_diagonals; base += BLOCK_SIZE)
    {
        // read a chunk of the diagonal offsets into shared memory
        const IndexType chunk_size = thrust::min(IndexType(BLOCK_SIZE), num_diagonals - base);

        if(threadIdx.x < chunk_size)
            offsets[threadIdx.x] = diagonal_offsets[base + threadIdx.x];

        __syncthreads();

        // process chunk
        for(IndexType row = thread_id; row < num_rows; row += grid_size)
        {
            ValueType sum = (base == 0) ? initialize(y[row]) : ValueType(0);

            // index into values array
            IndexType idx = row + pitch * base;

            for(IndexType n = 0; n < chunk_size; n++)
            {
                const IndexType col = row + offsets[n];

                if(col >= 0 && col < num_cols)
                {
                    const ValueType A_ij = values[idx];
                    sum = reduce(sum, combine(A_ij, x[col]));
                }

                idx += pitch;
            }

            y[row] = sum;
        }

        // wait until all threads are done reading offsets
        __syncthreads();
    }
}


template <typename DerivedPolicy,
          typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              MatrixType& A,
              VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::dia_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    typedef typename MatrixType::diagonal_offsets_array_type::const_iterator          OffsetsIterator;
    typedef typename MatrixType::values_array_type::values_array_type::const_iterator ValueIterator1;

    typedef typename VectorType1::const_iterator                                      ValueIterator2;
    typedef typename VectorType2::iterator                                            ValueIterator3;

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                               spmv_dia_kernel<OffsetsIterator, ValueIterator1, ValueIterator2, ValueIterator3, UnaryFunction, BinaryFunction1, BinaryFunction2, BLOCK_SIZE>,
                               BLOCK_SIZE, (size_t) sizeof(IndexType) * BLOCK_SIZE);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));

    const IndexType num_diagonals = A.values.num_cols;
    const IndexType pitch         = A.values.pitch;

    // TODO can this be removed?
    if (num_diagonals == 0)
    {
        // empty matrix
        thrust::transform(exec, y.begin(), y.begin() + A.num_rows, y.begin(), initialize);
        return;
    }

    cudaStream_t s = stream(thrust::detail::derived_cast(exec));

    spmv_dia_kernel<OffsetsIterator, ValueIterator1, ValueIterator2, ValueIterator3, UnaryFunction, BinaryFunction1, BinaryFunction2, BLOCK_SIZE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, s>>>
    (A.num_rows, A.num_cols, num_diagonals, pitch, A.diagonal_offsets.begin(), A.values.values.begin(), x.begin(), y.begin(), initialize, combine, reduce);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

