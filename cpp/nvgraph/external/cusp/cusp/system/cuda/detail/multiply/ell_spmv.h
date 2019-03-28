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

#include <cusp/ell_matrix.h>
#include <cusp/system/cuda/arch.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/device_ptr.h>

#include <cassert>
#include <algorithm>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template <typename IndexType,
          typename ValueType,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2,
          size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_ell_kernel(const IndexType num_rows,
                const IndexType num_cols,
                const IndexType num_cols_per_row,
                const IndexType pitch,
                const IndexType * Aj,
                const ValueType * Ax,
                const ValueType * x,
                ValueType * y,
                UnaryFunction initialize,
                BinaryFunction1 combine,
                BinaryFunction2 reduce)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::device_memory>::invalid_index;

    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        ValueType sum = initialize(y[row]);

        IndexType offset = row;

        for(IndexType n = 0; n < num_cols_per_row; n++)
        {
            const IndexType col = Aj[offset];

            if (col != invalid_index)
            {
                const ValueType A_ij = Ax[offset];
                sum = reduce(sum, combine(A_ij, x[col]));
            }

            offset += pitch;
        }

        y[row] = sum;
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
              cusp::ell_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    if(A.num_entries == 0)
    {
        thrust::transform(y.begin(), y.end(), y.begin(), initialize);
        return;
    }

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                                  spmv_ell_kernel<IndexType,ValueType,UnaryFunction,BinaryFunction1,BinaryFunction2,BLOCK_SIZE>, BLOCK_SIZE, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));

    const IndexType pitch               = A.column_indices.pitch;
    const IndexType num_entries_per_row = A.column_indices.num_cols;

    const IndexType * J = thrust::raw_pointer_cast(&A.column_indices(0,0));
    const ValueType * V = thrust::raw_pointer_cast(&A.values(0,0));

    const ValueType * x_ptr = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_ptr = thrust::raw_pointer_cast(&y[0]);

    // TODO generalize this
    assert(A.column_indices.pitch == A.values.pitch);

    cudaStream_t s = stream(thrust::detail::derived_cast(exec));

    spmv_ell_kernel<IndexType,ValueType,UnaryFunction,BinaryFunction1,BinaryFunction2,BLOCK_SIZE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, s>>>
    (A.num_rows, A.num_cols, num_entries_per_row, pitch, J, V, x_ptr, y_ptr, initialize, combine, reduce);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

