/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

template <typename RowIterator, typename ColumnIterator, typename ValueIterator1,
         typename ValueIterator2, typename ValueIterator3,
         typename UnaryFunction, typename BinaryFunction1, typename BinaryFunction2,
         unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(VECTORS_PER_BLOCK * THREADS_PER_VECTOR,1)
__global__ void BlockSpmvKernel(const unsigned int A_num_rows,
                                const RowIterator A_row_offsets,
                                const ColumnIterator A_column_indices,
                                const ValueIterator1 A_values,
                                const ValueIterator2 X_values,
                                ValueIterator3 Y_values,
                                UnaryFunction initialize,
                                BinaryFunction1 combine,
                                BinaryFunction2 reduce)
{
    typedef typename thrust::iterator_value<RowIterator>::type    IndexType;
    typedef typename thrust::iterator_value<ValueIterator1>::type ValueType;

    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];
    __shared__ volatile IndexType sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile ValueType vdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals

    const int THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const int thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType row = vector_id; row < A_num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = A_row_offsets[row + thread_lane];

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        // ValueType sum = X_values[row*THREADS_PER_VECTOR + thread_lane] * (row_end - row_start);
        ValueType sum = initialize(Y_values[row * THREADS_PER_VECTOR + thread_lane]);

        // accumulate local sums
        // for(IndexType jj = row_start; jj < row_end; jj += THREADS_PER_VECTOR)
        // {
        //     int num_cols = row_end - jj;

        //     if(num_cols < THREADS_PER_VECTOR)
        //     {
        //         sdata[threadIdx.x] = thread_lane < num_cols ? A_column_indices[jj + thread_lane] : 0;

        //         for(IndexType kk = 0; kk < num_cols; kk++)
        //         {
        //             IndexType col = sdata[vector_lane*THREADS_PER_VECTOR + kk];
        //             sum -= X_values[col*THREADS_PER_VECTOR + thread_lane];
        //         }
        //     }
        //     else
        //     {
        //         sdata[threadIdx.x] = A_column_indices[jj + thread_lane];

        //         #pragma unroll
        //         for(IndexType kk = 0; kk < THREADS_PER_VECTOR; kk++)
        //         {
        //             IndexType col = sdata[vector_lane*THREADS_PER_VECTOR + kk];
        //             sum -= X_values[col*THREADS_PER_VECTOR + thread_lane];
        //         }
        //     }
        // }
        for(IndexType jj = row_start; jj < row_end; jj += THREADS_PER_VECTOR)
        {
            int num_cols = min(THREADS_PER_VECTOR,row_end - jj);

            sdata[threadIdx.x] = 0;
            vdata[threadIdx.x] = 0;

            if( thread_lane < num_cols )
            {
                sdata[threadIdx.x] = A_column_indices[jj + thread_lane];
                vdata[threadIdx.x] = A_values[jj + thread_lane];
            }

            for(IndexType kk = 0; kk < num_cols; kk++)
            {
                IndexType col = sdata[vector_lane*THREADS_PER_VECTOR + kk];
                ValueType val = vdata[vector_lane*THREADS_PER_VECTOR + kk];
                sum = reduce(sum, combine(val, X_values[col*THREADS_PER_VECTOR + thread_lane]));
            }
        }

        Y_values[row*THREADS_PER_VECTOR + thread_lane] = sum;
    }
}

template<unsigned int NUM_COLS,
         typename DerivedPolicy,
         typename MatrixType,
         typename VectorType1,
         typename VectorType2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void __spmv_csr_block(cuda::execution_policy<DerivedPolicy>& exec,
                      const MatrixType& A,
                      const VectorType1& x,
                      VectorType2& y,
                      UnaryFunction   initialize,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    typedef typename MatrixType::row_offsets_array_type::const_iterator     RowIterator;
    typedef typename MatrixType::column_indices_array_type::const_iterator  ColumnIterator;
    typedef typename MatrixType::values_array_type::const_iterator          ValueIterator1;

    typedef typename VectorType1::values_array_type::const_iterator         ValueIterator2;
    typedef typename VectorType2::values_array_type::iterator               ValueIterator3;

    const size_t THREADS_PER_BLOCK  = 256;
    const size_t THREADS_PER_VECTOR = NUM_COLS;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                                  BlockSpmvKernel<RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3,
                                  UnaryFunction, BinaryFunction1, BinaryFunction2,
                                  VECTORS_PER_BLOCK, THREADS_PER_VECTOR>,
                                  THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = std::min(20 * MAX_BLOCKS, DIVIDE_INTO(A.num_rows, VECTORS_PER_BLOCK));

    cudaStream_t s = stream(thrust::detail::derived_cast(exec));

    // Reduce tiles into reduceDevice.
    BlockSpmvKernel<RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3,
                    UnaryFunction, BinaryFunction1, BinaryFunction2,
                    VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
      <<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, s>>>
      (A.num_rows, A.row_offsets.begin(), A.column_indices.begin(), A.values.begin(),
       x.values.begin(), y.values.begin(),
       initialize, combine, reduce);
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::csr_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    if (x.num_cols ==  1) {
        multiply(exec, A, x.values, y.values, initialize, combine, reduce);
        return;
    }
    if (x.num_cols <=  2) {
        __spmv_csr_block<2>(exec, A, x, y, initialize, combine, reduce);
        return;
    }
    if (x.num_cols <=  4) {
        __spmv_csr_block<4>(exec, A, x, y, initialize, combine, reduce);
        return;
    }
    if (x.num_cols <=  8) {
        __spmv_csr_block<8>(exec, A, x, y, initialize, combine, reduce);
        return;
    }
    if (x.num_cols <= 16) {
        __spmv_csr_block<16>(exec, A, x, y, initialize, combine, reduce);
        return;
    }

    __spmv_csr_block<32>(exec, A, x, y, initialize, combine, reduce);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

