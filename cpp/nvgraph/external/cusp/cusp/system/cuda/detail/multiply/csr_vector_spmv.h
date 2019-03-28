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

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with
//   the x vector, in parallel.  This division of work implies that
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of
//   work.  Since an entire 32-thread warp is assigned to each row, many
//   threads will remain idle when their row contains a small number
//   of elements.  This code relies on implicit synchronization among
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is
//   used for accessing the x vector.
//
//  Note: THREADS_PER_VECTOR must be one of [2,4,8,16,32]


template <typename RowIterator, typename ColumnIterator, typename ValueIterator1,
         typename ValueIterator2, typename ValueIterator3,
         typename UnaryFunction, typename BinaryFunction1, typename BinaryFunction2,
         unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void
spmv_csr_vector_kernel(const unsigned int num_rows,
                       const RowIterator    Ap,
                       const ColumnIterator Aj,
                       const ValueIterator1 Ax,
                       const ValueIterator2  x,
                       ValueIterator3        y,
                       UnaryFunction initialize,
                       BinaryFunction1 combine,
                       BinaryFunction2 reduce)
{
    typedef typename thrust::iterator_value<RowIterator>::type    IndexType;
    typedef typename thrust::iterator_value<ValueIterator1>::type ValueType;

    __shared__ volatile ValueType sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];

    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        ValueType sum = (thread_lane == 0) ? initialize(y[row]) : ValueType(0);

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
            // ensure aligned memory access to Aj and Ax

            IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum = reduce(sum, combine(Ax[jj], x[Aj[jj]]));

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                sum = reduce(sum, combine(Ax[jj], x[Aj[jj]]));
        }
        else
        {
            // accumulate local sums
            for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                sum = reduce(sum, combine(Ax[jj], x[Aj[jj]]));
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;

        // TODO: remove temp var WAR for MSVC
        ValueType temp;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) {
            temp = sdata[threadIdx.x + 16];
            sdata[threadIdx.x] = sum = reduce(sum, temp);
        }
        if (THREADS_PER_VECTOR >  8) {
            temp = sdata[threadIdx.x +  8];
            sdata[threadIdx.x] = sum = reduce(sum, temp);
        }
        if (THREADS_PER_VECTOR >  4) {
            temp = sdata[threadIdx.x +  4];
            sdata[threadIdx.x] = sum = reduce(sum, temp);
        }
        if (THREADS_PER_VECTOR >  2) {
            temp = sdata[threadIdx.x +  2];
            sdata[threadIdx.x] = sum = reduce(sum, temp);
        }
        if (THREADS_PER_VECTOR >  1) {
            temp = sdata[threadIdx.x +  1];
            sdata[threadIdx.x] = sum = reduce(sum, temp);
        }

        // first thread writes the result
        if (thread_lane == 0)
            y[row] = ValueType(sdata[threadIdx.x]);
    }
}

template <unsigned int THREADS_PER_VECTOR,
         typename DerivedPolicy,
         typename MatrixType,
         typename VectorType1,
         typename VectorType2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void __spmv_csr_vector(cuda::execution_policy<DerivedPolicy>& exec,
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

    typedef typename VectorType1::const_iterator                            ValueIterator2;
    typedef typename VectorType2::iterator                                  ValueIterator3;

    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                                  spmv_csr_vector_kernel<RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3,
                                  UnaryFunction, BinaryFunction1, BinaryFunction2,
                                  VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, VECTORS_PER_BLOCK));

    cudaStream_t s = stream(thrust::detail::derived_cast(exec));

    spmv_csr_vector_kernel<RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3,
                           UnaryFunction, BinaryFunction1, BinaryFunction2,
                           VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, s>>>
                           (A.num_rows, A.row_offsets.begin(), A.column_indices.begin(), A.values.begin(), x.begin(), y.begin(),
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
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename MatrixType::index_type IndexType;

    const IndexType nnz_per_row = A.num_entries / A.num_rows;

    if (nnz_per_row <=  2) {
        __spmv_csr_vector<2>(exec, A, x, y, initialize, combine, reduce);
        return;
    }
    if (nnz_per_row <=  4) {
        __spmv_csr_vector<4>(exec, A, x, y, initialize, combine, reduce);
        return;
    }
    if (nnz_per_row <=  8) {
        __spmv_csr_vector<8>(exec, A, x, y, initialize, combine, reduce);
        return;
    }
    if (nnz_per_row <= 16) {
        __spmv_csr_vector<16>(exec, A, x, y, initialize, combine, reduce);
        return;
    }

    __spmv_csr_vector<32>(exec, A, x, y, initialize, combine, reduce);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

