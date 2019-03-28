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

#include <cusp/coo_matrix.h>
#include <cusp/system/cuda/arch.h>
#include <cusp/system/cuda/utils.h>
#include <cusp/system/cuda/detail/multiply/coo_serial.h>
#include <cusp/system/cuda/detail/multiply/coo_flat_spmv.h>

#include <algorithm>

// Note: Unlike the other kernels this kernel implements y += A*x

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template<unsigned int CTA_SIZE,
         typename KeyIterator,
         typename ValueIterator>
__device__
void scan_by_key(KeyIterator keys, ValueIterator vals)
{
    typedef typename thrust::iterator_value<KeyIterator>::type   KeyType;
    typedef typename thrust::iterator_value<ValueIterator>::type ValueType;

    KeyType   key = keys[threadIdx.x];
    ValueType val = vals[threadIdx.x];

    if (CTA_SIZE >    1) {
        if(threadIdx.x >=   1 && key == keys[threadIdx.x -   1]) {
            val += vals[threadIdx.x -   1];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >    2) {
        if(threadIdx.x >=   2 && key == keys[threadIdx.x -   2]) {
            val += vals[threadIdx.x -   2];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >    4) {
        if(threadIdx.x >=   4 && key == keys[threadIdx.x -   4]) {
            val += vals[threadIdx.x -   4];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >    8) {
        if(threadIdx.x >=   8 && key == keys[threadIdx.x -   8]) {
            val += vals[threadIdx.x -   8];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >   16) {
        if(threadIdx.x >=  16 && key == keys[threadIdx.x -  16]) {
            val += vals[threadIdx.x -  16];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >   32) {
        if(threadIdx.x >=  32 && key == keys[threadIdx.x -  32]) {
            val += vals[threadIdx.x -  32];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >   64) {
        if(threadIdx.x >=  64 && key == keys[threadIdx.x -  64]) {
            val += vals[threadIdx.x -  64];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >  128) {
        if(threadIdx.x >= 128 && key == keys[threadIdx.x - 128]) {
            val += vals[threadIdx.x - 128];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >  256) {
        if(threadIdx.x >= 256 && key == keys[threadIdx.x - 256]) {
            val += vals[threadIdx.x - 256];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
    if (CTA_SIZE >  512) {
        if(threadIdx.x >= 512 && key == keys[threadIdx.x - 512]) {
            val += vals[threadIdx.x - 512];
        }
        __syncthreads();
        vals[threadIdx.x] = val;
        __syncthreads();
    }
}


template <unsigned int CTA_SIZE,
         unsigned int K,
         typename IndexType,
         typename ValueType>
__global__ void
spmv_coo_flat_k_kernel(const IndexType N,
                       const IndexType interval_size,
                       const IndexType * I,
                       const IndexType * J,
                       const ValueType * V,
                       const ValueType * x,
                       ValueType * y,
                       IndexType * temp_rows,
                       ValueType * temp_vals)
{
    __shared__ IndexType rows[K + 1][CTA_SIZE + 1];
    __shared__ ValueType vals[K + 1][CTA_SIZE + 1];

    __shared__ IndexType last_row;
    __shared__ ValueType last_val;

    const unsigned int interval_begin = interval_size * blockIdx.x;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    const unsigned int unit_size  = K * CTA_SIZE;

    unsigned int base = interval_begin;

    if (threadIdx.x == 0)
    {
        // initialize the carry in values
        last_row = I[interval_begin];
        last_val = 0;
    }

    // process full units
    for(; base + unit_size <= interval_end; base += unit_size)
    {
        // read data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*CTA_SIZE + threadIdx.x;
            const unsigned int n = base + offset;
            rows[offset % K][offset / K] = I[n];                               // i
            vals[offset % K][offset / K] = V[n] * x[J[n]];  // A(i,j) * x(j)
        }

        __syncthreads();

        // carry in
        if (threadIdx.x == 0)
        {
            if (rows[0][0] == last_row)
                vals[0][0] += last_val;
            else
                y[last_row] = last_val;
        }

        unsigned int terminated_rows = 0;

        // segmented scan of K values per thread
        for(unsigned int k = 1; k < K; k++)
        {
            if(rows[k-1][threadIdx.x] == rows[k][threadIdx.x])
                vals[k][threadIdx.x] += vals[k-1][threadIdx.x];
            else
                terminated_rows++;
        }

        rows[K][threadIdx.x] = rows[K-1][threadIdx.x];
        vals[K][threadIdx.x] = vals[K-1][threadIdx.x];

        __syncthreads();

        // scan horizontally across block
        scan_by_key<CTA_SIZE>(&rows[K][0], &vals[K].raw_data());

        IndexType prev_row = (threadIdx.x == 0) ? rows[0][0] : rows[K][threadIdx.x - 1];
        ValueType prev_val = (threadIdx.x == 0) ? 0          : vals[K][threadIdx.x - 1];

        if (prev_row != rows[0][threadIdx.x])
            terminated_rows++;

        // update local values
        for(unsigned int k = 0; k < K; k++)
        {
            if(rows[k][threadIdx.x] == prev_row)
                vals[k][threadIdx.x] += prev_val;
            // XXX could early out here
            //else
            //    break;
        }


        // TODO stream compact terminated rows
        //
        //y[rows[K][threadIdx.x]] = vals[K][threadIdx.x];

        if (prev_row != rows[0][threadIdx.x])
            y[prev_row] = prev_val;

        // write out terminated rows
        for(unsigned int k = 0; k < K - 1; k++)
        {
            if(rows[k][threadIdx.x] != rows[k + 1][threadIdx.x])
                y[rows[k][threadIdx.x]] = vals[k][threadIdx.x];
        }

        if (threadIdx.x == 0)
        {
            last_row = rows[K][CTA_SIZE-1];
            last_val = vals[K][CTA_SIZE-1];
        }

        __syncthreads();
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval_end)
    {
        if(threadIdx.x == 0)
        {
            IndexType prev_row = last_row;
            ValueType prev_val = last_val;

            for(IndexType n = base; n < interval_end; n++)
            {
                IndexType row = I[n];
                ValueType val = V[n] * x[ J[n] ];  // A(i,j) * x(j)

                if(row == prev_row)
                    val += prev_val;
                else
                    y[prev_row] = prev_val;

                prev_row = row;
                prev_val = val;
            }

            last_row = prev_row;
            last_val = prev_val;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0)
    {
        // write the carry out values
        temp_rows[blockIdx.x] = last_row;
        temp_vals[blockIdx.x] = last_val;
    }
}

template <typename IndexType, typename ValueType, typename Array1, typename Array2, bool InitializeY>
void __spmv_coo_flat_k(const coo_matrix<IndexType,ValueType,cusp::device_memory>& coo,
                       const Array1& d_x,
                             Array2& d_y)
{
    const IndexType * I = coo.row_indices.raw_data();
    const IndexType * J = coo.column_indices.raw_data();
    const ValueType * V = coo.values.raw_data();

    if (InitializeY)
        thrust::fill(d_y.begin(), d_y.begin() + coo.num_rows, ValueType(0));

    if(coo.num_entries == 0)
    {
        // empty matrix
        return;
    }
    else if (coo.num_entries < WARP_SIZE)
    {
        // small matrix
        spmv_coo_serial_kernel<IndexType,ValueType> <<<1,1>>>
        (coo.num_entries, I, J, V, d_x, d_y);
        return;
    }

    //TODO Determine optimal CTA_SIZE and max_blocks
    const unsigned int CTA_SIZE = 128;
    const unsigned int K        = 4;

    const unsigned int N = coo.num_entries;

    const unsigned int unit_size  = CTA_SIZE * K;
    const unsigned int num_units  = thrust::detail::util::divide_ri(N, unit_size);
    const unsigned int max_blocks = 120; //thrust::experimental::arch::max_active_blocks(scan_intervals<CTA_SIZE,K,InputIterator,OutputIterator,BinaryFunction>, CTA_SIZE, 0);
    const unsigned int num_blocks = std::min(max_blocks, num_units);
    const unsigned int num_iters  = thrust::detail::util::divide_ri(num_units, num_blocks);

    const unsigned int interval_size = unit_size * num_iters;

    cusp::array1d<IndexType,cusp::device_memory> temp_rows(num_blocks);
    cusp::array1d<ValueType,cusp::device_memory> temp_vals(num_blocks);

    spmv_coo_flat_k_kernel<CTA_SIZE,K,IndexType,ValueType> <<<num_blocks,CTA_SIZE>>>
    (N, interval_size, I, J, V, d_x.raw_data(), d_y.raw_data(),
     temp_rows.raw_data(), temp_vals.raw_data());

    spmv_coo_reduce_update_kernel<IndexType, ValueType, 512> <<<1, 512>>>
    (num_blocks, temp_rows.raw_data(), temp_vals.raw_data(), d_y);
}

template <typename IndexType, typename ValueType, typename Array1, typename Array2>
void spmv_coo_flat_k(const coo_matrix<IndexType,ValueType,cusp::device_memory>& coo,
                     const Array1& d_x,
                           Array2& d_y)
{
    __spmv_coo_flat_k<IndexType, ValueType, true>(coo, d_x, d_y);
}


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

