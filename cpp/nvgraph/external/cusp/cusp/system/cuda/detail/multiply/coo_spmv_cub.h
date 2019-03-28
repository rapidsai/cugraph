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

/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * An implementation of COO SpMV using prefix scan to implement a
 * reduce-value-by-row strategy
 ******************************************************************************/

#pragma once

#include <cusp/detail/temporary_array.h>

#include <cusp/system/cuda/arch.h>
#include <cusp/system/cuda/utils.h>
#include <cusp/system/cuda/detail/multiply/coo_serial.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace cub_coo_spmv_detail
{

using namespace thrust::system::cuda::detail::cub_;

/******************************************************************************
 * Texture referencing
 ******************************************************************************/

/**
 * Templated texture reference type for multiplicand vector
 */
template <typename ValueType>
struct TexVector
{
    // Texture type to actually use (e.g., because CUDA doesn't load doubles as texture items)
    typedef typename If<(Equals<ValueType, double>::VALUE), uint2, ValueType>::Type CastType;

    // Texture reference type
    typedef texture<CastType, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    /**
     * Bind textures
     */
    static void BindTexture(void *d_in, int elements)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<CastType>();
        if (d_in)
        {
            size_t offset;
            size_t bytes = sizeof(CastType) * elements;
            cudaBindTexture(&offset, ref, d_in, tex_desc, bytes);
        }
    }

    /**
     * Unbind textures
     */
    static void UnbindTexture()
    {
        cudaUnbindTexture(ref);
    }

    /**
     * Load
     */
    static __device__ __forceinline__ ValueType Load(int offset)
    {
        ValueType output;
        reinterpret_cast<typename TexVector<ValueType>::CastType &>(output) = tex1Dfetch(TexVector<ValueType>::ref, offset);
        return output;
    }
};

// Texture reference definitions
template <typename ValueType>
typename TexVector<ValueType>::TexRef TexVector<ValueType>::ref = 0;


/******************************************************************************
 * Utility types
 ******************************************************************************/


/**
 * A partial dot-product sum paired with a corresponding row-id
 */
template <typename IndexType, typename ValueType>
struct PartialProduct
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType    row;            /// Row-id
    ValueType    partial;        /// PartialProduct sum
};


/**
 * A partial dot-product sum paired with a corresponding row-id (specialized for double-int pairings)
 */
template <>
struct PartialProduct<int, double>
{
    typedef long long index_type;
    typedef double    value_type;

    long long   row;            /// Row-id
    double      partial;        /// PartialProduct sum
};

/**
 * A partial dot-product sum paired with a corresponding row-id (specialized for thrust::complex<float>-int pairings)
 */
template <>
struct PartialProduct<int, thrust::complex<float> >
{
    typedef long long                 index_type;
    typedef thrust::complex<float>    value_type;

    long long                   row;          /// Row-id
    thrust::complex<float>      partial;      /// PartialProduct sum
};


/**
 * Reduce-value-by-row scan operator
 */
template <typename PartialType, typename BinaryFunction>
struct ReduceByKeyOp
{
    BinaryFunction reduce_op;

    __device__ __forceinline__ PartialType operator()(
        const PartialType &first,
        const PartialType &second)
    {
        PartialType retval;

        retval.partial = (second.row != first.row) ?
                         second.partial :
                         reduce_op(first.partial, second.partial);

        retval.row = second.row;
        return retval;
    }
};


/**
 * Stateful block-wide prefix operator for BlockScan
 */
template <typename PartialProduct, typename BinaryFunction>
struct BlockPrefixCallbackOp
{
    // Running block-wide prefix
    PartialProduct running_prefix;
    ReduceByKeyOp<PartialProduct,BinaryFunction> scan_op;

    /**
     * Returns the block-wide running_prefix in thread-0
     */
    __device__ __forceinline__ PartialProduct operator()(
        const PartialProduct &block_aggregate)              ///< The aggregate sum of the BlockScan inputs
    {
        PartialProduct retval = running_prefix;
        running_prefix = scan_op(running_prefix, block_aggregate);
        return retval;
    }
};


/**
 * Operator for detecting discontinuities in a list of row identifiers.
 */
struct NewRowOp
{
    /// Returns true if row_b is the start of a new row
    template <typename IndexType>
    __device__ __forceinline__ bool operator()(
        const IndexType& row_a,
        const IndexType& row_b)
    {
        return (row_a != row_b);
    }
};



/******************************************************************************
 * Persistent thread block types
 ******************************************************************************/

/**
 * SpMV threadblock abstraction for processing a contiguous segment of
 * sparse COO tiles.
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        PartialIterator,
    typename        RowIterator,
    typename        ColumnIterator,
    typename        ValueIterator1,
    typename        ValueIterator2,
    typename        ValueIterator3,
    typename        BinaryFunction1,
    typename        BinaryFunction2>
struct PersistentBlockSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef int HeadFlag;

    // Partial dot product type
    typedef typename thrust::iterator_value<PartialIterator>::type    PartialProduct;

    // base types
    typedef typename thrust::iterator_value<ValueIterator1>::type     ValueType;
    typedef typename PartialProduct::index_type                       IndexType;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockExchange type for exchanging rows between warp-striped -> blocked arrangements
    typedef BlockExchange<IndexType, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeRows;

    // Parameterized BlockExchange type for exchanging values between warp-striped -> blocked arrangements
    typedef BlockExchange<ValueType, BLOCK_THREADS, ITEMS_PER_THREAD, true> BlockExchangeValueTypes;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<IndexType, BLOCK_THREADS> BlockDiscontinuity;

    // Parameterized BlockPrefixCallbackOp type for PartialProduct and BinaryFunction2
    typedef BlockPrefixCallbackOp<PartialProduct,BinaryFunction2> BlockPrefixOp;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        union
        {
            typename BlockExchangeRows::TempStorage         exchange_rows;      // Smem needed for BlockExchangeRows
            typename BlockExchangeValueTypes::TempStorage   exchange_values;    // Smem needed for BlockExchangeValueTypes
            struct
            {
                typename BlockScan::TempStorage             scan;               // Smem needed for BlockScan
                typename BlockDiscontinuity::TempStorage    discontinuity;      // Smem needed for BlockDiscontinuity
            };
        };

        IndexType        prev_block_row;     ///< The last row-ID of the previous thread block
        IndexType        first_block_row;    ///< The first row-ID seen by this thread block
        IndexType        last_block_row;     ///< The last row-ID seen by this thread block
        ValueType        first_product;      ///< The first dot-product written by this thread block
    };

    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                     &temp_storage;
    BlockPrefixOp                   prefix_op;
    RowIterator                     d_rows;
    ColumnIterator                  d_columns;
    ValueIterator1                  d_values;
    ValueIterator2                  d_vector;
    ValueIterator3                  d_result;
    PartialIterator                 d_block_partials;
    int                             block_offset;
    int                             block_end;
    BinaryFunction1                 combine_op;
    BinaryFunction2                 reduce_op;

    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    PersistentBlockSpmv(
        TempStorage                 &temp_storage,
        RowIterator                 d_rows,
        ColumnIterator              d_columns,
        ValueIterator1              d_values,
        ValueIterator2              d_vector,
        ValueIterator3              d_result,
        PartialIterator             d_block_partials,
        int                         block_offset,
        int                         block_end,
        BinaryFunction1             combine_op,
        BinaryFunction2             reduce_op)
        :
        temp_storage(temp_storage),
        d_rows(d_rows),
        d_columns(d_columns),
        d_values(d_values),
        d_vector(d_vector),
        d_result(d_result),
        d_block_partials(d_block_partials),
        block_offset(block_offset),
        block_end(block_end),
        combine_op(combine_op),
        reduce_op(reduce_op)
    {
        // Initialize scalar shared memory values
        if (threadIdx.x == 0)
        {
            IndexType first_block_row           = d_rows[block_offset];
            IndexType last_block_row            = d_rows[block_end - 1];
            IndexType prev_block_row            = blockIdx.x == 0 ? -1 : d_rows[block_offset - 1];

            temp_storage.prev_block_row         = prev_block_row;
            temp_storage.first_block_row        = first_block_row;
            temp_storage.last_block_row         = last_block_row;
            temp_storage.first_product          = ValueType(0);

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = ValueType(0);
        }

        __syncthreads();
    }


    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ProcessTile(
        int block_offset,
        int guarded_items = 0)
    {
        IndexType       columns[ITEMS_PER_THREAD];
        IndexType       rows[ITEMS_PER_THREAD];
        ValueType       values[ITEMS_PER_THREAD];
        PartialProduct  partial_sums[ITEMS_PER_THREAD];
        HeadFlag        head_flags[ITEMS_PER_THREAD];

        // Load a threadblock-striped tile of A (sparse row-ids, column-ids, and values)
        if (FULL_TILE)
        {
            // Unguarded loads
            LoadDirectWarpStriped(threadIdx.x, d_columns + block_offset, columns);
            LoadDirectWarpStriped(threadIdx.x, d_values + block_offset, values);
            LoadDirectWarpStriped(threadIdx.x, d_rows + block_offset, rows);
        }
        else
        {
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            LoadDirectWarpStriped(threadIdx.x, d_columns + block_offset, columns, guarded_items, IndexType(0));
            LoadDirectWarpStriped(threadIdx.x, d_values + block_offset, values, guarded_items, ValueType(0));
            LoadDirectWarpStriped(threadIdx.x, d_rows + block_offset, rows, guarded_items, temp_storage.last_block_row);
        }

        // Load the referenced values from x and compute the dot product partials sums
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            // values[ITEM] *= TexVector<ValueType>::Load(columns[ITEM]);
            values[ITEM] = combine_op(values[ITEM], d_vector[columns[ITEM]]);
        }

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeValueTypes(temp_storage.exchange_values).WarpStripedToBlocked(values);

        __syncthreads();

        // Transpose from warp-striped to blocked arrangement
        BlockExchangeRows(temp_storage.exchange_rows).WarpStripedToBlocked(rows);

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Flag row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
            head_flags,                     // (Out) Head flags
            rows,                           // Original row ids
            NewRowOp(),                     // Functor for detecting start of new rows
            prefix_op.running_prefix.row);  // Last row ID from previous tile to compare with first row ID in this tile

        // Assemble partial product structures
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            partial_sums[ITEM].partial = values[ITEM];
            partial_sums[ITEM].row = rows[ITEM];
        }

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Reduce reduce-value-by-row across partial_sums using exclusive prefix scan
        PartialProduct block_aggregate;
        PartialProduct identity;
        identity.row = -1;
        identity.partial = ValueType(0);
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan output
            identity,
            ReduceByKeyOp<PartialProduct,BinaryFunction2>(),// Scan operator
            block_aggregate,                // Block-wide total (unused)
            prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total

        // Barrier for smem reuse and coherence
        __syncthreads();

        // Scatter an accumulated dot product if it is the head of a valid row
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                // Save off the first partial product that this thread block will scatter
                if (partial_sums[ITEM].row == temp_storage.first_block_row)
                {
                    temp_storage.first_product = partial_sums[ITEM].partial;
                }
                else
                {
                    d_result[partial_sums[ITEM].row] = reduce_op(d_result[partial_sums[ITEM].row], partial_sums[ITEM].partial);
                }
            }
        }
    }


    /**
     * Iterate over input tiles belonging to this thread block
     */
    __device__ __forceinline__
    void ProcessTiles()
    {
        // Process full tiles
        while (block_offset <= block_end - TILE_ITEMS)
        {
            ProcessTile<true>(block_offset);
            block_offset += TILE_ITEMS;
        }

        // Process the last, partially-full tile (if present)
        int guarded_items = block_end - block_offset;
        if (guarded_items)
        {
            ProcessTile<false>(block_offset, guarded_items);
        }

        // Barrier for smem reuse and coherence
        __syncthreads();

        if (threadIdx.x == 0)
        {
            if (gridDim.x == 1)
            {
                // Scatter the final aggregate (this kernel contains only 1 threadblock)
                d_result[prefix_op.running_prefix.row] = reduce_op(d_result[prefix_op.running_prefix.row], prefix_op.running_prefix.partial);

                // Scatter the first aggregate (this kernel contains only 1 threadblock)
                if(temp_storage.first_block_row != prefix_op.running_prefix.row)
                {
                    d_result[temp_storage.first_block_row] = reduce_op(d_result[temp_storage.first_block_row], temp_storage.first_product);
                }
            }
            else
            {
                // Write the first and last partial products from this thread block so
                // that they can be subsequently "fixed up" in the next kernel.

                PartialProduct first_product;
                first_product.row       = temp_storage.first_block_row;
                first_product.partial   = temp_storage.first_product;

                if(first_product.row != temp_storage.prev_block_row)
                {
                     first_product.partial = reduce_op(d_result[first_product.row], first_product.partial);
                }

                if(temp_storage.first_block_row != prefix_op.running_prefix.row)
                {
                     prefix_op.running_prefix.partial = reduce_op(d_result[prefix_op.running_prefix.row], prefix_op.running_prefix.partial);
                }

                d_block_partials[blockIdx.x * 2]          = first_product;
                d_block_partials[(blockIdx.x * 2) + 1]    = prefix_op.running_prefix;
            }
        }
    }
};


/**
 * Threadblock abstraction for "fixing up" an array of interblock SpMV partial products.
 */
template <
    int             BLOCK_THREADS,
    int             ITEMS_PER_THREAD,
    typename        PartialIterator,
    typename        ValueIterator,
    typename        BinaryFunction>
struct FinalizeSpmvBlock
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    typedef typename thrust::iterator_value<PartialIterator>::type    PartialProduct;
    typedef typename thrust::iterator_value<ValueIterator>::type      ValueType;
    typedef typename PartialProduct::index_type                       IndexType;

    // Head flag type
    typedef int HeadFlag;

    // Parameterized BlockScan type for reduce-value-by-row scan
    typedef BlockScan<PartialProduct, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

    // Parameterized BlockDiscontinuity type for setting head-flags for each new row segment
    typedef BlockDiscontinuity<IndexType, BLOCK_THREADS> BlockDiscontinuity;

    // Parameterized BlockPrefixCallbackOp type for PartialProduct and BinaryFunction
    typedef BlockPrefixCallbackOp<PartialProduct,BinaryFunction> BlockPrefixOp;

    // Shared memory type for this threadblock
    struct TempStorage
    {
        typename BlockScan::TempStorage           scan;               // Smem needed for reduce-value-by-row scan
        typename BlockDiscontinuity::TempStorage  discontinuity;      // Smem needed for head-flagging

        IndexType last_block_row;
    };


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    TempStorage                             &temp_storage;
    BlockPrefixOp                           prefix_op;
    ValueIterator                           d_result;
    PartialIterator                         d_block_partials;
    int                                     num_partials;
    BinaryFunction                          reduce_op;


    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__
    FinalizeSpmvBlock(
        TempStorage                 &temp_storage,
        ValueIterator               d_result,
        PartialIterator             d_block_partials,
        int                         num_partials,
        BinaryFunction              reduce_op)
        :
        temp_storage(temp_storage),
        d_result(d_result),
        d_block_partials(d_block_partials),
        num_partials(num_partials),
        reduce_op(reduce_op)
    {
        // Initialize scalar shared memory values
        if (threadIdx.x == 0)
        {
            IndexType first_block_row           = d_block_partials[0].row;
            IndexType last_block_row            = d_block_partials[num_partials - 1].row;
            temp_storage.last_block_row         = last_block_row;

            // Initialize prefix_op to identity
            prefix_op.running_prefix.row        = first_block_row;
            prefix_op.running_prefix.partial    = ValueType(0);
        }

        __syncthreads();
    }


    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    template <bool FULL_TILE>
    __device__ __forceinline__
    void ProcessTile(
        int block_offset,
        int guarded_items = 0)
    {
        IndexType       rows[ITEMS_PER_THREAD];
        PartialProduct  partial_sums[ITEMS_PER_THREAD];
        IndexType       head_flags[ITEMS_PER_THREAD];

        // Load a tile of block partials from previous kernel
        if (FULL_TILE)
        {
            // Full tile
            LoadDirectBlocked(threadIdx.x, d_block_partials + block_offset, partial_sums);
        }
        else
        {
            // Partial tile (extend zero-valued coordinates of the last partial-product for out-of-bounds items)
            PartialProduct default_sum;
            default_sum.row = temp_storage.last_block_row;
            default_sum.partial = ValueType(0);

            LoadDirectBlocked(threadIdx.x, d_block_partials + block_offset, partial_sums, guarded_items, default_sum);
        }

        // Copy out row IDs for row-head flagging
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            rows[ITEM] = partial_sums[ITEM].row;
        }

        __syncthreads();

        // Flag row heads by looking for discontinuities
        BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
            head_flags,                     // (Out) Head flags
            rows,                           // Original row ids
            NewRowOp(),                     // Functor for detecting start of new rows
            prefix_op.running_prefix.row);   // Last row ID from previous tile to compare with first row ID in this tile

        __syncthreads();

        // Reduce reduce-value-by-row across partial_sums using exclusive prefix scan
        PartialProduct block_aggregate;
        PartialProduct identity;
        identity.row = -1;
        identity.partial = ValueType(0);
        BlockScan(temp_storage.scan).ExclusiveScan(
            partial_sums,                   // Scan input
            partial_sums,                   // Scan input
            identity,
            ReduceByKeyOp<PartialProduct,BinaryFunction>(),// Scan operator
            block_aggregate,                // Block-wide total (unused)
            prefix_op);                     // Prefix operator for seeding the block-wide scan with the running total

        __syncthreads();

        // Scatter an accumulated dot product if it is the head of a valid row
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
            }
        }
    }


    /**
     * Iterate over input tiles belonging to this thread block
     */
    __device__ __forceinline__
    void ProcessTiles()
    {
        // Process full tiles
        int block_offset = 0;
        while (block_offset <= (num_partials - TILE_ITEMS))
        {
            ProcessTile<true>(block_offset);
            block_offset += TILE_ITEMS;
        }

        // Process final partial tile (if present)
        int guarded_items = num_partials - block_offset;
        if (guarded_items)
        {
            ProcessTile<false>(block_offset, guarded_items);
        }

        __syncthreads();

        // Scatter the final aggregate (this kernel contains only 1 threadblock)
        if (threadIdx.x == 0)
        {
            d_result[prefix_op.running_prefix.row] = prefix_op.running_prefix.partial;
        }
    }
};


/******************************************************************************
 * Kernel entrypoints
 ******************************************************************************/



/**
 * SpMV kernel whose thread blocks each process a contiguous segment of sparse COO tiles.
 */
template <
    int                             BLOCK_THREADS,
    int                             ITEMS_PER_THREAD,
    typename                        PartialIterator,
    typename                        RowIterator,
    typename                        ColumnIterator,
    typename                        ValueIterator1,
    typename                        ValueIterator2,
    typename                        ValueIterator3,
    typename                        BinaryFunction1,
    typename                        BinaryFunction2>
__launch_bounds__ (BLOCK_THREADS)
__global__ void CooKernel(
    GridEvenShare<int>                    even_share,
    PartialIterator                       d_block_partials,
    const RowIterator                     d_rows,
    const ColumnIterator                  d_columns,
    const ValueIterator1                  d_values,
    const ValueIterator2                  d_vector,
    ValueIterator3                        d_result,
    BinaryFunction1                       combine_op,
    BinaryFunction2                       reduce_op)
{
    // Specialize SpMV threadblock abstraction type
    typedef PersistentBlockSpmv<BLOCK_THREADS, ITEMS_PER_THREAD, PartialIterator, RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3, BinaryFunction1, BinaryFunction2> PersistentBlockSpmv;

    // Shared memory allocation
    __shared__ typename PersistentBlockSpmv::TempStorage temp_storage;

    // Initialize threadblock even-share to tell us where to start and stop our tile-processing
    even_share.BlockInit();

    // Construct persistent thread block
    PersistentBlockSpmv persistent_block(
        temp_storage,
        d_rows,
        d_columns,
        d_values,
        d_vector,
        d_result,
        d_block_partials,
        even_share.block_offset,
        even_share.block_end,
        combine_op,
        reduce_op);

    // Process input tiles
    persistent_block.ProcessTiles();
}


/**
 * Kernel for "fixing up" an array of interblock SpMV partial products.
 */
template <
    int                             BLOCK_THREADS,
    int                             ITEMS_PER_THREAD,
    typename                        PartialIterator,
    typename                        ValueIterator,
    typename                        BinaryFunction>
__launch_bounds__ (BLOCK_THREADS,  1)
__global__ void CooFinalizeKernel(
    PartialIterator                      d_block_partials,
    int                                  num_partials,
    ValueIterator                        d_result,
    BinaryFunction                       reduce_op)
{
    // Specialize "fix-up" threadblock abstraction type
    typedef FinalizeSpmvBlock<BLOCK_THREADS, ITEMS_PER_THREAD, PartialIterator, ValueIterator, BinaryFunction> FinalizeSpmvBlock;

    // Shared memory allocation
    __shared__ typename FinalizeSpmvBlock::TempStorage temp_storage;

    // Construct persistent thread block
    FinalizeSpmvBlock persistent_block(temp_storage, d_result, d_block_partials, num_partials, reduce_op);

    // Process input tiles
    persistent_block.ProcessTiles();
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_coo(cusp::system::cuda::detail::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename MatrixType::index_type                                 IndexType;
    typedef typename VectorType2::value_type                                ValueType;
    typedef typename VectorType2::memory_space                              MemorySpace;
    typedef PartialProduct<IndexType, ValueType>                            PartialProduct;

    typedef typename MatrixType::row_indices_array_type::const_iterator     RowIterator;
    typedef typename MatrixType::column_indices_array_type::const_iterator  ColumnIterator;
    typedef typename MatrixType::values_array_type::const_iterator          ValueIterator1;

    typedef typename VectorType1::const_iterator                            ValueIterator2;
    typedef typename VectorType2::iterator                                  ValueIterator3;

    typedef typename cusp::array1d<PartialProduct,MemorySpace>::iterator    PartialIterator;

    if(A.num_entries == 0)
    {
        // empty matrix
        return;
    }

    cudaStream_t s = stream(thrust::detail::derived_cast(exec));

    // Parameterization for SM35
    enum
    {
        COO_BLOCK_THREADS           = 64,
        COO_ITEMS_PER_THREAD        = 10,
        COO_SUBSCRIPTION_FACTOR     = 4,
        FINALIZE_BLOCK_THREADS      = 256,
        FINALIZE_ITEMS_PER_THREAD   = 4,
    };

    const int COO_TILE_SIZE = COO_BLOCK_THREADS * COO_ITEMS_PER_THREAD;

    // Create SOA version of coo_graph on host
    // int num_cols    = A.num_cols;
    int num_edges   = A.num_entries;

    // Determine launch configuration from kernel properties
    int coo_sm_occupancy = 0;
    MaxSmOccupancy(coo_sm_occupancy,
                   CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD,
                   PartialIterator, RowIterator, ColumnIterator,
                   ValueIterator1, ValueIterator2, ValueIterator3,
                   BinaryFunction1, BinaryFunction2>,
                   COO_BLOCK_THREADS);

    // int sm_count = -1;
    // cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, 0);
    thrust::system::cuda::detail::device_properties_t properties = thrust::system::cuda::detail::device_properties();
    int sm_count = properties.multiProcessorCount;
    int max_coo_grid_size   = sm_count * coo_sm_occupancy * COO_SUBSCRIPTION_FACTOR;

    // Construct an even-share work distribution
    GridEvenShare<int> even_share(num_edges, max_coo_grid_size, COO_TILE_SIZE);
    int coo_grid_size  = even_share.grid_size;
    int num_partials   = coo_grid_size * 2;

    cusp::detail::temporary_array<PartialProduct, DerivedPolicy> block_partials(exec, num_partials);

    // Bind textures
    // void *d_x = (void *) thrust::raw_pointer_cast(&x[0]);
    // TexVector<ValueType>::BindTexture(d_x, num_cols);

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // Run the COO kernel
    CooKernel<COO_BLOCK_THREADS, COO_ITEMS_PER_THREAD><<<coo_grid_size, COO_BLOCK_THREADS, 0, s>>>(
        even_share,
        thrust::raw_pointer_cast(&block_partials[0]),
        A.row_indices.begin(),
        A.column_indices.begin(),
        A.values.begin(),
        x.begin(),
        y.begin(),
        combine,
        reduce);

    if (coo_grid_size > 1)
    {
        // Run the COO finalize kernel
        CooFinalizeKernel<FINALIZE_BLOCK_THREADS, FINALIZE_ITEMS_PER_THREAD><<<1, FINALIZE_BLOCK_THREADS, 0, s>>>(
            thrust::raw_pointer_cast(&block_partials[0]),
            num_partials,
            y.begin(),
            reduce);
    }
}

} // end namespace cub_coo_spmv_detail

template <typename DerivedPolicy,
          typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              thrust::identity<typename VectorType2::value_type> initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::coo_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cub_coo_spmv_detail::spmv_coo(exec, A, x, y, combine, reduce);
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
              cusp::coo_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    thrust::transform(exec, y.begin(), y.end(), y.begin(), initialize);
    cub_coo_spmv_detail::spmv_coo(exec, A, x, y, combine, reduce);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

