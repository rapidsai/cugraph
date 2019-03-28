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

#include <cusp/copy.h>
#include <cusp/coo_matrix.h>
#include <cusp/format_utils.h>

#include <cusp/blas/blas.h>

#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>

#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>
#include <cassert>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

// functors
template <typename IndexType, typename Orientation>
struct array2d_map_functor : public thrust::unary_function<IndexType,IndexType>
{
    IndexType pitch;

    array2d_map_functor(IndexType pitch)
        : pitch(pitch) {}

    template<typename Tuple>
    __host__ __device__
    IndexType operator()(const Tuple& t) const
    {
        IndexType row = thrust::get<0>(t);
        IndexType col = thrust::get<1>(t);

        return cusp::detail::index_of(row, col, pitch, Orientation());
    }
};

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::array1d_format&)
{
    // convert src -> coo_matrix -> dst
    typename cusp::detail::as_array2d_type<SourceType>::type tmp;

    cusp::convert(exec, src, tmp);
    cusp::convert(exec, tmp, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::array2d_format&)
{
    typedef typename DestinationType::index_type  IndexType;
    typedef typename DestinationType::value_type  ValueType;
    typedef typename DestinationType::orientation Orientation;

    dst.resize(src.num_rows, src.num_cols);

    thrust::fill(exec, dst.values.begin(), dst.values.end(), ValueType(0));

    if(src.num_entries == 0) return;

    thrust::scatter(exec,
                    src.values.begin(), src.values.end(),
                    thrust::make_transform_iterator(
                        thrust::make_zip_iterator(
                            thrust::make_tuple(src.row_indices.begin(), src.column_indices.begin())),
                        array2d_map_functor<IndexType,Orientation>(dst.pitch)),
                    dst.values.begin());
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::csr_format&)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    if(src.num_entries == 0) return;

    cusp::indices_to_offsets(exec, src.row_indices, dst.row_offsets);
    cusp::copy(exec, src.column_indices, dst.column_indices);
    cusp::copy(exec, src.values,         dst.values);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::dia_format&,
        size_t alignment = 32)
{
    typedef typename DestinationType::index_type   IndexType;
    typedef typename DestinationType::value_type   ValueType;

    if(src.num_entries == 0)
    {
        dst.resize(src.num_rows, src.num_cols, src.num_entries, 0);
        return;
    }

    const size_t occupied_diagonals =
        cusp::count_diagonals(exec, src.num_rows, src.num_cols, src.row_indices, src.column_indices);

    const float max_fill   = 3.0;
    const float threshold  = 1e6; // 1M entries
    const float size       = float(occupied_diagonals) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    // compute number of occupied diagonals and enumerate them
    cusp::detail::temporary_array<IndexType, DerivedPolicy> diag_map(exec, src.num_entries);
    thrust::transform(exec,
                      thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), src.column_indices.begin() ) ),
                      thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.end()  , src.column_indices.end() ) )  ,
                      diag_map.begin(),
                      cusp::detail::occupied_diagonal_functor<IndexType>(src.num_rows));

    // place ones in diagonals array locations with occupied diagonals
    cusp::detail::temporary_array<IndexType, DerivedPolicy> diagonals(exec, src.num_rows + src.num_cols,IndexType(0));
    thrust::scatter(exec,
                    thrust::constant_iterator<IndexType>(1),
                    thrust::constant_iterator<IndexType>(1) + src.num_entries,
                    diag_map.begin(),
                    diagonals.begin());

    const IndexType num_diagonals = thrust::reduce(exec, diagonals.begin(), diagonals.end());

    // allocate DIA structure
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, alignment);

    // fill in values array
    thrust::fill(exec, dst.values.values.begin(), dst.values.values.end(), ValueType(0));

    // fill in diagonal_offsets array
    thrust::copy_if(exec,
                    thrust::counting_iterator<IndexType>(0),
                    thrust::counting_iterator<IndexType>(src.num_rows + src.num_cols),
                    diagonals.begin(),
                    dst.diagonal_offsets.begin(),
                    cusp::greater_value<IndexType>(0));

    // replace shifted diagonals with index of diagonal in offsets array
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets( dst.diagonal_offsets );
    for( IndexType num_diag = 0; num_diag < num_diagonals; num_diag++ )
        thrust::replace(exec, diag_map.begin(), diag_map.end(), diagonal_offsets[num_diag], num_diag);

    // copy values to dst
    thrust::scatter(exec,
                    src.values.begin(), src.values.end(),
                    thrust::make_transform_iterator(
                        thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), diag_map.begin() ) ),
                        cusp::detail::diagonal_index_functor<IndexType>(dst.values.pitch)),
                    dst.values.values.begin());

    cusp::constant_array<IndexType> constant_view(num_diagonals, dst.num_rows);
    cusp::blas::axpy(exec,
                     constant_view,
                     dst.diagonal_offsets,
                     IndexType(-1));
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::ell_format&,
        size_t num_entries_per_row = 0,
        size_t alignment = 32)
{
    typedef typename DestinationType::index_type   IndexType;
    typedef typename DestinationType::value_type   ValueType;

    if(src.num_entries == 0)
    {
        dst.resize(src.num_rows, src.num_cols, src.num_entries, num_entries_per_row);
        return;
    }

    if(num_entries_per_row == 0)
    {
        cusp::detail::temporary_array<IndexType, DerivedPolicy> row_offsets(exec, src.num_rows + 1);
        cusp::indices_to_offsets(exec, src.row_indices, row_offsets);

        const size_t max_entries_per_row = cusp::compute_max_entries_per_row(exec, row_offsets);

        const float max_fill  = 3.0;
        const float threshold  = 1e6; // 1M entries
        const float size       = float(max_entries_per_row) * float(src.num_rows);
        const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

        if (max_fill < fill_ratio && size > threshold)
            throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

        num_entries_per_row = max_entries_per_row;
    }

    size_t num_entries = src.num_entries - thrust::count(exec, src.values.begin(), src.values.end(), ValueType(0));

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, num_entries, num_entries_per_row, alignment);

    // compute permutation from COO index to ELL index
    // first enumerate the entries within each row, e.g. [0, 1, 2, 0, 1, 2, 3, ...]
    cusp::detail::temporary_array<IndexType, DerivedPolicy> permutation(exec, src.num_entries);

    thrust::exclusive_scan_by_key(exec,
                                  src.row_indices.begin(), src.row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  permutation.begin(),
                                  IndexType(0));

    // next, scale by pitch and add row index
    cusp::blas::axpby(exec,
                      permutation, src.row_indices,
                      permutation,
                      IndexType(dst.column_indices.pitch),
                      IndexType(1));

    // fill output with padding
    thrust::fill(exec, dst.column_indices.values.begin(), dst.column_indices.values.end(), IndexType(-1));
    thrust::fill(exec, dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

    // scatter COO entries to ELL
    thrust::scatter(exec,
                    src.column_indices.begin(), src.column_indices.end(),
                    permutation.begin(),
                    dst.column_indices.values.begin());
    thrust::scatter(exec,
                    src.values.begin(), src.values.end(),
                    permutation.begin(),
                    dst.values.values.begin());
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format& format1,
        cusp::hyb_format& format2,
        size_t num_entries_per_row = 0,
        size_t alignment = 32)
{
    typedef typename DestinationType::index_type   IndexType;
    typedef typename DestinationType::value_type   ValueType;

    if(src.num_entries == 0)
    {
        dst.resize(src.num_rows, src.num_cols, 0, 0, num_entries_per_row, alignment);
        return;
    }

    if(num_entries_per_row == 0)
    {
        const float  relative_speed      = 3.0;
        const size_t breakeven_threshold = 4096;

        cusp::detail::temporary_array<IndexType, DerivedPolicy> row_offsets(exec, src.num_rows + 1);
        cusp::indices_to_offsets(src.row_indices, row_offsets);

        num_entries_per_row =
            cusp::compute_optimal_entries_per_row(exec, row_offsets, relative_speed, breakeven_threshold);
    }

    cusp::detail::temporary_array<IndexType, DerivedPolicy> indices(exec, src.num_entries);
    thrust::exclusive_scan_by_key(exec,
                                  src.row_indices.begin(), src.row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  indices.begin(),
                                  IndexType(0));

    size_t num_coo_entries = thrust::count_if(exec,
                                              indices.begin(), indices.end(),
                                              cusp::greater_equal_value<size_t>(num_entries_per_row));
    size_t num_ell_entries = src.num_entries - num_coo_entries;

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, num_ell_entries, num_coo_entries, num_entries_per_row, alignment);

    // fill output with padding
    thrust::fill(exec, dst.ell.column_indices.values.begin(), dst.ell.column_indices.values.end(), IndexType(-1));
    thrust::fill(exec, dst.ell.values.values.begin(),         dst.ell.values.values.end(),         ValueType(0));

    // write tail of each row to COO portion
    thrust::copy_if
    (exec,
     thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), src.column_indices.begin(), src.values.begin() ) ),
     thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.end()  , src.column_indices.end()  , src.values.end()   ) ),
     indices.begin(),
     thrust::make_zip_iterator( thrust::make_tuple( dst.coo.row_indices.begin(), dst.coo.column_indices.begin(), dst.coo.values.begin() ) ),
     cusp::greater_equal_value<size_t>(num_entries_per_row) );

    assert(dst.ell.column_indices.pitch == dst.ell.values.pitch);

    size_t pitch = dst.ell.column_indices.pitch;

    // next, scale by pitch and add row index
    cusp::blas::axpby(exec,
                      indices, src.row_indices,
                      indices,
                      IndexType(pitch),
                      IndexType(1));

    // scatter COO entries to ELL
    thrust::scatter_if(exec,
                       src.column_indices.begin(), src.column_indices.end(),
                       indices.begin(),
                       indices.begin(),
                       dst.ell.column_indices.values.begin(),
                       cusp::less_value<size_t>(dst.ell.column_indices.values.size()));
    thrust::scatter_if(exec,
                       src.values.begin(), src.values.end(),
                       indices.begin(),
                       indices.begin(),
                       dst.ell.values.values.begin(),
                       cusp::less_value<size_t>(dst.ell.values.values.size()));
//// fused version appears to be slightly slower
//  thrust::scatter_if(thrust::make_zip_iterator(thrust::make_tuple(src.column_indices.begin(), src.values.begin())),
//                     thrust::make_zip_iterator(thrust::make_tuple(src.column_indices.end(),   src.values.end())),
//                     indices.begin(),
//                     indices.begin(),
//                     thrust::make_zip_iterator(thrust::make_tuple(dst.ell.column_indices.values.begin(), dst.ell.values.values.begin())),
//                     less_than<size_t>(dst.ell.column_indices.values.size()));
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

