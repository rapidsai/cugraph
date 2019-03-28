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

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/copy.h>

#include <cusp/detail/format.h>

#include <thrust/count.h>
#include <thrust/copy.h>
#include <cusp/detail/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::array2d_format&,
        cusp::array1d_format&)
{
    typedef cusp::array2d_view<typename DestinationType::view, cusp::row_major>    RowView;
    typedef cusp::array2d_view<typename DestinationType::view, cusp::column_major> ColView;

    if (src.num_rows == 0 && src.num_cols == 0)
    {
        dst.resize(0);
    }
    else if (src.num_cols == 1)
    {
        dst.resize(src.num_rows);

        // interpret dst as a Nx1 column matrix and copy from src
        ColView view(src.num_rows, 1, src.num_rows, cusp::make_array1d_view(dst));

        cusp::copy(exec, src, view);
    }
    else if (src.num_rows == 1)
    {
        dst.resize(src.num_cols);

        // interpret dst as a 1xN row matrix and copy from src
        RowView view(1, src.num_cols, src.num_cols, cusp::make_array1d_view(dst));

        cusp::copy(exec, src, view);
    }
    else
    {
        throw cusp::format_conversion_exception("array2d to array1d conversion is only defined for row or column vectors");
    }
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::array1d_format&,
        cusp::array2d_format&)
{
    size_t N = src.size();

    // interpret src as a Nx1 column matrix and copy to dst
    cusp::copy(exec, cusp::make_array2d_view
               (N, 1, N,
                cusp::make_array1d_view(src),
                cusp::column_major()),
               dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::array1d_format&,
        cusp::coo_format&)
{
    size_t N = src.size();

    dst.resize(N, 1, N);

    thrust::sequence(exec, dst.row_indices.begin(), dst.row_indices.end());
    thrust::fill(exec, dst.column_indices.begin(), dst.column_indices.end(), 0);
    cusp::copy(exec, src, dst.values);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::array2d_format&,
        cusp::coo_format&)
{
    using namespace thrust::placeholders;

    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;

    // define types used to programatically generate row_indices
    typedef typename SourceType::orientation                                                            Orientation;
    typedef typename SourceType::values_array_type::const_iterator                                      ValueIterator;
    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, Orientation>    PermFunctor;
    typedef thrust::counting_iterator<IndexType>                                                        IndexIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>,  IndexIterator>                   RowIndexIterator;
    typedef thrust::transform_iterator<cusp::modulus_value<IndexType>, IndexIterator>                   ColumnIndexIterator;
    typedef thrust::transform_iterator<PermFunctor, IndexIterator>                                      PermIndexIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                              PermValueIterator;

    RowIndexIterator    row_indices_begin(IndexIterator(0),    cusp::divide_value<IndexType>(src.pitch));
    ColumnIndexIterator column_indices_begin(IndexIterator(0), cusp::modulus_value<IndexType>(src.pitch));
    PermIndexIterator   perm_indices_begin(IndexIterator(0),   PermFunctor(src.num_rows, src.num_cols, src.pitch));
    PermValueIterator   perm_values_begin(src.values.begin(),  perm_indices_begin);

    size_t num_coo_entries = thrust::count_if(exec, src.values.begin(), src.values.end(), _1 != ValueType(0));
    dst.resize(src.num_rows, src.num_cols, num_coo_entries);

    thrust::copy_if(exec,
                    thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, perm_values_begin)),
                    thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, perm_values_begin)) + src.num_entries,
                    perm_values_begin,
                    thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
                    _1 != ValueType(0));
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

