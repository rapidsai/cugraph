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
#include <cusp/ell_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/functional.h>
#include <cusp/sort.h>

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

#include <cassert>

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
             cusp::ell_format&,
             cusp::coo_format&)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;

    // define types used to programatically generate row_indices
    typedef thrust::counting_iterator<IndexType>                                                               IndexIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>, IndexIterator>                           RowIndexIterator;

    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>    PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, IndexIterator>                                             PermIndexIterator;
    typedef typename SourceType::column_indices_array_type::values_array_type::const_iterator                  IndicesIterator;
    typedef typename SourceType::values_array_type::values_array_type::const_iterator                          ValueIterator;
    typedef thrust::permutation_iterator<IndicesIterator, PermIndexIterator>                                   PermColumnIndicesIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                                     PermValueIterator;

    RowIndexIterator    row_indices_begin(IndexIterator(0),    cusp::divide_value<IndexType>(src.values.num_cols));
    PermIndexIterator   perm_indices_begin(IndexIterator(0),   PermFunctor(src.values.num_rows, src.values.num_cols, src.values.pitch));
    PermColumnIndicesIterator   perm_column_indices_begin(src.column_indices.values.begin(),  perm_indices_begin);
    PermValueIterator   perm_values_begin(src.values.values.begin(),  perm_indices_begin);

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    if(src.num_entries == 0) return;

    // copy valid entries to mixed COO/CSR format
    thrust::copy_if
     (exec,
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, perm_column_indices_begin, perm_values_begin)),
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, perm_column_indices_begin, perm_values_begin)) + src.values.num_entries,
      perm_values_begin,
      thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      thrust::placeholders::_1 != ValueType(0));
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::ell_format&,
        cusp::csr_format&)
{
    typedef typename DestinationType::index_type   IndexType;
    typedef typename DestinationType::value_type   ValueType;

    // define types used to programatically generate row_indices
    typedef thrust::counting_iterator<IndexType>                                                               IndexIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>, IndexIterator>                           RowIndexIterator;

    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>    PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, IndexIterator>                                             PermIndexIterator;
    typedef typename SourceType::column_indices_array_type::values_array_type::const_iterator                  IndicesIterator;
    typedef typename SourceType::values_array_type::values_array_type::const_iterator                          ValueIterator;
    typedef thrust::permutation_iterator<IndicesIterator, PermIndexIterator>                                   PermColumnIndicesIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                                     PermValueIterator;

    RowIndexIterator    row_indices_begin(IndexIterator(0),    cusp::divide_value<IndexType>(src.values.num_cols));
    PermIndexIterator   perm_indices_begin(IndexIterator(0),   PermFunctor(src.values.num_rows, src.values.num_cols, src.values.pitch));
    PermColumnIndicesIterator   perm_column_indices_begin(src.column_indices.values.begin(),  perm_indices_begin);
    PermValueIterator   perm_values_begin(src.values.values.begin(),  perm_indices_begin);

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    if(src.num_entries == 0) return;

    // create temporary row_indices array to capture valid ELL row indices
    cusp::detail::temporary_array<IndexType, DerivedPolicy> row_indices(exec, src.num_entries);

    // copy valid entries to mixed COO/CSR format
    thrust::copy_if
     (exec,
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, perm_column_indices_begin, perm_values_begin)),
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, perm_column_indices_begin, perm_values_begin)) + src.values.num_entries,
      perm_values_begin,
      thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      thrust::placeholders::_1 != ValueType(0));

    // convert COO row_indices to CSR row_offsets
    cusp::indices_to_offsets(exec, row_indices, dst.row_offsets);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::ell_format&,
        cusp::hyb_format&)
{
    // just copy into ell part of destination
    dst.resize(src.num_rows, src.num_cols,
               src.num_entries, 0,
               src.column_indices.num_cols);

    if(src.num_entries == 0) return;

    cusp::copy(exec, src, dst.ell);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
