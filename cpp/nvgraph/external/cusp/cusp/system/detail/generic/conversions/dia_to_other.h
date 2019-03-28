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

#include <cusp/detail/format.h>

#include <cusp/copy.h>
#include <cusp/dia_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/functional.h>
#include <cusp/sort.h>

#include <cusp/blas/blas.h>

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
        cusp::dia_format&,
        cusp::coo_format&)
{
    typedef typename SourceType::index_type   IndexType;
    typedef typename SourceType::value_type   ValueType;

    // define types used to programatically generate row_indices
    typedef thrust::counting_iterator<IndexType>                                                               IndexIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>, IndexIterator>                           RowIndexIterator;

    // define types used to programatically generate column_indices
    typedef typename SourceType::diagonal_offsets_array_type::const_iterator                                   ConstElementIterator;
    typedef thrust::transform_iterator<cusp::modulus_value<IndexType>, IndexIterator>                          ModulusIterator;
    typedef thrust::permutation_iterator<ConstElementIterator,ModulusIterator>                                 OffsetsPermIterator;
    typedef thrust::tuple<OffsetsPermIterator, RowIndexIterator>                                               IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple>                                                                ZipIterator;
    typedef thrust::transform_iterator<cusp::sum_pair_functor<IndexType>, ZipIterator>                         ColumnIndexIterator;

    typedef typename SourceType::values_array_type::values_array_type::const_iterator                          ValueIterator;
    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>    PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, IndexIterator>                                             PermIndexIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                                     PermValueIterator;

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    if( src.num_entries == 0 ) return;

    RowIndexIterator row_indices_begin(IndexIterator(0), cusp::divide_value<IndexType>(src.values.num_cols));

    ModulusIterator gather_indices_begin(IndexIterator(0), cusp::modulus_value<IndexType>(src.values.num_cols));
    OffsetsPermIterator offsets_begin(src.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, cusp::sum_pair_functor<IndexType>());

    PermIndexIterator   perm_indices_begin(IndexIterator(0),   PermFunctor(src.values.num_rows, src.values.num_cols, src.values.pitch));
    PermValueIterator   perm_values_begin(src.values.values.begin(),  perm_indices_begin);

    thrust::copy_if
     (exec,
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, perm_values_begin)),
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, perm_values_begin)) + src.values.num_entries,
      perm_values_begin,
      thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      thrust::placeholders::_1 != ValueType(0));
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::dia_format&,
        cusp::csr_format&)
{
    typedef typename SourceType::index_type   IndexType;
    typedef typename SourceType::value_type   ValueType;

    // define types used to programatically generate row_indices
    typedef thrust::counting_iterator<IndexType>                                                 IndexIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>, IndexIterator>             RowIndexIterator;

    // define types used to programatically generate column_indices
    typedef typename SourceType::diagonal_offsets_array_type::const_iterator                     ConstElementIterator;
    typedef thrust::transform_iterator<cusp::modulus_value<IndexType>, IndexIterator>            ModulusIterator;
    typedef thrust::permutation_iterator<ConstElementIterator,ModulusIterator>                   OffsetsPermIterator;
    typedef thrust::tuple<OffsetsPermIterator, RowIndexIterator>                                 IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple>                                                  ZipIterator;
    typedef thrust::transform_iterator<cusp::sum_pair_functor<IndexType>, ZipIterator>           ColumnIndexIterator;

    typedef typename SourceType::values_array_type::values_array_type::const_iterator            ValueIterator;
    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>    PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, IndexIterator>                               PermIndexIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                       PermValueIterator;

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    if( src.num_entries == 0 ) return;

    RowIndexIterator row_indices_begin(IndexIterator(0), cusp::divide_value<IndexType>(src.values.num_cols));

    ModulusIterator gather_indices_begin(IndexIterator(0), cusp::modulus_value<IndexType>(src.values.num_cols));
    OffsetsPermIterator offsets_begin(src.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, cusp::sum_pair_functor<IndexType>());

    PermIndexIterator   perm_indices_begin(IndexIterator(0), PermFunctor(src.values.num_rows, src.values.num_cols, src.values.pitch));
    PermValueIterator   perm_values_begin(src.values.values.begin(), perm_indices_begin);

    cusp::detail::temporary_array<IndexType, DerivedPolicy> row_indices(exec, src.num_entries);

    thrust::copy_if
     (exec,
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, perm_values_begin)),
      thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, perm_values_begin)) + src.values.num_entries,
      perm_values_begin,
      thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      thrust::placeholders::_1 != ValueType(0));

    cusp::indices_to_offsets(exec, row_indices, dst.row_offsets);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::dia_format&,
        cusp::ell_format&)
{
    typedef typename SourceType::index_type   IndexType;
    typedef typename SourceType::value_type   ValueType;
    typedef typename SourceType::memory_space MemorySpace;

    // define types used to programatically generate row_indices
    typedef thrust::counting_iterator<IndexType>                                        IndexIterator;
    typedef thrust::transform_iterator<cusp::modulus_value<IndexType>, IndexIterator>   RowIndexIterator;

    // define types used to programatically generate column_indices
    typedef typename cusp::array1d<IndexType,MemorySpace>::const_iterator               ConstElementIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>, IndexIterator>    DivideIterator;
    typedef thrust::permutation_iterator<ConstElementIterator,DivideIterator>           OffsetsPermIterator;
    typedef thrust::tuple<OffsetsPermIterator, RowIndexIterator>                        IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple>                                         ZipIterator;
    typedef thrust::transform_iterator<cusp::sum_pair_functor<IndexType>, ZipIterator>  ColumnIndexIterator;

    if( src.num_entries == 0 )
    {
      dst.resize(src.num_rows, src.num_cols, src.num_entries, 0);
      return;
    }

    const IndexType pitch = src.values.pitch;
    const size_t num_diagonals = src.diagonal_offsets.size();

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, src.values.pitch);

    RowIndexIterator row_indices_begin(IndexIterator(0), cusp::modulus_value<IndexType>(pitch));

    DivideIterator gather_indices_begin(IndexIterator(0), cusp::divide_value<IndexType>(pitch));
    OffsetsPermIterator offsets_begin(src.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, cusp::sum_pair_functor<IndexType>());

    thrust::replace_copy_if(exec,
                            column_indices_begin,
                            column_indices_begin + src.values.num_entries,
                            src.values.values.begin(),
                            dst.column_indices.values.begin(),
                            thrust::placeholders::_1 == ValueType(0), -1);

    thrust::copy(exec, src.values.values.begin(), src.values.values.end(), dst.values.values.begin());

}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
