/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <cusp/detail/config.h>
#include <cusp/detail/array2d_format_utils.h>
#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>
#include <cusp/detail/utils.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/functional.h>

#include <thrust/reduce.h>

#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename T>
struct valid_index_functor : public thrust::unary_function<T,T>
{
    const T size;

    valid_index_functor(const T size)
        : size(size) {}

    __host__ __device__
    T operator()(const T col)
    {
        return (col < 0 || col >= size) ? 0 : col;
    }
};

template <typename DerivedPolicy,
          typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(thrust::execution_policy<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1  combine,
                      BinaryFunction2  reduce,
                      cusp::dia_format,
                      cusp::array1d_format,
                      cusp::array1d_format,
                      cusp::array1d_format)
{
    typedef typename LinearOperator::index_type   IndexType;
    typedef typename LinearOperator::value_type   ValueType;

    // define types used to programatically generate row_indices
    typedef thrust::counting_iterator<IndexType>                                                 IndexIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>, IndexIterator>             RowIndexIterator;

    // define types used to programatically generate column_indices
    typedef typename LinearOperator::diagonal_offsets_array_type::const_iterator                 ConstElementIterator;
    typedef thrust::transform_iterator<cusp::modulus_value<IndexType>, IndexIterator>            ModulusIterator;
    typedef thrust::permutation_iterator<ConstElementIterator,ModulusIterator>                   OffsetsPermIterator;
    typedef thrust::tuple<OffsetsPermIterator, RowIndexIterator>                                 IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple>                                                  ZipIterator;
    typedef thrust::transform_iterator<cusp::sum_pair_functor<IndexType>, ZipIterator>          ColumnIndexIterator;

    typedef typename LinearOperator::values_array_type::values_array_type::const_iterator        ValueIterator;
    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>    PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, IndexIterator>                               PermIndexIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                       PermValueIterator;

    if(A.num_entries == 0)
    {
        thrust::copy(exec, y.begin(), y.end(), z.begin());
        return;
    }

    RowIndexIterator row_indices_begin(IndexIterator(0), cusp::divide_value<IndexType>(A.values.num_cols));

    ModulusIterator gather_indices_begin(IndexIterator(0), cusp::modulus_value<IndexType>(A.values.num_cols));
    OffsetsPermIterator offsets_begin(A.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, cusp::sum_pair_functor<IndexType>());

    PermIndexIterator   perm_indices_begin(IndexIterator(0),   PermFunctor(A.values.num_rows, A.values.num_cols, A.values.pitch));
    PermValueIterator   perm_values_begin(A.values.values.begin(),  perm_indices_begin);

    typedef typename Vector3::value_type ValueType3;
    cusp::detail::temporary_array<ValueType3, DerivedPolicy> vals(exec, A.num_rows);
    thrust::fill(exec, vals.begin(), vals.end(), ValueType(0));

    thrust::reduce_by_key(exec,
                          row_indices_begin, row_indices_begin + A.values.num_entries,
                          thrust::make_transform_iterator(
                              thrust::make_zip_iterator(
                                  thrust::make_tuple(
                                      perm_values_begin,
                                      thrust::make_permutation_iterator(x.begin(),
                                              thrust::make_transform_iterator(column_indices_begin,
                                                      valid_index_functor<IndexType>(A.num_cols)))
                                  )),
                              cusp::detail::combine_tuple_base_functor<BinaryFunction1>()),
                          thrust::make_discard_iterator(),
                          vals.begin(),
                          thrust::equal_to<IndexType>(),
                          reduce);

    thrust::transform(exec, vals.begin(), vals.end(), y.begin(), z.begin(), reduce);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(thrust::execution_policy<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1  combine,
                      BinaryFunction2  reduce,
                      cusp::ell_format,
                      cusp::array1d_format,
                      cusp::array1d_format,
                      cusp::array1d_format)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::value_type ValueType;

    typedef cusp::detail::logical_to_other_physical_functor<IndexType,cusp::row_major,cusp::column_major> LogicalFunctor;

    // define types used to programatically generate row_indices
    typedef thrust::counting_iterator<IndexType>                                            IndexIterator;
    typedef thrust::transform_iterator<cusp::divide_value<IndexType>, IndexIterator>        RowIndexIterator;
    typedef thrust::transform_iterator<LogicalFunctor, IndexIterator>                       StrideIndexIterator;

    if(A.num_entries == 0)
    {
        thrust::copy(exec, y.begin(), y.end(), z.begin());
        return;
    }

    size_t num_entries = A.values.num_entries;
    size_t num_entries_per_row = A.column_indices.num_cols;
    LogicalFunctor logical_functor(A.values.num_rows, A.values.num_cols, A.values.pitch);

    RowIndexIterator row_indices_begin(IndexIterator(0), cusp::divide_value<IndexType>(num_entries_per_row));
    StrideIndexIterator stride_indices_begin(IndexIterator(0), logical_functor);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    thrust::reduce_by_key(exec,
                          row_indices_begin, row_indices_begin + num_entries,
                          thrust::make_permutation_iterator(
                              thrust::make_transform_iterator(
                                  thrust::make_zip_iterator(
                                      thrust::make_tuple(
                                          A.values.values.begin(),
                                          thrust::make_permutation_iterator(x.begin(),
                                              thrust::make_transform_iterator(A.column_indices.values.begin(),
                                                      valid_index_functor<IndexType>(A.num_cols))))),
                                  cusp::detail::combine_tuple_base_functor<BinaryFunction1>()),
                              stride_indices_begin),
                          thrust::make_discard_iterator(),
                          vals.begin(),
                          thrust::equal_to<IndexType>(),
                          reduce);

    thrust::transform(exec, vals.begin(), vals.end(), y.begin(), z.begin(), reduce);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(thrust::execution_policy<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1  combine,
                      BinaryFunction2  reduce,
                      cusp::coo_format,
                      cusp::array1d_format,
                      cusp::array1d_format,
                      cusp::array1d_format)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::value_type ValueType;

    typedef cusp::detail::temporary_array<IndexType, DerivedPolicy> IndexArray;
    typedef cusp::detail::temporary_array<ValueType, DerivedPolicy> ValueArray;

    thrust::copy(exec, y.begin(), y.end(), z.begin());

    if(A.num_entries == 0) return;

    IndexArray rows(exec, A.num_rows);
    ValueArray vals(exec, A.num_rows);

    typename IndexArray::iterator rows_end;
    typename ValueArray::iterator vals_end;

    thrust::tie(rows_end, vals_end) =
        thrust::reduce_by_key(exec,
                              A.row_indices.begin(), A.row_indices.end(),
                              thrust::make_transform_iterator(
                                  thrust::make_zip_iterator(
                                      thrust::make_tuple(
                                          A.values.begin(),
                                          thrust::make_permutation_iterator(x.begin(), A.column_indices.begin()))),
                                  cusp::detail::combine_tuple_base_functor<BinaryFunction1>()),
                              rows.begin(),
                              vals.begin(),
                              thrust::equal_to<IndexType>(),
                              reduce);

    int num_entries = rows_end - rows.begin();
    thrust::transform(exec,
                      vals.begin(),
                      vals.begin() + num_entries,
                      thrust::make_permutation_iterator(z.begin(), rows.begin()),
                      thrust::make_permutation_iterator(z.begin(), rows.begin()),
                      reduce);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(thrust::execution_policy<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1  combine,
                      BinaryFunction2  reduce,
                      cusp::csr_format,
                      cusp::array1d_format,
                      cusp::array1d_format,
                      cusp::array1d_format)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef cusp::detail::temporary_array<IndexType, DerivedPolicy> TempArray;

    typedef typename TempArray::view                                        RowView;
    typedef typename LinearOperator::column_indices_array_type::const_view  ColView;
    typedef typename LinearOperator::values_array_type::const_view          ValView;

    if(A.num_entries == 0)
    {
        thrust::copy(exec, y.begin(), y.end(), z.begin());
        return;
    }

    TempArray row_indices(exec, A.num_entries);
    cusp::offsets_to_indices(A.row_offsets, row_indices);

    cusp::coo_matrix_view<RowView,ColView,ValView> A_coo_view(A.num_rows, A.num_cols, A.num_entries,
                                                              cusp::make_array1d_view(row_indices),
                                                              cusp::make_array1d_view(A.column_indices),
                                                              cusp::make_array1d_view(A.values));

    cusp::generalized_spmv(exec, A_coo_view, x, y, z, combine, reduce);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(thrust::execution_policy<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce,
                      cusp::hyb_format,
                      cusp::array1d_format,
                      cusp::array1d_format,
                      cusp::array1d_format)
{
    typedef typename Vector3::value_type ValueType;

    cusp::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    cusp::generalized_spmv(exec, A.ell, x, y, vals, combine, reduce);
    cusp::generalized_spmv(exec, A.coo, x, vals, z, combine, reduce);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

