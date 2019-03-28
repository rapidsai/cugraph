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

#include <cusp/copy.h>
#include <cusp/convert.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/sort.h>

#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/counting_iterator.h>
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

// Array2d format
template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void transpose(thrust::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A,
                     MatrixType2& At,
               cusp::array2d_format,
               cusp::array2d_format)
{
    typedef typename MatrixType1::orientation Orientation1;
    typedef typename MatrixType2::orientation Orientation2;

    At.resize(A.num_cols, A.num_rows);

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end(A.num_entries);

    // prefer coalesced writes to coalesced reads
    cusp::detail::transpose_index_functor    <size_t, Orientation1, Orientation2> func1(A.num_rows,  A.num_cols,  A.pitch);
    cusp::detail::logical_to_physical_functor<size_t, Orientation2>               func2(At.num_rows, At.num_cols, At.pitch);

    thrust::copy(exec,
                 thrust::make_permutation_iterator(A.values.begin(),  thrust::make_transform_iterator(begin, func1)),
                 thrust::make_permutation_iterator(A.values.begin(),  thrust::make_transform_iterator(end,   func1)),
                 thrust::make_permutation_iterator(At.values.begin(), thrust::make_transform_iterator(begin, func2)));
}

// COO format
template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void transpose(thrust::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A,
                     MatrixType2& At,
               cusp::coo_format,
               cusp::coo_format)
{
    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::copy(exec, A.row_indices,    At.column_indices);
    cusp::copy(exec, A.column_indices, At.row_indices);
    cusp::copy(exec, A.values,         At.values);

    At.sort_by_row();
}

// CSR format
template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void transpose(thrust::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A,
                     MatrixType2& At,
                     cusp::csr_format,
                     cusp::csr_format)
{
    typedef typename MatrixType2::index_type   IndexType2;

    cusp::detail::temporary_array<IndexType2, DerivedPolicy> At_row_indices(exec, A.column_indices);

    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::offsets_to_indices(exec, A.row_offsets, At.column_indices);

    cusp::copy(exec, A.values, At.values);

    cusp::sort_by_row(exec, At_row_indices, At.column_indices, At.values);

    cusp::indices_to_offsets(exec, At_row_indices, At.row_offsets);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename Format1,
          typename Format2>
void transpose(thrust::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A,
                     MatrixType2& At,
                     Format1,
                     Format2)
{
    typedef typename MatrixType1::const_coo_view_type              View;
    typedef typename cusp::detail::as_coo_type<MatrixType2>::type  CooType;

    View A_coo(A);
    CooType At_coo;

    transpose(thrust::detail::derived_cast(exec), A_coo, At_coo, coo_format(), coo_format());

    cusp::convert(exec, At_coo, At);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void transpose(thrust::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A,
                     MatrixType2& At)
{
    typedef typename MatrixType1::format Format1;
    typedef typename MatrixType2::format Format2;

    Format1 format1;
    Format2 format2;

    transpose(thrust::detail::derived_cast(exec), A, At, format1, format2);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

