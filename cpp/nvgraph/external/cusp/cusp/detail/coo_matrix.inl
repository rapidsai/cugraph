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

#include <cusp/array1d.h>
#include <cusp/convert.h>
#include <cusp/format_utils.h>
#include <cusp/functional.h>
#include <cusp/sort.h>

#include <cusp/iterator/join_iterator.h>

#include <cusp/detail/array2d_format_utils.h>

#include <thrust/copy.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
coo_matrix<IndexType,ValueType,MemorySpace>
::coo_matrix(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
}

////////////////////////////////
// Container Member Functions //
////////////////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries)
{
    Parent::resize(num_rows, num_cols, num_entries);
    row_indices.resize(num_entries);
    column_indices.resize(num_entries);
    values.resize(num_entries);
}

template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::swap(coo_matrix& matrix)
{
    Parent::swap(matrix);
    row_indices.swap(matrix.row_indices);
    column_indices.swap(matrix.column_indices);
    values.swap(matrix.values);
}

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
coo_matrix<IndexType,ValueType,MemorySpace>&
coo_matrix<IndexType,ValueType,MemorySpace>
::operator=(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);

    return *this;
}

// sort matrix elements by row index
template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::sort_by_row(void)
{
    cusp::sort_by_row(row_indices, column_indices, values, size_t(0), Parent::num_rows);
}

// sort matrix elements by row index
template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::sort_by_row_and_column(void)
{
    cusp::sort_by_row_and_column(row_indices, column_indices, values, size_t(0), Parent::num_rows, size_t(0), Parent::num_cols);
}

// determine whether matrix elements are sorted by row index
template <typename IndexType, typename ValueType, class MemorySpace>
bool
coo_matrix<IndexType,ValueType,MemorySpace>
::is_sorted_by_row(void)
{
    return thrust::is_sorted(row_indices.begin(), row_indices.end());
}

// determine whether matrix elements are sorted by row and column index
template <typename IndexType, typename ValueType, class MemorySpace>
bool
coo_matrix<IndexType,ValueType,MemorySpace>
::is_sorted_by_row_and_column(void)
{
    return thrust::is_sorted
           (thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   column_indices.end())));
}

///////////////////////
// View Constructors //
///////////////////////

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::coo_matrix_view(MatrixType& matrix)
{
    construct_from(matrix, typename MatrixType::format());
}

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::coo_matrix_view(const MatrixType& matrix)
{
    construct_from(const_cast<MatrixType&>(matrix), typename MatrixType::format());
}

///////////////////////////
// View Member Functions //
///////////////////////////

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
void coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::construct_from(MatrixType& matrix, cusp::csr_format)
{
    Parent::resize(matrix.num_rows, matrix.num_cols, matrix.num_entries);
    row_indices.resize(matrix.num_entries);
    cusp::offsets_to_indices(matrix.row_offsets, row_indices);

    column_indices = column_indices_array_type(matrix.column_indices.begin(), matrix.column_indices.end());
    values         = values_array_type(matrix.values.begin(), matrix.values.end());
}

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
void coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::construct_from(MatrixType& matrix, cusp::dia_format)
{
    typedef cusp::detail::coo_view_type<MatrixType>          dia_view_type;

    typedef typename dia_view_type::CountingIterator         CountingIterator;
    typedef typename dia_view_type::PermFunctor              PermFunctor;
    typedef typename dia_view_type::OffsetsPermIterator      OffsetsPermIterator;
    typedef typename dia_view_type::ModulusIterator          ModulusIterator;
    typedef typename dia_view_type::ZipIterator              ZipIterator;

    typedef typename dia_view_type::PermIndexIterator        PermIndexIterator;
    typedef typename dia_view_type::RowIndexIterator         RowIndexIterator;
    typedef typename dia_view_type::ColumnIndexIterator      ColumnIndexIterator;
    typedef typename dia_view_type::PermValueIterator        PermValueIterator;

    typedef typename dia_view_type::RowPermIterator          RowPermIterator;
    typedef typename dia_view_type::ColumnPermIterator       ColumnPermIterator;
    typedef typename dia_view_type::ValuePermIterator        ValuePermIterator;

    const size_t num_entries = matrix.values.num_entries;

    Parent::resize(matrix.num_rows, matrix.num_cols, matrix.num_entries);

    RowIndexIterator    row_indices_begin(CountingIterator(0), cusp::divide_value<IndexType>(matrix.values.num_cols));
    ModulusIterator     gather_indices_begin(CountingIterator(0), cusp::modulus_value<IndexType>(matrix.values.num_cols));
    OffsetsPermIterator offsets_begin(matrix.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator         offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));

    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, cusp::sum_pair_functor<IndexType>());
    PermIndexIterator   perm_indices_begin(CountingIterator(0),   PermFunctor(matrix.values.num_rows, matrix.values.num_cols, matrix.values.pitch));
    PermValueIterator   perm_values_begin(matrix.values.values.begin(), perm_indices_begin);

    indices.resize(num_entries);
    thrust::remove_copy_if(CountingIterator(0), CountingIterator(num_entries), perm_values_begin, indices.begin(), thrust::placeholders::_1 == ValueType(0));

    RowPermIterator           rows_iter(row_indices_begin,    indices.begin());
    ColumnPermIterator        cols_iter(column_indices_begin, indices.begin());
    ValuePermIterator         vals_iter(perm_values_begin,    indices.begin());

    row_indices_array_type    rows_array(rows_iter, rows_iter + matrix.num_entries);
    column_indices_array_type cols_array(cols_iter, cols_iter + matrix.num_entries);
    values_array_type         vals_array(vals_iter, vals_iter + matrix.num_entries);

    row_indices    = rows_array;
    column_indices = cols_array;
    values         = vals_array;
}

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
void coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::construct_from(MatrixType& matrix, cusp::ell_format)
{
    typedef cusp::detail::coo_view_type<MatrixType>          ell_view_type;

    typedef typename ell_view_type::CountingIterator         CountingIterator;
    typedef typename ell_view_type::PermFunctor              PermFunctor;
    typedef typename ell_view_type::PermIndexIterator        PermIndexIterator;

    typedef typename ell_view_type::RowIndexIterator         RowIndexIterator;
    typedef typename ell_view_type::PermColumnIndexIterator  PermColumnIndexIterator;
    typedef typename ell_view_type::PermValueIterator        PermValueIterator;

    typedef typename ell_view_type::RowPermIterator          RowPermIterator;
    typedef typename ell_view_type::ColumnPermIterator       ColumnPermIterator;
    typedef typename ell_view_type::ValuePermIterator        ValuePermIterator;

    const int    X               = MatrixType::invalid_index;
    const size_t ell_num_entries = matrix.column_indices.num_entries;

    Parent::resize(matrix.num_rows, matrix.num_cols, matrix.num_entries);

    PermIndexIterator       perm_indices_begin(CountingIterator(0), PermFunctor(matrix.values.num_rows, matrix.values.num_cols, matrix.values.pitch));

    RowIndexIterator        row_indices_begin(CountingIterator(0), cusp::divide_value<IndexType>(matrix.values.num_cols));
    PermColumnIndexIterator perm_column_indices_begin(matrix.column_indices.values.begin(), perm_indices_begin);
    PermValueIterator       perm_values_begin(matrix.values.values.begin(), perm_indices_begin);

    indices.resize(ell_num_entries);
    thrust::remove_copy_if(CountingIterator(0), CountingIterator(ell_num_entries), perm_column_indices_begin, indices.begin(), thrust::placeholders::_1 == X);

    RowPermIterator           rows_iter(row_indices_begin,         indices.begin());
    ColumnPermIterator        cols_iter(perm_column_indices_begin, indices.begin());
    ValuePermIterator         vals_iter(perm_values_begin,         indices.begin());

    row_indices_array_type    rows_array(rows_iter, rows_iter + matrix.num_entries);
    column_indices_array_type cols_array(cols_iter, cols_iter + matrix.num_entries);
    values_array_type         vals_array(vals_iter, vals_iter + matrix.num_entries);

    row_indices    = rows_array;
    column_indices = cols_array;
    values         = vals_array;
}

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
void coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::construct_from(MatrixType& matrix, cusp::hyb_format)
{
    using namespace cusp::detail;

    typedef cusp::detail::coo_view_type<MatrixType>                  hyb_view_type;

    typedef typename hyb_view_type::ell_view_type::CountingIterator  CountingIterator;
    typedef typename hyb_view_type::ell_view_type::PermFunctor       PermFunctor;

    typedef typename hyb_view_type::EllPermIndexIterator             PermIndexIterator;
    typedef typename hyb_view_type::EllRowIndexIterator              RowIndexIterator;
    typedef typename hyb_view_type::EllColumnIndexIterator           PermColumnIndexIterator;
    typedef typename hyb_view_type::EllValueIterator                 PermValueIterator;

    typedef typename hyb_view_type::JoinRowIterator::iterator        JoinRowIterator;
    typedef typename hyb_view_type::JoinColumnIterator::iterator     JoinColumnIterator;
    typedef typename hyb_view_type::JoinValueIterator::iterator      JoinValueIterator;

    const int    X               = MatrixType::ell_matrix_type::invalid_index;
    const size_t ell_num_entries = matrix.ell.column_indices.num_entries;
    const size_t coo_num_entries = matrix.coo.num_entries;
    const size_t total           = ell_num_entries + coo_num_entries;

    Parent::resize(matrix.num_rows, matrix.num_cols, matrix.num_entries);

    PermIndexIterator       perm_indices_begin(CountingIterator(0), PermFunctor(matrix.ell.values.num_rows, matrix.ell.values.num_cols, matrix.ell.values.pitch));
    RowIndexIterator        row_indices_begin(CountingIterator(0), cusp::divide_value<IndexType>(matrix.ell.values.num_cols));
    PermColumnIndexIterator perm_column_indices_begin(matrix.ell.column_indices.values.begin(), perm_indices_begin);
    PermValueIterator       perm_values_begin(matrix.ell.values.values.begin(), perm_indices_begin);

    // TODO : Remove this WAR when Thrust v1.8.3 is released, related to issue #635
#if THRUST_VERSION > 100802
    RowIndexIterator temp_row_indices_begin(row_indices_begin);
    PermColumnIndexIterator temp_column_indices_begin(perm_column_indices_begin);
#else
    cusp::array1d<IndexType,MemorySpace> temp_row_indices(row_indices_begin, row_indices_begin + ell_num_entries);
    cusp::array1d<IndexType,MemorySpace> temp_column_indices(perm_column_indices_begin, perm_column_indices_begin + ell_num_entries);

    typename cusp::array1d<IndexType,MemorySpace>::iterator temp_row_indices_begin(temp_row_indices.begin());
    typename cusp::array1d<IndexType,MemorySpace>::iterator temp_column_indices_begin(temp_column_indices.begin());
#endif

    indices.resize(total);

    if(coo_num_entries > 0)
    {
        // thrust::merge_by_key(thrust::make_zip_iterator(thrust::make_tuple(temp_row_indices_begin, temp_column_indices_begin)),
        //                      thrust::make_zip_iterator(thrust::make_tuple(temp_row_indices_begin, temp_column_indices_begin)) + ell_num_entries,
        //                      thrust::make_zip_iterator(thrust::make_tuple(matrix.coo.row_indices.begin(), matrix.coo.column_indices.begin())),
        //                      thrust::make_zip_iterator(thrust::make_tuple(matrix.coo.row_indices.begin(), matrix.coo.column_indices.begin())) + coo_num_entries,
        //                      thrust::counting_iterator<IndexType>(0),
        //                      thrust::counting_iterator<IndexType>(ell_num_entries),
        //                      thrust::make_discard_iterator(),
        //                      indices.begin(),
        //                      cusp::detail::coo_tuple_comp_functor<IndexType>());

        // filter out invalid indices
        JoinColumnIterator cols_iter = cusp::make_join_iterator(ell_num_entries, coo_num_entries, perm_column_indices_begin, matrix.coo.column_indices.begin(), indices.begin());
        thrust::remove_if(indices.begin(), indices.end(), cols_iter, thrust::placeholders::_1 == X);
    }
    else
    {
        thrust::remove_copy_if(CountingIterator(0), CountingIterator(ell_num_entries), temp_column_indices_begin, indices.begin(), thrust::placeholders::_1 == X);
    }

    JoinRowIterator    rows_iter = cusp::make_join_iterator(ell_num_entries, coo_num_entries, row_indices_begin,         matrix.coo.row_indices.begin(),    indices.begin());
    JoinColumnIterator cols_iter = cusp::make_join_iterator(ell_num_entries, coo_num_entries, perm_column_indices_begin, matrix.coo.column_indices.begin(), indices.begin());
    JoinValueIterator  vals_iter = cusp::make_join_iterator(ell_num_entries, coo_num_entries, perm_values_begin,         matrix.coo.values.begin(),         indices.begin());


    row_indices_array_type    rows_array(rows_iter, rows_iter + matrix.num_entries);
    column_indices_array_type cols_array(cols_iter, cols_iter + matrix.num_entries);
    values_array_type         vals_array(vals_iter, vals_iter + matrix.num_entries);

    row_indices    = rows_array;
    column_indices = cols_array;
    values         = vals_array;
}

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
void
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries)
{
    Parent::resize(num_rows, num_cols, num_entries);
    row_indices.resize(num_entries);
    column_indices.resize(num_entries);
    values.resize(num_entries);
}

// sort matrix elements by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
void
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::sort_by_row(void)
{
    cusp::sort_by_row(row_indices, column_indices, values, size_t(0), Parent::num_rows);
}

// sort matrix elements by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
void
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::sort_by_row_and_column(void)
{
    cusp::sort_by_row_and_column(row_indices, column_indices, values, size_t(0), Parent::num_rows, size_t(0), Parent::num_cols);
}

// determine whether matrix elements are sorted by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
bool
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::is_sorted_by_row(void)
{
    return thrust::is_sorted(row_indices.begin(), row_indices.end());
}

// determine whether matrix elements are sorted by row and column index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
bool
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::is_sorted_by_row_and_column(void)
{
    return thrust::is_sorted
           (thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   column_indices.end())));
}

} // end namespace cusp

