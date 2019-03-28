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

/*! \file dia_matrix.h
 *  \brief Diagonal matrix format.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/detail/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/detail/utils.h>

namespace cusp
{

// Forward definitions
template <typename ArrayType1, typename ArrayType2, typename IndexType, typename ValueType, typename MemorySpace> class dia_matrix_view;

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief Diagonal (DIA) representation a sparse matrix
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p dia_matrix is a sparse matrix container that stores each nonzero in
 *  a dense array2d container according to the diagonal each nonzero resides,
 *  the diagonal index of an ( \p i , \p j ) entry is | \p i - \p j |. This storage format is
 *  applicable to small set of matrices with significant structure. Storing the
 *  underlying entries in a \p array2d container avoids additional overhead
 *  associated with row or column indices but requires the storage of invalid
 *  entries associated with incomplete diagonals.
 *
 *  \note The diagonal offsets should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p dia_matrix on the host with 3 diagonals (6 total nonzeros)
 *  and then copies the matrix to the device.
 *
 *  \code
 * // include dia_matrix header file
 * #include <cusp/dia_matrix.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *  // allocate storage for (4,3) matrix with 6 nonzeros in 3 diagonals
 *  cusp::dia_matrix<int,float,cusp::host_memory> A(4,3,6,3);
 *
 *  // initialize diagonal offsets
 *  A.diagonal_offsets[0] = -2;
 *  A.diagonal_offsets[1] =  0;
 *  A.diagonal_offsets[2] =  1;
 *
 *  // initialize diagonal values
 *
 *  // first diagonal
 *  A.values(0,2) =  0;  // outside matrix
 *  A.values(1,2) =  0;  // outside matrix
 *  A.values(2,0) = 40;
 *  A.values(3,0) = 60;
 *
 *  // second diagonal
 *  A.values(0,1) = 10;
 *  A.values(1,1) =  0;
 *  A.values(2,1) = 50;
 *  A.values(3,1) = 50;  // outside matrix
 *
 *  // third diagonal
 *  A.values(0,2) = 20;
 *  A.values(1,2) = 30;
 *  A.values(2,2) =  0;  // outside matrix
 *  A.values(3,2) =  0;  // outside matrix
 *
 *  // A now represents the following matrix
 *  //    [10 20  0]
 *  //    [ 0  0 30]
 *  //    [40  0 50]
 *  //    [ 0 60  0]
 *
 *  // copy to the device
 *  cusp::dia_matrix<int,float,cusp::device_memory> B = A;
 *
 *  // print the constructed dia_matrix
 *  cusp::print(B);
 * }
 *  \endcode
 *
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class dia_matrix : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::dia_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::dia_format> Parent;

public:
    // TODO statically assert is_signed<IndexType>

    /*! \cond */
    typedef cusp::array1d<IndexType, MemorySpace>                     diagonal_offsets_array_type;
    typedef cusp::array2d<ValueType, MemorySpace, cusp::column_major> values_array_type;

    typedef cusp::dia_matrix<IndexType, ValueType, MemorySpace>       container;

    typedef cusp::dia_matrix_view<
            typename diagonal_offsets_array_type::view,
            typename values_array_type::view,
            IndexType, ValueType, MemorySpace> view;

    typedef cusp::dia_matrix_view<
            typename diagonal_offsets_array_type::const_view,
            typename values_array_type::const_view,
            IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<container,cusp::dia_format>::view              coo_view_type;
    // TODO : Why does GCC 4.4 fail using const type? Is it necessary?
    typedef typename cusp::detail::coo_view_type<container /*const*/,cusp::dia_format>::view        const_coo_view_type;

    template<typename MemorySpace2>
    struct rebind
    {
        typedef cusp::dia_matrix<IndexType, ValueType, MemorySpace2> type;
    };
    /*! \endcond */

    /*! Storage for the diagonal offsets.
     */
    diagonal_offsets_array_type diagonal_offsets;

    /*! Storage for the nonzero entries of the DIA data structure.
     */
    values_array_type values;

    /**
     * Construct an empty \p dia_matrix_view.
     */
    dia_matrix(void) {}

    /*! Construct a \p dia_matrix with a specific shape, number of nonzero entries,
     *  and number of occupied diagonals.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_diagonals Number of occupied diagonals.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    dia_matrix(const size_t num_rows, const size_t num_cols, const size_t num_entries,
               const size_t num_diagonals, const size_t alignment = 32);

    /*! Construct a \p dia_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    dia_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_diagonals Number of occupied diagonals.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_diagonals);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_diagonals Number of occupied diagonals.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_diagonals, const size_t alignment);

    /*! Swap the contents of two \p dia_matrix objects.
     *
     *  \param matrix Another \p dia_matrix with the same IndexType and ValueType.
     */
    void swap(dia_matrix& matrix);

    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    dia_matrix& operator=(const MatrixType& matrix);
}; // class dia_matrix
/*! \}
 */

/*! \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief View of a \p dia_matrix
 *
 * \tparam ArrayType1 Type of \c diagonal_offsets
 * \tparam ArrayType2 Type of \c values array view
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p dia_matrix_view is a sparse matrix view of a matrix in DIA format
 *  constructed from existing data or iterators. All entries in the \p dia_matrix are
 *  sorted according to row indices and internally within each row sorted by
 *  column indices.
 *
 *  \note The diagonal offsets should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p dia_matrix_view on the host with 3 diagonals and 6 nonzeros.
 *
 *  \code
 * // include dia_matrix header file
 * #include <cusp/dia_matrix.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    typedef cusp::array1d<int,cusp::host_memory> IndexArray;
 *    typedef cusp::array2d<float,cusp::host_memory,cusp::column_major> ValueArray;
 *
 *    typedef typename IndexArray::view IndexArrayView;
 *    typedef typename ValueArray::view ValueArrayView;
 *
 *    // initialize rows, columns, and values
 *    IndexArray diagonal_offsets(3);
 *    ValueArray values(4,3);
 *
 *    // initialize diagonal offsets
 *    diagonal_offsets[0] = -2;
 *    diagonal_offsets[1] =  0;
 *    diagonal_offsets[2] =  1;
 *
 *    // initialize diagonal values
 *
 *    // first diagonal
 *    values(0,2) =  0;  // outside matrix
 *    values(1,2) =  0;  // outside matrix
 *    values(2,0) = 40;
 *    values(3,0) = 60;
 *
 *    // second diagonal
 *    values(0,1) = 10;
 *    values(1,1) =  0;
 *    values(2,1) = 50;
 *    values(3,1) = 50;  // outside matrix
 *
 *    // third diagonal
 *    values(0,2) = 20;
 *    values(1,2) = 30;
 *    values(2,2) =  0;  // outside matrix
 *    values(3,2) =  0;  // outside matrix
 *
 *    // allocate storage for (4,3) matrix with 6 nonzeros
 *    cusp::dia_matrix_view<IndexArrayView,ValueArrayView> A(
 *    4,3,6,
 *    cusp::make_array1d_view(diagonal_offsets),
 *    cusp::make_array2d_view(values));
 *
 *    // A now represents the following matrix
 *    //    [10 20  0]
 *    //    [ 0  0 30]
 *    //    [40  0 50]
 *    //    [ 0 60  0]
 *
 *    // print the constructed coo_matrix
 *    cusp::print(A);
 *
 *    // change first entry in values array
 *    values(0,1) = -1;
 *
 *    // print the updated matrix view
 *    cusp::print(A);
 * }
 *  \endcode
 */
template <typename ArrayType1,
         typename ArrayType2,
         typename IndexType   = typename ArrayType1::value_type,
         typename ValueType   = typename ArrayType2::value_type,
         typename MemorySpace = typename cusp::minimum_space<typename ArrayType1::memory_space, typename ArrayType2::memory_space>::type >
class dia_matrix_view : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::dia_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::dia_format> Parent;

public:
    /*! \cond */
    typedef ArrayType1 diagonal_offsets_array_type;
    typedef ArrayType2 values_array_type;
    typedef cusp::dia_matrix<IndexType, ValueType, MemorySpace> container;
    typedef cusp::dia_matrix_view<ArrayType1, ArrayType2, IndexType, ValueType, MemorySpace> view;
    typedef cusp::dia_matrix_view<ArrayType1, ArrayType2, IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<view,cusp::dia_format>::view       coo_view_type;
    typedef typename cusp::detail::coo_view_type<view const,cusp::dia_format>::view const_coo_view_type;
    /*! \endcond */

    /**
     * View of the diagonal_offsets of the DIA data structure.
     */
    diagonal_offsets_array_type diagonal_offsets;

    /**
     * View for the nonzero entries of the DIA data structure.
     */
    values_array_type values;

    /**
     * Construct an empty \p dia_matrix_view.
     */
    dia_matrix_view(void) {}

    /*! Construct a \p dia_matrix_view with a specific shape and number of nonzero entries
     *  from existing arrays denoting the diagonal offsets and values.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param diagonal_offsets Array containing the diagonal offsets.
     *  \param values Array containing the values.
     */
    template <typename OtherArrayType1, typename OtherArrayType2>
    dia_matrix_view(size_t num_rows, size_t num_cols, size_t num_entries,
                    OtherArrayType1& diagonal_offsets, OtherArrayType2& values)
        : Parent(num_rows, num_cols, num_entries),
          diagonal_offsets(diagonal_offsets),
          values(values) {}

    /*! Construct a \p dia_matrix_view with a specific shape and number of nonzero entries
     *  from existing const arrays denoting the diagonal offsets and values.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param diagonal_offsets Array containing the diagonal offsets.
     *  \param values Array containing the values.
     */
    template <typename OtherArrayType1, typename OtherArrayType2>
    dia_matrix_view(size_t num_rows, size_t num_cols, size_t num_entries,
                    const OtherArrayType1& diagonal_offsets, const OtherArrayType2& values)
        : Parent(num_rows, num_cols, num_entries),
          diagonal_offsets(diagonal_offsets),
          values(values) {}

    /*! Construct a \p dia_matrix_view from a existing \p dia_matrix.
     *
     *  \param matrix \p dia_matrix used to create view.
     */
    dia_matrix_view(dia_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          diagonal_offsets(make_array1d_view(matrix.diagonal_offsets)),
          values(make_array2d_view(matrix.values)) {}

    /*! Construct a \p dia_matrix_view from a existing const \p dia_matrix.
     *
     *  \param matrix \p dia_matrix used to create view.
     */
    dia_matrix_view(const dia_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          diagonal_offsets(make_array1d_view(matrix.diagonal_offsets)),
          values(make_array2d_view(matrix.values)) {}

    /*! Construct a \p dia_matrix_view from a existing \p dia_matrix_view.
     *
     *  \param matrix \p dia_matrix_view used to create view.
     */
    dia_matrix_view(dia_matrix_view& matrix)
        : Parent(matrix),
          diagonal_offsets(matrix.diagonal_offsets),
          values(matrix.values) {}

    /*! Construct a \p dia_matrix_view from a existing const \p dia_matrix_view.
     *
     *  \param matrix \p dia_matrix_view used to create view.
     */
    dia_matrix_view(const dia_matrix_view& matrix)
        : Parent(matrix),
          diagonal_offsets(matrix.diagonal_offsets),
          values(matrix.values) {}

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_diagonals Number of occupied diagonals.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_diagonals);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_diagonals Number of occupied diagonals.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_diagonals, const size_t alignment);
}; // class dia_matrix_view

/* Convenience functions */

/**
 *  This is a convenience function for generating an \p dia_matrix_view
 *  using individual arrays
 *  \tparam ArrayType1 diagonal offsets array type
 *  \tparam ArrayType2 values array type
 *
 *  \param num_rows Number of rows.
 *  \param num_cols Number of columns.
 *  \param num_entries Number of nonzero matrix entries.
 *  \param diagonal_offsets Array containing the diagonal offsets.
 *  \param values Array containing the values.
 *
 *  \return \p dia_matrix_view constructed using input arrays
 */
template <typename ArrayType1,
         typename ArrayType2>
dia_matrix_view<ArrayType1,ArrayType2>
make_dia_matrix_view(size_t num_rows,
                     size_t num_cols,
                     size_t num_entries,
                     ArrayType1 diagonal_offsets,
                     ArrayType2 values)
{
    dia_matrix_view<ArrayType1,ArrayType2>
           view(num_rows, num_cols, num_entries, diagonal_offsets, values);

    return view;
}

/**
 *  This is a convenience function for generating an \p dia_matrix_view
 *  using individual arrays with explicit index, value, and memory space
 *  annotations.
 *
 *  \tparam ArrayType1 diagonal offsets array type
 *  \tparam ArrayType2 values array type
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p dia_matrix_view matrix to copy.
 *
 *  \return \p dia_matrix_view constructed using input arrays.
 */
template <typename ArrayType1,
         typename ArrayType2,
         typename IndexType,
         typename ValueType,
         typename MemorySpace>
dia_matrix_view<ArrayType1,ArrayType2,IndexType,ValueType,MemorySpace>
make_dia_matrix_view(const dia_matrix_view<ArrayType1,ArrayType2,IndexType,ValueType,MemorySpace>& m)
{
    return dia_matrix_view<ArrayType1,ArrayType2,IndexType,ValueType,MemorySpace>(m);
}

/**
 *  This is a convenience function for generating an \p dia_matrix_view
 *  using an existing \p dia_matrix.
 *
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p dia_matrix matrix to copy.
 *
 *  \return \p dia_matrix_view constructed using input arrays.
 */
template <typename IndexType, typename ValueType, class MemorySpace>
typename dia_matrix<IndexType,ValueType,MemorySpace>::view
make_dia_matrix_view(dia_matrix<IndexType,ValueType,MemorySpace>& m)
{
    return make_dia_matrix_view
           (m.num_rows, m.num_cols, m.num_entries,
            cusp::make_array1d_view(m.diagonal_offsets),
            cusp::make_array2d_view(m.values));
}

/**
 *  This is a convenience function for generating an const \p dia_matrix_view
 *  using an existing \p coo_matrix.
 *
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p dia_matrix matrix to copy.
 *
 *  \return \p dia_matrix_view constructed using input arrays.
 */
template <typename IndexType, typename ValueType, class MemorySpace>
typename dia_matrix<IndexType,ValueType,MemorySpace>::const_view
make_dia_matrix_view(const dia_matrix<IndexType,ValueType,MemorySpace>& m)
{
    return make_dia_matrix_view
           (m.num_rows, m.num_cols, m.num_entries,
            cusp::make_array1d_view(m.diagonal_offsets),
            cusp::make_array2d_view(m.values));
}

/*! \} // end Views
 */

} // end namespace cusp

#include <cusp/detail/dia_matrix.inl>
