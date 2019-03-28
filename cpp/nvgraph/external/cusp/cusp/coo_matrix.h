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

/*! \file coo_matrix.h
 *  \brief Coordinate matrix format
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/detail/format.h>
#include <cusp/detail/matrix_base.h>

#include <cusp/array1d.h>

namespace cusp
{

/*! \cond */
// forward definition
template <typename ArrayType1, typename ArrayType2, typename ArrayType3,
          typename IndexType, typename ValueType, typename MemorySpace> class coo_matrix_view;
/*! \endcond */

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \brief Sparse matrix containers represented in COO, CSR, DIA, ELL, HYB, and
 *  Permutation
 *  \{
 */

/**
 * \brief Coordinate (COO) representation a sparse matrix
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p coo_matrix is a sparse matrix container that stores one row and column
 *  entry per nonzero. The matrix may reside in either "host" or "device"
 *  memory depending on the MemorySpace. All entries in the \p coo_matrix are
 *  sorted according to row indices and internally within each row sorted by
 *  column indices.
 *
 * \note The matrix entries must be sorted by row index.
 * \note The matrix should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p coo_matrix on the host with 6 nonzeros and then copies the
 *  matrix to the device.
 *
 *  \code
 * // include coo_matrix header file
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *      // allocate storage for (4,3) matrix with 6 nonzeros
 *      cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *      // initialize matrix entries on host
 *      A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
 *      A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
 *      A.row_indices[2] = 2; A.column_indices[2] = 2; A.values[2] = 30;
 *      A.row_indices[3] = 3; A.column_indices[3] = 0; A.values[3] = 40;
 *      A.row_indices[4] = 3; A.column_indices[4] = 1; A.values[4] = 50;
 *      A.row_indices[5] = 3; A.column_indices[5] = 2; A.values[5] = 60;
 *
 *      // A now represents the following matrix
 *      //    [10  0 20]
 *      //    [ 0  0  0]
 *      //    [ 0  0 30]
 *      //    [40 50 60]
 *
 *      // copy to the device
 *      cusp::coo_matrix<int,float,cusp::device_memory> B(A);
 *
 *      // print the constructed coo_matrix
 *      cusp::print(B);
 *
 *      return 0;
 *  }
 *  \endcode
 *
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class coo_matrix : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format> Parent;

public:

    /*! \cond */
    typedef typename cusp::array1d<IndexType, MemorySpace> row_indices_array_type;
    typedef typename cusp::array1d<IndexType, MemorySpace> column_indices_array_type;
    typedef typename cusp::array1d<ValueType, MemorySpace> values_array_type;

    typedef typename cusp::coo_matrix<IndexType, ValueType, MemorySpace> container;

    typedef typename cusp::coo_matrix_view<
            typename row_indices_array_type::view,
            typename column_indices_array_type::view,
            typename values_array_type::view,
            IndexType, ValueType, MemorySpace> view;

    typedef typename cusp::coo_matrix_view<
            typename row_indices_array_type::const_view,
            typename column_indices_array_type::const_view,
            typename values_array_type::const_view,
            IndexType, ValueType, MemorySpace> const_view;

    typedef view        coo_view_type;
    typedef const_view  const const_coo_view_type;

    template<typename MemorySpace2>
    struct rebind
    {
        typedef cusp::coo_matrix<IndexType, ValueType, MemorySpace2> type;
    };
    /*! \endcond */

    /*! Storage for the row indices of the COO data structure.
     */
    row_indices_array_type row_indices;

    /*! Storage for the column indices of the COO data structure.
     */
    column_indices_array_type column_indices;

    /*! Storage for the nonzero entries of the COO data structure.
     */
    values_array_type values;

    /*! Construct an empty \p coo_matrix.
     */
    coo_matrix(void) {}

    /*! Construct a \p coo_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    coo_matrix(const size_t num_rows, const size_t num_cols, const size_t num_entries)
        : Parent(num_rows, num_cols, num_entries),
          row_indices(num_entries),
          column_indices(num_entries),
          values(num_entries) {}

    /*! Construct a \p coo_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    coo_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     */
    void resize(size_t num_rows, size_t num_cols, size_t num_entries);

    /*! Swap the contents of two \p coo_matrix objects.
     *
     *  \param matrix Another \p coo_matrix with the same IndexType and ValueType.
     */
    void swap(coo_matrix& matrix);

    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     *  \return \p coo_matrix constructed from existing matrix.
     */
    template <typename MatrixType>
    coo_matrix& operator=(const MatrixType& matrix);

    /*! Sort matrix elements by row index
     */
    void sort_by_row(void);

    /*! Sort matrix elements by row and column index
     */
    void sort_by_row_and_column(void);

    /*! Determine whether matrix elements are sorted by row index
     *
     *  \return \c false, if the row indices are unsorted; \c true, otherwise.
     */
    bool is_sorted_by_row(void);

    /*! Determine whether matrix elements are sorted by row and column index
     *
     *  \return \c false, if the row and column indices are unsorted; \c true, otherwise.
     */
    bool is_sorted_by_row_and_column(void);
}; // class coo_matrix
/*! \}
 */

/*! \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \brief Sparse matrix containers that wrap existing data or iterators in COO, CSR, DIA, ELL, HYB, and
 *  Permutation representations
 *  \{
 */

/**
 * \brief View of a \p coo_matrix
 *
 * \tparam ArrayType1 Type of \c row_indices array view
 * \tparam ArrayType2 Type of \c column_indices array view
 * \tparam ArrayType3 Type of \c values array view
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * \note The matrix entries must be sorted by row index.
 * \note The matrix should not contain duplicate entries.
 *
 *  A \p coo_matrix_view is a sparse matrix view of a matrix in COO format
 *  constructed from existing data or iterators. All entries in the \p coo_matrix are
 *  sorted according to row indices and internally within each row sorted by
 *  column indices.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p coo_matrix_view on the host with 6 nonzeros.
 *
 *  \code
 * // include coo_matrix header file
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    typedef cusp::array1d<int,cusp::host_memory> IndexArray;
 *    typedef cusp::array1d<float,cusp::host_memory> ValueArray;
 *
 *    typedef typename IndexArray::view IndexArrayView;
 *    typedef typename ValueArray::view ValueArrayView;
 *
 *    // initialize rows, columns, and values
 *    IndexArray row_indices(6);
 *    IndexArray column_indices(6);
 *    ValueArray values(6);
 *
 *    row_indices[0] = 0; column_indices[0] = 0; values[0] = 10;
 *    row_indices[1] = 0; column_indices[1] = 2; values[1] = 20;
 *    row_indices[2] = 2; column_indices[2] = 2; values[2] = 30;
 *    row_indices[3] = 3; column_indices[3] = 0; values[3] = 40;
 *    row_indices[4] = 3; column_indices[4] = 1; values[4] = 50;
 *    row_indices[5] = 3; column_indices[5] = 2; values[5] = 60;
 *
 *    // allocate storage for (4,3) matrix with 6 nonzeros
 *    cusp::coo_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView> A(
 *    4,3,6,
 *    cusp::make_array1d_view(row_indices),
 *    cusp::make_array1d_view(column_indices),
 *    cusp::make_array1d_view(values));
 *
 *    // A now represents the following matrix
 *    //    [10  0 20]
 *    //    [ 0  0  0]
 *    //    [ 0  0 30]
 *    //    [40 50 60]
 *
 *    // print the constructed coo_matrix
 *    cusp::print(A);
 *
 *    // change first entry in values array
 *    values[0] = -1;
 *
 *    // print the updated matrix view
 *    cusp::print(A);
 *  }
 *  \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename IndexType   = typename ArrayType2::value_type,
          typename ValueType   = typename ArrayType3::value_type,
          typename MemorySpace = typename cusp::minimum_space<
                                    typename ArrayType1::memory_space,
                                    typename ArrayType2::memory_space,
                                    typename ArrayType3::memory_space>::type >
class coo_matrix_view : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format> Parent;

public:

    /*! \cond */
    typedef ArrayType1 row_indices_array_type;
    typedef ArrayType2 column_indices_array_type;
    typedef ArrayType3 values_array_type;

    typedef typename cusp::coo_matrix<IndexType, ValueType, MemorySpace> container;
    typedef typename cusp::coo_matrix_view<ArrayType1, ArrayType2, ArrayType3, IndexType, ValueType, MemorySpace> view;
    typedef typename cusp::coo_matrix_view<ArrayType1, ArrayType2, ArrayType3, IndexType, ValueType, MemorySpace> const_view;

    typedef view       coo_view_type;
    typedef view const const_coo_view_type;
    /*! \endcond */

    /**
     * View of the row indices of the COO data structure.  Also called the "row pointer" array.
     */
    row_indices_array_type row_indices;

    /**
     * View of the column indices of the COO data structure.
     */
    column_indices_array_type column_indices;

    /**
     * View for the nonzero entries of the COO data structure.
     */
    values_array_type values;

    /**
     * Storage for indices used to generate COO view.
     */
    cusp::array1d<typename thrust::detail::remove_const<IndexType>::type,MemorySpace> indices;

    /**
     * Construct an empty \p coo_matrix_view.
     */
    coo_matrix_view(void)
        : Parent() {}

    /*! Construct a \p coo_matrix_view with a specific shape and number of nonzero entries
     *  from existing arrays denoting the row indices, column indices, and
     *  values.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param row_indices Array containing the row indices.
     *  \param column_indices Array containing the column indices.
     *  \param values Array containing the values.
     */
    coo_matrix_view(const size_t num_rows,
                    const size_t num_cols,
                    const size_t num_entries,
                    ArrayType1 row_indices,
                    ArrayType2 column_indices,
                    ArrayType3 values)
        : Parent(num_rows, num_cols, num_entries),
          row_indices(row_indices),
          column_indices(column_indices),
          values(values) {}

    /*! Construct a \p coo_matrix_view from a existing \p coo_matrix.
     *
     *  \param matrix \p coo_matrix used to create view.
     */
    coo_matrix_view(coo_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          row_indices(matrix.row_indices),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Construct a \p coo_matrix_view from a existing const \p coo_matrix.
     *
     *  \param matrix \p coo_matrix used to create view.
     */
    coo_matrix_view(const coo_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          row_indices(matrix.row_indices),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Construct a \p coo_matrix_view from a existing \p coo_matrix_view.
     *
     *  \param matrix \p coo_matrix_view used to create view.
     */
    coo_matrix_view(view& matrix)
        : Parent(matrix),
          row_indices(matrix.row_indices),
          column_indices(matrix.column_indices),
          values(matrix.values),
          indices(matrix.indices) {}

    /*! Construct a \p coo_matrix_view from a existing const \p coo_matrix.
     *
     *  \param matrix \p coo_matrix used to create view.
     */
    coo_matrix_view(const view& matrix)
        : Parent(matrix),
          row_indices(matrix.row_indices),
          column_indices(matrix.column_indices),
          values(matrix.values),
          indices(matrix.indices) {}

    /*! Construct a \p coo_matrix_view from a existing matrix in another
     * format.
     *
     *  \param matrix used to create view.
     */
    template<typename MatrixType>
    coo_matrix_view(MatrixType& matrix);

    /*! Construct a \p coo_matrix_view from a existing const matrix in another
     * format.
     *
     *  \param matrix used to create view.
     */
    template<typename MatrixType>
    coo_matrix_view(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries);

    /*! Sort matrix elements by row index
     */
    void sort_by_row(void);

    /*! Sort matrix elements by row and column index
     */
    void sort_by_row_and_column(void);

    /*! Determine whether matrix elements are sorted by row index
     *
     *  \return \c false, if the row indices are unsorted; \c true, otherwise.
     */
    bool is_sorted_by_row(void);

    /*! Determine whether matrix elements are sorted by row and column index
     *
     *  \return \c false, if the row and column indices are unsorted; \c true, otherwise.
     */
    bool is_sorted_by_row_and_column(void);

protected:
    /*! Initialize \p coo_matrix_view from  \p csr_matrix.
     *
     *  \param matrix Another matrix in csr_format.
     */
    template<typename MatrixType>
    void construct_from(MatrixType& matrix, csr_format);

    /*! Construct \p coo_matrix_view from  \p dia_matrix.
     *
     *  \param matrix Another matrix in dia_format.
     */
    template<typename MatrixType>
    void construct_from(MatrixType& matrix, dia_format);

    /*! Construct \p coo_matrix_view from  \p ell_matrix.
     *
     *  \param matrix Another matrix in ell_format.
     */
    template<typename MatrixType>
    void construct_from(MatrixType& matrix, ell_format);

    /*! Construct \p coo_matrix_view from  \p hyb_matrix.
     *
     *  \param matrix Another matrix in hyb_format.
     */
    template<typename MatrixType>
    void construct_from(MatrixType& matrix, hyb_format);
};

/* Convenience functions */

/**
 *  This is a convenience function for generating an \p coo_matrix_view
 *  using individual arrays
 *  \tparam ArrayType1 row indices array type
 *  \tparam ArrayType2 column indices array type
 *  \tparam ArrayType3 values array type
 *
 *  \param num_rows Number of rows.
 *  \param num_cols Number of columns.
 *  \param num_entries Number of nonzero matrix entries.
 *  \param row_indices Array containing the row indices.
 *  \param column_indices Array containing the column indices.
 *  \param values Array containing the values.
 *
 *  \return \p coo_matrix_view constructed using input arrays
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
coo_matrix_view<ArrayType1,ArrayType2,ArrayType3>
make_coo_matrix_view(const size_t num_rows,
                     const size_t num_cols,
                     const size_t num_entries,
                     ArrayType1 row_indices,
                     ArrayType2 column_indices,
                     ArrayType3 values)
{
    coo_matrix_view<ArrayType1,ArrayType2,ArrayType3>
           view(num_rows, num_cols, num_entries, row_indices, column_indices, values);

    return view;
}

/**
 *  This is a convenience function for generating an \p coo_matrix_view
 *  using individual arrays with explicit index, value, and memory space
 *  annotations.
 *
 *  \tparam ArrayType1 row indices array type
 *  \tparam ArrayType2 column indices array type
 *  \tparam ArrayType3 values array type
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p coo_matrix_view matrix to copy.
 *
 *  \return \p coo_matrix_view constructed using input arrays.
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename IndexType,
          typename ValueType,
          typename MemorySpace>
coo_matrix_view<ArrayType1,ArrayType2,ArrayType3,IndexType,ValueType,MemorySpace>
make_coo_matrix_view(const coo_matrix_view<ArrayType1,ArrayType2,ArrayType3,IndexType,ValueType,MemorySpace>& m)
{
    return coo_matrix_view<ArrayType1,ArrayType2,ArrayType3,IndexType,ValueType,MemorySpace>(m);
}

/**
 *  This is a convenience function for generating an \p coo_matrix_view
 *  using an existing matrix.
 *
 *  \tparam MatrixType Type of the input matrix
 *
 *  \param m Exemplar \p MatrixType matrix to copy.
 *
 *  \return \p coo_matrix_view constructed using input matrix.
 */
template <typename MatrixType>
typename MatrixType::coo_view_type
make_coo_matrix_view(MatrixType& m)
{
    typedef typename MatrixType::coo_view_type View;

    return View(m);
}

/**
 *  This is a convenience function for generating an const \p coo_matrix_view
 *  using an existing matrix.
 *
 *  \tparam MatrixType Type of the input matrix
 *
 *  \param m Exemplar \p MatrixType matrix to copy.
 *
 *  \return \p coo_matrix_view constructed using input matrix.
 */
template <typename MatrixType>
typename MatrixType::const_coo_view_type
make_coo_matrix_view(const MatrixType& m)
{
    typedef typename MatrixType::const_coo_view_type View;

    return View(m);
}
/*! \}
 */


} // end namespace cusp

#include <cusp/detail/coo_matrix.inl>
