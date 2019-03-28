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

/*! \file csr_matrix.h
 *  \brief Compressed Sparse Row matrix format.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/memory.h>

#include <cusp/detail/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/detail/type_traits.h>

namespace cusp
{

// forward definition
template <typename ArrayType1, typename ArrayType2, typename ArrayType3, typename IndexType, typename ValueType, typename MemorySpace> class csr_matrix_view;

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief Compressed sparse row (CSR) representation a sparse matrix
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p csr_matrix is a sparse matrix container that stores an offset to the
 *  first entry of each row in matrix and one column
 *  entry per nonzero. The matrix may reside in either "host" or "device"
 *  memory depending on the MemorySpace. All entries in the \p csr_matrix are
 *  sorted according to row and internally sorted within each row by column
 *  index.
 *
 * \note The matrix entries within the same row must be sorted by column index.
 * \note The matrix should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p csr_matrix on the host with 6 nonzeros and then copies the
 *  matrix to the device.
 *
 *  \code
 *  // include the csr_matrix header file
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/print.h>
 *
 *  int main()
 *  {
 *    // allocate storage for (4,3) matrix with 4 nonzeros
 *    cusp::csr_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *    // initialize matrix entries on host
 *    A.row_offsets[0] = 0;  // first offset is always zero
 *    A.row_offsets[1] = 2;
 *    A.row_offsets[2] = 2;
 *    A.row_offsets[3] = 3;
 *    A.row_offsets[4] = 6; // last offset is always num_entries
 *
 *    A.column_indices[0] = 0; A.values[0] = 10;
 *    A.column_indices[1] = 2; A.values[1] = 20;
 *    A.column_indices[2] = 2; A.values[2] = 30;
 *    A.column_indices[3] = 0; A.values[3] = 40;
 *    A.column_indices[4] = 1; A.values[4] = 50;
 *    A.column_indices[5] = 2; A.values[5] = 60;
 *
 *    // A now represents the following matrix
 *    //    [10  0 20]
 *    //    [ 0  0  0]
 *    //    [ 0  0 30]
 *    //    [40 50 60]
 *
 *    // copy to the device
 *    cusp::csr_matrix<int,float,cusp::device_memory> B(A);
 *
 *    cusp::print(B);
 *  }
 *  \endcode
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class csr_matrix : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format> Parent;

public:

    /*! \cond */
    typedef typename cusp::array1d<IndexType, MemorySpace> row_offsets_array_type;
    typedef typename cusp::array1d<IndexType, MemorySpace> column_indices_array_type;
    typedef typename cusp::array1d<ValueType, MemorySpace> values_array_type;

    typedef typename cusp::csr_matrix<IndexType, ValueType, MemorySpace> container;

    typedef typename cusp::csr_matrix_view<typename row_offsets_array_type::view,
            typename column_indices_array_type::view,
            typename values_array_type::view,
            IndexType, ValueType, MemorySpace> view;

    typedef typename cusp::csr_matrix_view<typename row_offsets_array_type::const_view,
            typename column_indices_array_type::const_view,
            typename values_array_type::const_view,
            IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<container,cusp::csr_format>::view            coo_view_type;
    // TODO : Why does GCC 4.4 fail using const type? Is it necessary?
    typedef typename cusp::detail::coo_view_type<container /*const*/,cusp::csr_format>::view  const_coo_view_type;

    template<typename MemorySpace2>
    struct rebind
    {
        typedef cusp::csr_matrix<IndexType, ValueType, MemorySpace2> type;
    };
    /*! \endcond */

    /*! Storage for the row offsets of the CSR data structure.  Also called the "row pointer" array.
     */
    row_offsets_array_type row_offsets;

    /*! Storage for the column indices of the CSR data structure.
     */
    column_indices_array_type column_indices;

    /*! Storage for the nonzero entries of the CSR data structure.
     */
    values_array_type values;

    /*! Construct an empty \p csr_matrix.
     */
    csr_matrix(void) {}

    /*! Construct a \p csr_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    csr_matrix(size_t num_rows, size_t num_cols, size_t num_entries)
        : Parent(num_rows, num_cols, num_entries),
          row_offsets(num_rows + 1),
          column_indices(num_entries),
          values(num_entries) {}

    /*! Construct a \p csr_matrix from another matrix.
     *
     *  \tparam MatrixType Type of input matrix used to create this \p
     *  csr_matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    csr_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries);

    /*! Swap the contents of two \p csr_matrix objects.
     *
     *  \param matrix Another \p csr_matrix with the same IndexType and ValueType.
     */
    void swap(csr_matrix& matrix);

    /*! Assignment from another matrix.
     *
     *  \tparam MatrixType Type of input matrix to copy into this \p
     *  csr_matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    csr_matrix& operator=(const MatrixType& matrix);

}; // class csr_matrix
/*! \}
 */

/**
 * \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief View of a \p csr_matrix
 *
 * \tparam ArrayType1 Type of \c row_offsets array view
 * \tparam ArrayType2 Type of \c column_indices array view
 * \tparam ArrayType3 Type of \c values array view
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p csr_matrix_view is a sparse matrix view of a matrix in CSR format
 *  constructed from existing data or iterators. All entries in the \p csr_matrix are
 *  sorted according to rows and internally within each row sorted by
 *  column indices.
 *
 * \note The matrix entries must be sorted by row index.
 * \note The matrix should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p csr_matrix_view on the host with 6 nonzeros.
 *
 *  \code
 * // include csr_matrix header file
 * #include <cusp/csr_matrix.h>
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
 *    IndexArray row_offsets(6);
 *    IndexArray column_indices(6);
 *    ValueArray values(6);
 *
 *    // initialize matrix entries on host
 *    row_offsets[0] = 0;  // first offset is always zero
 *    row_offsets[1] = 2;
 *    row_offsets[2] = 2;
 *    row_offsets[3] = 3;
 *    row_offsets[4] = 6; // last offset is always num_entries
 *
 *    column_indices[0] = 0; values[0] = 10;
 *    column_indices[1] = 2; values[1] = 20;
 *    column_indices[2] = 2; values[2] = 30;
 *    column_indices[3] = 0; values[3] = 40;
 *    column_indices[4] = 1; values[4] = 50;
 *    column_indices[5] = 2; values[5] = 60;
 *
 *    // allocate storage for (4,3) matrix with 6 nonzeros
 *    cusp::csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView> A(
 *    4,3,6,
 *    cusp::make_array1d_view(row_offsets),
 *    cusp::make_array1d_view(column_indices),
 *    cusp::make_array1d_view(values));
 *
 *    // A now represents the following matrix
 *    //    [10  0 20]
 *    //    [ 0  0  0]
 *    //    [ 0  0 30]
 *    //    [40 50 60]
 *
 *    // print the constructed csr_matrix
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
          typename IndexType   = typename ArrayType1::value_type,
          typename ValueType   = typename ArrayType3::value_type,
          typename MemorySpace = typename cusp::minimum_space<
                                    typename ArrayType1::memory_space,
                                    typename ArrayType2::memory_space,
                                    typename ArrayType3::memory_space>::type >
class csr_matrix_view : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format> Parent;

public:

    /*! \cond */
    typedef ArrayType1 row_offsets_array_type;
    typedef ArrayType2 column_indices_array_type;
    typedef ArrayType3 values_array_type;

    typedef typename cusp::csr_matrix<IndexType, ValueType, MemorySpace> container;
    typedef typename cusp::csr_matrix_view<ArrayType1, ArrayType2, ArrayType3, IndexType, ValueType, MemorySpace> view;
    typedef typename cusp::csr_matrix_view<ArrayType1, ArrayType2, ArrayType3, IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<view,cusp::csr_format>::view       coo_view_type;
    typedef typename cusp::detail::coo_view_type<view const,cusp::csr_format>::view const_coo_view_type;
    /*! \endcond */

    /**
     * View of the row_offsets of the CSR data structure.
     */
    row_offsets_array_type row_offsets;

    /**
     * View of the column indices of the CSR data structure.
     */
    column_indices_array_type column_indices;

    /**
     * View for the nonzero entries of the CSR data structure.
     */
    values_array_type values;

    /**
     * Construct an empty \p csr_matrix_view.
     */
    csr_matrix_view(void)
        : Parent() {}

    /*! Construct a \p csr_matrix_view with a specific shape and number of nonzero entries
     *  from existing arrays denoting the row offsets, column indices, and
     *  values.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param row_offsets Array containing the row offsets.
     *  \param column_indices Array containing the column indices.
     *  \param values Array containing the values.
     */
    csr_matrix_view(const size_t num_rows,
                    const size_t num_cols,
                    const size_t num_entries,
                    ArrayType1 row_offsets,
                    ArrayType2 column_indices,
                    ArrayType3 values)
        : Parent(num_rows, num_cols, num_entries),
          row_offsets(row_offsets),
          column_indices(column_indices),
          values(values) {}

    /*! Construct a \p csr_matrix_view from a existing \p csr_matrix.
     *
     *  \param matrix \p csr_matrix used to create view.
     */
    csr_matrix_view(csr_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          row_offsets(matrix.row_offsets),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Construct a \p csr_matrix_view from a existing const \p csr_matrix.
     *
     *  \param matrix \p csr_matrix used to create view.
     */
    csr_matrix_view(const csr_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          row_offsets(matrix.row_offsets),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Construct a \p csr_matrix_view from a existing \p csr_matrix_view.
     *
     *  \param matrix \p csr_matrix_view used to create view.
     */
    csr_matrix_view(csr_matrix_view& matrix)
        : Parent(matrix),
          row_offsets(matrix.row_offsets),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Construct a \p csr_matrix_view from a existing const \p csr_matrix_view.
     *
     *  \param matrix \p csr_matrix_view used to create view.
     */
    csr_matrix_view(const csr_matrix_view& matrix)
        : Parent(matrix),
          row_offsets(matrix.row_offsets),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries);
};

/* Convenience functions */

/**
 *  This is a convenience function for generating an \p csr_matrix_view
 *  using individual arrays
 *  \tparam ArrayType1 row offsets array type
 *  \tparam ArrayType2 column indices array type
 *  \tparam ArrayType3 values array type
 *
 *  \param num_rows Number of rows.
 *  \param num_cols Number of columns.
 *  \param num_entries Number of nonzero matrix entries.
 *  \param row_offsets Array containing the row offsets.
 *  \param column_indices Array containing the column indices.
 *  \param values Array containing the values.
 *
 *  \return \p csr_matrix_view constructed using input arrays
 */
template <typename ArrayType1,
         typename ArrayType2,
         typename ArrayType3>
csr_matrix_view<ArrayType1,ArrayType2,ArrayType3>
make_csr_matrix_view(size_t num_rows,
                     size_t num_cols,
                     size_t num_entries,
                     ArrayType1 row_offsets,
                     ArrayType2 column_indices,
                     ArrayType3 values)
{
    csr_matrix_view<ArrayType1,ArrayType2,ArrayType3>
           view(num_rows, num_cols, num_entries, row_offsets, column_indices, values);

	return view;
}

/**
 *  This is a convenience function for generating an \p csr_matrix_view
 *  using individual arrays with explicit index, value, and memory space
 *  annotations.
 *
 *  \tparam ArrayType1 row offsets array type
 *  \tparam ArrayType2 column indices array type
 *  \tparam ArrayType3 values array type
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p csr_matrix_view matrix to copy.
 *
 *  \return \p csr_matrix_view constructed using input arrays.
 */
template <typename ArrayType1,
         typename ArrayType2,
         typename ArrayType3,
         typename IndexType,
         typename ValueType,
         typename MemorySpace>
csr_matrix_view<ArrayType1,ArrayType2,ArrayType3,IndexType,ValueType,MemorySpace>
make_csr_matrix_view(const csr_matrix_view<ArrayType1,ArrayType2,ArrayType3,IndexType,ValueType,MemorySpace>& m)
{
    return csr_matrix_view<ArrayType1,ArrayType2,ArrayType3,IndexType,ValueType,MemorySpace>(m);
}

/**
 *  This is a convenience function for generating an \p csr_matrix_view
 *  using an existing \p csr_matrix.
 *
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p csr_matrix matrix to copy.
 *
 *  \return \p csr_matrix_view constructed using input arrays.
 */
template <typename IndexType, typename ValueType, class MemorySpace>
typename csr_matrix<IndexType,ValueType,MemorySpace>::view
make_csr_matrix_view(csr_matrix<IndexType,ValueType,MemorySpace>& m)
{
    return make_csr_matrix_view
           (m.num_rows, m.num_cols, m.num_entries,
            make_array1d_view(m.row_offsets),
            make_array1d_view(m.column_indices),
            make_array1d_view(m.values));
}

/**
 *  This is a convenience function for generating an const \p csr_matrix_view
 *  using an existing \p csr_matrix.
 *
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p csr_matrix matrix to copy.
 *
 *  \return \p csr_matrix_view constructed using input arrays.
 */
template <typename IndexType, typename ValueType, class MemorySpace>
typename csr_matrix<IndexType,ValueType,MemorySpace>::const_view
make_csr_matrix_view(const csr_matrix<IndexType,ValueType,MemorySpace>& m)
{
    return make_csr_matrix_view
           (m.num_rows, m.num_cols, m.num_entries,
            make_array1d_view(m.row_offsets),
            make_array1d_view(m.column_indices),
            make_array1d_view(m.values));
}
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/csr_matrix.inl>
