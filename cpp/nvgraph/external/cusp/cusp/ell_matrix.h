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

/*! \file ell_matrix.h
 *  \brief ELLPACK/ITPACK matrix format.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/memory.h>

#include <cusp/detail/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/detail/type_traits.h>
#include <cusp/detail/utils.h>

namespace cusp
{

/*! \cond */
template<typename,typename,typename>                   class array2d;
template<typename,typename>                            class array2d_view;
template<typename,typename,typename,typename,typename> class ell_matrix_view;

template<typename T, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<T,MemorySpace>::iterator>, Orientation>
make_array2d_view(cusp::array2d<T,MemorySpace,Orientation>& v);

template<typename T, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<T,MemorySpace>::const_iterator>, Orientation>
make_array2d_view(const cusp::array2d<T,MemorySpace,Orientation>& v);
/*! \endcond */

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief Packed row (ELLPACK/ITPACK) representation of a sparse matrix
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * \note The matrix entries must be sorted by column index.
 * \note The matrix entries within each row should be shifted to the left.
 * \note The matrix should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p ell_matrix on the host with 3 nonzeros per row (6 total nonzeros)
 *  and then copies the matrix to the device.
 *
 *  \code
 *  // include the ell_matrix header file
 *  #include <cusp/ell_matrix.h>
 *  #include <cusp/print.h>
 *
 *  int main()
 *  {
 *    // allocate storage for (4,3) matrix with 6 nonzeros and at most 3 nonzeros per row.
 *    cusp::ell_matrix<int,float,cusp::host_memory> A(4,3,6,3);
 *
 *    // X is used to fill unused entries in the matrix
 *    const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *    // initialize matrix entries on host
 *    A.column_indices(0,0) = 0; A.values(0,0) = 10;
 *    A.column_indices(0,1) = 2; A.values(0,1) = 20;  // shifted to leftmost position
 *    A.column_indices(0,2) = X; A.values(0,2) =  0;  // padding
 *
 *    A.column_indices(1,0) = X; A.values(1,0) =  0;  // padding
 *    A.column_indices(1,1) = X; A.values(1,1) =  0;  // padding
 *    A.column_indices(1,2) = X; A.values(1,2) =  0;  // padding
 *
 *    A.column_indices(2,0) = 2; A.values(2,0) = 30;  // shifted to leftmost position
 *    A.column_indices(2,1) = X; A.values(2,1) =  0;  // padding
 *    A.column_indices(2,2) = X; A.values(2,2) =  0;  // padding
 *
 *    A.column_indices(3,0) = 0; A.values(3,0) = 40;
 *    A.column_indices(3,1) = 1; A.values(3,1) = 50;
 *    A.column_indices(3,2) = 2; A.values(3,2) = 60;
 *
 *    // A now represents the following matrix
 *    //    [10  0 20]
 *    //    [ 0  0  0]
 *    //    [ 0  0 30]
 *    //    [40 50 60]
 *
 *    // copy to the device
 *    cusp::ell_matrix<int,float,cusp::device_memory> B(A);
 *
 *    cusp::print(B);
 *  }
 *  \endcode
 */
template <typename IndexType, typename ValueType, typename MemorySpace>
class ell_matrix : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::ell_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::ell_format> Parent;

public:

    /*! Value used to pad the rows of the column_indices array.
     */
    const static IndexType invalid_index = static_cast<IndexType>(-1);

    /*! \cond */
    typedef typename cusp::array2d<IndexType, MemorySpace, cusp::column_major> column_indices_array_type;
    typedef typename cusp::array2d<ValueType, MemorySpace, cusp::column_major> values_array_type;

    typedef typename cusp::ell_matrix<IndexType, ValueType, MemorySpace>       container;

    typedef typename cusp::ell_matrix_view<
            typename column_indices_array_type::view,
            typename values_array_type::view,
            IndexType, ValueType, MemorySpace> view;

    typedef typename cusp::ell_matrix_view<
            typename column_indices_array_type::const_view,
            typename values_array_type::const_view,
            IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<container, cusp::ell_format>::view           coo_view_type;
    // TODO : Why does GCC 4.4 fail using const type? Is it necessary?
    typedef typename cusp::detail::coo_view_type<container /*const*/, cusp::ell_format>::view const_coo_view_type;

    template<typename MemorySpace2>
    struct rebind
    {
        typedef cusp::ell_matrix<IndexType, ValueType, MemorySpace2> type;
    };
    /*! \endcond */

    /*! Storage for the column indices of the ELL data structure.
     */
    column_indices_array_type column_indices;

    /*! Storage for the nonzero entries of the ELL data structure.
     */
    values_array_type values;

    /*! Construct an empty \p ell_matrix.
     */
    ell_matrix(void) {}

    /*! Construct an \p ell_matrix with a specific shape, number of nonzero entries,
     *  and maximum number of nonzero entries per row.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    ell_matrix(const size_t num_rows, const size_t num_cols, const size_t num_entries,
               const size_t num_entries_per_row, const size_t alignment = 32);

    /*! Construct an \p ell_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    ell_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_entries_per_row);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_entries_per_row, const size_t alignment);

    /*! Swap the contents of two \p ell_matrix objects.
     *
     *  \param matrix Another \p ell_matrix with the same IndexType and ValueType.
     */
    void swap(ell_matrix& matrix);

    /*! Assignment from another matrix.
     *
     *  \tparam MatrixType Format type of input matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    ell_matrix& operator=(const MatrixType& matrix);
}; // class ell_matrix
/*! \}
 */


/*! \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief View of a \p ell_matrix
 *
 * \tparam ArrayType1 Type of \c column_indices array view
 * \tparam ArrayType2 Type of \c values array view
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * \note The matrix entries must be sorted by column index.
 * \note The matrix entries within each row should be shifted to the left.
 * \note The matrix should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p ell_matrix_view on the host with 3 nonzeros per row.
 *
 *  \code
 *  // include the ell_matrix header file
 *  #include <cusp/ell_matrix.h>
 *  #include <cusp/print.h>
 *
 *  int main()
 *  {
 *    typedef cusp::array2d<int,cusp::host_memory,cusp::column_major> IndexArray;
 *    typedef cusp::array2d<float,cusp::host_memory,cusp::column_major> ValueArray;
 *
 *    typedef typename IndexArray::view IndexArrayView;
 *    typedef typename ValueArray::view ValueArrayView;
 *
 *    // initialize columns and values
 *    IndexArray column_indices(4,3);
 *    ValueArray values(4,3);
 *
 *    // X is used to fill unused entries in the matrix
 *    const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *    // initialize matrix entries on host
 *    column_indices(0,0) = 0; values(0,0) = 10;
 *    column_indices(0,1) = 2; values(0,1) = 20;  // shifted to leftmost position
 *    column_indices(0,2) = X; values(0,2) =  0;  // padding
 *
 *    column_indices(1,0) = X; values(1,0) =  0;  // padding
 *    column_indices(1,1) = X; values(1,1) =  0;  // padding
 *    column_indices(1,2) = X; values(1,2) =  0;  // padding
 *
 *    column_indices(2,0) = 2; values(2,0) = 30;  // shifted to leftmost position
 *    column_indices(2,1) = X; values(2,1) =  0;  // padding
 *    column_indices(2,2) = X; values(2,2) =  0;  // padding
 *
 *    column_indices(3,0) = 0; values(3,0) = 40;
 *    column_indices(3,1) = 1; values(3,1) = 50;
 *    column_indices(3,2) = 2; values(3,2) = 60;
 *
 *    // allocate storage for (4,3) matrix with 6 nonzeros
 *    cusp::ell_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView> A(
 *    4,3,6,
 *    cusp::make_array2d_view(column_indices),
 *    cusp::make_array2d_view(values));
 *
 *    // A now represents the following matrix
 *    //    [10  0 20]
 *    //    [ 0  0  0]
 *    //    [ 0  0 30]
 *    //    [40 50 60]
 *
 *    // print the constructed ell_matrix
 *    cusp::print(A);
 *
 *    // change first entry in values array
 *    values(0,0) = -1;
 *
 *    // print the updated matrix view
 *    cusp::print(A);
 *  }
 *  \endcode
 */
template <typename ArrayType1,
         typename ArrayType2,
         typename IndexType   = typename ArrayType1::value_type,
         typename ValueType   = typename ArrayType2::value_type,
         typename MemorySpace = typename cusp::minimum_space<typename ArrayType1::memory_space, typename ArrayType2::memory_space>::type >
class ell_matrix_view : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::ell_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::ell_format> Parent;

public:

    /*! \cond */
    typedef ArrayType1 column_indices_array_type;
    typedef ArrayType2 values_array_type;

    typedef cusp::ell_matrix<IndexType, ValueType, MemorySpace>                              container;
    typedef cusp::ell_matrix_view<ArrayType1, ArrayType2, IndexType, ValueType, MemorySpace> view;
    typedef cusp::ell_matrix_view<ArrayType1, ArrayType2, IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<view,cusp::ell_format>::view       coo_view_type;
    typedef typename cusp::detail::coo_view_type<view const,cusp::ell_format>::view const_coo_view_type;
    /*! \endcond */

    /**
     * Value used to pad the rows of the column_indices array.
     */
    const static IndexType invalid_index = container::invalid_index;

    /**
     * View to column indices of the ELL data structure.
     */
    column_indices_array_type column_indices;

    /**
     * View to nonzero entries of the ELL data structure.
     */
    values_array_type values;

    /**
     * Construct an empty \p ell_matrix_view.
     */
    ell_matrix_view(void) {}

    /*! Construct a \p ell_matrix_view with a specific shape and number of nonzero entries
     *  from existing arrays denoting the column indices and values.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param column_indices Array containing the column indices.
     *  \param values Array containing the values.
     */
    template <typename OtherArrayType1, typename OtherArrayType2>
    ell_matrix_view(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                    const OtherArrayType1& column_indices, const OtherArrayType2& values)
        : Parent(num_rows, num_cols, num_entries),
          column_indices(column_indices),
          values(values) {}

    /*! Construct a \p ell_matrix_view from a existing \p ell_matrix.
     *
     *  \param matrix \p ell_matrix used to create view.
     */
    ell_matrix_view(ell_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          column_indices(make_array2d_view(matrix.column_indices)),
          values(make_array2d_view(matrix.values)) {}

    /*! Construct a \p ell_matrix_view from a existing const \p ell_matrix.
     *
     *  \param matrix \p ell_matrix used to create view.
     */
    ell_matrix_view(const ell_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix),
          column_indices(make_array2d_view(matrix.column_indices)),
          values(make_array2d_view(matrix.values)) {}

    /*! Construct a \p ell_matrix_view from a existing \p ell_matrix_view.
     *
     *  \param matrix \p ell_matrix_view used to create view.
     */
    ell_matrix_view(ell_matrix_view& matrix)
        : Parent(matrix),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Construct a \p ell_matrix_view from a existing const \p ell_matrix_view.
     *
     *  \param matrix \p ell_matrix_view used to create view.
     */
    ell_matrix_view(const ell_matrix_view& matrix)
        : Parent(matrix),
          column_indices(matrix.column_indices),
          values(matrix.values) {}

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     */
    void resize(size_t num_rows, size_t num_cols, size_t num_entries,
                size_t num_entries_per_row);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    void resize(size_t num_rows, size_t num_cols, size_t num_entries,
                size_t num_entries_per_row, size_t alignment);
}; // class ell_matrix_view

/**
 *  This is a convenience function for generating an \p ell_matrix_view
 *  using individual arrays
 *  \tparam ArrayType1 column indices array type
 *  \tparam ArrayType2 values array type
 *
 *  \param num_rows Number of rows.
 *  \param num_cols Number of columns.
 *  \param num_entries Number of nonzero matrix entries.
 *  \param column_indices Array containing the column indices.
 *  \param values Array containing the values.
 *
 *  \return \p ell_matrix_view constructed using input arrays
 */
template <typename ArrayType1, typename ArrayType2>
ell_matrix_view<ArrayType1,ArrayType2>
make_ell_matrix_view(size_t num_rows, size_t num_cols, size_t num_entries,
                     ArrayType1 column_indices, ArrayType2 values)
{
    ell_matrix_view<ArrayType1,ArrayType2>
           view(num_rows, num_cols, num_entries, column_indices, values);

    return view;
}

/**
 *  This is a convenience function for generating an \p ell_matrix_view
 *  using individual arrays with explicit index, value, and memory space
 *  annotations.
 *
 *  \tparam ArrayType1 column indices array type
 *  \tparam ArrayType2 values array type
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p ell_matrix_view matrix to copy.
 *
 *  \return \p ell_matrix_view constructed using input arrays.
 */
template <typename ArrayType1,
         typename ArrayType2,
         typename IndexType,
         typename ValueType,
         typename MemorySpace>
ell_matrix_view<ArrayType1,ArrayType2,IndexType,ValueType,MemorySpace>
make_ell_matrix_view(const ell_matrix_view<ArrayType1,ArrayType2,IndexType,ValueType,MemorySpace>& m)
{
    return ell_matrix_view<ArrayType1,ArrayType2,IndexType,ValueType,MemorySpace>(m);
}

/**
 *  This is a convenience function for generating an \p ell_matrix_view
 *  using an existing \p ell_matrix.
 *
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p ell_matrix matrix to copy.
 *
 *  \return \p ell_matrix_view constructed using input arrays.
 */
template <typename IndexType, typename ValueType, typename MemorySpace>
typename ell_matrix<IndexType,ValueType,MemorySpace>::view
make_ell_matrix_view(ell_matrix<IndexType,ValueType,MemorySpace>& m)
{
    return make_ell_matrix_view
           (m.num_rows, m.num_cols, m.num_entries,
            cusp::make_array2d_view(m.column_indices),
            cusp::make_array2d_view(m.values));
}

/**
 *  This is a convenience function for generating an const \p ell_matrix_view
 *  using an existing \p ell_matrix.
 *
 *  \tparam IndexType  indices type
 *  \tparam ValueType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p ell_matrix matrix to copy.
 *
 *  \return \p ell_matrix_view constructed using input arrays.
 */
template <typename IndexType, typename ValueType, typename MemorySpace>
typename ell_matrix<IndexType,ValueType,MemorySpace>::const_view
make_ell_matrix_view(const ell_matrix<IndexType,ValueType,MemorySpace>& m)
{
    return make_ell_matrix_view
           (m.num_rows, m.num_cols, m.num_entries,
            cusp::make_array2d_view(m.column_indices),
            cusp::make_array2d_view(m.values));
}

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/ell_matrix.inl>
