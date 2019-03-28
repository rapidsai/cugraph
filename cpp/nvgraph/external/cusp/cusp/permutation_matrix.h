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

/*! \file permutation_matrix.h
 *  \brief A permutation matrix.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>
#include <cusp/detail/matrix_base.h>

#include <cusp/array1d.h>
#include <cusp/memory.h>

namespace cusp
{

// forward definition
template <typename ArrayType, typename IndexType, typename MemorySpace> class permutation_matrix_view;

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief Simple representation a permutation matrix
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  This matrix represents a row permutation of the identity matrix.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 3-by-3
 *  \p permutation_matrix on the host with 3 nonzeros and permutes
 *  a coo_matrix by first by row and then by column.
 *
 *  \code
 *  // include the permutation_matrix header file
 *  #include <cusp/coo_matrix.h>
 *  #include <cusp/multiply.h>
 *  #include <cusp/permutation_matrix.h>
 *  #include <cusp/print.h>
 *  int main()
 *  {
 *    typedef cusp::host_memory MemorySpace;
 *
 *    // allocate storage for (3,3) matrix with 5 nonzeros
 *    cusp::coo_matrix<int,float,MemorySpace> A(3,3,7);
 *    // initialize matrix entries on host
 *    A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
 *    A.row_indices[1] = 0; A.column_indices[1] = 1; A.values[1] = 20;
 *    A.row_indices[2] = 0; A.column_indices[2] = 2; A.values[2] = 30;
 *    A.row_indices[3] = 1; A.column_indices[3] = 0; A.values[3] = 40;
 *    A.row_indices[4] = 1; A.column_indices[4] = 1; A.values[4] = 50;
 *    A.row_indices[5] = 2; A.column_indices[5] = 0; A.values[5] = 60;
 *    A.row_indices[6] = 2; A.column_indices[6] = 2; A.values[6] = 70;
 *    // A now represents the following matrix
 *    //    [10  0 20]
 *    //    [30 40  0]
 *    //    [ 0  0 50]
 *    // generate a index permutation that swaps row or column 0 and 2
 *    cusp::array1d<int,MemorySpace> permutation(3);
 *    permutation[0] = 2; // 0 maps to 2
 *    permutation[1] = 1; // 1 maps to 1
 *    permutation[2] = 0; // 2 maps to 0
 *    // allocate storage for (3,3) matrix with 3 nonzeros
 *    cusp::permutation_matrix<int,MemorySpace> P(3, permutation);
 *    // P now represents the following permutation matrix
 *    //    [0 0 1]
 *    //    [0 1 0]
 *    //    [1 0 0]
 *    // permute the rows of A
 *    cusp::coo_matrix<int,float,MemorySpace> PA;
 *    cusp::multiply(P, A, PA);
 *    // permute the column of PA
 *    cusp::coo_matrix<int,float,MemorySpace> PAP;
 *    cusp::multiply(PA, P, PAP);
 *    // print the matrix
 *    cusp::print(cusp::array2d<float,MemorySpace>(A));
 *    // print the permuted matrix
 *    cusp::print(cusp::array2d<float,MemorySpace>(PAP));
 *  }
 *  \endcode
 *
 */
template <typename IndexType, typename MemorySpace>
class permutation_matrix : public cusp::detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format> Parent;

public:

    /*! \cond */
    typedef typename cusp::array1d<IndexType, MemorySpace> permutation_array_type;

    typedef typename cusp::permutation_matrix<IndexType, MemorySpace> container;

    typedef typename cusp::permutation_matrix_view<
            typename permutation_array_type::view,
            IndexType,
            MemorySpace> view;

    typedef typename cusp::permutation_matrix_view<
            typename permutation_array_type::const_view,
            IndexType,
            MemorySpace> const_view;

    template<typename MemorySpace2>
    struct rebind {
        typedef cusp::permutation_matrix<IndexType, MemorySpace2> type;
    };
    /*! \endcond */

    /*! Storage for the permutation indices
     */
    permutation_array_type permutation;

    /*! Construct an empty \p permutation_matrix.
     */
    permutation_matrix(void) {}

    /*! Construct a \p permutation_matrix with a specific number of rows.
     *
     *  \param num_rows Number of rows.
     */
    permutation_matrix(const size_t num_rows)
        : Parent(num_rows, num_rows, num_rows),
          permutation(cusp::counting_array<int>(num_rows)) {}

    /*! Construct a \p permutation_matrix from another matrix.
     *
     *  \tparam MemorySpace2 Memory space of the input matrix
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template<typename MemorySpace2>
    permutation_matrix(const permutation_matrix<IndexType,MemorySpace2>& matrix)
        : Parent(matrix), permutation(matrix.permutation) {}

    /*! Construct a \p permutation_matrix from another matrix.
     *
     *  \tparam ArrayType permutation array type
     *
     *  \param num_rows Number of rows.
     *  \param permutation Array containing the permutation indices.
     */
    template<typename ArrayType>
    permutation_matrix(const size_t num_rows, const ArrayType& permutation)
        : Parent(num_rows, num_rows, num_rows), permutation(permutation) {}

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     */
    void resize(const size_t num_rows);

    /*! Swap the contents of two \p permutation_matrix objects.
     *
     *  \param matrix Another \p permutation_matrix with the same IndexType.
     */
    void swap(permutation_matrix& matrix);

    /*! Permute rows and columns of matrix elements
     *
     *  \tparam MatrixType Type of input matrix to permute
     *
     *  \param matrix Input matrix to apply symmetric permutation.
     */
    template<typename MatrixType>
    void symmetric_permute(MatrixType& matrix);
}; // class permutation_matrix
/*! \}
 */

/*! \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief View of a \p permutation_matrix
 *
 * \tparam Array Type of permutation array view
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \par Overview
 *
 *  A \p permutation_matrix_view is a view of a \p permutation_matrix
 *  constructed from existing data or iterators.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 3-by-3
 *  \p permutation_matrix on the host with 3 nonzeros and permutes
 *  a coo_matrix by first by row and then by column.
 *
 *  \code
 *  // include the permutation_matrix header file
 *  #include <cusp/coo_matrix.h>
 *  #include <cusp/multiply.h>
 *  #include <cusp/permutation_matrix.h>
 *  #include <cusp/print.h>
 *
 *  int main()
 *  {
 *    typedef cusp::array1d<int, cusp::host_memory> ArrayType;
 *
 *    // allocate storage for (3,3) matrix with 5 nonzeros
 *    cusp::coo_matrix<int,float,cusp::host_memory> A(3,3,5);
 *
 *    // initialize matrix entries on host
 *    A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
 *    A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
 *    A.row_indices[2] = 0; A.column_indices[2] = 0; A.values[2] = 30;
 *    A.row_indices[3] = 1; A.column_indices[3] = 1; A.values[3] = 40;
 *    A.row_indices[4] = 2; A.column_indices[4] = 2; A.values[4] = 50;
 *
 *    // A now represents the following matrix
 *    //    [10  0 20]
 *    //    [30 40  0]
 *    //    [ 0  0 50]
 *
 *    // generate a index permutation that swaps row or column 0 and 2
 *    ArrayType permutation(3);
 *    permutation[0] = 2; // 0 maps to 2
 *    permutation[1] = 1; // 1 maps to 1
 *    permutation[2] = 0; // 2 maps to 0
 *
 *    // allocate storage for (3,3) matrix with 3 nonzeros
 *    cusp::permutation_matrix_view<ArrayType> P(3, permutation);
 *
 *    // P now represents the following permutation matrix
 *    //    [0 0 1]
 *    //    [0 1 0]
 *    //    [1 0 0]
 *
 *    // permute the rows of A
 *    cusp::coo_matrix<int,float,cusp::host_memory> PA;
 *    cusp::multiply(P, A, PA);
 *
 *    // permute the column of PA
 *    cusp::coo_matrix<int,float,cusp::host_memory> PAP;
 *    cusp::multiply(PA, P, PAP);
 *
 *    // print the permuted matrix
 *    cusp::print(PAP);
 *  }
 *  \endcode
 */
template <typename ArrayType,
          typename IndexType   = typename ArrayType::value_type,
          typename MemorySpace = typename ArrayType::memory_space>
class permutation_matrix_view : public cusp::detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format> Parent;

public:

    /*! \cond */
    typedef ArrayType permutation_array_type;

    typedef typename cusp::permutation_matrix<IndexType, MemorySpace> container;

    typedef typename cusp::permutation_matrix_view<ArrayType, IndexType, MemorySpace> view;
    /*! \endcond */

    /*! Storage for the permutation indices
     */
    permutation_array_type permutation;

    /*! Construct an empty \p permutation_matrix_view.
     */
    permutation_matrix_view(void)
        : Parent() {}

    /*! Construct a \p permutation_matrix_view with a specific number of rows
     *  from an existing array denoting the permutation indices.
     *
     *  \param num_rows Number of rows.
     *  \param permutation Array containing the permutation indices.
     */
    permutation_matrix_view(const size_t num_rows, ArrayType& permutation)
        : Parent(num_rows, num_rows, num_rows),
          permutation(permutation) {}

    /*! Construct a \p permutation_matrix_view with a specific number of rows
     *  from an existing const array denoting the permutation indices.
     *
     *  \param num_rows Number of rows.
     *  \param permutation Array containing the permutation indices.
     */
    permutation_matrix_view(const size_t num_rows, const ArrayType& permutation)
        : Parent(num_rows, num_rows, num_rows),
          permutation(permutation) {}

    /*! Construct a \p permutation_matrix_view from a existing \p permutation_matrix.
     *
     *  \param matrix \p permutation_matrix used to create view.
     */
    permutation_matrix_view(permutation_matrix<IndexType,MemorySpace>& matrix)
        : Parent(matrix),
          permutation(matrix.permutation) {}

    /*! Construct a \p permutation_matrix_view from a existing const \p permutation_matrix.
     *
     *  \param matrix \p permutation_matrix used to create view.
     */
    permutation_matrix_view(const permutation_matrix<IndexType,MemorySpace>& matrix)
        : Parent(matrix),
          permutation(matrix.permutation) {}

    /*! Construct a \p permutation_matrix_view from a existing \p permutation_matrix_view.
     *
     *  \param matrix \p permutation_matrix_view used to create view.
     */
    permutation_matrix_view(permutation_matrix_view<ArrayType>& matrix)
        : Parent(matrix),
          permutation(matrix.permutation) {}

    /*! Construct a \p permutation_matrix_view from a existing const \p permutation_matrix_view.
     *
     *  \param matrix \p permutation_matrix_view used to create view.
     */
    permutation_matrix_view(const permutation_matrix_view<ArrayType>& matrix)
        : Parent(matrix),
          permutation(matrix.permutation) {}

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     */
    void resize(const size_t num_rows);

    /*! Permute rows and columns of matrix elements
     *
     *  \param matrix Input matrix to apply symmetric permutation.
     */
    template<typename MatrixType>
    void symmetric_permute(MatrixType& matrix);
};

/* Convenience functions */

/**
 *  This is a convenience function for generating a \p permutation_matrix_view
 *  using individual arrays
 *  \tparam ArrayType row offsets array type
 *
 *  \param num_rows Number of rows.
 *  \param permutation Array containing the permutation.
 *
 *  \return \p permutation_matrix_view constructed using input arrays
 */
template <typename ArrayType>
permutation_matrix_view<ArrayType>
make_permutation_matrix_view(size_t num_rows, ArrayType permutation)
{
    permutation_matrix_view<ArrayType> view(num_rows, permutation);

    return view;
}

/**
 *  This is a convenience function for generating a \p permutation_matrix_view
 *  using individual arrays with explicit value, and memory space
 *  annotations.
 *
 *  \tparam ArrayType permutation array type
 *  \tparam IndexType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p permutation_matrix_view matrix to copy.
 *
 *  \return \p permutation_matrix_view constructed using input arrays.
 */
template <typename ArrayType, typename IndexType, typename MemorySpace>
permutation_matrix_view<ArrayType,IndexType,MemorySpace>
make_permutation_matrix_view(const permutation_matrix_view<ArrayType,IndexType,MemorySpace>& m)
{
    return permutation_matrix_view<ArrayType,IndexType,MemorySpace>(m);
}

/**
 *  This is a convenience function for generating a \p permutation_matrix_view
 *  using an existing \p permutation_matrix.
 *
 *  \tparam IndexType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p permutation_matrix matrix to copy.
 *
 *  \return \p permutation_matrix_view constructed using input arrays.
 */
template <typename IndexType, class MemorySpace>
typename permutation_matrix<IndexType,MemorySpace>::view
make_permutation_matrix_view(permutation_matrix<IndexType,MemorySpace>& m)
{
    return make_permutation_matrix_view(m.num_rows, make_array1d_view(m.permutation));
}

/**
 *  This is a convenience function for generating a const \p permutation_matrix_view
 *  using an existing const \p permutation_matrix.
 *
 *  \tparam IndexType  values type
 *  \tparam MemorySpace memory space of the arrays
 *
 *  \param m Exemplar \p permutation_matrix matrix to copy.
 *
 *  \return \p permutation_matrix_view constructed using input arrays.
 */
template <typename IndexType, class MemorySpace>
typename permutation_matrix<IndexType,MemorySpace>::const_view
make_permutation_matrix_view(const permutation_matrix<IndexType,MemorySpace>& m)
{
    return make_permutation_matrix_view(m.num_rows, make_array1d_view(m.permutation));
}
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/permutation_matrix.inl>
