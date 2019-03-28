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

/*! \file hyb_matrix.h
 *  \brief Hybrid ELL/COO matrix format
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/detail/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/detail/type_traits.h>

namespace cusp
{

/*! \cond */
// Forward definitions
template <typename IndexType, typename ValueType, class MemorySpace> class ell_matrix;
template <typename IndexType, typename ValueType, class MemorySpace> class coo_matrix;
template <typename MatrixType1, typename MatrixType2, typename IndexType, typename ValueType, class MemorySpace> class hyb_matrix_view;
/*! \endcond */

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief Hybrid (HYB) representation a sparse matrix
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * The \p hyb_matrix is a combination of the \p ell_matrix and
 * \p coo_matrix formats.  Specifically, the \p hyb_matrix format
 * splits a matrix into two portions, one stored in ELL format
 * and one stored in COO format.
 *
 * While the ELL format is well-suited to vector and SIMD
 * architectures, its efficiency rapidly degrades when the number of
 * nonzeros per matrix row varies.  In contrast, the storage efficiency of
 * the COO format is invariant to the distribution of nonzeros per row, and
 * the use of segmented reduction makes its performance largely invariant
 * as well.  To obtain the advantages of both, we combine these
 * into a hybrid ELL/COO format.
 *
 * The purpose of the HYB format is to store the typical number of
 * nonzeros per row in the ELL data structure and the remaining entries of
 * exceptional rows in the COO format.
 *
 * \note The \p ell_matrix entries must be sorted by column index.
 * \note The \p ell_matrix entries within each row should be shifted to the left.
 * \note The \p coo_matrix entries must be sorted by row index.
 * \note The matrix should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a \p hyb_matrix.
 *  In practice we usually do not construct the HYB format directly and
 *  instead convert from a simpler format such as (COO, CSR) into HYB.
 *
 *  \code
 * // include hyb_matrix header file
 *  #include <cusp/hyb_matrix.h>
 *  #include <cusp/print.h>
 *
 *  int main()
 *  {
 *    // allocate storage for (4,3) matrix with 8 nonzeros
 *    //     ELL portion has 5 nonzeros and storage for 2 nonzeros per row
 *    //     COO portion has 3 nonzeros
 *
 *    cusp::hyb_matrix<int, float, cusp::host_memory> A(3, 4, 5, 3, 2);
 *
 *    // Initialize A to represent the following matrix
 *    // [10  20  30  40]
 *    // [ 0  50   0   0]
 *    // [60   0  70  80]
 *
 *    // A is split into ELL and COO parts as follows
 *    // [10  20  30  40]    [10  20   0   0]     [ 0   0  30  40]
 *    // [ 0  50   0   0]  = [ 0  50   0   0]  +  [ 0   0   0   0]
 *    // [60   0  70  80]    [60   0  70   0]     [ 0   0   0  80]
 *
 *    // Initialize ELL part
 *
 *    // X is used to fill unused entries in the ELL portion of the matrix
 *    const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *    // first row
 *    A.ell.column_indices(0,0) = 0; A.ell.values(0,0) = 10;
 *    A.ell.column_indices(0,1) = 1; A.ell.values(0,1) = 20;
 *
 *    // second row
 *    A.ell.column_indices(1,0) = 1; A.ell.values(1,0) = 50;  // shifted to leftmost position
 *    A.ell.column_indices(1,1) = X; A.ell.values(1,1) =  0;  // padding
 *
 *    // third row
 *    A.ell.column_indices(2,0) = 0; A.ell.values(2,0) = 60;
 *    A.ell.column_indices(2,1) = 2; A.ell.values(2,1) = 70;  // shifted to leftmost position
 *
 *
 *    // Initialize COO part
 *    A.coo.row_indices[0] = 0;  A.coo.column_indices[0] = 2;  A.coo.values[0] = 30;
 *    A.coo.row_indices[1] = 0;  A.coo.column_indices[1] = 3;  A.coo.values[1] = 40;
 *    A.coo.row_indices[2] = 2;  A.coo.column_indices[2] = 3;  A.coo.values[2] = 80;
 *
 *    // print the ELL portion
 *    cusp::print(A.ell);
 *    // print the COO portion
 *    cusp::print(A.coo);
 *    // print the aggregate
 *    cusp::print(A);
 *  }
 *  \endcode
 *
 *  \see \p ell_matrix
 *  \see \p coo_matrix
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class hyb_matrix : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format> Parent;

public:

    /*! \cond */
    typedef cusp::ell_matrix<IndexType,ValueType,MemorySpace> ell_matrix_type;
    typedef cusp::coo_matrix<IndexType,ValueType,MemorySpace> coo_matrix_type;

    typedef typename cusp::hyb_matrix<IndexType, ValueType, MemorySpace> container;

    typedef typename cusp::hyb_matrix_view<
            typename cusp::ell_matrix<IndexType,ValueType,MemorySpace>::view,
            typename cusp::coo_matrix<IndexType,ValueType,MemorySpace>::view,
            IndexType, ValueType, MemorySpace> view;

    typedef typename cusp::hyb_matrix_view<
            typename cusp::ell_matrix<IndexType,ValueType,MemorySpace>::const_view,
            typename cusp::coo_matrix<IndexType,ValueType,MemorySpace>::const_view,
            IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<container,cusp::hyb_format>::view           coo_view_type;
    // TODO : Why does GCC 4.4 fail using const type? Is it necessary?
    typedef typename cusp::detail::coo_view_type<container /*const*/,cusp::hyb_format>::view const_coo_view_type;

    template<typename MemorySpace2>
    struct rebind
    {
        typedef cusp::hyb_matrix<IndexType, ValueType, MemorySpace2> type;
    };
    /*! \endcond */

    /*! Storage for the \p ell_matrix portion.
     */
    ell_matrix_type ell;

    /*! Storage for the \p ell_matrix portion.
     */
    coo_matrix_type coo;

    /*! Construct an empty \p hyb_matrix.
     */
    hyb_matrix(void) {}

    /*! Construct a \p hyb_matrix with a specific shape and separation into ELL and COO portions.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_ell_entries Number of nonzero matrix entries in the ELL portion.
     *  \param num_coo_entries Number of nonzero matrix entries in the ELL portion.
     *  \param num_entries_per_row Maximum number of nonzeros per row in the ELL portion.
     *  \param alignment Amount of padding used to align the ELL data structure (default 32).
     */
    hyb_matrix(IndexType num_rows, IndexType num_cols,
               IndexType num_ell_entries, IndexType num_coo_entries,
               IndexType num_entries_per_row, IndexType alignment = 32)
        : Parent(num_rows, num_cols, num_ell_entries + num_coo_entries),
          ell(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment),
          coo(num_rows, num_cols, num_coo_entries) {}

    // TODO remove default alignment of 32

    /*! Construct a \p hyb_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    hyb_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     */
    void resize(IndexType num_rows, IndexType num_cols,
                IndexType num_ell_entries, IndexType num_coo_entries,
                IndexType num_entries_per_row, IndexType alignment = 32)
    {
        Parent::resize(num_rows, num_cols, num_ell_entries + num_coo_entries);
        ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment);
        coo.resize(num_rows, num_cols, num_coo_entries);
    }

    /*! Swap the contents of two \p hyb_matrix objects.
     *
     *  \param matrix Another \p hyb_matrix with the same IndexType and ValueType.
     */
    void swap(hyb_matrix& matrix)
    {
        Parent::swap(matrix);
        ell.swap(matrix.ell);
        coo.swap(matrix.coo);
    }

    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    hyb_matrix& operator=(const MatrixType& matrix);

}; // class hyb_matrix
/*! \}
 */


/*! \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \{
 */

/**
 * \brief View of a \p hyb_matrix
 *
 * \tparam MatrixType1 Type of \c ell
 * \tparam MatrixType2 Type of \c coo
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p hyb_matrix_view is a sparse matrix view of a \p hyb_matrix
 *  constructed from existing data or iterators. See \p ell_matrix and \p
 *  coo_matrix for format constraints.
 *
 *  \note The matrix entries must be sorted by row index.
 *  \note The matrix should not contain duplicate entries.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p hyb_matrix_view on the host with 6 nonzeros.
 *
 *  \code
 * // include coo_matrix header file
 * #include <cusp/hyb_matrix.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    typedef cusp::ell_matrix<int,float,cusp::host_memory>::view EllMatrixView;
 *    typedef cusp::coo_matrix<int,float,cusp::host_memory>::view CooMatrixView;
 *
 *    // allocate storage for (4,3) matrix with 8 nonzeros
 *    //     ELL portion has 5 nonzeros and storage for 2 nonzeros per row
 *    //     COO portion has 3 nonzeros
 *    cusp::ell_matrix<int,float,cusp::host_memory> ell(4,3,5,2);
 *    cusp::coo_matrix<int,float,cusp::host_memory> coo(4,3,3);
 *
 *    // Initialize A to represent the following matrix
 *    // [10  20  30  40]
 *    // [ 0  50   0   0]
 *    // [60   0  70  80]
 *
 *    // A is split into ELL and COO parts as follows
 *    // [10  20  30  40]    [10  20   0   0]     [ 0   0  30  40]
 *    // [ 0  50   0   0]  = [ 0  50   0   0]  +  [ 0   0   0   0]
 *    // [60   0  70  80]    [60   0  70   0]     [ 0   0   0  80]
 *
 *    // Initialize ELL part
 *
 *    // X is used to fill unused entries in the ELL portion of the matrix
 *    const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *    // first row
 *    ell.column_indices(0,0) = 0; ell.values(0,0) = 10;
 *    ell.column_indices(0,1) = 1; ell.values(0,1) = 20;
 *
 *    // second row
 *    ell.column_indices(1,0) = 1; ell.values(1,0) = 50;  // shifted to leftmost position
 *    ell.column_indices(1,1) = X; ell.values(1,1) =  0;  // padding
 *
 *    // third row
 *    ell.column_indices(2,0) = 0; ell.values(2,0) = 60;
 *    ell.column_indices(2,1) = 2; ell.values(2,1) = 70;  // shifted to leftmost position
 *
 *    // Initialize COO part
 *    coo.row_indices[0] = 0;  coo.column_indices[0] = 2;  coo.values[0] = 30;
 *    coo.row_indices[1] = 0;  coo.column_indices[1] = 3;  coo.values[1] = 40;
 *    coo.row_indices[2] = 2;  coo.column_indices[2] = 3;  coo.values[2] = 80;
 *
 *    // allocate storage for (4,3) matrix with 6 nonzeros
 *    cusp::hyb_matrix_view<EllMatrixView,CooMatrixView>
 *    A(cusp::make_ell_matrix_view(ell),
 *      cusp::make_coo_matrix_view(coo));
 *
 *    // print the constructed hyb_matrix
 *    cusp::print(A);
 *
 *    // change first entry in values array
 *    ell.values(0,0) = -1;
 *
 *    // print the updated matrix view
 *    cusp::print(A);
 * }
 *  \endcode
 */
template <typename MatrixType1,
         typename MatrixType2,
         typename IndexType   = typename MatrixType1::index_type,
         typename ValueType   = typename MatrixType1::value_type,
         typename MemorySpace = typename cusp::minimum_space<typename MatrixType1::memory_space, typename MatrixType2::memory_space>::type >
class hyb_matrix_view : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format> Parent;

public:

    /*! \cond */
    typedef MatrixType1 ell_matrix_type;
    typedef MatrixType2 coo_matrix_type;

    typedef typename cusp::hyb_matrix<IndexType, ValueType, MemorySpace> container;
    typedef typename cusp::hyb_matrix_view<MatrixType1, MatrixType2, IndexType, ValueType, MemorySpace> view;
    typedef typename cusp::hyb_matrix_view<MatrixType1, MatrixType2, IndexType, ValueType, MemorySpace> const_view;

    typedef typename cusp::detail::coo_view_type<view,cusp::hyb_format>::view       coo_view_type;
    typedef typename cusp::detail::coo_view_type<view const,cusp::hyb_format>::view const_coo_view_type;
    /*! \endcond */

    /*! View to the \p ELL portion of the HYB structure.
     */
    ell_matrix_type ell;

    /*! View to the \p COO portion of the HYB structure.
     */
    coo_matrix_type coo;

    /*! Construct an empty \p hyb_matrix_view.
     */
    hyb_matrix_view(void) {}

    /*! Construct a \p hyb_matrix_view with a specific shape and number of nonzero entries
     *  from existing \p ell_matrix and \p coo_matrix matrices.
     *
     *  \tparam OtherMatrixType1 Type of \p ell_matrix used to construct view
     *  \tparam OtherMatrixType2 Type of \p coo_matrix used to construct view
     *
     *  \param ell Matrix containing ELL portion of \p hyb_format.
     *  \param coo Matrix containing COO portion of \p hyb_format.
     */
    template <typename OtherMatrixType1, typename OtherMatrixType2>
    hyb_matrix_view(OtherMatrixType1& ell, OtherMatrixType2& coo)
        : Parent(ell.num_rows, ell.num_cols, ell.num_entries + coo.num_entries), ell(ell), coo(coo) {}

    /*! Construct a \p hyb_matrix_view with a specific shape and number of nonzero entries
     *  from existing const \p ell_matrix and const \p coo_matrix matrices.
     *
     *  \tparam OtherMatrixType1 Type of \p ell_matrix used to construct view
     *  \tparam OtherMatrixType2 Type of \p coo_matrix used to construct view
     *
     *  \param ell Matrix containing ELL portion of \p hyb_format.
     *  \param coo Matrix containing COO portion of \p hyb_format.
     */
    template <typename OtherMatrixType1, typename OtherMatrixType2>
    hyb_matrix_view(const OtherMatrixType1& ell, const OtherMatrixType2& coo)
        : Parent(ell.num_rows, ell.num_cols, ell.num_entries + coo.num_entries), ell(ell), coo(coo) {}

    /*! Construct a \p hyb_matrix_view from a existing \p hyb_matrix.
     *
     *  \param matrix \p hyb_matrix used to create view.
     */
    hyb_matrix_view(hyb_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix), ell(matrix.ell), coo(matrix.coo) {}

    /*! Construct a \p hyb_matrix_view from a existing const \p hyb_matrix.
     *
     *  \param matrix \p hyb_matrix used to create view.
     */
    hyb_matrix_view(const hyb_matrix<IndexType,ValueType,MemorySpace>& matrix)
        : Parent(matrix), ell(matrix.ell), coo(matrix.coo) {}

    /*! Construct a \p hyb_matrix_view from a existing \p hyb_matrix_view.
     *
     *  \param matrix \p hyb_matrix_view used to create view.
     */
    hyb_matrix_view(hyb_matrix_view& matrix)
        : Parent(matrix), ell(matrix.ell), coo(matrix.coo) {}

    /*! Construct a \p hyb_matrix_view from a existing const \p hyb_matrix_view.
     *
     *  \param matrix \p hyb_matrix_view used to create view.
     */
    hyb_matrix_view(const hyb_matrix_view& matrix)
        : Parent(matrix), ell(matrix.ell), coo(matrix.coo) {}

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_ell_entries Number of nonzero matrix entries in the ELL portion.
     *  \param num_coo_entries Number of nonzero matrix entries in the ELL portion.
     *  \param num_entries_per_row Maximum number of nonzeros per row in the ELL portion.
     *  \param alignment Amount of padding used to align the ELL data structure (default 32).
     */
    void resize(size_t num_rows, size_t num_cols,
                size_t num_ell_entries, size_t num_coo_entries,
                size_t num_entries_per_row, size_t alignment = 32);
};
/*! \} // end Views
 */

} // end namespace cusp

#include <cusp/detail/hyb_matrix.inl>
