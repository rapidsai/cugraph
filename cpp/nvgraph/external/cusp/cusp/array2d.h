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

/*! \file
 * \brief 2D array of elements that may reside in "host" or "device" memory space
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/memory.h>
#include <cusp/detail/format.h>
#include <cusp/array1d.h>

#include <cusp/detail/array2d_format_utils.h>
#include <cusp/detail/matrix_base.h>

#include <thrust/functional.h>

namespace cusp
{

// TODO document mapping of (i,j) onto values[pitch * i + j] or values[pitch * j + i]
// TODO document that array2d operations will try to respect .pitch of destination argument

/*! \addtogroup arrays Arrays
 */

/*! \addtogroup array_containers Array Containers
 *  \ingroup arrays
 *  \{
 */

/**
 * \brief The array2d class is a 2D vector container that may contain elements
 * stored in "host" or "device" memory space
 *
 * \tparam T value_type of the array
 * \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 * \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *
 * \par Overview
 * A array2d vector is a container that supports random access to elements.
 * The memory associated with a array2d vector may reside in either "host"
 * or "device" memory depending on the supplied allocator embedded in the
 * MemorySpace template argument. array2d vectors represent 2D matrices in
 * either row-major or column-major format.
 *
 * \par Example
 * \code
 * // include cusp array2d header file
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *   // Allocate a array of size 2 in "host" memory
 *   cusp::array2d<int,cusp::host_memory> a(3,3);
 *
 *   // Set the entries in the matrix using shorthand operator
 *   a(0,0) = 0; a(0,1) = 1; a(0,2) = 2;
 *   a(1,0) = 3; a(1,1) = 4; a(1,2) = 5;
 *   a(2,0) = 6; a(2,1) = 7; a(2,2) = 8;
 *
 *   // Allocate a seceond array2d in "device" memory that is
 *   // a copy of the first but in column major
 *   cusp::array2d<int,cusp::device_memory,cusp::column_major> b(a);
 *
 *   // print row-major layout of data
 *   // [0, 1, 2, 3, 4, 5, 6, 7, 8]
 *   cusp::print(a.values);
 *   // print column-major layout of data
 *   // [0, 3, 6, 1, 4, 7, 2, 5, 8]
 *   cusp::print(b.values);
 * }
 * \endcode
 */
template<typename ValueType, typename MemorySpace, typename Orientation = cusp::row_major>
class array2d : public cusp::detail::matrix_base<int,ValueType,MemorySpace,cusp::array2d_format>
{
private:

    typedef cusp::detail::matrix_base<int,ValueType,MemorySpace,cusp::array2d_format> Parent;

public:
    /*! \cond */
    typedef Orientation orientation;

    template<typename MemorySpace2>
    struct rebind {
        typedef cusp::array2d<ValueType, MemorySpace2, Orientation> type;
    };

    typedef cusp::array1d<ValueType, MemorySpace> values_array_type;
    typedef cusp::array2d<ValueType, MemorySpace, Orientation> container;

    typedef cusp::array2d_view<typename values_array_type::view, Orientation> view;
    typedef cusp::array2d_view<typename values_array_type::const_view, Orientation> const_view;

    typedef cusp::detail::row_or_column_view<
        typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value
      > row_view_type;

    typedef typename row_view_type::ArrayType row_view;

    typedef cusp::detail::row_or_column_view<
        typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value
      > column_view_type;

    typedef typename column_view_type::ArrayType column_view;

    typedef cusp::detail::row_or_column_view<
        typename values_array_type::const_iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value
      > const_row_view_type;

    typedef typename const_row_view_type::ArrayType const_row_view;

    typedef cusp::detail::row_or_column_view<
        typename values_array_type::const_iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value
      > const_column_view_type;

    typedef typename const_column_view_type::ArrayType const_column_view;

    typedef typename cusp::detail::transpose_orientation<Orientation>::type transpose_orientation;
    typedef cusp::array2d_view<typename values_array_type::const_view, transpose_orientation> transpose_const_view_type;
    /*! \endcond */

    /*! The stride between consecutive elements along the major dimension
     */
    size_t pitch;

    /*! 1D array of values
     */
    values_array_type values;

    /*! This constructor creates an empty \p array2d vector.
     */
    array2d(void)
        : Parent(), pitch(0), values(0) {}

    /*! This constructor creates a array2d vector with the given
     *  shape.
     *  \param num_rows The number of elements to initially create.
     *  \param num_cols The number of elements to initially create.
     */
    array2d(size_t num_rows, size_t num_cols)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
          values(num_rows * num_cols) {}

    /*! This constructor creates a array2d vector with the given
     *  shape and fills the entries with a given value.
     *  \param num_rows The number of array2d rows.
     *  \param num_cols The number of array2d columns.
     *  \param value The initial value of all entries.
     */
    array2d(size_t num_rows, size_t num_cols, const ValueType& value)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
          values(num_rows * num_cols, value) {}

    /*! This constructor creates a array2d vector with the given
     *  shape, fills the entries with a given value and sets the pitch
     *  \param num_rows The number of array2d rows.
     *  \param num_cols The number of array2d columns.
     *  \param value The initial value of all entries.
     *  \param pitch The stride between entries in the major dimension.
     */
    array2d(const size_t num_rows, const size_t num_cols, const ValueType& value, const size_t pitch);

    /*! This constructor creates a array2d vector from another matrix
     *  \tparam MatrixType Type of the input matrix
     *  \param matrix Input matrix used to create array2d matrix
     */
    template <typename MatrixType>
    array2d(const MatrixType& matrix);

    /*! Subscript access to the data contained in this array2d.
     *  \param i Row index for which data should be accessed.
     *  \param j Column index for which data should be accessed.
     *  \return Read/write reference to data.
     */
    typename values_array_type::reference operator()(const size_t i, const size_t j);

    /*! Subscript access to the data contained in this array2d.
     *  \param i Row index for which data should be accessed.
     *  \param j Column index for which data should be accessed.
     *  \return Read reference to data.
     */
    typename values_array_type::const_reference operator()(const size_t i, const size_t j) const;

    /*! This method will resize this array2d to the specified dimensions.
     *  If the number of total entries is smaller than this
     *  array2d's current size this array2d is truncated, otherwise this
     *  array2d is extended with the value of new entries undefined.
     *
     *  \param num_rows The number of rows this array2d should contain
     *  \param num_cols The number of columns this array2d should contain
     */
    void resize(const size_t num_rows, const size_t num_cols);

    /*! This method will resize this array2d to the specified dimensions.
     *  If the number of total entries is smaller than this
     *  array2d's current size this array2d is truncated, otherwise this
     *  array2d is extended with the value of new entries undefined.
     *
     *  \param num_rows The number of rows this array2d should contain
     *  \param num_cols The number of columns this array2d should contain
     *  \param pitch The stride between major dimension entries this array2d
     *  should contain
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t pitch);

    /*! This method swaps the contents of this array2d with another array2d.
     *  \param matrix The array2d with which to swap.
     */
    void swap(array2d& matrix);

    /*! This method generates a array1d_view of row i of this array2d matrix
     * \param i The row index used to create array1d_view
     * \return array1d_view of row i
     */
    row_view row(const size_t i);

    /*! This method generates a array1d_view of column i of this array2d matrix
     * \param i The column index used to create array1d_view
     * \return array1d_view of column i
     */
    column_view column(const size_t i);

    /*! This method generates a const array1d_view of row i of this array2d matrix
     * \param i The row index used to create array1d_view
     * \return const array1d_view of row i
     */
    const_row_view row(const size_t i) const;

    /*! This method generates a const array1d_view of column i of this array2d matrix
     * \param i The column index used to create array1d_view
     * \return const array1d_view of column i
     */
    const_column_view column(const size_t i) const;

    /*! Assign operator copies from an exemplar \p array2d container.
     *  \param matrix The \p array2d container to copy.
     *  \return array2d copy of input matrix
     */
    array2d& operator=(const array2d& matrix);

    /*! Assign operator copies from an exemplar matrix container.
     *  \tparam MatrixType The type of matrix to copy.
     *  \param matrix The matrix to copy.
     *  \return array2d copy of input matrix.
     */
    template <typename MatrixType>
    array2d& operator=(const MatrixType& matrix);

    /*! Construct a array2d_view of current matrix with opposite orientation.
     *  Mainly useful for interacting with BLAS routines.
     *  \return array2d_view with new orientation.
     */
    transpose_const_view_type T(void) const;

}; // class array2d
/*! \}
 */

/*! \addtogroup array_views Array Views
 *  \ingroup arrays
 *  \{
 */

/**
 * \brief The array2d_view is a view of a array2d container.
 *
 * \tparam Iterator The iterator type used to encapsulate the underlying data.
 *
 * \par Overview
 * array2d_view is a container that wraps existing iterators in array2d
 * datatypes to interoperate with cusp algorithms.
 *
 * \par Example
 * \code
 * // include cusp array2d header file
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *   // Define the container type
 *   typedef cusp::array2d<int, cusp::device_memory> Array;
 *
 *   // Get reference to array view type
 *   typedef Array::view View;
 *
 *   // Allocate array1d container with 10 elements
 *   Array v(3,3);
 *
 *   // Set the entries in the matrix using shorthand operator
 *   v(0,0) = 0; v(0,1) = 1; v(0,2) = 2;
 *   v(1,0) = 3; v(1,1) = 4; v(1,2) = 5;
 *   v(2,0) = 6; v(2,1) = 7; v(2,2) = 8;
 *
 *   // Create array2d_view, v_view, from array2d v
 *   View v_view(v);
 *
 *   v_view(0,0) = -1; v_view(1,1) = -2; v_view(2,2) = -3;
 *
 *   // print the updated array2d matrix
 *   cusp::print(v);
 * }
 * \endcode
 */
template<typename ArrayView, class Orientation = cusp::row_major>
class array2d_view
    : public cusp::detail::matrix_base<int,
                                       typename ArrayView::value_type,
                                       typename ArrayView::memory_space,
                                       cusp::array2d_format>
{
private:

    typedef cusp::detail::matrix_base<int,
                                      typename ArrayView::value_type,
                                      typename ArrayView::memory_space,
                                      cusp::array2d_format> Parent;

public:

    /*! \cond */
    typedef Orientation orientation;

    typedef ArrayView values_array_type;

    typedef cusp::array2d<typename Parent::value_type, typename Parent::memory_space, Orientation> container;

    typedef cusp::array2d_view<ArrayView, Orientation> view;

    typedef cusp::detail::row_or_column_view<
        typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value
      > row_view_type;
    typedef typename row_view_type::ArrayType row_view;

    typedef cusp::detail::row_or_column_view<
        typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value
      > column_view_type;
    typedef typename column_view_type::ArrayType column_view;

    typedef typename cusp::detail::transpose_orientation<Orientation>::type transpose_orientation;
    typedef cusp::array2d_view<ArrayView, transpose_orientation> transpose_const_view_type;
    /*! \endcond */

    /*! The stride between consecutive elements along the major dimension.
     */
    size_t pitch;

    /*! 1D array of values.
     */
    values_array_type values;

    /*! This constructor creates an empty \p array2d vector.
     */
    array2d_view(void)
        : Parent(), pitch(0), values(0) {}

    /*! This constructor creates a array2d_view from another array2d_view.
     *  \param a array2d_view used to create this array2d_view.
     */
    array2d_view(const array2d_view& a)
        : Parent(a), pitch(a.pitch), values(a.values) {}

    // TODO handle different Orientation (pitch = major)
    //template <typename Array2, typename Orientation2>
    //array2d_view(const array2d_view<Array2,Orientation2>& A)

    /*! This constructor creates a array2d_view from a array2d container.
     *  \param a array2d used to construct this array2d_view.
     */
    array2d_view(array2d<typename Parent::value_type, typename Parent::memory_space, orientation>& a)
        : Parent(a), pitch(a.pitch), values(a.values) {}

    /*! This constructor creates a array2d vector with the given
     *  shape, fills the entries with a given value and sets the pitch.
     *  \tparam Array2 The type of values used to construct this array2d_view.
     *  \param num_rows The number of array2d_view rows.
     *  \param num_cols The number of array2d_view columns.
     *  \param values The initial value of all entries.
     *  \param pitch The stride between entries in the major dimension.
     */
    template <typename Array2>
    array2d_view(size_t num_rows, size_t num_cols, size_t pitch, const Array2& values)
        : Parent(num_rows, num_cols, num_rows * num_cols), pitch(pitch), values(values) {}

    /*! Subscript access to the data contained in this array2d_view.
     *  \param i Row index for which data should be accessed.
     *  \param j Column index for which data should be accessed.
     *  \return Read reference to data.
     */
    typename values_array_type::reference operator()(const size_t i, const size_t j) const;

    /*! This method will resize this array2d_view to the specified dimensions.
     *  If the number of total entries is smaller than this
     *  array2d_view's current size this array2d_view is truncated, otherwise this
     *  array2d_view is extended with the value of new entries undefined.
     *
     *  \param num_rows The number of rows this array2d_view should contain.
     *  \param num_cols The number of columns this array2d_view should contain.
     */
    void resize(const size_t num_rows, const size_t num_cols);

    /*! This method will resize this array2d_view to the specified dimensions.
     *  If the number of total entries is smaller than this
     *  array2d_view's current size this array2d_view is truncated, otherwise this
     *  array2d_view is extended with the value of new entries undefined.
     *
     *  \param num_rows The number of rows this array2d_view should contain.
     *  \param num_cols The number of columns this array2d_view should contain.
     *  \param pitch The stride between major dimension entries this
     *  array2d_view should contain.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t pitch);

    /*! This method generates a array1d_view of row i of this array2d_view matrix
     * \param i The row index used to create array1d_view
     * \return array1d_view of row \p i
     */
    row_view row(size_t i);

    /*! This method generates a array1d_view of column i of this array2d_view matrix
     * \param i The column index used to create array1d_view
     * \return array1d_view of column \p i
     */
    column_view column(size_t i);

    /*! This method generates a const array1d_view of row i of this array2d_view matrix
     * \param i The row index used to create array1d_view
     * \return const array1d_view of row \p i
     */
    row_view row(size_t i) const;

    /*! This method generates a const array1d_view of column i of this array2d_view matrix
     * \param i The column index used to create array1d_view
     * \return const array1d_view of column \p i
     */
    column_view column(size_t i) const;

    /*! Construct a array2d_view of current matrix with opposite orientation.
     *  Mainly useful for interacting with BLAS routines.
     *  \return array2d_view with new orientation.
     */
    transpose_const_view_type T(void) const;
}; // end array2d_view class


/** This is a convenience function for generating an array2d_view
 *  using specified shape and values
 *  \tparam Iterator array1d_view iterator used to construct array2d_view
 *  \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *  \param num_rows The number of array2d_view rows.
 *  \param num_cols The number of array2d_view columns.
 *  \param pitch The stride between entries in the major dimension.
 *  \param values array1d_view containing this array2d_view's values.
 */
template <typename Iterator, typename Orientation>
array2d_view<typename cusp::array1d_view<Iterator>,Orientation>
make_array2d_view(size_t num_rows, size_t num_cols, size_t pitch, const cusp::array1d_view<Iterator>& values, Orientation)
{
    return array2d_view<typename cusp::array1d_view<Iterator>,Orientation>(num_rows, num_cols, pitch, values);
}

/**
 *  This is a convenience function for generating an array2d_view
 *  using an array2d_view
 *  \tparam Array Type of input array containing values
 *  \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *  \param v The array2d_view used to construct array2d_view
 *  \return array2d_view constructed using input array2d_view
 */
template <typename Array, typename Orientation>
array2d_view<Array,Orientation>
make_array2d_view(const array2d_view<Array, Orientation>& v)
{
    return array2d_view<Array,Orientation>(v);
}

/**
 *  This is a convenience function for generating an array2d_view
 *  using an array2d_view
 *  \tparam T value_type of the array
 *  \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 *  \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *  \param v The array2d used to construct array2d_view
 *  \return array2d_view constructed using input array2d
 */
template<typename T, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<T,MemorySpace>::iterator>, Orientation>
make_array2d_view(cusp::array2d<T,MemorySpace,Orientation>& v)
{
    return cusp::make_array2d_view(v.num_rows, v.num_cols, v.pitch, cusp::make_array1d_view(v.values), Orientation());
}

/**
 *  This is a convenience function for generating an array2d_view
 *  using an array2d
 *  \tparam T value_type of the array
 *  \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 *  \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *  \param v The const array2d used to construct array2d_view
 *  \return constant array2d_view constructed using input array2d
 */
template<typename T, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<T,MemorySpace>::const_iterator>, Orientation>
make_array2d_view(const cusp::array2d<T,MemorySpace,Orientation>& v)
{
    return cusp::make_array2d_view(v.num_rows, v.num_cols, v.pitch, cusp::make_array1d_view(v.values), Orientation());
}
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/array2d.inl>

