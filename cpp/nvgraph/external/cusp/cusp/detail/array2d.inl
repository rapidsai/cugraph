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

namespace cusp
{

// Forward definitions
template <typename T1, typename T2> void convert(const T1&, T2&);

//////////////////
// Constructors //
//////////////////

// construct from a different matrix
template<typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
array2d<ValueType,MemorySpace,Orientation>
::array2d(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
}

template<typename ValueType, class MemorySpace, class Orientation>
array2d<ValueType,MemorySpace,Orientation>
::array2d(const size_t num_rows, const size_t num_cols, const ValueType& value, size_t pitch)
    : Parent(num_rows, num_cols, num_rows * num_cols),
      pitch(pitch),
      values(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()), value)
{
    if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
        throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");
}

//////////////////////
// Member Functions //
//////////////////////

template <typename ValueType, class MemorySpace, class Orientation>
array2d<ValueType,MemorySpace,Orientation>&
array2d<ValueType,MemorySpace,Orientation>
::operator=(const array2d<ValueType,MemorySpace,Orientation>& matrix)
{
    cusp::convert(matrix, *this);

    return *this;
}

template <typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
array2d<ValueType,MemorySpace,Orientation>&
array2d<ValueType,MemorySpace,Orientation>
::operator=(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);

    return *this;
}

template <typename ValueType, class MemorySpace, class Orientation>
typename array2d<ValueType,MemorySpace,Orientation>::transpose_const_view_type
array2d<ValueType,MemorySpace,Orientation>
::T(void) const
{
    return transpose_const_view_type(this->num_cols,
                                     this->num_rows,
                                     this->pitch,
                                     this->values);
}


template <typename ValueType, class MemorySpace, class Orientation>
typename array2d<ValueType,MemorySpace,Orientation>::values_array_type::reference
array2d<ValueType,MemorySpace,Orientation>
::operator()(const size_t i, const size_t j)
{
    return values[cusp::detail::index_of(i, j, pitch, orientation())];
}

template <typename ValueType, class MemorySpace, class Orientation>
typename array2d<ValueType,MemorySpace,Orientation>::values_array_type::const_reference
array2d<ValueType,MemorySpace,Orientation>
::operator()(const size_t i, const size_t j) const
{
    return values[cusp::detail::index_of(i, j, pitch, orientation())];
}

template <typename ValueType, class MemorySpace, class Orientation>
void
array2d<ValueType,MemorySpace,Orientation>
::resize(const size_t num_rows, const size_t num_cols)
{
    // preserve .pitch if possible
    if (this->num_rows == num_rows && this->num_cols == num_cols)
        return;

    resize(num_rows, num_cols, cusp::detail::minor_dimension(num_rows, num_cols, orientation()));
}

template <typename ValueType, class MemorySpace, class Orientation>
void
array2d<ValueType,MemorySpace,Orientation>
::resize(const size_t num_rows, const size_t num_cols, const size_t pitch)
{
    if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
        throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");

    values.resize(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()));

    this->num_rows    = num_rows;
    this->num_cols    = num_cols;
    this->pitch       = pitch;
    this->num_entries = num_rows * num_cols;
}

template <typename ValueType, class MemorySpace, class Orientation>
void
array2d<ValueType,MemorySpace,Orientation>
::swap(array2d& matrix)
{
    Parent::swap(matrix);
    thrust::swap(this->pitch, matrix.pitch);
    values.swap(matrix.values);
}

template <typename ValueType, class MemorySpace, class Orientation>
typename array2d<ValueType,MemorySpace,Orientation>::row_view
array2d<ValueType,MemorySpace,Orientation>
::row(const size_t i)
{
    return row_view_type::get_array(*this, i);
}

template <typename ValueType, class MemorySpace, class Orientation>
typename array2d<ValueType,MemorySpace,Orientation>::column_view
array2d<ValueType,MemorySpace,Orientation>
::column(const size_t i)
{
    return column_view_type::get_array(*this, i);
}

template <typename ValueType, class MemorySpace, class Orientation>
typename array2d<ValueType,MemorySpace,Orientation>::const_row_view
array2d<ValueType,MemorySpace,Orientation>
::row(const size_t i) const
{
    return const_row_view_type::get_array(*this, i);
}

template <typename ValueType, class MemorySpace, class Orientation>
typename array2d<ValueType,MemorySpace,Orientation>::const_column_view
array2d<ValueType,MemorySpace,Orientation>
::column(const size_t i) const
{
    return const_column_view_type::get_array(*this, i);
}

template<typename ArrayView, class Orientation>
typename array2d_view<ArrayView,Orientation>::values_array_type::reference
array2d_view<ArrayView,Orientation>
::operator()(const size_t i, const size_t j) const
{
    return values[cusp::detail::index_of(i, j, pitch, orientation())];
}

template<typename ArrayView, class Orientation>
void
array2d_view<ArrayView,Orientation>
::resize(const size_t num_rows, const size_t num_cols)
{
    // preserve .pitch if possible
    if (this->num_rows == num_rows && this->num_cols == num_cols)
        return;

    resize(num_rows, num_cols, cusp::detail::minor_dimension(num_rows, num_cols, orientation()));
}

template<typename ArrayView, class Orientation>
void
array2d_view<ArrayView,Orientation>
::resize(const size_t num_rows, const size_t num_cols, const size_t pitch)
{
    if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
        throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");

    values.resize(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()));

    this->num_rows    = num_rows;
    this->num_cols    = num_cols;
    this->pitch       = pitch;
    this->num_entries = num_rows * num_cols;
}

template<typename ArrayView, class Orientation>
typename array2d_view<ArrayView,Orientation>::row_view
array2d_view<ArrayView,Orientation>
::row(const size_t i)
{
    return row_view_type::get_array(*this, i);
}

template<typename ArrayView, class Orientation>
typename array2d_view<ArrayView,Orientation>::column_view
array2d_view<ArrayView,Orientation>
::column(const size_t i)
{
    return column_view_type::get_array(*this, i);
}

template<typename ArrayView, class Orientation>
typename array2d_view<ArrayView,Orientation>::row_view
array2d_view<ArrayView,Orientation>
::row(const size_t i) const
{
    return row_view_type::get_array(*this, i);
}

template<typename ArrayView, class Orientation>
typename array2d_view<ArrayView,Orientation>::column_view
array2d_view<ArrayView,Orientation>
::column(const size_t i) const
{
    return column_view_type::get_array(*this, i);
}

template<typename ArrayView, class Orientation>
typename array2d_view<ArrayView,Orientation>::transpose_const_view_type
array2d_view<ArrayView,Orientation>
::T(void) const
{
    return transpose_const_view_type(this->num_cols,
                                     this->num_rows,
                                     this->pitch,
                                     this->values);
}

/////////////////////
// Other Functions //
/////////////////////

namespace detail
{
template <typename Array1, typename Array2>
bool array2d_equal(const Array1& lhs, const Array2& rhs)
{
    typedef typename Array1::orientation Orientation1;
    typedef typename Array2::orientation Orientation2;

    if (lhs.num_rows != rhs.num_rows || lhs.num_cols != rhs.num_cols)
        return false;

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end(lhs.num_entries);

    cusp::detail::logical_to_physical_functor      <size_t, Orientation1>               func1(lhs.num_rows, lhs.num_cols, lhs.pitch);
    cusp::detail::logical_to_other_physical_functor<size_t, Orientation1, Orientation2> func2(rhs.num_rows, rhs.num_cols, rhs.pitch);

    // like a boss
    return thrust::equal(thrust::make_permutation_iterator(lhs.values.begin(), thrust::make_transform_iterator(begin, func1)),
                         thrust::make_permutation_iterator(lhs.values.begin(), thrust::make_transform_iterator(end,   func1)),
                         thrust::make_permutation_iterator(rhs.values.begin(), thrust::make_transform_iterator(begin, func2)));
}

} // end namespace detail

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator==(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
    return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename Array1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator==(const array2d_view<Array1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
    return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator==(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
    return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename Array1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator==(const array2d_view<Array1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
    return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator!=(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}

template<typename Array1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator!=(const array2d_view<Array1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator!=(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}

template<typename Array1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator!=(const array2d_view<Array1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}

} // end namespace cusp

#include <cusp/convert.h>

