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

/*! \file array2d_format_utils.h
 *  \brief Array2d formatting and indexing routines
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/iterator/strided_iterator.h>

namespace cusp
{

// forward definitions
template<typename, class> class array2d_view;

namespace detail
{
// (i,j) -> major dimension
// (i,j) -> minor dimension
// logical n -> (i,j)
// (i,j) -> logical n
// (i,j) -> physical n
// logical n -> physical n
// logical n -> physical n (translated)

template <typename IndexType>
__host__ __device__
IndexType minor_dimension(IndexType num_rows, IndexType num_cols, row_major)    {
    return num_cols;
}

template <typename IndexType>
__host__ __device__
IndexType minor_dimension(IndexType num_rows, IndexType num_cols, column_major) {
    return num_rows;
}

template <typename IndexType>
__host__ __device__
IndexType major_dimension(IndexType num_rows, IndexType num_cols, row_major)    {
    return num_rows;
}

template <typename IndexType>
__host__ __device__
IndexType major_dimension(IndexType num_rows, IndexType num_cols, column_major) {
    return num_cols;
}

// convert logical linear index into a logical (i,j) index
template <typename IndexType>
__host__ __device__
IndexType linear_index_to_row_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, row_major)    {
    return linear_index / num_cols;
}

template <typename IndexType>
__host__ __device__
IndexType linear_index_to_col_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, row_major)    {
    return linear_index % num_cols;
}

template <typename IndexType>
__host__ __device__
IndexType linear_index_to_row_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, column_major)    {
    return linear_index % num_rows;
}

template <typename IndexType>
__host__ __device__
IndexType linear_index_to_col_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, column_major)    {
    return linear_index / num_rows;
}

// convert a logical (i,j) index into a physical linear index
template <typename IndexType, typename Bool>
__host__ __device__ inline
IndexType index_of(IndexType i, IndexType j, IndexType pitch, row_major_base<Bool>)    {
    return i * pitch + j;
}

template <typename IndexType, typename Bool>
__host__ __device__ inline
IndexType index_of(IndexType i, IndexType j, IndexType pitch, column_major_base<Bool>) {
    return j * pitch + i;
}

template <typename IndexType, typename Orientation>
__host__ __device__
IndexType logical_to_physical(IndexType linear_index, IndexType num_rows, IndexType num_cols, IndexType pitch, Orientation)
{
    IndexType i = linear_index_to_row_index(linear_index, num_rows, num_cols, Orientation());
    IndexType j = linear_index_to_col_index(linear_index, num_rows, num_cols, Orientation());

    return index_of(i, j, pitch, Orientation());
}

// convert logical linear index in the source into a physical linear index in the destination
template <typename IndexType, typename Orientation1, typename Orientation2>
__host__ __device__
IndexType logical_to_other_physical(IndexType linear_index, IndexType num_rows, IndexType num_cols, IndexType pitch, Orientation1, Orientation2)
{
    IndexType i = linear_index_to_row_index(linear_index, num_rows, num_cols, Orientation1());
    IndexType j = linear_index_to_col_index(linear_index, num_rows, num_cols, Orientation1());

    return index_of(i, j, pitch, Orientation2());
}

// functors
template <typename IndexType, typename Orientation>
struct logical_to_physical_functor : public thrust::unary_function<IndexType,IndexType>
{
    IndexType num_rows, num_cols, pitch;

    __host__ __device__
    logical_to_physical_functor(void) {}

    logical_to_physical_functor(IndexType num_rows, IndexType num_cols, IndexType pitch)
        : num_rows(num_rows), num_cols(num_cols), pitch(pitch) {}

    __host__ __device__
    IndexType operator()(const IndexType i) const
    {
        return logical_to_physical(i, num_rows, num_cols, pitch, Orientation());
    }
};

// convert logical linear index in the (tranposed) destination into a physical index in the source
template <typename IndexType, typename Orientation1, typename Orientation2>
struct transpose_index_functor : public thrust::unary_function<IndexType,IndexType>
{
    IndexType num_rows, num_cols, pitch; // source dimensions

    __host__ __device__
    transpose_index_functor(void) {}

    transpose_index_functor(IndexType num_rows, IndexType num_cols, IndexType pitch)
        : num_rows(num_rows), num_cols(num_cols), pitch(pitch) {}

    __host__ __device__
    IndexType operator()(IndexType linear_index)
    {
        IndexType i = linear_index_to_row_index(linear_index, num_cols, num_rows, Orientation2());
        IndexType j = linear_index_to_col_index(linear_index, num_cols, num_rows, Orientation2());

        return index_of(j, i, pitch, Orientation1());
    }
};


template <typename IndexType, typename Orientation1, typename Orientation2>
struct logical_to_other_physical_functor : public thrust::unary_function<IndexType,IndexType>
{
    IndexType num_rows, num_cols, pitch;

    __host__ __device__
    logical_to_other_physical_functor(void) {}

    __host__ __device__
    logical_to_other_physical_functor(IndexType num_rows, IndexType num_cols, IndexType pitch)
        : num_rows(num_rows), num_cols(num_cols), pitch(pitch) {}

    __host__ __device__
    IndexType operator()(const IndexType i) const
    {
        return logical_to_other_physical(i, num_rows, num_cols, pitch, Orientation1(), Orientation2());
    }
};

template <typename Iterator, bool same_orientation>
struct row_or_column_view {};

template <typename Iterator>
struct row_or_column_view<Iterator,true>
{
    typedef cusp::array1d_view<Iterator> ArrayType;

    template< typename Array >
    static ArrayType get_array(Array& A, size_t i) {

        ArrayType x(A.values.begin() + A.pitch * i,
                    A.values.begin() + A.pitch * i +
                    cusp::detail::minor_dimension(A.num_rows, A.num_cols, typename Array::orientation()));

        return x;
    }
};

template <typename Iterator>
struct row_or_column_view<Iterator,false>
{
    typedef typename cusp::strided_iterator<Iterator> StrideType;
    typedef cusp::array1d_view<typename StrideType::iterator> ArrayType;

    template< typename Array >
    static ArrayType get_array(Array& A, size_t i) {

        cusp::strided_iterator<Iterator> strided_range(A.values.begin() + i,
                                                       A.values.begin() + A.pitch *
                                                       cusp::detail::major_dimension(A.num_rows, A.num_cols, typename Array::orientation()),
                                                       A.pitch);

        ArrayType x(strided_range.begin(), strided_range.end());

        return x;
    }
};

template <typename Orientation, typename IsTranspose = typename Orientation::transpose>
struct transpose_orientation
      : thrust::detail::eval_if<
          thrust::detail::is_same<Orientation, cusp::row_major_base<IsTranspose> >::value,
          thrust::detail::identity_<cusp::column_major_base<thrust::detail::not_<IsTranspose> > >,
          thrust::detail::identity_<cusp::row_major_base<thrust::detail::not_<IsTranspose> > >
        > // if orientation
{};

} // end namespace detail

} // end namespace cusp
