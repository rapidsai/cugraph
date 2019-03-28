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

#include <cusp/detail/array2d_format_utils.h>
#include <cusp/detail/format.h>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/detail/type_traits.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename T1, typename T2>
void copy_matrix_dimensions(const T1& src, T2& dst)
{
    dst.num_rows    = src.num_rows;
    dst.num_cols    = src.num_cols;
    dst.num_entries = src.num_entries;
}

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::coo_format,
          cusp::coo_format)
{
    copy_matrix_dimensions(src, dst);
    cusp::copy(exec, src.row_indices,    dst.row_indices);
    cusp::copy(exec, src.column_indices, dst.column_indices);
    cusp::copy(exec, src.values,         dst.values);
}

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::csr_format,
          cusp::csr_format)
{
    copy_matrix_dimensions(src, dst);
    cusp::copy(exec, src.row_offsets,    dst.row_offsets);
    cusp::copy(exec, src.column_indices, dst.column_indices);
    cusp::copy(exec, src.values,         dst.values);
}

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::dia_format,
          cusp::dia_format)
{
    copy_matrix_dimensions(src, dst);
    cusp::copy(exec, src.diagonal_offsets, dst.diagonal_offsets);
    cusp::copy(exec, src.values,           dst.values);
}

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::ell_format,
          cusp::ell_format)
{
    copy_matrix_dimensions(src, dst);
    cusp::copy(exec, src.column_indices, dst.column_indices);
    cusp::copy(exec, src.values,         dst.values);
}

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::hyb_format,
          cusp::hyb_format)
{
    copy_matrix_dimensions(src, dst);
    cusp::copy(exec, src.ell, dst.ell);
    cusp::copy(exec, src.coo, dst.coo);
}

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::array1d_format,
          cusp::array1d_format)
{
    dst.resize(src.size());
    thrust::copy(exec, src.begin(), src.end(), dst.begin());
}

// same orientation
template <typename DerivedPolicy, typename T1, typename T2, typename Orientation>
void copy_array2d(thrust::execution_policy<DerivedPolicy>& exec,
                  const T1& src, T2& dst, Orientation, Orientation)
{
    // will preserve destination pitch if possible
    dst.resize(src.num_rows, src.num_cols);

    if (dst.pitch == src.pitch)
    {
        cusp::copy(src.values, dst.values);
    }
    else
    {
        thrust::counting_iterator<size_t> begin(0);
        thrust::counting_iterator<size_t> end(src.num_entries);

        cusp::detail::logical_to_physical_functor<size_t, Orientation> func1(src.num_rows, src.num_cols, src.pitch);
        cusp::detail::logical_to_physical_functor<size_t, Orientation> func2(dst.num_rows, dst.num_cols, dst.pitch);

        thrust::copy(exec,
                     thrust::make_permutation_iterator(src.values.begin(), thrust::make_transform_iterator(begin, func1)),
                     thrust::make_permutation_iterator(src.values.begin(), thrust::make_transform_iterator(end,   func1)),
                     thrust::make_permutation_iterator(dst.values.begin(), thrust::make_transform_iterator(begin, func2)));
    }
}

template <typename DerivedPolicy, typename T1, typename T2,
          typename Orientation1, typename Orientation2>
void copy_array2d(thrust::execution_policy<DerivedPolicy>& exec,
                  const T1& src, T2& dst, Orientation1, Orientation2)
{
    // note: pitch does not carry over when orientation differs
    dst.resize(src.num_rows, src.num_cols);

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end(src.num_entries);

    // prefer coalesced writes to coalesced reads
    cusp::detail::logical_to_other_physical_functor<size_t, Orientation2, Orientation1> func1(src.num_rows, src.num_cols, src.pitch);
    cusp::detail::logical_to_physical_functor      <size_t, Orientation2>               func2(dst.num_rows, dst.num_cols, dst.pitch);

    thrust::copy(exec,
                 thrust::make_permutation_iterator(src.values.begin(), thrust::make_transform_iterator(begin, func1)),
                 thrust::make_permutation_iterator(src.values.begin(), thrust::make_transform_iterator(end,   func1)),
                 thrust::make_permutation_iterator(dst.values.begin(), thrust::make_transform_iterator(begin, func2)));
}

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::array2d_format,
          cusp::array2d_format)
{
    copy_array2d(exec, src, dst, typename T1::orientation(), typename T2::orientation());
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void copy(thrust::execution_policy<DerivedPolicy> &exec,
             const SourceType& src, DestinationType& dst)
{
    typedef typename SourceType::format      Format1;
    typedef typename DestinationType::format Format2;

    Format1 format1;
    Format2 format2;

    copy(exec, src, dst, format1, format2);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp


