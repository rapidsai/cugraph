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

/*! \file sort.inl
 *  \brief Inline file for sort.h.
 */

#include <cusp/detail/config.h>
#include <cusp/system/detail/generic/sort.h>
#include <cusp/system/detail/adl/sort.h>

#include <thrust/system/detail/generic/select_system.h>

namespace cusp
{

template <typename DerivedPolicy,
          typename ArrayType>
void counting_sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   ArrayType& v,
                   typename ArrayType::value_type min,
                   typename ArrayType::value_type max)
{
    using cusp::system::detail::generic::counting_sort;

    return counting_sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), v, min, max);
}

template <typename ArrayType>
void counting_sort(ArrayType& v,
                   typename ArrayType::value_type min,
                   typename ArrayType::value_type max)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::counting_sort(select_system(system), v, min, max);
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          ArrayType1& keys,
                          ArrayType2& vals,
                          typename ArrayType1::value_type min,
                          typename ArrayType1::value_type max)
{
    using cusp::system::detail::generic::counting_sort_by_key;

    return counting_sort_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys, vals, min, max);
}

template <typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min,
                          typename ArrayType1::value_type max)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::counting_sort_by_key(select_system(system1,system2), keys, vals, min, max);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ArrayType1& row_indices,
                 ArrayType2& column_indices,
                 ArrayType3& values,
                 typename ArrayType1::value_type min_row,
                 typename ArrayType1::value_type max_row)
{
    using cusp::system::detail::generic::sort_by_row;

    return sort_by_row(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), row_indices, column_indices, values, min_row, max_row);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row(ArrayType1& row_indices,
                 ArrayType2& column_indices,
                 ArrayType3& values,
                 typename ArrayType1::value_type min_row,
                 typename ArrayType1::value_type max_row)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    return cusp::sort_by_row(select_system(system1,system2,system3), row_indices, column_indices, values, min_row, max_row);
}

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row_and_column(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ArrayType1& row_indices,
                            ArrayType2& column_indices,
                            ArrayType3& values,
                            typename ArrayType1::value_type min_row,
                            typename ArrayType1::value_type max_row,
                            typename ArrayType2::value_type min_col,
                            typename ArrayType2::value_type max_col)
{
    using cusp::system::detail::generic::sort_by_row_and_column;

    return sort_by_row_and_column(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), row_indices, column_indices, values, min_row, max_row, min_col, max_col);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void sort_by_row_and_column(ArrayType1& row_indices,
                            ArrayType2& column_indices,
                            ArrayType3& values,
                            typename ArrayType1::value_type min_row,
                            typename ArrayType1::value_type max_row,
                            typename ArrayType2::value_type min_col,
                            typename ArrayType2::value_type max_col)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    return cusp::sort_by_row_and_column(select_system(system1,system2,system3), row_indices, column_indices, values, min_row, max_row, min_col, max_col);
}

} // end namespace cusp

