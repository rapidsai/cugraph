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

/*! \file counting_sort.h
 *  \brief Sequential implementations of counting sort algorithms.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>

#include <cusp/array1d.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

template <typename DerivedPolicy, typename ArrayType>
void counting_sort(thrust::cpp::execution_policy<DerivedPolicy>& exec,
                   ArrayType& keys,
                   typename ArrayType::value_type min,
                   typename ArrayType::value_type max)
{
    typedef typename ArrayType::value_type IndexType;

    if(min < IndexType(0))
      throw cusp::invalid_input_exception("counting_sort min element less than 0");

    if(max < min)
      throw cusp::invalid_input_exception("counting_sort min element less than max element");

    if(min > 0) min = 0;

    // compute the number of bins
    size_t size = max - min;

    // allocate temporary arrays
    cusp::detail::temporary_array<size_t,    DerivedPolicy> counts(exec, size + 2);
    cusp::detail::temporary_array<IndexType, DerivedPolicy> temp_keys(exec, keys.begin(), keys.end());

    // initialize counts
    thrust::fill(exec, counts.begin(), counts.end(), size_t(0));

    // count the number of occurences of each key
    for(size_t i = 0; i < keys.size(); i++)
      counts[keys[i] + 1]++;

    // scan the sum of each bin
    for(size_t i = 0; i < size; i++)
      counts[i + 1] += counts[i];

    // generate output in sorted order
    for(size_t i = 0; i < keys.size(); i++)
      keys[counts[temp_keys[i]]++] = temp_keys[i];
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(thrust::cpp::execution_policy<DerivedPolicy>& exec,
                          ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min,
                          typename ArrayType1::value_type max)
{
    typedef typename ArrayType1::value_type IndexType1;
    typedef typename ArrayType2::value_type IndexType2;

    if(min < IndexType1(0))
      throw cusp::invalid_input_exception("counting_sort min element less than 0");

    if(max < min)
      throw cusp::invalid_input_exception("counting_sort min element less than max element");

    if(keys.size() < vals.size())
      throw cusp::invalid_input_exception("counting_sort keys.size() less than vals.size()");

    if(min > 0) min = 0;

    // compute the number of bins
    size_t size = max - min;

    // allocate temporary arrays
    cusp::detail::temporary_array<size_t,     DerivedPolicy> counts(exec, size + 2);
    cusp::detail::temporary_array<IndexType1, DerivedPolicy> temp_keys(exec, keys.begin(), keys.end());
    cusp::detail::temporary_array<IndexType2, DerivedPolicy> temp_vals(exec, vals.begin(), vals.end());

    // initialize counts
    thrust::fill(exec, counts.begin(), counts.end(), size_t(0));

    // count the number of occurences of each key
    for(size_t i = 0; i < keys.size(); i++)
      counts[keys[i] + 1]++;

    // scan the sum of each bin
    for(size_t i = 0; i < size; i++)
      counts[i + 1] += counts[i];

    // generate output in sorted order
    for(size_t i = 0; i < keys.size(); i++)
    {
      keys[counts[temp_keys[i]]] = temp_keys[i];
      vals[counts[temp_keys[i]]++] = temp_vals[i];
    }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

