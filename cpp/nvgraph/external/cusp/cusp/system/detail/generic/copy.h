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


#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/type_traits.h>

#include <cusp/detail/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::coo_format,
          cusp::coo_format);

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::csr_format,
          cusp::csr_format);

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::dia_format,
          cusp::dia_format);

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::ell_format,
          cusp::ell_format);

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::hyb_format,
          cusp::hyb_format);

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::array1d_format,
          cusp::array1d_format);

template <typename DerivedPolicy, typename T1, typename T2, typename Orientation>
void copy_array2d(thrust::execution_policy<DerivedPolicy>& exec,
                  const T1& src, T2& dst,
                  Orientation, Orientation);

template <typename DerivedPolicy, typename T1, typename T2,
          typename Orientation1, typename Orientation2>
void copy_array2d(thrust::execution_policy<DerivedPolicy>& exec,
                  const T1& src, T2& dst,
                  Orientation1, Orientation2);

template <typename DerivedPolicy, typename T1, typename T2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const T1& src, T2& dst,
          cusp::array2d_format,
          cusp::array2d_format);

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

#include <cusp/system/detail/generic/copy.inl>

