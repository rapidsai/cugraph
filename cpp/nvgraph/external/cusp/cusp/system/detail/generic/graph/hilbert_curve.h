/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <cusp/exception.h>
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

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void hilbert_curve(thrust::execution_policy<DerivedPolicy>& exec,
                   const Array2d& coord,
                   const size_t num_parts,
                         Array1d& parts)
{
  throw cusp::not_implemented_exception("No generic Hilbert curve");
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
