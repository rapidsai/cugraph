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

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cusp/exception.h>
#include <cusp/graph/hilbert_curve.h>

#include <cusp/system/detail/adl/graph/hilbert_curve.h>
#include <cusp/system/detail/generic/graph/hilbert_curve.h>

namespace cusp
{
namespace graph
{

template <typename DerivedPolicy,
          typename Array2dType,
          typename ArrayType>
void hilbert_curve(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                   const Array2dType& G,
                   const size_t num_parts,
                   ArrayType& parts)
{
    using cusp::system::detail::generic::hilbert_curve;

    hilbert_curve(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), G, num_parts, parts);
}

template<typename Array2dType,
         typename ArrayType>
void hilbert_curve(const Array2dType& G,
                   const size_t num_parts,
                   ArrayType& parts)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2dType::memory_space System1;
    typedef typename ArrayType::memory_space   System2;

    System1 system1;
    System2 system2;

    cusp::graph::hilbert_curve(select_system(system1,system2), G, num_parts, parts);
}

} // end namespace graph
} // end namespace cusp

