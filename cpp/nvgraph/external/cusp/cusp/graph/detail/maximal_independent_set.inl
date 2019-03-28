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
#include <cusp/graph/maximal_independent_set.h>

#include <cusp/system/detail/adl/graph/maximal_independent_set.h>
#include <cusp/system/detail/generic/graph/maximal_independent_set.h>

namespace cusp
{
namespace graph
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
size_t maximal_independent_set(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                               const MatrixType& G,
                                     ArrayType& stencil,
                               const size_t k)
{
    using cusp::system::detail::generic::maximal_independent_set;

    return maximal_independent_set(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), G, stencil, k);
}

template <typename MatrixType,
          typename ArrayType>
size_t maximal_independent_set(const MatrixType& G,
                                     ArrayType& stencil,
                               const size_t k)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System1;
    typedef typename ArrayType::memory_space  System2;

    System1 system1;
    System2 system2;

    return cusp::graph::maximal_independent_set(select_system(system1,system2), G, stencil, k);
}

} // end namespace graph
} // end namespace cusp

