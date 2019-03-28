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

#include <cusp/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cusp/exception.h>
#include <cusp/graph/vertex_coloring.h>

#include <cusp/system/detail/adl/graph/vertex_coloring.h>
#include <cusp/system/detail/generic/graph/vertex_coloring.h>

namespace cusp
{
namespace graph
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
size_t vertex_coloring(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                       const MatrixType& G,
                             ArrayType& colors)
{
    using cusp::system::detail::generic::vertex_coloring;

    return vertex_coloring(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), G, colors);
}

template<typename MatrixType,
         typename ArrayType>
size_t vertex_coloring(const MatrixType& G,
                             ArrayType& colors)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System1;
    typedef typename ArrayType::memory_space  System2;

    System1 system1;
    System2 system2;

    return cusp::graph::vertex_coloring(select_system(system1,system2), G, colors);
}

} // end namespace graph
} // end namespace cusp

