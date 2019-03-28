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

#include <cusp/system/detail/adl/graph/pseudo_peripheral.h>
#include <cusp/system/detail/generic/graph/pseudo_peripheral.h>

namespace cusp
{
namespace graph
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                         const MatrixType& G,
                         ArrayType& levels)
{
    using cusp::system::detail::generic::pseudo_peripheral_vertex;

    return pseudo_peripheral_vertex(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), G, levels);
}

template<typename MatrixType, typename ArrayType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const MatrixType& G, ArrayType& levels)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System1;
    typedef typename ArrayType::memory_space  System2;

    System1 system1;
    System2 system2;

    return cusp::graph::pseudo_peripheral_vertex(select_system(system1,system2), G, levels);
}

template<typename DerivedPolicy,
         typename MatrixType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                         const MatrixType& G)
{
    using cusp::system::detail::generic::pseudo_peripheral_vertex;

    return pseudo_peripheral_vertex(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), G);
}

template<typename MatrixType>
typename MatrixType::index_type
pseudo_peripheral_vertex(const MatrixType& G)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System;

    System system;

    return cusp::graph::pseudo_peripheral_vertex(select_system(system), G);
}

} // end namespace graph
} // end namespace cusp
