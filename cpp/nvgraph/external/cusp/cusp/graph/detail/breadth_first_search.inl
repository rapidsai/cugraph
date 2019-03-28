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
#include <cusp/graph/breadth_first_search.h>

#include <cusp/system/detail/adl/graph/breadth_first_search.h>
#include <cusp/system/detail/generic/graph/breadth_first_search.h>

namespace cusp
{
namespace graph
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType>
void breadth_first_search(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                          const MatrixType& G,
                          const typename MatrixType::index_type src,
                          ArrayType& labels,
                          const bool mark_levels)
{
    using cusp::system::detail::generic::breadth_first_search;

    return breadth_first_search(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), G, src, labels, mark_levels);
}

template<typename MatrixType,
         typename ArrayType>
void breadth_first_search(const MatrixType& G,
                          const typename MatrixType::index_type src,
                          ArrayType& labels,
                          const bool mark_levels)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System1;
    typedef typename ArrayType::memory_space  System2;

    System1 system1;
    System2 system2;

    return cusp::graph::breadth_first_search(select_system(system1,system2), G, src, labels, mark_levels);
}

} // end namespace graph
} // end namespace cusp

