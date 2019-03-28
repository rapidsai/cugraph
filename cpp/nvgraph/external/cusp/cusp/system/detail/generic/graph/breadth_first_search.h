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

template<typename DerivedPolicy,
         typename MatrixType,
         typename ArrayType>
void breadth_first_search(thrust::execution_policy<DerivedPolicy>& exec,
                          const MatrixType& G,
                          const typename MatrixType::index_type src,
                                ArrayType& labels,
                          const bool mark_levels,
                                cusp::known_format)
{
    typedef typename cusp::detail::as_csr_type<MatrixType>::type CsrMatrix;

    CsrMatrix G_csr(G);

    cusp::graph::breadth_first_search(exec, G_csr, src, labels, mark_levels);
}

template<typename DerivedPolicy,
         typename MatrixType,
         typename ArrayType>
void breadth_first_search(thrust::execution_policy<DerivedPolicy>& exec,
                          const MatrixType& G,
                          const typename MatrixType::index_type src,
                                ArrayType& labels,
                          const bool mark_levels)
{
    typedef typename MatrixType::format Format;

    Format format;

    breadth_first_search(thrust::detail::derived_cast(exec), G, src, labels, mark_levels, format);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

