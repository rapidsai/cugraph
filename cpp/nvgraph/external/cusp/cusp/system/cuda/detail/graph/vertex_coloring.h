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
#include <cusp/detail/format.h>

#include <cusp/array1d.h>
#include <cusp/exception.h>
#include <cusp/detail/type_traits.h>

#include <cusp/graph/vertex_coloring.h>
#include <cusp/system/cuda/detail/execution_policy.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template<typename DerivedPolicy, typename MatrixType, typename ArrayType>
size_t vertex_coloring(cuda::execution_policy<DerivedPolicy>& exec,
                       const MatrixType& G,
                       ArrayType& colors,
                       cusp::csr_format)
{
  typedef typename ArrayType::value_type IndexType;
  typedef typename cusp::detail::as_csr_type<MatrixType,cusp::host_memory>::type CsrHost;

  CsrHost G_host(G);
  cusp::array1d<IndexType,cusp::host_memory> colors_host(colors.size());

  size_t max_colors = cusp::graph::vertex_coloring(G_host, colors_host);
  colors = colors_host;

  return max_colors;
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

