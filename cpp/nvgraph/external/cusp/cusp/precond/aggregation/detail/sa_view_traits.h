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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename IndexType, typename ValueType, typename MemorySpace>
struct select_sa_matrix_type
{
  typedef cusp::csr_matrix<IndexType,ValueType,MemorySpace> CSRType;
  typedef cusp::coo_matrix<IndexType,ValueType,MemorySpace> COOType;

  typedef typename thrust::detail::eval_if<
        thrust::detail::is_convertible<MemorySpace, cusp::host_memory>::value
      , thrust::detail::identity_<CSRType>
      , thrust::detail::identity_<COOType>
    >::type type;
};

template <typename MatrixType>
struct select_sa_matrix_view
{
  typedef typename MatrixType::memory_space MemorySpace;
  typedef typename MatrixType::format       Format;

  typedef typename thrust::detail::eval_if<
        thrust::detail::is_convertible<MemorySpace, cusp::host_memory>::value
      , typename thrust::detail::eval_if<
          thrust::detail::is_same<Format, cusp::csr_format>::value
          , thrust::detail::identity_<typename MatrixType::const_view>
          , cusp::detail::as_csr_type<MatrixType>
          >
      , thrust::detail::identity_<typename MatrixType::const_coo_view_type>
    >::type type;
};

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

