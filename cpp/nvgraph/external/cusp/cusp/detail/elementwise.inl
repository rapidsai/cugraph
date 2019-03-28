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

/*! \file elementwise.inl
 *  \brief Inline file for elementwise.h.
 */

#include <cusp/detail/config.h>

#include <cusp/system/detail/adl/elementwise.h>
#include <cusp/system/detail/generic/elementwise.h>

#include <thrust/system/detail/generic/select_system.h>

namespace cusp
{

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename BinaryFunction>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A,
                 const MatrixType2& B,
                       MatrixType3& C,
                       BinaryFunction op)
{
    using cusp::system::detail::generic::elementwise;

    return elementwise(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C, op);
}

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename BinaryFunction>
void elementwise(const MatrixType1& A,
                 const MatrixType2& B,
                       MatrixType3& C,
                       BinaryFunction op)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;
    typedef typename MatrixType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    return cusp::elementwise(select_system(system1,system2,system3), A, B, C, op);
}

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void add(const MatrixType1& A,
         const MatrixType2& B,
               MatrixType3& C)
{
    typedef typename MatrixType1::value_type   ValueType;
    typedef thrust::plus<ValueType>            Op;

    Op op;

    cusp::elementwise(A, B, C, op);
}

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void subtract(const MatrixType1& A,
              const MatrixType2& B,
                    MatrixType3& C)
{
    typedef typename MatrixType1::value_type   ValueType;
    typedef thrust::minus<ValueType>           Op;

    Op op;

    cusp::elementwise(A, B, C, op);
}

} // end namespace cusp

