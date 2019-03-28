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
#include <cusp/detail/execution_policy.h>

#include <cusp/multiply.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void galerkin_product(thrust::execution_policy<DerivedPolicy> &exec,
                      const MatrixType1& R,
                      const MatrixType2& A,
                      const MatrixType1& P,
                            MatrixType3& RAP)
{
    // TODO test speed of R * (A * P) vs. (R * A) * P
    MatrixType3 AP;
    cusp::multiply(exec, A, P, AP);
    cusp::multiply(exec, R, AP, RAP);
}

} // end detail

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void galerkin_product(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      const MatrixType1& R,
                      const MatrixType2& A,
                      const MatrixType1& P,
                            MatrixType3& RAP)
{
    using cusp::precond::aggregation::detail::galerkin_product;

    return galerkin_product(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), R, A, P, RAP);
}

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void galerkin_product(const MatrixType1& R,
                      const MatrixType2& A,
                      const MatrixType1& P,
                            MatrixType3& RAP)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;
    typedef typename MatrixType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    return cusp::precond::aggregation::galerkin_product(select_system(system1,system2,system3), R, A, P, RAP);
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

