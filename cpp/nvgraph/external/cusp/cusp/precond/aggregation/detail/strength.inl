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

#include <cusp/precond/aggregation/system/detail/generic/symmetric_strength.h>
#include <cusp/precond/aggregation/system/detail/generic/evolution_strength.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void symmetric_strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                      const MatrixType1& A, MatrixType2& S, const double theta)
{
    using cusp::precond::aggregation::detail::symmetric_strength_of_connection;

    symmetric_strength_of_connection(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, S, theta);
}

template <typename MatrixType1, typename MatrixType2>
void symmetric_strength_of_connection(const MatrixType1& A, MatrixType2& S, const double theta)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::precond::aggregation::symmetric_strength_of_connection(select_system(system1,system2), A, S, theta);
}

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2, typename ArrayType>
void evolution_strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                      const MatrixType1& A, MatrixType2& S, const ArrayType& B,
                                      const double rho_DinvA, const double epsilon)
{
    using cusp::precond::aggregation::detail::evolution_strength_of_connection;

    evolution_strength_of_connection(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, S, B, rho_DinvA, epsilon);
}

template <typename MatrixType1, typename MatrixType2, typename ArrayType>
void evolution_strength_of_connection(const MatrixType1& A, MatrixType2& S, const ArrayType& B,
                                      const double rho_DinvA, const double epsilon)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::precond::aggregation::evolution_strength_of_connection(select_system(system1,system2), A, S, B, rho_DinvA, epsilon);
}


template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                            const MatrixType1& A,
                                  MatrixType2& S)
{
    return cusp::precond::aggregation::symmetric_strength_of_connection(exec, A, S);
}

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            const MatrixType1& A,
                                  MatrixType2& S)
{
    return strength_of_connection(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, S);
}

template <typename MatrixType1,
          typename MatrixType2>
void strength_of_connection(const MatrixType1& A,
                                  MatrixType2& S)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::precond::aggregation::strength_of_connection(select_system(system1,system2), A, S);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                            const MatrixType1& A,
                                  MatrixType2& S,
                                  sa_level<MatrixType3>& level)
{
    cusp::precond::aggregation::symmetric_strength_of_connection(exec, A, S);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            const MatrixType1& A,
                                  MatrixType2& S,
                                  sa_level<MatrixType3>& level)
{
    return strength_of_connection(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, S, level);
}

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void strength_of_connection(const MatrixType1& A,
                                  MatrixType2& S,
                                  sa_level<MatrixType3>& level)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::precond::aggregation::strength_of_connection(select_system(system1,system2), A, S, level);
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

