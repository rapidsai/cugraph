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

#include <cusp/eigen/spectral_radius.h>
#include <cusp/precond/aggregation/system/detail/generic/smooth_prolongator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void smooth_prolongator(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const MatrixType1& S,
                        const MatrixType2& T,
                        MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega)
{
    using cusp::precond::aggregation::detail::smooth_prolongator;

    double rho = rho_Dinv_S;

    if(rho == double(0))
    {
        // compute spectral radius of diag(C)^-1 * C
        rho = cusp::eigen::estimate_rho_Dinv_A(S);
    }

    smooth_prolongator(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), S, T, P, rho, omega);
}

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void smooth_prolongator(const MatrixType1& S,
                        const MatrixType2& T,
                              MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;
    typedef typename MatrixType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    smooth_prolongator(select_system(system1,system2,system3), S, T, P, rho_Dinv_S, omega);
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

