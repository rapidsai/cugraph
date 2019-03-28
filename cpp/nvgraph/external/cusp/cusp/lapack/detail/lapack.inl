/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

/*! \file lapack.inl
 *  \brief Definition of lapack interface routines
 */

#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/lapack/detail/generic.h>

namespace cusp
{
namespace lapack
{

template<typename DerivedPolicy, typename Array2d, typename Array1d>
void getrf( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            Array2d& A, Array1d& piv )
{
    using cusp::lapack::generic::getrf;

    return getrf(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, piv);
}

template<typename Array2d, typename Array1d>
void getrf( Array2d& A, Array1d& piv )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::lapack::getrf(select_system(system1,system2), A, piv);
}

template<typename DerivedPolicy, typename Array2d>
void potrf( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            Array2d& A, char uplo )
{
    using cusp::lapack::generic::potrf;

    return potrf(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, uplo);
}

template<typename Array2d>
void potrf( Array2d& A, char uplo )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System;

    System system;

    return cusp::lapack::potrf(select_system(system), A);
}

template<typename DerivedPolicy, typename Array2d, typename Array1d>
void sytrf( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            Array2d& A, Array1d& piv, char uplo )
{
    using cusp::lapack::generic::sytrf;

    return sytrf(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, piv, uplo);
}

template<typename Array2d, typename Array1d>
void sytrf( Array2d& A, Array1d& piv, char uplo )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::lapack::sytrf(select_system(system1,system2), A, piv, uplo);
}

template<typename DerivedPolicy, typename Array2d, typename Array1d>
void getrs( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            const Array2d& A, const Array1d& piv, Array2d& B, char trans )
{
    using cusp::lapack::generic::getrs;

    return getrs(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, piv, B, trans);
}

template<typename Array2d, typename Array1d>
void getrs( const Array2d& A, const Array1d& piv, Array2d& B, char trans )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::lapack::getrs(select_system(system1,system2), A, piv, B, trans);
}

template<typename DerivedPolicy, typename Array2d>
void potrs( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            const Array2d& A, Array2d& B, char uplo )
{
    using cusp::lapack::generic::potrs;

    return potrs(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, uplo);
}

template<typename Array2d>
void potrs( const Array2d& A, Array2d& B, char uplo )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System;

    System system;

    return cusp::lapack::potrs(select_system(system), A, B, uplo);
}

template<typename DerivedPolicy, typename Array2d, typename Array1d>
void sytrs( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            const Array2d& A, const Array1d& piv, Array2d& B, char uplo )
{
    using cusp::lapack::generic::sytrs;

    return sytrs(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, piv, B, uplo);
}

template<typename Array2d, typename Array1d>
void sytrs( const Array2d& A, const Array1d& piv, Array2d& B, char uplo )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::lapack::sytrs(select_system(system1, system2), A, piv, B, uplo);
}

template<typename DerivedPolicy, typename Array2d>
void trtrs( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            const Array2d& A, Array2d& B, char uplo, char trans, char diag )
{
    using cusp::lapack::generic::trtrs;

    return trtrs(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, uplo, trans, diag);
}

template<typename Array2d>
void trtrs( const Array2d& A, Array2d& B, char uplo, char trans, char diag )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System;

    System system;

    return cusp::lapack::trtrs(select_system(system), A, B, uplo, trans, diag);
}

template<typename DerivedPolicy, typename Array2d>
void trtri( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            Array2d& A, char uplo, char diag )
{
    using cusp::lapack::generic::trtri;

    return trtri(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, uplo, diag);
}

template<typename Array2d>
void trtri( Array2d& A, char uplo, char diag )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System;

    System system;

    return cusp::lapack::trtri(select_system(system), A, uplo, diag);
}

template<typename DerivedPolicy, typename Array2d, typename Array1d>
void syev( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array2d& A, Array1d& eigvals, Array2d& eigvecs, char uplo )
{
    using cusp::lapack::generic::syev;

    return syev(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, eigvals, eigvecs, uplo);
}

template<typename Array2d, typename Array1d>
void syev( const Array2d& A, Array1d& eigvals, Array2d& eigvecs, char uplo )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::lapack::syev(select_system(system1, system2), A, eigvals, eigvecs, uplo);
}

template<typename DerivedPolicy, typename Array1d1, typename Array1d2, typename Array1d3, typename Array2d>
void stev( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array1d1& alphas, const Array1d2& betas, Array1d3& eigvals, Array2d& eigvecs, char job )
{
    using cusp::lapack::generic::stev;

    return stev(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), alphas, betas, eigvals, eigvecs, job);
}

template<typename Array1d1, typename Array1d2, typename Array1d3, typename Array2d>
void stev( const Array1d1& alphas, const Array1d2& betas, Array1d3& eigvals, Array2d& eigvecs, char job )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1d1::memory_space System1;
    typedef typename Array1d2::memory_space System2;
    typedef typename Array1d3::memory_space System3;
    typedef typename Array2d::memory_space  System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;

    return cusp::lapack::stev(select_system(system1, system2, system3, system4), alphas, betas, eigvals, eigvecs, job);
}

template<typename DerivedPolicy, typename Array2d1, typename Array2d2, typename Array1d, typename Array2d3>
void sygv( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array2d1& A, const Array2d2& B, Array1d& eigvals, Array2d3& eigvecs )
{
    using cusp::lapack::generic::sygv;

    return sygv(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, eigvals, eigvecs);
}

template<typename Array2d1, typename Array2d2, typename Array1d, typename Array2d3>
void sygv( const Array2d1& A, const Array2d2& B, Array1d& eigvals, Array2d3& eigvecs )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;
    typedef typename Array2d3::memory_space System3;
    typedef typename Array1d::memory_space  System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;

    return cusp::lapack::sygv(select_system(system1, system2, system3, system4), A, B, eigvals, eigvecs);
}

template<typename DerivedPolicy, typename Array2d, typename Array1d>
void gesv( const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array2d& A, Array2d& B, Array1d& pivots )
{
    using cusp::lapack::generic::gesv;

    return gesv(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, pivots);
}

template<typename Array2d, typename Array1d>
void gesv( const Array2d& A, Array2d& B, Array1d& pivots )
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1d::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::lapack::gesv(select_system(system1, system2), A, B, pivots);
}

} // end namespace lapack
} // end namespace cusp

