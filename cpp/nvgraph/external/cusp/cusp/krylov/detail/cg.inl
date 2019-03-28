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

#include <cusp/array1d.h>
#include <cusp/linear_operator.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>

#include <cusp/detail/temporary_array.h>

#include <cusp/blas/blas.h>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{
namespace cg_detail
{

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void cg(thrust::execution_policy<DerivedPolicy> &exec,
        const LinearOperator& A,
              VectorType1& x,
        const VectorType2& b,
              Monitor& monitor,
              Preconditioner& M)
{
    typedef typename LinearOperator::value_type           ValueType;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;

    // allocate workspace
    cusp::detail::temporary_array<ValueType, DerivedPolicy> y(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> z(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> r(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> p(exec, N);

    assert(A.num_rows == A.num_cols);        // sanity check

    // y <- Ax
    cusp::multiply(exec, A, x, y);

    // r <- b - A*x
    cusp::blas::axpby(exec, b, y, r, ValueType(1), ValueType(-1));

    // z <- M*r
    cusp::multiply(exec, M, r, z);

    // p <- z
    blas::copy(exec, z, p);

    // rz = <r^H, z>
    ValueType rz = blas::dotc(exec, r, z);

    while (!monitor.finished(exec, r))
    {
        // y <- Ap
        cusp::multiply(exec, A, p, y);

        // alpha <- <r,z>/<y,p>
        ValueType alpha =  rz / blas::dotc(exec, y, p);

        // x <- x + alpha * p
        blas::axpy(exec, p, x, alpha);

        // r <- r - alpha * y
        blas::axpy(exec, y, r, -alpha);

        // z <- M*r
        cusp::multiply(exec, M, r, z);

        ValueType rz_old = rz;

        // rz = <r^H, z>
        rz = blas::dotc(exec, r, z);

        // beta <- <r_{i+1},r_{i+1}>/<r,r>
        ValueType beta = rz / rz_old;

        // p <- r + beta*p
        blas::axpby(exec, z, p, p, ValueType(1), beta);

        ++monitor;
    }
}

} // end cg_detail namespace

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void cg(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        const LinearOperator& A,
              VectorType1& x,
        const VectorType2& b,
              Monitor& monitor,
              Preconditioner& M)
{
    using cusp::krylov::cg_detail::cg;

    return cg(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, b, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void cg(const LinearOperator& A,
              VectorType1& x,
        const VectorType2& b,
              Monitor& monitor,
              Preconditioner& M)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename VectorType2::memory_space    System2;

    System1 system1;
    System2 system2;

    return cusp::krylov::cg(select_system(system1,system2), A, x, b, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor>
void cg(const LinearOperator& A,
              VectorType1& x,
        const VectorType2& b,
              Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    return cusp::krylov::cg(A, x, b, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2>
void cg(const LinearOperator& A,
              VectorType1& x,
        const VectorType2& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::monitor<ValueType> monitor(b);

    return cusp::krylov::cg(A, x, b, monitor);
}

} // end namespace krylov
} // end namespace cusp

