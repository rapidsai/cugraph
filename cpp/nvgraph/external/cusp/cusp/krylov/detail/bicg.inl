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

#include <cusp/blas/blas.h>

#include <cusp/detail/temporary_array.h>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{
namespace bicg_detail
{

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void bicg(thrust::execution_policy<DerivedPolicy> &exec,
          const LinearOperator& A,
          const LinearOperator& At,
                VectorType1& x,
          const VectorType2& b,
                Monitor& monitor,
                Preconditioner& M,
                Preconditioner& Mt)
{
    typedef typename LinearOperator::value_type           ValueType;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;

    // allocate workspace
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  y(exec, N);

    cusp::detail::temporary_array<ValueType, DerivedPolicy>  p(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  p_star(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  q(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  q_star(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  r(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  r_star(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  z(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  z_star(exec, N);

    // y <- Ax
    cusp::multiply(exec, A, x, y);

    // r <- b - A*x
    blas::axpby(exec, b, y, r, ValueType(1), ValueType(-1));

    if(monitor.finished(r)) {
        return;
    }

    // r_star <- r
    blas::copy(exec, r, r_star);

    // z = M r
    cusp::multiply(exec, M, r, z);

    // z_star = Mt r_star
    cusp::multiply(exec, Mt, r_star, z_star);

    // rho = (z,r_star)
    ValueType rho = blas::dotc(exec, z, r_star);

    // p <- z
    blas::copy(exec, z, p);

    // p_star <- r
    blas::copy(exec, z_star, p_star);

    while (1)
    {
        // q = A p
        cusp::multiply(exec, A, p, q);

        // q_star = At p_star
        cusp::multiply(exec, At, p_star, q_star);

        // alpha = (rho) / (p_star, q)
        ValueType alpha = rho / blas::dotc(exec, p_star, q);

        // x += alpha*p
        blas::axpby(exec, x, p, x, ValueType(1), ValueType(alpha));

        // r -= alpha*q
        blas::axpby(exec, r, q, r, ValueType(1), ValueType(-alpha));

        // r_star -= alpha*q_star
        blas::axpby(exec, r_star, q_star, r_star, ValueType(1), ValueType(-alpha));

        if (monitor.finished(exec, r)) {
            break;
        }

        // z = M r
        cusp::multiply(exec, M, r, z);

        // z_star = Mt r_star
        cusp::multiply(exec, Mt, r_star, z_star);

        ValueType prev_rho = rho;

        // rho = (z,r_star)
        rho = blas::dotc(exec, z, r_star);

        if(rho == ValueType(0)) {
            // Failure!
            // TODO: Make the failure more apparent to the user
            break;
        }

        ValueType beta = rho/prev_rho;

        // p = beta*p + z
        blas::axpby(exec, p, z, p, ValueType(beta), ValueType(1));

        // p_star = beta*p_star + z_star
        blas::axpby(exec, p_star, z_star, p_star, ValueType(beta), ValueType(1));

        ++monitor;
    }
}

} // end bicg_detail namespace

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void bicg(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const LinearOperator& A,
          const LinearOperator& At,
                VectorType1& x,
          const VectorType2& b,
                Monitor& monitor,
                Preconditioner& M,
                Preconditioner& Mt)
{
    using cusp::krylov::bicg_detail::bicg;

    return bicg(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, At, x, b, monitor, M, Mt);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void bicg(const LinearOperator& A,
          const LinearOperator& At,
                VectorType1& x,
          const VectorType2& b,
                Monitor& monitor,
                Preconditioner& M,
                Preconditioner& Mt)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename VectorType2::memory_space    System2;

    System1 system1;
    System2 system2;

    return cusp::krylov::bicg(select_system(system1,system2), A, At, x, b, monitor, M, Mt);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor>
void bicg(const LinearOperator& A,
          const LinearOperator& At,
                VectorType1& x,
          const VectorType2& b,
                Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    return cusp::krylov::bicg(A, At, x, b, monitor, M, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2>
void bicg(const LinearOperator& A,
          const LinearOperator& At,
                VectorType1& x,
          const VectorType2& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::monitor<ValueType> monitor(b);

    return cusp::krylov::bicg(A, At, x, b, monitor);
}

} // end namespace krylov
} // end namespace cusp

