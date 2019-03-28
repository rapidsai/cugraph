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
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>

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
void bicgstab(thrust::execution_policy<DerivedPolicy> &exec,
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
    cusp::detail::temporary_array<ValueType, DerivedPolicy>   p(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>   r(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> r_star(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>   s(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  Mp(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> AMp(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy>  Ms(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> AMs(exec, N);

    // r <- Ax
    cusp::multiply(exec, A, x, r);

    // r <- b - A*x
    blas::axpby(exec, b, r, r, ValueType(1), ValueType(-1));

    // p <- r
    blas::copy(exec, r, p);

    // r_star <- r
    blas::copy(exec, r, r_star);

    ValueType r_r_star_old = blas::dotc(exec, r_star, r);

    while (!monitor.finished(exec, r))
    {
        // Mp = M*p
        cusp::multiply(exec, M, p, Mp);

        // AMp = A*Mp
        cusp::multiply(exec, A, Mp, AMp);

        // alpha = (r_j, r_star) / (A*M*p, r_star)
        ValueType alpha = r_r_star_old / blas::dotc(exec, r_star, AMp);

        // s_j = r_j - alpha * AMp
        blas::axpby(exec, r, AMp, s, ValueType(1), ValueType(-alpha));

        if (monitor.finished(exec, s)) {
            // x += alpha*M*p_j
            blas::axpby(exec, x, Mp, x, ValueType(1), ValueType(alpha));
            break;
        }

        // Ms = M*s_j
        cusp::multiply(exec, M, s, Ms);

        // AMs = A*Ms
        cusp::multiply(exec, A, Ms, AMs);

        // omega = (AMs, s) / (AMs, AMs)
        ValueType omega = blas::dotc(exec, AMs, s) / blas::dotc(exec, AMs, AMs);

        // x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
        blas::axpbypcz(exec, x, Mp, Ms, x, ValueType(1), alpha, omega);

        // r_{j+1} = s_j - omega*A*M*s
        blas::axpby(exec, s, AMs, r, ValueType(1), -omega);

        // beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
        ValueType r_r_star_new = blas::dotc(exec, r_star, r);
        ValueType beta = (r_r_star_new / r_r_star_old) * (alpha / omega);
        r_r_star_old = r_r_star_new;

        // p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
        blas::axpbypcz(exec, r, p, AMp, p, ValueType(1), beta, -beta*omega);

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
void bicgstab(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              const LinearOperator& A,
                    VectorType1& x,
              const VectorType2& b,
                    Monitor& monitor,
                    Preconditioner& M)
{
    using cusp::krylov::bicg_detail::bicgstab;

    return bicgstab(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, b, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void bicgstab(const LinearOperator& A,
                    VectorType1& x,
              const VectorType2& b,
                    Monitor& monitor,
                    Preconditioner& M)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename VectorType1::memory_space    System2;

    System1 system1;
    System2 system2;

    return cusp::krylov::bicgstab(select_system(system1,system2), A, x, b, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor>
void bicgstab(const LinearOperator& A,
                    VectorType1& x,
              const VectorType2& b,
                    Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    return cusp::krylov::bicgstab(A, x, b, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2>
void bicgstab(const LinearOperator& A,
                    VectorType1& x,
              const VectorType2& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::monitor<ValueType> monitor(b);

    return cusp::krylov::bicgstab(A, x, b, monitor);
}

} // end namespace krylov
} // end namespace cusp

