/*
 *  Copyright 2011 The Regents of the University of California
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
#include <cusp/complex.h>
#include <cusp/linear_operator.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>

#include <cusp/blas/blas.h>

#include <cusp/detail/temporary_array.h>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{
namespace gmres_detail
{

template <typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          typename ValueType4>
void ApplyPlaneRotation(      ValueType1& dx,
                              ValueType2& dy,
                        const ValueType3& cs,
                        const ValueType4& sn)
{
    ValueType3 temp = cs * dx + sn * dy;
    dy = -cusp::conj(sn) * dx + cs * dy;
    dx = temp;
}

template <typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          typename ValueType4>
void GeneratePlaneRotation(const ValueType1& dx,
                           const ValueType2& dy,
                                 ValueType3& cs,
                                 ValueType4& sn)
{
    typedef typename cusp::norm_type<ValueType1>::type NormType;

    if (dx == ValueType1(0)) {
        cs = ValueType3(0);
        sn = ValueType4(1);
    } else {
        NormType scale = cusp::abs(dx) + cusp::abs(dy);
        NormType norm = scale * std::sqrt(cusp::abs(dx / scale) * cusp::abs(dx / scale) +
                                          cusp::abs(dy / scale) * cusp::abs(dy / scale));
        ValueType4 alpha = dx / cusp::abs(dx);
        cs = cusp::abs(dx) / norm;
        sn = alpha * cusp::conj(dy) / norm;
    }
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3>
void PlaneRotation(LinearOperator &H,
                   VectorType1    &cs,
                   VectorType2    &sn,
                   VectorType3    &s,
                   const int i)
{
    for (int k = 0; k < i; k++) {
        ApplyPlaneRotation(H(k, i), H(k + 1, i), cs[k], sn[k]);
    }

    GeneratePlaneRotation(H(i, i), H(i + 1, i), cs[i], sn[i]);
    ApplyPlaneRotation(H(i, i), H(i + 1, i), cs[i], sn[i]);
    ApplyPlaneRotation(s[i], s[i + 1], cs[i], sn[i]);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void gmres(thrust::execution_policy<DerivedPolicy> &exec,
           const LinearOperator &A,
                 VectorType1 &x,
           const VectorType2 &b,
           const size_t restart,
                 Monitor &monitor,
                 Preconditioner &M)
{
    typedef typename LinearOperator::value_type ValueType;
    typedef typename cusp::norm_type<ValueType>::type NormType;
    typedef typename cusp::minimum_space<
    typename LinearOperator::memory_space, typename VectorType1::memory_space,
             typename Preconditioner::memory_space>::type MemorySpace;

    assert(A.num_rows == A.num_cols);  // sanity check

    const size_t N = A.num_rows;
    const int R = restart;
    int i, j, k;
    NormType beta = 0;

    // allocate workspace
    cusp::detail::temporary_array<ValueType, DerivedPolicy>   w(exec, N);
    // Arnoldi matrix pos 0
    cusp::detail::temporary_array<ValueType, DerivedPolicy>   V0(exec, N);
    // Arnoldi matrix
    cusp::array2d<ValueType, MemorySpace, cusp::column_major> V(N, R + 1, ValueType(0.0));

    // duplicate copy of s on GPU
    cusp::detail::temporary_array<ValueType, DerivedPolicy> sDev(exec, R + 1);

    // HOST WORKSPACE
    cusp::host_memory host_exec;
    cusp::array2d<ValueType, cusp::host_memory, cusp::column_major> H(R + 1, R);  // Hessenberg matrix
    cusp::array1d<ValueType, cusp::host_memory> s(R + 1);
    cusp::array1d<ValueType, cusp::host_memory> cs(R);
    cusp::array1d<ValueType, cusp::host_memory> sn(R);
    cusp::array1d<ValueType, cusp::host_memory> resid(1);

    do
    {
        // compute initial residual and its norm
        cusp::multiply(exec, A, x, w);                // V(0) = A*x
        blas::axpy(exec, b, w, ValueType(-1));        // V(0) = V(0) - b
        cusp::multiply(exec, M, w, w);                // V(0) = M*V(0)
        beta = blas::nrm2(exec, w);                   // beta = norm(V(0))
        blas::scal(exec, w, ValueType(-1.0 / beta));  // V(0) = -V(0)/beta
        blas::copy(exec, w, V.column(0));

        // s = 0 //
        blas::fill(host_exec, s, ValueType(0.0));
        s[0] = beta;
        i = -1;
        resid[0] = cusp::abs(s[0]);
        if (monitor.finished(resid))
        {
            break;
        }

        do
        {
            ++i;
            ++monitor;

            // apply preconditioner
            // can't pass in ref to column in V so need to use copy (w)
            cusp::multiply(exec, A, w, V0);
            // V(i+1) = A*w = M*A*V(i)
            cusp::multiply(exec, M, V0, w);

            for (k = 0; k <= i; k++)
            {
                //  H(k,i) = <V(i+1),V(k)>
                H(k, i) = blas::dotc(exec, V.column(k), w);
                // V(i+1) -= H(k, i) * V(k)
                blas::axpy(exec, V.column(k), w, -H(k, i));
            }

            H(i + 1, i) = blas::nrm2(exec, w);
            // V(i+1) = V(i+1) / H(i+1, i)
            blas::scal(exec, w, ValueType(1.0) / H(i + 1, i));
            blas::copy(exec, w, V.column(i + 1));

            PlaneRotation(H, cs, sn, s, i);

            resid[0] = cusp::abs(s[i + 1]);

            // check convergence condition
            if (monitor.finished(resid))
            {
                break;
            }
        }
        while (i + 1 < R && monitor.iteration_count() + 1 <= monitor.iteration_limit());

        // solve upper triangular system in place
        for (j = i; j >= 0; j--) {
            s[j] /= H(j, j);
            // S(0:j) = s(0:j) - s[j] H(0:j,j)
            for (k = j - 1; k >= 0; k--) {
                s[k] -= H(k, j) * s[j];
            }
        }

        // update the solution

        // copy s to gpu
        blas::copy(s, sDev);
        // x= V(1:N,0:i)*s(0:i)+x
        for (j = 0; j <= i; j++) {
            // x = x + s[j] * V(j)
            blas::axpy(exec, V.column(j), x, s[j]);
        }
    } while (!monitor.finished(resid));
}

}  // end gmres_detail namespace

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void gmres(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const LinearOperator &A,
                 VectorType1 &x,
           const VectorType2 &b,
           const size_t restart,
                 Monitor &monitor,
                 Preconditioner &M)
{
    using cusp::krylov::gmres_detail::gmres;

    return gmres(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, b, restart, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void gmres(const LinearOperator &A,
                 VectorType1 &x,
           const VectorType2 &b,
           const size_t restart,
                 Monitor &monitor,
                 Preconditioner &M)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename VectorType1::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::krylov::gmres(select_system(system1, system2), A, x, b, restart, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor>
void gmres(const LinearOperator &A,
                 VectorType1 &x,
           const VectorType2 &b,
           const size_t restart,
                 Monitor &monitor)
{
    typedef typename LinearOperator::value_type ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType, MemorySpace> M(A.num_rows, A.num_cols);

    return cusp::krylov::gmres(A, x, b, restart, monitor, M);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2>
void gmres(const LinearOperator &A,
                 VectorType1 &x,
           const VectorType2 &b,
           const size_t restart)
{
    typedef typename LinearOperator::value_type ValueType;

    cusp::monitor<ValueType> monitor(b);

    return cusp::krylov::gmres(A, x, b, restart, monitor);
}

}  // end namespace krylov
}  // end namespace cusp

