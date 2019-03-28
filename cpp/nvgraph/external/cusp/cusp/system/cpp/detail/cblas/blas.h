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

#include <cusp/array1d.h>
#include <cusp/exception.h>

#include <cusp/system/cpp/detail/cblas/complex_stubs.h>
#include <cusp/system/cpp/detail/cblas/stubs.h>
#include <cusp/system/cpp/detail/cblas/defs.h>
#include <cusp/system/cpp/detail/cblas/execution_policy.h>

namespace cusp
{
namespace system
{
namespace cpp
{
namespace detail
{
namespace cblas
{

template <typename DerivedPolicy,
          typename Array>
int amax(cblas::execution_policy<DerivedPolicy>& exec,
         const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::amax<0>(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
asum(cblas::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::asum<0>(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(cblas::execution_policy<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    typedef typename Array1::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::axpy<0>(n, alpha, x_p, 1, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(cblas::execution_policy<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::copy<0>(n, x_p, 1, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(cblas::execution_policy<DerivedPolicy>& exec,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    return cblas::dot<0>(n, x_p, 1, y_p, 1);
}

// template <typename Array1,
//           typename Array2>
// typename Array1::value_type
// dotc(cblas::execution_policy& exec,
//      const Array1& x,
//      const Array2& y)
// {
//     typedef typename Array2::value_type ValueType;
//
//     int n = y.size();
//
//     const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
//     const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);
//
//     return cblas::detail::dotc(n, x_p, 1, y_p, 1);
// }

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm2(cblas::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::nrm2<0>(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void scal(cblas::execution_policy<DerivedPolicy>& exec,
          Array& x,
          const ScalarType alpha)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    cblas::scal<0>(n, alpha, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void swap(cblas::execution_policy<DerivedPolicy>& exec,
          Array1& x,
          Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::swap<0>(n, x_p, 1, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2,
          typename ScalarType1,
          typename ScalarType2>
void gemv(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER     order = LayoutType::order;
    CBLAS_TRANSPOSE trans = CblasNoTrans;

    int m   = A.num_rows;
    int n   = A.num_cols;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType * y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::gemv<0>(order, trans, m, n, alpha,
                   A_p, lda, x_p, 1, beta, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array1d1,
          typename Array1d2,
          typename Array2d1,
          typename ScalarType>
void ger(cblas::execution_policy<DerivedPolicy>& exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A,
         const ScalarType alpha)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER order = LayoutType::order;

    int m   = A.num_rows;
    int n   = A.num_cols;
    int lda = A.pitch;

    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType * y_p = thrust::raw_pointer_cast(&y[0]);
          ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));

    cblas::ger<0>(order, m, n, alpha,
                  x_p, 1, y_p, 1, A_p, lda);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2,
          typename ScalarType1,
          typename ScalarType2>
void symv(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER order = LayoutType::order;
    CBLAS_UPLO  uplo  = CblasUpper;

    int n   = A.num_rows;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType * y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::symv<0>(order, uplo, n, alpha,
                   A_p, lda, x_p, 1, beta, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array1d,
          typename Array2d,
          typename ScalarType>
void syr(cblas::execution_policy<DerivedPolicy>& exec,
         const Array1d& x,
               Array2d& A,
         const ScalarType alpha)
{
    typedef typename Array2d::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d::orientation>::type LayoutType;

    CBLAS_ORDER order = LayoutType::order;
    CBLAS_UPLO  uplo  = CblasUpper;

    int n   = A.num_rows;
    int lda = A.pitch;

    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));

    cblas::syr<0>(order, uplo, n, alpha,
                  x_p, 1, A_p, lda);
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trmv(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d::orientation>::type LayoutType;

    CBLAS_ORDER     order = LayoutType::order;
    CBLAS_UPLO      uplo  = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG      diag  = CblasNonUnit;

    int n   = A.num_rows;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType * x_p = thrust::raw_pointer_cast(&x[0]);

    cblas::trmv<0>(order, uplo, trans, diag, n,
                   A_p, lda, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trsv(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d::orientation>::type LayoutType;

    CBLAS_ORDER     order = LayoutType::order;
    CBLAS_UPLO      uplo  = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG      diag  = CblasNonUnit;

    int n   = A.num_rows;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType * x_p = thrust::raw_pointer_cast(&x[0]);

    cblas::trsv<0>(order, uplo, trans, diag, n,
                   A_p, lda, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3,
          typename ScalarType1,
          typename ScalarType2>
void gemm(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d3::orientation>::type LayoutType3;

    CBLAS_ORDER order  = LayoutType3::order;

    bool a_trans = Array2d1::orientation::transpose::value;
    bool b_trans = Array2d2::orientation::transpose::value;

    CBLAS_TRANSPOSE transa = a_trans ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transb = b_trans ? CblasTrans : CblasNoTrans;

    int m = A.num_rows;
    int n = B.num_cols;
    int k = B.num_rows;

    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha0 = ValueType(alpha);
    ValueType beta0  = ValueType(beta);

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
          ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cblas::gemm<0>(order, transa, transb,
                   m, n, k, alpha0, A_p, lda,
                   B_p, ldb, beta0, C_p, ldc);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3,
          typename ScalarType1,
          typename ScalarType2>
void symm(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER order = LayoutType::order;
    CBLAS_SIDE  side  = CblasLeft;
    CBLAS_UPLO  uplo  = CblasUpper;

    int m   = C.num_rows;
    int n   = C.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
          ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cblas::symm<0>(order, side, uplo,
                   m, n, alpha, A_p, lda,
                   B_p, ldb, beta, C_p, ldc);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename ScalarType1,
          typename ScalarType2>
void syrk(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER     order = LayoutType::order;
    CBLAS_UPLO      uplo  = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;

    int n   = A.num_rows;
    int k   = B.num_rows;
    int lda = A.pitch;
    int ldb = B.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cblas::syrk<0>(order, uplo, trans,
                   n, k, alpha, A_p, lda,
                   beta, B_p, ldb);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3,
          typename ScalarType1,
          typename ScalarType2>
void syr2k(cblas::execution_policy<DerivedPolicy>& exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER     order = LayoutType::order;
    CBLAS_UPLO      uplo  = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;

    int n   = A.num_rows;
    int k   = B.num_rows;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
          ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cblas::syr2k<0>(order, uplo, trans,
                    n, k, alpha, A_p, lda,
                    B_p, ldb, beta, C_p, ldc);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename ScalarType>
void trmm(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType alpha)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER     order = LayoutType::order;
    CBLAS_SIDE      side  = CblasLeft;
    CBLAS_UPLO      uplo  = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG      diag  = CblasNonUnit;

    int m   = B.num_rows;
    int n   = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cblas::trmm<0>(order, side, uplo, trans, diag,
                   m, n, alpha, A_p, lda, B_p, ldb);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename ScalarType>
void trsm(cblas::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType alpha)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cblas::Orientation<typename Array2d1::orientation>::type LayoutType;

    CBLAS_ORDER     order = LayoutType::order;
    CBLAS_SIDE      side  = CblasLeft;
    CBLAS_UPLO      uplo  = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    CBLAS_DIAG      diag  = CblasNonUnit;

    int m   = B.num_rows;
    int n   = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cblas::trsm<0>(order, side, uplo, trans, diag,
                   m, n, alpha, A_p, lda, B_p, ldb);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm1(cblas::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::asum(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrmmax(cblas::execution_policy<DerivedPolicy>& exec,
       const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    int index = cblas::amax(n, x_p, 1);

    return cusp::abs(x[index]);
}

} // end namespace cblas
} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace cusp

