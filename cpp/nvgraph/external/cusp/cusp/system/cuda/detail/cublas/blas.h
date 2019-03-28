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

#include <cusp/system/cuda/detail/cublas/defs.h>
#include <cusp/system/cuda/detail/cublas/exception.h>
#include <cusp/system/cuda/detail/cublas/execute_with_cublas.h>
#include <cusp/system/cuda/detail/cublas/stubs.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace cublas
{

template <typename DerivedPolicy,
          typename Array>
int amax(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
         const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    int result;

    cublasStatus_t stat = cublas::amax<0>(handle(exec), n, x_p, 1, result);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("amax", stat);
    }

    return result - 1;
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
asum(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::norm_type<typename Array::value_type>::type Real;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    Real result;

    cublasStatus_t stat = cublas::asum<0>(handle(exec), n, x_p, 1, result);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("asum", stat);
    }

    return result;
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t stat = cublas::axpy<0>(handle(exec), n, ValueType(alpha), x_p, 1, y_p, 1);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("axpy", stat);
    }
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t stat = cublas::copy<0>(handle(exec), n, x_p, 1, y_p, 1);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("copy", stat);
    }
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    ValueType result;

    cublasStatus_t stat = cublas::dot<0>(handle(exec), n, x_p, 1, y_p, 1, result);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("dot", stat);
    }

    return result;
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dotc(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
     const Array1& x,
     const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    ValueType result;

    cublasStatus_t stat = cublas::dotc<0>(handle(exec), n, x_p, 1, y_p, 1, &result);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("dotc", stat);
    }

    return result;
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm2(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    ResultType result;

    cublasStatus_t stat = cublas::nrm2<0>(handle(exec), n, x_p, 1, result);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("nrm2", stat);
    }

    return result;
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void scal(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          Array& x,
          const ScalarType alpha)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    cublasStatus_t stat = cublas::scal<0>(handle(exec), n, alpha, x_p, 1);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("scal", stat);
    }
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void swap(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          Array1& x,
          Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t stat = cublas::swap<0>(handle(exec), n, x_p, 1, y_p, 1);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("swap", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2,
          typename ScalarType1,
          typename ScalarType2>
void gemv(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d1::orientation>::type LayoutType;

    cublasOperation_t trans = LayoutType::order;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;

    ValueType alpha0 = ValueType(alpha);
    ValueType beta0  = ValueType(beta);

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType *y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t stat =
        cublas::gemv<0>(handle(exec), trans, m, n, alpha0,
                        A_p, lda, x_p, 1, beta0, y_p, 1);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("gemv", stat);
    }
}

template <typename DerivedPolicy,
          typename Array1d1,
          typename Array1d2,
          typename Array2d1,
          typename ScalarType>
void ger(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A,
         const ScalarType alpha)
{
    typedef typename Array2d1::value_type ValueType;

    int m   = A.num_rows;
    int n   = A.num_cols;
    int lda = A.pitch;

    ValueType alpha0 = ValueType(alpha);

    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType *y_p = thrust::raw_pointer_cast(&y[0]);
          ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));

    cublasStatus_t stat =
        cublas::ger<0>(handle(exec), m, n, alpha0,
                       x_p, 1, y_p, 1, A_p, lda);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("ger", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2,
          typename ScalarType1,
          typename ScalarType2>
void symv(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    int n   = A.num_rows;
    int lda = A.pitch;

    ValueType alpha0 = ValueType(alpha);
    ValueType beta0  = ValueType(beta);

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType *y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t stat =
        cublas::symv<0>(handle(exec), uplo, n, alpha0,
                        A_p, lda, x_p, 1, beta0, y_p, 1);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("symv", stat);
    }
}

template <typename DerivedPolicy,
          typename Array1d,
          typename Array2d,
          typename ScalarType>
void syr(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
         const Array1d& x,
               Array2d& A,
         const ScalarType alpha)
{
    typedef typename Array2d::value_type ValueType;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    int n   = A.num_cols;
    int lda = A.pitch;

    ValueType alpha0 = ValueType(alpha);

    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
          ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));

    cublasStatus_t stat =
        cublas::syr<0>(handle(exec), uplo, n, alpha0,
                       x_p, 1, A_p, lda);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("syr", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trmv(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d::orientation>::type LayoutType;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;
    cublasOperation_t trans = LayoutType::order;

    int n   = A.num_rows;
    int lda = A.pitch;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType *x_p = thrust::raw_pointer_cast(&x[0]);

    cublasStatus_t stat =
        cublas::trmv<0>(handle(exec), uplo, trans, diag, n,
                        A_p, lda, x_p, 1);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("trmv", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trsv(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d::orientation>::type LayoutType;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;
    cublasOperation_t trans = LayoutType::order;

    int n   = A.num_rows;
    int lda = A.pitch;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType *x_p = thrust::raw_pointer_cast(&x[0]);

    cublasStatus_t stat =
        cublas::trsv<0>(handle(exec), uplo, trans, diag, n,
                        A_p, lda, x_p, 1);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("trsv", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3,
          typename ScalarType1,
          typename ScalarType2>
void gemm(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d1::orientation>::type LayoutType1;
    typedef typename cublas::Orientation<typename Array2d2::orientation>::type LayoutType2;
    typedef typename cublas::Orientation<typename Array2d3::orientation>::type LayoutType3;

    assert(A.num_cols == B.num_rows);
    assert(C.num_rows == A.num_rows);
    assert(C.num_cols == B.num_cols);

    cublasOperation_t transa = LayoutType1::order;
    cublasOperation_t transb = LayoutType2::order;
    cublasOperation_t transc = LayoutType3::order;

    int m = A.num_rows;
    int n = B.num_cols;
    int k = B.num_rows;

    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType * A_p = (ValueType *) thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = (ValueType *) thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p =               thrust::raw_pointer_cast(&C(0,0));

    if(transc == CUBLAS_OP_T)
    {
        bool a_trans = Array2d1::orientation::transpose::value;
        bool b_trans = Array2d2::orientation::transpose::value;

        if(!b_trans)
            transa = (transa == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;
        if(!a_trans)
            transb = (transb == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;

        std::swap(lda, ldb);
        std::swap(A_p, B_p);
    }

    ValueType alpha0 = ValueType(alpha);
    ValueType beta0  = ValueType(beta);

    cublasStatus_t stat =
        cublas::gemm<0>(handle(exec), transa, transb,
                        m, n, k, alpha0, A_p, lda,
                        B_p, ldb, beta0, C_p, ldc);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("gemm", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3,
          typename ScalarType1,
          typename ScalarType2>
void symm(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;

    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    int m   = A.num_rows;
    int n   = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha0 = ValueType(alpha);
    ValueType beta0  = ValueType(beta);

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
          ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t stat =
        cublas::symm<0>(handle(exec), side, uplo,
                        m, n, alpha0, A_p, lda,
                        B_p, ldb, beta0, C_p, ldc);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("symm", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename ScalarType1,
          typename ScalarType2>
void syrk(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d1::orientation>::type LayoutType1;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = LayoutType1::order;

    int n   = A.num_rows;
    int k   = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha0 = ValueType(alpha);
    ValueType beta0  = ValueType(beta);

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cublasStatus_t stat =
        cublas::syrk<0>(handle(exec), uplo, trans,
                        n, k, alpha0, A_p, lda,
                        beta0, B_p, ldb);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("syrk", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3,
          typename ScalarType1,
          typename ScalarType2>
void syr2k(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C,
           const ScalarType1 alpha,
           const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d1::orientation>::type LayoutType1;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = LayoutType1::order;

    int n   = A.num_rows;
    int k   = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha0 = ValueType(alpha);
    ValueType beta0  = ValueType(beta);

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
          ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t stat =
        cublas::syr2k<0>(handle(exec), uplo, trans,
                         n, k, alpha0, A_p, lda,
                         B_p, ldb, beta0, C_p, ldc);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("syr2k", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename ScalarType>
void trmm(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType alpha)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d1::orientation>::type LayoutType1;

    cublasSideMode_t  side  = CUBLAS_SIDE_LEFT;
    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;
    cublasOperation_t trans = LayoutType1::order;

    int m   = B.num_rows;
    int n   = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = B.pitch;

    ValueType alpha0 = ValueType(alpha);

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
          ValueType * C_p = thrust::raw_pointer_cast(&B(0,0));

    cublasStatus_t stat =
        cublas::trmm<0>(handle(exec), side, uplo, trans, diag,
                        m, n, alpha0, A_p, lda,
                        B_p, ldb, C_p, ldc);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("trmm", stat);
    }
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename ScalarType>
void trsm(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType alpha)
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename cublas::Orientation<typename Array2d1::orientation>::type LayoutType1;

    cublasSideMode_t  side  = CUBLAS_SIDE_LEFT;
    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;
    cublasOperation_t trans = LayoutType1::order;

    int n   = B.num_rows;
    int k   = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha0 = ValueType(alpha);

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
          ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cublasStatus_t stat =
        cublas::trsm<0>(handle(exec), side, uplo, trans, diag,
                        n, k, alpha0, A_p, lda, B_p, ldb);

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("trsm", stat);
    }
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm1(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    ResultType result;

    cublasStatus_t stat = cublas::asum(handle(exec), n, x_p, 1, result);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("nrm1", stat);
    }

    return result;
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrmmax(cublas::execute_with_cublas_base<DerivedPolicy>& exec,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::norm_type<ValueType>::type ResultType;

    int index;
    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    cublasStatus_t stat = cublas::amax(handle(exec), n, x_p, 1, index);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_exception("nrmmax", stat);
    }

    return cusp::abs(x[index - 1]);
}

} // end namespace cublas
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

