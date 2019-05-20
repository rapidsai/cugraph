/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include "include/nvgraph_cublas.hxx"

namespace nvgraph
{

cublasHandle_t Cublas::m_handle = 0;

namespace
{
    cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                               const float* alpha,
                               const float* x, int incx,
                               float* y, int incy)
    {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }

    cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                               const double* alpha,
                               const double* x, int incx,
                               double* y, int incy)
    {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }

    cublasStatus_t cublas_copy(cublasHandle_t handle, int n,
                               const float* x, int incx,
                               float* y, int incy)
    {
        return cublasScopy(handle, n, x, incx, y, incy);
    }

    cublasStatus_t cublas_copy(cublasHandle_t handle, int n,
                               const double* x, int incx,
                               double* y, int incy)
    {
        return cublasDcopy(handle, n, x, incx, y, incy);
    }

    cublasStatus_t cublas_dot(cublasHandle_t handle, int n,
                              const float* x, int incx, const float* y, int incy,
                              float* result)
    {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }

    cublasStatus_t cublas_dot(cublasHandle_t handle, int n,
                              const double* x, int incx, const double* y, int incy,
                              double* result)
    {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }
    
    
     cublasStatus_t cublas_trsv_v2(cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx)
    {
	return cublasStrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
    }
    cublasStatus_t cublas_trsv_v2(cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx)
    {
	return cublasDtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
    }
    
    cublasStatus_t cublas_gemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *B, int ldb,
                               const float           *beta,
                               float           *C, int ldc)
    {
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    cublasStatus_t cublas_gemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *B, int ldb,
                               const double          *beta,
                               double          *C, int ldc)
    {
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    cublasStatus_t cublas_gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                               const float *alpha, const float *A, int lda,
                               const float *x, int incx,
                               const float *beta, float* y, int incy)
    {
        return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    cublasStatus_t cublas_gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                               const double *alpha, const double *A, int lda,
                               const double *x, int incx,
                               const double *beta, double* y, int incy)
    {
        return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    cublasStatus_t cublas_ger(cublasHandle_t handle, int m, int n,
                              const float* alpha,
                              const float* x, int incx,
                              const float* y, int incy,
                              float* A, int lda)
    {
        return cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
    }

    cublasStatus_t cublas_ger(cublasHandle_t handle, int m, int n,
                              const double* alpha,
                              const double* x, int incx,
                              const double* y, int incy,
                              double *A, int lda)
    {
        return cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
    }

    cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n,
                               const float *x, int incx, float *result)
    {
        return cublasSnrm2(handle, n, x, incx, result);
    }

    cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n,
                               const double *x, int incx, double *result)
    {
        return cublasDnrm2(handle, n, x, incx, result);
    }

    cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                               const float* alpha,
                               float* x, int incx)
    {
        return cublasSscal(handle, n, alpha, x, incx);
    }

    cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                               const double* alpha,
                               double* x, int incx)
    {
        return cublasDscal(handle, n, alpha, x, incx);
    }

    cublasStatus_t cublas_geam(cublasHandle_t handle,
			       cublasOperation_t transa,
			       cublasOperation_t transb,
			       int m, int n,
			       const float * alpha,
			       const float * A, int lda,
			       const float * beta,
			       const float * B, int ldb,
			       float * C, int ldc) 
    {
        return cublasSgeam(handle, transa, transb, m, n,
			   alpha, A, lda, beta, B, ldb, C, ldc);
    }

    cublasStatus_t cublas_geam(cublasHandle_t handle,
			       cublasOperation_t transa,
			       cublasOperation_t transb,
			       int m, int n,
			       const double * alpha,
			       const double * A, int lda,
			       const double * beta,
			       const double * B, int ldb,
			       double * C, int ldc) 
    {
        return cublasDgeam(handle, transa, transb, m, n,
			   alpha, A, lda, beta, B, ldb, C, ldc);
    }
			     

} // anonymous namespace.

void Cublas::set_pointer_mode_device()
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

void Cublas::set_pointer_mode_host()
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
}

template <typename T>
void Cublas::axpy(int n, T alpha,
                  const T* x, int incx,
                  T* y, int incy)
{
    cublasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_axpy(handle, n, &alpha, x, incx, y, incy));
}

template <typename T>
void Cublas::copy(int n, const T* x, int incx,
                  T* y, int incy)
{
    cublasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_copy(handle, n, x, incx, y, incy));
}

template <typename T>
void Cublas::dot(int n, const T* x, int incx,
                 const T* y, int incy,
                 T* result)
{
    cublasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_dot(handle, n, x, incx, y, incy, result));
}

template <typename T>
T Cublas::nrm2(int n, const T* x, int incx)
{
    Cublas::get_handle();
    T result;
    Cublas::nrm2(n, x, incx, &result);
    return result;
}

template <typename T>
void Cublas::nrm2(int n, const T* x, int incx, T* result)
{
    cublasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_nrm2(handle, n, x, incx, result));
}

template <typename T>
void Cublas::scal(int n, T alpha, T* x, int incx)
{
    Cublas::scal(n, &alpha, x, incx);
}

template <typename T>
void Cublas::scal(int n, T* alpha, T* x, int incx)
{
    cublasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_scal(handle, n, alpha, x, incx));
}

template <typename T>
void Cublas::gemv(bool transposed, int m, int n,
                  const T* alpha, const T* A, int lda,
                  const T* x, int incx,
                  const T* beta, T* y, int incy)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasOperation_t trans = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    CHECK_CUBLAS(cublas_gemv(handle, trans, m, n, alpha, A, lda,
                                 x, incx, beta, y, incy));
}

template <typename T>
void Cublas::gemv_ext(bool transposed, const int m, const int n,
                  const T* alpha, const T* A, const int lda,
                  const T* x, const int incx,
                  const T* beta, T* y, const int incy, const int offsetx, const int offsety, const int offseta)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasOperation_t trans = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    CHECK_CUBLAS(cublas_gemv(handle, trans, m, n, alpha, A+offseta, lda,
                                 x+offsetx, incx, beta, y+offsety, incy));
}

template <typename T>
void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, 
			      const T *A, int lda, T *x, int incx, int offseta)
{
    cublasHandle_t handle = Cublas::get_handle();

    CHECK_CUBLAS( cublas_trsv_v2(handle, uplo, trans, diag, n, A+offseta, lda, x, incx));
}
      
      
template <typename T>
void Cublas::ger(int m, int n, const T* alpha,
                 const T* x, int incx,
                 const T* y, int incy,
                 T* A, int lda)
{
    cublasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}


template <typename T>
void Cublas::gemm(bool transa,
		  bool transb,
		  int m, int n, int k,
		  const T * alpha,
		  const T * A, int lda,
		  const T * B, int ldb,
		  const T * beta,
		  T * C, int ldc)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasOperation_t cublasTransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublasTransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    CHECK_CUBLAS(cublas_gemm(handle, cublasTransA, cublasTransB, m, n, k,
			     alpha, A, lda, B, ldb, beta, C, ldc));
}


template <typename T>
void Cublas::geam(bool transa, bool transb, int m, int n,
		  const T * alpha, const T * A, int lda,
		  const T * beta,  const T * B, int ldb,
		  T * C, int ldc)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasOperation_t cublasTransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublasTransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    CHECK_CUBLAS(cublas_geam(handle, cublasTransA, cublasTransB, m, n,
			     alpha, A, lda, beta, B, ldb, C, ldc));
}

template void Cublas::axpy(int n, float alpha,
                           const float* x, int incx,
                           float* y, int incy);
template void Cublas::axpy(int n, double alpha,
                           const double* x, int incx,
                           double* y, int incy);

template void Cublas::copy(int n, const float* x, int incx, float* y, int incy);
template void Cublas::copy(int n, const double* x, int incx, double* y, int incy);

template void Cublas::dot(int n, const float* x, int incx,
                          const float* y, int incy,
                          float* result);
template void Cublas::dot(int n, const double* x, int incx,
                          const double* y, int incy,
                          double* result);

template void Cublas::gemv(bool transposed, int m, int n,
                           const float* alpha, const float* A, int lda,
                           const float* x, int incx,
                           const float* beta, float* y, int incy);
template void Cublas::gemv(bool transposed, int m, int n,
                           const double* alpha, const double* A, int lda,
                           const double* x, int incx,
                           const double* beta, double* y, int incy);

template void Cublas::ger(int m, int n, const float* alpha,
                          const float* x, int incx,
                          const float* y, int incy,
                          float* A, int lda);
template void Cublas::ger(int m, int n, const double* alpha,
                          const double* x, int incx,
                          const double* y, int incy,
                          double* A, int lda);


template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                           const float* alpha, const float* A, const int lda,
                           const float* x, const int incx,
                           const float* beta, float* y, const int incy, const int offsetx, const int offsety, const int offseta);
template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                           const double* alpha, const double* A, const int lda,
                           const double* x, const int incx,
                           const double* beta, double* y, const int incy, const int offsetx, const int offsety, const int offseta);


template void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, 
			      const float *A, int lda, float *x, int incx, int offseta);
template void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, 
			      const double *A, int lda, double *x, int incx, int offseta);

template double Cublas::nrm2(int n, const double* x, int incx);
template float Cublas::nrm2(int n, const float* x, int incx);

template void Cublas::scal(int n, float alpha, float* x, int incx);
template void Cublas::scal(int n, double alpha, double* x, int incx);

template void Cublas::gemm(bool transa, bool transb,
			   int m, int n, int k,
			   const float * alpha,
			   const float * A, int lda,
			   const float * B, int ldb,
			   const float * beta,
			   float * C, int ldc);
template void Cublas::gemm(bool transa, bool transb,
			   int m, int n, int k,
			   const double * alpha,
			   const double * A, int lda,
			   const double * B, int ldb,
			   const double * beta,
			   double * C, int ldc);

template void Cublas::geam(bool transa, bool transb, int m, int n,
			   const float * alpha, const float * A, int lda,
			   const float * beta,  const float * B, int ldb,
			   float * C, int ldc);
template void Cublas::geam(bool transa, bool transb, int m, int n,
			   const double * alpha, const double * A, int lda,
			   const double * beta,  const double * B, int ldb,
			   double * C, int ldc);


} // end namespace nvgraph

