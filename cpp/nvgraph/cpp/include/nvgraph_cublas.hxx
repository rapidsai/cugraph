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
 
#pragma once

#include <cublas_v2.h>
#include <iostream>
#include "debug_macros.h"

namespace nvgraph
{
class Cublas;

class Cublas
{
private:
    static cublasHandle_t m_handle;
    // Private ctor to prevent instantiation.
    Cublas();
    ~Cublas();
public:

    // Get the handle.
    static cublasHandle_t get_handle()
    {
        if (m_handle == 0)
            CHECK_CUBLAS(cublasCreate(&m_handle));
        return m_handle;
    }

    static void destroy_handle()
    {
        if (m_handle != 0)
            CHECK_CUBLAS(cublasDestroy(m_handle));
        m_handle = 0;
    }

    static void set_pointer_mode_device();
    static void set_pointer_mode_host();
    static void setStream(cudaStream_t stream) 
    {   
        cublasHandle_t handle = Cublas::get_handle();
        CHECK_CUBLAS(cublasSetStream(handle, stream));
    }

    template <typename T>
    static void axpy(int n, T alpha,
                     const T* x, int incx,
                     T* y, int incy);

    template <typename T>
    static void copy(int n, const T* x, int incx,
                     T* y, int incy);

    template <typename T>
    static void dot(int n, const T* x, int incx,
                    const T* y, int incy,
                    T* result);

    template <typename T>
    static void gemv(bool transposed, int m, int n,
                     const T* alpha, const T* A, int lda,
                     const T* x, int incx,
                     const T* beta, T* y, int incy);

    template <typename T>
    static void gemv_ext(bool transposed, const int m, const int n,
                     const T* alpha, const T* A, const int lda,
                     const T* x, const int incx,
                     const T* beta, T* y, const int incy, const int offsetx, const int offsety, const int offseta);
    
    template <typename T>
    static void trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, 
			      const T *A, int lda, T *x, int incx, int offseta);

    template <typename T>
    static void ger(int m, int n, const T* alpha,
                    const T* x, int incx,
                    const T* y, int incy,
                    T* A, int lda);

    template <typename T>
    static T nrm2(int n, const T* x, int incx);
    template <typename T>
    static void nrm2(int n, const T* x, int incx, T* result);

    template <typename T>
    static void scal(int n, T alpha, T* x, int incx);
    template <typename T>
    static void scal(int n, T* alpha, T* x, int incx);

    template <typename T>
    static void gemm(bool transa, bool transb, int m, int n, int k,
		     const T * alpha, const T * A, int lda,
		     const T * B, int ldb,
		     const T * beta, T * C, int ldc);

    template <typename T>
    static void geam(bool transa, bool transb, int m, int n,
		     const T * alpha, const T * A, int lda,
		     const T * beta,  const T * B, int ldb,
		     T * C, int ldc);

};

} // end namespace nvgraph

