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
 

#include "include/nvgraph_lapack.hxx"

//#include <f2c.h>
//#include <complex>

//#define NVGRAPH_USE_LAPACK 1

namespace nvgraph
{

#define lapackCheckError(status)                         \
    {                                                    \
        if (status < 0)                                  \
        {                                                \
            std::stringstream ss;                        \
            ss << "Lapack error: argument number "       \
               << -status << " had an illegal value.";   \
            FatalError(ss.str(), NVGRAPH_ERR_UNKNOWN);      \
        }                                                \
        else if (status > 0)                             \
            FatalError("Lapack error: internal error.",  \
                       NVGRAPH_ERR_UNKNOWN);                \
    }                                                    \

template <typename T>
void Lapack<T>::check_lapack_enabled()
{
#ifndef NVGRAPH_USE_LAPACK
    FatalError("Error: LAPACK not enabled.", NVGRAPH_ERR_UNKNOWN);
#endif
}


typedef enum{
    CUSOLVER_STATUS_SUCCESS=0,
    CUSOLVER_STATUS_NOT_INITIALIZED=1,
    CUSOLVER_STATUS_ALLOC_FAILED=2,
    CUSOLVER_STATUS_INVALID_VALUE=3,
    CUSOLVER_STATUS_ARCH_MISMATCH=4,
    CUSOLVER_STATUS_MAPPING_ERROR=5,
    CUSOLVER_STATUS_EXECUTION_FAILED=6,
    CUSOLVER_STATUS_INTERNAL_ERROR=7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT=10,
    CUSOLVER_STATUS_INVALID_LICENSE=11
} cusolverStatus_t;

typedef enum {
    CUBLAS_OP_N=0,
    CUBLAS_OP_T=1,
    CUBLAS_OP_C=2
} cublasOperation_t;

namespace {
// XGEMM
//extern "C"
//void sgemm_(const char *transa, const char *transb,
//        const int *m, const int *n, const int *k,
//        const float *alpha, const float *a, const int *lda,
//        const float *b, const int *ldb,
//        const float *beta, float *c, const int *ldc);
//extern "C"
//void dgemm_(const char *transa, const char *transb,
//        const int *m, const int *n, const int *k,
//        const double *alpha, const double *a, const int *lda,
//        const double *b, const int *ldb,
//        const double *beta, double *c, const int *ldc);



extern "C" cusolverStatus_t cusolverDnSgemmHost(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc);


void lapack_gemm(const char transa, const char transb, int m, int n, int k,
         float alpha, const float *a, int lda,
         const float *b, int ldb,
         float beta, float *c, int ldc)
{
    cublasOperation_t cublas_transa = (transa == 'N')? CUBLAS_OP_N : CUBLAS_OP_T ;
    cublasOperation_t cublas_transb = (transb == 'N')? CUBLAS_OP_N : CUBLAS_OP_T ;
    cusolverDnSgemmHost(cublas_transa, cublas_transb, m, n, k,
       &alpha, (float*)a, lda, (float*)b, ldb, &beta, c, ldc);
}

extern "C" cusolverStatus_t cusolverDnDgemmHost(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const double *alpha,
    const double *A,
    int lda,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc);

void lapack_gemm(const signed char transa, const signed char transb, int m, int n, int k,
         double alpha, const double *a, int lda,
         const double *b, int ldb,
         double beta, double *c, int ldc)
{
    cublasOperation_t cublas_transa = (transa == 'N')? CUBLAS_OP_N : CUBLAS_OP_T ;
    cublasOperation_t cublas_transb = (transb == 'N')? CUBLAS_OP_N : CUBLAS_OP_T ;
    cusolverDnDgemmHost(cublas_transa, cublas_transb, m, n, k,
       &alpha, (double*)a, lda, (double*)b, ldb, &beta, c, ldc);
}

// XSTERF
//extern "C"
//void ssterf_(const int *n, float *d, float *e, int *info);
//
//extern "C"
//void dsterf_(const int *n, double *d, double *e, int *info);
//

extern "C" cusolverStatus_t cusolverDnSsterfHost(
    int n,
    float *d,
    float *e,
    int *info);

void lapack_sterf(int n, float * d, float * e, int * info)
{
    cusolverDnSsterfHost(n, d, e, info);
}

extern "C" cusolverStatus_t cusolverDnDsterfHost(
    int n,
    double *d,
    double *e,
    int *info);

void lapack_sterf(int n, double * d, double * e, int * info)
{
    cusolverDnDsterfHost(n, d, e, info);
}

// XSTEQR
//extern "C"
//void ssteqr_(const char *compz, const int *n, float *d, float *e,
//       float *z, const int *ldz, float *work, int * info);
//extern "C"
//void dsteqr_(const char *compz, const int *n, double *d, double *e,
//       double *z, const int *ldz, double *work, int *info);


extern "C" cusolverStatus_t cusolverDnSsteqrHost(
    const signed char *compz,
    int n,
    float *d,
    float *e,
    float *z,
    int ldz,
    float *work,
    int *info);

void lapack_steqr(const signed char compz, int n, float * d, float * e,
          float * z, int ldz, float * work, int * info)
{
    cusolverDnSsteqrHost(&compz, n, d, e, z, ldz, work, info);
}

extern "C" cusolverStatus_t cusolverDnDsteqrHost(
    const signed char *compz,
    int n,
    double *d,
    double *e,
    double *z,
    int ldz,
    double *work,
    int *info);

void lapack_steqr(const signed char compz, int n, double * d, double * e,
          double * z, int ldz, double * work, int * info)
{
    cusolverDnDsteqrHost(&compz, n, d, e, z, ldz, work, info);
}

#ifdef NVGRAPH_USE_LAPACK


extern "C"
void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
extern "C"
void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
//extern "C"
//void cgeqrf_(int *m, int *n, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
//extern "C"
//void zgeqrf_(int *m, int *n, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);

void lapack_geqrf(int m, int n, float *a, int lda, float *tau, float *work, int *lwork, int *info)
{
    sgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
}
void lapack_geqrf(int m, int n, double *a, int lda, double *tau, double *work, int *lwork, int *info)
{
    dgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
}
//void lapack_geqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info)
//{
//    cgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
//}
//void lapack_geqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info)
//{
//    zgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
//}

extern "C"
void sormqr_ (char* side, char* trans, int *m, int *n, int *k, float *a, int *lda, const float *tau, float* c, int *ldc, float *work, int *lwork, int *info);
extern "C"
void dormqr_(char* side, char* trans, int *m, int *n, int *k, double *a, int *lda, const double *tau,  double* c, int *ldc, double *work, int *lwork, int *info);
//extern "C"
//void cunmqr_ (char* side, char* trans, int *m, int *n, int *k, std::complex<float> *a, int *lda, const std::complex<float> *tau, std::complex<float>* c, int *ldc, std::complex<float> *work, int *lwork, int *info);
//extern "C"
//void zunmqr_(char* side, char* trans, int *m, int *n, int *k, std::complex<double> *a, int *lda, const std::complex<double> *tau,  std::complex<double>* c, int *ldc, std::complex<double> *work, int *lwork, int *info);

void lapack_ormqr(char side, char trans, int m, int n, int k, float *a, int lda, float *tau, float* c, int ldc, float *work, int *lwork, int *info)
{
    sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
}
void lapack_ormqr(char side, char trans, int m, int n, int k, double *a, int lda, double *tau, double* c, int ldc, double *work, int *lwork, int *info)
{
    dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
}
//void lapack_unmqr(char side, char trans, int m, int n, int k, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float>* c, int ldc, std::complex<float> *work, int *lwork, int *info)
//{
//    cunmqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
//}
//void lapack_unmqr(char side, char trans, int m, int n, int k, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double>* c, int ldc, std::complex<double> *work, int *lwork, int *info)
//{
//    zunmqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
//}

// extern "C"
// void sorgqr_ ( int* m, int* n, int* k, float* a, int* lda, const float* tau, float* work, int* lwork, int *info );
// extern "C"
// void dorgqr_ ( int* m, int* n, int* k, double* a, int* lda, const double* tau, double* work, int* lwork, int *info );
// 
// void lapack_orgqr( int m, int n, int k, float* a, int lda, const float* tau, float* work, int *lwork, int *info) 
// {
//     sorgqr_(&m, &n, &k, a, &lda, tau, work, lwork, info);
// }
// void lapack_orgqr( int m, int n, int k, double* a, int lda, const double* tau, double* work, int* lwork, int *info )
// {
//     dorgqr_(&m, &n, &k, a, &lda, tau, work, lwork, info);
// }

//int lapack_hseqr_dispatch(char *jobvl, char *jobvr, int* n, int*ilo, int*ihi, 
//                          double *h, int* ldh, double *wr, double *wi, double *z, 
//                          int*ldz, double *work, int *lwork, int *info)
//{
//    return dhseqr_(jobvl, jobvr, n, ilo, ihi, h, ldh, wr, wi, z, ldz, work, lwork, info);
//}
//
//int lapack_hseqr_dispatch(char *jobvl, char *jobvr, int* n, int*ilo, int*ihi, 
//                          float *h, int* ldh, float *wr, float *wi, float *z, 
//                          int*ldz, float *work, int *lwork, int *info)
//{
//    return shseqr_(jobvl, jobvr, n, ilo, ihi, h, ldh, wr, wi, z, ldz, work, lwork, info);
//}


// XGEEV
extern "C"
int dgeev_(char *jobvl, char *jobvr, int *n, double *a,
           int *lda, double *wr, double *wi, double *vl,
           int *ldvl, double *vr, int *ldvr, double *work,
           int *lwork, int *info);

extern "C"
int sgeev_(char *jobvl, char *jobvr, int *n, float *a,
           int *lda, float *wr, float *wi, float *vl,
           int *ldvl, float *vr, int *ldvr, float *work,
           int *lwork, int *info);

//extern "C"
//int dhseqr_(char *jobvl, char *jobvr, int* n, int*ilo, int*ihi, 
//            double *h, int* ldh, double *wr, double *wi, double *z, 
//            int*ldz, double *work, int *lwork, int *info);
//extern "C"
//int shseqr_(char *jobvl, char *jobvr, int* n, int*ilo, int*ihi, 
//            float *h, int* ldh, float *wr, float *wi, float *z, 
//            int*ldz, float *work, int *lwork, int *info);
//
int lapack_geev_dispatch(char *jobvl, char *jobvr, int *n, double *a,
                         int *lda, double *wr, double *wi, double *vl,
                         int *ldvl, double *vr, int *ldvr, double *work,
                         int *lwork, int *info)
{
    return dgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

int lapack_geev_dispatch(char *jobvl, char *jobvr, int *n, float *a,
                         int *lda, float *wr, float *wi, float *vl,
                         int *ldvl, float *vr, int *ldvr, float *work,
                         int *lwork, int *info)
{
    return sgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}




// real eigenvalues
template <typename T>
void lapack_geev(T* A, T* eigenvalues, int dim, int lda)
{
    char job = 'N';
    T* WI = new T[dim];
    int ldv = 1;
    T* vl = 0;
    int work_size = 6 * dim;
    T* work = new T[work_size];
    int info;
    lapack_geev_dispatch(&job, &job, &dim, A, &lda, eigenvalues, WI, vl, &ldv,
                         vl, &ldv, work, &work_size, &info);
    lapackCheckError(info);
    delete [] WI;
    delete [] work;
}
//real eigenpairs
template <typename T>
void lapack_geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr)
{
    char jobvl = 'N';
    char jobvr = 'V';
    T* WI = new T[dim];
    int work_size = 6 * dim;
    T* vl = 0;
    int ldvl = 1;
    T* work = new T[work_size];
    int info;
    lapack_geev_dispatch(&jobvl, &jobvr, &dim, A, &lda, eigenvalues, WI, vl, &ldvl,
                         eigenvectors, &ldvr, work, &work_size, &info);
    lapackCheckError(info);
    delete [] WI;
    delete [] work;
}
//complex eigenpairs
template <typename T>
void lapack_geev(T* A, T* eigenvalues_r, T* eigenvalues_i, T* eigenvectors_r, T* eigenvectors_i, int dim, int lda, int ldvr)
{
    char jobvl = 'N';
    char jobvr = 'V';
    int work_size = 8 * dim;
    int ldvl = 1;
    T* work = new T[work_size];
    int info;
    lapack_geev_dispatch(&jobvl, &jobvr, &dim, A, &lda, eigenvalues_r, eigenvalues_i, 0, &ldvl,
                         eigenvectors_r, &ldvr, work, &work_size, &info);
    lapackCheckError(info);
    delete [] work;
}

//template <typename T>
//void lapack_hseqr(T* Q, T* H, T* eigenvalues, int dim, int ldh, int ldq)
//{
//    char job = 'S'; // S compute eigenvalues and the Schur form T. On entry, the upper Hessenberg matrix H. 
//                    // On exit H contains the upper quasi-triangular matrix T from the Schur decomposition
//    char jobvr = 'V'; //Take Q on entry, and the product Q*Z is returned.
//    //ILO and IHI are normally set by a previous call to DGEBAL, Otherwise ILO and IHI should be set to 1 and N
//    int ilo = 1;
//    int ihi = dim;
//    T* WI = new T[dim];
//    int ldv = 1;
//    T* vl = 0;
//    int work_size = 11 * dim; //LWORK as large as 11*N may be required for optimal performance. It is CPU memory and the matrix is assumed to be small
//    T* work = new T[work_size];
//    int info;
//    lapack_hseqr_dispatch(&job, &jobvr, &dim, &ilo, &ihi, H, &ldh, eigenvalues, WI, Q, &ldq, work, &work_size, &info);
//    lapackCheckError(info);
//    delete [] WI;
//    delete [] work;
//}

#endif

} // end anonymous namespace

template <typename T>
void Lapack< T >::gemm(bool transa, bool transb,
		       int m, int n, int k,
		       T alpha, const T * A, int lda,
		       const T * B, int ldb,
		       T beta, T * C, int ldc)
{
//check_lapack_enabled();
//#ifdef NVGRAPH_USE_LAPACK
    const char transA_char = transa ? 'T' : 'N';
    const char transB_char = transb ? 'T' : 'N';
    lapack_gemm(transA_char, transB_char, m, n, k,
		alpha, A, lda, B, ldb, beta, C, ldc);
//#endif
}

template <typename T>
void Lapack< T >::sterf(int n, T * d, T * e)
{
//    check_lapack_enabled();
//#ifdef NVGRAPH_USE_LAPACK
    int info;
    lapack_sterf(n, d, e, &info);
    lapackCheckError(info);
//#endif
}

template <typename T>
void Lapack< T >::steqr(char compz, int n, T * d, T * e,
			T * z, int ldz, T * work)
{
//    check_lapack_enabled();
//#ifdef NVGRAPH_USE_LAPACK
    int info;
    lapack_steqr(compz, n, d, e, z, ldz, work, &info);
    lapackCheckError(info);
//#endif
}

template <typename T>
void Lapack< T >::geqrf(int m, int n, T *a, int lda, T *tau, T *work, int *lwork)
{
    check_lapack_enabled();
    #ifdef NVGRAPH_USE_LAPACK
        int info;
        lapack_geqrf(m, n, a, lda, tau, work, lwork, &info);
        lapackCheckError(info);
    #endif
}
template <typename T>
void Lapack< T >::ormqr(bool right_side, bool transq, int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc, T *work, int *lwork)
{
    check_lapack_enabled();
    #ifdef NVGRAPH_USE_LAPACK
        char side = right_side ? 'R' : 'L';
        char trans = transq ? 'T' : 'N';
        int info;
        lapack_ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &info);
        lapackCheckError(info);
    #endif
}

//template <typename T>
//void Lapack< T >::unmqr(bool right_side, bool transq, int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc, T *work, int *lwork)
//{
//    check_lapack_enabled();
//    #ifdef NVGRAPH_USE_LAPACK
//        char side = right_side ? 'R' : 'L';
//        char trans = transq ? 'T' : 'N';
//        int info;
//        lapack_unmqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &info);
//        lapackCheckError(info);
//    #endif
//}

//template <typename T>
//void Lapack< T >::orgqr( int m, int n, int k, T* a, int lda, const T* tau, T* work, int* lwork)
//{
//    check_lapack_enabled();
//    #ifdef NVGRAPH_USE_LAPACK
//        int info;
//        lapack_orgqr(m, n, k, a, lda, tau, work, lwork, &info);
//        lapackCheckError(info);
//    #endif
//}
//template <typename T>
//void Lapack< T >::qrf(int n, int k, T *H, T *C, T *Q, T *R)
//{
//    check_lapack_enabled();
//    #ifdef NVGRAPH_USE_LAPACK
//    //   int m = n, k = n, lda=n, lwork=2*n, info;
//    //   lapack_geqrf(m, n, H, lda, C, work, lwork, &info);
//    //   lapackCheckError(info);
//    //   lapack_ormqr(m, n, k, H, lda, tau, c, ldc, work, lwork, &info);
//    //   lapackCheckError(info);
//    #endif
//}

//real eigenvalues 
template <typename T>
void Lapack< T >::geev(T* A, T* eigenvalues, int dim, int lda)
{
    check_lapack_enabled();
#ifdef NVGRAPH_USE_LAPACK
    lapack_geev(A, eigenvalues, dim, lda);
#endif
}
//real eigenpairs
template <typename T>
void Lapack< T >::geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr)
{
    check_lapack_enabled();
#ifdef NVGRAPH_USE_LAPACK
    lapack_geev(A, eigenvalues, eigenvectors, dim, lda, ldvr);
#endif
}
//complex eigenpairs
template <typename T>
void Lapack< T >::geev(T* A, T* eigenvalues_r, T* eigenvalues_i, T* eigenvectors_r, T* eigenvectors_i, int dim, int lda, int ldvr)
{
    check_lapack_enabled();
#ifdef NVGRAPH_USE_LAPACK
    lapack_geev(A, eigenvalues_r, eigenvalues_i, eigenvectors_r, eigenvectors_i, dim, lda, ldvr);
#endif
}

//template <typename T>
//void Lapack< T >::hseqr(T* Q, T* H, T* eigenvalues,T* eigenvectors, int dim, int ldh, int ldq)
//{
//    check_lapack_enabled();
//#ifdef NVGRAPH_USE_LAPACK
//    lapack_hseqr(Q, H, eigenvalues, dim, ldh, ldq);
//#endif
//}

// Explicit instantiation
template void Lapack<float>::check_lapack_enabled();
template void Lapack<float>::gemm(bool transa, bool transb,int m, int n, int k,float alpha, const float * A, int lda, const float * B, int ldb, float beta, float * C, int ldc);
template void Lapack<float>::sterf(int n, float * d, float * e);
template void Lapack<float>::geev (float* A, float* eigenvalues, float* eigenvectors, int dim, int lda, int ldvr);
template void Lapack<float>::geev (float* A, float* eigenvalues_r, float* eigenvalues_i, float* eigenvectors_r, float* eigenvectors_i, int dim, int lda, int ldvr);
//template void Lapack<float>::hseqr(float* Q, float* H, float* eigenvalues, float* eigenvectors, int dim, int ldh, int ldq);
template void Lapack<float>::steqr(char compz, int n, float * d, float * e, float * z, int ldz, float * work);
template void Lapack<float>::geqrf(int m, int n, float *a, int lda, float *tau, float *work, int *lwork);
template void Lapack<float>::ormqr(bool right_side, bool transq, int m, int n, int k, float *a, int lda, float *tau, float *c, int ldc, float *work, int *lwork);
//template void Lapack<float>::orgqr(int m, int n, int k, float* a, int lda, const float* tau, float* work, int* lwork);

template void Lapack<double>::check_lapack_enabled();
template void Lapack<double>::gemm(bool transa, bool transb, int m, int n, int k, double alpha, const double * A, int lda, const double * B, int ldb, double beta, double * C, int ldc);
template void Lapack<double>::sterf(int n, double * d, double * e);
template void Lapack<double>::geev (double* A, double* eigenvalues, double* eigenvectors, int dim, int lda, int ldvr);
template void Lapack<double>::geev (double* A, double* eigenvalues_r, double* eigenvalues_i, double* eigenvectors_r, double* eigenvectors_i, int dim, int lda, int ldvr);
//template void Lapack<double>::hseqr(double* Q, double* H, double* eigenvalues, double* eigenvectors, int dim, int ldh, int ldq);
template void Lapack<double>::steqr(char compz, int n, double * d, double * e, double * z, int ldz, double * work);
template void Lapack<double>::geqrf(int m, int n, double *a, int lda, double *tau, double *work, int *lwork);
template void Lapack<double>::ormqr(bool right_side, bool transq, int m, int n, int k, double *a, int lda, double *tau, double *c, int ldc, double *work, int *lwork);
//template void Lapack<double>::orgqr(int m, int n, int k, double* a, int lda, const double* tau, double* work, int* lwork);

//template void Lapack<std::complex<float> >::geqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float> *work, int *lwork);
//template void Lapack<std::complex<double> >::geqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double> *work, int *lwork);
//template void Lapack<std::complex<float> >::unmqr(bool right_side, bool transq, int m, int n, int k, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float> *c, int ldc, std::complex<float> *work, int *lwork);
//template void Lapack<std::complex<double> >::unmqr(bool right_side, bool transq, int m, int n, int k, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double> *c, int ldc, std::complex<double> *work, int *lwork);


}  // end namespace nvgraph

