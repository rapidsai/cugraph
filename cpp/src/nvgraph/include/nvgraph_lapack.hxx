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
#include <nvgraph_error.hxx>
namespace nvgraph
{
template <typename T> class Lapack;

template <typename T>
class Lapack
{
private:
    Lapack();
    ~Lapack();
public:
	static void check_lapack_enabled();

	static void gemm(bool transa, bool transb, int m, int n, int k, T alpha, const T * A, int lda, const T * B, int ldb, T beta, T * C, int ldc);

	// special QR for lanczos
	static void sterf(int n, T * d, T * e);
	static void steqr(char compz, int n, T * d, T * e, T * z, int ldz, T * work);

	// QR
	// computes the QR factorization of a general matrix
	static void geqrf (int m, int n, T *a, int lda, T *tau, T *work, int *lwork);
	// Generates the real orthogonal matrix Q of the QR factorization formed by geqrf.
	//static void orgqr( int m, int n, int k, T* a, int lda, const T* tau, T* work, int* lwork );
	// multiply C by implicit Q
	static void ormqr (bool right_side, bool transq, int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc, T *work, int *lwork);
	//static void unmqr (bool right_side, bool transq, int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc, T *work, int *lwork);
    //static void qrf (int n, T *H, T *Q, T *R);

    //static void hseqr (T* Q, T* R, T* eigenvalues,T* eigenvectors, int dim, int ldh, int ldq);
	static void geev(T* A, T* eigenvalues, int dim, int lda);
	static void geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr);
	static void geev(T* A, T* eigenvalues_r, T* eigenvalues_i, T* eigenvectors_r, T* eigenvectors_i, int dim, int lda, int ldvr);

};
}  // end namespace nvgraph

