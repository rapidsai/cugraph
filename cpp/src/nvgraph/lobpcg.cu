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
//#if SPECTRAL_USE_LOBPCG
#include "include/lobpcg.hxx"

#include <stdio.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <curand.h>
//#include "spectral_parameters.h"
//#include "cuda_helper.h"
//#include "cublas_helper.h"
//#include "cusolver_helper.h"
//#include "cusparse_helper.h"
//#include "curand_helper.h"
//#include "magma_helper.h"
//#define COLLECT_TIME_STATISTICS 1
#undef COLLECT_TIME_STATISTICS

#ifdef COLLECT_TIME_STATISTICS
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#endif

static double timer (void) {
#ifdef COLLECT_TIME_STATISTICS
    struct timeval tv;
    cudaDeviceSynchronize();
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
#else
    return 0.0; 
#endif
}

namespace nvgraph {

    template <typename IndexType_, typename ValueType_, bool Device_>
    static int print_matrix(IndexType_ m, IndexType_ n, ValueType_ * A, IndexType_ lda, const char *s){
        IndexType_ i,j;
        ValueType_ * h_A;

        if (m > lda) {
            WARNING("print_matrix - invalid parameter (m > lda)");
            return -1;
        }
        if (Device_) {
            h_A = (ValueType_ *)malloc(lda*n*sizeof(ValueType_));
            if (!h_A) {
                WARNING("print_matrix - malloc failed");
                return -1;
            }
            cudaMemcpy(h_A, A, lda*n*sizeof(ValueType_), cudaMemcpyDeviceToHost); cudaCheckError();
        }
        else {
            h_A = A;
        }

        printf("%s\n",s);
        for (i=0; i<m; i++) { //assumption m<lda
            for (j=0; j<n; j++) {
                 printf("%8.5f ", h_A[i+j*lda]);
            }
            printf("\n");
        }

        if (Device_) {
            if (h_A) free(h_A);
        }
        return 0;
    }

    template <typename IndexType_, typename ValueType_> 
    static __global__ void random_matrix_kernel(IndexType_ m, IndexType_ n, ValueType_ * A, IndexType_ lda, IndexType_ seed) {
        IndexType_ i,j,index;

        for (j=threadIdx.y+blockIdx.y*blockDim.y; j<n; j+=blockDim.y*gridDim.y) {
            for (i=threadIdx.x+blockIdx.x*blockDim.x; i<m; i+=blockDim.x*gridDim.x) {
                index = i+j*lda;
                A[index] = ((ValueType_)(((index+seed) % 253)+1))/256.0;
            }            
        }
    }


    template <typename IndexType_, typename ValueType_>
    int random_matrix(IndexType_ m, IndexType_ n, ValueType_ * A, IndexType_ lda, IndexType_ seed, cudaStream_t s){

        if (m > lda) {
            WARNING("random_matrix - invalid parameter (m > lda)");
            return -1;
        }
        
        //device code  
        dim3 gridDim, blockDim;
        blockDim.x = 256;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x  = min((m+blockDim.x-1)/blockDim.x, 65535);
        gridDim.y  = min((n+blockDim.y-1)/blockDim.y, 65535);
        gridDim.z  = 1;
        random_matrix_kernel<IndexType_,ValueType_><<<gridDim,blockDim,0,s>>>(m,n,A,lda,seed);
        cudaCheckError();

        /*
        //host code
        IndexType_ i,j,index;
        ValueType_ * h_A;

        h_A = (ValueType_ *)malloc(lda*n*sizeof(ValueType_));
        if (!h_A) {
            WARNING("random_matrix - malloc failed");
            return -1;
        }
        cudaMemcpy(h_A, A, lda*n*sizeof(ValueType_), cudaMemcpyDeviceToHost); cudaCheckError();
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                index = i+j*lda;
                h_A[index] = ((ValueType_)(((index+seed) % 253)+1))/256.0;
                //printf("%d, %d, %f, ",index, (index+seed) % 253, ((ValueType_)(((index+seed) % 253)+1))/256.0);
            }
            printf("\n");
        }
        cudaMemcpy(A, h_A, lda*n*sizeof(ValueType_), cudaMemcpyHostToDevice); cudaCheckError();
        */
        return 0;
    }

    template <typename IndexType_, typename ValueType_> 
    static __global__ void block_axmy_kernel(IndexType_ n, IndexType_ k, ValueType_ * alpha, ValueType_ *X, IndexType_ ldx, ValueType_ *Y, IndexType_ ldy) {
        IndexType_ i,j,index;

        for (j=threadIdx.y+blockIdx.y*blockDim.y; j<k; j+=blockDim.y*gridDim.y) {
            for (i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) {
                index = i+j*ldx;
                Y[index] = Y[index] - alpha[j]*X[index];
            }            
        }
    }

    template <typename IndexType_, typename ValueType_>
    int block_axmy(IndexType_ n, IndexType_ k, ValueType_ * alpha, ValueType_ *X, IndexType_ ldx, ValueType_ *Y, IndexType_ ldy, cudaStream_t s) {
        //device code  
        dim3 gridDim, blockDim;
        blockDim.x = 256;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x  = min((n+blockDim.x-1)/blockDim.x, 65535);
        gridDim.y  = min((k+blockDim.y-1)/blockDim.y, 65535);
        gridDim.z  = 1;
        block_axmy_kernel<IndexType_,ValueType_><<<gridDim,blockDim,0,s>>>(n,k,alpha,X,ldx,Y,ldy);
        cudaCheckError();

        return 0;
    }

    template <typename IndexType_, typename ValueType_> 
    static __global__ void collect_sqrt_kernel(IndexType_ n, ValueType_ *A, IndexType_ lda, ValueType_ *E) {
        IndexType_ i,index;

        for (i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) {
            index = i+i*lda;
            E[i] = std::sqrt(static_cast<ValueType_>(A[index]));
        }                    
    }

    template <typename IndexType_, typename ValueType_>
    int collect_sqrt_memcpy(IndexType_ n, ValueType_ *A, IndexType_ lda, ValueType_ * E, cudaStream_t s) {
        //device code  
        dim3 gridDim, blockDim;
        blockDim.x = min(n,256);
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x  = min((n+blockDim.x-1)/blockDim.x, 65535);
        gridDim.y  = 1;
        gridDim.z  = 1;
        collect_sqrt_kernel<IndexType_,ValueType_><<<gridDim,blockDim,0,s>>>(n,A,lda,E);
        cudaCheckError();

        return 0;
    }

    template <typename IndexType_, typename ValueType_, bool eigenvecs> 
    static __global__ void convert_to_ascending_order_kernel(IndexType_ n, ValueType_ * H_dst, IndexType_ ldd, ValueType_ * E_dst, ValueType_ * H_src, IndexType_ lds, ValueType_ * E_src){
        IndexType_ i,j,indexs,indexd;

        for (i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) {
            E_dst[n-(i+1)] = E_src[i];
        }

        if (eigenvecs) {
            for (j=threadIdx.y+blockIdx.y*blockDim.y; j<n; j+=blockDim.y*gridDim.y) {
                for (i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) {
                    indexs = i+j*lds;
                    indexd = i+(n-(j+1))*ldd;
                    H_dst[indexd] = H_src[indexs];
                }            
            }
        }
    }

    template <typename IndexType_, typename ValueType_, bool eigenvecs>
    int convert_to_ascending_order(IndexType_ n, ValueType_ * H_dst, IndexType_ ldd, ValueType_ * E_dst, ValueType_ * H_src, IndexType_ lds, ValueType_ * E_src, cudaStream_t s){
        //device code  
        dim3 gridDim, blockDim;
        blockDim.x = min(n,256);
        blockDim.y = (256+blockDim.x-1)/blockDim.x;
        blockDim.z = 1;
        gridDim.x  = min((n+blockDim.x-1)/blockDim.x, 65535);
        gridDim.y  = min((n+blockDim.y-1)/blockDim.y, 65535);
        gridDim.z  = 1;
        convert_to_ascending_order_kernel<IndexType_,ValueType_,eigenvecs><<<gridDim,blockDim,0,s>>>(n,H_dst,ldd,E_dst,H_src,lds,E_src);
        cudaCheckError();

        return 0;
    }

    template <typename IndexType_, typename ValueType_> 
    static __global__ void compute_cond_kernel (IndexType_ n, ValueType_ *E) {         
        //WARNING: must be launched with a single thread and block only 
        E[0] = E[0]/E[n-1];
    }

    template <typename IndexType_, typename ValueType_>
    int compute_cond(IndexType_ n, ValueType_ *E, cudaStream_t s) { 
        //device code  
        dim3 gridDim, blockDim;
        blockDim.x = 1;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x  = 1;
        gridDim.y  = 1;
        gridDim.z  = 1;
        compute_cond_kernel<IndexType_,ValueType_><<<gridDim,blockDim,0,s>>>(n,E);
        cudaCheckError();

        return 0;
    } 

    template <typename IndexType_, typename ValueType_>
    int lobpcg_simplified(cublasHandle_t cublasHandle,
                          cusolverDnHandle_t cusolverHandle,
                          IndexType_ n, IndexType_ k,
                          /*const*/ Matrix<IndexType_,ValueType_> * A,
                          ValueType_ * __restrict__ eigVecs_dev,
                          ValueType_ * __restrict__ eigVals_dev,
                          IndexType_ mit, ValueType_ tol,
                          ValueType_ * __restrict__ work_dev,
                          IndexType_ & iter) {
        
        // -------------------------------------------------------
        // Variable declaration
        // -------------------------------------------------------
        LaplacianMatrix<IndexType_,ValueType_>* L = dynamic_cast< LaplacianMatrix<IndexType_,ValueType_>* >(A);
        //LaplacianMatrix<IndexType_,ValueType_>* L = static_cast< LaplacianMatrix<IndexType_,ValueType_>* >(A);

        cudaEvent_t event=NULL;
        cudaStream_t s_alg=NULL,s_cublas=NULL,s_cusolver=NULL,s_cusparse=NULL;
        //cudaStream_t s_magma=NULL; //magma_types.h: typedef cudaStream_t magma_queue_t;

        // Useful constants
        const ValueType_ zero = 0.0;
        const ValueType_ one  = 1.0;
        const ValueType_ mone =-1.0;
        const bool sp = (sizeof(ValueType_) == 4);
        const ValueType_ eps      = (sp) ? 1.1920929e-7f : 2.220446049250313e-16;
        const ValueType_ max_kappa= (sp) ? 4    : 8;
        //const bool use_magma = SPECTRAL_USE_MAGMA; //true; //false;
        const bool use_throttle = SPECTRAL_USE_THROTTLE; //true; //false;
        const bool use_normalized_laplacian = SPECTRAL_USE_NORMALIZED_LAPLACIAN; //true; //false;
        const bool use_R_orthogonalization = SPECTRAL_USE_R_ORTHOGONALIZATION; //true; //false;

        // Status flags
        //int minfo;
        //int nb;
        //int lwork;
        //int liwork;
        int Lwork;
        int k3 = 3*k;
        int k2 = 2*k;
        int sz = k2;
        //int nb1;
        //int nb2;
        //int nb3;
        ValueType_ kappa;
        ValueType_ kappa_average;        
        //ValueType_ * h_wa=NULL;
        //ValueType_ * h_work=NULL;
        //IndexType_ * h_iwork=NULL;
        //ValueType_ * h_E=NULL;

        // Loop indices
        IndexType_ i,j,start;
        
        //LOBPCG subspaces        
        ValueType_ * E=NULL;
        ValueType_ * Y=NULL;
        ValueType_ * X=NULL;
        ValueType_ * R=NULL;
        ValueType_ * P=NULL;
        ValueType_ * Z=NULL;
        ValueType_ * AX=NULL;
        ValueType_ * AR=NULL;
        ValueType_ * AP=NULL;
        ValueType_ * Q=NULL;
        ValueType_ * BX=NULL;
        ValueType_ * BR=NULL;
        ValueType_ * BP=NULL;
        ValueType_ * G=NULL;
        ValueType_ * H=NULL;
        ValueType_ * HU=NULL;
        ValueType_ * HVT=NULL;
        ValueType_ * nrmR=NULL;
        ValueType_ * h_nrmR=NULL;
        ValueType_ * h_kappa_history=NULL;
        ValueType_ * Workspace=NULL;

        double t_start=0.0,t_end=0.0,t_total=0.0,t_setup=0.0,t_mm=0.0,t_bdot=0.0,t_gemm=0.0,t_potrf=0.0,t_trsm=0.0,t_syevd=0.0,t_custom=0.0,t_prec=0.0,t1=0.0,t2=0.0;
 
        t_start =timer();

        // Random number generator
        curandGenerator_t randGen;
        
        // -------------------------------------------------------
        // Check that parameters are valid
        // -------------------------------------------------------
        if(n < 1) {
            WARNING("lobpcg_simplified - invalid parameter (n<1)");
            return -1;
        }
        if(k < 1) {
            WARNING("lobpcg_simplified - invalid parameter (k<1)");
            return -1;
        }
        if(tol < 0) {
            WARNING("lobpcg_simplified - invalid parameter (tol<0)");
            return -1;
        }
        if(k > n) {
            WARNING("lobpcg_simplified - invalid parameters (k>n)");
            return -1;
        }
        
        E = eigVals_dev;      //array, not matrix, of eigenvalues 
        Y = &work_dev[0];     //alias Y = [X,R,P]
        X = &work_dev[0];     //notice that X, R and P must be continuous in memory
        R = &work_dev[k*n];   //R = A*X-B*X*E
        P = &work_dev[2*k*n];        
        Z = &work_dev[3*k*n]; //alias Z = A*Y = [AX,AR,AP] 
        AX= &work_dev[3*k*n]; //track A*X
        AR= &work_dev[4*k*n]; //track A*R (also used as temporary storage)
        AP= &work_dev[5*k*n]; //track A*P
        Q = &work_dev[6*k*n]; //alias Q = B*Y = [BX,BR,BP] 
        BX= &work_dev[6*k*n]; //track B*X
        BR= &work_dev[7*k*n]; //track B*R
        BP= &work_dev[8*k*n]; //track B*P
        G   = &work_dev[9*k*n];
        H   = &work_dev[9*k*n +   k3*k3];
        HU  = &work_dev[9*k*n + 2*k3*k3];
        HVT = &work_dev[9*k*n + 3*k3*k3];
        nrmR= &work_dev[9*k*n + 4*k3*k3];
        Workspace = &work_dev[9*k*n + 4*k3*k3+k];

        // -------------------------------------------------------
        // Variable initialization
        // -------------------------------------------------------
        t1 =timer();

        // create a CUDA stream
        cudaEventCreate(&event); cudaCheckError();
        cudaStreamCreate(&s_alg); cudaCheckError();
        ///s_alg=NULL;

        // set pointer mode in CUBLAS
        CHECK_CUBLAS(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
        
        // save and set streams in CUBLAS and CUSOLVER/MAGMA
        CHECK_CUBLAS(cublasGetStream(cublasHandle, &s_cublas));
        CHECK_CUBLAS(cublasSetStream(cublasHandle, s_alg));
        //if (use_magma) {
        //    CHECK_CUBLAS(magmablasGetKernelStream(&s_magma)); //returns cublasStatus_t
        //    CHECK_CUBLAS(magmablasSetKernelStream(s_alg));    //returns cublasStatus_t
        //}
        //else {
            CHECK_CUSOLVER(cusolverDnGetStream(cusolverHandle, &s_cusolver));
            CHECK_CUSOLVER(cusolverDnSetStream(cusolverHandle, s_alg));
        //}
        // save and set streams in Laplacian/CUSPARSE    
        L->getCUDAStream(&s_cusparse);     
        L->setCUDAStream(s_alg);    

        // Initialize random number generator
        CHECK_CURAND(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen, 123456/*time(NULL)*/));

        // Initialize initial LOBPCG subspace
        CHECK_CURAND(curandGenerateNormalX(randGen, X, k*n, zero, one));
        ///random_matrix<IndexType_,ValueType_>(n,k,X,n,17,s_alg);       
        //print_matrix<IndexType_,ValueType_,true>(3,3,X,n,"X");
 
        // set nxk matrices P=0, AP=0 and BP=0
        cudaMemsetAsync(P,  0, n*k*sizeof(ValueType_), s_alg); cudaCheckError();
        cudaMemsetAsync(AP, 0, n*k*sizeof(ValueType_), s_alg);cudaCheckError();
        cudaMemsetAsync(BP, 0, n*k*sizeof(ValueType_), s_alg);cudaCheckError();

        //if (use_magma) {
        //    //NB can be obtained through magma_get_dsytrd_nb(N). 
        //    //If JOBZ = MagmaVec and N > 1, LWORK >= max( 2*N + N*NB, 1 + 6*N + 2*N**2 ). 
        //    //If JOBZ = MagmaVec and N > 1, LIWORK >= 3 + 5*N.
        //    nb1    = magma_get_xsytrd_nb(k, zero);
        //    nb2    = magma_get_xsytrd_nb(k2,zero);
        //    nb3    = magma_get_xsytrd_nb(k3,zero);
        //    nb     = max(nb1,max(nb2,nb3)); //this is needed to ensure allocations are correct even if sz is changed from k, 2*k to 3*k below       
        //    lwork  = max(2*k3+k3*nb, 1+6*k3+2*k3*k3); 
        //    liwork = 3 + 5*k3;
        //    //printf("k=%d, nb=%d, lwork=%d, liwork=%d\n",k,nb,lwork,liwork);
        //    h_E    = (ValueType_ *)malloc(k3*sizeof(h_E[0]));
        //    h_wa   = (ValueType_ *)malloc(k3*k3*sizeof(h_wa[0]));
        //    h_work = (ValueType_ *)malloc(lwork*sizeof(h_work[0])); 
        //    h_iwork= (IndexType_ *)malloc(liwork*sizeof(h_iwork[0]));
        //    if ((!h_E) || (!h_wa) || (!h_work) || (!h_iwork)) {
        //        WARNING("lobpcg_simplified - malloc failed");
        //        return -1;
        //    }
        //}

        if(use_throttle) {
            cudaHostAlloc(&h_nrmR, 2*sizeof(h_nrmR[0]), cudaHostAllocDefault); //pinned memory 
            cudaCheckError();
        }
        else{
            h_nrmR = (ValueType_ *)malloc((k+1)*sizeof(h_nrmR[0]));
        }

        h_kappa_history = (ValueType_ *)malloc((mit+1)*sizeof(h_kappa_history[0]));
        if ((!h_kappa_history) || (!h_nrmR) ) {
            WARNING("lobpcg_simplified - malloc/cudaHostAlloc failed");
            return -1;
        }
        h_kappa_history[0] = -log10(eps)/2.0; 
        //printf("h_kappa_history[0] = %f\n",h_kappa_history[0]);
        t2 =timer();
        t_setup+=t2-t1;

        // -------------------------------------------------------
        // Algorithm
        // -------------------------------------------------------
        //BX= B*X
        if (use_normalized_laplacian) {
            L->dm(k, one, X, zero, BX);
        }
        else { 
            cudaMemcpyAsync(BX, X, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError(); 
        }
        //print_matrix<IndexType_,ValueType_,true>(3,3,BX,n,"BX=B*X");

        //G = X'*BX
        t1 =timer();
        CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, k, k, n, &one, X, n, BX, n, &zero, G, k));
        t2 =timer();
        t_bdot+=t2-t1;
        //print_matrix<IndexType_,ValueType_,true>(k,k,G,k,"G=X'*BX");

        //S = chol(G);
        t1 =timer();
        //if (false /*use_magma*/) {
        //    MAGMACHECK(magma_xpotrf(k, G, k, &minfo));
        //}
        //else{
            CHECK_CUSOLVER(cusolverXpotrf_bufferSize(cusolverHandle,k,G,k,&Lwork)); //Workspace was already over allocated earlier
            CHECK_CUSOLVER(cusolverXpotrf(cusolverHandle,k,G,k,Workspace,Lwork,(int *)&Workspace[Lwork]));
        //}
        t2 =timer();
        t_potrf+=t2-t1;
        //print_matrix<IndexType_,ValueType_,true>(k,k,G,k,"S=chol(G,lower_part_stored)");

        //X = X/S (notice that in MATLAB S has L', therefore extra transpose (CUBLAS_OP_T) is required below)
        t1 =timer();
        CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,k,&one,G,k, X,n));
        //BX=BX/S
        CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,k,&one,G,k,BX,n));
        t2 =timer();
        t_trsm+=t2-t1;
        //print_matrix<IndexType_,ValueType_,true>(3,3,X, n,"X = X/S");
        //print_matrix<IndexType_,ValueType_,true>(3,3,BX,n,"BX=BX/S");

        //AX = A*X        
        t1 =timer();
        L->mm(k, one, X, zero, AX);
        t2 =timer();
        t_mm+=t2-t1;
        //print_matrix<IndexType_,ValueType_,true>(3,3,AX,n,"AX=A*X");

        //H = X'*AX
        t1 =timer();
        CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, k, k, n, &one, X, n, AX, n, &zero, H, k));
        t2 =timer();
        t_bdot+=t2-t1;
        //print_matrix<IndexType_,ValueType_,true>(k,k,H,k,"H=X'*A*X");

        //[W,E]=eig(H)
        t1 =timer();
        //if (use_magma) {
        //    MAGMACHECK(magma_xsyevd(k, H, k, h_E, h_wa, k, h_work, lwork, h_iwork, liwork, &minfo));
        //    cudaMemcpy(E, h_E, k*sizeof(ValueType_), cudaMemcpyHostToDevice); cudaCheckError();
         //}
        //else {
            //WARNING: using eigVecs_dev as a temporary space
            CHECK_CUSOLVER(cusolverXgesvd_bufferSize(cusolverHandle,k,k,H,k,HU,k,HVT,k,&Lwork)); //Workspace was already over allocated earlier
            CHECK_CUSOLVER(cusolverXgesvd(cusolverHandle,k,k,H,k,eigVecs_dev,HU,k,HVT,k,Workspace,Lwork,NULL,(int *)&Workspace[Lwork]));
            convert_to_ascending_order<IndexType_,ValueType_,true>(k,H,k,E,HU,k,eigVecs_dev,s_alg);
        //}
        t2 =timer();
        t_syevd+=t2-t1;
        //print_matrix<IndexType_,ValueType_,true>(k,1,E,k,"E, from [W,E]=eig(H)");
        //print_matrix<IndexType_,ValueType_,true>(k,k,H,k,"W, from [W,E]=eig(H)");
          
        //X = X*W 
        t1 =timer();
        CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k, &one, X, n, H, k, &zero, AR, n));
        cudaMemcpyAsync(X, AR, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError(); 
        //BX = BX*W
        CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k, &one,BX, n, H, k, &zero, AR, n));
        cudaMemcpyAsync(BX,AR, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError(); 
        //AX = AX*W (notice that R=AX below, which we will use later on when computing residual R)
        CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k, &one, AX, n, H, k, &zero, R, n));
        cudaMemcpyAsync(AX, R, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError(); 
        t2 =timer();
        t_gemm+=t2-t1;
        //print_matrix<IndexType_,ValueType_,true>(3,3,X, n,"X = X*W");
        //print_matrix<IndexType_,ValueType_,true>(3,3,BX,n,"BX=BX*W");
        //print_matrix<IndexType_,ValueType_,true>(3,3,AX,n,"AX=AX*W");

        // start main loop
        for(i=0; i<mit; i++){
            //save iteration number (an output parameter)
            iter = i;

            //R = AX - BX*E
            t1 =timer();
            block_axmy<IndexType_,ValueType_>(n,k,E,BX,n,R,n,s_alg);
            t2 =timer();
            t_custom+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(3,3,R,n,"R=AX-X*E");

            //check convergence
            t1 =timer();
            if (use_throttle) { //use throttle technique
                if ((i % 2) == 0) {
                    //notice can not use G=R'*BR, because it is != R'*R, which is needed at this point
                    CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, k, k, n, &one, R, n, R, n, &zero, G, k));
                    collect_sqrt_memcpy<IndexType_,ValueType_>(k,G,k,nrmR,s_alg);
                    cudaMemcpyAsync(h_nrmR, &nrmR[k-1], sizeof(ValueType_), cudaMemcpyDeviceToHost, s_alg); cudaCheckError();
                    cudaEventRecord(event, s_alg); cudaCheckError();
                }
                if (((i+1) % 2) == 0) {
                    cudaEventSynchronize(event); cudaCheckError();
                    if (h_nrmR[0] < tol) {
                        break;
                    }            
                }
            }
            else { //use naive approach
                for (j=0; j<k; j++) {
                    CHECK_CUBLAS(cublasXnrm2(cublasHandle, n, &R[j*n], 1, &h_nrmR[j])); 
                    //printf("h_nrmR[%d]=%f \n", j,h_nrmR[j]);
                }
                if (h_nrmR[k-1] < tol) {
                    break;
                }     
            }          
            t2 =timer();
            t_custom+=t2-t1;

            //R=M\R preconditioning step
            t1 =timer();
            L->prec_solve(k,one,R,eigVecs_dev);
            t2 =timer();
            t_prec+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(3,3,R,n,"R=M\R");
 
            //make residuals B orthogonal to X (I'm not sure this is needed)
            //R = R - X*(BX'*R);
            if (use_R_orthogonalization) {
                t1 =timer();
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, k, k, n, &one, BX, n, R, n, &zero, G, k));
                t2 =timer();
                t_bdot+=t2-t1;
                
                t1 =timer();
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k, &mone, X, n, G, k, &one, R, n));
                t2 =timer();
                t_gemm+=t2-t1;
            }

            //BX= B*X
            if (use_normalized_laplacian) {
                L->dm(k, one, R, zero, BR);
            }
            else { 
                cudaMemcpyAsync(BR, R, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError(); 
            }
            //G=R'*BR
            t1 =timer();
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, k, k, n, &one, R, n, BR, n, &zero, G, k));
            t2 =timer();
            t_bdot+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(k,k,G,k,"G=R'*BR");
           
            //S = chol(G);
            t1 =timer();
            //if (false /*use_magma*/) {
            //    MAGMACHECK(magma_xpotrf(k, G, k, &minfo));
            //}
            //else{
                CHECK_CUSOLVER(cusolverXpotrf_bufferSize(cusolverHandle,k,G,k,&Lwork)); //Workspace was already over allocated earlier
                CHECK_CUSOLVER(cusolverXpotrf(cusolverHandle,k,G,k,Workspace,Lwork,(int *)&Workspace[Lwork]));
            // }
            t2 =timer();
            t_potrf+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(k,k,G,k,"S=chol(G,lower_part_stored)");

            //R = R/S (notice that in MATLAB S has L', therefore extra transpose (CUBLAS_OP_T) is required below)
            t1 =timer();
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,k,&one,G,k,R,n));
            //BR=BR/S
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,k,&one,G,k,BR,n));
            t2 =timer();
            t_trsm+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(3,3, R,n,"R = R/S");
            //print_matrix<IndexType_,ValueType_,true>(3,3,BR,n,"BR=BR/S");

            //G=Y'*Q (where Q=B*Y) 
            //std::cout<<"size : "<< sz<< std::endl;
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,Y,sz,"Y");
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,Q,sz,"Q");
            t1 =timer();
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, sz, sz, n, &one, Y, n, Q, n, &zero, G, sz));
            t2 =timer();
            t_bdot+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,G,sz,"G=Y'*Q");

            //check conditioning of the subspace restart strategy
            //WARNING: We need to compute condition number of matrix G in ||.||_2. 
            //Normally to compute these condition number we would perform a singular value 
            //decomposition and have kappa(G) = max_singular_value/min_singular_value of G.
            t1 =timer();
            //if (use_magma) {
            //    //Notice also that MAGMA does not have GPU interface to singular_value decomposition,
            //    //but it does have one for the eigenvalue routine. We will take advantage of it:    
            //    //Since G is symmetric we can also say that singular_value(G) = sqrt(eigenvalue(A'*A)) = eigenvalue(A), 
            //    //therefore kappa(G) = max_eigenvalue_G/min_eigenvalue_G
            //    //[W,E]=eig(H)
            //    MAGMACHECK(magma_xsyevd_cond(sz, G, sz, h_E, h_wa, sz, h_work, lwork, h_iwork, liwork, &minfo));
            //    kappa = log10(h_E[sz-1]/h_E[0])+1; 
            //    //printf("cond=%f (%f/%f),  %f\n",h_E[sz-1]/h_E[0],h_E[sz-1],h_E[0],log10(h_E[sz-1]/h_E[0])+1);
            //    //print_matrix<IndexType_,ValueType_,false>(sz,1,h_E,sz,"h_E, sing_values(G)=eig(G) in cond(G)");
            //}
            //else { 
                if (sz > n*k) { //WARNING: using eigVecs_dev as a temporary space (for sz singular values)
                    WARNING("lobpcg_simplified - temporary space insufficient (sz > n*k)");
                    return -1;
                }
                CHECK_CUSOLVER(cusolverXgesvd_bufferSize(cusolverHandle,sz,sz,G,sz,HU,sz,HVT,sz,&Lwork)); //Workspace was already over allocated earlier
                CHECK_CUSOLVER(cusolverXgesvd(cusolverHandle,sz,sz,G,sz,eigVecs_dev,HU,sz,HVT,sz,Workspace,Lwork,NULL,(int *)&Workspace[Lwork]));
                compute_cond<IndexType_,ValueType_>(sz,eigVecs_dev,s_alg); //condition number is eigVecs_dev[0] = eigVecs_dev[0]/eigVecs_dev[sz-1]
                cudaMemcpy(&kappa, eigVecs_dev, sizeof(ValueType_), cudaMemcpyDeviceToHost); cudaCheckError();//FIX LATER using throttle technique
                kappa = log10(kappa)+1.0;
                ///kappa =1;
            //}
            t2 =timer();
            t_syevd+=t2-t1;
            //printf("cond=%f\n", kappa);
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,G,sz,"G, should not have changed cond(G)");
            

            //WARNING: will compute average (not mean, like MATLAB code) because it is easier to code
            start = max(0,i-10-((int)round(log(static_cast<float>(k)))));
            kappa_average = zero;
            for(j=start; j<=i; j++) {
                //printf("%f ",h_kappa_history[j]);
                kappa_average += h_kappa_history[j];
            }
            //printf("\n");
            kappa_average = kappa_average/(i-start+1); 
            if (((kappa/kappa_average) > 2 && (kappa > 2)) || (kappa > max_kappa)) {
                //exclude P from Y=[X,R] 
                sz = k2;
                //printf("restart=%d (%d, %d, %d, %d) (%f %f %f)\n",i,(int)round(log(k)),i-10-((int)round(log(k))),start,i-start+1,kappa,kappa_average,max_kappa);
                //recompute G=Y'*Q and corresponding condition number (excluding P)
                t1 =timer();
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, sz, sz, n, &one, Y, n, Q, n, &zero, G, sz));
                t2 =timer();
                t_bdot+=t2-t1;
                //print_matrix<IndexType_,ValueType_,true>(sz,sz,G,sz,"G=Y'*Y");

                t1 =timer();
                //if (use_magma) {
                //    MAGMACHECK(magma_xsyevd_cond(sz, G, sz, h_E, h_wa, sz, h_work, lwork, h_iwork, liwork, &minfo));
                //    kappa = log10(h_E[sz-1]/h_E[0])+1;                 
                //}
                //else {
                    if (sz > n*k) { //WARNING: using eigVecs_dev as a temporary space (for sz singular values)
                        WARNING("lobpcg_simplified - temporary space insufficient (sz > n*k)");
                        return -1;
                    }
                    CHECK_CUSOLVER(cusolverXgesvd_bufferSize(cusolverHandle,sz,sz,G,sz,HU,sz,HVT,sz,&Lwork)); //Workspace was already over allocated earlier
                    CHECK_CUSOLVER(cusolverXgesvd(cusolverHandle,sz,sz,G,sz,eigVecs_dev,HU,sz,HVT,sz,Workspace,Lwork,NULL,(int *)&Workspace[Lwork]));
                    compute_cond<IndexType_,ValueType_>(sz,eigVecs_dev,s_alg); //condition number is eigVecs_dev[0] = eigVecs_dev[0]/eigVecs_dev[sz-1]
                    cudaMemcpy(&kappa, eigVecs_dev, sizeof(ValueType_), cudaMemcpyDeviceToHost); cudaCheckError(); //FIX LATER using throttle technique
                    kappa = log10(kappa)+1.0;
                    ///kappa =1;
                //}    
                t2 =timer();
                t_syevd+=t2-t1;
                //printf("cond=%f\n", kappa);
                //print_matrix<IndexType_,ValueType_,false>(sz,1,h_E,sz,"h_E, sing_values(G)=eig(G) in cond(G)");
                //print_matrix<IndexType_,ValueType_,true>(sz,sz,G,sz,"G, should not have changed cond(G)");
            }
            h_kappa_history[i+1] = kappa;

            //WARNING: the computation of condition number destroys the  
            //lower triangle of G (including diagonal), so it must be recomputed again.
            //recompute G=Y'*Q 
            t1 =timer(); 
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, sz, sz, n, &one, Y, n, Q, n, &zero, G, sz));
            t2 =timer();
            t_bdot+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,G,sz,"G=Y'*Q (recomputing)");
            
            //AR = A*R        
            t1 =timer();
            L->mm(k, one, R, zero, AR);
            t2 =timer();
            t_mm+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(3,k,AR,n,"AR=A*R");

            //H = Y'*Z
            t1 =timer();
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, sz, sz, n, &one, Y, n, Z, n, &zero, H, sz));
            t2 =timer();
            t_bdot+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,H,sz,"H=Y'*A*Y");

            //Approach 1:
            //S = chol(G);
            t1 =timer();
            //if (false /*use_magma*/) {
            //    MAGMACHECK(magma_xpotrf(sz, G, sz, &minfo));
            //}
            //else{
                CHECK_CUSOLVER(cusolverXpotrf_bufferSize(cusolverHandle,sz,G,sz,&Lwork)); //Workspace was over already over allocated earlier
                CHECK_CUSOLVER(cusolverXpotrf(cusolverHandle,sz,G,sz,Workspace,Lwork,(int *)&Workspace[Lwork]));
            //}
            t2 =timer();
            t_potrf+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,G,sz,"S=chol(G,lower_part_stored)");

            //H = S'\ H /S (notice that in MATLAB S has L', therefore extra transpose (CUBLAS_OP_T) is required below)
            t1 =timer();
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,sz,sz,&one,G,sz,H,sz));
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,sz,sz,&one,G,sz,H,sz));
            t2 =timer();
            t_trsm+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,H,sz,"H = S'\\ H /S");

            //[W,E]=eig(S'\ H /S);
            t1 =timer();
            //if (use_magma) {
            //    MAGMACHECK(magma_xsyevd(sz, H, sz, h_E, h_wa, sz, h_work, lwork, h_iwork, liwork, &minfo));
            //    cudaMemcpy(E, h_E, k*sizeof(ValueType_), cudaMemcpyHostToDevice); cudaCheckError(); //only have k spaces in E, but h_E have sz eigs
            //}
            //else {
                if (sz > n*k) { //WARNING: using eigVecs_dev as a temporary space (for sz singular values)
                    WARNING("lobpcg_simplified - temporary space insufficient (sz > n*k)");
                    return -1;
                }
                CHECK_CUSOLVER(cusolverXgesvd_bufferSize(cusolverHandle,sz,sz,H,sz,HU,sz,HVT,sz,&Lwork)); //Workspace was already over allocated earlier
                CHECK_CUSOLVER(cusolverXgesvd(cusolverHandle,sz,sz,H,sz,eigVecs_dev,HU,sz,HVT,sz,Workspace,Lwork,NULL,(int *)&Workspace[Lwork]));
                convert_to_ascending_order<IndexType_,ValueType_,true>(sz,H,sz,E,HU,sz,eigVecs_dev,s_alg);
            //}
            t2 =timer();
            t_syevd+=t2-t1;
            //print_matrix<IndexType_,ValueType_,false>(sz,1,h_E,sz,"h_E, from [W,E]=eig(S'\\ H /S)");
            //print_matrix<IndexType_,ValueType_, true>(k,1,E,k,"E, smallest k eigs from [W,E]=eig(S'\\ H /S)");
            //print_matrix<IndexType_,ValueType_, true>(sz,sz,H,sz,"W, from [W,E]=eig(S'\\ H /S)");

            //W=S\W (recover original eigvectors)
            t1 =timer();
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,sz,sz,&one,G,sz,H,sz));
            t2 =timer();
            t_trsm+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(sz,sz,H,sz,"W=S\\W");

            //WARNING: using eigVecs_dev as a temporary space
            //X =Y*W(:,1:k); //notice can not use X  for the result directly, because it is part of Y (and aliased by Y)
            t1 =timer();
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, sz, &one, Y, n, H, sz, &zero, eigVecs_dev, n));
            cudaMemcpyAsync(X, eigVecs_dev, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg);  cudaCheckError();
            //BX=Q*W(:,1:k); //notice can not use BX for the result directly, because it is part of Q (and aliased by Q)
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, sz, &one, Q, n, H, sz, &zero, eigVecs_dev, n));
            cudaMemcpyAsync(BX, eigVecs_dev, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError(); 
            //AX=Z*W(:,1:k); //notice can not use AX for the result directly, because it is part of Z (and aliased by Z) 
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, sz, &one, Z, n, H, sz, &zero, eigVecs_dev, n));
            cudaMemcpyAsync(AX, eigVecs_dev, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError(); 
            t2 =timer();
            t_gemm+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(3,3, X,n,"X =Y*W(:,1:k)");
            //print_matrix<IndexType_,ValueType_,true>(3,3,BX,n,"BX=Q*W(:,1:k)");
            //print_matrix<IndexType_,ValueType_,true>(3,3,AX,n,"AX=Z*W(:,1:k)");

            //update P
            t1 =timer();
            if (sz == k2) {
                //P = R*W(k+1:2*k,1:k);
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k, &one, R, n, &H[k], sz, &zero, P, n));
                //BP=BR*W(k+1:2*k,1:k);
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k, &one,BR, n, &H[k], sz, &zero,BP, n));
                //AP=AR*W(k+1:2*k,1:k);
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k, &one,AR, n, &H[k], sz, &zero,AP, n));              
                //print_matrix<IndexType_,ValueType_,true>(3,3, P,n,"P = R*W(k+1:2*k,1:k)");
                //print_matrix<IndexType_,ValueType_,true>(3,3,BP,n,"BP=BR*W(k+1:2*k,1:k)");
                //print_matrix<IndexType_,ValueType_,true>(3,3,AP,n,"AP=AR*W(k+1:2*k,1:k)");
            }
            else { //(sz == k3) 
                //P= R*W(k+1:2*k,1:k) +  P*W(2*k+1:3*k,1:k); and recall that Y = [X,R,P]
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k2, &one, &Y[n*k], n, &H[k], sz, &zero, eigVecs_dev, n));
                cudaMemcpyAsync(P, eigVecs_dev, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg);cudaCheckError();
                //BP=BR*W(k+1:2*k,1:k) + BP*W(2*k+1:3*k,1:k); and recall that Q = [BX,BR,BP]
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k2, &one, &Q[n*k], n, &H[k], sz, &zero, eigVecs_dev, n));
                cudaMemcpyAsync(BP, eigVecs_dev, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg);cudaCheckError();
                //AP=AR*W(k+1:2*k,1:k) + AP*W(2*k+1:3*k,1:k); and recall that Z = [AX,AR,AP]
                CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, n, k, k2, &one, &Z[n*k], n, &H[k], sz, &zero, eigVecs_dev, n));
                cudaMemcpyAsync(AP, eigVecs_dev, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg);cudaCheckError();
                //print_matrix<IndexType_,ValueType_,true>(3,3, P,n,"P = R*W(k+1:2*k,1:k) +  P*W(2*k+1:3*k,1:k)");
                //print_matrix<IndexType_,ValueType_,true>(3,3,BP,n,"BP=BR*W(k+1:2*k,1:k) + BP*W(2*k+1:3*k,1:k)");
                //print_matrix<IndexType_,ValueType_,true>(3,3,AP,n,"AP=AR*W(k+1:2*k,1:k) + AP*W(2*k+1:3*k,1:k)");
            }
            t2 =timer();
            t_gemm+=t2-t1;

            //orthonormalize P
            //G = P'*BP
            t1 =timer();
            CHECK_CUBLAS(cublasXgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, k, k, n, &one, P, n, BP, n, &zero, G, k));
            t2 =timer();
            t_bdot+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(k,k,G,k,"G=P'*BP");

            //S = chol(G);
            t1 =timer();
            //if (false /*use_magma*/) {
            //    MAGMACHECK(magma_xpotrf(k, G, k, &minfo));
            //}
            //else{
                CHECK_CUSOLVER(cusolverXpotrf_bufferSize(cusolverHandle,k,G,k,&Lwork)); //Workspace was already over allocated earlier
                CHECK_CUSOLVER(cusolverXpotrf(cusolverHandle,k,G,k,Workspace,Lwork,(int *)&Workspace[Lwork]));
            //}
            t2 =timer();
            t_potrf+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(k,k,G,k,"S=chol(G,lower_part_stored)");

            //P  =  P/S (notice that in MATLAB S has L', therefore extra transpose (CUBLAS_OP_T) is required below)
            t1 =timer();
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,k,&one,G,k,P,n));
            //BP = BP/S 
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,k,&one,G,k,BP,n));
            //AP = AP/S 
            CHECK_CUBLAS(cublasXtrsm(cublasHandle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,k,&one,G,k,AP,n));
            t2 =timer();
            t_trsm+=t2-t1;
            //print_matrix<IndexType_,ValueType_,true>(3,3, P,n,"P = P/S");
            //print_matrix<IndexType_,ValueType_,true>(3,3,BP,n,"BP=BP/S");
            //print_matrix<IndexType_,ValueType_,true>(3,3,AP,n,"AP=AP/S");

            //copy AX into R (to satisfy assumption in the next iteration)
            cudaMemcpyAsync(R, AX, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg);cudaCheckError(); 
            //reset sz for the next iteration
            sz=k3;
            //printf("--- %d ---\n",i);
        }
        t_end =timer();
        t_total+=t_end-t_start;

        //WARNING: In the MATLAB code at this point X is made a section of A,
        //which I don't think is necessary, but something to keep in mind,
        //in case something goes wrong in the future.
        cudaMemcpyAsync(eigVecs_dev, X, n*k*sizeof(ValueType_), cudaMemcpyDeviceToDevice, s_alg); cudaCheckError();

        //free temporary host memory
        cudaStreamSynchronize(s_alg); cudaCheckError();
        //if (use_magma) {
        //    if (h_E) free(h_E);
        //    if (h_wa) free(h_wa);
        //    if (h_work) free(h_work);
        //    if (h_iwork) free(h_iwork);
        //}
        if(use_throttle) {
            cudaFreeHost(h_nrmR);cudaCheckError(); //pinned
        }
        else {
            if (h_nrmR) free(h_nrmR);
        }
        if (h_kappa_history) free(h_kappa_history);
        cudaEventDestroy(event);cudaCheckError();
        if (s_alg) {cudaStreamDestroy(s_alg);cudaCheckError();}
        //revert CUBLAS and CUSOLVER/MAGMA streams
        CHECK_CUBLAS(cublasSetStream(cublasHandle, s_cublas));
        //if (use_magma) {
        //    CHECK_CUBLAS(magmablasSetKernelStream(s_magma)); //returns cublasStatus_t
        //}
        //else {
            CHECK_CUSOLVER(cusolverDnSetStream(cusolverHandle, s_cusolver));
        //}
        //revert Laplacian/CUSPARSE streams     
        L->setCUDAStream(s_cusparse);
    
#ifdef COLLECT_TIME_STATISTICS
        //timing statistics
        printf("-------------------------\n");
        printf("time eigsolver [total] %f\n",t_total);
        printf("time eigsolver [L->pr] %f\n",t_prec);
        printf("time eigsolver [potrf] %f\n",t_potrf);
        printf("time eigsolver [syevd] %f\n",t_syevd);
        printf("time eigsolver [trsm]  %f\n",t_trsm);
        printf("time eigsolver [bdot]  %f\n",t_bdot);
        printf("time eigsolver [gemm]  %f\n",t_gemm);
        printf("time eigsolver [L->mm] %f\n",t_mm);
        printf("time eigsolver [custom]%f\n",t_custom);
        printf("time eigsolver [setup] %f\n",t_setup);
        printf("time eigsolver [other] %f\n",t_total-(t_prec+t_potrf+t_syevd+t_trsm+t_bdot+t_gemm+t_mm+t_custom+t_setup));
#endif        
        return 0;
    }

    // =========================================================
    // Explicit instantiation
    // =========================================================

    template int lobpcg_simplified<int,float>
    (cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
     int n, int k,
     /*const*/ Matrix<int,float> * A,
     float * __restrict__ eigVecs_dev,
     float * __restrict__ eigVals_dev,
     int maxIter, float tol,
     float * __restrict__ work_dev,
     int &iter); 

    template int lobpcg_simplified<int,double>
    (cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
     int n, int k,
     /*const*/ Matrix<int,double> * A,
     double * __restrict__ eigVecs_dev,
     double * __restrict__ eigVals_dev,
     int maxIter, double tol,
     double * __restrict__ work_dev,
     int &iter);

}
//#endif //enable/disable lobpcg

