//#ifdef NVGRAPH_PARTITION

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

#include "include/partition.hxx"

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "include/nvgraph_error.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/matrix.hxx"
#include "include/lanczos.hxx"
#include "include/kmeans.hxx"
#include "include/debug_macros.h"
#include "include/lobpcg.hxx"
#include "include/sm_utils.h"

//#define COLLECT_TIME_STATISTICS 1
//#undef COLLECT_TIME_STATISTICS

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

  // =========================================================
  // Useful macros
  // =========================================================

  // Get index of matrix entry
#define IDX(i,j,lda) ((i)+(j)*(lda))

//    namespace {
//      /// Get string associated with NVGRAPH error flag
//      static
//      const char* nvgraphGetErrorString(NVGRAPH_ERROR e) {
//  switch(e) {
//  case NVGRAPH_OK:                  return "NVGRAPH_OK";
//  case NVGRAPH_ERR_BAD_PARAMETERS:  return "NVGRAPH_ERR_BAD_PARAMETERS";
//  case NVGRAPH_ERR_UNKNOWN:         return "NVGRAPH_ERR_UNKNOWN";
//  case NVGRAPH_ERR_CUDA_FAILURE:    return "NVGRAPH_ERR_CUDA_FAILURE";
//  case NVGRAPH_ERR_THRUST_FAILURE:  return "NVGRAPH_ERR_THRUST_FAILURE";
//  case NVGRAPH_ERR_IO:              return "NVGRAPH_ERR_IO";
//  case NVGRAPH_ERR_NOT_IMPLEMENTED: return "NVGRAPH_ERR_NOT_IMPLEMENTED";
//  case NVGRAPH_ERR_NO_MEMORY:       return "NVGRAPH_ERR_NO_MEMORY";
//  default:                       return "unknown NVGRAPH error";
//  }
//      }
//    }

     template <typename IndexType_, typename ValueType_, bool Device_, bool print_transpose>
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
            cudaMemcpy(h_A, A, lda*n*sizeof(ValueType_), cudaMemcpyDeviceToHost); cudaCheckError()
        }
        else {
            h_A = A;
        }

        printf("%s\n",s);
        if(print_transpose){
            for (j=0; j<n; j++) {
                for (i=0; i<m; i++) { //assumption m<lda
                    printf("%8.5f, ", h_A[i+j*lda]);
                }
                printf("\n");
            }
        }
        else {
            for (i=0; i<m; i++) { //assumption m<lda
                for (j=0; j<n; j++) {
                    printf("%8.5f, ", h_A[i+j*lda]);
                }
                printf("\n");
            }
        }

        if (Device_) {
            if (h_A) free(h_A);
        }
        return 0;
    }

    template <typename IndexType_, typename ValueType_>
    static __global__ void scale_obs_kernel(IndexType_ m, IndexType_ n, ValueType_ *obs) {
        IndexType_ i,j,k,index,mm;
        ValueType_ alpha,v,last;
        bool valid;
        //ASSUMPTION: kernel is launched with either 2, 4, 8, 16 or 32 threads in x-dimension

        //compute alpha
        mm =(((m+blockDim.x-1)/blockDim.x)*blockDim.x); //m in multiple of blockDim.x
        alpha=0.0;
        //printf("[%d,%d,%d,%d] n=%d, li=%d, mn=%d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y, n, li, mn);    
        for (j=threadIdx.y+blockIdx.y*blockDim.y; j<n; j+=blockDim.y*gridDim.y) {
            for (i=threadIdx.x; i<mm; i+=blockDim.x) {
                //check if the thread is valid
                valid  = i<m;
                
                //get the value of the last thread
                last = utils::shfl(alpha, blockDim.x-1, blockDim.x);      
                
                //if you are valid read the value from memory, otherwise set your value to 0
                alpha = (valid) ? obs[i+j*m] : 0.0;
                alpha = alpha*alpha;

                //do prefix sum (of size warpSize=blockDim.x =< 32)
                for (k=1; k<blockDim.x; k*=2) {
                    v = utils::shfl_up(alpha, k, blockDim.x);
                    if (threadIdx.x >= k) alpha+=v;
                }
                //shift by last
                alpha+=last;
            }
        }

        //scale by alpha      
        alpha = utils::shfl(alpha, blockDim.x-1, blockDim.x);
        alpha = std::sqrt(alpha); 
        for (j=threadIdx.y+blockIdx.y*blockDim.y; j<n; j+=blockDim.y*gridDim.y) {
            for (i=threadIdx.x; i<m; i+=blockDim.x) { //blockDim.x=32
                index = i+j*m;
                obs[index] = obs[index]/alpha;
            }            
        }
    }

    template <typename IndexType_>
    IndexType_ next_pow2(IndexType_ n) {
        IndexType_ v;
        //Reference: 
        //http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float
        v = n-1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v+1;
    }

    template <typename IndexType_, typename ValueType_>
    cudaError_t scale_obs(IndexType_ m, IndexType_ n, ValueType_ *obs) {
        IndexType_ p2m;
        dim3 nthreads, nblocks;

        //find next power of 2
        p2m = next_pow2<IndexType_>(m);
        //setup launch configuration
        nthreads.x = max(2,min(p2m,32));
        nthreads.y = 256/nthreads.x;
        nthreads.z = 1;
        nblocks.x  = 1;
        nblocks.y  = (n + nthreads.y - 1)/nthreads.y;
        nblocks.z  = 1;
        //printf("m=%d(%d),n=%d,obs=%p, nthreads=(%d,%d,%d),nblocks=(%d,%d,%d)\n",m,p2m,n,obs,nthreads.x,nthreads.y,nthreads.z,nblocks.x,nblocks.y,nblocks.z);

        //launch scaling kernel (scale each column of obs by its norm)
        scale_obs_kernel<IndexType_,ValueType_><<<nblocks,nthreads>>>(m,n,obs);
        cudaCheckError();

        return cudaSuccess;
    }

  // =========================================================
  // Spectral partitioner
  // =========================================================

  /// Compute spectral graph partition
  /** Compute partition for a weighted undirected graph. This
   *  partition attempts to minimize the cost function:
   *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
   *
   *  @param G Weighted graph in CSR format
   *  @param nParts Number of partitions.
   *  @param nEigVecs Number of eigenvectors to compute.
   *  @param maxIter_lanczos Maximum number of Lanczos iterations.
   *  @param restartIter_lanczos Maximum size of Lanczos system before
   *    implicit restart.
   *  @param tol_lanczos Convergence tolerance for Lanczos method.
   *  @param maxIter_kmeans Maximum number of k-means iterations.
   *  @param tol_kmeans Convergence tolerance for k-means algorithm.
   *  @param parts (Output, device memory, n entries) Partition
   *    assignments.
   *  @param iters_lanczos On exit, number of Lanczos iterations
   *    performed.
   *  @param iters_kmeans On exit, number of k-means iterations
   *    performed.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR partition( ValuedCsrGraph<IndexType_,ValueType_>& G,
           IndexType_ nParts,
           IndexType_ nEigVecs,
           IndexType_ maxIter_lanczos,
           IndexType_ restartIter_lanczos,
           ValueType_ tol_lanczos,
           IndexType_ maxIter_kmeans,
           ValueType_ tol_kmeans,
           IndexType_ * __restrict__ parts,
           Vector<ValueType_> &eigVals,
           Vector<ValueType_> &eigVecs,
           IndexType_ & iters_lanczos,
           IndexType_ & iters_kmeans) {

    // -------------------------------------------------------
    // Check that parameters are valid
    // -------------------------------------------------------

    if(nParts < 1) {
      WARNING("invalid parameter (nParts<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs < 1) {
      WARNING("invalid parameter (nEigVecs<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter_lanczos < nEigVecs) {
      WARNING("invalid parameter (maxIter_lanczos<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter_lanczos < nEigVecs) {
      WARNING("invalid parameter (restartIter_lanczos<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol_lanczos < 0) {
      WARNING("invalid parameter (tol_lanczos<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter_kmeans < 0) {
      WARNING("invalid parameter (maxIter_kmeans<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol_kmeans < 0) {
      WARNING("invalid parameter (tol_kmeans<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // -------------------------------------------------------
    // Variable declaration
    // -------------------------------------------------------

    // Useful constants
    const ValueType_ zero = 0;
    const ValueType_ one  = 1;

    // Loop index
    IndexType_ i;

    // Matrix dimension
    IndexType_ n = G.get_num_vertices();

    // CUDA stream
    //   TODO: handle non-zero streams
    cudaStream_t stream = 0;

    // Matrices
    Matrix<IndexType_, ValueType_> * A;  // Adjacency matrix
    Matrix<IndexType_, ValueType_> * L;  // Laplacian matrix

    // Whether to perform full reorthogonalization in Lanczos
    bool reorthogonalize_lanczos = false;

    // k-means residual
    ValueType_ residual_kmeans;

    bool scale_eigevec_rows=SPECTRAL_USE_SCALING_OF_EIGVECS; //true; //false;

    double t1=0.0,t2=0.0,t_kmeans=0.0;

    // -------------------------------------------------------
    // Spectral partitioner
    // -------------------------------------------------------

    // Compute eigenvectors of Laplacian
    
    // Initialize Laplacian
    A = new CsrMatrix<IndexType_,ValueType_>(G);
    L = new LaplacianMatrix<IndexType_,ValueType_>(*A);

    // Compute smallest eigenvalues and eigenvectors
    CHECK_NVGRAPH(computeSmallestEigenvectors(*L, nEigVecs, maxIter_lanczos,
             restartIter_lanczos, tol_lanczos,
             reorthogonalize_lanczos, iters_lanczos,
             eigVals.raw(), eigVecs.raw()));   
    //eigVals.dump(0, nEigVecs);
    //eigVecs.dump(0, nEigVecs);
    //eigVecs.dump(n, nEigVecs);
    //eigVecs.dump(2*n, nEigVecs);
    // Whiten eigenvector matrix
    for(i=0; i<nEigVecs; ++i) {
      ValueType_ mean, std;
      mean = thrust::reduce(thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i,n)),
                            thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i+1,n)));
      cudaCheckError();
      mean /= n;
      thrust::transform(thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i,n)),
                        thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i+1,n)), 
                        thrust::make_constant_iterator(mean), 
                        thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i,n)), 
                        thrust::minus<ValueType_>());
      cudaCheckError();
      std = Cublas::nrm2(n, eigVecs.raw()+IDX(0,i,n), 1)/std::sqrt(static_cast<ValueType_>(n));
      thrust::transform(thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i,n)),
                        thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i+1,n)),
                        thrust::make_constant_iterator(std),
                        thrust::device_pointer_cast(eigVecs.raw()+IDX(0,i,n)), 
                        thrust::divides<ValueType_>());
      cudaCheckError();
    }

   delete L;
   delete A;

    // Transpose eigenvector matrix
    //   TODO: in-place transpose
    {
      Vector<ValueType_> work(nEigVecs*n, stream);
      Cublas::set_pointer_mode_host();
      Cublas::geam(true, false, nEigVecs, n,
       &one, eigVecs.raw(), n,
       &zero, (ValueType_*) NULL, nEigVecs,
       work.raw(), nEigVecs);
      CHECK_CUDA(cudaMemcpyAsync(eigVecs.raw(), work.raw(),
         nEigVecs*n*sizeof(ValueType_),
         cudaMemcpyDeviceToDevice));
    }

     // Clean up
  

    if (scale_eigevec_rows) {
        //WARNING: notice that at this point the matrix has already been transposed, so we are scaling columns
        scale_obs(nEigVecs,n,eigVecs.raw()); cudaCheckError()
        //print_matrix<IndexType_,ValueType_,true,false>(nEigVecs-ifirst,n,obs,nEigVecs-ifirst,"Scaled obs");
        //print_matrix<IndexType_,ValueType_,true,true>(nEigVecs-ifirst,n,obs,nEigVecs-ifirst,"Scaled obs");
    }

    t1=timer();

    //eigVecs.dump(0, nEigVecs*n);
    // Find partition with k-means clustering
    CHECK_NVGRAPH(kmeans(n, nEigVecs, nParts, 
          tol_kmeans, maxIter_kmeans,
          eigVecs.raw(), parts,
          residual_kmeans, iters_kmeans));
    t2=timer();
    t_kmeans+=t2-t1;
#ifdef COLLECT_TIME_STATISTICS
    printf("time k-means %f\n",t_kmeans);
#endif        


    return NVGRAPH_OK;
  }

  // =========================================================
  // Spectral partitioner
  // =========================================================

  /// Compute spectral graph partition
  /** Compute partition for a weighted undirected graph. This
   *  partition attempts to minimize the cost function:
   *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
   *
   *  @param G Weighted graph in CSR format
   *  @param nParts Number of partitions.
   *  @param nEigVecs Number of eigenvectors to compute.
   *  @param maxIter_lanczos Maximum number of Lanczos iterations.
   *  @param restartIter_lanczos Maximum size of Lanczos system before
   *    implicit restart.
   *  @param tol_lanczos Convergence tolerance for Lanczos method.
   *  @param maxIter_kmeans Maximum number of k-means iterations.
   *  @param tol_kmeans Convergence tolerance for k-means algorithm.
   *  @param parts (Output, device memory, n entries) Partition
   *    assignments.
   *  @param iters_lanczos On exit, number of Lanczos iterations
   *    performed.
   *  @param iters_kmeans On exit, number of k-means iterations
   *    performed.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR partition_lobpcg( ValuedCsrGraph<IndexType_,ValueType_>& G, Matrix<IndexType_,ValueType_> * M, cusolverDnHandle_t cusolverHandle,
           IndexType_ nParts,
           IndexType_ nEigVecs,
           IndexType_ maxIter_lanczos,
           ValueType_ tol_lanczos,
           IndexType_ maxIter_kmeans,
           ValueType_ tol_kmeans,
           IndexType_ * __restrict__ parts,
           Vector<ValueType_> &eigVals,
           Vector<ValueType_> &eigVecs,
           IndexType_ & iters_lanczos,
           IndexType_ & iters_kmeans) {

    // -------------------------------------------------------
    // Check that parameters are valid
    // -------------------------------------------------------

    if(nParts < 1) {
      WARNING("invalid parameter (nParts<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs < 1) {
      WARNING("invalid parameter (nEigVecs<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter_lanczos < nEigVecs) {
      WARNING("invalid parameter (maxIter_lanczos<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol_lanczos < 0) {
      WARNING("invalid parameter (tol_lanczos<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter_kmeans < 0) {
      WARNING("invalid parameter (maxIter_kmeans<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol_kmeans < 0) {
      WARNING("invalid parameter (tol_kmeans<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // -------------------------------------------------------
    // Variable declaration
    // -------------------------------------------------------

    // Useful constants
    const ValueType_ zero = 0;
    const ValueType_ one  = 1;

    // Loop index
    //IndexType_ i;

    // Matrix dimension
    IndexType_ n = G.get_num_vertices();

    // CUDA stream
    //   TODO: handle non-zero streams
    cudaStream_t stream = 0;

    // Matrices
    Matrix<IndexType_, ValueType_> * A;  // Adjacency matrix
    Matrix<IndexType_, ValueType_> * L;  // Laplacian matrix

    // k-means residual
    ValueType_ residual_kmeans;

    bool scale_eigevec_rows=SPECTRAL_USE_SCALING_OF_EIGVECS; //true; //false;

    double t1=0.0,t2=0.0,t_kmeans=0.0;

    // Compute eigenvectors of Laplacian
    
    // Initialize Laplacian
    A = new CsrMatrix<IndexType_,ValueType_>(G);
    L = new LaplacianMatrix<IndexType_,ValueType_>(*A);

    // LOBPCG use
    //bool use_lobpcg=SPECTRAL_USE_LOBPCG; //true; //false;
    bool use_preconditioning=SPECTRAL_USE_PRECONDITIONING; //true; //false;
    int lwork=0,lwork1=0,lwork2=0,lwork3=0,lwork_potrf=0,lwork_gesvd=0;
    double t_setup=0.0,t_solve=0.0;
    //ValueType_ * eigVals;
    //ValueType_ * work;
    ValueType_ * lanczosVecs=0;
    //ValueType_ * obs;

    //lanczosVecs are not allocated yet, but should not be touched in *_bufferSize routine
    CHECK_CUSOLVER(cusolverXpotrf_bufferSize(cusolverHandle,  nEigVecs,lanczosVecs,  nEigVecs,&lwork1));
    CHECK_CUSOLVER(cusolverXpotrf_bufferSize(cusolverHandle,2*nEigVecs,lanczosVecs,2*nEigVecs,&lwork2));
    CHECK_CUSOLVER(cusolverXpotrf_bufferSize(cusolverHandle,3*nEigVecs,lanczosVecs,3*nEigVecs,&lwork3));
    lwork_potrf = max(lwork1,max(lwork2,lwork3));
    CHECK_CUSOLVER(cusolverXgesvd_bufferSize(cusolverHandle,  nEigVecs,  nEigVecs,lanczosVecs,nEigVecs,lanczosVecs,nEigVecs,lanczosVecs,nEigVecs,&lwork1));
    CHECK_CUSOLVER(cusolverXgesvd_bufferSize(cusolverHandle,2*nEigVecs,2*nEigVecs,lanczosVecs,nEigVecs,lanczosVecs,nEigVecs,lanczosVecs,nEigVecs,&lwork2));
    CHECK_CUSOLVER(cusolverXgesvd_bufferSize(cusolverHandle,3*nEigVecs,3*nEigVecs,lanczosVecs,nEigVecs,lanczosVecs,nEigVecs,lanczosVecs,nEigVecs,&lwork3));
    lwork_gesvd = max(lwork1,max(lwork2,lwork3));
    lwork = max(lwork_potrf,lwork_gesvd);
    //allocating +2 to hold devInfo for cuSolver, which is of type int, using 2 rather than 1 just in case
    //sizeof(ValueType_) < sizeof(IntType_). Notice that this ratio will not be more than 2.
    //6*nEigVecs*n - Y=[X,R,P] and Z=[Q,T,V], where X and others are of size nEigVecs x n
    //36*nEigVecs*nEigVecs for G, H, HU and HVT, each of max size 3*nEigVecs x 3*nEigVecs
    //nEigVecs - nrmR
    //lwork - Workspace max Lwork value (for either potrf or gesvd)
    //2 - devInfo
    auto rmm_result = RMM_ALLOC(&lanczosVecs, (9*nEigVecs*n + 36*nEigVecs*nEigVecs + nEigVecs + lwork+2)*sizeof(ValueType_), stream); 
    rmmCheckError(rmm_result);

    //Setup preconditioner M for Laplacian L
    t1=timer();
    if (use_preconditioning) {
        L->prec_setup(M);
    }
    t2=timer();
    t_setup+=t2-t1;

    //Run the eigensolver (with preconditioning)
    t1=timer();
    if(lobpcg_simplified(Cublas::get_handle(),cusolverHandle, 
                                  n, nEigVecs, L, 
                                  eigVecs.raw(), eigVals.raw(),
                                  maxIter_lanczos,tol_lanczos,
                                  lanczosVecs, //work array (on device)
                                  iters_lanczos) != 0)
    {
      WARNING("error in eigensolver");
      return NVGRAPH_ERR_UNKNOWN;
    }
                
    t2=timer();
    t_solve+=t2-t1;
    #ifdef COLLECT_TIME_STATISTICS
    printf("time eigsolver setup %f\n",t_setup);
    printf("time eigsolver solve %f\n",t_solve);
    #endif    

    delete L;
    delete A;
    // Transpose eigenvector matrix
    //   TODO: in-place transpose
    {
      Vector<ValueType_> work(nEigVecs*n, stream);
      Cublas::set_pointer_mode_host();
      Cublas::geam(true, false, nEigVecs, n,
       &one, eigVecs.raw(), n,
       &zero, (ValueType_*) NULL, nEigVecs,
       work.raw(), nEigVecs);
      CHECK_CUDA(cudaMemcpyAsync(eigVecs.raw(), work.raw(),
         nEigVecs*n*sizeof(ValueType_),
         cudaMemcpyDeviceToDevice));
    }

    if (scale_eigevec_rows) {
        //WARNING: notice that at this point the matrix has already been transposed, so we are scaling columns
        scale_obs(nEigVecs,n,eigVecs.raw()); cudaCheckError();
        //print_matrix<IndexType_,ValueType_,true,false>(nEigVecs-ifirst,n,obs,nEigVecs-ifirst,"Scaled obs");
        //print_matrix<IndexType_,ValueType_,true,true>(nEigVecs-ifirst,n,obs,nEigVecs-ifirst,"Scaled obs");
    }

    t1=timer();

    //eigVecs.dump(0, nEigVecs*n);
    // Find partition with k-means clustering
    CHECK_NVGRAPH(kmeans(n, nEigVecs, nParts, 
          tol_kmeans, maxIter_kmeans,
          eigVecs.raw(), parts,
          residual_kmeans, iters_kmeans));
    t2=timer();
    t_kmeans+=t2-t1;
#ifdef COLLECT_TIME_STATISTICS
    printf("time k-means %f\n",t_kmeans);
#endif        

    return NVGRAPH_OK;
  }

  // =========================================================
  // Analysis of graph partition
  // =========================================================

  namespace {
    /// Functor to generate indicator vectors
    /** For use in Thrust transform
     */
    template <typename IndexType_, typename ValueType_>
    struct equal_to_i_op {
      const IndexType_ i;
    public:
      equal_to_i_op(IndexType_ _i) : i(_i) {}
      template<typename Tuple_>
      __host__ __device__ void operator()(Tuple_ t) {
  thrust::get<1>(t)
    = (thrust::get<0>(t) == i) ? (ValueType_) 1.0 : (ValueType_) 0.0;
      }
    };
  }

  /// Compute cost function for partition
  /** This function determines the edges cut by a partition and a cost
   *  function:
   *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
   *  Graph is assumed to be weighted and undirected.
   *
   *  @param G Weighted graph in CSR format
   *  @param nParts Number of partitions.
   *  @param parts (Input, device memory, n entries) Partition
   *    assignments.
   *  @param edgeCut On exit, weight of edges cut by partition.
   *  @param cost On exit, partition cost function.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR analyzePartition(ValuedCsrGraph<IndexType_,ValueType_> & G,
            IndexType_ nParts,
            const IndexType_ * __restrict__ parts,
            ValueType_ & edgeCut, ValueType_ & cost) {
    
    //using namespace thrust;

    // -------------------------------------------------------
    // Variable declaration
    // -------------------------------------------------------

    // Loop index
    IndexType_ i;

    // Matrix dimension
    IndexType_ n = G.get_num_vertices();

    // Values for computing partition cost
    ValueType_ partEdgesCut, partSize;

    // CUDA stream
    //   TODO: handle non-zero streams
    cudaStream_t stream = 0;
    
    // Device memory
    Vector<ValueType_> part_i(n, stream);
    Vector<ValueType_> Lx(n, stream);

    // Adjacency and Laplacian matrices
    Matrix<IndexType_, ValueType_> * A;
    Matrix<IndexType_, ValueType_> * L;

    // -------------------------------------------------------
    // Implementation
    // -------------------------------------------------------

    // Check that parameters are valid
    if(nParts < 1) {
      WARNING("invalid parameter (nParts<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // Initialize cuBLAS
    Cublas::set_pointer_mode_host();

    // Initialize Laplacian
    A = new CsrMatrix<IndexType_,ValueType_>(G);
    L = new LaplacianMatrix<IndexType_,ValueType_>(*A);

    // Initialize output
    cost    = 0;
    edgeCut = 0;

    // Iterate through partitions
    for(i=0; i<nParts; ++i) {
    
      // Construct indicator vector for ith partition
      thrust::for_each( thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(parts),
                thrust::device_pointer_cast(part_i.raw()))),
                thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(parts+n),
                thrust::device_pointer_cast(part_i.raw()+n))),
                equal_to_i_op<IndexType_,ValueType_>(i));
      cudaCheckError();

      // Compute size of ith partition
      Cublas::dot(n, part_i.raw(), 1, part_i.raw(), 1, &partSize);
      partSize = round(partSize);
      if(partSize < 0.5) {
  WARNING("empty partition");
  continue;
      }
         
      // Compute number of edges cut by ith partition
      L->mv(1, part_i.raw(), 0, Lx.raw());
      Cublas::dot(n, Lx.raw(), 1, part_i.raw(), 1, &partEdgesCut);

      // Record results
      cost    += partEdgesCut/partSize;
      edgeCut += partEdgesCut/2;

    }

    // Clean up and return
    delete L;
    delete A;
    return NVGRAPH_OK;

  }

  // =========================================================
  // Explicit instantiation
  // =========================================================
  template
  NVGRAPH_ERROR partition<int,float>( ValuedCsrGraph<int,float> & G,
          int nParts,
          int nEigVecs,
          int maxIter_lanczos,
          int restartIter_lanczos,
          float tol_lanczos,
          int maxIter_kmeans,
          float tol_kmeans,
          int * __restrict__ parts,
          Vector<float> &eigVals,
          Vector<float> &eigVecs,
          int & iters_lanczos,
          int & iters_kmeans);
  template
  NVGRAPH_ERROR partition<int,double>( ValuedCsrGraph<int,double> & G,
           int nParts,
           int nEigVecs,
           int maxIter_lanczos,
           int restartIter_lanczos,
           double tol_lanczos,
           int maxIter_kmeans,
           double tol_kmeans,
           int * __restrict__ parts,
           Vector<double> &eigVals,
           Vector<double> &eigVecs,
           int & iters_lanczos,
           int & iters_kmeans);



  template 
  NVGRAPH_ERROR partition_lobpcg<int,float>(ValuedCsrGraph<int,float> & G,
           Matrix<int,float> * M, 
           cusolverDnHandle_t cusolverHandle,
           int nParts,
           int nEigVecs,
           int maxIter_lanczos,
           float tol_lanczos,
           int maxIter_kmeans,
           float tol_kmeans,
           int * __restrict__ parts,
           Vector<float> &eigVals,
           Vector<float> &eigVecs,
           int & iters_lanczos,
           int & iters_kmeans);

  template 
  NVGRAPH_ERROR partition_lobpcg<int,double>(ValuedCsrGraph<int,double> & G,
           Matrix<int,double> * M, 
           cusolverDnHandle_t cusolverHandle,
           int nParts,
           int nEigVecs,
           int maxIter_lanczos,
           double tol_lanczos,
           int maxIter_kmeans,
           double tol_kmeans,
           int * __restrict__ parts,
           Vector<double> &eigVals,
           Vector<double> &eigVecs,
           int & iters_lanczos,
           int & iters_kmeans);
  template
  NVGRAPH_ERROR analyzePartition<int,float>(ValuedCsrGraph<int,float> & G,
           int nParts,
           const int * __restrict__ parts,
           float & edgeCut, float & cost);
  template
  NVGRAPH_ERROR analyzePartition<int,double>(ValuedCsrGraph<int,double> & G,
            int nParts,
            const int * __restrict__ parts,
            double & edgeCut, double & cost);

}
//#endif //NVGRAPH_PARTITION

