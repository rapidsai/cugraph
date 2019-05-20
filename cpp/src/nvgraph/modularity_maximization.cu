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
//#ifdef NVGRAPH_PARTITION

#include "include/modularity_maximization.hxx"

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
#include "cuda_profiler_api.h"
#endif

#ifdef COLLECT_TIME_STATISTICS
static double timer (void) {
    struct timeval tv;
    cudaDeviceSynchronize();
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif        

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
//	switch(e) {
//	case NVGRAPH_OK:                  return "NVGRAPH_OK";
//	case NVGRAPH_ERR_BAD_PARAMETERS:  return "NVGRAPH_ERR_BAD_PARAMETERS";
//	case NVGRAPH_ERR_UNKNOWN:         return "NVGRAPH_ERR_UNKNOWN";
//	case NVGRAPH_ERR_CUDA_FAILURE:    return "NVGRAPH_ERR_CUDA_FAILURE";
//	case NVGRAPH_ERR_THRUST_FAILURE:  return "NVGRAPH_ERR_THRUST_FAILURE";
//	case NVGRAPH_ERR_IO:              return "NVGRAPH_ERR_IO";
//	case NVGRAPH_ERR_NOT_IMPLEMENTED: return "NVGRAPH_ERR_NOT_IMPLEMENTED";
//	case NVGRAPH_ERR_NO_MEMORY:       return "NVGRAPH_ERR_NO_MEMORY";
//	default:                       return "unknown NVGRAPH error";
//	}
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
  // Spectral modularity_maximization
  // =========================================================

  /** Compute partition for a weighted undirected graph. This
   *  partition attempts to minimize the cost function:
   *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
   *
   *  @param G Weighted graph in CSR format
   *  @param nClusters Number of partitions.
   *  @param nEigVecs Number of eigenvectors to compute.
   *  @param maxIter_lanczos Maximum number of Lanczos iterations.
   *  @param restartIter_lanczos Maximum size of Lanczos system before
   *    implicit restart.
   *  @param tol_lanczos Convergence tolerance for Lanczos method.
   *  @param maxIter_kmeans Maximum number of k-means iterations.
   *  @param tol_kmeans Convergence tolerance for k-means algorithm.
   *  @param parts (Output, device memory, n entries) Cluster
   *    assignments.
   *  @param iters_lanczos On exit, number of Lanczos iterations
   *    performed.
   *  @param iters_kmeans On exit, number of k-means iterations
   *    performed.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR modularity_maximization( ValuedCsrGraph<IndexType_,ValueType_>& G,
           IndexType_ nClusters,
           IndexType_ nEigVecs,
           IndexType_ maxIter_lanczos,
           IndexType_ restartIter_lanczos,
           ValueType_ tol_lanczos,
           IndexType_ maxIter_kmeans,
           ValueType_ tol_kmeans,
           IndexType_ * __restrict__ clusters,
           Vector<ValueType_> &eigVals,
           Vector<ValueType_> &eigVecs,
           IndexType_ & iters_lanczos,
           IndexType_ & iters_kmeans) {

    // -------------------------------------------------------
    // Check that parameters are valid
    // -------------------------------------------------------

    if(nClusters < 1) {
      WARNING("invalid parameter (nClusters<1)");
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
    Matrix<IndexType_, ValueType_> * B;  // Modularity matrix

    // Whether to perform full reorthogonalization in Lanczos
    bool reorthogonalize_lanczos = false;

    // k-means residual
    ValueType_ residual_kmeans;

    bool scale_eigevec_rows=true; //true; //false;
#ifdef COLLECT_TIME_STATISTICS
    double t1=0.0,t2=0.0;
#endif 
    // -------------------------------------------------------
    // Spectral partitioner
    // -------------------------------------------------------

    // Compute eigenvectors of Modularity Matrix
 #ifdef COLLECT_TIME_STATISTICS
    t1=timer();
 #endif        
    // Initialize Modularity Matrix
    A = new CsrMatrix<IndexType_,ValueType_>(G);
    B = new ModularityMatrix<IndexType_,ValueType_>(*A, static_cast<IndexType_>(G.get_num_edges()));

    // Compute smallest eigenvalues and eigenvectors
#ifdef COLLECT_TIME_STATISTICS
    t2=timer();
    printf("%f\n",t2-t1);
#endif        

#ifdef COLLECT_TIME_STATISTICS
    t1=timer();
    cudaProfilerStart();
#endif        

    CHECK_NVGRAPH(computeLargestEigenvectors(*B, nEigVecs, maxIter_lanczos,
             restartIter_lanczos, tol_lanczos,
             reorthogonalize_lanczos, iters_lanczos,
             eigVals.raw(), eigVecs.raw()));   

 #ifdef COLLECT_TIME_STATISTICS
    cudaProfilerStop();
    t2=timer();
    printf("%f\n",t2-t1);
#endif         

#ifdef COLLECT_TIME_STATISTICS
    t1=timer();
#endif    
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
   delete B;
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
        scale_obs(nEigVecs,n,eigVecs.raw()); cudaCheckError()
        //print_matrix<IndexType_,ValueType_,true,false>(nEigVecs-ifirst,n,obs,nEigVecs-ifirst,"Scaled obs");
        //print_matrix<IndexType_,ValueType_,true,true>(nEigVecs-ifirst,n,obs,nEigVecs-ifirst,"Scaled obs");
    }
#ifdef COLLECT_TIME_STATISTICS
    t2=timer();
    printf("%f\n",t2-t1);
#endif        

#ifdef COLLECT_TIME_STATISTICS
    t1=timer();
#endif        
    //eigVecs.dump(0, nEigVecs*n);
    // Find partition with k-means clustering
    CHECK_NVGRAPH(kmeans(n, nEigVecs, nClusters, 
          tol_kmeans, maxIter_kmeans,
          eigVecs.raw(), clusters,
          residual_kmeans, iters_kmeans));
#ifdef COLLECT_TIME_STATISTICS
    t2=timer();
    printf("%f\n\n",t2-t1);
#endif        


    return NVGRAPH_OK;
  }
  //===================================================
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

  /// Compute modularity
  /** This function determines the modularity based on a graph and cluster assignments 
   *  @param G Weighted graph in CSR format
   *  @param nClusters Number of clusters.
   *  @param parts (Input, device memory, n entries) Cluster assignments.
   *  @param modularity On exit, modularity
   */
 template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR analyzeModularity(ValuedCsrGraph<IndexType_,ValueType_> & G,
            IndexType_ nClusters,
            const IndexType_ * __restrict__ parts,
            ValueType_ & modularity) {
    
    //using namespace thrust;

    // -------------------------------------------------------
    // Variable declaration
    // -------------------------------------------------------

    // Loop index
    IndexType_ i;

    // Matrix dimension
    IndexType_ n = G.get_num_vertices();

    // Values for computing partition cost
    ValueType_ partModularity, partSize;

    // CUDA stream
    //   TODO: handle non-zero streams
    cudaStream_t stream = 0;
    
    // Device memory
    Vector<ValueType_> part_i(n, stream);
    Vector<ValueType_> Bx(n, stream);

    // Adjacency and Modularity matrices
    Matrix<IndexType_, ValueType_> * A;
    Matrix<IndexType_, ValueType_> * B;

    // -------------------------------------------------------
    // Implementation
    // -------------------------------------------------------

    // Check that parameters are valid
    if(nClusters < 1) {
      WARNING("invalid parameter (nClusters<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // Initialize cuBLAS
    Cublas::set_pointer_mode_host();

    // Initialize Modularity
    A = new CsrMatrix<IndexType_,ValueType_>(G);
    B = new ModularityMatrix<IndexType_,ValueType_>(*A, static_cast<IndexType_>(G.get_num_edges()));

    // Debug
    //Vector<ValueType_> ones(n,0);
    //ones.fill(1.0);
    //B->mv(1, ones.raw(), 0, Bx.raw());
    //Bx.dump(0,n);
    //Cublas::dot(n, Bx.raw(), 1, ones.raw(), 1, &partModularity);
    //std::cout<< "sum " <<partModularity<< std::endl;

    // Initialize output
     modularity = 0;

    // Iterate through partitions
    for(i=0; i<nClusters; ++i) {
    
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
         
      // Compute modularity
      B->mv(1, part_i.raw(), 0, Bx.raw());
      Cublas::dot(n, Bx.raw(), 1, part_i.raw(), 1, &partModularity);

      // Record results
      modularity += partModularity;
      //std::cout<< "partModularity " <<partModularity<< std::endl;
    }
    //modularity = modularity/nClusters;
    // devide by nnz
    modularity= modularity/B->getEdgeSum();
    // Clean up and return
    delete B;
    delete A;
    return NVGRAPH_OK;

  }

  // =========================================================
  // Explicit instantiation
  // =========================================================
  template
  NVGRAPH_ERROR modularity_maximization<int,float>( ValuedCsrGraph<int,float> & G,
				  int nClusters,
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
  NVGRAPH_ERROR modularity_maximization<int,double>( ValuedCsrGraph<int,double> & G,
				   int nClusters,
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
  NVGRAPH_ERROR analyzeModularity<int,float>(ValuedCsrGraph<int,float> & G,
					 int nClusters,
					 const int * __restrict__ parts,
					 float & modularity);
  template
  NVGRAPH_ERROR analyzeModularity<int,double>(ValuedCsrGraph<int,double> & G,
					  int nClusters,
					  const int * __restrict__ parts,
					  double & modularity);

}
//#endif //NVGRAPH_PARTITION

