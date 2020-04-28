/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <nvgraph/include/nvgraph_error.hxx>
#include <nvgraph/include/nvgraph_vector.hxx>
#include <nvgraph/include/nvgraph_cublas.hxx>
#include <nvgraph/include/spectral_matrix.hxx>
#include <nvgraph/include/lanczos.hxx>
#include <nvgraph/include/kmeans.hxx>
#include <nvgraph/include/debug_macros.h>
#include <nvgraph/include/sm_utils.h>

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
  template <typename vertex_t, typename edge_t, typename weight_t>
  NVGRAPH_ERROR partition(cugraph::experimental::GraphCSR<vertex_t, edge_t, weight_t> const &graph,
                          vertex_t nParts,
                          vertex_t nEigVecs,
                          int maxIter_lanczos,
                          int restartIter_lanczos,
                          weight_t tol_lanczos,
                          int maxIter_kmeans,
                          weight_t tol_kmeans,
                          vertex_t * __restrict__ parts,
                          weight_t *eigVals,
                          weight_t *eigVecs) {

    cudaStream_t stream = 0;

    const weight_t zero{0.0};
    const weight_t one{1.0};

    int iters_lanczos;
    int iters_kmeans;

    edge_t i;
    edge_t n = graph.number_of_vertices;

    // k-means residual
    weight_t residual_kmeans;

    // -------------------------------------------------------
    // Spectral partitioner
    // -------------------------------------------------------

    // Compute eigenvectors of Laplacian
    
    // Initialize Laplacian
    CsrMatrix<vertex_t,weight_t> A(false,
                                   false,
                                   graph.number_of_vertices,
                                   graph.number_of_vertices,
                                   graph.number_of_edges,
                                   0,
                                   graph.edge_data,
                                   graph.offsets,
                                   graph.indices);
    LaplacianMatrix<vertex_t,weight_t>  L(A);

    // Compute smallest eigenvalues and eigenvectors
    CHECK_NVGRAPH(computeSmallestEigenvectors(L, nEigVecs, maxIter_lanczos,
                                              restartIter_lanczos, tol_lanczos,
                                              false, iters_lanczos,
                                              eigVals, eigVecs));   

    // Whiten eigenvector matrix
    for(i=0; i<nEigVecs; ++i) {
      weight_t mean, std;

      mean = thrust::reduce(thrust::device_pointer_cast(eigVecs+IDX(0,i,n)),
                            thrust::device_pointer_cast(eigVecs+IDX(0,i+1,n)));
      cudaCheckError();
      mean /= n;
      thrust::transform(thrust::device_pointer_cast(eigVecs+IDX(0,i,n)),
                        thrust::device_pointer_cast(eigVecs+IDX(0,i+1,n)), 
                        thrust::make_constant_iterator(mean), 
                        thrust::device_pointer_cast(eigVecs+IDX(0,i,n)), 
                        thrust::minus<weight_t>());
      cudaCheckError();
      std = Cublas::nrm2(n, eigVecs+IDX(0,i,n), 1)/std::sqrt(static_cast<weight_t>(n));
      thrust::transform(thrust::device_pointer_cast(eigVecs+IDX(0,i,n)),
                        thrust::device_pointer_cast(eigVecs+IDX(0,i+1,n)),
                        thrust::make_constant_iterator(std),
                        thrust::device_pointer_cast(eigVecs+IDX(0,i,n)), 
                        thrust::divides<weight_t>());
      cudaCheckError();
    }

    // Transpose eigenvector matrix
    //   TODO: in-place transpose
    {
      Vector<weight_t> work(nEigVecs*n, stream);
      Cublas::set_pointer_mode_host();
      Cublas::geam(true, false, nEigVecs, n,
                   &one, eigVecs, n,
                   &zero, (weight_t*) NULL, nEigVecs,
                   work.raw(), nEigVecs);
      CHECK_CUDA(cudaMemcpyAsync(eigVecs, work.raw(),
                                 nEigVecs*n*sizeof(weight_t),
                                 cudaMemcpyDeviceToDevice));
    }

     // Clean up
  

    //eigVecs.dump(0, nEigVecs*n);
    // Find partition with k-means clustering
    CHECK_NVGRAPH(kmeans(n, nEigVecs, nParts, 
                         tol_kmeans, maxIter_kmeans,
                         eigVecs, parts,
                         residual_kmeans, iters_kmeans));

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
  template <typename vertex_t, typename edge_t, typename weight_t>
  NVGRAPH_ERROR analyzePartition(cugraph::experimental::GraphCSR<vertex_t, edge_t, weight_t> const &graph,
                                 vertex_t nParts,
                                 const vertex_t * __restrict__ parts,
                                 weight_t & edgeCut, weight_t & cost) {
    
    cudaStream_t stream = 0;

    edge_t i;
    edge_t n = graph.number_of_vertices;

    weight_t partEdgesCut, partSize;

    // Device memory
    Vector<weight_t> part_i(n, stream);
    Vector<weight_t> Lx(n, stream);

    // Initialize cuBLAS
    Cublas::set_pointer_mode_host();

    // Initialize Laplacian
    CsrMatrix<vertex_t,weight_t> A(false,
                                   false,
                                   graph.number_of_vertices,
                                   graph.number_of_vertices,
                                   graph.number_of_edges,
                                   0,
                                   graph.edge_data,
                                   graph.offsets,
                                   graph.indices);
    LaplacianMatrix<vertex_t,weight_t>  L(A);

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
                        equal_to_i_op<vertex_t, weight_t>(i));
      cudaCheckError();

      // Compute size of ith partition
      Cublas::dot(n, part_i.raw(), 1, part_i.raw(), 1, &partSize);
      partSize = round(partSize);
      if(partSize < 0.5) {
        WARNING("empty partition");
        continue;
      }
         
      // Compute number of edges cut by ith partition
      L.mv(1, part_i.raw(), 0, Lx.raw());
      Cublas::dot(n, Lx.raw(), 1, part_i.raw(), 1, &partEdgesCut);

      // Record results
      cost    += partEdgesCut/partSize;
      edgeCut += partEdgesCut/2;
    }

    // Clean up and return
    return NVGRAPH_OK;
  }

  // =========================================================
  // Explicit instantiation
  // =========================================================
  //template <typename vertex_t, typename edge_t, typename weight_t>
  //NVGRAPH_ERROR partition(cugraph::experimental::GraphCSR<vertex_t, edge_t, weight_t> const &graph,

  template
  NVGRAPH_ERROR partition<int,int,float>(cugraph::experimental::GraphCSR<int, int, float> const &graph,
                                         int nParts,
                                         int nEigVecs,
                                         int maxIter_lanczos,
                                         int restartIter_lanczos,
                                         float tol_lanczos,
                                         int maxIter_kmeans,
                                         float tol_kmeans,
                                         int * __restrict__ parts,
                                         float *eigVals,
                                         float *eigVecs);

  template
  NVGRAPH_ERROR partition<int,int,double>(cugraph::experimental::GraphCSR<int, int, double> const &graph,
                                          int nParts,
                                          int nEigVecs,
                                          int maxIter_lanczos,
                                          int restartIter_lanczos,
                                          double tol_lanczos,
                                          int maxIter_kmeans,
                                          double tol_kmeans,
                                          int * __restrict__ parts,
                                          double *eigVals,
                                          double *eigVecs);



  template
  NVGRAPH_ERROR analyzePartition<int,int,float>(cugraph::experimental::GraphCSR<int,int,float> const &graph,
           int nParts,
           const int * __restrict__ parts,
           float & edgeCut, float & cost);
  template
  NVGRAPH_ERROR analyzePartition<int,int,double>(cugraph::experimental::GraphCSR<int,int,double> const &graph,
            int nParts,
            const int * __restrict__ parts,
            double & edgeCut, double & cost);

}
