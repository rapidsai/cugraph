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
//#ifdef NVGRAPH_PARTITION

#include "include/modularity_maximization.hxx"

#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "include/debug_macros.h"
#include "include/kmeans.hxx"
#include "include/lanczos.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_error.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/sm_utils.h"
#include "include/spectral_matrix.hxx"

//#define COLLECT_TIME_STATISTICS 1
//#undef COLLECT_TIME_STATISTICS

#ifdef COLLECT_TIME_STATISTICS
#include <stddef.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include "cuda_profiler_api.h"
#endif

#ifdef COLLECT_TIME_STATISTICS
static double timer(void)
{
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
#define IDX(i, j, lda) ((i) + (j) * (lda))

template <typename IndexType_, typename ValueType_>
static __global__ void scale_obs_kernel(IndexType_ m, IndexType_ n, ValueType_ *obs)
{
  IndexType_ i, j, k, index, mm;
  ValueType_ alpha, v, last;
  bool valid;
  // ASSUMPTION: kernel is launched with either 2, 4, 8, 16 or 32 threads in x-dimension

  // compute alpha
  mm    = (((m + blockDim.x - 1) / blockDim.x) * blockDim.x);  // m in multiple of blockDim.x
  alpha = 0.0;
  // printf("[%d,%d,%d,%d] n=%d, li=%d, mn=%d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y, n,
  // li, mn);
  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n; j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < mm; i += blockDim.x) {
      // check if the thread is valid
      valid = i < m;

      // get the value of the last thread
      last = utils::shfl(alpha, blockDim.x - 1, blockDim.x);

      // if you are valid read the value from memory, otherwise set your value to 0
      alpha = (valid) ? obs[i + j * m] : 0.0;
      alpha = alpha * alpha;

      // do prefix sum (of size warpSize=blockDim.x =< 32)
      for (k = 1; k < blockDim.x; k *= 2) {
        v = utils::shfl_up(alpha, k, blockDim.x);
        if (threadIdx.x >= k) alpha += v;
      }
      // shift by last
      alpha += last;
    }
  }

  // scale by alpha
  alpha = utils::shfl(alpha, blockDim.x - 1, blockDim.x);
  alpha = std::sqrt(alpha);
  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n; j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < m; i += blockDim.x) {  // blockDim.x=32
      index      = i + j * m;
      obs[index] = obs[index] / alpha;
    }
  }
}

template <typename IndexType_>
IndexType_ next_pow2(IndexType_ n)
{
  IndexType_ v;
  // Reference:
  // http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float
  v = n - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

template <typename IndexType_, typename ValueType_>
cudaError_t scale_obs(IndexType_ m, IndexType_ n, ValueType_ *obs)
{
  IndexType_ p2m;
  dim3 nthreads, nblocks;

  // find next power of 2
  p2m = next_pow2<IndexType_>(m);
  // setup launch configuration
  nthreads.x = max(2, min(p2m, 32));
  nthreads.y = 256 / nthreads.x;
  nthreads.z = 1;
  nblocks.x  = 1;
  nblocks.y  = (n + nthreads.y - 1) / nthreads.y;
  nblocks.z  = 1;
  // printf("m=%d(%d),n=%d,obs=%p,
  // nthreads=(%d,%d,%d),nblocks=(%d,%d,%d)\n",m,p2m,n,obs,nthreads.x,nthreads.y,nthreads.z,nblocks.x,nblocks.y,nblocks.z);

  // launch scaling kernel (scale each column of obs by its norm)
  scale_obs_kernel<IndexType_, ValueType_><<<nblocks, nthreads>>>(m, n, obs);
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
template <typename vertex_t, typename edge_t, typename weight_t>
NVGRAPH_ERROR modularity_maximization(
  cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  vertex_t nClusters,
  vertex_t nEigVecs,
  int maxIter_lanczos,
  int restartIter_lanczos,
  weight_t tol_lanczos,
  int maxIter_kmeans,
  weight_t tol_kmeans,
  vertex_t *__restrict__ clusters,
  weight_t *eigVals,
  weight_t *eigVecs,
  int &iters_lanczos,
  int &iters_kmeans)
{
  cudaStream_t stream = 0;
  const weight_t zero{0.0};
  const weight_t one{1.0};

  edge_t i;
  edge_t n = graph.number_of_vertices;

  // k-means residual
  weight_t residual_kmeans;

  // Compute eigenvectors of Modularity Matrix
  // Initialize Modularity Matrix
  CsrMatrix<vertex_t, weight_t> A(false,
                                  false,
                                  graph.number_of_vertices,
                                  graph.number_of_vertices,
                                  graph.number_of_edges,
                                  0,
                                  graph.edge_data,
                                  graph.offsets,
                                  graph.indices);
  ModularityMatrix<vertex_t, weight_t> B(A, graph.number_of_edges);

  // Compute smallest eigenvalues and eigenvectors
  CHECK_NVGRAPH(computeLargestEigenvectors(B,
                                           nEigVecs,
                                           maxIter_lanczos,
                                           restartIter_lanczos,
                                           tol_lanczos,
                                           false,
                                           iters_lanczos,
                                           eigVals,
                                           eigVecs));

  // eigVals.dump(0, nEigVecs);
  // eigVecs.dump(0, nEigVecs);
  // eigVecs.dump(n, nEigVecs);
  // eigVecs.dump(2*n, nEigVecs);
  // Whiten eigenvector matrix
  for (i = 0; i < nEigVecs; ++i) {
    weight_t mean, std;
    mean = thrust::reduce(thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                          thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)));
    cudaCheckError();
    mean /= n;
    thrust::transform(thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(mean),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::minus<weight_t>());
    cudaCheckError();
    std = Cublas::nrm2(n, eigVecs + IDX(0, i, n), 1) / std::sqrt(static_cast<weight_t>(n));
    thrust::transform(thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(std),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::divides<weight_t>());
    cudaCheckError();
  }

  // Transpose eigenvector matrix
  //   TODO: in-place transpose
  {
    Vector<weight_t> work(nEigVecs * n, stream);
    Cublas::set_pointer_mode_host();
    Cublas::geam(true,
                 false,
                 nEigVecs,
                 n,
                 &one,
                 eigVecs,
                 n,
                 &zero,
                 (weight_t *)NULL,
                 nEigVecs,
                 work.raw(),
                 nEigVecs);
    CHECK_CUDA(cudaMemcpyAsync(
      eigVecs, work.raw(), nEigVecs * n * sizeof(weight_t), cudaMemcpyDeviceToDevice));
  }

  // WARNING: notice that at this point the matrix has already been transposed, so we are scaling
  // columns
  scale_obs(nEigVecs, n, eigVecs);
  cudaCheckError();

  // eigVecs.dump(0, nEigVecs*n);
  // Find partition with k-means clustering
  CHECK_NVGRAPH(kmeans(n,
                       nEigVecs,
                       nClusters,
                       tol_kmeans,
                       maxIter_kmeans,
                       eigVecs,
                       clusters,
                       residual_kmeans,
                       iters_kmeans));

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
  template <typename Tuple_>
  __host__ __device__ void operator()(Tuple_ t)
  {
    thrust::get<1>(t) = (thrust::get<0>(t) == i) ? (ValueType_)1.0 : (ValueType_)0.0;
  }
};
}  // namespace

/// Compute modularity
/** This function determines the modularity based on a graph and cluster assignments
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of clusters.
 *  @param parts (Input, device memory, n entries) Cluster assignments.
 *  @param modularity On exit, modularity
 */
template <typename vertex_t, typename edge_t, typename weight_t>
NVGRAPH_ERROR analyzeModularity(
  cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  vertex_t nClusters,
  const vertex_t *__restrict__ parts,
  weight_t &modularity)
{
  cudaStream_t stream = 0;
  edge_t i;
  edge_t n = graph.number_of_vertices;
  weight_t partModularity, partSize;

  // Device memory
  Vector<weight_t> part_i(n, stream);
  Vector<weight_t> Bx(n, stream);

  // Initialize cuBLAS
  Cublas::set_pointer_mode_host();

  // Initialize Modularity
  CsrMatrix<vertex_t, weight_t> A(false,
                                  false,
                                  graph.number_of_vertices,
                                  graph.number_of_vertices,
                                  graph.number_of_edges,
                                  0,
                                  graph.edge_data,
                                  graph.offsets,
                                  graph.indices);
  ModularityMatrix<vertex_t, weight_t> B(A, graph.number_of_edges);

  // Initialize output
  modularity = 0;

  // Iterate through partitions
  for (i = 0; i < nClusters; ++i) {
    // Construct indicator vector for ith partition
    thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(parts),
                                                   thrust::device_pointer_cast(part_i.raw()))),
      thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(parts + n),
                                                   thrust::device_pointer_cast(part_i.raw() + n))),
      equal_to_i_op<vertex_t, weight_t>(i));
    cudaCheckError();

    // Compute size of ith partition
    Cublas::dot(n, part_i.raw(), 1, part_i.raw(), 1, &partSize);
    partSize = round(partSize);
    if (partSize < 0.5) {
      WARNING("empty partition");
      continue;
    }

    // Compute modularity
    B.mv(1, part_i.raw(), 0, Bx.raw());
    Cublas::dot(n, Bx.raw(), 1, part_i.raw(), 1, &partModularity);

    // Record results
    modularity += partModularity;
    // std::cout<< "partModularity " <<partModularity<< std::endl;
  }
  // modularity = modularity/nClusters;
  // devide by nnz
  modularity = modularity / B.getEdgeSum();
  // Clean up and return

  return NVGRAPH_OK;
}

// =========================================================
// Explicit instantiation
// =========================================================
template NVGRAPH_ERROR modularity_maximization<int, int, float>(
  cugraph::experimental::GraphCSRView<int, int, float> const &graph,
  int nClusters,
  int nEigVecs,
  int maxIter_lanczos,
  int restartIter_lanczos,
  float tol_lanczos,
  int maxIter_kmeans,
  float tol_kmeans,
  int *__restrict__ parts,
  float *eigVals,
  float *eigVecs,
  int &iters_lanczos,
  int &iters_kmeans);
template NVGRAPH_ERROR modularity_maximization<int, int, double>(
  cugraph::experimental::GraphCSRView<int, int, double> const &graph,
  int nClusters,
  int nEigVecs,
  int maxIter_lanczos,
  int restartIter_lanczos,
  double tol_lanczos,
  int maxIter_kmeans,
  double tol_kmeans,
  int *__restrict__ parts,
  double *eigVals,
  double *eigVecs,
  int &iters_lanczos,
  int &iters_kmeans);
template NVGRAPH_ERROR analyzeModularity<int, int, float>(
  cugraph::experimental::GraphCSRView<int, int, float> const &graph,
  int nClusters,
  const int *__restrict__ parts,
  float &modularity);
template NVGRAPH_ERROR analyzeModularity<int, int, double>(
  cugraph::experimental::GraphCSRView<int, int, double> const &graph,
  int nClusters,
  const int *__restrict__ parts,
  double &modularity);

}  // namespace nvgraph
//#endif //NVGRAPH_PARTITION
