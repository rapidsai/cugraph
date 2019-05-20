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
//#ifdef DEBUG

#include "include/kmeans.hxx"

#include <stdio.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/gather.h>

#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/atomics.hxx"
#include "include/sm_utils.h"
#include "include/debug_macros.h"

using namespace nvgraph;

// =========================================================
// Useful macros
// =========================================================

#define BLOCK_SIZE 1024
#define WARP_SIZE  32
#define BSIZE_DIV_WSIZE (BLOCK_SIZE/WARP_SIZE)

// Get index of matrix entry
#define IDX(i,j,lda) ((i)+(j)*(lda))

namespace {

  // =========================================================
  // CUDA kernels
  // =========================================================

  /// Compute distances between observation vectors and centroids
  /** Block dimensions should be (warpSize, 1,
   *  blockSize/warpSize). Ideally, the grid is large enough so there
   *  are d threads in the x-direction, k threads in the y-direction,
   *  and n threads in the z-direction.
   *
   *  @param n Number of observation vectors.
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param obs (Input, d*n entries) Observation matrix. Matrix is
   *    stored column-major and each column is an observation
   *    vector. Matrix dimensions are d x n.
   *  @param centroids (Input, d*k entries) Centroid matrix. Matrix is
   *    stored column-major and each column is a centroid. Matrix
   *    dimensions are d x k.
   *  @param dists (Output, n*k entries) Distance matrix. Matrix is
   *    stored column-major and the (i,j)-entry is the square of the
   *    Euclidean distance between the ith observation vector and jth
   *    centroid. Matrix dimensions are n x k. Entries must be
   *    initialized to zero.
   */
  template <typename IndexType_, typename ValueType_>
  static __global__
  void computeDistances(IndexType_ n, IndexType_ d, IndexType_ k,
      const ValueType_ * __restrict__ obs,
      const ValueType_ * __restrict__ centroids,
      ValueType_ * __restrict__ dists) {

    // Loop index
    IndexType_ i;

    // Block indices
    IndexType_ bidx;
    // Global indices
    IndexType_ gidx, gidy, gidz;

    // Private memory
    ValueType_ centroid_private, dist_private;

    // Global x-index indicates index of vector entry
    bidx = blockIdx.x;
    while(bidx*blockDim.x < d) {
      gidx = threadIdx.x + bidx*blockDim.x;

      // Global y-index indicates centroid
      gidy = threadIdx.y + blockIdx.y*blockDim.y;
      while(gidy < k) {

        // Load centroid coordinate from global memory
        centroid_private
          = (gidx < d) ? centroids[IDX(gidx,gidy,d)] : 0;

        // Global z-index indicates observation vector
        gidz = threadIdx.z + blockIdx.z*blockDim.z;
        while(gidz < n) {

          // Load observation vector coordinate from global memory
          dist_private
            = (gidx < d) ? obs[IDX(gidx,gidz,d)] : 0;

          // Compute contribution of current entry to distance
          dist_private = centroid_private - dist_private;
          dist_private = dist_private*dist_private;

          // Perform reduction on warp
          for(i=WARP_SIZE/2; i>0; i/=2)
            dist_private += utils::shfl_down(dist_private, i, 2*i);
        
          // Write result to global memory
          if(threadIdx.x == 0)
            atomicFPAdd(dists+IDX(gidz,gidy,n), dist_private);

          // Move to another observation vector
          gidz += blockDim.z*gridDim.z;
        }

        // Move to another centroid
        gidy += blockDim.y*gridDim.y;
      }
      
      // Move to another vector entry
      bidx += gridDim.x;
    }
  
  }

  /// Find closest centroid to observation vectors
  /** Block and grid dimensions should be 1-dimensional. Ideally the
   *  grid is large enough so there are n threads.
   *
   *  @param n Number of observation vectors.
   *  @param k Number of clusters.
   *  @param centroids (Input, d*k entries) Centroid matrix. Matrix is
   *    stored column-major and each column is a centroid. Matrix
   *    dimensions are d x k.
   *  @param dists (Input/output, n*k entries) Distance matrix. Matrix
   *    is stored column-major and the (i,j)-entry is the square of
   *    the Euclidean distance between the ith observation vector and
   *    jth centroid. Matrix dimensions are n x k. On exit, the first
   *    n entries give the square of the Euclidean distance between
   *    observation vectors and closest centroids.
   *  @param codes (Output, n entries) Cluster assignments.
   *  @param clusterSizes (Output, k entries) Number of points in each
   *    cluster. Entries must be initialized to zero.
   */
  template <typename IndexType_, typename ValueType_>
  static __global__
  void minDistances(IndexType_ n, IndexType_ k,
        ValueType_ * __restrict__ dists,
        IndexType_ * __restrict__ codes,
        IndexType_ * __restrict__ clusterSizes) {

    // Loop index
    IndexType_ i, j;

    // Current matrix entry
    ValueType_ dist_curr;

    // Smallest entry in row
    ValueType_ dist_min;
    IndexType_ code_min;

    // Each row in observation matrix is processed by a thread
    i = threadIdx.x + blockIdx.x*blockDim.x;
    while(i<n) {

      // Find minimum entry in row
      code_min = 0;
      dist_min = dists[IDX(i,0,n)];
      for(j=1; j<k; ++j) {
        dist_curr = dists[IDX(i,j,n)];
        code_min = (dist_curr<dist_min) ? j : code_min;
        dist_min = (dist_curr<dist_min) ? dist_curr : dist_min;
      }

      // Transfer result to global memory
      dists[i] = dist_min;
      codes[i] = code_min;

      // Increment cluster sizes
      atomicAdd(clusterSizes+code_min, 1);
    
      // Move to another row
      i += blockDim.x*gridDim.x;

    }

  }

  /// Check if newly computed distances are smaller than old distances
  /** Block and grid dimensions should be 1-dimensional. Ideally the
   *  grid is large enough so there are n threads.
   *
   *  @param n Number of observation vectors.
   *  @param dists_old (Input/output, n entries) Distances between
   *    observation vectors and closest centroids. On exit, entries
   *    are replaced by entries in 'dists_new' if the corresponding
   *    observation vectors are closest to the new centroid.
   *  @param dists_new (Input, n entries) Distance between observation
   *    vectors and new centroid.
   *  @param codes_old (Input/output, n entries) Cluster
   *    assignments. On exit, entries are replaced with 'code_new' if
   *    the corresponding observation vectors are closest to the new
   *    centroid.
   *  @param code_new Index associated with new centroid.
   */
  template <typename IndexType_, typename ValueType_>
  static __global__
  void minDistances2(IndexType_ n,
         ValueType_ * __restrict__ dists_old,
         const ValueType_ * __restrict__ dists_new,
         IndexType_ * __restrict__ codes_old,
         IndexType_ code_new) {

    // Loop index
    IndexType_ i;

    // Distances
    ValueType_ dist_old_private;
    ValueType_ dist_new_private;

    // Each row is processed by a thread
    i = threadIdx.x + blockIdx.x*blockDim.x;
    while(i<n) {

      // Get old and new distances
      dist_old_private = dists_old[i];
      dist_new_private = dists_new[i];

      // Update if new distance is smaller than old distance
      if(dist_new_private < dist_old_private) {
        dists_old[i] = dist_new_private;
        codes_old[i] = code_new;
      }
    
      // Move to another row
      i += blockDim.x*gridDim.x;
    }

  }

  /// Compute size of k-means clusters
  /** Block and grid dimensions should be 1-dimensional. Ideally the
   *  grid is large enough so there are n threads.
   *
   *  @param n Number of observation vectors.
   *  @param k Number of clusters.
   *  @param codes (Input, n entries) Cluster assignments.
   *  @param clusterSizes (Output, k entries) Number of points in each
   *    cluster. Entries must be initialized to zero.
   */
  template <typename IndexType_> static __global__
  void computeClusterSizes(IndexType_ n, IndexType_ k,
         const IndexType_ * __restrict__ codes,
         IndexType_ * __restrict__ clusterSizes) {
    IndexType_ i = threadIdx.x + blockIdx.x*blockDim.x;
    while(i<n) {
      atomicAdd(clusterSizes+codes[i], 1);
      i += blockDim.x*gridDim.x;
    }
  }

  /// Divide rows of centroid matrix by cluster sizes
  /** Divides the ith column of the sum matrix by the size of the ith
   *  cluster. If the sum matrix has been initialized so that the ith
   *  row is the sum of all observation vectors in the ith cluster,
   *  this kernel produces cluster centroids. The grid and block
   *  dimensions should be 2-dimensional. Ideally the grid is large
   *  enough so there are d threads in the x-direction and k threads
   *  in the y-direction.
   *
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param clusterSizes (Input, k entries) Number of points in each
   *    cluster.
   *  @param centroids (Input/output, d*k entries) Sum matrix. Matrix
   *    is stored column-major and matrix dimensions are d x k. The
   *    ith column is the sum of all observation vectors in the ith
   *    cluster. On exit, the matrix is the centroid matrix (each
   *    column is the mean position of a cluster).
   */
  template <typename IndexType_, typename ValueType_>
  static __global__
  void divideCentroids(IndexType_ d, IndexType_ k,
           const IndexType_ * __restrict__ clusterSizes,
           ValueType_ * __restrict__ centroids) {


    // Global indices
    IndexType_ gidx, gidy;

    // Current cluster size
    IndexType_ clusterSize_private;

    // Observation vector is determined by global y-index
    gidy = threadIdx.y + blockIdx.y*blockDim.y;
    while(gidy < k) {
    
      // Get cluster size from global memory
      clusterSize_private = clusterSizes[gidy];

      // Add vector entries to centroid matrix
      //   Vector entris are determined by global x-index
      gidx = threadIdx.x + blockIdx.x*blockDim.x;
      while(gidx < d) {
        centroids[IDX(gidx,gidy,d)] /= clusterSize_private;
        gidx += blockDim.x*gridDim.x;
      }

      // Move to another centroid
      gidy += blockDim.y*gridDim.y;
    }

  }

  // =========================================================
  // Helper functions
  // =========================================================

  /// Randomly choose new centroids
  /** Centroid is randomly chosen with k-means++ algorithm.
   *
   *  @param n Number of observation vectors.
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param rand Random number drawn uniformly from [0,1).
   *  @param obs (Input, device memory, d*n entries) Observation
   *    matrix. Matrix is stored column-major and each column is an
   *    observation vector. Matrix dimensions are n x d.
   *  @param dists (Input, device memory, 2*n entries) Workspace. The
   *    first n entries should be the distance between observation
   *    vectors and the closest centroid.
   *  @param centroid (Output, device memory, d entries) Centroid
   *    coordinates.
   *  @return Zero if successful. Otherwise non-zero.
   */
  template <typename IndexType_, typename ValueType_> static
  int chooseNewCentroid(IndexType_ n, IndexType_ d, IndexType_ k,
      ValueType_ rand,
      const ValueType_ * __restrict__ obs,
      ValueType_ * __restrict__ dists,
      ValueType_ * __restrict__ centroid) {
  
    using namespace thrust;

    // Cumulative sum of distances
    ValueType_ * distsCumSum = dists + n;
    // Residual sum of squares
    ValueType_ distsSum;
    // Observation vector that is chosen as new centroid
    IndexType_ obsIndex;

    // Compute cumulative sum of distances
    inclusive_scan(device_pointer_cast(dists),
       device_pointer_cast(dists+n),
       device_pointer_cast(distsCumSum));
    cudaCheckError();
    CHECK_CUDA(cudaMemcpy(&distsSum, distsCumSum+n-1,
        sizeof(ValueType_),
        cudaMemcpyDeviceToHost));
  
    // Randomly choose observation vector
    //   Probabilities are proportional to square of distance to closest
    //   centroid (see k-means++ algorithm)
    obsIndex = (lower_bound(device_pointer_cast(distsCumSum),
          device_pointer_cast(distsCumSum+n),
          distsSum*rand)
    - device_pointer_cast(distsCumSum));
    cudaCheckError();
    obsIndex = max(obsIndex, 0);
    obsIndex = min(obsIndex, n-1);

    // Record new centroid position
    CHECK_CUDA(cudaMemcpyAsync(centroid, obs+IDX(0,obsIndex,d),
             d*sizeof(ValueType_),
             cudaMemcpyDeviceToDevice));

    return 0;

  }

  /// Choose initial cluster centroids for k-means algorithm
  /** Centroids are randomly chosen with k-means++ algorithm
   *
   *  @param n Number of observation vectors.
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param obs (Input, device memory, d*n entries) Observation
   *    matrix. Matrix is stored column-major and each column is an
   *    observation vector. Matrix dimensions are d x n.
   *  @param centroids (Output, device memory, d*k entries) Centroid
   *    matrix. Matrix is stored column-major and each column is a
   *    centroid. Matrix dimensions are d x k.
   *  @param codes (Output, device memory, n entries) Cluster
   *    assignments.
   *  @param clusterSizes (Output, device memory, k entries) Number of
   *    points in each cluster.
   *  @param dists (Output, device memory, 2*n entries) Workspace. On
   *    exit, the first n entries give the square of the Euclidean
   *    distance between observation vectors and the closest centroid.
   *  @return Zero if successful. Otherwise non-zero.
   */
  template <typename IndexType_, typename ValueType_> static
  int initializeCentroids(IndexType_ n, IndexType_ d, IndexType_ k,
        const ValueType_ * __restrict__ obs,
        ValueType_ * __restrict__ centroids,
        IndexType_ * __restrict__ codes,
        IndexType_ * __restrict__ clusterSizes,
        ValueType_ * __restrict__ dists) {

    // -------------------------------------------------------
    // Variable declarations
    // -------------------------------------------------------

    // Loop index
    IndexType_ i;

    // CUDA grid dimensions
    dim3 blockDim_warp, gridDim_warp, gridDim_block;

    // Random number generator
    thrust::default_random_engine rng(123456);
    thrust::uniform_real_distribution<ValueType_> uniformDist(0,1);
  
    // -------------------------------------------------------
    // Implementation
    // -------------------------------------------------------

    // Initialize grid dimensions
    blockDim_warp.x = WARP_SIZE;
    blockDim_warp.y = 1;
    blockDim_warp.z = BSIZE_DIV_WSIZE;
    gridDim_warp.x = min((d+WARP_SIZE-1)/WARP_SIZE, 65535);
    gridDim_warp.y = 1;
    gridDim_warp.z 
      = min((n+BSIZE_DIV_WSIZE-1)/BSIZE_DIV_WSIZE, 65535);
    gridDim_block.x = min((n+BLOCK_SIZE-1)/BLOCK_SIZE, 65535);
    gridDim_block.y = 1;
    gridDim_block.z = 1;

    // Assign observation vectors to code 0
    CHECK_CUDA(cudaMemsetAsync(codes, 0, n*sizeof(IndexType_)));

    // Choose first centroid
    thrust::fill(thrust::device_pointer_cast(dists),
     thrust::device_pointer_cast(dists+n), 1);
    cudaCheckError();
    if(chooseNewCentroid(n, d, k, uniformDist(rng), obs, dists, centroids))
      WARNING("error in k-means++ (could not pick centroid)");

    // Compute distances from first centroid
    CHECK_CUDA(cudaMemsetAsync(dists, 0, n*sizeof(ValueType_)));
    computeDistances <<< gridDim_warp, blockDim_warp >>>
      (n, d, 1, obs, centroids, dists);
    cudaCheckError()

    // Choose remaining centroids
    for(i=1; i<k; ++i) {
    
      // Choose ith centroid
      if(chooseNewCentroid(n, d, k, uniformDist(rng),obs, dists, centroids+IDX(0,i,d))) 
        WARNING("error in k-means++ (could not pick centroid)");

      // Compute distances from ith centroid
      CHECK_CUDA(cudaMemsetAsync(dists+n, 0, n*sizeof(ValueType_)));
      computeDistances <<< gridDim_warp, blockDim_warp >>>
        (n, d, 1, obs, centroids+IDX(0,i,d), dists+n);
      cudaCheckError();

      // Recompute minimum distances
      minDistances2 <<< gridDim_block, BLOCK_SIZE >>>
        (n, dists, dists+n, codes, i);
      cudaCheckError();

    }

    // Compute cluster sizes
    CHECK_CUDA(cudaMemsetAsync(clusterSizes, 0, k*sizeof(IndexType_)));
    computeClusterSizes <<< gridDim_block, BLOCK_SIZE >>>
      (n, k, codes, clusterSizes);
    cudaCheckError();

    return 0;

  }

  /// Find cluster centroids closest to observation vectors
  /** Distance is measured with Euclidean norm.
   *
   *  @param n Number of observation vectors.
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param obs (Input, device memory, d*n entries) Observation
   *    matrix. Matrix is stored column-major and each column is an
   *    observation vector. Matrix dimensions are d x n.
   *  @param centroids (Input, device memory, d*k entries) Centroid
   *    matrix. Matrix is stored column-major and each column is a
   *    centroid. Matrix dimensions are d x k.
   *  @param dists (Output, device memory, n*k entries) Workspace. On
   *    exit, the first n entries give the square of the Euclidean
   *    distance between observation vectors and the closest centroid.
   *  @param codes (Output, device memory, n entries) Cluster
   *    assignments.
   *  @param clusterSizes (Output, device memory, k entries) Number of
   *    points in each cluster.
   *  @param residual_host (Output, host memory, 1 entry) Residual sum
   *    of squares of assignment.
   *  @return Zero if successful. Otherwise non-zero.
   */
  template <typename IndexType_, typename ValueType_> static
  int assignCentroids(IndexType_ n, IndexType_ d, IndexType_ k,
          const ValueType_ * __restrict__ obs,
          const ValueType_ * __restrict__ centroids,
          ValueType_ * __restrict__ dists,
          IndexType_ * __restrict__ codes,
          IndexType_ * __restrict__ clusterSizes,
          ValueType_ * residual_host) {

    // CUDA grid dimensions
    dim3 blockDim, gridDim;

    // Compute distance between centroids and observation vectors
    CHECK_CUDA(cudaMemsetAsync(dists, 0, n*k*sizeof(ValueType_)));
    blockDim.x = WARP_SIZE;
    blockDim.y = 1;
    blockDim.z = BLOCK_SIZE/WARP_SIZE;
    gridDim.x  = min((d+WARP_SIZE-1)/WARP_SIZE, 65535);
    gridDim.y  = min(k, 65535);
    gridDim.z  = min((n+BSIZE_DIV_WSIZE-1)/BSIZE_DIV_WSIZE, 65535);
    computeDistances <<< gridDim, blockDim >>> (n, d, k,
            obs, centroids,
            dists);
    cudaCheckError();

    // Find centroid closest to each observation vector
    CHECK_CUDA(cudaMemsetAsync(clusterSizes,0,k*sizeof(IndexType_)));
    blockDim.x = BLOCK_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x  = min((n+BLOCK_SIZE-1)/BLOCK_SIZE, 65535);
    gridDim.y  = 1;
    gridDim.z  = 1;
    minDistances <<< gridDim, blockDim >>> (n, k, dists, codes,
              clusterSizes);
    cudaCheckError();

    // Compute residual sum of squares
    *residual_host
      = thrust::reduce(thrust::device_pointer_cast(dists),
           thrust::device_pointer_cast(dists+n));

    return 0;

  }

  /// Update cluster centroids for k-means algorithm
  /** All clusters are assumed to be non-empty.
   *
   *  @param n Number of observation vectors.
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param obs (Input, device memory, d*n entries) Observation
   *    matrix. Matrix is stored column-major and each column is an
   *    observation vector. Matrix dimensions are d x n.
   *  @param codes (Input, device memory, n entries) Cluster
   *    assignments.
   *  @param clusterSizes (Input, device memory, k entries) Number of
   *    points in each cluster.
   *  @param centroids (Output, device memory, d*k entries) Centroid
   *    matrix. Matrix is stored column-major and each column is a
   *    centroid. Matrix dimensions are d x k.
   *  @param work (Output, device memory, n*d entries) Workspace.
   *  @param work_int (Output, device memory, 2*d*n entries)
   *    Workspace.
   *  @return Zero if successful. Otherwise non-zero.
   */
  template <typename IndexType_, typename ValueType_> static
  int updateCentroids(IndexType_ n, IndexType_ d, IndexType_ k,
          const ValueType_ * __restrict__ obs,
          const IndexType_ * __restrict__ codes,
          const IndexType_ * __restrict__ clusterSizes,
          ValueType_ * __restrict__ centroids,
          ValueType_ * __restrict__ work,
          IndexType_ * __restrict__ work_int) {

    using namespace thrust;

    // -------------------------------------------------------
    // Variable declarations
    // -------------------------------------------------------

    // Useful constants
    const ValueType_ one  = 1;
    const ValueType_ zero = 0;

    // CUDA grid dimensions
    dim3 blockDim, gridDim;

    // Device memory
    device_ptr<ValueType_> obs_copy(work);
    device_ptr<IndexType_> codes_copy(work_int);
    device_ptr<IndexType_> rows(work_int+d*n);

    // Take transpose of observation matrix
    Cublas::geam(true, false, n, d,
     &one, obs, d, &zero, (ValueType_*) NULL, n,
     raw_pointer_cast(obs_copy), n);

    // Cluster assigned to each observation matrix entry
    sequence(rows, rows+d*n);
    cudaCheckError();
    transform(rows, rows+d*n, make_constant_iterator<IndexType_>(n),
        rows, modulus<IndexType_>());
    cudaCheckError();
    gather(rows, rows+d*n, device_pointer_cast(codes), codes_copy);
    cudaCheckError();

    // Row associated with each observation matrix entry
    sequence(rows, rows+d*n);
    cudaCheckError();
    transform(rows, rows+d*n, make_constant_iterator<IndexType_>(n),
        rows, divides<IndexType_>());
    cudaCheckError();

    // Sort and reduce to add observation vectors in same cluster
    stable_sort_by_key(codes_copy, codes_copy+d*n,
           make_zip_iterator(make_tuple(obs_copy, rows)));
    cudaCheckError();
    reduce_by_key(rows, rows+d*n, obs_copy,
      codes_copy, // Output to codes_copy is ignored
      device_pointer_cast(centroids));
    cudaCheckError();

    // Divide sums by cluster size to get centroid matrix
    blockDim.x = WARP_SIZE;
    blockDim.y = BLOCK_SIZE/WARP_SIZE;
    blockDim.z = 1;
    gridDim.x  = min((d+WARP_SIZE-1)/WARP_SIZE, 65535);
    gridDim.y  = min((k+BSIZE_DIV_WSIZE-1)/BSIZE_DIV_WSIZE, 65535);
    gridDim.z  = 1;
    divideCentroids <<< gridDim, blockDim >>> (d, k, clusterSizes,
                 centroids);
    cudaCheckError();

    return 0;

  }

}

namespace nvgraph {

  // =========================================================
  // k-means algorithm
  // =========================================================

  /// Find clusters with k-means algorithm
  /** Initial centroids are chosen with k-means++ algorithm. Empty
   *  clusters are reinitialized by choosing new centroids with
   *  k-means++ algorithm.
   *
   *  @param n Number of observation vectors.
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param tol Tolerance for convergence. k-means stops when the
   *    change in residual divided by n is less than tol.
   *  @param maxiter Maximum number of k-means iterations.
   *  @param obs (Input, device memory, d*n entries) Observation
   *    matrix. Matrix is stored column-major and each column is an
   *    observation vector. Matrix dimensions are d x n.
   *  @param codes (Output, device memory, n entries) Cluster
   *    assignments.
   *  @param clusterSizes (Output, device memory, k entries) Number of
   *    points in each cluster.
   *  @param centroids (Output, device memory, d*k entries) Centroid
   *    matrix. Matrix is stored column-major and each column is a
   *    centroid. Matrix dimensions are d x k.
   *  @param work (Output, device memory, n*max(k,d) entries)
   *    Workspace.
   *  @param work_int (Output, device memory, 2*d*n entries)
   *    Workspace.
   *  @param residual_host (Output, host memory, 1 entry) Residual sum
   *    of squares (sum of squares of distances between observation
   *    vectors and centroids).
   *  @param iters_host (Output, host memory, 1 entry) Number of
   *    k-means iterations.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR kmeans(IndexType_ n, IndexType_ d, IndexType_ k,
        ValueType_ tol, IndexType_ maxiter,
        const ValueType_ * __restrict__ obs,
        IndexType_ * __restrict__ codes,
        IndexType_ * __restrict__ clusterSizes,
        ValueType_ * __restrict__ centroids,
        ValueType_ * __restrict__ work,
        IndexType_ * __restrict__ work_int,
        ValueType_ * residual_host,
        IndexType_ * iters_host) {
  
    // -------------------------------------------------------
    // Variable declarations
    // -------------------------------------------------------

    // Current iteration
    IndexType_ iter;

    // Residual sum of squares at previous iteration
    ValueType_ residualPrev = 0;

    // Random number generator
    thrust::default_random_engine rng(123456);
    thrust::uniform_real_distribution<ValueType_> uniformDist(0,1);

    // -------------------------------------------------------
    // Initialization
    // -------------------------------------------------------

    // Check that parameters are valid
    if(n < 1) {
      WARNING("invalid parameter (n<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(d < 1) {
      WARNING("invalid parameter (d<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(k < 1) {
      WARNING("invalid parameter (k<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol < 0) {
      WARNING("invalid parameter (tol<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxiter < 0) {
      WARNING("invalid parameter (maxiter<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // Trivial cases
    if(k == 1) {
      CHECK_CUDA(cudaMemsetAsync(codes, 0, n*sizeof(IndexType_)));
      CHECK_CUDA(cudaMemcpyAsync(clusterSizes, &n, sizeof(IndexType_),
         cudaMemcpyHostToDevice));
      if(updateCentroids(n, d, k, obs, codes,
           clusterSizes, centroids,
           work, work_int)) 
        WARNING("could not compute k-means centroids");
      dim3 blockDim, gridDim;
      blockDim.x = WARP_SIZE;
      blockDim.y = 1;
      blockDim.z = BLOCK_SIZE/WARP_SIZE;
      gridDim.x = min((d+WARP_SIZE-1)/WARP_SIZE, 65535);
      gridDim.y = 1;
      gridDim.z = min((n+BLOCK_SIZE/WARP_SIZE-1)/(BLOCK_SIZE/WARP_SIZE), 65535);
      CHECK_CUDA(cudaMemsetAsync(work, 0, n*k*sizeof(ValueType_)));
      computeDistances <<< gridDim, blockDim >>> (n, d, 1,
              obs,
              centroids,
              work);
      cudaCheckError();
      *residual_host = thrust::reduce(thrust::device_pointer_cast(work), 
        thrust::device_pointer_cast(work+n));
      cudaCheckError();
      return NVGRAPH_OK;
    }
    if(n <= k) {
      thrust::sequence(thrust::device_pointer_cast(codes),
           thrust::device_pointer_cast(codes+n));
      cudaCheckError();
      thrust::fill_n(thrust::device_pointer_cast(clusterSizes), n, 1);
      cudaCheckError();

      if(n < k)
        CHECK_CUDA(cudaMemsetAsync(clusterSizes+n, 0, (k-n)*sizeof(IndexType_)));
      CHECK_CUDA(cudaMemcpyAsync(centroids, obs, d*n*sizeof(ValueType_),
        cudaMemcpyDeviceToDevice));
      *residual_host = 0;
      return NVGRAPH_OK;
    }

    // Initialize cuBLAS
    Cublas::set_pointer_mode_host();

    // -------------------------------------------------------
    // k-means++ algorithm
    // -------------------------------------------------------

    // Choose initial cluster centroids
    if(initializeCentroids(n, d, k, obs, centroids, codes,
             clusterSizes, work)) 
      WARNING("could not initialize k-means centroids");

    // Apply k-means iteration until convergence
    for(iter=0; iter<maxiter; ++iter) {

      // Update cluster centroids
      if(updateCentroids(n, d, k, obs, codes, 
           clusterSizes, centroids,
           work, work_int)) WARNING("could not update k-means centroids");

      // Determine centroid closest to each observation
      residualPrev = *residual_host;
      if(assignCentroids(n, d, k, obs, centroids, work,
           codes, clusterSizes, residual_host))
       WARNING("could not assign observation vectors to k-means clusters");

      // Reinitialize empty clusters with new centroids
      IndexType_ emptyCentroid = (thrust::find(thrust::device_pointer_cast(clusterSizes), 
        thrust::device_pointer_cast(clusterSizes+k), 0) - thrust::device_pointer_cast(clusterSizes));
      while(emptyCentroid < k) {
        if(chooseNewCentroid(n, d, k, uniformDist(rng), obs, work, centroids+IDX(0,emptyCentroid,d)))
          WARNING("could not replace empty centroid");
        if(assignCentroids(n, d, k, obs, centroids, work, codes, clusterSizes, residual_host))
          WARNING("could not assign observation vectors to k-means clusters");
        emptyCentroid = (thrust::find(thrust::device_pointer_cast(clusterSizes), 
            thrust::device_pointer_cast(clusterSizes+k), 0) - thrust::device_pointer_cast(clusterSizes));
        cudaCheckError();
      }

      // Check for convergence
      if(fabs(residualPrev-(*residual_host))/n < tol) {
        ++iter;
        break;
      }

    }

    // Warning if k-means has failed to converge
    if(fabs(residualPrev-(*residual_host))/n >= tol)
      WARNING("k-means failed to converge");

    *iters_host = iter;
    return NVGRAPH_OK;

  }

  /// Find clusters with k-means algorithm
  /** Initial centroids are chosen with k-means++ algorithm. Empty
   *  clusters are reinitialized by choosing new centroids with
   *  k-means++ algorithm.
   *
   *  CNMEM must be initialized before calling this function.
   *
   *  @param n Number of observation vectors.
   *  @param d Dimension of observation vectors.
   *  @param k Number of clusters.
   *  @param tol Tolerance for convergence. k-means stops when the
   *    change in residual divided by n is less than tol.
   *  @param maxiter Maximum number of k-means iterations.
   *  @param obs (Input, device memory, d*n entries) Observation
   *    matrix. Matrix is stored column-major and each column is an
   *    observation vector. Matrix dimensions are d x n.
   *  @param codes (Output, device memory, n entries) Cluster
   *    assignments.
   *  @param residual On exit, residual sum of squares (sum of squares
   *    of distances between observation vectors and centroids).
   *  @param On exit, number of k-means iterations.
   *  @return NVGRAPH error flag
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR kmeans(IndexType_ n, IndexType_ d, IndexType_ k,
        ValueType_ tol, IndexType_ maxiter,
        const ValueType_ * __restrict__ obs,
        IndexType_ * __restrict__ codes,
        ValueType_ & residual,
        IndexType_ & iters) {

    // Check that parameters are valid
    if(n < 1) {
      WARNING("invalid parameter (n<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(d < 1) {
      WARNING("invalid parameter (d<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(k < 1) {
      WARNING("invalid parameter (k<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol < 0) {
      WARNING("invalid parameter (tol<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxiter < 0) {
      WARNING("invalid parameter (maxiter<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // Allocate memory
    // TODO: handle non-zero CUDA streams
    cudaStream_t stream = 0;
    Vector<IndexType_> clusterSizes(k, stream);
    Vector<ValueType_> centroids(d*k, stream);
    Vector<ValueType_> work(n*max(k,d), stream);
    Vector<IndexType_> work_int(2*d*n, stream);
    
    // Perform k-means
    return kmeans<IndexType_,ValueType_>(n, d, k, tol, maxiter,
           obs, codes, 
           clusterSizes.raw(),
           centroids.raw(),
           work.raw(), work_int.raw(),
           &residual, &iters);
    
  }


  // =========================================================
  // Explicit instantiations
  // =========================================================

  template
  NVGRAPH_ERROR kmeans<int, float>(int n, int d, int k,
        float tol, int maxiter,
        const float * __restrict__ obs,
        int * __restrict__ codes,
        float & residual,
        int & iters);
  template
  NVGRAPH_ERROR kmeans<int, double>(int n, int d, int k,
         double tol, int maxiter,
         const double * __restrict__ obs,
         int * __restrict__ codes,
         double & residual,
         int & iters);
}
//#endif //NVGRAPH_PARTITION
//#endif //debug

