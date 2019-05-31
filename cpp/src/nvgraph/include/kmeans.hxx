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

#include "nvgraph_error.hxx"

namespace nvgraph {

  /// Find clusters with k-means algorithm
  /** Initial centroids are chosen with k-means++ algorithm. Empty
   *  clusters are reinitialized by choosing new centroids with
   *  k-means++ algorithm.
   *
   *  CNMEM must be initialized before calling this function.
   *
   *  @param cublasHandle_t cuBLAS handle.
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
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR kmeans(IndexType_ n, IndexType_ d, IndexType_ k,
		    ValueType_ tol, IndexType_ maxiter,
		    const ValueType_ * __restrict__ obs,
		    IndexType_ * __restrict__ codes,
		    ValueType_ & residual,
		    IndexType_ & iters);

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
		    IndexType_ * iters_host);

}

