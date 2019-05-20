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
#include "valued_csr_graph.hxx"
#include "matrix.hxx"


namespace nvgraph {
  #define SPECTRAL_USE_COLORING true
  
  #define SPECTRAL_USE_LOBPCG true 
  #define SPECTRAL_USE_PRECONDITIONING true
  #define SPECTRAL_USE_SCALING_OF_EIGVECS false
  
  #define SPECTRAL_USE_MAGMA false
  #define SPECTRAL_USE_THROTTLE true
  #define SPECTRAL_USE_NORMALIZED_LAPLACIAN true
  #define SPECTRAL_USE_R_ORTHOGONALIZATION false

  /// Spectral graph partition
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
		       IndexType_ & iters_kmeans);

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
           IndexType_ & iters_kmeans);


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
			      ValueType_ & edgeCut, ValueType_ & cost);

}

