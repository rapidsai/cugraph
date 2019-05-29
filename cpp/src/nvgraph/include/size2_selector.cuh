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

#include <thrust/device_vector.h>
#include <thrust/count.h> //count
#include <thrust/sort.h> //sort
#include <thrust/binary_search.h> //lower_bound
#include <thrust/unique.h> //unique
#include <cusparse.h>
#include "async_event.cuh"
#include "graph_utils.cuh"
#include "common_selector.cuh"
#include "valued_csr_graph.cuh"


// This should be enabled
#define EXPERIMENTAL_ITERATIVE_MATCHING

using namespace nvlouvain;

namespace nvlouvain{

typedef enum
{
   USER_PROVIDED = 0, // using edge values as is
   SCALED_BY_ROW_SUM   = 1,  // 0.5*(A_ij+A_ji)/max(d(i),d (j)), where d(i) is the sum of the row i
   SCALED_BY_DIAGONAL   = 2,  // 0.5*(A_ij+A_ji)/max(diag(i),diag(j)) 
}Matching_t;

typedef enum{
  NVGRAPH_OK = 0,
  NVGRAPH_ERR_BAD_PARAMETERS = 1,
}NVGRAPH_ERROR;



template <typename IndexType, typename ValueType>
class Size2Selector
{

  public:

    Size2Selector();

    Size2Selector(Matching_t similarity_metric,  int deterministic = 1, int max_iterations = 15 , ValueType numUnassigned_tol = 0.05 ,bool two_phase = false, bool merge_singletons = true, cudaStream_t stream = 0) 
       :m_similarity_metric(similarity_metric), m_deterministic(deterministic), m_max_iterations(max_iterations), m_numUnassigned_tol(numUnassigned_tol), m_two_phase(two_phase), m_merge_singletons(merge_singletons), m_stream(stream)
    {
        m_aggregation_edge_weight_component = 0;
        m_weight_formula = 0;
    }

//    NVGRAPH_ERROR setAggregates(const CsrGraph<IndexType, ValueType> &A, Vector<IndexType> &aggregates, int &num_aggregates);
    NVGRAPH_ERROR setAggregates(cusparseHandle_t, const IndexType n_vertex, const IndexType n_edges, IndexType* csr_ptr, IndexType* csr_ind, ValueType* csr_val, Vector<IndexType> &aggregates, int &num_aggregates);


  protected:
//    NVGRAPH_ERROR setAggregates_common_sqblocks(const CsrGraph<IndexType, ValueType> &A, Vector<IndexType> &aggregates, int &num_aggregates);
    NVGRAPH_ERROR setAggregates_common_sqblocks(cusparseHandle_t, const IndexType n_vertex, const IndexType n_edges, IndexType* csr_ptr, IndexType* csr_ind, ValueType* csr_val, Vector<IndexType> &aggregates, int &num_aggregates);
 
    Matching_t m_similarity_metric;
    int m_deterministic;
    int m_max_iterations;
    ValueType m_numUnassigned_tol;
    bool m_two_phase;
    bool m_merge_singletons;
    cudaStream_t m_stream;    
    int m_aggregation_edge_weight_component;
    int m_weight_formula;
};

}


template <typename IndexType>
void renumberAndCountAggregates(Vector<IndexType> &aggregates, const IndexType n, IndexType& num_aggregates)
{
  // renumber aggregates
  Vector<IndexType> scratch(n+1);
  scratch.fill(0);
  thrust::device_ptr<IndexType> aggregates_thrust_dev_ptr(aggregates.raw());
  thrust::device_ptr<IndexType> scratch_thrust_dev_ptr(scratch.raw());

  // set scratch[aggregates[i]] = 1
  thrust::fill(thrust::make_permutation_iterator(scratch_thrust_dev_ptr, aggregates_thrust_dev_ptr),
               thrust::make_permutation_iterator(scratch_thrust_dev_ptr, aggregates_thrust_dev_ptr + n), 1);
  //scratch.dump(0,scratch.get_size());

  // do prefix sum on scratch
  thrust::exclusive_scan(scratch_thrust_dev_ptr, scratch_thrust_dev_ptr + n + 1, scratch_thrust_dev_ptr);
 // scratch.dump(0,scratch.get_size());

  // aggregates[i] = scratch[aggregates[i]]
  thrust::copy(thrust::make_permutation_iterator(scratch_thrust_dev_ptr, aggregates_thrust_dev_ptr),
               thrust::make_permutation_iterator(scratch_thrust_dev_ptr, aggregates_thrust_dev_ptr + n),
               aggregates_thrust_dev_ptr);
  cudaCheckError();
  cudaMemcpy(&num_aggregates, &scratch.raw()[scratch.get_size()-1], sizeof(int), cudaMemcpyDefault); //num_aggregates = scratch.raw()[scratch.get_size()-1];
  cudaCheckError();

}

// ------------------
// Constructors
// ------------------

template <typename IndexType, typename ValueType>
Size2Selector<IndexType, ValueType>::Size2Selector()
{
  //Using default vaues from AmgX
  m_deterministic = 1;
  m_stream=0;
  m_max_iterations = 15;
  m_numUnassigned_tol = 0.05;
  m_two_phase =  0;
  m_aggregation_edge_weight_component= 0;
  m_merge_singletons = 1;
  m_weight_formula = 0;
  m_similarity_metric = SCALED_BY_ROW_SUM;
}

// ------------------
// Methods
// ------------------

// setAggregates for block_dia_csr_matrix_d format
template <typename IndexType, typename ValueType>
NVGRAPH_ERROR Size2Selector<IndexType, ValueType>::setAggregates_common_sqblocks(
cusparseHandle_t cusp_handle,
const IndexType n_vertex,
const IndexType n_edges, 
IndexType *csr_ptr,
IndexType *csr_ind,
ValueType *csr_val, 
Vector<IndexType> &aggregates, int &num_aggregates)
{
  const IndexType n = n_vertex;
  const IndexType nnz = n_edges;
  const IndexType *A_row_offsets_ptr = csr_ptr;
  const IndexType *A_column_indices_ptr = csr_ind;
  const ValueType *A_nonzero_values_ptr = csr_val;
  
  // compute row indices
  Vector<IndexType> row_indices(nnz);
  IndexType* row_indices_raw_ptr = row_indices.raw();
//  Cusparse::csr2coo( n, nnz, A_row_offsets_ptr, row_indices.raw()); // note : amgx uses cusp for that
  //cusparseHandle_t cusp_handle;
  //cusparseCreate(&cusp_handle);

  cusparseXcsr2coo(cusp_handle, A_row_offsets_ptr,
                 nnz, n, row_indices_raw_ptr, 
                 CUSPARSE_INDEX_BASE_ZERO);  

  const IndexType *A_row_indices_ptr = row_indices.raw();
  
  //All vectors should be initialized to -1.
  aggregates.fill(-1);
  Vector<IndexType> strongest_neighbour(n);
  strongest_neighbour.fill(-1);
  Vector<IndexType> strongest_neighbour_1phase(n);
  strongest_neighbour_1phase.fill(-1);
  Vector<float> edge_weights(nnz);
  edge_weights.fill(-1);
  float *edge_weights_ptr  = edge_weights.raw();
  float *rand_edge_weights_ptr = NULL;
  cudaCheckError();

  IndexType *strongest_neighbour_ptr = strongest_neighbour.raw();
  IndexType *strongest_neighbour_1phase_ptr = strongest_neighbour_1phase.raw();
  IndexType *aggregates_ptr = aggregates.raw();

  const int threads_per_block = 256;
  const int max_grid_size = 256;
  const int num_blocks = min( max_grid_size, (n-1)/threads_per_block+ 1 );
  const int num_blocks_V2 = min( max_grid_size, (nnz-1)/threads_per_block + 1);
  int bsize = 1; // AmgX legacy: we don't use block CSR matrices, this is just to specify that we run on regular matrices

  int numUnassigned = n;
  int numUnassigned_previous = numUnassigned;
  thrust::device_ptr<IndexType> aggregates_thrust_dev_ptr(aggregates_ptr);
  switch(m_similarity_metric)
  {
     case USER_PROVIDED : 
     {          
         //printf("user provided !!!!!!!!!!!!!!!! \n");
         //copy non wero values of A in edge_weights (float)
         convert_type<<<num_blocks_V2,threads_per_block,0,this->m_stream>>>(nnz, A_nonzero_values_ptr, edge_weights_ptr);
         cudaCheckError();
         //edge_weights.dump(0,nnz);
         break; 
     }
     case SCALED_BY_ROW_SUM : 
     {  /* comment out by Tin-Yin 
        // Compute the edge weights using .5*(A_ij+A_ji)/max(d(i),d(j)) where d(i) is the sum of outgoing edges of i

        Vector<ValueType> row_sum(n);
        const ValueType *A_row_sum_ptr = row_sum.raw(); 
        Vector<ValueType> ones(n);
        ones.fill(1.0);
        ValueType alpha = 1.0, beta =0.0;
        Cusparse::csrmv(false, false, n, n, nnz,&alpha,A_nonzero_values_ptr, A_row_offsets_ptr, A_column_indices_ptr, ones.raw(),&beta, row_sum.raw());
        cudaFuncSetCacheConfig(computeEdgeWeightsBlockDiaCsr_V2<IndexType,ValueType,float>,cudaFuncCachePreferL1);
        computeEdgeWeights_simple<<<num_blocks_V2,threads_per_block,0,this->m_stream>>>(A_row_offsets_ptr, A_row_indices_ptr, A_column_indices_ptr, A_row_sum_ptr, A_nonzero_values_ptr, nnz, edge_weights_ptr, rand_edge_weights_ptr, n, this->m_weight_formula);
        cudaCheckError();  
        break; 
*/
        
     }
     case SCALED_BY_DIAGONAL : 
     { 
       // Compute the edge weights using AmgX formula (works only if there is a diagonal entry for each row)
       Vector<IndexType> diag_idx(n);
       const IndexType *A_dia_idx_ptr = diag_idx.raw();

       computeDiagonalKernelCSR<<<num_blocks,threads_per_block,0,this->m_stream>>>(n, csr_ptr, csr_ind, diag_idx.raw());
       cudaCheckError();

       cudaFuncSetCacheConfig(computeEdgeWeightsBlockDiaCsr_V2<IndexType,ValueType,float>,cudaFuncCachePreferL1);
       computeEdgeWeightsBlockDiaCsr_V2<<<num_blocks_V2,threads_per_block,0,this->m_stream>>>(A_row_offsets_ptr, A_row_indices_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, nnz, edge_weights_ptr, rand_edge_weights_ptr, n, bsize,this->m_aggregation_edge_weight_component, this->m_weight_formula);
       cudaCheckError();  
       break; 
     }
     default: return NVGRAPH_ERR_BAD_PARAMETERS;
  }
  
#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
  // TODO (from amgx): allocate host pinned memory
  AsyncEvent *throttle_event = new AsyncEvent;
  throttle_event->create();
  std::vector<IndexType> h_unagg_vec(1);
  Vector<IndexType> d_unagg_vec(1);

  int *unaggregated = &h_unagg_vec[0];
  int *d_unaggregated = d_unagg_vec.raw();

#endif

  int icount, s = 1;
  {
    icount = 0;
    float *weights_ptr = edge_weights_ptr;
    
    do 
    {
      if( !this->m_two_phase ) {
      // 1-phase handshaking
        findStrongestNeighbourBlockDiaCsr_V2<<<num_blocks,threads_per_block,0,this->m_stream>>>(A_row_offsets_ptr, A_column_indices_ptr, weights_ptr, n, aggregates_ptr, strongest_neighbour_ptr, strongest_neighbour_ptr, bsize, 1, this->m_merge_singletons);
        cudaCheckError();

      } 
      else { 
        // 2-phase handshaking
        findStrongestNeighbourBlockDiaCsr_V2<<<num_blocks,threads_per_block,0,this->m_stream>>>(A_row_offsets_ptr, A_column_indices_ptr, weights_ptr, n, aggregates_ptr, strongest_neighbour_1phase_ptr, strongest_neighbour_ptr, bsize, 1, this->m_merge_singletons);
        cudaCheckError();
        
 
        // 2nd phase: for each block_row, find the strongest neighbour among those who gave hand on 1st phase
        findStrongestNeighbourBlockDiaCsr_V2<<<num_blocks,threads_per_block,0,this->m_stream>>>(A_row_offsets_ptr, A_column_indices_ptr, weights_ptr, n, aggregates_ptr, strongest_neighbour_1phase_ptr, strongest_neighbour_ptr, bsize, 2, this->m_merge_singletons);
        cudaCheckError();
      }

      // Look for perfect matches. Also, for nodes without unaggregated neighbours, merge with aggregate containing strongest neighbour
      matchEdges<<<num_blocks,threads_per_block,0,this->m_stream>>>(n, aggregates_ptr, strongest_neighbour_ptr);
      cudaCheckError();

#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
      s = (icount & 1);
      if( s == 0 ) 
      {
        // count unaggregated vertices
        cudaMemsetAsync(d_unaggregated, 0, sizeof(int), this->m_stream);
        countAggregates<IndexType,threads_per_block><<<num_blocks,threads_per_block,0,this->m_stream>>>(n, aggregates_ptr, d_unaggregated);
        cudaCheckError();

        cudaMemcpyAsync(unaggregated, d_unaggregated, sizeof(int), cudaMemcpyDeviceToHost, this->m_stream);
        throttle_event->record(this->m_stream);
        cudaCheckError();
      }
      else 
      {
        throttle_event->sync();

        numUnassigned_previous = numUnassigned;
        numUnassigned = *unaggregated;
      }
#else
      cudaStreamSynchronize(this->m_stream);
      numUnassigned_previous = numUnassigned;
      numUnassigned = (int)thrust::count(aggregates_thrust_dev_ptr, aggregates_thrust_dev_ptr+n,-1);
      cudaCheckError();
#endif

      icount++;
    } while ( (s == 0) || !(numUnassigned==0 || icount > this->m_max_iterations || 1.0*numUnassigned/n < this->m_numUnassigned_tol || numUnassigned == numUnassigned_previous));
  }
  
  //print
  //printf("icount=%i, numUnassiged=%d, numUnassigned_tol=%f\n", icount, numUnassigned, this->m_numUnassigned_tol);

#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
  delete throttle_event;
#endif

  if( this->m_merge_singletons )
  {
    // Merge remaining vertices with current aggregates
    if (!this->m_deterministic)
    {
      while (numUnassigned != 0) 
      {
        mergeWithExistingAggregatesBlockDiaCsr_V2<<<num_blocks,threads_per_block,0,this->m_stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, n, aggregates_ptr, bsize,this->m_deterministic,(IndexType*) NULL);
        cudaCheckError();

         numUnassigned = (int)thrust::count(aggregates_thrust_dev_ptr, aggregates_thrust_dev_ptr+n,-1);
        cudaCheckError();
      }

    }
    else 
    {
      Vector<int> aggregates_candidate(n);
      aggregates_candidate.fill(-1);

      while (numUnassigned != 0) 
      {
        mergeWithExistingAggregatesBlockDiaCsr_V2<<<num_blocks,threads_per_block,0,this->m_stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, n, aggregates_ptr, bsize,this->m_deterministic,aggregates_candidate.raw());
        cudaCheckError();

        joinExistingAggregates<<<num_blocks,threads_per_block,0,this->m_stream>>>(n, aggregates_ptr, aggregates_candidate.raw());
        cudaCheckError();

        numUnassigned = (int)thrust::count(aggregates_thrust_dev_ptr, aggregates_thrust_dev_ptr+n,-1);
        cudaCheckError();
      }
    }
  }
  else
  {
      //make singletons
      aggregateSingletons<<<num_blocks,threads_per_block,0,this->m_stream>>>( aggregates_ptr, n );
      cudaCheckError();
  }

    renumberAndCountAggregates(aggregates, n, num_aggregates);

    return NVGRAPH_OK; 
}
/*
template <typename IndexType, typename ValueType>
NVGRAPH_ERROR Size2Selector<IndexType, ValueType>::setAggregates(const CsrGraph<IndexType, ValueType> &A, Vector<IndexType> &aggregates, int &num_aggregates)
{
    return setAggregates_common_sqblocks( A, aggregates, num_aggregates);
}
*/

template <typename IndexType, typename ValueType>
NVGRAPH_ERROR Size2Selector<IndexType, ValueType>::setAggregates(
cusparseHandle_t cusp_handle,
const IndexType n_vertex,
const IndexType n_edges, 
IndexType *csr_ptr,
IndexType *csr_ind,
ValueType *csr_val, 
Vector<IndexType> &aggregates, int &num_aggregates)
{
    return setAggregates_common_sqblocks(cusp_handle, n_vertex, n_edges, csr_ptr, csr_ind, csr_val, aggregates, num_aggregates);
}

//template class Size2Selector<int, float>;
//template class Size2Selector<int, double>;
//template void renumberAndCountAggregates  <int> (Vector<int> &aggregates, const int n, int& num_aggregates);

