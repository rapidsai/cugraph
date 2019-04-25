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
#include <thrust/device_vector.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/detail/temporary_array.h>
#include "util.cuh"
#include "graph_utils.cuh"
//#include <cusp/format_utils.h> //indices_to_offsets

template <typename DerivedPolicy, typename IndexArray, typename OffsetArray>
void indices_to_offsets(const thrust::execution_policy<DerivedPolicy> &exec,
                        const IndexArray& indices, OffsetArray& offsets)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert uncompressed row indices into compressed row offsets
    thrust::lower_bound(exec,
                        indices.begin(),
                        indices.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()),
                        offsets.begin());
}


template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(const thrust::execution_policy<DerivedPolicy> &exec,
                          ArrayType1& keys, ArrayType2& vals//,
                          /*typename ArrayType1::value_type min, typename ArrayType1::value_type max*/)
{
/*
    std::cout<<"## stable_sort_by_key\n" ;
    if(keys.size()!= vals.size()){
          std::cout<<"Error keys.size()!= vals.size()\n" ;
    }
*/
    CUDA_CALL(cudaDeviceSynchronize());
    thrust::stable_sort_by_key(exec, keys.begin(), keys.end(), vals.begin());
    CUDA_CALL(cudaDeviceSynchronize());
//    std::cout<<"## done stable_sort_by_key\n";
}


template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(const thrust::execution_policy<DerivedPolicy> &exec,
                            ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                            typename ArrayType1::value_type min_row = 0,
                            typename ArrayType1::value_type max_row = 0,
                            typename ArrayType2::value_type min_col = 0,
                            typename ArrayType2::value_type max_col = 0)
{
    typedef typename ArrayType1::value_type IndexType1;
    typedef typename ArrayType2::value_type IndexType2;
    typedef typename ArrayType3::value_type ValueType;

    size_t N = row_indices.size();

    
    thrust::detail::temporary_array<IndexType1, DerivedPolicy> permutation(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), N);
    thrust::sequence(exec, permutation.begin(), permutation.end());

/*
    IndexType1 minr = min_row;
    IndexType1 maxr = max_row;
    IndexType2 minc = min_col;
    IndexType2 maxc = max_col;
*/
    //std::cout<<"## max element\n";

/*
    if(maxr == 0){
//        maxr = *thrust::max_element(exec, row_indices.begin(), row_indices.end());
        ArrayType1::iterator maxr_iter = thrust::max_element(exec, row_indices.begin(), row_indices.end());
        maxr = *maxr_ptr;
    }
    if(maxc == 0){
//        maxc = *thrust::max_element(exec, column_indices.begin(), column_indices.end());
        ArrayType2::iterator maxc_iter = thrust::max_element(exec, column_indices.begin(), column_indices.end());
        thrust::copy()
        maxc = *maxc_ptr;
    }
*/
//    std::cout<<"## compute permutation and sort by (I,J)\n";
    // compute permutation and sort by (I,J)
    {
        thrust::detail::temporary_array<IndexType1, DerivedPolicy> temp(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                                                        column_indices.begin(), column_indices.end());
        counting_sort_by_key(exec, temp, permutation/*, minc, maxc*/);
  
        thrust::copy(exec, row_indices.begin(), row_indices.end(), temp.begin());

        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), row_indices.begin());
        counting_sort_by_key(exec, row_indices, permutation/*, minr, maxr*/);
//        thrust::stable_sort_by_key(exec, row_indices.begin(), row_indices.end(), permutation.begin());

        thrust::copy(exec, column_indices.begin(), column_indices.end(), temp.begin());
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), column_indices.begin());

    }
    // use permutation to reorder the values
    {
        thrust::detail::temporary_array<ValueType, DerivedPolicy> temp(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), 
                                                                       values.begin(), values.end()); 
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), values.begin());
    }
}

//#include <cusp/system/detail/generic/format_utils.h>
// --------------------
// Kernels
// --------------------

// Kernel to store aggregate I of each fine point index i
template <typename IndexType>
__global__
void iToIKernel(const IndexType *row_offsets, const IndexType *aggregates, IndexType *I, const int num_rows)
{  
  for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < num_rows; tid += gridDim.x * blockDim.x)
  {
    int agg = aggregates[tid];
    for (int j=row_offsets[tid];j<row_offsets[tid+1];j++)
    {
        I[j] = agg;        
    }
  }
}

// Kernel to store aggregate J of each fine point index j
template <typename IndexType>
__global__
void jToJKernel(const IndexType *column_indices, const IndexType *aggregates, IndexType *J, const int num_entries)
{
  for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < num_entries; tid += gridDim.x * blockDim.x)
  {
    int j = column_indices[tid];
    J[tid] = aggregates[j];
  }
}

//-----------------------------------------------------
// Method to compute the Galerkin product: A_c=R*A*P 
//-----------------------------------------------------

// Method to compute Ac on DEVICE using csr format
template <typename IndexType, typename ValueType>
void generate_superverticies_graph(const int n_vertex, const int num_aggregates, 
                                   thrust::device_vector<IndexType> &csr_ptr_d, 
                                   thrust::device_vector<IndexType> &csr_ind_d,
                                   thrust::device_vector<ValueType> &csr_val_d,
                                   thrust::device_vector<IndexType> &new_csr_ptr_d, 
                                   thrust::device_vector<IndexType> &new_csr_ind_d,
                                   thrust::device_vector<ValueType> &new_csr_val_d,
                                   const thrust::device_vector<IndexType> &aggregates  
                                   ){
  
  const int n_edges = csr_ptr_d[n_vertex];

  
  thrust::device_vector<IndexType> I(n_edges,-1);
  thrust::device_vector<IndexType> J(n_edges,-1);
  thrust::device_vector<ValueType> V(n_edges,-1);

  const int block_size_I = 128;
  const int block_size_J = 256;

  const int num_blocks_I = min( GRID_MAX_SIZE, (int) ((n_vertex-1)/block_size_I + 1) );
  const int num_blocks_J = min( GRID_MAX_SIZE, (int) ((n_edges-1)/block_size_J + 1) );

  const IndexType *row_offsets_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
  const IndexType *column_indices_ptr = thrust::raw_pointer_cast(csr_ind_d.data());
  const IndexType *aggregates_ptr= thrust::raw_pointer_cast(aggregates.data());
  IndexType *I_ptr= thrust::raw_pointer_cast(&I[0]);
  IndexType *J_ptr= thrust::raw_pointer_cast(&J[0]);




  // Kernel to fill array I with aggregates number for fine points i
  iToIKernel<<<num_blocks_I,block_size_I>>>(row_offsets_ptr, aggregates_ptr, I_ptr, (int)n_vertex);
  cudaCheckError();

  // Kernel to fill array J with aggregates number for fine points j
  jToJKernel<<<num_blocks_J,block_size_J>>>(column_indices_ptr, aggregates_ptr, J_ptr, (int)n_edges);
  cudaCheckError();

  // Copy A.values to V array
  thrust::copy(thrust::device, csr_val_d.begin(), csr_val_d.begin() + n_edges, V.begin()); 
  cudaCheckError();
  //cudaDeviceSynchronize();
  

  // Sort (I,J,V) by rows and columns (I,J)
  // TODO : remove cusp depedency
  sort_by_row_and_column(thrust::device, I, J, V);
  cudaCheckError();

  cudaDeviceSynchronize();

  // compute unique number of nonzeros in the output
  IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                        thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                        IndexType(0),
                                        thrust::plus<IndexType>(),
                                        thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;
  cudaCheckError();

  // allocate space for coarse matrix Ac
  new_csr_ptr_d.resize(num_aggregates+1);
  new_csr_ind_d.resize(NNZ);
  new_csr_val_d.resize(NNZ);


  // Reduce by key to fill in Ac.column_indices and Ac.values
  thrust::device_vector<IndexType> new_row_indices(NNZ,0);


  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
         V.begin(),
         thrust::make_zip_iterator(thrust::make_tuple(new_row_indices.begin(), new_csr_ind_d.begin())),
         new_csr_val_d.begin(),
         thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
         thrust::plus<ValueType>());
  cudaCheckError();
  
  indices_to_offsets(thrust::device, new_row_indices, new_csr_ptr_d);
  cudaCheckError();

}

