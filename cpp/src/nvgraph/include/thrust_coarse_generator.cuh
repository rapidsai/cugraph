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

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "graph_utils.cuh"
#include "util.cuh"

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


template<typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(const thrust::execution_policy<DerivedPolicy> &exec,
                          ArrayType1& keys,
                          ArrayType2& vals) {
  thrust::stable_sort_by_key(exec, keys.begin(), keys.end(), vals.begin());
}


template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(const thrust::execution_policy<DerivedPolicy> &exec,
                            ArrayType1& row_indices,
                            ArrayType2& column_indices, ArrayType3& values,
                            typename ArrayType1::value_type min_row = 0,
                            typename ArrayType1::value_type max_row = 0,
                            typename ArrayType2::value_type min_col = 0,
                            typename ArrayType2::value_type max_col = 0)
                            {
  typedef typename ArrayType1::value_type IndexType1;
  typedef typename ArrayType2::value_type IndexType2;
  typedef typename ArrayType3::value_type ValueType;

  size_t N = row_indices.size();

  thrust::detail::temporary_array<IndexType1, DerivedPolicy> permutation(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                                                         N);
  thrust::sequence(exec, permutation.begin(), permutation.end());

  // compute permutation and sort by (I,J)
  {
    thrust::detail::temporary_array<IndexType1, DerivedPolicy> temp(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                                                    column_indices.begin(),
                                                                    column_indices.end());
    counting_sort_by_key(exec, temp, permutation/*, minc, maxc*/);
    thrust::copy(exec, row_indices.begin(), row_indices.end(), temp.begin());
    thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), row_indices.begin());
    counting_sort_by_key(exec, row_indices, permutation/*, minr, maxr*/);
    thrust::copy(exec, column_indices.begin(), column_indices.end(), temp.begin());
    thrust::gather(exec,
                   permutation.begin(),
                   permutation.end(),
                   temp.begin(),
                   column_indices.begin());

  }
  // use permutation to reorder the values
  {
    thrust::detail::temporary_array<ValueType, DerivedPolicy> temp(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                                                   values.begin(),
                                                                   values.end());
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

template <typename IdxT, typename ValT>
__global__ void ijToIJKernel(IdxT nnz,
                             IdxT num_verts,
                             IdxT* csr_off,
                             IdxT* csr_ind,
                             ValT* edge_weights,
                             IdxT* clusters,
                             IdxT* I,
                             IdxT* J,
                             ValT* V) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < nnz) {
    IdxT startVertex = binsearch_maxle(csr_off, tid, (IdxT)0, num_verts);
    IdxT endVertex = csr_ind[tid];
    IdxT startCluster = clusters[startVertex];
    IdxT endCluster = clusters[endVertex];
    if (startCluster == endCluster){
      I[tid] = -1;
      J[tid] = -1;
      V[tid] = 0.0;
    }
    else {
      I[tid] = startCluster;
      J[tid] = endCluster;
      V[tid] = edge_weights[tid];
    }

    tid += gridDim.x * blockDim.x;
  }
}

//-----------------------------------------------------
// Method to compute the Galerkin product: A_c=R*A*P 
//-----------------------------------------------------

// Method to compute Ac on DEVICE using csr format
template <typename IndexType, typename ValueType>
void generate_superverticies_graph(const int n_vertex,
                                   const int num_aggregates,
                                   rmm::device_vector<IndexType> &csr_ptr_d, 
                                   rmm::device_vector<IndexType> &csr_ind_d,
                                   rmm::device_vector<ValueType> &csr_val_d,
                                   rmm::device_vector<IndexType> &new_csr_ptr_d, 
                                   rmm::device_vector<IndexType> &new_csr_ind_d,
                                   rmm::device_vector<ValueType> &new_csr_val_d,
                                   rmm::device_vector<IndexType> &aggregates){
  
  int n_edges = csr_ptr_d[n_vertex];

  
  rmm::device_vector<IndexType> I(n_edges,-1);
  rmm::device_vector<IndexType> J(n_edges,-1);
  rmm::device_vector<ValueType> V(n_edges,-1);

  IndexType *row_offsets_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
  IndexType *column_indices_ptr = thrust::raw_pointer_cast(csr_ind_d.data());
  ValueType *edge_weights_ptr = thrust::raw_pointer_cast(csr_val_d.data());
  IndexType *aggregates_ptr= thrust::raw_pointer_cast(aggregates.data());
  IndexType *I_ptr= thrust::raw_pointer_cast(&I[0]);
  IndexType *J_ptr= thrust::raw_pointer_cast(&J[0]);
  ValueType *V_ptr = thrust::raw_pointer_cast(V.data());

  dim3 grid, block;
  block.x = 512;
  grid.x = min((IndexType) CUDA_MAX_BLOCKS, (n_edges / 512 + 1));
  ijToIJKernel<<<grid, block, 0, nullptr>>>(n_edges,
                                            n_vertex,
                                            row_offsets_ptr,
                                            column_indices_ptr,
                                            edge_weights_ptr,
                                            aggregates_ptr,
                                            I_ptr,
                                            J_ptr,
                                            V_ptr);
  
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
  NNZ = NNZ - 1;
  new_csr_ptr_d.resize(num_aggregates+1);
  new_csr_ind_d.resize(NNZ);
  new_csr_val_d.resize(NNZ);


  // Reduce by key to fill in Ac.column_indices and Ac.values
  rmm::device_vector<IndexType> new_row_indices(NNZ,0);
  rmm::device_vector<IndexType> I_out(NNZ + 1, 0);
  rmm::device_vector<IndexType> J_out(NNZ + 1, 0);
  rmm::device_vector<ValueType> V_out(NNZ + 1, 0);

  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
                        V.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(I_out.begin(),
                                                                     J_out.begin())),
                        V_out.begin(),
                        thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                        thrust::plus<ValueType>());
  cudaCheckError();
  
  thrust::copy(I_out.begin() + 1, I_out.end(), new_row_indices.begin());
  thrust::copy(J_out.begin() + 1, J_out.end(), new_csr_ind_d.begin());
  thrust::copy(V_out.begin() + 1, V_out.end(), new_csr_val_d.begin());

  indices_to_offsets(thrust::device, new_row_indices, new_csr_ptr_d);
  cudaCheckError();
}

template <typename IndexType, typename ValueType>
void generate_supervertices_graph(const int n_vertex,
                                   const int nnz,
                                   const int num_aggregates,
                                   IndexType* csr_off,
                                   IndexType* csr_ind,
                                   ValueType* csr_val,
                                   IndexType** new_csr_off,
                                   IndexType** new_csr_ind,
                                   ValueType** new_csr_val,
                                   IndexType* aggregates,
                                   IndexType& new_nnz){

  IndexType n_edges = nnz;

  rmm::device_vector<IndexType> I(n_edges,-1);
  rmm::device_vector<IndexType> J(n_edges,-1);
  rmm::device_vector<ValueType> V(n_edges,-1);

  IndexType *I_ptr= thrust::raw_pointer_cast(&I[0]);
  IndexType *J_ptr= thrust::raw_pointer_cast(&J[0]);
  ValueType *V_ptr = thrust::raw_pointer_cast(V.data());

  dim3 grid, block;
  block.x = 512;
  grid.x = min((IndexType) CUDA_MAX_BLOCKS, (n_edges / 512 + 1));
  ijToIJKernel<<<grid, block, 0, nullptr>>>(n_edges,
                                            n_vertex,
                                            csr_off,
                                            csr_ind,
                                            csr_val,
                                            aggregates,
                                            I_ptr,
                                            J_ptr,
                                            V_ptr);

  // Sort (I,J,V) by rows and columns (I,J)
  sort_by_row_and_column(thrust::device, I, J, V);
  cudaCheckError();

  // compute unique number of nonzeros in the output
  IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                        thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                        IndexType(0),
                                        thrust::plus<IndexType>(),
                                        thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;
  cudaCheckError();

  // allocate space for coarse matrix Ac
  NNZ = NNZ - 1;
  new_nnz = NNZ;
  ALLOC_TRY(new_csr_off, sizeof(IndexType) * (num_aggregates + 1), nullptr);
  ALLOC_TRY(new_csr_ind, sizeof(IndexType) * NNZ, nullptr);
  ALLOC_TRY(new_csr_val, sizeof(ValueType) * NNZ, nullptr);

  // Reduce by key to fill in Ac.column_indices and Ac.values
  rmm::device_vector<IndexType> new_row_indices(NNZ,0);
  rmm::device_vector<IndexType> I_out(NNZ + 1, 0);
  rmm::device_vector<IndexType> J_out(NNZ + 1, 0);
  rmm::device_vector<ValueType> V_out(NNZ + 1, 0);

  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
                        V.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(I_out.begin(),
                                                                     J_out.begin())),
                        V_out.begin(),
                        thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                        thrust::plus<ValueType>());
  cudaCheckError();

  thrust::copy(I_out.begin() + 1, I_out.end(), new_row_indices.begin());
  thrust::copy(J_out.begin() + 1, J_out.end(), *new_csr_ind);
  thrust::copy(V_out.begin() + 1, V_out.end(), *new_csr_val);

  thrust::lower_bound(rmm::exec_policy(nullptr)->on(nullptr),
                      new_row_indices.begin(),
                      new_row_indices.end(),
                      thrust::counting_iterator<IndexType>(0),
                      thrust::counting_iterator<IndexType>(num_aggregates + 1),
                      *new_csr_off);
  cudaCheckError();
}

