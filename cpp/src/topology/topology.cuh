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

//Andrei Schaffer, 6/10/19; 
//

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/distance.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
//#include <thrust/iterator/counting_iterator.h>//
#include <thrust/copy.h>

#include <iostream>
#include <vector>
#include <iterator>


namespace cugraph {
namespace detail {

/**
 * @brief Check symmetry of CSR adjacency matrix;
 * Algorithm outline:
 *  flag = true;
 *  for j in rows:
 *    for k in [row_offsets[j]..row_offsets[j+1]):
 *      col_indx = col_indices[k];
 *      if col_indx > j && col_indx < n-1: # only look above the diagonal
 *         flag &= find(j, [col_indices[row_offsets[col_indx]]..col_indices[row_offsets[col_indx+1]]));
 *  return flag;
 *
 * @tparam IndexT type of indices for rows and columns
 * @tparam Vector type of the container used to hold buffers
 * @param d_row_offsets CSR row ofssets array
 * @param d_col_indices CSR column indices array
 */
template<typename IndexT, template<typename, typename ...> typename Vector>
bool check_symmetry(const Vector<IndexT>& d_row_offsets, const Vector<IndexT>& d_col_indices)
{
  auto nnz   = d_col_indices.size();
  auto nrows = d_row_offsets.size()-1;
  using BoolT = bool;
  Vector<BoolT> d_flags(nrows, 1);

  const IndexT* ptr_r_o = thrust::raw_pointer_cast( &d_row_offsets.front() );
  const IndexT* ptr_c_i = thrust::raw_pointer_cast( &d_col_indices.front() );
  BoolT* start_flags = thrust::raw_pointer_cast( &d_flags.front() ) ;//d_flags.begin();
  BoolT* end_flags = start_flags + nrows;
  BoolT init{1};
  return thrust::transform_reduce(thrust::device,
                                  start_flags, end_flags,
                                  [ptr_r_o, ptr_c_i,start_flags, nnz] __device__ (BoolT& crt_flag){
                                    IndexT row_indx = thrust::distance(start_flags, &crt_flag);
                                    BoolT flag{1};
                                    for(auto k=ptr_r_o[row_indx];k<ptr_r_o[row_indx+1];++k)
                                      {
                                        auto col_indx = ptr_c_i[k];
                                        if( col_indx > row_indx )
                                          {
                                            auto begin = ptr_c_i + ptr_r_o[col_indx];
                                            auto end   = ptr_c_i + ptr_r_o[col_indx+1];//end is okay to point beyond last element of ptr_c_i
                                            auto it = thrust::find(thrust::seq, begin, end, row_indx);
                                            flag &= (it != end);
                                          }
                                      }
                                    return crt_flag & flag;
                                  },
                                  init,
                                  thrust::logical_and<BoolT>());                           
}


/**
 * @brief Check symmetry of CSR adjacency matrix (raw pointers version);
 * Algorithm outline:
 *  flag = true;
 *  for j in rows:
 *    for k in [row_offsets[j]..row_offsets[j+1]):
 *      col_indx = col_indices[k];
 *      if col_indx > j && col_indx < n-1: # only look above the diagonal
 *         flag &= find(j, [col_indices[row_offsets[col_indx]]..col_indices[row_offsets[col_indx+1]]));
 *  return flag;
 *
 * @tparam IndexT type of indices for rows and columns
 * @param nrows number of vertices
 * @param ptr_r_o CSR row ofssets array
 * @param nnz number of edges
 * @param ptr_c_i CSR column indices array
 */
template<typename IndexT>
bool check_symmetry(IndexT nrows, const IndexT* ptr_r_o, IndexT nnz, const IndexT* ptr_c_i)
{
  using BoolT = bool;
  using Vector = thrust::device_vector<BoolT>;
  Vector d_flags(nrows, 1);

  BoolT* start_flags = thrust::raw_pointer_cast( &d_flags.front() ) ;//d_flags.begin();
  BoolT* end_flags = start_flags + nrows;
  BoolT init{1};
  return thrust::transform_reduce(thrust::device,
                                  start_flags, end_flags,
                                  [ptr_r_o, ptr_c_i,start_flags, nnz] __device__ (BoolT& crt_flag){
                                    IndexT row_indx = thrust::distance(start_flags, &crt_flag);
                                    BoolT flag{1};
                                    for(auto k=ptr_r_o[row_indx];k<ptr_r_o[row_indx+1];++k)
                                      {
                                        auto col_indx = ptr_c_i[k];
                                        if( col_indx > row_indx )
                                          {
                                            auto begin = ptr_c_i + ptr_r_o[col_indx];
                                            auto end   = ptr_c_i + ptr_r_o[col_indx+1];//end is okay to point beyond last element of ptr_c_i
                                            auto it = thrust::find(thrust::seq, begin, end, row_indx);
                                            flag &= (it != end);
                                          }
                                      }
                                    return crt_flag & flag;
                                  },
                                  init,
                                  thrust::logical_and<BoolT>());                           
}
} } //end namespace

namespace{ //unnamed namespace for debugging tools:
  template<typename T, typename...Args, template<typename,typename...> class Vector>
    void print_v(const Vector<T, Args...>& v, std::ostream& os)
  {
    thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));//okay
    os<<"\n";
  }

  template<typename T, typename...Args, template<typename,typename...> class Vector>
    void print_v(const Vector<T, Args...>& v, typename Vector<T, Args...>::const_iterator pos, std::ostream& os)
  { 
    thrust::copy(v.begin(), pos, std::ostream_iterator<T>(os,","));//okay
    os<<"\n";
  }

  template<typename T, typename...Args, template<typename,typename...> class Vector>
    void print_v(const Vector<T, Args...>& v, size_t n, std::ostream& os)
  { 
    thrust::copy_n(v.begin(), n, std::ostream_iterator<T>(os,","));//okay
    os<<"\n";
  }

  template<typename T>
    void print_v(const T* p_v, size_t n, std::ostream& os)
  { 
    thrust::copy_n(p_v, n, std::ostream_iterator<T>(os,","));//okay
    os<<"\n";
  }
}
