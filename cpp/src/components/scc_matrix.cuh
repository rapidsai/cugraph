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
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>

#include <type_traits>
//

//Convergence check logic;
//
/**
 * @brief Provide convergence check logic for GEMM SCC via a device pointer
 */
struct CStableChecker
{
  explicit CStableChecker(int flag):
    d_flag_(1, flag)
  {
  }

  //hopefully might be cheaper than copying the value from device to host:
  //
  bool is_set(void) const
  {
    auto pos = thrust::find(d_flag_.begin(), d_flag_.end(), 1);
    return (pos != d_flag_.end());
  }

  void set(int flag)
  {
    thrust::for_each(d_flag_.begin(), d_flag_.end(),
                     [flag] __device__ (int& val){
                       val = flag;
                     });
  }

  int* get_ptr(void)
  {
    return d_flag_.data().get();
  }
private:
  thrust::device_vector<int> d_flag_;
};


/**
 * @brief SCC Algorithm
 * (Adapted from John Gilbert's "Graph Algorithms in the Language of Linear Algebra")
 *
 * C = I | A | A^2 |...| A^k
 * where matrix multiplication is through
 * (combine, reduce) == (&, |) (bitwise ops)
 * Then: X = C & transpose(C);
 * apply get_labels(X);
 */
template<typename ByteT,
         typename IndexT = int>
struct SCC_Data
{
  SCC_Data(size_t nrows,
           const IndexT* p_d_r_o, //row_offsets
           const IndexT* p_d_c_i): //column indices
    nrows_(nrows),
    p_d_r_o_(p_d_r_o),
    p_d_c_i_(p_d_c_i),
    d_C(nrows*nrows, 0),
    d_Cprev(nrows*nrows, 0)
  {
    init();
  }

  const thrust::device_vector<ByteT>& get_C(void) const
  {
    return d_C;
  }

  size_t nrows(void) const
  {
    return nrows_;
  }

  const IndexT* r_o(void) const
  {
    return p_d_r_o_;
  }

  const IndexT* c_i(void) const
  {
    return p_d_c_i_;
  }
  
  //protected: cannot have device lambda inside protected memf
  void init(void)
  {    
    //init d_Cprev to identity:
    //
    auto* p_d_Cprev = d_Cprev.data().get();
    size_t n = nrows_; // for lambda capture, since I cannot capture `this` (host), or `nrows_`
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nrows_),
                     [p_d_Cprev, n] __device__ (size_t indx){
                       p_d_Cprev[indx*n + indx] = ByteT{1};
                     });
  }



  void get_labels(IndexT* d_labels) const
  {
    auto* p_d_C = d_C.data().get();
    size_t n = nrows_; // for lambda capture, since I cannot capture `this` (host), or `nrows_`
    thrust::transform(thrust::device,
                      thrust::make_counting_iterator<IndexT>(0), thrust::make_counting_iterator<IndexT>(nrows_),
                      d_labels,
                      [n, p_d_C] __device__ (IndexT k){
                        auto begin = p_d_C + k*n;
                        auto end = begin + n;
                        ByteT one{1};
                        
                        auto pos = thrust::find_if(thrust::seq,
                                                    begin, end,
                                                    [one] (IndexT entry){
                                                      return (entry == one);
                                                    });


                        //if( pos != end ) // always the case, because C starts as I + A
                          return IndexT(pos-begin);
                      });
                      
  }

  size_t run_scc(IndexT* d_labels)
  {
    size_t nrows = nrows_;
    size_t count = 0;
  

    ByteT* p_d_C = d_C.data().get();
    ByteT* p_d_Cprev = get_Cprev().data().get();
  
    size_t n2 = nrows*nrows;
    const IndexT* p_d_ro = r_o();
    const IndexT* p_d_ci = c_i();
  
    CStableChecker flag(0);
    int* p_d_flag = flag.get_ptr();
    do
      {
        flag.set(0);
      
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(n2),
                         [nrows, p_d_C, p_d_Cprev, p_d_flag, p_d_ro, p_d_ci] __device__ (size_t indx){
                           ByteT one{1};
                         
                           auto i = indx / nrows;
                           auto j = indx % nrows;

                           if( (i == j) || (p_d_Cprev[indx] == one) )
                             p_d_C[indx] = one;
                           else
                             {
                               //this is where a hash-map could help:
                               //only need hashmap[(i,j)]={0,1} (`1` for "hit");
                               //and only for new entries!
                               //already existent entries are covered by
                               //the `if`-branch above!
                               //Hence, hashmap[] can use limited space:
                               //M = max_l{number(new `1` entries)}, where
                               //l = #iterations in the do-loop!
                               //M ~ new `1` entries between A^k and A^{k+1},
                               //    k=1,2,...
                               //Might M actually be M ~ nnz(A) = |E| ?!
                               //Probably, because the primitive hash
                               //(via find_if) uses a search space of nnz(A)
                               //
                               //But, what if more than 1 entry pops-up in a row?
                               //Not an issue! Because the hash key is (i,j), and no
                               //more than one entry can exist in position (i,j)!
                               //
                               //And remember, we only need to store the new (i,j) keys
                               //that an iteration produces wrt to the previous iteration!
                               //
                               auto begin = p_d_ci + p_d_ro[i];
                               auto end   = p_d_ci + p_d_ro[i+1];
                               auto pos = thrust::find_if(thrust::seq,
                                                          begin, end,
                                                          [one, j, nrows, p_d_Cprev, p_d_ci] (IndexT k){
                                                            return (p_d_Cprev[k*nrows+j] == one);
                                                          });


                               if( pos != end )
                                 p_d_C[indx] = one;
                             }

                           if( p_d_C[indx] != p_d_Cprev[indx] )
                             *p_d_flag = 1;//race-condition: harmless, worst case many threads write the same value
                         });
        ++count;
        cudaDeviceSynchronize();
      
        std::swap(p_d_C, p_d_Cprev);
      } while( flag.is_set() );

    //C & Ct:
    //This is the actual reason we need both C and Cprev: 
    //to avoid race condition on C1 = C0 & transpose(C0):
    //
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(n2),
                     [nrows, p_d_C, p_d_Cprev] __device__ (size_t indx){
                       auto i = indx / nrows;
                       auto j = indx % nrows;
                       auto tindx = j*nrows + i;
                     
                       p_d_C[indx] = (p_d_Cprev[indx]) & (p_d_Cprev[tindx]);
                     });

    
    get_labels(d_labels);
      
  
    return count;
  }

private:
  size_t nrows_;
  const IndexT* p_d_r_o_; //row_offsets
  const IndexT* p_d_c_i_; //column indices  
  thrust::device_vector<ByteT> d_C;
  thrust::device_vector<ByteT> d_Cprev;

  thrust::device_vector<ByteT>& get_Cprev(void)
  {
    return d_Cprev;
  }

};
