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
#include <thrust/random.h>


namespace nvlouvain{

template<typename IdxType, typename IdxIter>
struct link_to_cluster{

  IdxType key;
  IdxIter cluster_iter;
  __host__ __device__
  link_to_cluster(IdxType _key, IdxIter _iter): key(_key), cluster_iter(_iter){}

  __host__ __device__ 
  bool operator()(const IdxType& csr_idx){ 
    return ((*(cluster_iter + csr_idx)) == key);    
  }
};

template<typename IdxType, typename IdxIter>
struct link_inside_cluster{

  IdxType idx_i;
  IdxType key;
  IdxIter cluster_iter;
  __host__ __device__
  link_inside_cluster(IdxType _idx_i, IdxType _key, IdxIter _iter):idx_i(_idx_i), key(_key), cluster_iter(_iter){}

  __host__ __device__ 
  bool operator()(const IdxType& csr_idx){ 
    return ((*(cluster_iter + csr_idx)) == (*(cluster_iter + idx_i))) && ((*(cluster_iter + csr_idx)) == key);    
  }
};

template<typename IdxType, typename IdxIter>
struct link_incident_cluster{

  IdxType key;
  IdxIter cluster_iter;
  IdxType i;
  __host__ __device__
  link_incident_cluster(IdxType _key, IdxIter _iter, IdxType _i): key(_key), cluster_iter(_iter), i(_i){}

  __host__ __device__ 
  bool operator()(const IdxType& csr_idx){ 
    //if(csr_idx == i) return false; 
    return (csr_idx == i) ? false : ((key) == (IdxType)(*(cluster_iter + csr_idx)) );    
  }
};

template<typename IdxType, typename IdxIter>
struct ci_not_equal_cj{

  IdxType key;
  IdxIter cluster_iter;
  __host__ __device__
  ci_not_equal_cj( IdxType _key, IdxIter _iter): key(_key), cluster_iter(_iter){}

  __host__ __device__ 
  bool operator()(const IdxType& idx){ 
    IdxType cj = *(cluster_iter+idx);

    return (cj != key);    
  }
};

template<typename IdxType, typename IdxIter>
struct ci_is_cj{

  IdxType key;
  IdxIter cluster_iter;
  __host__ __device__
  ci_is_cj( IdxType _key, IdxIter _iter): key(_key), cluster_iter(_iter){}

  __host__ __device__ 
  bool operator()(const IdxType& idx){ 
    IdxType cj = *(cluster_iter+idx);
    
    return (cj == key);    
  }
};


template<typename IdxType>
struct rand_functor{
  IdxType low;
  IdxType up;

  __host__ __device__
  rand_functor(IdxType _low, IdxType _up): low(_low), up(_up){}

  __host__ __device__
  bool operator()(const IdxType& idx){
    thrust::random::default_random_engine rand_eng;
    thrust::random::uniform_int_distribution< IdxType > random_op(low, up);
    rand_eng.discard(idx);
    return random_op(rand_eng);
    
  }
};

template<typename IdxType>
struct not_zero{
  __host__ __device__
  bool operator()(const IdxType& idx){
    return (idx != 0);
    
  }
};

template<typename IdxType>
struct is_one{
  __host__ __device__
  bool operator()(const IdxType& x){
    return x == 1;
  }
};

template<typename IdxType>
struct is_c{
  IdxType c;
  __host__ __device__
  is_c(int _c):c(_c){}

  __host__ __device__
  bool operator()(const IdxType& x){
    return x == c;
  }
};


template<typename ValType>
struct not_best{
  ValType best_val;
  __host__ __device__
  not_best(ValType _b):best_val(_b){}
  __host__ __device__ 
  bool operator()(const ValType& val){ 
    return (val != best_val);    
  }
};

template<typename ValType>
struct assign_k_functor{
    ValType* k_ptr;
    __host__ __device__
    assign_k_functor(ValType* _k):k_ptr(_k){}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t){
        //output[i] = k_ptr[ ind[i] ];
        thrust::get<1>(t) = *(k_ptr + thrust::get<0>(t));
       // t.first = *(k_ptr + t.second);
    }
};

template<typename IdxType, typename IdxIter>
struct assign_table_functor{
    IdxType* table_array;
    IdxIter cluster_iter;
    __host__ __device__
    assign_table_functor(IdxIter _c, IdxType* _t):cluster_iter(_c),table_array(_t){}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t){
        //output[i] = k_ptr[ ind[i] ];
//        thrust::get<1>(t) = *(k_ptr + thrust::get<0>(t));
        table_array[*(cluster_iter + thrust::get<0>(t))] = 1;
       // t.first = *(k_ptr + t.second);
    }
};


template<typename IdxType, typename ValType>
struct minus_idx{

    __host__ __device__
    ValType operator()(const IdxType & x, const IdxType & y) const{
      return (ValType) (x - y);
    }
};

template<typename IdxType, typename IdxIter>
struct sort_by_cluster{
    IdxIter cluster_iter;
    __host__ __device__
    sort_by_cluster(IdxIter _c):cluster_iter(_c){}

    __host__ __device__
    bool operator()(const IdxType& a, const IdxType& b){   
      return (IdxType)(*(cluster_iter + a)) < (IdxType)(*(cluster_iter + b));
    }

};


template<typename IdxType>
__device__ inline IdxType not_delta_function(IdxType c1, IdxType c2){
  return (IdxType)(c1!=c2);
}


template<typename IdxType>
__device__ inline IdxType delta_function(IdxType c1, IdxType c2){
  return (IdxType)(c1==c2);
}


}// nvlouvain
