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

namespace nvlouvain{


template <typename ValType>
class Vector: public thrust::device_vector<ValType>{
  public:
    Vector(): thrust::device_vector<ValType>(){}
    Vector(int size): thrust::device_vector<ValType>(size){}
 
    template <typename Iter> 
    Vector(Iter begin, Iter end): thrust::device_vector<ValType>(begin, end){}
 
    inline void fill(const ValType val){
      thrust::fill(thrust::cuda::par, this->begin(), this->end(), val);
    }
    inline thrust::device_vector<ValType>& to_device_vector(){
      return static_cast<thrust::device_vector<ValType>> (*this);
    }

    inline ValType* raw(){
      return (ValType*)thrust::raw_pointer_cast( thrust::device_vector<ValType>::data() );
    }

    inline int get_size(){
      return this->size();
    }
};


template <typename IndexType, typename ValueType>
class CsrGraph{
     
  public:
    CsrGraph( thrust::device_vector<IndexType>& csr_ptr_d, thrust::device_vector<IndexType>& csr_ind_d,  thrust::device_vector<ValueType>& csr_val_d, IndexType v, IndexType e, bool _w=false):
    _n_vertices(v), _n_edges(e), csr_ptr(csr_ptr_d.begin(), csr_ptr_d.end()), csr_ind(csr_ind_d.begin(), csr_ind_d.end()), csr_val(csr_val_d.begin(), csr_val_d.end()), weighted(_w){
    }
    
    CsrGraph( thrust::host_vector<IndexType>& csr_ptr_d, thrust::host_vector<IndexType>& csr_ind_d,  thrust::host_vector<ValueType>& csr_val_d, IndexType v, IndexType e, bool _w=false):
    _n_vertices(v), _n_edges(e), csr_ptr(csr_ptr_d.begin(), csr_ptr_d.end()), csr_ind(csr_ind_d.begin(), csr_ind_d.end()), csr_val(csr_val_d.begin(), csr_val_d.end()), weighted(_w){
    }


    inline const IndexType get_num_vertices() const{
      return _n_vertices;
    }

    inline const IndexType get_num_edges() const{
      return csr_ptr.back();
    } 
    inline const IndexType* get_raw_row_offsets() const{
      return thrust::raw_pointer_cast(csr_ptr.data());
    }
    inline const IndexType* get_raw_column_indices()const {
      return thrust::raw_pointer_cast(csr_ind.data());;
    }
    inline const ValueType* get_raw_values() const{
      return thrust::raw_pointer_cast(csr_val.data());
    }
    inline const Vector<IndexType> & get_row_offsets() const{
      return csr_ptr;
    }
    inline const Vector<IndexType> & get_column_indices() const{
      return csr_ind;
    }
    inline const Vector<ValueType> & get_values() const{
      return csr_val;
    }
    inline const Vector<IndexType> & get_csr_ptr() const{
      return csr_ptr;
    }
    inline const Vector<IndexType> & get_csr_ind() const{
      return csr_ind;
    }
    inline const Vector<ValueType> & get_csr_val() const{
      return csr_val;
    }
 
    inline void update_csr_ptr(thrust::device_vector<IndexType> & d_v){
      thrust::copy(thrust::cuda::par, d_v.begin(), d_v.end(), csr_ptr.begin());
    }
    inline void update_csr_ptr_n(thrust::device_vector<IndexType> & d_v,unsigned size){
      csr_ptr.resize(size);
      thrust::copy_n(thrust::cuda::par, d_v.begin(), size, csr_ptr.begin());
    } 


    inline void update_csr_ind(thrust::device_vector<IndexType> & d_v){
      thrust::copy(thrust::cuda::par, d_v.begin(), d_v.end(), csr_ind.begin());
    }
    inline void update_csr_ind_n(thrust::device_vector<IndexType> & d_v,unsigned size){
      csr_ind.resize(size);
      thrust::copy_n(thrust::cuda::par, d_v.begin(), size, csr_ind.begin());
    } 


    inline void update_csr_val(thrust::device_vector<ValueType> & d_v){
      thrust::copy(thrust::cuda::par, d_v.begin(), d_v.end(), csr_val.begin());
    }  
    inline void update_csr_val_n(thrust::device_vector<ValueType> & d_v,unsigned size){
      csr_val.resize(size); 
      thrust::copy_n(thrust::cuda::par, d_v.begin(), size, csr_val.begin());
    } 
    inline void update_graph(size_t n_v, size_t n_e, thrust::device_vector<IndexType> & ptr, thrust::device_vector<IndexType> & ind, thrust::device_vector<ValueType> & val, bool w){
      _n_vertices = n_v;
      _n_edges = n_e;
#ifdef DEBUG
      if(n_v != ptr.size()){
        std::cout<<"n_vertex size not match\n";
      }
      if(n_e != ind.size() || n_e != val.size()){
        std::cout<<"n_edges size not match\n";
      }
#endif 
      update_csr_ptr_n(ptr, _n_vertices);
      update_csr_ind_n(ind, _n_edges);
      update_csr_val_n(val, _n_edges);
      weighted = w;
    }
  private:
    size_t _n_vertices;
    size_t _n_edges;
    Vector<IndexType> csr_ptr;
    Vector<IndexType> csr_ind;
    Vector<ValueType> csr_val;
    bool weighted;
};




}; //nvlouvain
