/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

namespace nvlouvain{


template <typename ValType>
class Vector: public rmm::device_vector<ValType>{
  public:
    Vector(): rmm::device_vector<ValType>(){}
    Vector(int size): rmm::device_vector<ValType>(size){}
 
    template <typename Iter> 
    Vector(Iter begin, Iter end): rmm::device_vector<ValType>(begin, end){}
 
    inline void fill(const ValType val){
      thrust::fill(thrust::cuda::par, this->begin(), this->end(), val);
    }
    inline rmm::device_vector<ValType>& to_device_vector(){
      return static_cast<rmm::device_vector<ValType>> (*this);
    }

    inline ValType* raw(){
      return (ValType*)thrust::raw_pointer_cast( rmm::device_vector<ValType>::data() );
    }

    inline int get_size(){
      return this->size();
    }
};

}; //nvlouvain
