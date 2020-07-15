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
#include "nvgraph_vector_kernels.hxx"

#include <rmm/thrust_rmm_allocator.h>

#include "debug_macros.h"

namespace nvgraph
{

/*! A Vector contains a device vector of size |E| and type T
 */
template <typename ValueType_>
class Vector {
public:
  typedef ValueType_ ValueType;

protected:
  rmm::device_vector<ValueType> values;

public:
  /*! Construct an empty \p Vector.
   */
  Vector(void) {}
  ~Vector(void) {}
  /*! Construct a \p Vector of size vertices.
   *
   *  \param vertices The size of the Vector
   */
  Vector(size_t vertices, cudaStream_t stream = 0)
    : values(vertices) {}
    
  size_t get_size() const { return values.size(); }
  size_t bytes() const { return values.size()*sizeof(ValueType);}
  ValueType const *raw() const { return values.data().get();  }
  ValueType *raw() { return values.data().get();  }

  void allocate(size_t n, cudaStream_t stream = 0) 
  {
    values.resize(n);
  }

  void fill(ValueType val, cudaStream_t stream = 0) 
  {
    fill_raw_vec(this->raw(), this->get_size(), val, stream); 
  } 

  void copy(Vector<ValueType> &vec1, cudaStream_t stream = 0)
  {
    if (this->get_size() == 0 && vec1.get_size()>0) {
      allocate(vec1.get_size(), stream);
      copy_vec(vec1.raw(), this->get_size(), this->raw(), stream);
    } else if (this->get_size() == vec1.get_size()) 
      copy_vec(vec1.raw(),  this->get_size(), this->raw(), stream);
    else if (this->get_size() > vec1.get_size()) {
      copy_vec(vec1.raw(),  vec1.get_size(), this->raw(), stream);
    } else {
      FatalError("Cannot copy a vector into a smaller one", NVGRAPH_ERR_BAD_PARAMETERS);
    }
  }

  ValueType nrm1(cudaStream_t stream = 0) { 
    ValueType res = 0;
    nrm1_raw_vec(this->raw(), this->get_size(), &res, stream);
    return res;
  }
}; // class Vector
} // end namespace nvgraph

