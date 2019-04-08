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
#include <cnmem_shared_ptr.hxx>
#include "nvgraph_error.hxx"
#include "nvgraph_vector_kernels.hxx"

#include "debug_macros.h"

namespace nvgraph
{

/*! A Vector contains a device vector of size |E| and type T
 */
template <typename ValueType_>
class Vector 
{
public:
    //typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;

protected:
    /*! Storage for the values.
     */
    SHARED_PREFIX::shared_ptr<ValueType> values;

    /*! Size of the array
     */
    size_t size;

    /*! Storage for a cuda stream
     */
    //, cudaStream_t stream = 0

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
        : values(allocateDevice<ValueType>(vertices, stream)),
          size(vertices) {}

    
    size_t get_size() const { return size; }
    size_t bytes() const { return size*sizeof(ValueType);}
    ValueType* raw() const { return values.get(); }
    //cudaStream_t get_stream() const { return stream_; }
    void allocate(size_t n, cudaStream_t stream = 0) 
    {
        size = n; 
        values = allocateDevice<ValueType>(n, stream); 
    }

    void attach(size_t n, ValueType* vals, cudaStream_t stream = 0) 
    {
        size = n;
        values = attachDevicePtr<ValueType>(vals, stream); 
    }

    Vector(size_t vertices, ValueType * vals, cudaStream_t stream = 0)
        : values(attachDevicePtr<ValueType>(vals, stream)),
          size(vertices) {}

    void fill(ValueType val, cudaStream_t stream = 0) 
    {
        fill_raw_vec(this->raw(), this->get_size(), val, stream); 
    } 
    void copy(Vector<ValueType> &vec1, cudaStream_t stream = 0)
    {
        if (this->get_size() == 0 && vec1.get_size()>0)
        {
            allocate(vec1.get_size(), stream);
            copy_vec(vec1.raw(), this->get_size(), this->raw(), stream);
        }
        else if (this->get_size() == vec1.get_size()) 
            copy_vec(vec1.raw(),  this->get_size(), this->raw(), stream);
        else if (this->get_size() > vec1.get_size()) 
        {
            //COUT() << "Warning Copy : sizes mismatch "<< this->get_size() <<':'<< vec1.get_size() <<std::endl;
            copy_vec(vec1.raw(),  vec1.get_size(), this->raw(), stream);
            //dump_raw_vec (this->raw(), vec1.get_size(), 0);
        }
        else
        {
            FatalError("Cannot copy a vector into a smaller one", NVGRAPH_ERR_BAD_PARAMETERS);
        }
    }
    void dump(size_t off, size_t sz, cudaStream_t stream = 0)
    {
        if ((off+sz)<= this->size) 
            dump_raw_vec(this->raw(), sz, off, stream);
        else
            FatalError("Offset and Size values doesn't make sense", NVGRAPH_ERR_BAD_PARAMETERS);
    }
    void flag_zeros(Vector<int> & flags, cudaStream_t stream = 0) 
    {
        flag_zeros_raw_vec(this->get_size(), this->raw(), flags.raw(), stream);
    }

    ValueType nrm1(cudaStream_t stream = 0) 
    { 
        ValueType res = 0;
        nrm1_raw_vec(this->raw(), this->get_size(), &res, stream);
        return res;
    }
}; // class Vector
} // end namespace nvgraph

