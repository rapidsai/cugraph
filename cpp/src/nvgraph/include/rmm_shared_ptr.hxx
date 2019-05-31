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

#include <cstring>
#include <iostream>
#include <memory>

#include "rmm/rmm.h"

#include "nvgraph_error.hxx"

namespace nvgraph
{

template< typename T >
class DeviceDeleter
{
    cudaStream_t mStream;
public:
    DeviceDeleter(cudaStream_t stream) : mStream(stream) {}
    void operator()(T *ptr)
    {
        auto status = RMM_FREE(ptr, mStream);
        if (status != RMM_SUCCESS) {
            FatalError("Memory manager internal error (free)", NVGRAPH_ERR_UNKNOWN);
        }
    }
};


template< typename T >
inline std::shared_ptr<T> allocateDevice(size_t n, cudaStream_t stream)
{
    T *ptr = NULL;
    auto status = RMM_ALLOC(&ptr, n * sizeof(T), stream);
    if (status == RMM_ERROR_OUT_OF_MEMORY) {
        FatalError("Not enough memory", NVGRAPH_ERR_NO_MEMORY);
    }
    else if (status != RMM_SUCCESS) {
        FatalError("Memory manager internal error (alloc)", NVGRAPH_ERR_UNKNOWN);
    }
    return std::shared_ptr<T>(ptr, DeviceDeleter<T>(stream));
}

template< typename T >
class DeviceReleaser
{
    cudaStream_t mStream;
public:
    DeviceReleaser(cudaStream_t stream) : mStream(stream) {}
    void operator()(T *ptr)
    {

    }
};

template< typename T >
inline std::shared_ptr<T> attachDevicePtr(T * ptr_in, cudaStream_t stream)
{
    T *ptr = ptr_in;
    return std::shared_ptr<T>(ptr, DeviceReleaser<T>(stream));
}


} // end namespace nvgraph

