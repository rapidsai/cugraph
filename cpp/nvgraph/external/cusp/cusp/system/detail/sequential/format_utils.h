/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

template <typename DerivedPolicy,
          typename OffsetArray,
          typename IndexArray>
void offsets_to_indices(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const OffsetArray& offsets,
                        IndexArray& indices)
{
    typedef typename OffsetArray::value_type OffsetType;

    typename IndexArray::iterator iter(indices.begin());

    for(size_t i = 0; i < offsets.size() - 1; i++)
        for(OffsetType j = 0; j < (offsets[i + 1] - offsets[i]); j++)
            *iter++ = i;
}

template <typename DerivedPolicy,
          typename IndexArray,
          typename OffsetArray>
void indices_to_offsets(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const IndexArray& indices,
                        OffsetArray& offsets)
{
    for(size_t i = 0; i < offsets.size(); i++)
        offsets[i] = 0;

    for(size_t i = 0; i < indices.size(); i++)
        offsets[indices[i] + 1]++;

    for(size_t i = 1; i < offsets.size(); i++)
        offsets[i] += offsets[i - 1];
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

