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

#include <thrust/pair.h>

namespace cusp
{
namespace system
{
namespace cuda
{

template <typename Size1, typename Size2>
__host__ __device__
Size1 DIVIDE_INTO(Size1 N, Size2 granularity)
{
    return (N + (granularity - 1)) / granularity;
}

template <typename T>
thrust::pair<T,T> uniform_splitting(const T N, const T granularity, const T max_intervals)
{
    const T grains  = DIVIDE_INTO(N, granularity);

    // one grain per interval
    if (grains <= max_intervals)
        return thrust::make_pair(granularity, grains);

    // insures that:
    //     num_intervals * interval_size is >= N
    //   and
    //     (num_intervals - 1) * interval_size is < N
    const T grains_per_interval = DIVIDE_INTO(grains, max_intervals);
    const T interval_size       = grains_per_interval * granularity;
    const T num_intervals       = DIVIDE_INTO(N, interval_size);

    return thrust::make_pair(interval_size, num_intervals);
}

} // end namespace cuda
} // end namespace system
} // end namespace cusp

