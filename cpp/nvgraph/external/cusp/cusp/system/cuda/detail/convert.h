/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <cusp/system/cuda/detail/execution_policy.h>

#include <cusp/copy.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template <typename System1,
          typename System2,
          typename SourceType,
          typename DestinationType>
void convert(
#if THRUST_VERSION >= 100900
    thrust::cuda_cub::cross_system<System1,System2>& exec,
#else
    thrust::system::cuda::detail::cross_system<System1,System2>& exec,
#endif
             const SourceType& src,
                   DestinationType& dst)
{
    typedef typename DestinationType::container                Container;
    typedef typename Container::template rebind<System1>::type DestinationType2;

    DestinationType2 tmp;

    cusp::convert(src, tmp);
    cusp::copy(tmp, dst);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

