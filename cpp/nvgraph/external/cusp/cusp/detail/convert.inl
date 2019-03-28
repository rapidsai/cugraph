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


#include <cusp/detail/config.h>

#include <cusp/system/detail/adl/convert.h>
#include <cusp/system/detail/generic/convert.h>

#include <cusp/detail/execution_policy.h>
#include <thrust/system/detail/generic/select_system.h>

namespace cusp
{

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void convert(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             const SourceType& src,
                   DestinationType& dst)
{
    using cusp::system::detail::generic::convert;

    return convert(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), src, dst);
}

template <typename SourceType,
          typename DestinationType>
void convert(const SourceType& src,
                   DestinationType& dst)
{
    using thrust::system::detail::generic::select_system;

    typedef typename SourceType::memory_space System1;
    typedef typename DestinationType::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::convert(select_system(system1,system2), src, dst);
}

} // end namespace cusp

