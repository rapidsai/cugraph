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

/*! \file cusp/system/cuda/detail/execution_policy.h
 *  \brief Execution policies for CUSP's CUDA system.
 */

#include <cusp/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/cuda/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/cuda/detail/par.h>

#if THRUST_VERSION < 100800
#include <cusp/detail/thrust/system/cuda/detail/execute_on_stream.h>
#endif

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{
// allow conversion to tag when it is not a successor
template<typename Derived>
struct execution_policy
    : public
#if THRUST_VERSION >= 100900
    thrust::system::cuda::execution_policy<Derived>
#else
    thrust::system::cuda::detail::execution_policy<Derived>
#endif
    {};

} // end detail

// alias execution_policy and tag here
using cusp::system::cuda::detail::execution_policy;

} // end cuda
} // end system

// alias items at top-level
namespace cuda
{

using cusp::system::cuda::execution_policy;

} // end cuda
} // end cusp

