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

/*! \file cusp/system/tbb/detail/execution_policy.h
 *  \brief Execution policies for Intel's TBB system.
 */

#include <cusp/detail/config.h>
// get the execution policies definitions first
#include <cusp/system/cpp/detail/execution_policy.h>
#include <thrust/system/tbb/detail/execution_policy.h>

namespace cusp
{
namespace system
{

namespace tbb
{
namespace detail
{
// allow conversion to tag when it is not a successor
template<typename Derived>
struct execution_policy
        : public thrust::system::tbb::detail::execution_policy<Derived> {};

} // end detail

// alias execution_policy and tag here
using cusp::system::tbb::detail::execution_policy;

} // end tbb
} // end system

// alias items at top-level
namespace tbb
{

using cusp::system::tbb::execution_policy;

} // end tbb
} // end cusp

