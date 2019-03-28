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

/*! \file cusp/system/detail/sequential/execution_policy.h
 *  \brief Execution policies for CUSP's standard sequential system.
 */

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

#include <thrust/system/cpp/detail/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

template<typename Derived>
struct execution_policy
      : public thrust::cpp::execution_policy<Derived> {};

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

