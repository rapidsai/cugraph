/*
 *  Copyright 2008-2012 NVIDIA Corporation
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
#include <cusp/system/cpp/detail/execution_policy.h>

namespace cusp
{
namespace system
{
namespace cpp
{
namespace detail
{
namespace cblas
{

template<typename Derived>
struct execution_policy
        : public cusp::cpp::execution_policy<Derived> {};

struct par_t : public execution_policy<par_t>
{
  public:
  par_t() : execution_policy<par_t>() {}
};

} // end namespace cblas
} // end namespace detail
} // end namespace cpp
} // end namespace system

// alias items at top-level
namespace cblas
{

using cusp::system::cpp::detail::cblas::execution_policy;
static const cusp::system::cpp::detail::cblas::par_t par;

} // end cblas
} // end namespace cusp

