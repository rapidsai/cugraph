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
#include <cusp/system/tbb/detail/execution_policy.h>
#include <cusp/system/cpp/detail/par.h>

#include <thrust/detail/execute_with_allocator.h>

namespace cusp
{
namespace system
{
namespace tbb
{
namespace detail
{


struct par_t : public cusp::system::tbb::detail::execution_policy<par_t>
{
  public:
  par_t() : cusp::system::tbb::detail::execution_policy<par_t>() {}

  template<typename Allocator>
    thrust::detail::execute_with_allocator<Allocator, cusp::system::tbb::detail::execution_policy>
      operator()(Allocator &alloc) const
  {
    return thrust::detail::execute_with_allocator<Allocator, cusp::system::tbb::detail::execution_policy>(alloc);
  }
};

// overloads of select_system
// XXX select_system(cusp::cpp::par, thrust::cpp::par) & select_system(thrust::cpp:par, cusp::cpp::par) are ambiguous
//     because cusp::cpp::par does not directly inherit from thrust::cpp::par, we arbitrarily define the necessary
//     select_system calls here to resolve this issue
template<typename System1, typename System2>
inline __host__ __device__
  System1 select_system(execution_policy<System1> s, cusp::system::cpp::detail::execution_policy<System2>)
{
  return thrust::detail::derived_cast(s);
} // end select_system()


template<typename System1, typename System2>
inline __host__ __device__
  System2 select_system(cusp::system::cpp::detail::execution_policy<System1>, execution_policy<System2> s)
{
  return thrust::detail::derived_cast(s);
} // end select_system()

} // end detail


static const detail::par_t par;


} // end tbb
} // end system


// alias par here
namespace tbb
{


using cusp::system::tbb::par;


} // end tbb
} // end cusp

