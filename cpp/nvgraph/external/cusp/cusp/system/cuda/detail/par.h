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
#include <cusp/system/cuda/detail/cublas/execute_with_cublas.h>

#if THRUST_VERSION >= 100900
#include <thrust/system/cuda/detail/cross_system.h>
#endif
#include <thrust/detail/execute_with_allocator.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

struct par_t : public cusp::system::cuda::detail::execution_policy<par_t>
{
  public:
  par_t() : cusp::system::cuda::detail::execution_policy<par_t>() {}

  template<typename Allocator>
    thrust::detail::execute_with_allocator<Allocator, cusp::system::cuda::detail::execution_policy>
      operator()(Allocator &alloc) const
  {
    return thrust::detail::execute_with_allocator<Allocator, cusp::system::cuda::detail::execution_policy>(alloc);
  }

  __host__ __device__
  inline cublas::execute_with_cublas with(const cublasHandle_t &handle) const
  {
    return cublas::execute_with_cublas(handle);
  }
};

// overloads of select_system

// cpp interop
template<typename System1, typename System2>
inline __host__ __device__
#if THRUST_VERSION >= 100900
thrust::cuda_cub::cross_system<System1,System2>
#else
thrust::system::cuda::detail::cross_system<System1,System2>
#endif
select_system(const execution_policy<System1> &system1, const cusp::cpp::execution_policy<System2> &system2)
{
  #if THRUST_VERSION >= 100900
  using namespace thrust::cuda_cub;
  #else
  using namespace thrust::system::cuda::detail;
  #endif

  thrust::execution_policy<System1> &non_const_system1 = const_cast<execution_policy<System1>&>(system1);
  cusp::cpp::execution_policy<System2> &non_const_system2 = const_cast<cusp::cpp::execution_policy<System2>&>(system2);
  return cross_system<System1,System2>(non_const_system1,non_const_system2);
}

template<typename System1, typename System2>
inline __host__ __device__
#if THRUST_VERSION >= 100900
thrust::cuda_cub::cross_system<System1,System2>
#else
thrust::system::cuda::detail::cross_system<System1,System2>
#endif
select_system(const cusp::cpp::execution_policy<System1> &system1, const execution_policy<System2> &system2)
{
  #if THRUST_VERSION >= 100900
  using namespace thrust::cuda_cub;
  #else
  using namespace thrust::system::cuda::detail;
  #endif

  cusp::cpp::execution_policy<System1> &non_const_system1 = const_cast<cusp::cpp::execution_policy<System1>&>(system1);
  thrust::execution_policy<System2> &non_const_system2 = const_cast<execution_policy<System2>&>(system2);
  return cross_system<System1,System2>(non_const_system1,non_const_system2);
}

template<typename System>
inline __host__ __device__
typename thrust::detail::disable_if<thrust::detail::is_convertible<thrust::any_system_tag,System>::value,execution_policy<System> &>::type
// execution_policy<System>
select_system(const execution_policy<System> &system, const par_t &)
{
  return system;
}

// template<typename System>
// inline __host__ __device__
// thrust::execution_policy<System>
// select_system(const thrust::execution_policy<par_t>&, const thrust::execution_policy<System> &system)
// {
//   return system;
// }

// template<typename System>
// inline __host__ __device__
// cusp::system::cuda::detail::execution_policy<System>
// select_system(const par_t&, const cusp::cuda::execution_policy<System> &system)
// {
//   return system;
// }

} // end detail


static const detail::par_t par;


} // end cuda
} // end system


// alias par here
namespace cuda
{

using cusp::system::cuda::par;

} // end cuda
} // end cusp

