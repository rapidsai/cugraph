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

/*! \file temporary_array.h
 *  \brief Container-like class temporary storage inside algorithms.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>
#include <cusp/memory.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/detail/temporary_array.h>

#if THRUST_VERSION >= 100800
#define TEMP_HOST_DEVICE_DECORATORS __host__ __device__
#else
#define TEMP_HOST_DEVICE_DECORATORS
#endif

namespace cusp
{

template <typename, typename> class array1d;
template <typename> class array1d_view;

namespace detail
{

template<typename T, typename System>
class temporary_array
    : public thrust::detail::temporary_array<T,System>
{
private:

    typedef thrust::detail::temporary_array<T,System> super_t;

public:

    typedef System                                      memory_space;
    typedef cusp::array1d_format                        format;

    typedef typename super_t::iterator                  iterator;
    typedef typename super_t::size_type                 size_type;

    typedef typename cusp::array1d_view<iterator>       view;

    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system) : super_t(system) {};

    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system, size_type n) : super_t(system, n) {};

    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system, size_type n, T init) : super_t(system, n) {
        super_t::uninitialized_copy(system, thrust::constant_iterator<T>(init), thrust::constant_iterator<T>(init) + n, super_t::begin());
    };

    // provide a kill-switch to explicitly avoid initialization
    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(int uninit, thrust::execution_policy<System> &system, size_type n) : super_t(uninit, system, n) {};

    template<typename InputIterator>
    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system,
                    InputIterator first,
                    size_type n,
                    typename thrust::detail::disable_if<thrust::detail::is_integral<InputIterator>::value>::type* = 0)
        : super_t(system, first, n) {}

    template<typename InputIterator, typename InputSystem>
    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system,
                    thrust::execution_policy<InputSystem> &input_system,
                    InputIterator first,
                    size_type n) : super_t(system, input_system, first, n) {}

    template<typename InputIterator>
    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system,
                    InputIterator first,
                    InputIterator last) : super_t(system, first, last) {}

    template<typename InputSystem, typename InputIterator>
    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system,
                    thrust::execution_policy<InputSystem> &input_system,
                    InputIterator first,
                    InputIterator last) : super_t(system, input_system, first, last) {}

    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system,
                    const cusp::array1d<T,System>& v) : super_t(system, v.begin(), v.end()) {}

    TEMP_HOST_DEVICE_DECORATORS
    temporary_array(thrust::execution_policy<System> &system,
                    const temporary_array<T,System>& v) : super_t(system, v.begin(), v.end()) {}

    void resize(size_type new_size) {
        super_t::allocate(new_size);
    }

}; // end temporary_array

} // end detail

template <typename T, typename DerivedPolicy>
typename detail::temporary_array<T,DerivedPolicy>::view
make_array1d_view(detail::temporary_array<T,DerivedPolicy>& v)
{
    return make_array1d_view(v.begin(), v.end());
}

} // end cusp


