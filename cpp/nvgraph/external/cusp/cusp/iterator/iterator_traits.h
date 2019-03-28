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


/*! \file cusp/iterator/iterator_traits.h
 *  \brief Traits and metafunctions for reasoning about the traits of iterators
 */

/*
 * (C) Copyright David Abrahams 2003.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <cusp/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <iterator>

#include <cusp/iterator/detail/host_system_tag.h>
#include <cusp/iterator/detail/device_system_tag.h>
#include <cusp/iterator/detail/any_system_tag.h>

namespace cusp
{
template<typename Space>
struct iterator_system
        // convertible to host iterator?
        : thrust::detail::eval_if<
          thrust::detail::or_<
            thrust::detail::is_same<Space, thrust::host_system_tag>,
            thrust::detail::is_same<Space, cusp::host_memory>
        >::value,

        thrust::detail::identity_<cusp::host_memory>,

        // convertible to device iterator?
        thrust::detail::eval_if<
          thrust::detail::or_<
            thrust::detail::is_same<Space, thrust::device_system_tag>,
            thrust::detail::is_same<Space, cusp::device_memory>
        >::value,

        thrust::detail::identity_<cusp::device_memory>,

        // convertible to any iterator?
        thrust::detail::eval_if<
          thrust::detail::or_<
            thrust::detail::is_same<Space, thrust::any_system_tag>,
            thrust::detail::is_same<Space, cusp::any_memory>
        >::value,

        thrust::detail::identity_<cusp::any_memory>,

        // unknown system
        thrust::detail::identity_<void>
        > // if any
        > // if device
        > // if host
{
}; // end iterator_system

} // end namespace cusp
