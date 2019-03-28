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

// #include the host system's execution_policy header
#define __CUSP_DEVICE_SYSTEM_TAG_HEADER <__CUSP_DEVICE_SYSTEM_ROOT/detail/par.h>
#include __CUSP_DEVICE_SYSTEM_TAG_HEADER
#undef __CUSP_DEVICE_SYSTEM_TAG_HEADER

namespace cusp
{

typedef cusp::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::detail::par_t device_memory;

} // end cusp

