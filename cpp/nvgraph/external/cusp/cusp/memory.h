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

/*! \file memory.h
 *  \brief Memory spaces and allocators
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/iterator/iterator_traits.h>

namespace cusp
{

template<typename T, typename MemorySpace>
struct default_memory_allocator;

template <typename MemorySpace1,
          typename MemorySpace2=any_memory,
          typename MemorySpace3=any_memory,
          typename MemorySpace4=any_memory>
struct minimum_space;


} // end namespace cusp

#include <cusp/detail/memory.inl>

