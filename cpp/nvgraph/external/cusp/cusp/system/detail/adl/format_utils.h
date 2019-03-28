/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a count of the License at
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

// the purpose of this header is to #include the format_utils.h header
// of the sequential, host, and device systems. It should be #included in any
// code which uses adl to dispatch format_utils

#include <cusp/system/detail/sequential/format_utils.h>

// SCons can't see through the #defines below to figure out what this header
// includes, so we fake it out by specifying all possible files we might end up
// including inside an #if 0.
#if 0
#include <cusp/system/cpp/detail/format_utils.h>
#include <cusp/system/cuda/detail/format_utils.h>
#include <cusp/system/omp/detail/format_utils.h>
#include <cusp/system/tbb/detail/format_utils.h>
#endif

#define __CUSP_HOST_SYSTEM_FORMAT_UTILS_HEADER <__CUSP_HOST_SYSTEM_ROOT/detail/format_utils.h>
#include __CUSP_HOST_SYSTEM_FORMAT_UTILS_HEADER
#undef __CUSP_HOST_SYSTEM_FORMAT_UTILS_HEADER

#define __CUSP_DEVICE_SYSTEM_FORMAT_UTILS_HEADER <__CUSP_DEVICE_SYSTEM_ROOT/detail/format_utils.h>
#include __CUSP_DEVICE_SYSTEM_FORMAT_UTILS_HEADER
#undef __CUSP_DEVICE_SYSTEM_FORMAT_UTILS_HEADER

