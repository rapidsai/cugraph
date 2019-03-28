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

// the purpose of this header is to #include the blas.h header
// of the sequential, host, and device systems. It should be #included in any
// code which uses adl to dispatch blas

#include <cusp/system/detail/sequential/blas.h>

// SCons can't see through the #defines below to figure out what this header
// includes, so we fake it out by specifying all possible files we might end up
// including inside an #if 0.
#if 0
#include <cusp/system/cpp/detail/cblas/blas.h>
#include <cusp/system/cuda/detail/cublas/blas.h>

#include <cusp/system/cpp/detail/blas.h>
#include <cusp/system/omp/detail/blas.h>
#include <cusp/system/tbb/detail/blas.h>
#include <cusp/system/cuda/detail/blas.h>
#endif

#define __CUSP_HOST_BLAS_SYSTEM <__CUSP_HOST_BLAS_ROOT/blas.h>
#include __CUSP_HOST_BLAS_SYSTEM
#undef __CUSP_HOST_BLAS_SYSTEM

#define __CUSP_DEVICE_BLAS_SYSTEM <__CUSP_DEVICE_BLAS_ROOT/blas.h>
#include __CUSP_DEVICE_BLAS_SYSTEM
#undef __CUSP_DEVICE_BLAS_SYSTEM

