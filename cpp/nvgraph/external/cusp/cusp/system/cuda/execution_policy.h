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

/*! \file cusp/system/cuda/execution_policy.h
 *  \brief Execution policies for CUSP's CUDA system.
 */

#include <thrust/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/cuda/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/cuda/detail/par.h>

#if THRUST_VERSION < 100800
#include <cusp/detail/thrust/system/cuda/detail/execute_on_stream.h>
#endif

// now get all the algorithm definitions

#include <cusp/system/cuda/detail/blas.h>
#include <cusp/system/cuda/detail/convert.h>
#include <cusp/system/cuda/detail/copy.h>
#include <cusp/system/cuda/detail/elementwise.h>
#include <cusp/system/cuda/detail/format_utils.h>
#include <cusp/system/cuda/detail/multiply.h>
#include <cusp/system/cuda/detail/sort.h>
#include <cusp/system/cuda/detail/transpose.h>

#include <cusp/system/cuda/detail/graph/breadth_first_search.h>
#include <cusp/system/cuda/detail/graph/connected_components.h>
#include <cusp/system/cuda/detail/graph/hilbert_curve.h>
#include <cusp/system/cuda/detail/graph/maximal_independent_set.h>
#include <cusp/system/cuda/detail/graph/pseudo_peripheral.h>
#include <cusp/system/cuda/detail/graph/symmetric_rcm.h>
#include <cusp/system/cuda/detail/graph/vertex_coloring.h>

#include <cusp/system/cuda/detail/relaxation/gauss_seidel.h>

