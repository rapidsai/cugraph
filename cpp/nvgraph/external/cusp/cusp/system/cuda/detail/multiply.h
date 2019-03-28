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

#if THRUST_VERSION >= 100800 && THRUST_VERSION < 100900
#include <cusp/system/cuda/detail/multiply/coo_spmv_cub.h>
#include <cusp/system/cuda/detail/multiply/csr_vector_spmv_cub.h>
#else
#include <cusp/system/cuda/detail/multiply/coo_flat_spmv.h>
#include <cusp/system/cuda/detail/multiply/csr_vector_spmv.h>
#endif

#include <cusp/system/cuda/detail/multiply/csr_block_spmv.h>

#include <cusp/system/cuda/detail/multiply/dense.h>
#include <cusp/system/cuda/detail/multiply/dia_spmv.h>
#include <cusp/system/cuda/detail/multiply/ell_spmv.h>
// #include <cusp/system/cuda/detail/multiply/hyb_spmv.h>

#include <cusp/system/cuda/detail/multiply/spgemm.h>

