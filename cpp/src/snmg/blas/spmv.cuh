/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// snmg spmv
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include "cub/cub.cuh"
#include <omp.h>
#include "rmm_utils.h"
#include "utilities/cusparse_helper.h"
#include "utilities/graph_utils.cuh"
#include "snmg/utils.cuh"
//#define SNMG_DEBUG

namespace cugraph
{

template <typename IndexType, typename ValueType>
class SNMGcsrmv 
{ 

  private:
    size_t v_glob;
    size_t v_loc;
    size_t e_loc;
    SNMGinfo env;
    size_t* part_off;
    int i;
    int p;
    IndexType * off;
    IndexType * ind;
    ValueType * val;
    ValueType * y_loc;
    cudaStream_t stream;
    CusparseCsrMV<ValueType> spmv;
  public: 
    SNMGcsrmv(SNMGinfo & env_, size_t* part_off_, 
              IndexType * off_, IndexType * ind_, ValueType * val_, ValueType ** x);

    ~SNMGcsrmv();

    void run (ValueType ** x);
};


} //namespace cugraph
