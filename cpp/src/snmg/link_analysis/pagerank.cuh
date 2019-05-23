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

// snmg pagerank
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include "cub/cub.cuh"
#include <omp.h>
#include "utilities/graph_utils.cuh"
#include "snmg/utils.cuh"
//#define SNMG_DEBUG

namespace cugraph
{

template <typename IndexType, typename ValueType>
class SNMGpagerank 
{ 
  private:
    size_t v_glob; //global number of vertices
    size_t v_loc;  //local number of vertices
    size_t e_loc;  //local number of edges
    int id; // thread id
    int nt; // number of threads
    ValueType alpha; // damping factor
    SNMGinfo env;  //info about the snmg env setup
    cudaStream_t stream;  
    
    //Vertex offsets for each partition. 
    //This information should be available on all threads/devices
    //part_offsets[device_id] contains the global ID 
    //of the first vertex of the partion owned by device_id. 
    //part_offsets[num_devices] contains the global number of vertices
    size_t* part_off; 
    
    // local CSR matrix
    IndexType * off;
    IndexType * ind;
    ValueType * val;

    // vectors of size v_glob 
    ValueType * bookmark; // constant vector with dangling node info

    bool is_setup;

  public: 
    SNMGpagerank(SNMGinfo & env_, size_t* part_off_, 
                 IndexType * off_, IndexType * ind_);
    ~SNMGpagerank();

    void transition_vals(const IndexType *degree);

    void flag_leafs(const IndexType *degree);

    // Artificially create the google matrix by setting val and bookmark
    void setup(ValueType _alpha, IndexType** degree);

    // run the power iteration on the google matrix
    void solve (int max_iter, ValueType ** pagerank);
};

} //namespace cugraph
