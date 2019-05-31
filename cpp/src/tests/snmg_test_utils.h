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

// Interanl helper functions 
// Author: Alex Fender afender@nvidia.com

#pragma once

#include <omp.h>
#include "test_utils.h"
#include <fstream>      // std::ifstream


// global to local offsets by shifting all offsets by the first offset value
template <typename T>
void shift_offsets(std::vector<T> & off_loc) {
  auto start = off_loc.front();
  for (auto i = size_t{0}; i < off_loc.size(); ++i)
    off_loc[i] -= start;
}

// 1D partitioning such as each GPU has about the same number of edges
template <typename T>
void edge_partioning(std::vector<T> & off_h, std::vector<size_t> & part_offset, std::vector<size_t> & v_loc, std::vector<size_t> & e_loc) {
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads();

  //set first and last partition offsets
  part_offset[0] = 0;
  part_offset[p] = off_h.size()-1;
  
  if (i>0) {
    //get the first vertex ID of each partition
    auto loc_nnz = off_h.back()/p;
    auto start_nnz = i*loc_nnz;
    auto start_v = 0;
    for (auto j = size_t{0}; j < off_h.size(); ++j) {
      if (off_h[j] > start_nnz) {
        start_v = j;
        break;
      }
    }
    part_offset[i] = start_v;
  }
  // all threads must know their partition offset 
  #pragma omp barrier 

  // Store the local number of V and E for convinience
  v_loc[i] = part_offset[i+1] - part_offset[i];
  e_loc[i] = off_h[part_offset[i+1]] - off_h[part_offset[i]];
}

// csv for HiBench
template <typename idx_t>
int read_single_file(std::string fileName,
        std::vector<idx_t>& s,
        std::vector<idx_t>& d) {
    s.clear();
    d.clear();
    std::ifstream f(fileName);
    if (!f) { return 1; }
    idx_t src, dst;
    while (f>>src>>dst) {
        s.push_back(src);
        d.push_back(dst);
    }
    f.close();
    return 0;
}

template <typename idx_t,typename val_t>
void load_csr_loc(std::vector<idx_t> & off_h, std::vector<idx_t> & ind_h, std::vector<val_t> & val_h, 
                  std::vector<size_t> & v_loc, std::vector<size_t> & e_loc, std::vector<size_t> & part_offset,
                  gdf_column* col_off, gdf_column* col_ind, gdf_column* col_val)
{
 
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads(); 
  edge_partioning(off_h, part_offset, v_loc, e_loc);
  
  ASSERT_EQ(part_offset[i+1]-part_offset[i], v_loc[i]);
  
  std::vector<idx_t> off_loc(off_h.begin()+part_offset[i],off_h.begin()+part_offset[i+1]+1), 
                     ind_loc(ind_h.begin()+off_h[part_offset[i]],ind_h.begin()+off_h[part_offset[i+1]]);
  std::vector<val_t> val_loc(val_h.begin()+off_h[part_offset[i]],val_h.begin()+off_h[part_offset[i+1]]);
  ASSERT_EQ(off_loc.size(), v_loc[i]+1);
  ASSERT_EQ(ind_loc.size(), e_loc[i]);
  ASSERT_EQ(val_loc.size(), e_loc[i]);

  #ifdef SNMG_VERBOSE
  #pragma omp barrier 
  #pragma omp master 
  { 
    std::cout << off_h[part_offset[i]]<< std::endl;
    std::cout << off_h[part_offset[i+1]]<< std::endl;
    for (auto j = part_offset.begin(); j != part_offset.end(); ++j)
      std::cout << *j << ' ';
    std::cout << std::endl;
    for (auto j = v_loc.begin(); j != v_loc.end(); ++j)
      std::cout << *j << ' ';
    std::cout << std::endl;  
    for (auto j = e_loc.begin(); j != e_loc.end(); ++j)
      std::cout << *j << ' ';
    std::cout << std::endl;
  }
  #pragma omp barrier 
  #endif


  shift_offsets(off_loc);

  ASSERT_EQ(static_cast<size_t>(off_loc[part_offset[i+1]-part_offset[i]]),e_loc[i]);

  create_gdf_column(off_loc, col_off);
  ASSERT_EQ(off_loc.size(), static_cast<size_t>(col_off->size));
  
  create_gdf_column(ind_loc, col_ind);
  create_gdf_column(val_loc, col_val);
}
