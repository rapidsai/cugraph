// -*-c++-*-

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

#pragma once

#include <cudf/types.h>
#include "sort_impl.cuh"

namespace cusort {

  /**
   * @brief  Sort key value pairs distributed across multiple GPUs
   *
   *   This sort function takes arrays of keys and values distributed
   *   around multiple GPUs, redistributes them so that GPU 0 contains
   *   the smallest elements, GPU 1 the next smallest elements, etc.
   *
   *   The sort function should be called from a serial region of code.
   *   it executes multiple openmp parallel regions to execute functions
   *   on each GPU.
   *
   *   This function will be more efficient if each GPU has been configured
   *   to allow peer access to every other GPU.
   *
   *   The device arrays in d_output_keys and d_output_values are
   *   allocated by this function - since the ultimate partitioning of
   *   the output cannot be known a priori.
   *
   * @param[in]   d_input_keys               The unsorted keys, stored in
   *                                         device arrays. input_keys_d[i]
   *                                         is the array of keys on GPU i
   * @param[in]   d_input_values             The unsorted values, stored in
   *                                         device arrays. input_values_d[i]
   *                                         is the array of values on GPU i
   * @param[in]   h_input_partition_offsets  Host array containing the starting
   *                                         offset of elements on each GPU in
   *                                         the input key/value arrays.
   * @param[out]  d_output_keys              The sorted keys, stored in device
   *                                         arrays. output_keys_d[i] is the
   *                                         array of keys on GPU i
   * @param[out]  d_output_values            The sorted values, stored in
   *                                         device arrays. output_values_d[i]
   *                                         is the array of values on GPU i
   * @param[out]  h_output_partition_offsets Host array containing the starting
   *                                         offset of elements on each GPU in
   *                                         the output key/value arrays.
   * @param[in]   num_gpus                   The number of GPUs
   *
   * @return   GDF_SUCCESS upon successful completion
   */
  template <typename Key_t, typename Value_t, typename Length_t>
  void sort_key_value(Key_t **d_input_keys,
                           Value_t **d_input_values,
                           Length_t *h_input_partition_offsets,
                           Key_t **d_output_keys,
                           Value_t **d_output_values,
                           Length_t *h_output_partition_offsets,
                           int num_gpus) {

    Cusort<Key_t, Value_t, Length_t, 16, 16> sort;
    
    return sort.sort(d_input_keys,
                     d_input_values,
                     h_input_partition_offsets,
                     d_output_keys,
                     d_output_values,
                     h_output_partition_offsets,
                     num_gpus);
  }
  
  /**
   * @brief  Sort keys distributed across multiple GPUs
   *
   *   This sort function takes an array of keys distributed
   *   around multiple GPUs, redistributes them so that GPU 0 contains
   *   the smallest elements, GPU 1 the next smallest elements, etc.
   *
   *   The sort function should be called from a serial region of code.
   *   it executes multiple openmp parallel regions to execute functions
   *   on each GPU.
   *
   *   This function will be more efficient if each GPU has been configured
   *   to allow peer access to every other GPU.
   *
   *   The device arrays in d_output_keys and d_output_values are
   *   allocated by this function - since the ultimate partitioning of
   *   the output cannot be known a priori.
   *
   * @param[in]   d_input_keys               The unsorted keys, stored in
   *                                         device arrays. input_keys_d[i]
   *                                         is the array of keys on GPU i
   * @param[in]   h_input_partition_offset   Host array containing the number
   *                                         of elements on each GPU in the
   *                                         input key/value arrays.
   * @param[out]  d_output_keys              The sorted keys, stored in device
   *                                         arrays. output_keys_d[i] is the
   *                                         array of keys on GPU i
   * @param[out]  h_output_partition_offset  Host array containing the number
   *                                         of elements on each GPU in the
   *                                         output key/value arrays.
   * @param[in]   num_gpus                   The number of GPUs
   *
   * @return   GDF_SUCCESS upon successful completion
   */
  template <typename Key_t, typename Length_t>
  void sort_key(Key_t **d_input_keys,
                     Length_t *h_input_partition_offsets,
                     Key_t **d_output_keys,
                     Length_t *h_output_partition_offsets,
                     int num_gpus) {

    Cusort<Key_t, int, Length_t, 16, 16> sort;
    
    return sort.sort(d_input_keys,
                     h_input_partition_offsets,
                     d_output_keys,
                     h_output_partition_offsets,
                     num_gpus);
  }

  /**
   * @brief Initialize peer-to-peer communications on the GPU
   *
   *   This function should be called from a serial region of code.
   *   It executes an openmp parallel region to execute functions
   *   on each GPU.
   *
   * @param[in]   numGPUs    The number of GPUs we want to communicate
   */
  void initialize_snmg_communication(int numGPUs) {
    omp_set_num_threads(numGPUs);

#pragma omp parallel 
    {
      int gpuId = omp_get_thread_num();

      cudaSetDevice(gpuId);
      for (int g = 0 ; g < numGPUs ; ++g) {
        if (g != gpuId) {
          int isCapable;

          cudaDeviceCanAccessPeer(&isCapable, gpuId, g);
          if (isCapable == 1) {
            cudaError_t err = cudaDeviceEnablePeerAccess(g, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
              cudaGetLastError();
            }
          }
        }
      }
    }
  }
}
