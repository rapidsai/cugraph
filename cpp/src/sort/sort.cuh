// -*-c++-*-

#pragma once

#include <cudf/types.h>

namespace cusort {

  /**
   * @brief  Sort key value pairs distributed across multiple GPUs
   *
   *   This sort function takes arrays of keys and values distributed
   *   around multiple GPUs, redistributes them so that GPU 0 contains
   *   the smallest elements, GPU 1 the next smallest elements, etc.
   *
   * @param[in]   d_input_keys     The unsorted keys, stored in device arrays.
   *                               input_keys_d[i] is the array of keys on GPU i
   * @param[in]   d_input_values   The unsorted values, stored in device arrays.
   *                               input_values_d[i] is the array of values on GPU i
   * @param[in]   h_input_length   Host array containing the number of elements on
   *                               each GPU in the input key/value arrays.
   * @param[out]  d_output_keys    The sorted keys, stored in device arrays.
   *                               output_keys_d[i] is the array of keys on GPU i
   * @param[out]  d_output_values  The sorted values, stored in device arrays.
   *                               output_values_d[i] is the array of values on GPU i
   * @param[out]  h_output_length  Host array containing the number of elements on
   *                               each GPU in the output key/value arrays.
   * @param[in]   num_gpus         The number of GPUs
   *
   * @return   GDF_SUCCESS upon successful completion
   */
  template <typename Key_t, typename Value_t, typename Length_t>
  gdf_error sort_key_value(Key_t **d_input_keys,
                           Value_t **d_input_values,
                           Length_t *h_input_length,
                           Key_t **d_output_keys,
                           Value_t **d_output_values,
                           Length_t *h_output_length,
                           int num_gpus) {

    return GDF_SUCCESS;
  }
  
  /**
   * @brief  Sort keys distributed across multiple GPUs
   *
   *   This sort function takes an array of keys distributed
   *   around multiple GPUs, redistributes them so that GPU 0 contains
   *   the smallest elements, GPU 1 the next smallest elements, etc.
   *
   * @param[in]   d_input_keys     The unsorted keys, stored in device arrays.
   *                               input_keys_d[i] is the array of keys on GPU i
   * @param[in]   h_input_length   Host array containing the number of elements on
   *                               each GPU in the input key/value arrays.
   * @param[out]  d_output_keys    The sorted keys, stored in device arrays.
   *                               output_keys_d[i] is the array of keys on GPU i
   * @param[out]  h_output_length  Host array containing the number of elements on
   *                               each GPU in the output key/value arrays.
   * @param[in]   num_gpus         The number of GPUs
   *
   * @return   GDF_SUCCESS upon successful completion
   */
  template <typename Key_t, typename Value_t, typename Length_t>
  gdf_error sort_key(Key_t **d_input_keys,
                     Length_t *h_input_lengths,
                     Key_t **d_output_keys,
                     Length_t *h_output_lengths,
                     int num_gpus) {

    return GDF_SUCCESS;
  }
}
