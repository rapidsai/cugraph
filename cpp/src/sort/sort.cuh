// -*-c++-*-

#pragma once

#include <cudf/types.h>

namespace cusort {

  /**
   * @brief  Sort key value pairs distributed across multiple GPUs
   *
   *   This sort function takes arrays of keys and values distributed
   *   around multiple GPUs 
   *
   * @param[in]   input_keys_d     The unsorted keys, stored in device arrays.
   *                               input_keys_d[i] is the array of keys on GPU i
   * @param[in]   input_values_d   The unsorted values, stored in device arrays.
   *                               input_values_d[i] is the array of values on GPU i
   * @param[in]   input_length_h   Host array containing the number of elements on
   *                               each GPU in the input key/value arrays.
   * @param[out]  output_keys_d    The sorted keys, stored in device arrays.
   *                               output_keys_d[i] is the array of keys on GPU i
   * @param[out]  output_values_d  The sorted values, stored in device arrays.
   *                               output_values_d[i] is the array of values on GPU i
   * @param[out]  output_length_h  Host array containing the number of elements on
   *                               each GPU in the output key/value arrays.
   * @param[in]   num_gpus         The number of GPUs
   *
   * @return   GDF_SUCCESS upon successful completion
   */
  template <typename Key_t, typename Value_t, typename Length_t>
  gdf_error sortKeyValue(Key_t **input_keys_d,
                         Value_t **input_values_d,
                         Length_t *input_length_h,
                         Key_t **output_keys_d,
                         Value_t **output_values_d,
                         Length_t *output_length_h,
                         int num_gpus) {

    return GDF_SUCCESS;
  }
  
  /**
   * @brief  Sort key value pairs distributed across multiple GPUs
   *
   *   This sort function takes arrays of keys and values distributed
   *   around multiple GPUs 
   *
   * @param[in]   input_keys_d     The unsorted keys, stored in device arrays.
   *                               input_keys_d[i] is the array of keys on GPU i
   * @param[in]   input_length_h   Host array containing the number of elements on
   *                               each GPU in the input key/value arrays.
   * @param[out]  output_keys_d    The sorted keys, stored in device arrays.
   *                               output_keys_d[i] is the array of keys on GPU i
   * @param[out]  output_length_h  Host array containing the number of elements on
   *                               each GPU in the output key/value arrays.
   * @param[in]   num_gpus         The number of GPUs
   *
   * @return   GDF_SUCCESS upon successful completion
   */
  template <typename Key_t, typename Value_t, typename Length_t>
  gdf_error sortKey(Key_t **input_keys_d,
                    Length_t *input_lengths_h,
                    Key_t **output_keys_d,
                    Length_t *output_lengths_h,
                    int num_gpus) {

    return GDF_SUCCESS;
  }
}
