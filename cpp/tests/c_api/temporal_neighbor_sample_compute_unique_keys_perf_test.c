/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h"

#include <cugraph_c/coo.h>
#include <cugraph_c/graph_generators.h>
#include <cugraph_c/sampling_algorithms.h>

#include <cuda_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  RMAT_SCALE       = 18,
  RMAT_EDGE_FACTOR = 16,
  BATCH_SIZE       = 8192,
  NUM_WARMUPS      = 10,
  NUM_ITERATIONS   = 50
};

typedef struct {
  cugraph_graph_t* graph;
  cugraph_type_erased_device_array_t* starts;
  cugraph_type_erased_device_array_view_t* starts_view;
  cugraph_type_erased_device_array_t* start_times;
  cugraph_type_erased_device_array_view_t* start_times_view;
  cugraph_type_erased_device_array_t* label_offsets;
  cugraph_type_erased_device_array_view_t* label_offsets_view;
} benchmark_context_t;

static double elapsed_ms(struct timespec start, struct timespec stop)
{
  return ((double)(stop.tv_sec - start.tv_sec) * 1000.0) +
         ((double)(stop.tv_nsec - start.tv_nsec) / 1000000.0);
}

static int compare_doubles(const void* lhs, const void* rhs)
{
  double a = *(const double*)lhs;
  double b = *(const double*)rhs;
  return (a > b) - (a < b);
}

static int compare_sizes(const void* lhs, const void* rhs)
{
  size_t a = *(const size_t*)lhs;
  size_t b = *(const size_t*)rhs;
  return (a > b) - (a < b);
}

static double median_double(double* values, size_t size)
{
  qsort(values, size, sizeof(double), compare_doubles);
  return (size % 2 == 0) ? (values[size / 2 - 1] + values[size / 2]) / 2.0 : values[size / 2];
}

static size_t median_size(size_t* values, size_t size)
{
  qsort(values, size, sizeof(size_t), compare_sizes);
  return values[size / 2];
}

static void free_benchmark_context(benchmark_context_t* context)
{
  cugraph_type_erased_device_array_view_free(context->label_offsets_view);
  cugraph_type_erased_device_array_view_free(context->start_times_view);
  cugraph_type_erased_device_array_view_free(context->starts_view);
  cugraph_type_erased_device_array_free(context->label_offsets);
  cugraph_type_erased_device_array_free(context->start_times);
  cugraph_type_erased_device_array_free(context->starts);
  cugraph_graph_free(context->graph);
  memset(context, 0, sizeof(*context));
}

static int create_benchmark_context(const cugraph_resource_handle_t* handle,
                                    benchmark_context_t* context)
{
  int test_ret_value                                       = 0;
  cugraph_error_code_t ret_code                            = CUGRAPH_SUCCESS;
  cugraph_error_t* error                                   = NULL;
  cugraph_rng_state_t* rng_state                           = NULL;
  cugraph_coo_t* coo                                       = NULL;
  cugraph_type_erased_device_array_view_t* src_view        = NULL;
  cugraph_type_erased_device_array_view_t* dst_view        = NULL;
  cugraph_type_erased_device_array_t* edge_times           = NULL;
  cugraph_type_erased_device_array_view_t* edge_times_view = NULL;
  cugraph_type_erased_device_array_view_t* seed_view       = NULL;
  size_t* label_offsets                                    = NULL;
  size_t num_edges = ((size_t)1 << RMAT_SCALE) * RMAT_EDGE_FACTOR;

  memset(context, 0, sizeof(*context));

  ret_code = cugraph_rng_state_create(handle, 42, &rng_state, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("RMAT RNG creation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }

  ret_code = cugraph_generate_rmat_edgelist(
    handle, rng_state, RMAT_SCALE, num_edges, 0.57, 0.19, 0.19, FALSE, TRUE, &coo, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("RMAT generation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }

  src_view = cugraph_coo_get_sources(coo);
  dst_view = cugraph_coo_get_destinations(coo);
  TEST_ASSERT(test_ret_value,
              cugraph_type_erased_device_array_view_size(src_view) >= BATCH_SIZE,
              "RMAT edge list is too small for the requested batch.");
  if (test_ret_value != 0) goto cleanup;

  ret_code = cugraph_type_erased_device_array_create(handle, num_edges, INT32, &edge_times, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("Edge-time allocation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }
  edge_times_view = cugraph_type_erased_device_array_view(edge_times);
  if (cudaMemset((void*)cugraph_type_erased_device_array_view_pointer(edge_times_view),
                 0,
                 num_edges * sizeof(int32_t)) != cudaSuccess) {
    printf("Edge-time initialization failed.\n");
    test_ret_value = 1;
    goto cleanup;
  }

  cugraph_graph_properties_t properties = {.is_symmetric = FALSE, .is_multigraph = TRUE};
  ret_code                              = cugraph_graph_create_with_times_sg(handle,
                                                &properties,
                                                NULL,
                                                src_view,
                                                dst_view,
                                                NULL,
                                                NULL,
                                                NULL,
                                                edge_times_view,
                                                NULL,
                                                FALSE,
                                                TRUE,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                &context->graph,
                                                &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("RMAT graph creation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }

  // Drawing seeds from the RMAT source distribution gives the first hop enough active
  // vertices to reproduce the large second-hop frontier in issue #5604.
  seed_view = cugraph_type_erased_device_array_view_create(
    (void*)cugraph_type_erased_device_array_view_pointer(src_view), BATCH_SIZE, INT32);
  ret_code =
    cugraph_type_erased_device_array_create_from_view(handle, seed_view, &context->starts, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("Start-vertex creation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }
  context->starts_view = cugraph_type_erased_device_array_view(context->starts);

  ret_code = cugraph_type_erased_device_array_create(
    handle, BATCH_SIZE, INT32, &context->start_times, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("Start-time allocation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }
  context->start_times_view = cugraph_type_erased_device_array_view(context->start_times);
  if (cudaMemset((void*)cugraph_type_erased_device_array_view_pointer(context->start_times_view),
                 0,
                 BATCH_SIZE * sizeof(int32_t)) != cudaSuccess) {
    printf("Start-time initialization failed.\n");
    test_ret_value = 1;
    goto cleanup;
  }

  label_offsets = (size_t*)malloc((BATCH_SIZE + 1) * sizeof(size_t));
  if (label_offsets == NULL) {
    printf("Label-offset allocation failed.\n");
    test_ret_value = 1;
    goto cleanup;
  }
  for (size_t i = 0; i <= BATCH_SIZE; ++i) {
    label_offsets[i] = i;
  }

  ret_code = cugraph_type_erased_device_array_create(
    handle, BATCH_SIZE + 1, SIZE_T, &context->label_offsets, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("Label-offset device allocation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }
  context->label_offsets_view = cugraph_type_erased_device_array_view(context->label_offsets);
  ret_code                    = cugraph_type_erased_device_array_view_copy_from_host(
    handle, context->label_offsets_view, (byte_t*)label_offsets, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("Label-offset copy failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }

cleanup:
  free(label_offsets);
  cugraph_type_erased_device_array_view_free(seed_view);
  cugraph_type_erased_device_array_view_free(edge_times_view);
  cugraph_type_erased_device_array_free(edge_times);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_coo_free(coo);
  cugraph_rng_state_free(rng_state);
  cugraph_error_free(error);

  if (test_ret_value != 0) free_benchmark_context(context);
  return test_ret_value;
}

static int run_temporal_sampling_benchmark(const cugraph_resource_handle_t* handle,
                                           const benchmark_context_t* context,
                                           int* fan_out,
                                           size_t fan_out_size)
{
  int test_ret_value                                  = 0;
  cugraph_error_code_t ret_code                       = CUGRAPH_SUCCESS;
  cugraph_error_t* error                              = NULL;
  cugraph_rng_state_t* rng_state                      = NULL;
  cugraph_sampling_options_t* options                 = NULL;
  cugraph_type_erased_host_array_view_t* fan_out_view = NULL;
  double durations[NUM_ITERATIONS];
  size_t result_sizes[NUM_ITERATIONS];

  ret_code = cugraph_rng_state_create(handle, 7, &rng_state, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("Sampling RNG creation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }

  ret_code = cugraph_sampling_options_create(&options, &error);
  if (ret_code != CUGRAPH_SUCCESS) {
    printf("Sampling-options creation failed: %s\n", cugraph_error_message(error));
    test_ret_value = 1;
    goto cleanup;
  }

  cugraph_sampling_set_with_replacement(options, FALSE);
  cugraph_sampling_set_return_hops(options, TRUE);
  cugraph_sampling_set_prior_sources_behavior(options, DEFAULT);
  cugraph_sampling_set_dedupe_sources(options, FALSE);
  cugraph_sampling_set_renumber_results(options, FALSE);
  cugraph_sampling_set_temporal_sampling_comparison(options, MONOTONICALLY_INCREASING);
  fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  for (size_t iteration = 0; iteration < NUM_WARMUPS + NUM_ITERATIONS; ++iteration) {
    cugraph_sample_result_t* result = NULL;
    struct timespec start;
    struct timespec stop;

    if (cudaDeviceSynchronize() != cudaSuccess) {
      printf("Pre-sampling device synchronization failed.\n");
      test_ret_value = 1;
      goto cleanup;
    }
    if (iteration >= NUM_WARMUPS) clock_gettime(CLOCK_MONOTONIC, &start);

    ret_code = cugraph_homogeneous_uniform_temporal_neighbor_sample(handle,
                                                                    rng_state,
                                                                    context->graph,
                                                                    "edge_start_time",
                                                                    context->starts_view,
                                                                    context->start_times_view,
                                                                    context->label_offsets_view,
                                                                    fan_out_view,
                                                                    options,
                                                                    FALSE,
                                                                    &result,
                                                                    &error);
    if (ret_code != CUGRAPH_SUCCESS) {
      printf("Temporal neighbor sampling failed: %s\n", cugraph_error_message(error));
      cugraph_sample_result_free(result);
      test_ret_value = 1;
      goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      printf("Post-sampling device synchronization failed.\n");
      cugraph_sample_result_free(result);
      test_ret_value = 1;
      goto cleanup;
    }

    if (iteration >= NUM_WARMUPS) {
      cugraph_type_erased_device_array_view_t* sources = cugraph_sample_result_get_sources(result);
      clock_gettime(CLOCK_MONOTONIC, &stop);
      durations[iteration - NUM_WARMUPS]    = elapsed_ms(start, stop);
      result_sizes[iteration - NUM_WARMUPS] = cugraph_type_erased_device_array_view_size(sources);
      cugraph_type_erased_device_array_view_free(sources);
    }
    cugraph_sample_result_free(result);
  }

  {
    double median_ms   = median_double(durations, NUM_ITERATIONS);
    size_t median_rows = median_size(result_sizes, NUM_ITERATIONS);
    printf(
      "compute_unique_keys temporal sampling: batch=%d fanout=[16%s] "
      "warmups=%d iterations=%d median=%.3f ms median_rows=%zu\n",
      BATCH_SIZE,
      fan_out_size == 2 ? ",16" : "",
      NUM_WARMUPS,
      NUM_ITERATIONS,
      median_ms,
      median_rows);
    TEST_ASSERT(test_ret_value, median_rows > 0, "Temporal sampling returned no edges.");
  }

cleanup:
  cugraph_type_erased_host_array_view_free(fan_out_view);
  cugraph_sampling_options_free(options);
  cugraph_rng_state_free(rng_state);
  cugraph_error_free(error);
  return test_ret_value;
}

int test_temporal_neighbor_sampling_compute_unique_keys_performance(
  const cugraph_resource_handle_t* handle)
{
  benchmark_context_t context;
  int one_hop_fan_out[] = {16};
  int two_hop_fan_out[] = {16, 16};
  int test_ret_value    = create_benchmark_context(handle, &context);

  if (test_ret_value == 0)
    test_ret_value |= run_temporal_sampling_benchmark(handle, &context, one_hop_fan_out, 1);
  if (test_ret_value == 0)
    test_ret_value |= run_temporal_sampling_benchmark(handle, &context, two_hop_fan_out, 2);

  free_benchmark_context(&context);
  return test_ret_value;
}

int main(void)
{
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  int result =
    RUN_TEST_NEW(test_temporal_neighbor_sampling_compute_unique_keys_performance, handle);
  cugraph_free_resource_handle(handle);
  return result;
}
