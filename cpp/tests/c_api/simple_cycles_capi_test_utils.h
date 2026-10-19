/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "c_test_utils.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t simple_cycles_vertex_t;

#define SIMPLE_CYCLES_CAPI_MAX_CYCLES    16
#define SIMPLE_CYCLES_CAPI_MAX_CYCLE_LEN 16

typedef struct {
  simple_cycles_vertex_t vertices[SIMPLE_CYCLES_CAPI_MAX_CYCLE_LEN];
  size_t length;
} simple_cycles_capi_cycle_t;

typedef struct {
  simple_cycles_capi_cycle_t cycles[SIMPLE_CYCLES_CAPI_MAX_CYCLES];
  size_t num_cycles;
} simple_cycles_capi_cycle_list_t;

static int simple_cycles_capi_compare_vertex_sequences(const simple_cycles_vertex_t* a,
                                                       size_t len_a,
                                                       const simple_cycles_vertex_t* b,
                                                       size_t len_b)
{
  if (len_a < len_b) { return -1; }
  if (len_a > len_b) { return 1; }
  for (size_t i = 0; i < len_a; ++i) {
    if (a[i] < b[i]) { return -1; }
    if (a[i] > b[i]) { return 1; }
  }
  return 0;
}

static void simple_cycles_capi_canonicalize_cycle(simple_cycles_vertex_t* cycle_vertices,
                                                  size_t cycle_length,
                                                  simple_cycles_capi_cycle_t* out)
{
  out->length = cycle_length;
  if (cycle_length == 0) { return; }
  if (cycle_length > SIMPLE_CYCLES_CAPI_MAX_CYCLE_LEN) { return; }

  size_t min_index = 0;
  for (size_t i = 1; i < cycle_length; ++i) {
    if (cycle_vertices[i] < cycle_vertices[min_index]) { min_index = i; }
  }

  for (size_t i = 0; i < cycle_length; ++i) {
    out->vertices[i] = cycle_vertices[(min_index + i) % cycle_length];
  }
}

static int simple_cycles_capi_compare_cycles_qsort(const void* a, const void* b)
{
  const simple_cycles_capi_cycle_t* cycle_a = (const simple_cycles_capi_cycle_t*)a;
  const simple_cycles_capi_cycle_t* cycle_b = (const simple_cycles_capi_cycle_t*)b;
  return simple_cycles_capi_compare_vertex_sequences(
    cycle_a->vertices, cycle_a->length, cycle_b->vertices, cycle_b->length);
}

static void simple_cycles_capi_build_cycle_list(const simple_cycles_vertex_t* flat_vertices,
                                                const size_t* offsets,
                                                size_t num_cycles,
                                                simple_cycles_capi_cycle_list_t* out)
{
  out->num_cycles = num_cycles;
  for (size_t i = 0; i < num_cycles; ++i) {
    size_t begin = offsets[i];
    size_t end   = offsets[i + 1];
    simple_cycles_vertex_t cycle_buffer[SIMPLE_CYCLES_CAPI_MAX_CYCLE_LEN];
    size_t cycle_length = end - begin;
    if (cycle_length > SIMPLE_CYCLES_CAPI_MAX_CYCLE_LEN) {
      cycle_length = SIMPLE_CYCLES_CAPI_MAX_CYCLE_LEN;
    }
    memcpy(cycle_buffer, flat_vertices + begin, cycle_length * sizeof(simple_cycles_vertex_t));
    simple_cycles_capi_canonicalize_cycle(cycle_buffer, end - begin, &out->cycles[i]);
  }
  qsort(out->cycles,
        out->num_cycles,
        sizeof(simple_cycles_capi_cycle_t),
        simple_cycles_capi_compare_cycles_qsort);
}

static int simple_cycles_capi_validate_cycle_lists(const simple_cycles_capi_cycle_list_t* actual,
                                                   const simple_cycles_capi_cycle_list_t* expected)
{
  if (actual->num_cycles != expected->num_cycles) { return 0; }
  for (size_t i = 0; i < actual->num_cycles; ++i) {
    if (simple_cycles_capi_compare_vertex_sequences(actual->cycles[i].vertices,
                                                    actual->cycles[i].length,
                                                    expected->cycles[i].vertices,
                                                    expected->cycles[i].length) != 0) {
      return 0;
    }
  }
  return 1;
}

static int simple_cycles_capi_validate_flat_cycles(const simple_cycles_vertex_t* actual_vertices,
                                                   const size_t* actual_offsets,
                                                   size_t actual_num_cycles,
                                                   const simple_cycles_vertex_t* expected_vertices,
                                                   const size_t* expected_offsets,
                                                   size_t expected_num_cycles)
{
  simple_cycles_capi_cycle_list_t actual_list   = {0};
  simple_cycles_capi_cycle_list_t expected_list = {0};

  simple_cycles_capi_build_cycle_list(
    actual_vertices, actual_offsets, actual_num_cycles, &actual_list);
  simple_cycles_capi_build_cycle_list(
    expected_vertices, expected_offsets, expected_num_cycles, &expected_list);

  return simple_cycles_capi_validate_cycle_lists(&actual_list, &expected_list);
}
