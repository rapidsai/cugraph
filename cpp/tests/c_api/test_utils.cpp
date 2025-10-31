/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../sampling/detail/nbr_sampling_validate.hpp"
#include "c_api/array.hpp"
#include "c_api/c_test_utils.h"
#include "c_api/resource_handle.hpp"
#include "c_api/sampling_common.hpp"

#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/span.hpp>
#include <raft/util/cudart_utils.hpp>

#include <math.h>

namespace {
template <typename T>
raft::device_span<T const> make_span(cugraph_type_erased_device_array_view_t const* view)
{
  auto internal_view =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(view);
  return raft::device_span<T const>{internal_view->as_type<T const>(), internal_view->size_};
}

template <typename vertex_t>
int vertex_id_compare_function(const void* a, const void* b)
{
  return (*((vertex_t*)a) < *((vertex_t*)b)) ? -1 : (*((vertex_t*)a) > *((vertex_t*)b)) ? 1 : 0;
}

}  // namespace

extern "C" int nearlyEqual(float a, float b, float epsilon)
{
  // FIXME:  There is a better test than this,
  //   perhaps use the gtest comparison for consistency
  //   with C++ and wrap it in a C wrapper.
  return (fabsf(a - b) <= (((fabsf(a) < fabsf(b)) ? fabs(b) : fabs(a)) * epsilon));
}

extern "C" int nearlyEqualDouble(double a, double b, double epsilon)
{
  // FIXME:  There is a better test than this,
  //   perhaps use the gtest comparison for consistency
  //   with C++ and wrap it in a C wrapper.
  return (fabsf(a - b) <= (((fabsf(a) < fabsf(b)) ? fabs(b) : fabs(a)) * epsilon));
}

/*
 * Simple check of creating a graph from a COO on device memory.
 */
extern "C" int create_test_graph(const cugraph_resource_handle_t* p_handle,
                                 int32_t* h_src,
                                 int32_t* h_dst,
                                 float* h_wgt,
                                 size_t num_edges,
                                 bool_t store_transposed,
                                 bool_t renumber,
                                 bool_t is_symmetric,
                                 cugraph_graph_t** p_graph,
                                 cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, weight_tid, &wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, wgt_view, (byte_t*)h_wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_graph_create_with_times_sg(p_handle,
                                                &properties,
                                                nullptr,
                                                src_view,
                                                dst_view,
                                                wgt_view,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                store_transposed,
                                                renumber,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                p_graph,
                                                ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

extern "C" int create_test_graph_double(const cugraph_resource_handle_t* p_handle,
                                        int32_t* h_src,
                                        int32_t* h_dst,
                                        double* h_wgt,
                                        size_t num_edges,
                                        bool_t store_transposed,
                                        bool_t renumber,
                                        bool_t is_symmetric,
                                        cugraph_graph_t** p_graph,
                                        cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT64;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, weight_tid, &wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, wgt_view, (byte_t*)h_wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_graph_create_with_times_sg(p_handle,
                                                &properties,
                                                nullptr,
                                                src_view,
                                                dst_view,
                                                wgt_view,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                store_transposed,
                                                renumber,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                p_graph,
                                                ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

/*
 * Runs the function pointed to by "test" and returns the return code.  Also
 * prints reporting info (using "test_name"): pass/fail and run time, to stdout.
 *
 * Intended to be used by the RUN_TEST macro.
 */
extern "C" int run_sg_test(int (*test)(), const char* test_name)
{
  int ret_val = 0;
  time_t start_time, end_time;

  printf("RUNNING: %s...", test_name);
  fflush(stdout);

  time(&start_time);

  ret_val = test();

  time(&end_time);

  printf("done (%f seconds).", difftime(end_time, start_time));
  if (ret_val == 0) {
    printf(" - passed\n");
  } else {
    printf(" - FAILED\n");
  }
  fflush(stdout);

  return ret_val;
}

extern "C" int run_sg_test_new(int (*test)(const cugraph_resource_handle_t*),
                               const char* test_name,
                               const cugraph_resource_handle_t* handle)
{
  int ret_val = 0;
  time_t start_time, end_time;

  printf("RUNNING: %s...", test_name);
  fflush(stdout);

  time(&start_time);

  ret_val = test(handle);

  time(&end_time);

  printf("done (%f seconds).", difftime(end_time, start_time));
  if (ret_val == 0) {
    printf(" - passed\n");
  } else {
    printf(" - FAILED\n");
  }
  fflush(stdout);

  return ret_val;
}

int create_sg_test_graph(const cugraph_resource_handle_t* handle,
                         cugraph_data_type_id_t vertex_tid,
                         cugraph_data_type_id_t edge_tid,
                         void* h_src,
                         void* h_dst,
                         cugraph_data_type_id_t weight_tid,
                         void* h_wgt,
                         cugraph_data_type_id_t edge_type_tid,
                         void* h_edge_type,
                         cugraph_data_type_id_t edge_id_tid,
                         void* h_edge_id,
                         cugraph_data_type_id_t edge_time_tid,
                         void* h_edge_start_times,
                         void* h_edge_end_times,
                         size_t num_edges,
                         bool_t store_transposed,
                         bool_t renumber,
                         bool_t is_symmetric,
                         bool_t is_multigraph,
                         cugraph_graph_t** graph,
                         cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = is_multigraph;

  cugraph_type_erased_device_array_t* src              = NULL;
  cugraph_type_erased_device_array_t* dst              = NULL;
  cugraph_type_erased_device_array_t* wgt              = NULL;
  cugraph_type_erased_device_array_t* edge_type        = NULL;
  cugraph_type_erased_device_array_t* edge_id          = NULL;
  cugraph_type_erased_device_array_t* edge_start_times = NULL;
  cugraph_type_erased_device_array_t* edge_end_times   = NULL;

  cugraph_type_erased_device_array_view_t* src_view              = NULL;
  cugraph_type_erased_device_array_view_t* dst_view              = NULL;
  cugraph_type_erased_device_array_view_t* wgt_view              = NULL;
  cugraph_type_erased_device_array_view_t* edge_type_view        = NULL;
  cugraph_type_erased_device_array_view_t* edge_id_view          = NULL;
  cugraph_type_erased_device_array_view_t* edge_start_times_view = NULL;
  cugraph_type_erased_device_array_view_t* edge_end_times_view   = NULL;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  if (h_wgt != NULL) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

    wgt_view = cugraph_type_erased_device_array_view(wgt);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, wgt_view, (byte_t*)h_wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");
  }

  if (h_edge_type != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_edges, edge_type_tid, &edge_type, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_type create failed.");

    edge_type_view = cugraph_type_erased_device_array_view(edge_type);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, edge_type_view, (byte_t*)h_edge_type, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_type copy_from_host failed.");
  }

  if (h_edge_id != NULL) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, edge_id_tid, &edge_id, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_id create failed.");

    edge_id_view = cugraph_type_erased_device_array_view(edge_id);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, edge_id_view, (byte_t*)h_edge_id, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_id copy_from_host failed.");
  }

  if (h_edge_start_times != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_edges, edge_time_tid, &edge_start_times, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_start_times create failed.");

    edge_start_times_view = cugraph_type_erased_device_array_view(edge_start_times);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, edge_start_times_view, (byte_t*)h_edge_start_times, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_type copy_from_host failed.");
  }

  if (h_edge_end_times != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_edges, edge_time_tid, &edge_end_times, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_end_times create failed.");

    edge_end_times_view = cugraph_type_erased_device_array_view(edge_end_times);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, edge_end_times_view, (byte_t*)h_edge_end_times, ret_error);
    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_end_times copy_from_host failed.");
  }

  ret_code = cugraph_graph_create_with_times_sg(handle,
                                                &properties,
                                                nullptr,
                                                src_view,
                                                dst_view,
                                                wgt_view,
                                                edge_id_view,
                                                edge_type_view,
                                                edge_start_times_view,
                                                edge_end_times_view,
                                                store_transposed,
                                                renumber,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                graph,
                                                ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  if (edge_end_times != NULL) {
    cugraph_type_erased_device_array_view_free(edge_end_times_view);
    cugraph_type_erased_device_array_free(edge_end_times);
  }

  if (edge_start_times != NULL) {
    cugraph_type_erased_device_array_view_free(edge_start_times_view);
    cugraph_type_erased_device_array_free(edge_start_times);
  }

  if (edge_id != NULL) {
    cugraph_type_erased_device_array_view_free(edge_id_view);
    cugraph_type_erased_device_array_free(edge_id);
  }

  if (edge_type != NULL) {
    cugraph_type_erased_device_array_view_free(edge_type_view);
    cugraph_type_erased_device_array_free(edge_type);
  }

  if (wgt != NULL) {
    cugraph_type_erased_device_array_view_free(wgt_view);
    cugraph_type_erased_device_array_free(wgt);
  }

  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

extern "C" size_t cugraph_size_t_allreduce(const cugraph_resource_handle_t* handle, size_t value)
{
  auto internal_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  return cugraph::host_scalar_allreduce(internal_handle->handle_->get_comms(),
                                        value,
                                        raft::comms::op_t::SUM,
                                        internal_handle->handle_->get_stream());
}

extern "C" int validate_sample_result(const cugraph_resource_handle_t* handle,
                                      const cugraph_sample_result_t* result,
                                      int32_t* h_src,
                                      int32_t* h_dst,
                                      float* h_wgt,
                                      int32_t* h_edge_ids,
                                      int32_t* h_edge_types,
                                      int32_t* h_edge_start_times,
                                      int32_t* h_edge_end_times,
                                      size_t num_vertices,
                                      size_t num_edges,
                                      int32_t* h_start_vertices,
                                      size_t num_start_vertices,
                                      size_t* h_start_label_offsets,
                                      size_t num_start_label_offsets,
                                      int* fan_out,
                                      size_t fan_out_size,
                                      cugraph_sampling_options_t* sampling_options,
                                      bool validate_edge_times)
{
  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  auto internal_sampling_options =
    reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(sampling_options);
  auto prior_sources_behavior = internal_sampling_options->prior_sources_behavior_;
  bool renumber_results       = (internal_sampling_options->renumber_results_ == TRUE);
  bool dedupe_sources         = (internal_sampling_options->dedupe_sources_ == TRUE);
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison =
    internal_sampling_options->temporal_sampling_comparison_;

  using vertex_t    = int32_t;
  using weight_t    = float;
  using edge_t      = int32_t;
  using edge_time_t = int32_t;

  edge_time_t const MAX_EDGE_TIME = INT32_MAX;

  cugraph_type_erased_device_array_view_t* result_renumber_map_offsets   = nullptr;
  cugraph_type_erased_device_array_view_t* result_renumber_map           = nullptr;
  cugraph_type_erased_device_array_view_t* result_srcs                   = nullptr;
  cugraph_type_erased_device_array_view_t* result_dsts                   = nullptr;
  cugraph_type_erased_device_array_view_t* result_weights                = nullptr;
  cugraph_type_erased_device_array_view_t* result_edge_ids               = nullptr;
  cugraph_type_erased_device_array_view_t* result_edge_types             = nullptr;
  cugraph_type_erased_device_array_view_t* result_edge_start_times       = nullptr;
  cugraph_type_erased_device_array_view_t* result_edge_end_times         = nullptr;
  cugraph_type_erased_device_array_view_t* result_hops                   = nullptr;
  cugraph_type_erased_device_array_view_t* result_label_type_hop_offsets = nullptr;
  cugraph_type_erased_device_array_view_t* result_label_hop_offsets      = nullptr;
  cugraph_type_erased_device_array_view_t* result_labels                 = nullptr;

  result_renumber_map_offsets   = cugraph_sample_result_get_renumber_map_offsets(result);
  result_renumber_map           = cugraph_sample_result_get_renumber_map(result);
  result_srcs                   = cugraph_sample_result_get_majors(result);
  result_dsts                   = cugraph_sample_result_get_minors(result);
  result_weights                = cugraph_sample_result_get_edge_weight(result);
  result_edge_ids               = cugraph_sample_result_get_edge_id(result);
  result_edge_types             = cugraph_sample_result_get_edge_type(result);
  result_edge_start_times       = cugraph_sample_result_get_edge_start_time(result);
  result_edge_end_times         = cugraph_sample_result_get_edge_end_time(result);
  result_label_hop_offsets      = cugraph_sample_result_get_label_hop_offsets(result);
  result_label_type_hop_offsets = cugraph_sample_result_get_label_type_hop_offsets(result);
  result_labels                 = cugraph_sample_result_get_start_labels(result);
  result_hops                   = cugraph_sample_result_get_hop(result);

  size_t result_renumber_map_offsets_size = 0;
  size_t result_renumber_map_size         = 0;
  size_t result_size                      = cugraph_type_erased_device_array_view_size(result_srcs);

  if (result_renumber_map_offsets != NULL) {
    result_renumber_map_offsets_size =
      cugraph_type_erased_device_array_view_size(result_renumber_map_offsets);
  }
  if (result_renumber_map != NULL) {
    result_renumber_map_size = result_renumber_map != NULL
                                 ? cugraph_type_erased_device_array_view_size(result_renumber_map)
                                 : 0;
  }

  auto raft_handle =
    *(reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_);

  auto result_srcs_span       = make_span<vertex_t>(result_srcs);
  auto result_dsts_span       = make_span<vertex_t>(result_dsts);
  auto result_weights_span    = (result_weights != NULL)
                                  ? std::make_optional(make_span<weight_t>(result_weights))
                                  : std::nullopt;
  auto result_edge_ids_span   = (result_edge_ids != NULL)
                                  ? std::make_optional(make_span<edge_t>(result_edge_ids))
                                  : std::nullopt;
  auto result_edge_types_span = (result_edge_types != NULL)
                                  ? std::make_optional(make_span<edge_t>(result_edge_types))
                                  : std::nullopt;
  auto result_edge_start_times_span =
    (result_edge_start_times != NULL)
      ? std::make_optional(make_span<edge_time_t>(result_edge_start_times))
      : std::nullopt;
  auto result_edge_end_times_span =
    (result_edge_end_times != NULL)
      ? std::make_optional(make_span<edge_time_t>(result_edge_end_times))
      : std::nullopt;
  auto result_label_span =
    (result_labels != NULL) ? std::make_optional(make_span<int32_t>(result_labels)) : std::nullopt;
  auto result_label_type_hop_offsets_span =
    (result_label_type_hop_offsets != NULL)
      ? std::make_optional(make_span<size_t>(result_label_type_hop_offsets))
      : std::nullopt;
  auto result_label_hop_offsets_span =
    (result_label_hop_offsets != NULL)
      ? std::make_optional(make_span<size_t>(result_label_hop_offsets))
      : std::nullopt;

  rmm::device_uvector<vertex_t> graph_srcs(num_edges, raft_handle.get_stream());
  rmm::device_uvector<vertex_t> graph_dsts(num_edges, raft_handle.get_stream());
  auto graph_weights =
    (result_weights != NULL)
      ? std::make_optional(rmm::device_uvector<weight_t>(num_edges, raft_handle.get_stream()))
      : std::nullopt;
  auto graph_edge_ids =
    (result_edge_ids != NULL)
      ? std::make_optional(rmm::device_uvector<edge_t>(num_edges, raft_handle.get_stream()))
      : std::nullopt;
  auto graph_edge_types =
    (result_edge_types != NULL)
      ? std::make_optional(rmm::device_uvector<edge_t>(num_edges, raft_handle.get_stream()))
      : std::nullopt;
  auto graph_edge_start_times =
    (result_edge_start_times != NULL)
      ? std::make_optional(rmm::device_uvector<edge_time_t>(num_edges, raft_handle.get_stream()))
      : std::nullopt;
  auto graph_edge_end_times =
    (result_edge_end_times != NULL)
      ? std::make_optional(rmm::device_uvector<edge_time_t>(num_edges, raft_handle.get_stream()))
      : std::nullopt;

  raft::update_device(graph_srcs.data(), h_src, num_edges, raft_handle.get_stream());
  raft::update_device(graph_dsts.data(), h_dst, num_edges, raft_handle.get_stream());
  if (graph_weights) {
    raft::update_device(graph_weights->data(), h_wgt, num_edges, raft_handle.get_stream());
  }
  if (graph_edge_ids) {
    raft::update_device(graph_edge_ids->data(), h_edge_ids, num_edges, raft_handle.get_stream());
  }
  if (graph_edge_types) {
    raft::update_device(
      graph_edge_types->data(), h_edge_types, num_edges, raft_handle.get_stream());
  }
  if (graph_edge_start_times) {
    raft::update_device(
      graph_edge_start_times->data(), h_edge_start_times, num_edges, raft_handle.get_stream());
  }
  if (graph_edge_end_times) {
    raft::update_device(
      graph_edge_end_times->data(), h_edge_end_times, num_edges, raft_handle.get_stream());
  }

  rmm::device_uvector<vertex_t> renumbered_srcs(result_srcs_span.size(), raft_handle.get_stream());
  rmm::device_uvector<vertex_t> renumbered_dsts(result_dsts_span.size(), raft_handle.get_stream());

  raft::copy(renumbered_srcs.data(),
             result_srcs_span.data(),
             result_srcs_span.size(),
             raft_handle.get_stream());
  raft::copy(renumbered_dsts.data(),
             result_dsts_span.data(),
             result_dsts_span.size(),
             raft_handle.get_stream());

  int32_t h_result_labels[result_size];
  edge_time_t h_result_edge_start_times[result_size];
  edge_time_t h_result_edge_end_times[result_size];

  if (result_label_span) {
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_labels, result_labels, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "result_label copy_to_host failed.");
  }

  int32_t h_result_hops[result_size];
  if (result_hops != NULL) {
    raft::update_host(
      h_result_hops,
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(result_hops)
        ->as_type<int32_t const>(),
      result_size,
      raft_handle.get_stream());
  } else if (result_label_type_hop_offsets != nullptr) {
    size_t label_type_hop_offsets_size =
      cugraph_type_erased_device_array_view_size(result_label_type_hop_offsets);
    size_t h_result_label_type_hop_offsets[label_type_hop_offsets_size];
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_label_type_hop_offsets, result_label_type_hop_offsets, &ret_error);
    TEST_ASSERT(test_ret_value,
                ret_code == CUGRAPH_SUCCESS,
                "result_label_type_hop_offsets copy_to_host failed.");

    size_t local_hop_size = h_result_label_type_hop_offsets[label_type_hop_offsets_size - 1];
    int32_t hop           = 0;
    for (size_t i = 0; i < label_type_hop_offsets_size - 1; ++i) {
      for (size_t j = h_result_label_type_hop_offsets[i];
           j < h_result_label_type_hop_offsets[i + 1];
           ++j) {
        h_result_hops[j] = hop;
      }
      hop = (hop + 1) % fan_out_size;
    }
  } else if (result_label_hop_offsets != nullptr) {
    size_t label_hop_offsets_size =
      cugraph_type_erased_device_array_view_size(result_label_hop_offsets);
    size_t h_result_label_hop_offsets[label_hop_offsets_size];
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_label_hop_offsets, result_label_hop_offsets, &ret_error);
    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "result_label_hop_offsets copy_to_host failed.");

    size_t local_hop_size = h_result_label_hop_offsets[label_hop_offsets_size - 1];
    int32_t hop           = 0;
    for (size_t i = 0; i < label_hop_offsets_size - 1; ++i) {
      for (size_t j = h_result_label_hop_offsets[i]; j < h_result_label_hop_offsets[i + 1]; ++j) {
        h_result_hops[j] = hop;
      }
      hop = (hop + 1) % fan_out_size;
    }
  } else {
    std::fill(h_result_hops, h_result_hops + result_size, 0);
  }
  if (result_renumber_map_offsets != NULL) {
    size_t h_result_renumber_map_offsets[result_renumber_map_offsets_size];
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_renumber_map_offsets, result_renumber_map_offsets, &ret_error);
    TEST_ASSERT(test_ret_value,
                ret_code == CUGRAPH_SUCCESS,
                "result_renumber_map_offsets copy_to_host failed.");

    // Renumber all of the results by label
    if (result_label_type_hop_offsets_span) {
      size_t h_result_label_type_hop_offsets[result_label_type_hop_offsets_span->size()];
      ret_code =
        cugraph_type_erased_device_array_view_copy_to_host(handle,
                                                           (byte_t*)h_result_label_type_hop_offsets,
                                                           result_label_type_hop_offsets,
                                                           &ret_error);
      TEST_ASSERT(test_ret_value,
                  ret_code == CUGRAPH_SUCCESS,
                  "result_label_type_hop_offsets copy_to_host failed.");

      for (size_t i = 0; i < (result_label_type_hop_offsets_span->size() - 1); ++i) {
        if (h_result_label_type_hop_offsets[i] < result_size) {
          int32_t label_id = h_result_labels[h_result_label_type_hop_offsets[i]];

          cugraph::unrenumber_local_int_vertices(
            raft_handle,
            renumbered_srcs.data() + h_result_label_type_hop_offsets[i],
            h_result_label_type_hop_offsets[i + 1] - h_result_label_type_hop_offsets[i],
            reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
              result_renumber_map)
                ->as_type<vertex_t const>() +
              h_result_renumber_map_offsets[label_id],
            vertex_t{0},
            static_cast<vertex_t>(h_result_renumber_map_offsets[label_id + 1] -
                                  h_result_renumber_map_offsets[label_id]),
            false);

          cugraph::unrenumber_local_int_vertices(
            raft_handle,
            renumbered_dsts.data() + h_result_label_type_hop_offsets[i],
            h_result_label_type_hop_offsets[i + 1] - h_result_label_type_hop_offsets[i],
            reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
              result_renumber_map)
                ->as_type<vertex_t const>() +
              h_result_renumber_map_offsets[label_id],
            vertex_t{0},
            static_cast<vertex_t>(h_result_renumber_map_offsets[label_id + 1] -
                                  h_result_renumber_map_offsets[label_id]),
            false);
        }
      }
    } else if (result_label_hop_offsets_span) {
      size_t h_result_label_hop_offsets[result_label_hop_offsets_span->size()];
      raft::update_host(h_result_label_hop_offsets,
                        result_label_hop_offsets_span->data(),
                        result_label_hop_offsets_span->size(),
                        raft_handle.get_stream());

      for (size_t i = 0; i < (result_label_hop_offsets_span->size() - 1); ++i) {
        if (h_result_label_hop_offsets[i] < result_size) {
          int32_t label_id = h_result_labels[h_result_label_hop_offsets[i]];

          cugraph::unrenumber_local_int_vertices(
            raft_handle,
            renumbered_srcs.data() + h_result_label_hop_offsets[i],
            h_result_label_hop_offsets[i + 1] - h_result_label_hop_offsets[i],
            reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
              result_renumber_map)
                ->as_type<vertex_t const>() +
              h_result_renumber_map_offsets[label_id],
            vertex_t{0},
            static_cast<vertex_t>(h_result_renumber_map_offsets[label_id + 1] -
                                  h_result_renumber_map_offsets[label_id]),
            false);

          cugraph::unrenumber_local_int_vertices(
            raft_handle,
            renumbered_dsts.data() + h_result_label_hop_offsets[i],
            h_result_label_hop_offsets[i + 1] - h_result_label_hop_offsets[i],
            reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
              result_renumber_map)
                ->as_type<vertex_t const>() +
              h_result_renumber_map_offsets[label_id],
            vertex_t{0},
            static_cast<vertex_t>(h_result_renumber_map_offsets[label_id + 1] -
                                  h_result_renumber_map_offsets[label_id]),
            false);
        }
      }
    }
  }

  if (result_edge_start_times != NULL) {
    raft::update_host(
      h_result_edge_start_times,
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        result_edge_start_times)
        ->as_type<edge_time_t const>(),
      result_size,
      raft_handle.get_stream());
  }
  if (result_edge_end_times != NULL) {
    raft::update_host(
      h_result_edge_end_times,
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        result_edge_end_times)
        ->as_type<edge_time_t const>(),
      result_size,
      raft_handle.get_stream());
  }

  TEST_ASSERT(test_ret_value,
              cugraph::test::validate_extracted_graph_is_subgraph(
                raft_handle,
                raft::device_span<vertex_t const>{graph_srcs.data(), graph_srcs.size()},
                raft::device_span<vertex_t const>{graph_dsts.data(), graph_dsts.size()},
                (graph_weights) ? std::make_optional(raft::device_span<weight_t const>{
                                    graph_weights->data(), graph_weights->size()})
                                : std::nullopt,
                raft::device_span<vertex_t const>{renumbered_srcs.data(), renumbered_srcs.size()},
                raft::device_span<vertex_t const>{renumbered_dsts.data(), renumbered_dsts.size()},
                result_weights_span ? std::make_optional(raft::device_span<weight_t const>{
                                        result_weights_span->data(), result_weights_span->size()})
                                    : std::nullopt),
              "extracted graph is not a subgraph of the original graph");

  //
  // For the sampling result to make sense, all sources in hop 0 must be in the seeds,
  // all sources in hop 1 must be a result from hop 0, etc.
  //
  vertex_t check_v1[result_size];
  vertex_t check_v2[result_size];
  vertex_t* check_sources      = check_v1;
  vertex_t* check_destinations = check_v2;

  size_t degree[num_vertices];
  for (size_t i = 0; i < num_vertices; ++i)
    degree[i] = 0;

  for (size_t i = 0; i < num_edges; ++i) {
    degree[h_src[i]]++;
  }

  int32_t h_result_srcs[result_size];
  int32_t h_result_dsts[result_size];

  raft::update_host(h_result_srcs, renumbered_srcs.data(), result_size, raft_handle.get_stream());
  raft::update_host(h_result_dsts, renumbered_dsts.data(), result_size, raft_handle.get_stream());

  for (int label_id = 0;
       label_id < (h_start_label_offsets != NULL ? (num_start_label_offsets - 1) : 1);
       ++label_id) {
    size_t sources_size      = 0;
    size_t destinations_size = 0;

    // Fill sources with the input sources
    if (h_start_label_offsets != NULL) {
      for (size_t i = h_start_label_offsets[label_id]; i < h_start_label_offsets[label_id + 1];
           ++i) {
        check_sources[sources_size] = h_start_vertices[i];
        ++sources_size;
      }
    } else {
      for (size_t i = 0; i < num_start_vertices; ++i) {
        check_sources[sources_size] = h_start_vertices[i];
        ++sources_size;
      }
    }

    if (renumber_results) {
      size_t num_vertex_ids = 0;
      vertex_t vertex_ids[2 * result_size];

      int32_t h_unrenumbered_result_srcs[result_size];
      int32_t h_unrenumbered_result_dsts[result_size];

      raft::update_host(h_unrenumbered_result_srcs,
                        result_srcs_span.data(),
                        result_srcs_span.size(),
                        raft_handle.get_stream());
      raft::update_host(h_unrenumbered_result_dsts,
                        result_dsts_span.data(),
                        result_dsts_span.size(),
                        raft_handle.get_stream());

      for (size_t i = 0; i < result_size; ++i) {
        if (h_result_labels[i] == label_id) {
          vertex_ids[num_vertex_ids]     = h_unrenumbered_result_srcs[i];
          vertex_ids[num_vertex_ids + 1] = h_unrenumbered_result_dsts[i];
          num_vertex_ids += 2;
        }
      }

      qsort(vertex_ids, num_vertex_ids, sizeof(vertex_t), vertex_id_compare_function<vertex_t>);

      vertex_t current_v = 0;
      for (size_t i = 0; (i < num_vertex_ids) && (test_ret_value == 0); ++i) {
        if (vertex_ids[i] == current_v)
          ++current_v;
        else {
          TEST_ASSERT(test_ret_value,
                      vertex_ids[i] == (current_v - 1),
                      "vertices are not properly renumbered");
        }
      }
    }

    if ((result_hops != NULL) || (result_label_type_hop_offsets != NULL) ||
        (result_label_hop_offsets != NULL)) {
      // Can't check this if we don't have the result_hop, result_label_type_hop_offsets, or
      // result_label_hop_offsets
      for (int hop = 0; hop < fan_out_size; ++hop) {
        if (prior_sources_behavior == cugraph::prior_sources_behavior_t::CARRY_OVER) {
          destinations_size = sources_size;
          for (size_t i = 0; i < sources_size; ++i) {
            check_destinations[i] = check_sources[i];
          }
        }

        for (size_t i = 0; i < result_size; ++i) {
          if (h_result_labels[i] == label_id) {
            if (h_result_hops[i] == hop) {
              bool found = false;
              for (size_t j = 0; (!found) && (j < sources_size); ++j) {
                found = (h_result_srcs[i] == check_sources[j]);
              }

              TEST_ASSERT(test_ret_value,
                          found,
                          "encountered source vertex that was not part of previous frontier");

              if (prior_sources_behavior == cugraph::prior_sources_behavior_t::CARRY_OVER) {
                // Make sure destination isn't already in the source list
                bool found = false;
                for (size_t j = 0; (!found) && (j < destinations_size); ++j) {
                  found = (h_result_dsts[i] == check_destinations[j]);
                }

                if (!found) {
                  check_destinations[destinations_size] = h_result_dsts[i];
                  ++destinations_size;
                }
              } else {
                check_destinations[destinations_size] = h_result_dsts[i];
                ++destinations_size;
              }

              if (prior_sources_behavior == cugraph::prior_sources_behavior_t::EXCLUDE) {
                // Make sure vertex v only appears as source in the first hop after it is
                // encountered
                if (h_result_labels[i] == label_id) {
                  for (size_t j = i + 1; (j < result_size) && (test_ret_value == 0); ++j) {
                    if (h_result_labels[j] == label_id) {
                      if (h_result_srcs[i] == h_result_srcs[j]) {
                        TEST_ASSERT(test_ret_value,
                                    h_result_hops[i] == h_result_hops[j],
                                    "source vertex should not have been used in diferent hops");
                      }
                    }
                  }
                }
              }
            }
          }
        }

        vertex_t* tmp      = check_sources;
        check_sources      = check_destinations;
        check_destinations = tmp;
        sources_size       = destinations_size;
        destinations_size  = 0;
      }

      if (dedupe_sources) {
        // Make sure vertex v only appears as source once for each edge after it appears as
        // destination Externally test this by verifying that vertex v only appears in <= hop
        // size/degree
        for (size_t i = 0; i < result_size; ++i) {
          if (h_result_labels[i] == label_id) {
            if (h_result_hops[i] > 0) {
              size_t num_occurrences = 1;
              for (size_t j = i + 1; j < result_size; ++j) {
                if (h_result_labels[j] == label_id) {
                  if ((h_result_srcs[j] == h_result_srcs[i]) &&
                      (h_result_hops[j] == h_result_hops[i]))
                    num_occurrences++;
                }
              }

              if (fan_out[h_result_hops[i]] < 0) {
                TEST_ASSERT(test_ret_value,
                            num_occurrences <= degree[h_result_srcs[i]],
                            "source vertex used in too many return edges");
              } else {
                TEST_ASSERT(test_ret_value,
                            num_occurrences <= fan_out[h_result_hops[i]],
                            "source vertex used in too many return edges");
              }
            }
          }
        }
      }

      if (validate_edge_times) {
        // Check that the edge times are moving in the correct direction
        edge_time_t previous_vertex_times[num_vertices];
        for (size_t i = 0; i < num_vertices; ++i)
          if (temporal_sampling_comparison == STRICTLY_INCREASING) {
            previous_vertex_times[i] = -1;
          } else if (temporal_sampling_comparison == MONOTONICALLY_INCREASING) {
            previous_vertex_times[i] = -1;
          } else if (temporal_sampling_comparison == MONOTONICALLY_DECREASING) {
            previous_vertex_times[i] = MAX_EDGE_TIME;
          } else if (temporal_sampling_comparison == STRICTLY_DECREASING) {
            previous_vertex_times[i] = MAX_EDGE_TIME;
          }

        for (size_t hop = 0; hop < fan_out_size; ++hop) {
          for (size_t i = 0; i < result_size; ++i) {
            if (h_result_labels[i] == label_id) {
              if (h_result_hops[i] == hop) {
                if (h_result_edge_start_times[i] > previous_vertex_times[h_result_srcs[i]]) {
                  if (temporal_sampling_comparison == STRICTLY_INCREASING) {
                    TEST_ASSERT(
                      test_ret_value,
                      h_result_edge_start_times[i] > previous_vertex_times[h_result_srcs[i]],
                      "edge times are not strictly increasing");
                  } else if (temporal_sampling_comparison == MONOTONICALLY_INCREASING) {
                    TEST_ASSERT(
                      test_ret_value,
                      h_result_edge_start_times[i] >= previous_vertex_times[h_result_srcs[i]],
                      "edge times are not monotonically increasing");
                  } else if (temporal_sampling_comparison == MONOTONICALLY_DECREASING) {
                    TEST_ASSERT(
                      test_ret_value,
                      h_result_edge_start_times[i] <= previous_vertex_times[h_result_srcs[i]],
                      "edge times are not monotonically decreasing");
                  } else if (temporal_sampling_comparison == STRICTLY_DECREASING) {
                    TEST_ASSERT(
                      test_ret_value,
                      h_result_edge_start_times[i] < previous_vertex_times[h_result_srcs[i]],
                      "edge times are not strictly decreasing");
                  }
                }
              }
            }
          }

          for (size_t i = 0; i < result_size; ++i) {
            if (h_result_labels[i] == label_id) {
              if (h_result_hops[i] == hop) {
                if ((previous_vertex_times[h_result_dsts[i]] == -1) ||
                    (previous_vertex_times[h_result_dsts[i]] > h_result_edge_start_times[i])) {
                  previous_vertex_times[h_result_dsts[i]] = h_result_edge_start_times[i];
                }
              }
            }
          }
        }
      }
    }
  }

  // FIXME: Add other C++ checks here

  return test_ret_value;
}
