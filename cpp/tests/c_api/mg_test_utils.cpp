/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <c_api/c_test_utils.h>
#include <c_api/resource_handle.hpp>

#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/comms/mpi_comms.hpp>

#include <rmm/device_uvector.hpp>

extern "C" int run_mg_test(int (*test)(const cugraph_resource_handle_t*),
                           const char* test_name,
                           const cugraph_resource_handle_t* handle)
{
  int ret_val = 0;
  time_t start_time, end_time;
  int rank = 0;

  auto raft_handle =
    reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_;
  auto& comm = raft_handle->get_comms();

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank == 0) {
    printf("RUNNING: %s...", test_name);
    fflush(stdout);

    time(&start_time);
  }

  ret_val = test(handle);

  // FIXME:  This is copied from host_scalar_allreduce
  //         which is in a file of thrust enabled code which can't
  //         be included in a cpp file.  Either make this file a .cu
  //         or refactor host_scalar_comm.hpp to separate the thrust
  //         code from the non-thrust code
  rmm::device_uvector<int> d_input(1, raft_handle->get_stream());
  raft::update_device(d_input.data(), &ret_val, 1, raft_handle->get_stream());
  comm.allreduce(
    d_input.data(), d_input.data(), 1, raft::comms::op_t::SUM, raft_handle->get_stream());
  raft::update_host(&ret_val, d_input.data(), 1, raft_handle->get_stream());
  auto status = comm.sync_stream(raft_handle->get_stream());
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  if (rank == 0) {
    time(&end_time);

    printf("done (%f seconds).", difftime(end_time, start_time));
    if (ret_val == 0) {
      printf(" - passed\n");
    } else {
      printf(" - FAILED\n");
    }
    fflush(stdout);
  }

  return ret_val;
}

extern "C" void* create_raft_handle(int prows)
{
  raft::handle_t* handle = new raft::handle_t{};
  raft::comms::initialize_mpi_comms(handle, MPI_COMM_WORLD);

  cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t> subcomm_factory(
    *handle, prows);

  return handle;
}

extern "C" void free_raft_handle(void* raft_handle)
{
  raft::handle_t* handle = reinterpret_cast<raft::handle_t*>(raft_handle);
  delete handle;
}

/*
 * Simple function to create an MG test graph from COO arrays.  COO is
 * assumed to be defined entirely on rank 0 and will be shuffled to
 * the proper location.
 */
extern "C" int create_mg_test_graph(const cugraph_resource_handle_t* handle,
                                    int32_t* h_src,
                                    int32_t* h_dst,
                                    float* h_wgt,
                                    size_t num_edges,
                                    bool_t store_transposed,
                                    bool_t is_symmetric,
                                    cugraph_graph_t** p_graph,
                                    cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank == 0) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");
  } else {
    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code = cugraph_type_erased_device_array_create(handle, 0, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");
  }

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_mg_graph_create(handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     store_transposed,
                                     num_edges,
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
 * Simple function to create an MG test graph from COO arrays.  COO is
 * assumed to be defined entirely on rank 0 and will be shuffled to
 * the proper location.
 */
extern "C" int create_mg_test_graph_double(const cugraph_resource_handle_t* handle,
                                           int32_t* h_src,
                                           int32_t* h_dst,
                                           double* h_wgt,
                                           size_t num_edges,
                                           bool_t store_transposed,
                                           bool_t is_symmetric,
                                           cugraph_graph_t** p_graph,
                                           cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT64;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank == 0) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");
  } else {
    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code = cugraph_type_erased_device_array_create(handle, 0, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");
  }

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_mg_graph_create(handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     store_transposed,
                                     num_edges,
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

extern "C" int create_mg_test_graph_with_edge_ids(const cugraph_resource_handle_t* handle,
                                                  int32_t* h_src,
                                                  int32_t* h_dst,
                                                  int32_t* h_idx,
                                                  size_t num_edges,
                                                  bool_t store_transposed,
                                                  bool_t is_symmetric,
                                                  cugraph_graph_t** p_graph,
                                                  cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* idx;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* idx_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank == 0) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &idx, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "idx create failed.");
  } else {
    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code = cugraph_type_erased_device_array_create(handle, 0, weight_tid, &idx, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");
  }

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  idx_view = cugraph_type_erased_device_array_view(idx);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, idx_view, (byte_t*)h_idx, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_as_type(idx, weight_tid, &wgt_view, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt cast from idx failed.");

  ret_code = cugraph_mg_graph_create(handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     store_transposed,
                                     num_edges,
                                     FALSE,
                                     p_graph,
                                     ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(idx_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(idx);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

extern "C" int create_mg_test_graph_with_properties(const cugraph_resource_handle_t* handle,
                                                    int32_t* h_src,
                                                    int32_t* h_dst,
                                                    int32_t* h_idx,
                                                    int32_t* h_type,
                                                    float* h_wgt,
                                                    size_t num_edges,
                                                    bool_t store_transposed,
                                                    bool_t is_symmetric,
                                                    cugraph_graph_t** p_graph,
                                                    cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t index_tid  = INT32;
  data_type_id_t type_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* idx;
  cugraph_type_erased_device_array_t* type;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* idx_view;
  cugraph_type_erased_device_array_view_t* type_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank == 0) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, index_tid, &idx, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "index create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, type_tid, &type, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "type create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");
  } else {
    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &src, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");

    ret_code = cugraph_type_erased_device_array_create(handle, 0, vertex_tid, &dst, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code = cugraph_type_erased_device_array_create(handle, 0, index_tid, &idx, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "index create failed.");

    ret_code = cugraph_type_erased_device_array_create(handle, 0, type_tid, &type, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "type create failed.");

    ret_code = cugraph_type_erased_device_array_create(handle, 0, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");
  }

  src_view  = cugraph_type_erased_device_array_view(src);
  dst_view  = cugraph_type_erased_device_array_view(dst);
  idx_view  = cugraph_type_erased_device_array_view(idx);
  type_view = cugraph_type_erased_device_array_view(type);
  wgt_view  = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, idx_view, (byte_t*)h_idx, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "index copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, type_view, (byte_t*)h_type, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "type copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_mg_graph_create(handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     idx_view,
                                     type_view,
                                     store_transposed,
                                     num_edges,
                                     FALSE,
                                     p_graph,
                                     ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(type_view);
  cugraph_type_erased_device_array_view_free(idx_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(type);
  cugraph_type_erased_device_array_free(idx);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}
