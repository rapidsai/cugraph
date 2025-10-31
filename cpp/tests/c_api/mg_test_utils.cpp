/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "c_api/mg_test_utils.h"

#include "../sampling/detail/nbr_sampling_validate.hpp"
#include "c_api/array.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/sampling_common.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"

#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

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

extern "C" void* create_mg_raft_handle(int argc, char** argv)
{
  int comm_rank;
  int comm_size;
  int num_gpus_per_node;
  cudaError_t status;
  int mpi_status;

  C_MPI_TRY(MPI_Init(&argc, &argv));
  C_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
  C_MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
  C_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  C_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));

  raft::handle_t* handle = new raft::handle_t{};
  raft::comms::initialize_mpi_comms(handle, MPI_COMM_WORLD);

#if 1
  int gpu_row_comm_size = 1;
#else
  // TODO:  Need something a bit more sophisticated for bigger systems
  gpu_row_comm_size = (int)sqrt((double)comm_size);
  while (comm_size % gpu_row_comm_size != 0) {
    --gpu_row_comm_size;
  }
#endif
  cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

  return handle;
}

extern "C" void free_mg_raft_handle(void* raft_handle)
{
  raft::handle_t* handle = reinterpret_cast<raft::handle_t*>(raft_handle);
  delete handle;

  C_MPI_TRY(MPI_Finalize());
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
  properties.is_multigraph = TRUE;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank != 0) num_edges = 0;

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

  ret_code = cugraph_graph_create_with_times_mg(handle,
                                                &properties,
                                                NULL,
                                                &src_view,
                                                &dst_view,
                                                wgt_view == nullptr ? NULL : &wgt_view,
                                                NULL,
                                                NULL,
                                                NULL,
                                                NULL,
                                                store_transposed,
                                                1,
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
  properties.is_multigraph = TRUE;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT64;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank != 0) num_edges = 0;

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

  ret_code = cugraph_graph_create_with_times_mg(handle,
                                                &properties,
                                                NULL,
                                                &src_view,
                                                &dst_view,
                                                wgt_view == nullptr ? NULL : &wgt_view,
                                                NULL,
                                                NULL,
                                                NULL,
                                                NULL,
                                                store_transposed,
                                                1,
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
  properties.is_multigraph = TRUE;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* idx;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* idx_view;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank != 0) num_edges = 0;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code = cugraph_type_erased_device_array_create(handle, num_edges, edge_tid, &idx, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "idx create failed.");

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

  ret_code = cugraph_graph_create_with_times_mg(handle,
                                                &properties,
                                                NULL,
                                                &src_view,
                                                &dst_view,
                                                NULL,
                                                idx_view == nullptr ? NULL : &idx_view,
                                                NULL,
                                                NULL,
                                                NULL,
                                                store_transposed,
                                                1,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                p_graph,
                                                ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

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
  properties.is_multigraph = TRUE;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t index_tid  = INT32;
  cugraph_data_type_id_t type_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src            = NULL;
  cugraph_type_erased_device_array_t* dst            = NULL;
  cugraph_type_erased_device_array_t* idx            = NULL;
  cugraph_type_erased_device_array_t* type           = NULL;
  cugraph_type_erased_device_array_t* wgt            = NULL;
  cugraph_type_erased_device_array_view_t* src_view  = NULL;
  cugraph_type_erased_device_array_view_t* dst_view  = NULL;
  cugraph_type_erased_device_array_view_t* idx_view  = NULL;
  cugraph_type_erased_device_array_view_t* type_view = NULL;
  cugraph_type_erased_device_array_view_t* wgt_view  = NULL;

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank != 0) num_edges = 0;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");

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

  if (h_idx != nullptr) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, index_tid, &idx, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "index create failed.");

    idx_view = cugraph_type_erased_device_array_view(idx);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, idx_view, (byte_t*)h_idx, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "index copy_from_host failed.");
  }

  if (h_type != nullptr) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, type_tid, &type, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "type create failed.");

    type_view = cugraph_type_erased_device_array_view(type);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, type_view, (byte_t*)h_type, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "type copy_from_host failed.");
  }

  if (h_wgt != nullptr) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

    wgt_view = cugraph_type_erased_device_array_view(wgt);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, wgt_view, (byte_t*)h_wgt, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");
  }

  ret_code = cugraph_graph_create_with_times_mg(handle,
                                                &properties,
                                                NULL,
                                                &src_view,
                                                &dst_view,
                                                wgt_view == nullptr ? NULL : &wgt_view,
                                                idx_view == nullptr ? NULL : &idx_view,
                                                type_view == nullptr ? NULL : &type_view,
                                                NULL,
                                                NULL,
                                                store_transposed,
                                                1,
                                                FALSE,
                                                FALSE,
                                                FALSE,
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

int create_mg_test_graph_new(const cugraph_resource_handle_t* handle,
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

  int rank = 0;

  rank = cugraph_resource_handle_get_rank(handle);

  if (rank != 0) num_edges = 0;

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
    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "edge_start_times copy_from_host failed.");
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

  ret_code = cugraph_graph_create_with_times_mg(
    handle,
    &properties,
    NULL,
    &src_view,
    &dst_view,
    wgt_view == nullptr ? NULL : &wgt_view,
    edge_id_view == nullptr ? NULL : &edge_id_view,
    edge_type_view == nullptr ? NULL : &edge_type_view,
    edge_start_times_view == nullptr ? NULL : &edge_start_times_view,
    edge_end_times_view == nullptr ? NULL : &edge_end_times_view,
    store_transposed,
    1,
    FALSE,
    FALSE,
    FALSE,
    TRUE,
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

extern "C" size_t cugraph_test_device_gatherv_size(
  const cugraph_resource_handle_t* handle, const cugraph_type_erased_device_array_view_t* array)
{
  auto internal_array =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(array);

  size_t ret_value = internal_array->size_;

  auto raft_handle =
    reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_;
  auto& comm = raft_handle->get_comms();

  rmm::device_uvector<size_t> d_input(1, raft_handle->get_stream());
  raft::update_device(d_input.data(), &ret_value, 1, raft_handle->get_stream());
  comm.allreduce(
    d_input.data(), d_input.data(), 1, raft::comms::op_t::SUM, raft_handle->get_stream());
  raft::update_host(&ret_value, d_input.data(), 1, raft_handle->get_stream());
  auto status = comm.sync_stream(raft_handle->get_stream());
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  return (comm.get_rank() == 0) ? ret_value : 0;
}

extern "C" size_t cugraph_test_scalar_reduce(const cugraph_resource_handle_t* handle, size_t value)
{
  auto raft_handle =
    reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_;
  auto& comm = raft_handle->get_comms();

  rmm::device_uvector<size_t> d_input(1, raft_handle->get_stream());
  raft::update_device(d_input.data(), &value, 1, raft_handle->get_stream());
  comm.allreduce(
    d_input.data(), d_input.data(), 1, raft::comms::op_t::SUM, raft_handle->get_stream());
  raft::update_host(&value, d_input.data(), 1, raft_handle->get_stream());
  auto status = comm.sync_stream(raft_handle->get_stream());
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  return (comm.get_rank() == 0) ? value : 0;
}

extern "C" cugraph_error_code_t cugraph_test_host_gatherv_fill(
  const cugraph_resource_handle_t* handle,
  void* input,
  size_t input_size,
  cugraph_data_type_id_t input_type,
  void* output)
{
  auto raft_handle =
    reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_;
  auto& comm = raft_handle->get_comms();

  switch (input_type) {
    case cugraph_data_type_id_t::INT32: {
      auto tmp = cugraph::test::to_device(
        *raft_handle,
        raft::host_span<int32_t const>{reinterpret_cast<int32_t const*>(input), input_size});
      tmp = cugraph::test::device_gatherv(*raft_handle, tmp.data(), tmp.size());
      raft::update_host(
        reinterpret_cast<int32_t*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::INT64: {
      auto tmp = cugraph::test::to_device(
        *raft_handle,
        raft::host_span<int64_t const>{reinterpret_cast<int64_t const*>(input), input_size});
      tmp = cugraph::test::device_gatherv(*raft_handle, tmp.data(), tmp.size());
      raft::update_host(
        reinterpret_cast<int64_t*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::FLOAT32: {
      auto tmp = cugraph::test::to_device(
        *raft_handle,
        raft::host_span<float const>{reinterpret_cast<float const*>(input), input_size});
      tmp = cugraph::test::device_gatherv(*raft_handle, tmp.data(), tmp.size());
      raft::update_host(
        reinterpret_cast<float*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::FLOAT64: {
      auto tmp = cugraph::test::to_device(
        *raft_handle,
        raft::host_span<double const>{reinterpret_cast<double const*>(input), input_size});
      tmp = cugraph::test::device_gatherv(*raft_handle, tmp.data(), tmp.size());
      raft::update_host(
        reinterpret_cast<double*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::SIZE_T: {
      auto tmp = cugraph::test::to_device(
        *raft_handle,
        raft::host_span<size_t const>{reinterpret_cast<size_t const*>(input), input_size});
      tmp = cugraph::test::device_gatherv(*raft_handle, tmp.data(), tmp.size());
      raft::update_host(
        reinterpret_cast<size_t*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    default: {
      return CUGRAPH_UNKNOWN_ERROR;
    }
  }

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_test_device_gatherv_fill(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* input,
  void* output)
{
  auto internal_array =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(input);
  auto raft_handle =
    reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_;
  auto& comm = raft_handle->get_comms();

  switch (internal_array->type_) {
    case cugraph_data_type_id_t::INT32: {
      auto tmp = cugraph::test::device_gatherv(
        *raft_handle,
        raft::device_span<int32_t const>{internal_array->as_type<int32_t const>(),
                                         internal_array->size_});
      raft::update_host(
        reinterpret_cast<int32_t*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::INT64: {
      auto tmp = cugraph::test::device_gatherv(
        *raft_handle,
        raft::device_span<int64_t const>{internal_array->as_type<int64_t const>(),
                                         internal_array->size_});
      raft::update_host(
        reinterpret_cast<int64_t*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::FLOAT32: {
      auto tmp = cugraph::test::device_gatherv(
        *raft_handle,
        raft::device_span<float const>{internal_array->as_type<float const>(),
                                       internal_array->size_});
      raft::update_host(
        reinterpret_cast<float*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::FLOAT64: {
      auto tmp = cugraph::test::device_gatherv(
        *raft_handle,
        raft::device_span<double const>{internal_array->as_type<double const>(),
                                        internal_array->size_});
      raft::update_host(
        reinterpret_cast<double*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    case cugraph_data_type_id_t::SIZE_T: {
      auto tmp = cugraph::test::device_gatherv(
        *raft_handle,
        raft::device_span<size_t const>{internal_array->as_type<size_t const>(),
                                        internal_array->size_});
      raft::update_host(
        reinterpret_cast<size_t*>(output), tmp.data(), tmp.size(), raft_handle->get_stream());
    } break;
    default: {
      return CUGRAPH_UNKNOWN_ERROR;
    }
  };

  return CUGRAPH_SUCCESS;
}

int mg_validate_sample_result(const cugraph_resource_handle_t* handle,
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
                              int32_t* h_fan_out,
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
  result_srcs                   = cugraph_sample_result_get_sources(result);
  result_dsts                   = cugraph_sample_result_get_destinations(result);
  result_edge_ids               = cugraph_sample_result_get_edge_id(result);
  result_weights                = cugraph_sample_result_get_edge_weight(result);
  result_edge_types             = cugraph_sample_result_get_edge_type(result);
  result_edge_start_times       = cugraph_sample_result_get_edge_start_time(result);
  result_edge_end_times         = cugraph_sample_result_get_edge_end_time(result);
  result_hops                   = cugraph_sample_result_get_hop(result);
  result_label_type_hop_offsets = cugraph_sample_result_get_label_type_hop_offsets(result);
  result_label_hop_offsets      = cugraph_sample_result_get_label_hop_offsets(result);
  result_labels                 = cugraph_sample_result_get_start_labels(result);

  size_t result_size = cugraph_test_device_gatherv_size(handle, result_srcs);

  vertex_t h_result_srcs[result_size];
  vertex_t h_result_dsts[result_size];
  vertex_t h_original_result_srcs[result_size];
  vertex_t h_original_result_dsts[result_size];
  edge_t h_result_edge_ids[result_size];
  weight_t h_result_weights[result_size];
  int32_t h_result_edge_types[result_size];
  int32_t h_result_hops[result_size];
  int32_t h_result_labels[result_size];
  edge_time_t h_result_edge_start_times[result_size];
  edge_time_t h_result_edge_end_times[result_size];

  size_t result_renumber_map_offsets_size = 0;
  size_t result_renumber_map_size         = 0;

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

  auto result_srcs_span = make_span<vertex_t>(result_srcs);
  auto result_dsts_span = make_span<vertex_t>(result_dsts);

  rmm::device_uvector<vertex_t> renumbered_srcs(result_srcs_span.size(), raft_handle.get_stream());
  rmm::device_uvector<vertex_t> renumbered_dsts(result_dsts_span.size(), raft_handle.get_stream());
  rmm::device_uvector<vertex_t> original_srcs(result_srcs_span.size(), raft_handle.get_stream());
  rmm::device_uvector<vertex_t> original_dsts(result_dsts_span.size(), raft_handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> gathered_weights{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> gathered_labels{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> gathered_hops{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> gathered_edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> gathered_edge_types{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> gathered_edge_start_times{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> gathered_edge_end_times{std::nullopt};

  raft::copy(renumbered_srcs.data(),
             result_srcs_span.data(),
             result_srcs_span.size(),
             raft_handle.get_stream());
  raft::copy(renumbered_dsts.data(),
             result_dsts_span.data(),
             result_dsts_span.size(),
             raft_handle.get_stream());
  raft::copy(original_srcs.data(),
             result_srcs_span.data(),
             result_srcs_span.size(),
             raft_handle.get_stream());
  raft::copy(original_dsts.data(),
             result_dsts_span.data(),
             result_dsts_span.size(),
             raft_handle.get_stream());

  // Renumber the source and destination vertices if necessary
  if (result_labels != NULL) {
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_labels, result_labels, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "result_label copy_to_host failed.");
  }

  if (result_renumber_map_offsets != NULL) {
    size_t h_result_renumber_map_offsets[result_renumber_map_offsets_size];
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_renumber_map_offsets, result_renumber_map_offsets, &ret_error);
    TEST_ASSERT(test_ret_value,
                ret_code == CUGRAPH_SUCCESS,
                "result_renumber_map_offsets copy_to_host failed.");

    // Renumber all of the results by label
    if (result_label_type_hop_offsets != NULL) {
      size_t h_result_label_type_hop_offsets[cugraph_type_erased_device_array_view_size(
        result_label_type_hop_offsets)];
      ret_code =
        cugraph_type_erased_device_array_view_copy_to_host(handle,
                                                           (byte_t*)h_result_label_type_hop_offsets,
                                                           result_label_type_hop_offsets,
                                                           &ret_error);
      TEST_ASSERT(test_ret_value,
                  ret_code == CUGRAPH_SUCCESS,
                  "result_label_type_hop_offsets copy_to_host failed.");

      for (size_t i = 0;
           i < (cugraph_type_erased_device_array_view_size(result_label_type_hop_offsets) - 1);
           ++i) {
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
    } else if (result_label_hop_offsets != NULL) {
      size_t h_result_label_hop_offsets[cugraph_type_erased_device_array_view_size(
        result_label_hop_offsets)];
      ret_code = cugraph_type_erased_device_array_view_copy_to_host(
        handle, (byte_t*)h_result_label_hop_offsets, result_label_hop_offsets, &ret_error);
      TEST_ASSERT(test_ret_value,
                  ret_code == CUGRAPH_SUCCESS,
                  "result_label_hop_offsets copy_to_host failed.");

      for (size_t i = 0;
           i < (cugraph_type_erased_device_array_view_size(result_label_hop_offsets) - 1);
           ++i) {
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

  renumbered_srcs = cugraph::test::device_gatherv(
    raft_handle, raft::device_span<vertex_t const>{renumbered_srcs.data(), renumbered_srcs.size()});
  raft::update_host(
    h_result_srcs, renumbered_srcs.data(), renumbered_srcs.size(), raft_handle.get_stream());
  renumbered_dsts = cugraph::test::device_gatherv(
    raft_handle, raft::device_span<vertex_t const>{renumbered_dsts.data(), renumbered_dsts.size()});
  raft::update_host(
    h_result_dsts, renumbered_dsts.data(), renumbered_dsts.size(), raft_handle.get_stream());

  original_srcs = cugraph::test::device_gatherv(
    raft_handle, raft::device_span<vertex_t const>{original_srcs.data(), original_srcs.size()});
  raft::update_host(
    h_original_result_srcs, original_srcs.data(), original_srcs.size(), raft_handle.get_stream());
  original_dsts = cugraph::test::device_gatherv(
    raft_handle, raft::device_span<vertex_t const>{original_dsts.data(), original_dsts.size()});
  raft::update_host(
    h_original_result_dsts, original_dsts.data(), original_dsts.size(), raft_handle.get_stream());

  if (result_weights != NULL) {
    gathered_weights = cugraph::test::device_gatherv(
      raft_handle,
      raft::device_span<weight_t const>{
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_weights)
          ->as_type<weight_t const>(),
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_weights)
          ->size_});
    raft::update_host(h_result_weights,
                      gathered_weights->data(),
                      gathered_weights->size(),
                      raft_handle.get_stream());
  }
  if (result_edge_ids != NULL) {
    gathered_edge_ids = cugraph::test::device_gatherv(
      raft_handle,
      raft::device_span<edge_t const>{
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_ids)
          ->as_type<edge_t const>(),
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_ids)
          ->size_});
    raft::update_host(h_result_edge_ids,
                      gathered_edge_ids->data(),
                      gathered_edge_ids->size(),
                      raft_handle.get_stream());
  }
  if (result_edge_types != NULL) {
    gathered_edge_types = cugraph::test::device_gatherv(
      raft_handle,
      raft::device_span<int32_t const>{
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_types)
          ->as_type<int32_t const>(),
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_types)
          ->size_});
    raft::update_host(h_result_edge_types,
                      gathered_edge_types->data(),
                      gathered_edge_types->size(),
                      raft_handle.get_stream());
  }
  if (result_edge_start_times != NULL) {
    gathered_edge_start_times = cugraph::test::device_gatherv(
      raft_handle,
      raft::device_span<edge_time_t const>{
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_start_times)
          ->as_type<edge_time_t const>(),
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_start_times)
          ->size_});
    raft::update_host(h_result_edge_start_times,
                      gathered_edge_start_times->data(),
                      gathered_edge_start_times->size(),
                      raft_handle.get_stream());
  }
  if (result_edge_end_times != NULL) {
    gathered_edge_end_times = cugraph::test::device_gatherv(
      raft_handle,
      raft::device_span<edge_time_t const>{
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_end_times)
          ->as_type<edge_time_t const>(),
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_edge_end_times)
          ->size_});
    raft::update_host(h_result_edge_end_times,
                      gathered_edge_end_times->data(),
                      gathered_edge_end_times->size(),
                      raft_handle.get_stream());
  }
  if (result_labels != NULL) {
    gathered_labels = cugraph::test::device_gatherv(
      raft_handle,
      raft::device_span<int32_t const>{
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_labels)
          ->as_type<int32_t const>(),
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_labels)
          ->size_});
    raft::update_host(
      h_result_labels, gathered_labels->data(), gathered_labels->size(), raft_handle.get_stream());
  }
  if (result_hops != NULL) {
    gathered_hops = cugraph::test::device_gatherv(
      raft_handle,
      raft::device_span<int32_t const>{
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_hops)
          ->as_type<int32_t const>(),
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          result_hops)
          ->size_});
    raft::update_host(
      h_result_hops, gathered_hops->data(), gathered_hops->size(), raft_handle.get_stream());
  } else if (result_label_type_hop_offsets != nullptr) {
    size_t label_type_hop_offsets_size =
      cugraph_type_erased_device_array_view_size(result_label_type_hop_offsets);
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
    rmm::device_uvector<int32_t> tmp(local_hop_size, raft_handle.get_stream());
    raft::update_device(tmp.data(), h_result_hops, local_hop_size, raft_handle.get_stream());
    gathered_hops = cugraph::test::device_gatherv(
      raft_handle, raft::device_span<int32_t const>{tmp.data(), tmp.size()});
    raft::update_host(
      h_result_hops, gathered_hops->data(), gathered_hops->size(), raft_handle.get_stream());
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
    rmm::device_uvector<int32_t> tmp(local_hop_size, raft_handle.get_stream());
    raft::update_device(tmp.data(), h_result_hops, local_hop_size, raft_handle.get_stream());
    gathered_hops = cugraph::test::device_gatherv(
      raft_handle, raft::device_span<int32_t const>{tmp.data(), tmp.size()});
    raft::update_host(
      h_result_hops, gathered_hops->data(), gathered_hops->size(), raft_handle.get_stream());
  } else {
    std::fill(h_result_hops, h_result_hops + result_size, 0);
  }

  if (h_edge_types != NULL) {
    ret_code = cugraph_test_device_gatherv_fill(handle, result_edge_types, h_result_edge_types);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");
  }

  int rank = cugraph_resource_handle_get_rank(handle);
  if (rank == 0) {
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
    auto result_label_span = (result_labels != NULL)
                               ? std::make_optional(make_span<int32_t>(result_labels))
                               : std::nullopt;
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
                  gathered_weights ? std::make_optional(raft::device_span<weight_t const>{
                                       gathered_weights->data(), gathered_weights->size()})
                                   : std::nullopt),
                "extracted graph is not a subgraph of the original graph");

    //
    // For the sampling result to make sense, all sources in hop 0 must be in the seeds,
    // all sources in hop 1 must be a result from hop 0, etc.
    //
    vertex_t check_v1[result_size];
    vertex_t check_v2[result_size];
    size_t v1_counts[result_size];
    size_t v2_counts[result_size];
    vertex_t* check_sources      = check_v1;
    vertex_t* check_destinations = check_v2;
    size_t* source_counts        = v1_counts;
    size_t* destination_counts   = v2_counts;

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

    for (int label_id = 0; label_id <= num_start_label_offsets; ++label_id) {
      size_t sources_size      = 0;
      size_t destinations_size = 0;

      // Fill sources with the input sources
      for (size_t i = 0; i < result_size; ++i) {
        if (h_result_labels[i] == label_id) {
          check_sources[sources_size] = h_result_srcs[i];
          source_counts[sources_size] = 1;
          ++sources_size;
        }
      }

      if (renumber_results) {
        size_t num_vertex_ids = 0;
        vertex_t vertex_ids[2 * result_size];

        for (size_t i = 0; i < result_size; ++i) {
          if (h_result_labels[i] == label_id) {
            vertex_ids[num_vertex_ids]     = h_original_result_srcs[i];
            vertex_ids[num_vertex_ids + 1] = h_original_result_dsts[i];
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
        // Can't check this if we don't have the result_hop
        for (int hop = 0; hop < fan_out_size; ++hop) {
          if (prior_sources_behavior == cugraph::prior_sources_behavior_t::CARRY_OVER) {
            destinations_size = sources_size;
            for (size_t i = 0; i < sources_size; ++i) {
              check_destinations[i] = check_sources[i];
              destination_counts[i] = source_counts[i];
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
              }
            }

            // Make sure destination isn't already in the source list
            bool found = false;
            size_t j   = 0;
            for (j = 0; (!found) && (j < destinations_size); ++j) {
              found = (h_result_dsts[i] == check_destinations[j]);
            }

            if (!found) {
              check_destinations[destinations_size] = h_result_dsts[i];
              destination_counts[destinations_size] = 1;
              ++destinations_size;
            } else {
              destination_counts[j]++;
            }

            if (prior_sources_behavior == cugraph::prior_sources_behavior_t::EXCLUDE) {
              // Make sure vertex v only appears as source in the first hop after it is encountered
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

                if (h_fan_out[h_result_hops[i]] < 0) {
                  TEST_ASSERT(test_ret_value,
                              num_occurrences <= degree[h_result_srcs[i]],
                              "source vertex used in too many return edges");
                } else {
                  TEST_ASSERT(test_ret_value,
                              num_occurrences <= h_fan_out[h_result_hops[i]],
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

      vertex_t* tmp      = check_sources;
      check_sources      = check_destinations;
      check_destinations = tmp;
      size_t* tmp_counts = source_counts;
      source_counts      = destination_counts;
      destination_counts = tmp_counts;
      sources_size       = destinations_size;
      destinations_size  = 0;
    }
  }

  cugraph_error_free(ret_error);
  return test_ret_value;
}
