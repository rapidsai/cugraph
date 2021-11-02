/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph_c/array.h>
#include <c_api/error.hpp>

#include <raft/handle.hpp>

#include <rmm/device_buffer.hpp>

namespace c_api {

typedef struct {
  rmm::device_buffer* data_;
  size_t size_;
  size_t nbytes_;
  data_type_id_t type_;
} cugraph_type_erased_device_array_t;

typedef struct {
  std::byte* data_;
  size_t size_;
  size_t nbytes_;
  data_type_id_t type_;
} cugraph_type_erased_host_array_t;

}  // namespace c_api

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_create(
  const cugraph_resource_handle_t* handle,
  data_type_id_t dtype,
  size_t n_elems,
  cugraph_type_erased_device_array_t** array,
  cugraph_error_t** error)
{
  *array = nullptr;
  *error = nullptr;

  try {
    raft::handle_t const* raft_handle = reinterpret_cast<raft::handle_t const*>(handle);

    if (!raft_handle) return CUGRAPH_INVALID_HANDLE;

    size_t nbytes = n_elems * (::data_type_sz[dtype]);

    c_api::cugraph_type_erased_device_array_t* ret_value =
      new c_api::cugraph_type_erased_device_array_t{
        new rmm::device_buffer(nbytes, raft_handle->get_stream()), n_elems, nbytes, dtype};

    *array = reinterpret_cast<cugraph_type_erased_device_array_t*>(ret_value);
    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" void cugraph_type_erased_device_array_free(cugraph_type_erased_device_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_device_array_t*>(p);
  delete internal_pointer->data_;
  delete internal_pointer;
}

extern "C" size_t cugraph_type_erased_device_array_size(const cugraph_type_erased_device_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_device_array_t const*>(p);
  return internal_pointer->size_;
}

extern "C" data_type_id_t cugraph_type_erased_device_array_type(
  const cugraph_type_erased_device_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_device_array_t const*>(p);
  return internal_pointer->type_;
}

extern "C" void* cugraph_type_erased_device_array_pointer(
  const cugraph_type_erased_device_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_device_array_t const*>(p);
  return internal_pointer->data_->data();
}

extern "C" cugraph_error_code_t cugraph_type_erased_host_array_create(
  const cugraph_resource_handle_t* handle,
  data_type_id_t dtype,
  size_t n_elems,
  cugraph_type_erased_host_array_t** array,
  cugraph_error_t** error)
{
  *array = nullptr;
  *error = nullptr;

  try {
    raft::handle_t const* raft_handle = reinterpret_cast<raft::handle_t const*>(handle);

    if (!raft_handle) return CUGRAPH_INVALID_HANDLE;

    size_t nbytes = n_elems * (::data_type_sz[dtype]);

    c_api::cugraph_type_erased_host_array_t* ret_value =
      new c_api::cugraph_type_erased_host_array_t{new std::byte[nbytes], n_elems, nbytes, dtype};

    *array = reinterpret_cast<cugraph_type_erased_host_array_t*>(ret_value);
    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" void cugraph_type_erased_host_array_free(cugraph_type_erased_host_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_host_array_t*>(p);
  delete[] internal_pointer->data_;
  delete internal_pointer;
}

extern "C" size_t cugraph_type_erased_host_array_size(const cugraph_type_erased_host_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_host_array_t const*>(p);
  return internal_pointer->size_;
}

extern "C" data_type_id_t cugraph_type_erased_host_array_type(
  const cugraph_type_erased_host_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_host_array_t const*>(p);
  return internal_pointer->type_;
}

extern "C" void* cugraph_type_erased_host_array_pointer(const cugraph_type_erased_host_array_t* p)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_host_array_t const*>(p);
  return internal_pointer->data_;
}

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_copy_from_host(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_device_array_t* dst,
  const byte_t* h_src,
  cugraph_error_t** error)
{
  *error = nullptr;

  try {
    raft::handle_t const* raft_handle = reinterpret_cast<raft::handle_t const*>(handle);
    auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_device_array_t*>(dst);

    if (!raft_handle) return CUGRAPH_ALLOC_ERROR;

    raft::update_device(reinterpret_cast<byte_t*>(internal_pointer->data_->data()),
                        h_src,
                        internal_pointer->nbytes_,
                        raft_handle->get_stream());

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_copy_to_host(
  const cugraph_resource_handle_t* handle,
  byte_t* h_dst,
  const cugraph_type_erased_device_array_t* src,
  cugraph_error_t** error)
{
  *error = nullptr;

  try {
    raft::handle_t const* raft_handle = reinterpret_cast<raft::handle_t const*>(handle);
    auto internal_pointer = reinterpret_cast<c_api::cugraph_type_erased_device_array_t const*>(src);

    if (!raft_handle) return CUGRAPH_ALLOC_ERROR;

    raft::update_host(h_dst,
                      reinterpret_cast<byte_t*>(internal_pointer->data_->data()),
                      internal_pointer->nbytes_,
                      raft_handle->get_stream());

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}
