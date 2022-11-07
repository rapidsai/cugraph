/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <c_api/array.hpp>
#include <c_api/error.hpp>
#include <c_api/resource_handle.hpp>

namespace cugraph {
namespace c_api {

cugraph::visitors::DTypes dtypes_mapping[] = {
  cugraph::visitors::DTypes::INT32,
  cugraph::visitors::DTypes::INT64,
  cugraph::visitors::DTypes::FLOAT32,
  cugraph::visitors::DTypes::FLOAT64,
  cugraph::visitors::DTypes::SIZE_T,
};

size_t data_type_sz[] = {4, 8, 4, 8, 8};

}  // namespace c_api
}  // namespace cugraph

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_create_from_view(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* view,
  cugraph_type_erased_device_array_t** array,
  cugraph_error_t** error)
{
  *array = nullptr;
  *error = nullptr;

  try {
    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
    auto internal_pointer =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(view);

    size_t n_bytes =
      internal_pointer->size_ * (cugraph::c_api::data_type_sz[internal_pointer->type_]);

    auto ret_value = new cugraph::c_api::cugraph_type_erased_device_array_t(
      internal_pointer->size_, n_bytes, internal_pointer->type_, p_handle->handle_->get_stream());

    raft::copy(reinterpret_cast<byte_t*>(ret_value->data_.data()),
               reinterpret_cast<byte_t const*>(internal_pointer->data_),
               internal_pointer->num_bytes(),
               p_handle->handle_->get_stream());

    *array = reinterpret_cast<cugraph_type_erased_device_array_t*>(ret_value);
    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_create(
  const cugraph_resource_handle_t* handle,
  size_t n_elems,
  data_type_id_t dtype,
  cugraph_type_erased_device_array_t** array,
  cugraph_error_t** error)
{
  *array = nullptr;
  *error = nullptr;

  try {
    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);

    size_t n_bytes = n_elems * (cugraph::c_api::data_type_sz[dtype]);

    auto ret_value = new cugraph::c_api::cugraph_type_erased_device_array_t(
      n_elems, n_bytes, dtype, p_handle->handle_->get_stream());

    *array = reinterpret_cast<cugraph_type_erased_device_array_t*>(ret_value);
    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" void cugraph_type_erased_device_array_free(cugraph_type_erased_device_array_t* p)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(p);
  delete internal_pointer;
}

#if 0
// NOTE:  This can't work.  rmm::device_buffer doesn't support release, that would leave a raw
//        pointer in the wild with no idea how to free it.  I suppose that could be done
//        (I imagine you can do that with unique_ptr), but it's not currently supported and I'm
//        not sure *this* use case is sufficient justification to adding a potentially
//        dangerous feature.
extern "C" void* cugraph_type_erased_device_array_release(cugraph_type_erased_device_array_t* p)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(p);
  return internal_pointer->data_.release();
}
#endif

extern "C" cugraph_type_erased_device_array_view_t* cugraph_type_erased_device_array_view(
  cugraph_type_erased_device_array_t* array)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(array);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->view());
}

cugraph_type_erased_device_array_view_t* cugraph_type_erased_device_array_view_create(
  void* pointer, size_t n_elems, data_type_id_t dtype)
{
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    new cugraph::c_api::cugraph_type_erased_device_array_view_t{
      pointer, n_elems, n_elems * (cugraph::c_api::data_type_sz[dtype]), dtype});
}

extern "C" void cugraph_type_erased_device_array_view_free(
  cugraph_type_erased_device_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(p);
  delete internal_pointer;
}

extern "C" size_t cugraph_type_erased_device_array_view_size(
  const cugraph_type_erased_device_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(p);
  return internal_pointer->size_;
}

extern "C" data_type_id_t cugraph_type_erased_device_array_view_type(
  const cugraph_type_erased_device_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(p);
  return internal_pointer->type_;
}

extern "C" const void* cugraph_type_erased_device_array_view_pointer(
  const cugraph_type_erased_device_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(p);
  return internal_pointer->data_;
}

extern "C" cugraph_error_code_t cugraph_type_erased_host_array_create(
  const cugraph_resource_handle_t* handle,
  size_t n_elems,
  data_type_id_t dtype,
  cugraph_type_erased_host_array_t** array,
  cugraph_error_t** error)
{
  *array = nullptr;
  *error = nullptr;

  try {
    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);

    size_t n_bytes = n_elems * (cugraph::c_api::data_type_sz[dtype]);

    *array = reinterpret_cast<cugraph_type_erased_host_array_t*>(
      new cugraph::c_api::cugraph_type_erased_host_array_t{n_elems, n_bytes, dtype});

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new cugraph::c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" void cugraph_type_erased_host_array_free(cugraph_type_erased_host_array_t* p)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_t*>(p);
  delete internal_pointer;
}

#if 0
// Leaving this one out since we're not doing the more important device version
extern "C" void* cugraph_type_erased_host_array_release(const cugraph_type_erased_host_array_t* p)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_t*>(p);
  return internal_pointer->data_.release();
}
#endif

extern "C" cugraph_type_erased_host_array_view_t* cugraph_type_erased_host_array_view(
  cugraph_type_erased_host_array_t* array)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_t*>(array);
  return reinterpret_cast<cugraph_type_erased_host_array_view_t*>(internal_pointer->view());
}

extern "C" cugraph_type_erased_host_array_view_t* cugraph_type_erased_host_array_view_create(
  void* pointer, size_t n_elems, data_type_id_t dtype)
{
  return reinterpret_cast<cugraph_type_erased_host_array_view_t*>(
    new cugraph::c_api::cugraph_type_erased_host_array_view_t{
      static_cast<std::byte*>(pointer),
      n_elems,
      n_elems * (cugraph::c_api::data_type_sz[dtype]),
      dtype});
}

extern "C" void cugraph_type_erased_host_array_view_free(cugraph_type_erased_host_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t*>(p);
  delete internal_pointer;
}

extern "C" size_t cugraph_type_erased_host_array_size(
  const cugraph_type_erased_host_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(p);
  return internal_pointer->size_;
}

extern "C" data_type_id_t cugraph_type_erased_host_array_view_type(
  const cugraph_type_erased_host_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(p);
  return internal_pointer->type_;
}

extern "C" void* cugraph_type_erased_host_array_pointer(
  const cugraph_type_erased_host_array_view_t* p)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(p);
  return internal_pointer->data_;
}

extern "C" cugraph_error_code_t cugraph_type_erased_host_array_view_copy(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_host_array_view_t* dst,
  const cugraph_type_erased_host_array_view_t* src,
  cugraph_error_t** error)
{
  *error = nullptr;

  try {
    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
    auto internal_pointer_dst =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t*>(dst);
    auto internal_pointer_src =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(src);

    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    if (internal_pointer_src->num_bytes() != internal_pointer_dst->num_bytes()) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"source and destination arrays are different sizes"});
      return CUGRAPH_INVALID_INPUT;
    }

    raft::copy(reinterpret_cast<byte_t*>(internal_pointer_dst->data_),
               reinterpret_cast<byte_t const*>(internal_pointer_src->data_),
               internal_pointer_src->num_bytes(),
               p_handle->handle_->get_stream());

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new cugraph::c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_view_copy_from_host(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_device_array_view_t* dst,
  const byte_t* h_src,
  cugraph_error_t** error)
{
  *error = nullptr;

  try {
    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
    auto internal_pointer =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(dst);

    raft::update_device(reinterpret_cast<byte_t*>(internal_pointer->data_),
                        h_src,
                        internal_pointer->num_bytes(),
                        p_handle->handle_->get_stream());

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new cugraph::c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_view_copy_to_host(
  const cugraph_resource_handle_t* handle,
  byte_t* h_dst,
  const cugraph_type_erased_device_array_view_t* src,
  cugraph_error_t** error)
{
  *error = nullptr;

  try {
    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
    auto internal_pointer =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(src);

    raft::update_host(h_dst,
                      reinterpret_cast<byte_t const*>(internal_pointer->data_),
                      internal_pointer->num_bytes(),
                      p_handle->handle_->get_stream());

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new cugraph::c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}
extern "C" cugraph_error_code_t cugraph_type_erased_device_array_view_copy(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* src,
  cugraph_error_t** error)
{
  *error = nullptr;

  try {
    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
    auto internal_pointer_dst =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(dst);
    auto internal_pointer_src =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(src);

    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    if (internal_pointer_src->num_bytes() != internal_pointer_dst->num_bytes()) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"source and destination arrays are different sizes"});
      return CUGRAPH_INVALID_INPUT;
    }

    raft::copy(reinterpret_cast<byte_t*>(internal_pointer_dst->data_),
               reinterpret_cast<byte_t const*>(internal_pointer_src->data_),
               internal_pointer_src->num_bytes(),
               p_handle->handle_->get_stream());

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    auto tmp_error = new cugraph::c_api::cugraph_error_t{ex.what()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" cugraph_error_code_t cugraph_type_erased_device_array_view_as_type(
  cugraph_type_erased_device_array_t* array,
  data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** result_view,
  cugraph_error_t** error)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(array);

  if (cugraph::c_api::data_type_sz[dtype] ==
      cugraph::c_api::data_type_sz[internal_pointer->type_]) {
    *result_view = reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
      new cugraph::c_api::cugraph_type_erased_device_array_view_t{internal_pointer->data_.data(),
                                                                  internal_pointer->size_,
                                                                  internal_pointer->data_.size(),
                                                                  dtype});
    return CUGRAPH_SUCCESS;
  } else {
    std::stringstream ss;
    ss << "Could not treat type " << internal_pointer->type_ << " as type " << dtype;
    auto tmp_error = new cugraph::c_api::cugraph_error_t{ss.str().c_str()};
    *error         = reinterpret_cast<cugraph_error_t*>(tmp_error);
    return CUGRAPH_INVALID_INPUT;
  }
}
