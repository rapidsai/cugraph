/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/array.hpp"
#include "c_api/sampling_common.hpp"
#include "c_api/utils.hpp"

extern "C" cugraph_error_code_t cugraph_sampling_options_create(
  cugraph_sampling_options_t** options, cugraph_error_t** error)
{
  *options =
    reinterpret_cast<cugraph_sampling_options_t*>(new cugraph::c_api::cugraph_sampling_options_t());
  if (*options == nullptr) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  return CUGRAPH_SUCCESS;
}

extern "C" void cugraph_sampling_set_retain_seeds(cugraph_sampling_options_t* options, bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->retain_seeds_ = value;
}

extern "C" void cugraph_sampling_set_renumber_results(cugraph_sampling_options_t* options,
                                                      bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->renumber_results_ = value;
}

extern "C" void cugraph_sampling_set_compress_per_hop(cugraph_sampling_options_t* options,
                                                      bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->compress_per_hop_ = value;
}

extern "C" void cugraph_sampling_set_with_replacement(cugraph_sampling_options_t* options,
                                                      bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->with_replacement_ = value;
}

extern "C" void cugraph_sampling_set_return_hops(cugraph_sampling_options_t* options, bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->return_hops_ = value;
}

extern "C" void cugraph_sampling_set_compression_type(cugraph_sampling_options_t* options,
                                                      cugraph_compression_type_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  switch (value) {
    case COO: internal_pointer->compression_type_ = cugraph_compression_type_t::COO; break;
    case CSR: internal_pointer->compression_type_ = cugraph_compression_type_t::CSR; break;
    case CSC: internal_pointer->compression_type_ = cugraph_compression_type_t::CSC; break;
    case DCSR: internal_pointer->compression_type_ = cugraph_compression_type_t::DCSR; break;
    case DCSC: internal_pointer->compression_type_ = cugraph_compression_type_t::DCSC; break;
    default: CUGRAPH_FAIL("Invalid compression type");
  }
}

extern "C" void cugraph_sampling_set_prior_sources_behavior(cugraph_sampling_options_t* options,
                                                            cugraph_prior_sources_behavior_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  switch (value) {
    case CARRY_OVER:
      internal_pointer->prior_sources_behavior_ = cugraph::prior_sources_behavior_t::CARRY_OVER;
      break;
    case EXCLUDE:
      internal_pointer->prior_sources_behavior_ = cugraph::prior_sources_behavior_t::EXCLUDE;
      break;
    default:
      internal_pointer->prior_sources_behavior_ = cugraph::prior_sources_behavior_t::DEFAULT;
      break;
  }
}

extern "C" void cugraph_sampling_set_dedupe_sources(cugraph_sampling_options_t* options,
                                                    bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->dedupe_sources_ = value;
}

extern "C" void cugraph_sampling_set_temporal_sampling_comparison(
  cugraph_sampling_options_t* options, cugraph_temporal_sampling_comparison_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->temporal_sampling_comparison_ = value;
}

extern "C" void cugraph_sampling_options_free(cugraph_sampling_options_t* options)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  delete internal_pointer;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_sources(
  const cugraph_sample_result_t* result)
{
  // Deprecated.
  return cugraph_sample_result_get_majors(result);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_destinations(
  const cugraph_sample_result_t* result)
{
  // Deprecated.
  return cugraph_sample_result_get_minors(result);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_majors(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return (internal_pointer->majors_ != nullptr)
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->majors_->view())

           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_major_offsets(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return (internal_pointer->major_offsets_ != nullptr)
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->major_offsets_->view())

           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_minors(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->minors_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_start_labels(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->label_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->label_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_id(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_id_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_id_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_type(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_type_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_type_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_weight(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->wgt_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->wgt_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_start_time(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_start_time_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_start_time_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_end_time(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_end_time_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_end_time_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_hop(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->hop_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->hop_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_label_hop_offsets(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->label_hop_offsets_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->label_hop_offsets_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_sample_result_get_label_type_hop_offsets(const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->label_type_hop_offsets_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->label_type_hop_offsets_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_index(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->edge_id_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_offsets(
  const cugraph_sample_result_t* result)
{
  // Deprecated.
  return cugraph_sample_result_get_label_hop_offsets(result);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_renumber_map(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->renumber_map_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_renumber_map_offsets(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->renumber_map_offsets_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_renumber_map(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_renumber_map_->view());
}

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_sample_result_get_edge_renumber_map_offsets(const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_renumber_map_offsets_->view());
}

extern "C" cugraph_error_code_t cugraph_test_uniform_neighborhood_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* sampled_edge_ids,
  const cugraph_type_erased_device_array_view_t* sampled_edge_type,
  const cugraph_type_erased_device_array_view_t* weight,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;
  size_t n_bytes{0};
  cugraph_error_code_t error_code{CUGRAPH_SUCCESS};

  if (!handle) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  // Create unique_ptrs and release them during cugraph_sample_result_t
  // construction. This allows the arrays to be cleaned up if this function
  // returns early on error.
  using device_array_unique_ptr_t =
    std::unique_ptr<cugraph_type_erased_device_array_t,
                    decltype(&cugraph_type_erased_device_array_free)>;

  // copy srcs to new device array
  cugraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, srcs, &new_device_srcs_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy dsts to new device array
  cugraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, dsts, &new_device_dsts_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy weights to new device array
  cugraph_type_erased_device_array_t* new_device_weight_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, weight, &new_device_weight_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_weight(new_device_weight_ptr,
                                              &cugraph_type_erased_device_array_free);

  // copy edge ids to new device array
  cugraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, sampled_edge_ids, &new_device_edge_id_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                               &cugraph_type_erased_device_array_free);

  // copy edge types to new device array
  cugraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, sampled_edge_type, &new_device_edge_type_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                 &cugraph_type_erased_device_array_free);
  // copy hop ids to new device array
  cugraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, hop, &new_device_hop_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy labels to new device array
  cugraph_type_erased_device_array_t* new_device_label_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, label, &new_device_label_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                             &cugraph_type_erased_device_array_free);

  // create new cugraph_sample_result_t
  *result = reinterpret_cast<cugraph_sample_result_t*>(new cugraph::c_api::cugraph_sample_result_t{
    nullptr,
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_srcs.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_dsts.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_id.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_type.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_weight.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(new_device_hop.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_label.release())});

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_test_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* sampled_edge_ids,
  const cugraph_type_erased_device_array_view_t* sampled_edge_type,
  const cugraph_type_erased_device_array_view_t* sampled_weights,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;
  size_t n_bytes{0};
  cugraph_error_code_t error_code{CUGRAPH_SUCCESS};

  if (!handle) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  // Create unique_ptrs and release them during cugraph_sample_result_t
  // construction. This allows the arrays to be cleaned up if this function
  // returns early on error.
  using device_array_unique_ptr_t =
    std::unique_ptr<cugraph_type_erased_device_array_t,
                    decltype(&cugraph_type_erased_device_array_free)>;

  // copy srcs to new device array
  cugraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, srcs, &new_device_srcs_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy dsts to new device array
  cugraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, dsts, &new_device_dsts_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy sampled_edge_ids to new device array
  cugraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};

  if (sampled_edge_ids != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, sampled_edge_ids, &new_device_edge_id_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                               &cugraph_type_erased_device_array_free);

  // copy sampled_edge_type to new device array
  cugraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};

  if (sampled_edge_type != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, sampled_edge_type, &new_device_edge_type_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                 &cugraph_type_erased_device_array_free);

  // copy sampled_weights to new device array
  cugraph_type_erased_device_array_t* new_device_wgt_ptr{nullptr};
  if (sampled_weights != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, sampled_weights, &new_device_wgt_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_wgt(new_device_wgt_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy hop to new device array
  cugraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, hop, &new_device_hop_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy label to new device array
  cugraph_type_erased_device_array_t* new_device_label_ptr{nullptr};

  if (label != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, label, &new_device_label_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                             &cugraph_type_erased_device_array_free);

  // create new cugraph_sample_result_t
  *result = reinterpret_cast<cugraph_sample_result_t*>(new cugraph::c_api::cugraph_sample_result_t{
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_srcs.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_dsts.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_id.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_type.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(new_device_wgt.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_label.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_hop.release())});

  return CUGRAPH_SUCCESS;
}

extern "C" void cugraph_sample_result_free(cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t*>(result);

  delete internal_pointer->major_offsets_;
  delete internal_pointer->majors_;
  delete internal_pointer->minors_;
  delete internal_pointer->edge_id_;
  delete internal_pointer->edge_type_;
  delete internal_pointer->wgt_;
  delete internal_pointer->edge_start_time_;
  delete internal_pointer->edge_end_time_;
  delete internal_pointer->hop_;
  delete internal_pointer->label_hop_offsets_;
  delete internal_pointer->label_type_hop_offsets_;
  delete internal_pointer->label_;
  delete internal_pointer->renumber_map_;
  delete internal_pointer->renumber_map_offsets_;
  delete internal_pointer->edge_renumber_map_;
  delete internal_pointer->edge_renumber_map_offsets_;

  delete internal_pointer;
}
