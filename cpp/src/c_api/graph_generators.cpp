/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cugraph_c/graph_generators.h>

#include <c_api/array.hpp>
#include <c_api/error.hpp>
#include <c_api/random.hpp>
#include <c_api/resource_handle.hpp>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_coo_t {
  std::unique_ptr<cugraph_type_erased_device_array_t> src_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> dst_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> wgt_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> id_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> type_{};
};

struct cugraph_coo_list_t {
  std::vector<std::unique_ptr<cugraph_coo_t>> list_;
};

}  // namespace c_api
}  // namespace cugraph

namespace {

template <typename vertex_t>
cugraph_error_code_t cugraph_generate_rmat_edgelist(raft::handle_t const& handle,
                                                    raft::random::RngState& rng_state,
                                                    cugraph_data_type_id_t vertex_dtype,
                                                    size_t scale,
                                                    size_t num_edges,
                                                    double a,
                                                    double b,
                                                    double c,
                                                    bool_t clip_and_flip,
                                                    bool_t scramble_vertex_ids,
                                                    cugraph::c_api::cugraph_coo_t** result,
                                                    cugraph::c_api::cugraph_error_t** error)
{
  try {
    auto [src, dst] = cugraph::generate_rmat_edgelist<vertex_t>(
      handle, rng_state, scale, num_edges, a, b, c, clip_and_flip, scramble_vertex_ids);

    *result = new cugraph::c_api::cugraph_coo_t{
      std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(src, vertex_dtype),
      std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(dst, vertex_dtype),
      nullptr,
      nullptr,
      nullptr};

  } catch (std::exception const& ex) {
    *error = new cugraph::c_api::cugraph_error_t{ex.what()};
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

template <typename vertex_t>
cugraph_error_code_t cugraph_generate_rmat_edgelists(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  cugraph_data_type_id_t vertex_dtype,
  size_t n_edgelists,
  size_t min_scale,
  size_t max_scale,
  size_t edge_factor,
  cugraph_generator_distribution_t size_distribution,
  cugraph_generator_distribution_t edge_distribution,
  bool_t clip_and_flip,
  bool_t scramble_vertex_ids,
  cugraph::c_api::cugraph_coo_list_t** result,
  cugraph::c_api::cugraph_error_t** error)
{
  try {
    auto tuple_vector = cugraph::generate_rmat_edgelists<vertex_t>(
      handle,
      rng_state,
      n_edgelists,
      min_scale,
      max_scale,
      edge_factor,
      static_cast<cugraph::generator_distribution_t>(size_distribution),
      static_cast<cugraph::generator_distribution_t>(edge_distribution),
      clip_and_flip,
      scramble_vertex_ids);

    *result = new cugraph::c_api::cugraph_coo_list_t;
    (*result)->list_.resize(tuple_vector.size());

    std::transform(
      tuple_vector.begin(),
      tuple_vector.end(),
      (*result)->list_.begin(),
      [vertex_dtype](auto& tuple) {
        auto result = std::make_unique<cugraph::c_api::cugraph_coo_t>();

        auto& src = std::get<0>(tuple);
        auto& dst = std::get<1>(tuple);

        result->src_ =
          std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(src, vertex_dtype);
        result->dst_ =
          std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(dst, vertex_dtype);

        return result;
      });

    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    *error = new cugraph::c_api::cugraph_error_t{ex.what()};
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_coo_get_sources(cugraph_coo_t* coo)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->src_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_coo_get_destinations(cugraph_coo_t* coo)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->dst_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_coo_get_edge_weights(cugraph_coo_t* coo)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->wgt_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_coo_get_edge_id(cugraph_coo_t* coo)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->id_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_coo_get_edge_type(cugraph_coo_t* coo)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->type_->view());
}

extern "C" size_t cugraph_coo_list_size(const cugraph_coo_list_t* coo_list)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_list_t const*>(coo_list);
  return internal_pointer->list_.size();
}

extern "C" cugraph_coo_t* cugraph_coo_list_element(cugraph_coo_list_t* coo_list, size_t index)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_list_t*>(coo_list);
  return reinterpret_cast<::cugraph_coo_t*>(
    (index < internal_pointer->list_.size()) ? internal_pointer->list_[index].get() : nullptr);
}

extern "C" void cugraph_coo_free(cugraph_coo_t* coo)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);
  delete internal_pointer;
}

extern "C" void cugraph_coo_list_free(cugraph_coo_list_t* coo_list)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_coo_list_t*>(coo_list);
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_generate_rmat_edgelist(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  bool_t clip_and_flip,
  bool_t scramble_vertex_ids,
  cugraph_coo_t** result,
  cugraph_error_t** error)
{
  auto& local_handle{
    *reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_};
  auto& local_rng_state{
    reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)->rng_state_};

  if (scale < 32) {
    return cugraph_generate_rmat_edgelist<int32_t>(
      local_handle,
      local_rng_state,
      cugraph_data_type_id_t::INT32,
      scale,
      num_edges,
      a,
      b,
      c,
      clip_and_flip,
      scramble_vertex_ids,
      reinterpret_cast<cugraph::c_api::cugraph_coo_t**>(result),
      reinterpret_cast<cugraph::c_api::cugraph_error_t**>(error));
  } else {
    return cugraph_generate_rmat_edgelist<int64_t>(
      local_handle,
      local_rng_state,
      cugraph_data_type_id_t::INT64,
      scale,
      num_edges,
      a,
      b,
      c,
      clip_and_flip,
      scramble_vertex_ids,
      reinterpret_cast<cugraph::c_api::cugraph_coo_t**>(result),
      reinterpret_cast<cugraph::c_api::cugraph_error_t**>(error));
  }
}

extern "C" cugraph_error_code_t cugraph_generate_rmat_edgelists(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  size_t n_edgelists,
  size_t min_scale,
  size_t max_scale,
  size_t edge_factor,
  cugraph_generator_distribution_t size_distribution,
  cugraph_generator_distribution_t edge_distribution,
  bool_t clip_and_flip,
  bool_t scramble_vertex_ids,
  cugraph_coo_list_t** result,
  cugraph_error_t** error)
{
  auto& local_handle{
    *reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_};
  auto& local_rng_state{
    reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)->rng_state_};

  if (max_scale < 32) {
    return cugraph_generate_rmat_edgelists<int32_t>(
      local_handle,
      local_rng_state,
      cugraph_data_type_id_t::INT32,
      n_edgelists,
      min_scale,
      max_scale,
      edge_factor,
      size_distribution,
      edge_distribution,
      clip_and_flip,
      scramble_vertex_ids,
      reinterpret_cast<cugraph::c_api::cugraph_coo_list_t**>(result),
      reinterpret_cast<cugraph::c_api::cugraph_error_t**>(error));
  } else {
    return cugraph_generate_rmat_edgelists<int64_t>(
      local_handle,
      local_rng_state,
      cugraph_data_type_id_t::INT64,
      n_edgelists,
      min_scale,
      max_scale,
      edge_factor,
      size_distribution,
      edge_distribution,
      clip_and_flip,
      scramble_vertex_ids,
      reinterpret_cast<cugraph::c_api::cugraph_coo_list_t**>(result),
      reinterpret_cast<cugraph::c_api::cugraph_error_t**>(error));
  }
}

extern "C" cugraph_error_code_t cugraph_generate_edge_weights(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_coo_t* coo,
  cugraph_data_type_id_t dtype,
  double minimum_weight,
  double maximum_weight,
  cugraph_error_t** error)
{
  auto& local_handle{
    *reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_};
  auto& local_rng_state{
    reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)->rng_state_};

  auto local_coo = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);

  switch (dtype) {
    case cugraph_data_type_id_t::FLOAT32: {
      rmm::device_uvector<float> tmp(local_coo->src_->size_, local_handle.get_stream());
      cugraph::detail::uniform_random_fill(local_handle.get_stream(),
                                           tmp.data(),
                                           tmp.size(),
                                           static_cast<float>(minimum_weight),
                                           static_cast<float>(maximum_weight),
                                           local_rng_state);
      local_coo->wgt_ =
        std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(tmp, dtype);
      break;
    }
    case cugraph_data_type_id_t::FLOAT64: {
      rmm::device_uvector<double> tmp(local_coo->src_->size_, local_handle.get_stream());
      cugraph::detail::uniform_random_fill(local_handle.get_stream(),
                                           tmp.data(),
                                           tmp.size(),
                                           minimum_weight,
                                           maximum_weight,
                                           local_rng_state);
      local_coo->wgt_ =
        std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(tmp, dtype);
      break;
    }
    otherwise : {
      *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t(
        "Only FLOAT and DOUBLE supported as generated edge weights"));
      return CUGRAPH_INVALID_INPUT;
    }
  }

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_generate_edge_ids(const cugraph_resource_handle_t* handle,
                                                          cugraph_coo_t* coo,
                                                          bool_t multi_gpu,
                                                          cugraph_error_t** error)
{
  auto& local_handle{
    *reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_};

  auto local_coo = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);

  constexpr size_t int32_threshold{std::numeric_limits<int32_t>::max()};

  size_t num_edges{local_coo->src_->size_};
  size_t base_edge_id{0};

  if (multi_gpu) {
    auto edge_counts = cugraph::host_scalar_allgather(
      local_handle.get_comms(), num_edges, local_handle.get_stream());
    std::vector<size_t> edge_starts(edge_counts.size());

    std::exclusive_scan(edge_counts.begin(), edge_counts.end(), edge_starts.begin(), size_t{0});

    num_edges    = edge_starts.back() + edge_counts.back();
    base_edge_id = edge_starts[local_handle.get_comms().get_rank()];
  }

  if (num_edges < int32_threshold) {
    rmm::device_uvector<int32_t> tmp(local_coo->src_->size_, local_handle.get_stream());

    cugraph::detail::sequence_fill(
      local_handle.get_stream(), tmp.data(), tmp.size(), static_cast<int32_t>(base_edge_id));

    local_coo->id_ = std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(
      tmp, cugraph_data_type_id_t::INT32);
  } else {
    rmm::device_uvector<int64_t> tmp(local_coo->src_->size_, local_handle.get_stream());

    cugraph::detail::sequence_fill(
      local_handle.get_stream(), tmp.data(), tmp.size(), static_cast<int64_t>(base_edge_id));

    local_coo->id_ = std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(
      tmp, cugraph_data_type_id_t::INT64);
  }

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_generate_edge_types(const cugraph_resource_handle_t* handle,
                                                            cugraph_rng_state_t* rng_state,
                                                            cugraph_coo_t* coo,
                                                            int32_t min_edge_type,
                                                            int32_t max_edge_type,
                                                            cugraph_error_t** error)
{
  auto& local_handle{
    *reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_};
  auto& local_rng_state{
    reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)->rng_state_};

  auto local_coo = reinterpret_cast<cugraph::c_api::cugraph_coo_t*>(coo);

  rmm::device_uvector<int32_t> tmp(local_coo->src_->size_, local_handle.get_stream());
  cugraph::detail::uniform_random_fill(local_handle.get_stream(),
                                       tmp.data(),
                                       tmp.size(),
                                       min_edge_type,
                                       max_edge_type,
                                       local_rng_state);
  local_coo->type_ = std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(
    tmp, cugraph_data_type_id_t::INT32);

  return CUGRAPH_SUCCESS;
}
