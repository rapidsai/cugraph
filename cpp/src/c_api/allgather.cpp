/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/induced_subgraph_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph_generators.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace {

struct create_allgather_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* src_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_type_ids_;
  cugraph::c_api::cugraph_induced_subgraph_result_t* result_{};

  create_allgather_functor(
    raft::handle_t const& handle,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* src,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_type_ids)
    : abstract_functor(),
      handle_(handle),
      src_(src),
      dst_(dst),
      weights_(weights),
      edge_ids_(edge_ids),
      edge_type_ids_(edge_type_ids)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_id_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    std::optional<rmm::device_uvector<vertex_t>> edgelist_srcs{std::nullopt};
    if (src_) {
      edgelist_srcs = rmm::device_uvector<vertex_t>(src_->size_, handle_.get_stream());
      raft::copy(
        edgelist_srcs->data(), src_->as_type<vertex_t>(), src_->size_, handle_.get_stream());
    }

    std::optional<rmm::device_uvector<vertex_t>> edgelist_dsts{std::nullopt};
    if (dst_) {
      edgelist_dsts = rmm::device_uvector<vertex_t>(dst_->size_, handle_.get_stream());
      raft::copy(
        edgelist_dsts->data(), dst_->as_type<vertex_t>(), dst_->size_, handle_.get_stream());
    }

    std::optional<rmm::device_uvector<weight_t>> edgelist_weights{std::nullopt};
    if (weights_) {
      edgelist_weights = rmm::device_uvector<weight_t>(weights_->size_, handle_.get_stream());
      raft::copy(edgelist_weights->data(),
                 weights_->as_type<weight_t>(),
                 weights_->size_,
                 handle_.get_stream());
    }

    std::optional<rmm::device_uvector<edge_t>> edgelist_ids{std::nullopt};
    if (edge_ids_) {
      edgelist_ids = rmm::device_uvector<edge_t>(edge_ids_->size_, handle_.get_stream());
      raft::copy(
        edgelist_ids->data(), edge_ids_->as_type<edge_t>(), edge_ids_->size_, handle_.get_stream());
    }

    std::optional<rmm::device_uvector<edge_type_id_t>> edgelist_type_ids{std::nullopt};
    if (edge_type_ids_) {
      edgelist_type_ids =
        rmm::device_uvector<edge_type_id_t>(edge_type_ids_->size_, handle_.get_stream());
      raft::copy(edgelist_type_ids->data(),
                 edge_type_ids_->as_type<edge_type_id_t>(),
                 edge_type_ids_->size_,
                 handle_.get_stream());
    }

    auto& comm = handle_.get_comms();

    if (edgelist_srcs) {
      edgelist_srcs = cugraph::detail::device_allgatherv(
        handle_,
        comm,
        raft::device_span<vertex_t const>(edgelist_srcs->data(), edgelist_srcs->size()));
    }

    if (edgelist_dsts) {
      edgelist_dsts = cugraph::detail::device_allgatherv(
        handle_,
        comm,
        raft::device_span<vertex_t const>(edgelist_dsts->data(), edgelist_dsts->size()));
    }

    rmm::device_uvector<size_t> edge_offsets(2, handle_.get_stream());

    std::vector<size_t> h_edge_offsets{
      {0, edgelist_srcs ? edgelist_srcs->size() : edgelist_weights->size()}};
    raft::update_device(
      edge_offsets.data(), h_edge_offsets.data(), h_edge_offsets.size(), handle_.get_stream());

    cugraph::c_api::cugraph_induced_subgraph_result_t* result = NULL;

    if (edgelist_weights) {
      edgelist_weights = cugraph::detail::device_allgatherv(
        handle_,
        comm,
        raft::device_span<weight_t const>(edgelist_weights->data(), edgelist_weights->size()));
    }

    if (edgelist_ids) {
      edgelist_ids = cugraph::detail::device_allgatherv(
        handle_, comm, raft::device_span<edge_t const>(edgelist_ids->data(), edgelist_ids->size()));
    }

    if (edgelist_type_ids) {
      edgelist_type_ids =
        cugraph::detail::device_allgatherv(handle_,
                                           comm,
                                           raft::device_span<edge_type_id_t const>(
                                             edgelist_type_ids->data(), edgelist_type_ids->size()));
    }

    result = new cugraph::c_api::cugraph_induced_subgraph_result_t{
      edgelist_srcs
        ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edgelist_srcs, src_->type_)
        : NULL,
      edgelist_dsts
        ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edgelist_dsts, dst_->type_)
        : NULL,
      edgelist_weights
        ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edgelist_weights, weights_->type_)
        : NULL,
      edgelist_ids
        ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edgelist_ids, edge_ids_->type_)
        : NULL,
      edgelist_type_ids ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                            *edgelist_type_ids, edge_type_ids_->type_)
                        : NULL,
      new cugraph::c_api::cugraph_type_erased_device_array_t(edge_offsets,
                                                             cugraph_data_type_id_t::SIZE_T)};

    result_ = reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(result);
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_allgather(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_device_array_view_t* edge_ids,
  const cugraph_type_erased_device_array_view_t* edge_type_ids,
  cugraph_induced_subgraph_result_t** edgelist,
  cugraph_error_t** error)
{
  *edgelist = nullptr;
  *error    = nullptr;

  auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto p_src =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(src);
  auto p_dst =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(dst);
  auto p_weights =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(weights);

  auto p_edge_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(edge_ids);

  auto p_edge_type_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(edge_type_ids);

  CAPI_EXPECTS((dst == nullptr) || (src == nullptr) || p_src->size_ == p_dst->size_,
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != dst size.",
               *error);
  CAPI_EXPECTS((dst == nullptr) || (src == nullptr) || p_src->type_ == p_dst->type_,
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src type != dst type.",
               *error);

  CAPI_EXPECTS((weights == nullptr) || (src == nullptr) || (p_weights->size_ == p_src->size_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != weights size.",
               *error);

  cugraph_data_type_id_t vertex_type;
  cugraph_data_type_id_t edge_type;
  cugraph_data_type_id_t weight_type;
  cugraph_data_type_id_t edge_type_id_type;

  if (src != nullptr) {
    vertex_type = p_src->type_;
  } else {
    vertex_type = cugraph_data_type_id_t::INT32;
  }

  if (weights != nullptr) {
    weight_type = p_weights->type_;
  } else {
    weight_type = cugraph_data_type_id_t::FLOAT32;
  }

  if (edge_ids != nullptr) {
    edge_type = p_edge_ids->type_;
  } else {
    edge_type = cugraph_data_type_id_t::INT32;
  }

  if (edge_type_ids != nullptr) {
    edge_type_id_type = p_edge_type_ids->type_;
  } else {
    edge_type_id_type = cugraph_data_type_id_t::INT32;
  }

  if (src != nullptr) {
    CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids->size_ == p_src->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != edge id prop size",
                 *error);

    CAPI_EXPECTS((edge_type_ids == nullptr) || (p_edge_type_ids->size_ == p_src->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != edge type prop size",
                 *error);
  }

  constexpr bool multi_gpu        = false;
  constexpr bool store_transposed = false;

  ::create_allgather_functor functor(
    *p_handle->handle_, p_src, p_dst, p_weights, p_edge_ids, p_edge_type_ids);

  try {
    cugraph::c_api::vertex_dispatcher(
      vertex_type, edge_type, weight_type, edge_type_id_type, store_transposed, multi_gpu, functor);

    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }

    *edgelist = reinterpret_cast<cugraph_induced_subgraph_result_t*>(functor.result_);
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}
