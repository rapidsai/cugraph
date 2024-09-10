/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
//#include <cugraph_c/graph_generators.h>

#include <cugraph/algorithms.hpp>
//#include <cugraph/detail/collect_comm_wrapper.hpp>
//#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace {

struct create_allgather_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* src_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights_;
  bool_t reciprocal_;
  cugraph::c_api::cugraph_induced_subgraph_result_t* result_{};

  create_allgather_functor(
    raft::handle_t const& handle,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* src,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights,
    bool_t reciprocal)
    : abstract_functor(),
      handle_(handle),
      src_(src),
      dst_(dst),
      weights_(weights),
      reciprocal_(reciprocal)
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

    rmm::device_uvector<vertex_t> edgelist_srcs(src_->size_, handle_.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(dst_->size_, handle_.get_stream());

    raft::copy<vertex_t>(
    edgelist_srcs.data(), src_->as_type<vertex_t>(), src_->size_, handle_.get_stream());
    raft::copy<vertex_t>(
    edgelist_dsts.data(), dst_->as_type<vertex_t>(), dst_->size_, handle_.get_stream());

    std::optional<rmm::device_uvector<weight_t>> edgelist_weights =
    weights_
        ? std::make_optional(rmm::device_uvector<weight_t>(weights_->size_, handle_.get_stream()))
        : std::nullopt;

    if (edgelist_weights) {
        raft::copy<weight_t>(edgelist_weights->data(),
                                weights_->as_type<weight_t>(),
                                weights_->size_,
                                handle_.get_stream());
    }

    cugraph::detail::symmetrize_edgelist(
        edgelist_srcs,
        edgelist_dsts,
        edgelist_weights,
        reciprocal
    );




    


  
  }
};

}  // namespace


extern "C" cugraph_error_code_t symmetrize_edgelist(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  bool_t reciprocal,
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
  cugraph_data_type_id_t weight_type;

  cugraph_data_type_id_t edge_type = cugraph_data_type_id_t::INT32;
  cugraph_data_type_id_t edge_type_id_type = cugraph_data_type_id_t::INT32;

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

  constexpr bool multi_gpu        = false;
  constexpr bool store_transposed = false;

  ::symmetrize_edgelist_functor functor(
    *p_handle->handle_, p_src, p_dst, p_weights);
  

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



}