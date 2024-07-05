/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include "c_api/graph_functions.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_similarity_result_t {
  cugraph_type_erased_device_array_t* similarity_coefficients_;
  cugraph_vertex_pairs_t* vertex_pairs_;
};

}  // namespace c_api
}  // namespace cugraph

namespace {

template <typename call_similarity_functor_t>
struct similarity_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  cugraph::c_api::cugraph_vertex_pairs_t const* vertex_pairs_;
  call_similarity_functor_t call_similarity_;
  bool use_weight_;
  bool do_expensive_check_;

  cugraph::c_api::cugraph_similarity_result_t* result_{};

  similarity_functor(::cugraph_resource_handle_t const* handle,
                     ::cugraph_graph_t* graph,
                     ::cugraph_vertex_pairs_t const* vertex_pairs,
                     call_similarity_functor_t call_similarity,
                     bool use_weight,
                     bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      vertex_pairs_(reinterpret_cast<cugraph::c_api::cugraph_vertex_pairs_t const*>(vertex_pairs)),
      call_similarity_(call_similarity),
      use_weight_(use_weight),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // similarity algorithms expect store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> v1(vertex_pairs_->first_->size_, handle_.get_stream());
      rmm::device_uvector<vertex_t> v2(vertex_pairs_->second_->size_, handle_.get_stream());
      raft::copy(
        v1.data(), vertex_pairs_->first_->as_type<vertex_t>(), v1.size(), handle_.get_stream());
      raft::copy(
        v2.data(), vertex_pairs_->second_->as_type<vertex_t>(), v2.size(), handle_.get_stream());

      //
      // Need to renumber vertex pairs
      //
      cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        v1.data(),
        v1.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        false);

      cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        v2.data(),
        v2.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        false);

      auto similarity_coefficients =
        call_similarity_(handle_,
                         graph_view,
                         use_weight_ ? std::make_optional(edge_weights->view()) : std::nullopt,
                         std::make_tuple(raft::device_span<vertex_t const>{v1.data(), v1.size()},
                                         raft::device_span<vertex_t const>{v2.data(), v2.size()}));

      result_ = new cugraph::c_api::cugraph_similarity_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(similarity_coefficients,
                                                               graph_->weight_type_),
        nullptr};
    }
  }
};

template <typename call_similarity_functor_t>
struct all_pairs_similarity_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices_;
  call_similarity_functor_t call_similarity_;
  bool use_weight_;
  size_t topk_;
  bool do_expensive_check_;

  cugraph::c_api::cugraph_similarity_result_t* result_{};

  all_pairs_similarity_functor(::cugraph_resource_handle_t const* handle,
                               ::cugraph_graph_t* graph,
                               ::cugraph_type_erased_device_array_view_t const* vertices,
                               call_similarity_functor_t call_similarity,
                               bool use_weight,
                               size_t topk,
                               bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(vertices)),
      call_similarity_(call_similarity),
      use_weight_(use_weight),
      topk_(topk),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // similarity algorithms expect store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto [v1, v2, similarity_coefficients] =
        call_similarity_(handle_,
                         graph_view,
                         use_weight_ ? std::make_optional(edge_weights->view()) : std::nullopt,
                         vertices_ ? std::make_optional(raft::device_span<vertex_t const>{
                                       vertices_->as_type<vertex_t const>(), vertices_->size_})
                                   : std::nullopt,
                         topk_ != SIZE_MAX ? std::make_optional(topk_) : std::nullopt);
      
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                   v1.data(),
                                                   v1.size(),
                                                   number_map->data(),
                                                   graph_view.vertex_partition_range_lasts(),
                                                   false);
      
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                   v2.data(),
                                                   v2.size(),
                                                   number_map->data(),
                                                   graph_view.vertex_partition_range_lasts(),
                                                   false);

      result_ = new cugraph::c_api::cugraph_similarity_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(similarity_coefficients,
                                                               graph_->weight_type_),
        new cugraph::c_api::cugraph_vertex_pairs_t{
          new cugraph::c_api::cugraph_type_erased_device_array_t(v1, graph_->vertex_type_),
          new cugraph::c_api::cugraph_type_erased_device_array_t(v2, graph_->vertex_type_)}};
    }
  }
};

struct jaccard_functor {
  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  rmm::device_uvector<weight_t> operator()(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs)
  {
    return cugraph::jaccard_coefficients(handle, graph_view, edge_weight_view, vertex_pairs);
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             rmm::device_uvector<weight_t>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             std::optional<raft::device_span<vertex_t const>> vertices,
             std::optional<size_t> topk)
  {
    return cugraph::jaccard_all_pairs_coefficients(
      handle, graph_view, edge_weight_view, vertices, topk);
  }
};

struct sorensen_functor {
  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  rmm::device_uvector<weight_t> operator()(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs)
  {
    return cugraph::sorensen_coefficients(handle, graph_view, edge_weight_view, vertex_pairs);
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             rmm::device_uvector<weight_t>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             std::optional<raft::device_span<vertex_t const>> vertices,
             std::optional<size_t> topk)
  {
    return cugraph::sorensen_all_pairs_coefficients(
      handle, graph_view, edge_weight_view, vertices, topk);
  }
};

struct overlap_functor {
  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  rmm::device_uvector<weight_t> operator()(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs)
  {
    return cugraph::overlap_coefficients(handle, graph_view, edge_weight_view, vertex_pairs);
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             rmm::device_uvector<weight_t>>
  operator()(raft::handle_t const& handle,
             cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             std::optional<raft::device_span<vertex_t const>> vertices,
             std::optional<size_t> topk)
  {
    return cugraph::overlap_all_pairs_coefficients(
      handle, graph_view, edge_weight_view, vertices, topk);
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_similarity_result_get_similarity(
  cugraph_similarity_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_similarity_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->similarity_coefficients_->view());
}

extern "C" cugraph_vertex_pairs_t* cugraph_similarity_result_get_vertex_pairs(
  cugraph_similarity_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_similarity_result_t*>(result);
  return reinterpret_cast<cugraph_vertex_pairs_t*>(internal_pointer->vertex_pairs_);
}

extern "C" void cugraph_similarity_result_free(cugraph_similarity_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_similarity_result_t*>(result);
  delete internal_pointer->similarity_coefficients_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_jaccard_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_vertex_pairs_t* vertex_pairs,
  bool_t use_weight,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  if (use_weight) {
    CAPI_EXPECTS(
      reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr,
      CUGRAPH_INVALID_INPUT,
      "use_weight is true but edge weights are not provided.",
      *error);
  }
  similarity_functor functor(
    handle, graph, vertex_pairs, jaccard_functor{}, use_weight, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_sorensen_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_vertex_pairs_t* vertex_pairs,
  bool_t use_weight,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  if (use_weight) {
    CAPI_EXPECTS(
      reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr,
      CUGRAPH_INVALID_INPUT,
      "use_weight is true but edge weights are not provided.",
      *error);
  }
  similarity_functor functor(
    handle, graph, vertex_pairs, sorensen_functor{}, use_weight, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_overlap_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_vertex_pairs_t* vertex_pairs,
  bool_t use_weight,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  if (use_weight) {
    CAPI_EXPECTS(
      reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr,
      CUGRAPH_INVALID_INPUT,
      "use_weight is true but edge weights are not provided.",
      *error);
  }
  similarity_functor functor(
    handle, graph, vertex_pairs, overlap_functor{}, use_weight, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_all_pairs_jaccard_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* vertices,
  bool_t use_weight,
  size_t topk,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  if (use_weight) {
    CAPI_EXPECTS(
      reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr,
      CUGRAPH_INVALID_INPUT,
      "use_weight is true but edge weights are not provided.",
      *error);
  }
  all_pairs_similarity_functor functor(
    handle, graph, vertices, jaccard_functor{}, use_weight, topk, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_all_pairs_sorensen_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* vertices,
  bool_t use_weight,
  size_t topk,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  if (use_weight) {
    CAPI_EXPECTS(
      reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr,
      CUGRAPH_INVALID_INPUT,
      "use_weight is true but edge weights are not provided.",
      *error);
  }
  all_pairs_similarity_functor functor(
    handle, graph, vertices, sorensen_functor{}, use_weight, topk, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_all_pairs_overlap_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* vertices,
  bool_t use_weight,
  size_t topk,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  if (use_weight) {
    CAPI_EXPECTS(
      reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr,
      CUGRAPH_INVALID_INPUT,
      "use_weight is true but edge weights are not provided.",
      *error);
  }
  all_pairs_similarity_functor functor(
    handle, graph, vertices, overlap_functor{}, use_weight, topk, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}