/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <cugraph_c/graph.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/array.hpp>
#include <c_api/error.hpp>
#include <c_api/generic_cascaded_dispatch.hpp>
#include <c_api/graph.hpp>
#include <c_api/resource_handle.hpp>

#include <limits>

namespace {

template <typename value_t>
rmm::device_uvector<value_t> concatenate(
  raft::handle_t const& handle,
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* values,
  size_t num_arrays)
{
  size_t num_values = std::transform_reduce(
    values, values + num_arrays, size_t{0}, std::plus{}, [](auto p) { return p->size_; });

  rmm::device_uvector<value_t> results(num_values, handle.get_stream());
  size_t concat_pos{0};

  for (size_t i = 0; i < num_arrays; ++i) {
    raft::copy<value_t>(results.data() + concat_pos,
                        values[i]->as_type<value_t>(),
                        values[i]->size_,
                        handle.get_stream());
    concat_pos += values[i]->size_;
  }

  return results;
}

struct create_graph_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_properties_t const* properties_;
  cugraph_data_type_id_t vertex_type_;
  cugraph_data_type_id_t edge_type_;
  cugraph_data_type_id_t weight_type_;
  cugraph_data_type_id_t edge_type_id_type_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* vertices_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* src_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* dst_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* weights_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_ids_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_type_ids_;
  size_t num_arrays_;
  bool_t renumber_;
  bool_t drop_self_loops_;
  bool_t drop_multi_edges_;
  bool_t do_expensive_check_;
  cugraph::c_api::cugraph_graph_t* result_{};

  create_graph_functor(
    raft::handle_t const& handle,
    cugraph_graph_properties_t const* properties,
    cugraph_data_type_id_t vertex_type,
    cugraph_data_type_id_t edge_type,
    cugraph_data_type_id_t weight_type,
    cugraph_data_type_id_t edge_type_id_type,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* vertices,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* src,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* dst,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* weights,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_ids,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_type_ids,
    size_t num_arrays,
    bool_t renumber,
    bool_t drop_self_loops,
    bool_t drop_multi_edges,
    bool_t do_expensive_check)
    : abstract_functor(),
      properties_(properties),
      vertex_type_(vertex_type),
      edge_type_(edge_type),
      weight_type_(weight_type),
      edge_type_id_type_(edge_type_id_type),
      handle_(handle),
      vertices_(vertices),
      src_(src),
      dst_(dst),
      weights_(weights),
      edge_ids_(edge_ids),
      edge_type_ids_(edge_type_ids),
      num_arrays_(num_arrays),
      renumber_(renumber),
      drop_self_loops_(drop_self_loops),
      drop_multi_edges_(drop_multi_edges),
      do_expensive_check_(do_expensive_check)
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
    if constexpr (!multi_gpu || !cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      std::optional<rmm::device_uvector<vertex_t>> new_number_map;

      std::optional<cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        weight_t>>
        new_edge_weights{std::nullopt};

      std::optional<cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        edge_t>>
        new_edge_ids{std::nullopt};

      std::optional<cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        edge_type_id_t>>
        new_edge_types{std::nullopt};

      std::optional<rmm::device_uvector<vertex_t>> vertex_list =
        vertices_ ? std::make_optional(concatenate<vertex_t>(handle_, vertices_, num_arrays_))
                  : std::nullopt;

      rmm::device_uvector<vertex_t> edgelist_srcs =
        concatenate<vertex_t>(handle_, src_, num_arrays_);
      rmm::device_uvector<vertex_t> edgelist_dsts =
        concatenate<vertex_t>(handle_, dst_, num_arrays_);

      std::optional<rmm::device_uvector<weight_t>> edgelist_weights =
        weights_ ? std::make_optional(concatenate<weight_t>(handle_, weights_, num_arrays_))
                 : std::nullopt;

      std::optional<rmm::device_uvector<edge_t>> edgelist_edge_ids =
        edge_ids_ ? std::make_optional(concatenate<edge_t>(handle_, edge_ids_, num_arrays_))
                  : std::nullopt;

      std::optional<rmm::device_uvector<edge_type_id_t>> edgelist_edge_types =
        edge_type_ids_
          ? std::make_optional(concatenate<edge_type_id_t>(handle_, edge_type_ids_, num_arrays_))
          : std::nullopt;

      std::tie(store_transposed ? edgelist_dsts : edgelist_srcs,
               store_transposed ? edgelist_srcs : edgelist_dsts,
               edgelist_weights,
               edgelist_edge_ids,
               edgelist_edge_types) =
        cugraph::detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
          handle_,
          std::move(store_transposed ? edgelist_dsts : edgelist_srcs),
          std::move(store_transposed ? edgelist_srcs : edgelist_dsts),
          std::move(edgelist_weights),
          std::move(edgelist_edge_ids),
          std::move(edgelist_edge_types));

      if (vertex_list) {
        vertex_list = cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
          handle_, std::move(*vertex_list));
      }

      auto graph = new cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle_);

      rmm::device_uvector<vertex_t>* number_map =
        new rmm::device_uvector<vertex_t>(0, handle_.get_stream());

      auto edge_weights = new cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        weight_t>(handle_);

      auto edge_ids = new cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        edge_t>(handle_);

      auto edge_types = new cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        edge_type_id_t>(handle_);

      if (drop_self_loops_) {
        std::tie(
          edgelist_srcs, edgelist_dsts, edgelist_weights, edgelist_edge_ids, edgelist_edge_types) =
          cugraph::remove_self_loops(handle_,
                                     std::move(edgelist_srcs),
                                     std::move(edgelist_dsts),
                                     std::move(edgelist_weights),
                                     std::move(edgelist_edge_ids),
                                     std::move(edgelist_edge_types));
      }

      if (drop_multi_edges_) {
        std::tie(
          edgelist_srcs, edgelist_dsts, edgelist_weights, edgelist_edge_ids, edgelist_edge_types) =
          cugraph::remove_multi_edges(handle_,
                                      std::move(edgelist_srcs),
                                      std::move(edgelist_dsts),
                                      std::move(edgelist_weights),
                                      std::move(edgelist_edge_ids),
                                      std::move(edgelist_edge_types));
      }

      std::tie(*graph, new_edge_weights, new_edge_ids, new_edge_types, new_number_map) =
        cugraph::create_graph_from_edgelist<vertex_t,
                                            edge_t,
                                            weight_t,
                                            edge_t,
                                            edge_type_id_t,
                                            store_transposed,
                                            multi_gpu>(
          handle_,
          std::move(vertex_list),
          std::move(edgelist_srcs),
          std::move(edgelist_dsts),
          std::move(edgelist_weights),
          std::move(edgelist_edge_ids),
          std::move(edgelist_edge_types),
          cugraph::graph_properties_t{properties_->is_symmetric, properties_->is_multigraph},
          renumber_,
          do_expensive_check_);

      if (renumber_) {
        *number_map = std::move(new_number_map.value());
      } else {
        number_map->resize(graph->number_of_vertices(), handle_.get_stream());
        cugraph::detail::sequence_fill(handle_.get_stream(),
                                       number_map->data(),
                                       number_map->size(),
                                       graph->view().local_vertex_partition_range_first());
      }

      if (new_edge_weights) { *edge_weights = std::move(new_edge_weights.value()); }
      if (new_edge_ids) { *edge_ids = std::move(new_edge_ids.value()); }
      if (new_edge_types) { *edge_types = std::move(new_edge_types.value()); }

      // Set up return
      auto result = new cugraph::c_api::cugraph_graph_t{vertex_type_,
                                                        edge_type_,
                                                        weight_type_,
                                                        edge_type_id_type_,
                                                        store_transposed,
                                                        multi_gpu,
                                                        graph,
                                                        number_map,
                                                        new_edge_weights ? edge_weights : nullptr,
                                                        new_edge_ids ? edge_ids : nullptr,
                                                        new_edge_types ? edge_types : nullptr};

      result_ = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(result);
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_graph_create_mg(
  cugraph_resource_handle_t const* handle,
  cugraph_graph_properties_t const* properties,
  cugraph_type_erased_device_array_view_t const* const* vertices,
  cugraph_type_erased_device_array_view_t const* const* src,
  cugraph_type_erased_device_array_view_t const* const* dst,
  cugraph_type_erased_device_array_view_t const* const* weights,
  cugraph_type_erased_device_array_view_t const* const* edge_ids,
  cugraph_type_erased_device_array_view_t const* const* edge_type_ids,
  bool_t store_transposed,
  size_t num_arrays,
  bool_t drop_self_loops,
  bool_t drop_multi_edges,
  bool_t do_expensive_check,
  cugraph_graph_t** graph,
  cugraph_error_t** error)
{
  constexpr bool multi_gpu = true;
  constexpr size_t int32_threshold{std::numeric_limits<int32_t>::max()};

  *graph = nullptr;
  *error = nullptr;

  auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto p_vertices =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(
      vertices);
  auto p_src =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(src);
  auto p_dst =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(dst);
  auto p_weights =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(
      weights);
  auto p_edge_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(
      edge_ids);
  auto p_edge_type_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(
      edge_type_ids);

  size_t local_num_edges{0};

  //
  // Determine the type of vertex, weight, edge_type_id across
  // multiple input arrays and acros multiple GPUs.  Also compute
  // the number of edges so we can determine what type to use for
  // edge_t
  //
  cugraph_data_type_id_t vertex_type{cugraph_data_type_id_t::NTYPES};
  cugraph_data_type_id_t weight_type{cugraph_data_type_id_t::NTYPES};

  for (size_t i = 0; i < num_arrays; ++i) {
    CAPI_EXPECTS(p_src[i]->size_ == p_dst[i]->size_,
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != dst size.",
                 *error);

    CAPI_EXPECTS(p_src[i]->type_ == p_dst[i]->type_,
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src type != dst type.",
                 *error);

    CAPI_EXPECTS((p_vertices == nullptr) || (p_src[i]->type_ == p_vertices[i]->type_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src type != vertices type.",
                 *error);

    CAPI_EXPECTS((weights == nullptr) || (p_weights[i]->size_ == p_src[i]->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != weights size.",
                 *error);

    local_num_edges += p_src[i]->size_;

    if (vertex_type == cugraph_data_type_id_t::NTYPES) vertex_type = p_src[i]->type_;

    if (weights != nullptr) {
      if (weight_type == cugraph_data_type_id_t::NTYPES) weight_type = p_weights[i]->type_;
    }

    CAPI_EXPECTS(p_src[i]->type_ == vertex_type,
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: all vertex types must match",
                 *error);

    CAPI_EXPECTS((weights == nullptr) || (p_weights[i]->type_ == weight_type),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: all weight types must match",
                 *error);
  }

  size_t num_edges = cugraph::host_scalar_allreduce(p_handle->handle_->get_comms(),
                                                    local_num_edges,
                                                    raft::comms::op_t::SUM,
                                                    p_handle->handle_->get_stream());

  auto vertex_types = cugraph::host_scalar_allgather(
    p_handle->handle_->get_comms(), static_cast<int>(vertex_type), p_handle->handle_->get_stream());

  auto weight_types = cugraph::host_scalar_allgather(
    p_handle->handle_->get_comms(), static_cast<int>(weight_type), p_handle->handle_->get_stream());

  if (vertex_type == cugraph_data_type_id_t::NTYPES) {
    // Only true if this GPU had no vertex arrays
    vertex_type = static_cast<cugraph_data_type_id_t>(
      *std::min_element(vertex_types.begin(), vertex_types.end()));
  }

  if (weight_type == cugraph_data_type_id_t::NTYPES) {
    // Only true if this GPU had no weight arrays
    weight_type = static_cast<cugraph_data_type_id_t>(
      *std::min_element(weight_types.begin(), weight_types.end()));
  }

  CAPI_EXPECTS(std::all_of(vertex_types.begin(),
                           vertex_types.end(),
                           [vertex_type](auto t) { return vertex_type == static_cast<int>(t); }),
               CUGRAPH_INVALID_INPUT,
               "different vertex type used on different GPUs",
               *error);

  CAPI_EXPECTS(std::all_of(weight_types.begin(),
                           weight_types.end(),
                           [weight_type](auto t) { return weight_type == static_cast<int>(t); }),
               CUGRAPH_INVALID_INPUT,
               "different weight type used on different GPUs",
               *error);

  cugraph_data_type_id_t edge_type;

  if (num_edges < int32_threshold) {
    edge_type = static_cast<cugraph_data_type_id_t>(vertex_types[0]);
  } else {
    edge_type = cugraph_data_type_id_t::INT64;
  }

  if (weight_type == cugraph_data_type_id_t::NTYPES) {
    weight_type = cugraph_data_type_id_t::FLOAT32;
  }

  cugraph_data_type_id_t edge_type_id_type{cugraph_data_type_id_t::NTYPES};

  for (size_t i = 0; i < num_arrays; ++i) {
    CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids[i]->type_ == edge_type),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: Edge id type must match edge type",
                 *error);

    CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids[i]->size_ == p_src[i]->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != edge id prop size",
                 *error);

    if (edge_type_ids != nullptr) {
      CAPI_EXPECTS(p_edge_type_ids[i]->size_ == p_src[i]->size_,
                   CUGRAPH_INVALID_INPUT,
                   "Invalid input arguments: src size != edge type prop size",
                   *error);

      if (edge_type_id_type == cugraph_data_type_id_t::NTYPES)
        edge_type_id_type = p_edge_type_ids[i]->type_;

      CAPI_EXPECTS(p_edge_type_ids[i]->type_ == edge_type_id_type,
                   CUGRAPH_INVALID_INPUT,
                   "Invalid input arguments: src size != edge type prop size",
                   *error);
    }
  }

  auto edge_type_id_types = cugraph::host_scalar_allgather(p_handle->handle_->get_comms(),
                                                           static_cast<int>(edge_type_id_type),
                                                           p_handle->handle_->get_stream());

  if (edge_type_id_type == cugraph_data_type_id_t::NTYPES) {
    // Only true if this GPU had no edge_type_id arrays
    edge_type_id_type = static_cast<cugraph_data_type_id_t>(
      *std::min_element(edge_type_id_types.begin(), edge_type_id_types.end()));
  }

  CAPI_EXPECTS(
    std::all_of(edge_type_id_types.begin(),
                edge_type_id_types.end(),
                [edge_type_id_type](auto t) { return edge_type_id_type == static_cast<int>(t); }),
    CUGRAPH_INVALID_INPUT,
    "different edge_type_id type used on different GPUs",
    *error);

  if (edge_type_id_type == cugraph_data_type_id_t::NTYPES) {
    edge_type_id_type = cugraph_data_type_id_t::INT32;
  }

  //
  // Now we know enough to create the graph
  //
  create_graph_functor functor(*p_handle->handle_,
                               properties,
                               vertex_type,
                               edge_type,
                               weight_type,
                               edge_type_id_type,
                               p_vertices,
                               p_src,
                               p_dst,
                               p_weights,
                               p_edge_ids,
                               p_edge_type_ids,
                               num_arrays,
                               bool_t::TRUE,
                               drop_self_loops,
                               drop_multi_edges,
                               do_expensive_check);

  try {
    cugraph::c_api::vertex_dispatcher(
      vertex_type, edge_type, weight_type, edge_type_id_type, store_transposed, multi_gpu, functor);

    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }

    *graph = reinterpret_cast<cugraph_graph_t*>(functor.result_);
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_mg_graph_create(
  cugraph_resource_handle_t const* handle,
  cugraph_graph_properties_t const* properties,
  cugraph_type_erased_device_array_view_t const* src,
  cugraph_type_erased_device_array_view_t const* dst,
  cugraph_type_erased_device_array_view_t const* weights,
  cugraph_type_erased_device_array_view_t const* edge_ids,
  cugraph_type_erased_device_array_view_t const* edge_type_ids,
  bool_t store_transposed,
  size_t num_edges,
  bool_t do_expensive_check,
  cugraph_graph_t** graph,
  cugraph_error_t** error)
{
  return cugraph_graph_create_mg(handle,
                                 properties,
                                 NULL,
                                 &src,
                                 &dst,
                                 &weights,
                                 &edge_ids,
                                 &edge_type_ids,
                                 store_transposed,
                                 1,
                                 FALSE,
                                 FALSE,
                                 do_expensive_check,
                                 graph,
                                 error);
}

extern "C" void cugraph_mg_graph_free(cugraph_graph_t* ptr_graph)
{
  if (ptr_graph != NULL) { cugraph_graph_free(ptr_graph); }
}
