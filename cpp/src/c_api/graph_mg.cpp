/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/array.hpp"
#include "c_api/error.hpp"
#include "c_api/generic_cascaded_dispatch.hpp"
#include "c_api/graph.hpp"
#include "c_api/graph_helper.hpp"
#include "c_api/resource_handle.hpp"
#include "cugraph/utilities/host_scalar_comm.hpp"
#include "cugraph_c/types.h"

#include <cugraph_c/graph.h>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>

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
    cugraph::c_api::copy_or_transform(
      raft::device_span<value_t>{results.data() + concat_pos, values[i]->size_},
      values[i],
      handle.get_stream());
    concat_pos += values[i]->size_;
  }

  return results;
}

struct create_graph_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_properties_t const* properties_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* vertices_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* src_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* dst_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* weights_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_ids_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_type_ids_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_start_times_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_end_times_;
  size_t num_arrays_;
  bool_t drop_self_loops_;
  bool_t drop_multi_edges_;
  bool_t symmetrize_;
  bool_t do_expensive_check_;
  cugraph::c_api::cugraph_graph_t* result_{};

  create_graph_functor(
    raft::handle_t const& handle,
    cugraph_graph_properties_t const* properties,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* vertices,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* src,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* dst,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* weights,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_ids,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_type_ids,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_start_times,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* const* edge_end_times,
    size_t num_arrays,
    bool_t drop_self_loops,
    bool_t drop_multi_edges,
    bool_t symmetrize,
    bool_t do_expensive_check)
    : abstract_functor(),
      properties_(properties),
      handle_(handle),
      vertices_(vertices),
      src_(src),
      dst_(dst),
      weights_(weights),
      edge_ids_(edge_ids),
      edge_type_ids_(edge_type_ids),
      edge_start_times_(edge_start_times),
      edge_end_times_(edge_end_times),
      num_arrays_(num_arrays),
      drop_self_loops_(drop_self_loops),
      drop_multi_edges_(drop_multi_edges),
      symmetrize_(symmetrize),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!multi_gpu || !cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      std::optional<rmm::device_uvector<vertex_t>> vertex_list =
        vertices_ ? std::make_optional(concatenate<vertex_t>(handle_, vertices_, num_arrays_))
                  : std::nullopt;

      rmm::device_uvector<vertex_t> edgelist_srcs =
        concatenate<vertex_t>(handle_, src_, num_arrays_);
      rmm::device_uvector<vertex_t> edgelist_dsts =
        concatenate<vertex_t>(handle_, dst_, num_arrays_);

      std::vector<cugraph::arithmetic_device_uvector_t> edgelist_edge_properties{};

      if (weights_)
        edgelist_edge_properties.push_back(concatenate<weight_t>(handle_, weights_, num_arrays_));
      if (edge_ids_)
        edgelist_edge_properties.push_back(concatenate<edge_t>(handle_, edge_ids_, num_arrays_));
      if (edge_type_ids_)
        edgelist_edge_properties.push_back(
          concatenate<edge_type_t>(handle_, edge_type_ids_, num_arrays_));
      if (edge_start_times_)
        edgelist_edge_properties.push_back(
          concatenate<time_stamp_t>(handle_, edge_start_times_, num_arrays_));
      if (edge_end_times_)
        edgelist_edge_properties.push_back(
          concatenate<time_stamp_t>(handle_, edge_end_times_, num_arrays_));

      std::tie(edgelist_srcs, edgelist_dsts, edgelist_edge_properties) =
        cugraph::shuffle_ext_edges(handle_,
                                   std::move(edgelist_srcs),
                                   std::move(edgelist_dsts),
                                   std::move(edgelist_edge_properties),
                                   store_transposed);

      size_t pos{0};
      auto edgelist_weights =
        weights_ ? std::make_optional(std::move(
                     std::get<rmm::device_uvector<weight_t>>(edgelist_edge_properties[pos++])))
                 : std::nullopt;
      auto edgelist_edge_ids =
        edge_ids_ ? std::make_optional(std::move(
                      std::get<rmm::device_uvector<edge_t>>(edgelist_edge_properties[pos++])))
                  : std::nullopt;
      auto edgelist_edge_types =
        edge_type_ids_ ? std::make_optional(std::move(std::get<rmm::device_uvector<edge_type_t>>(
                           edgelist_edge_properties[pos++])))
                       : std::nullopt;
      auto edgelist_edge_start_times =
        edge_start_times_
          ? std::make_optional(std::move(
              std::get<rmm::device_uvector<time_stamp_t>>(edgelist_edge_properties[pos++])))
          : std::nullopt;
      auto edgelist_edge_end_times =
        edge_end_times_ ? std::make_optional(std::move(std::get<rmm::device_uvector<time_stamp_t>>(
                            edgelist_edge_properties[pos++])))
                        : std::nullopt;

      if (vertex_list) {
        std::tie(vertex_list, std::ignore) = cugraph::shuffle_ext_vertices(
          handle_, std::move(*vertex_list), std::vector<cugraph::arithmetic_device_uvector_t>{});
      }

      if (drop_self_loops_) {
        std::tie(edgelist_srcs,
                 edgelist_dsts,
                 edgelist_weights,
                 edgelist_edge_ids,
                 edgelist_edge_types,
                 edgelist_edge_start_times,
                 edgelist_edge_end_times) =
          cugraph::remove_self_loops(handle_,
                                     std::move(edgelist_srcs),
                                     std::move(edgelist_dsts),
                                     std::move(edgelist_weights),
                                     std::move(edgelist_edge_ids),
                                     std::move(edgelist_edge_types),
                                     std::move(edgelist_edge_start_times),
                                     std::move(edgelist_edge_end_times));
      }

      if (drop_multi_edges_) {
        std::tie(edgelist_srcs,
                 edgelist_dsts,
                 edgelist_weights,
                 edgelist_edge_ids,
                 edgelist_edge_types,
                 edgelist_edge_start_times,
                 edgelist_edge_end_times) =
          cugraph::remove_multi_edges(handle_,
                                      std::move(edgelist_srcs),
                                      std::move(edgelist_dsts),
                                      std::move(edgelist_weights),
                                      std::move(edgelist_edge_ids),
                                      std::move(edgelist_edge_types),
                                      std::move(edgelist_edge_start_times),
                                      std::move(edgelist_edge_end_times),
                                      properties_->is_symmetric
                                        ? true /* keep minimum weight edges to maintain symmetry */
                                        : false);
      }

      if (symmetrize_) {
        // Symmetrize the edgelist
        std::tie(edgelist_srcs,
                 edgelist_dsts,
                 edgelist_weights,
                 edgelist_edge_ids,
                 edgelist_edge_types,
                 edgelist_edge_start_times,
                 edgelist_edge_end_times) =
          cugraph::symmetrize_edgelist<vertex_t,
                                       edge_t,
                                       weight_t,
                                       edge_type_t,
                                       time_stamp_t,
                                       store_transposed,
                                       multi_gpu>(handle_,
                                                  std::move(edgelist_srcs),
                                                  std::move(edgelist_dsts),
                                                  std::move(edgelist_weights),
                                                  std::move(edgelist_edge_ids),
                                                  std::move(edgelist_edge_types),
                                                  std::move(edgelist_edge_start_times),
                                                  std::move(edgelist_edge_end_times),
                                                  false);
      }

      auto graph = new cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle_);

      std::optional<rmm::device_uvector<vertex_t>> new_number_map;

      edgelist_edge_properties.clear();
      if (edgelist_weights) edgelist_edge_properties.push_back(std::move(*edgelist_weights));
      if (edgelist_edge_ids) edgelist_edge_properties.push_back(std::move(*edgelist_edge_ids));
      if (edgelist_edge_types) edgelist_edge_properties.push_back(std::move(*edgelist_edge_types));
      if (edgelist_edge_start_times)
        edgelist_edge_properties.push_back(std::move(*edgelist_edge_start_times));
      if (edgelist_edge_end_times)
        edgelist_edge_properties.push_back(std::move(*edgelist_edge_end_times));

      std::vector<cugraph::edge_arithmetic_property_t<edge_t>> new_edge_properties{};

      std::tie(*graph, new_edge_properties, new_number_map) =
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
          handle_,
          std::move(vertex_list),
          std::move(edgelist_srcs),
          std::move(edgelist_dsts),
          std::move(edgelist_edge_properties),
          cugraph::graph_properties_t{properties_->is_symmetric, properties_->is_multigraph},
          true,
          std::nullopt,
          std::nullopt,
          do_expensive_check_);

      rmm::device_uvector<vertex_t>* number_map =
        new rmm::device_uvector<vertex_t>(std::move(new_number_map.value()));

      cugraph::edge_property_t<edge_t, weight_t>* edge_weights_property{nullptr};
      cugraph::edge_property_t<edge_t, edge_t>* edge_ids_property{nullptr};
      cugraph::edge_property_t<edge_t, edge_type_t>* edge_types_property{nullptr};
      cugraph::edge_property_t<edge_t, time_stamp_t>* edge_start_times_property{nullptr};
      cugraph::edge_property_t<edge_t, time_stamp_t>* edge_end_times_property{nullptr};

      {
        size_t pos = 0;
        if (edgelist_weights) {
          edge_weights_property = new cugraph::edge_property_t<edge_t, weight_t>(std::move(
            std::get<cugraph::edge_property_t<edge_t, weight_t>>(new_edge_properties[pos++])));
        }

        if (edgelist_edge_ids) {
          edge_ids_property = new cugraph::edge_property_t<edge_t, edge_t>(std::move(
            std::get<cugraph::edge_property_t<edge_t, edge_t>>(new_edge_properties[pos++])));
        }

        if (edgelist_edge_types) {
          edge_types_property = new cugraph::edge_property_t<edge_t, edge_type_t>(std::move(
            std::get<cugraph::edge_property_t<edge_t, edge_type_t>>(new_edge_properties[pos++])));
        }
        if (edgelist_edge_start_times) {
          edge_start_times_property = new cugraph::edge_property_t<edge_t, time_stamp_t>(std::move(
            std::get<cugraph::edge_property_t<edge_t, time_stamp_t>>(new_edge_properties[pos++])));
        }
        if (edgelist_edge_end_times) {
          edge_end_times_property = new cugraph::edge_property_t<edge_t, time_stamp_t>(std::move(
            std::get<cugraph::edge_property_t<edge_t, time_stamp_t>>(new_edge_properties[pos++])));
        }
      }

      // Set up return
      auto result =
        new cugraph::c_api::cugraph_graph_t{cugraph::c_api::data_type_id<vertex_t>::id,
                                            cugraph::c_api::data_type_id<edge_t>::id,
                                            cugraph::c_api::data_type_id<weight_t>::id,
                                            cugraph::c_api::data_type_id<edge_type_t>::id,
                                            cugraph::c_api::data_type_id<time_stamp_t>::id,
                                            store_transposed,
                                            multi_gpu,
                                            graph,
                                            number_map,
                                            edge_weights_property,
                                            edge_ids_property,
                                            edge_types_property,
                                            edge_start_times_property,
                                            edge_end_times_property};

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
  bool_t symmetrize,
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

    CAPI_EXPECTS((weights == nullptr) || (p_weights[i]->size_ == p_src[i]->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != weights size.",
                 *error);

    local_num_edges += p_src[i]->size_;

    // FIXME: Might be better to move this out of the for loop
    bool cast_vertex_t = false;
    if (vertex_type == cugraph_data_type_id_t::NTYPES) vertex_type = p_src[i]->type_;

    if (!((p_vertices == nullptr) || (p_src[i]->type_ == p_vertices[i]->type_))) {
      cast_vertex_t = true;
    }

    if (!((p_edge_ids == nullptr) || (p_src[i]->type_ == p_edge_ids[i]->type_))) {
      cast_vertex_t = true;
    }

    if (!(p_src[i]->type_ == p_dst[i]->type_)) { cast_vertex_t = true; }

    if (cast_vertex_t) { vertex_type = cugraph_data_type_id_t::INT64; }

    if (weights != nullptr) {
      if (weight_type == cugraph_data_type_id_t::NTYPES) weight_type = p_weights[i]->type_;
    }

    if (symmetrize == TRUE) {
      CAPI_EXPECTS((properties->is_symmetric == TRUE),
                   CUGRAPH_INVALID_INPUT,
                   "Invalid input arguments: The graph property must be symmetric if 'symmetrize' "
                   "is set to True.",
                   *error);
    }

    CAPI_EXPECTS((weights == nullptr) || (p_weights[i]->type_ == weight_type),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: all weight types must match",
                 *error);
  }

  size_t num_edges = cugraph::host_scalar_allreduce(p_handle->handle_->get_comms(),
                                                    local_num_edges,
                                                    raft::comms::op_t::SUM,
                                                    p_handle->handle_->get_stream());

  cugraph_data_type_id_t edge_type{vertex_type};

  if (vertex_type == cugraph_data_type_id_t::INT32)
    CAPI_EXPECTS(num_edges < int32_threshold,
                 CUGRAPH_INVALID_INPUT,
                 "Number of edges won't fit in 32-bit integer, using 32-bit type",
                 *error);

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

  if (weight_type == cugraph_data_type_id_t::NTYPES) {
    weight_type = cugraph_data_type_id_t::FLOAT32;
  }

  cugraph_data_type_id_t edge_type_id_type{cugraph_data_type_id_t::NTYPES};

  for (size_t i = 0; i < num_arrays; ++i) {
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
                               p_vertices,
                               p_src,
                               p_dst,
                               p_weights,
                               p_edge_ids,
                               p_edge_type_ids,
                               nullptr,
                               nullptr,
                               num_arrays,
                               drop_self_loops,
                               drop_multi_edges,
                               symmetrize,
                               do_expensive_check);

  try {
    cugraph::c_api::vertex_dispatcher(vertex_type,
                                      edge_type,
                                      weight_type,
                                      edge_type_id_type,
                                      INT32,
                                      store_transposed,
                                      multi_gpu,
                                      functor);

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

extern "C" cugraph_error_code_t cugraph_graph_create_with_times_mg(
  cugraph_resource_handle_t const* handle,
  cugraph_graph_properties_t const* properties,
  cugraph_type_erased_device_array_view_t const* const* vertices,
  cugraph_type_erased_device_array_view_t const* const* src,
  cugraph_type_erased_device_array_view_t const* const* dst,
  cugraph_type_erased_device_array_view_t const* const* weights,
  cugraph_type_erased_device_array_view_t const* const* edge_ids,
  cugraph_type_erased_device_array_view_t const* const* edge_type_ids,
  cugraph_type_erased_device_array_view_t const* const* edge_start_times,
  cugraph_type_erased_device_array_view_t const* const* edge_end_times,
  bool_t store_transposed,
  size_t num_arrays,
  bool_t drop_self_loops,
  bool_t drop_multi_edges,
  bool_t symmetrize,
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
  auto p_edge_start_times =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(
      edge_start_times);
  auto p_edge_end_times =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const* const*>(
      edge_end_times);

  size_t local_num_edges{0};

  //
  // Determine the type of vertex, weight, edge_type_id across
  // multiple input arrays and acros multiple GPUs.  Also compute
  // the number of edges so we can determine what type to use for
  // edge_t
  //
  cugraph_data_type_id_t vertex_type{cugraph_data_type_id_t::NTYPES};
  cugraph_data_type_id_t weight_type{cugraph_data_type_id_t::NTYPES};
  cugraph_data_type_id_t edge_time_type{cugraph_data_type_id_t::NTYPES};

  for (size_t i = 0; i < num_arrays; ++i) {
    CAPI_EXPECTS(p_src[i]->size_ == p_dst[i]->size_,
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != dst size.",
                 *error);

    CAPI_EXPECTS((weights == nullptr) || (p_weights[i]->size_ == p_src[i]->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != weights size.",
                 *error);

    local_num_edges += p_src[i]->size_;

    // FIXME: Might be better to move this out of the for loop
    bool cast_vertex_t = false;
    if (vertex_type == cugraph_data_type_id_t::NTYPES) vertex_type = p_src[i]->type_;

    if (!((p_vertices == nullptr) || (p_src[i]->type_ == p_vertices[i]->type_))) {
      cast_vertex_t = true;
    }

    if (!((p_edge_ids == nullptr) || (p_src[i]->type_ == p_edge_ids[i]->type_))) {
      cast_vertex_t = true;
    }

    if (!(p_src[i]->type_ == p_dst[i]->type_)) { cast_vertex_t = true; }

    if (cast_vertex_t) { vertex_type = cugraph_data_type_id_t::INT64; }

    if (weights != nullptr) {
      if (weight_type == cugraph_data_type_id_t::NTYPES) weight_type = p_weights[i]->type_;
    }

    if (symmetrize == TRUE) {
      CAPI_EXPECTS((properties->is_symmetric == TRUE),
                   CUGRAPH_INVALID_INPUT,
                   "Invalid input arguments: The graph property must be symmetric if 'symmetrize' "
                   "is set to True.",
                   *error);
    }

    CAPI_EXPECTS((weights == nullptr) || (p_weights[i]->type_ == weight_type),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: all weight types must match",
                 *error);

    if ((edge_time_type == cugraph_data_type_id_t::NTYPES) && (p_edge_start_times != nullptr)) {
      edge_time_type = p_edge_start_times[i]->type_;
    }

    if (!((p_edge_end_times == nullptr) ||
          (p_edge_start_times[i]->type_ == p_edge_end_times[i]->type_))) {
      edge_time_type = cugraph_data_type_id_t::INT64;
    }
  }

  size_t num_edges = cugraph::host_scalar_allreduce(p_handle->handle_->get_comms(),
                                                    local_num_edges,
                                                    raft::comms::op_t::SUM,
                                                    p_handle->handle_->get_stream());

  cugraph_data_type_id_t edge_type{vertex_type};

  if (vertex_type == cugraph_data_type_id_t::INT32)
    CAPI_EXPECTS(num_edges < int32_threshold,
                 CUGRAPH_INVALID_INPUT,
                 "Number of edges won't fit in 32-bit integer, using 32-bit type",
                 *error);

  auto vertex_types = cugraph::host_scalar_allgather(
    p_handle->handle_->get_comms(), static_cast<int>(vertex_type), p_handle->handle_->get_stream());

  auto weight_types = cugraph::host_scalar_allgather(
    p_handle->handle_->get_comms(), static_cast<int>(weight_type), p_handle->handle_->get_stream());

  auto time_types = cugraph::host_scalar_allgather(p_handle->handle_->get_comms(),
                                                   static_cast<int>(edge_time_type),
                                                   p_handle->handle_->get_stream());

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

  if (edge_time_type == cugraph_data_type_id_t::NTYPES) {
    // Only true if this GPU had no weight arrays
    edge_time_type =
      static_cast<cugraph_data_type_id_t>(*std::min_element(time_types.begin(), time_types.end()));
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

  CAPI_EXPECTS(
    std::all_of(time_types.begin(),
                time_types.end(),
                [edge_time_type](auto t) { return edge_time_type == static_cast<int>(t); }),
    CUGRAPH_INVALID_INPUT,
    "different time type used on different GPUs",
    *error);

  if (weight_type == cugraph_data_type_id_t::NTYPES) {
    weight_type = cugraph_data_type_id_t::FLOAT32;
  }

  if (edge_time_type == cugraph_data_type_id_t::NTYPES) {
    edge_time_type = cugraph_data_type_id_t::INT32;
  }

  cugraph_data_type_id_t edge_type_id_type{cugraph_data_type_id_t::NTYPES};

  for (size_t i = 0; i < num_arrays; ++i) {
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

    CAPI_EXPECTS((edge_start_times == nullptr) || (p_edge_start_times[i]->size_ == p_src[i]->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != edge start time prop size",
                 *error);

    CAPI_EXPECTS((edge_end_times == nullptr) || (p_edge_end_times[i]->size_ == p_src[i]->size_),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: src size != edge end time prop size",
                 *error);
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
                               p_vertices,
                               p_src,
                               p_dst,
                               p_weights,
                               p_edge_ids,
                               p_edge_type_ids,
                               p_edge_start_times,
                               p_edge_end_times,
                               num_arrays,
                               drop_self_loops,
                               drop_multi_edges,
                               symmetrize,
                               do_expensive_check);

  try {
    cugraph::c_api::vertex_dispatcher(vertex_type,
                                      edge_type,
                                      weight_type,
                                      edge_type_id_type,
                                      edge_time_type,
                                      store_transposed,
                                      multi_gpu,
                                      functor);

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
