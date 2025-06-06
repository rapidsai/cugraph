/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include "c_api/array.hpp"
#include "c_api/error.hpp"
#include "c_api/generic_cascaded_dispatch.hpp"
#include "c_api/graph.hpp"
#include "c_api/graph_helper.hpp"
#include "c_api/resource_handle.hpp"

#include <cugraph_c/graph.h>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <limits>

namespace {

struct create_graph_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_properties_t const* properties_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* src_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_type_ids_;
  bool_t renumber_;
  bool_t drop_self_loops_;
  bool_t drop_multi_edges_;
  bool_t symmetrize_;
  bool_t do_expensive_check_;
  cugraph_data_type_id_t edge_type_;
  cugraph::c_api::cugraph_graph_t* result_{};

  create_graph_functor(raft::handle_t const& handle,
                       cugraph_graph_properties_t const* properties,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* src,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids,
                       cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_type_ids,
                       bool_t renumber,
                       bool_t drop_self_loops,
                       bool_t drop_multi_edges,
                       bool_t symmetrize,
                       bool_t do_expensive_check,
                       cugraph_data_type_id_t edge_type)
    : abstract_functor(),
      properties_(properties),
      handle_(handle),
      vertices_(vertices),
      src_(src),
      dst_(dst),
      weights_(weights),
      edge_ids_(edge_ids),
      edge_type_ids_(edge_type_ids),
      renumber_(renumber),
      drop_self_loops_(drop_self_loops),
      drop_multi_edges_(drop_multi_edges),
      symmetrize_(symmetrize),
      do_expensive_check_(do_expensive_check),
      edge_type_(edge_type)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    using edge_time_t = int32_t;

    if constexpr (multi_gpu || !cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      if (do_expensive_check_) {
        // FIXME:  Need an implementation here.
      }

      std::optional<rmm::device_uvector<vertex_t>> new_number_map;

      std::optional<cugraph::edge_property_t<edge_t, weight_t>> new_edge_weights{std::nullopt};

      std::optional<cugraph::edge_property_t<edge_t, edge_t>> new_edge_ids{std::nullopt};

      std::optional<cugraph::edge_property_t<edge_t, edge_type_t>> new_edge_types{std::nullopt};

      std::optional<rmm::device_uvector<vertex_t>> vertex_list =
        vertices_ ? std::make_optional(
                      rmm::device_uvector<vertex_t>(vertices_->size_, handle_.get_stream()))
                  : std::nullopt;

      if (vertex_list) {
        raft::copy<vertex_t>(vertex_list->data(),
                             vertices_->as_type<vertex_t>(),
                             vertices_->size_,
                             handle_.get_stream());
      }

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

      std::optional<rmm::device_uvector<edge_t>> edgelist_edge_ids =
        edge_ids_
          ? std::make_optional(rmm::device_uvector<edge_t>(edge_ids_->size_, handle_.get_stream()))
          : std::nullopt;

      if (edgelist_edge_ids) {
        raft::copy<edge_t>(edgelist_edge_ids->data(),
                           edge_ids_->as_type<edge_t>(),
                           edge_ids_->size_,
                           handle_.get_stream());
      }

      std::optional<rmm::device_uvector<edge_type_t>> edgelist_edge_types =
        edge_type_ids_ ? std::make_optional(rmm::device_uvector<edge_type_t>(edge_type_ids_->size_,
                                                                             handle_.get_stream()))
                       : std::nullopt;

      if (edgelist_edge_types) {
        raft::copy<edge_type_t>(edgelist_edge_types->data(),
                                edge_type_ids_->as_type<edge_type_t>(),
                                edge_type_ids_->size_,
                                handle_.get_stream());
      }

      auto graph = new cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle_);

      rmm::device_uvector<vertex_t>* number_map =
        new rmm::device_uvector<vertex_t>(0, handle_.get_stream());

      auto edge_weights = new cugraph::edge_property_t<edge_t, weight_t>(handle_);

      auto edge_ids = new cugraph::edge_property_t<edge_t, edge_t>(handle_);

      auto edge_types = new cugraph::edge_property_t<edge_t, edge_type_t>(handle_);

      if (drop_self_loops_) {
        std::optional<rmm::device_uvector<edge_time_t>> dummy_start_times{std::nullopt};
        std::optional<rmm::device_uvector<edge_time_t>> dummy_end_times{std::nullopt};

        std::tie(edgelist_srcs,
                 edgelist_dsts,
                 edgelist_weights,
                 edgelist_edge_ids,
                 edgelist_edge_types,
                 std::ignore,
                 std::ignore) = cugraph::remove_self_loops(handle_,
                                                           std::move(edgelist_srcs),
                                                           std::move(edgelist_dsts),
                                                           std::move(edgelist_weights),
                                                           std::move(edgelist_edge_ids),
                                                           std::move(edgelist_edge_types),
                                                           std::move(dummy_start_times),
                                                           std::move(dummy_end_times));
      }

      if (drop_multi_edges_) {
        std::optional<rmm::device_uvector<edge_time_t>> dummy_start_times{std::nullopt};
        std::optional<rmm::device_uvector<edge_time_t>> dummy_end_times{std::nullopt};

        std::tie(edgelist_srcs,
                 edgelist_dsts,
                 edgelist_weights,
                 edgelist_edge_ids,
                 edgelist_edge_types,
                 std::ignore,
                 std::ignore) =
          cugraph::remove_multi_edges(handle_,
                                      std::move(edgelist_srcs),
                                      std::move(edgelist_dsts),
                                      std::move(edgelist_weights),
                                      std::move(edgelist_edge_ids),
                                      std::move(edgelist_edge_types),
                                      std::move(dummy_start_times),
                                      std::move(dummy_end_times),
                                      properties_->is_symmetric
                                        ? true /* keep minimum weight edges to maintain symmetry */
                                        : false);
      }

      if (symmetrize_) {
        std::optional<rmm::device_uvector<edge_time_t>> dummy_start_times{std::nullopt};
        std::optional<rmm::device_uvector<edge_time_t>> dummy_end_times{std::nullopt};

        // Symmetrize the edgelist
        std::tie(edgelist_srcs,
                 edgelist_dsts,
                 edgelist_weights,
                 edgelist_edge_ids,
                 edgelist_edge_types,
                 std::ignore,
                 std::ignore) =
          cugraph::symmetrize_edgelist<vertex_t,
                                       edge_t,
                                       weight_t,
                                       edge_type_t,
                                       edge_time_t,
                                       store_transposed,
                                       multi_gpu>(handle_,
                                                  std::move(edgelist_srcs),
                                                  std::move(edgelist_dsts),
                                                  std::move(edgelist_weights),
                                                  std::move(edgelist_edge_ids),
                                                  std::move(edgelist_edge_types),
                                                  std::move(dummy_start_times),
                                                  std::move(dummy_end_times),
                                                  false);
      }

      std::optional<rmm::device_uvector<edge_time_t>> dummy_start_times{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> dummy_end_times{std::nullopt};

      std::tie(*graph,
               new_edge_weights,
               new_edge_ids,
               new_edge_types,
               std::ignore,
               std::ignore,
               new_number_map) = cugraph::create_graph_from_edgelist<vertex_t,
                                                                     edge_t,
                                                                     weight_t,
                                                                     edge_type_t,
                                                                     edge_time_t,
                                                                     store_transposed,
                                                                     multi_gpu>(
        handle_,
        std::move(vertex_list),
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::move(edgelist_weights),
        std::move(edgelist_edge_ids),
        std::move(edgelist_edge_types),
        std::move(dummy_start_times),
        std::move(dummy_end_times),
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

        if (vertices_) {
          vertex_list = rmm::device_uvector<vertex_t>(vertices_->size_, handle_.get_stream());
          raft::copy<vertex_t>(vertex_list->data(),
                               vertices_->as_type<vertex_t>(),
                               vertices_->size_,
                               handle_.get_stream());

          auto is_consecutive = cugraph::detail::is_equal(
            handle_.get_stream(),
            raft::device_span<vertex_t>{vertex_list->data(), vertex_list->size()},
            raft::device_span<vertex_t>{number_map->data(), number_map->size()});

          if (!is_consecutive) {
            mark_error(
              CUGRAPH_INVALID_INPUT,
              "Vertex list must be numbered consecutively from 0 when 'renumber' is 'false'");
            return;
          }
        }
      }

      if (new_edge_weights) { *edge_weights = std::move(new_edge_weights.value()); }
      if (new_edge_ids) { *edge_ids = std::move(new_edge_ids.value()); }
      if (new_edge_types) { *edge_types = std::move(new_edge_types.value()); }

      // Set up return
      auto result = new cugraph::c_api::cugraph_graph_t{
        src_->type_,
        edge_type_,
        weights_ ? weights_->type_ : cugraph_data_type_id_t::FLOAT32,
        edge_type_ids_ ? edge_type_ids_->type_ : cugraph_data_type_id_t::INT32,
        store_transposed,
        multi_gpu,
        graph,
        number_map,
        new_edge_weights ? edge_weights : nullptr,
        new_edge_ids ? edge_ids : nullptr,
        new_edge_types ? edge_types : nullptr,
        nullptr,
        nullptr};

      result_ = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(result);
    }
  }
};

struct create_graph_csr_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_properties_t const* properties_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* offsets_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* indices_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_type_ids_;
  bool_t renumber_;
  bool_t symmetrize_;
  bool_t do_expensive_check_;
  cugraph::c_api::cugraph_graph_t* result_{};

  create_graph_csr_functor(
    raft::handle_t const& handle,
    cugraph_graph_properties_t const* properties,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* offsets,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* indices,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids,
    cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_type_ids,
    bool_t renumber,
    bool_t symmetrize,
    bool_t do_expensive_check)
    : abstract_functor(),
      properties_(properties),
      handle_(handle),
      offsets_(offsets),
      indices_(indices),
      weights_(weights),
      edge_ids_(edge_ids),
      edge_type_ids_(edge_type_ids),
      renumber_(renumber),
      symmetrize_(symmetrize),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    using edge_time_t = int32_t;

    if constexpr (multi_gpu || !cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      if (do_expensive_check_) {
        // FIXME:  Need an implementation here.
      }

      std::optional<rmm::device_uvector<vertex_t>> new_number_map;

      std::optional<cugraph::edge_property_t<edge_t, weight_t>> new_edge_weights{std::nullopt};

      std::optional<cugraph::edge_property_t<edge_t, edge_t>> new_edge_ids{std::nullopt};

      std::optional<cugraph::edge_property_t<edge_t, edge_type_t>> new_edge_types{std::nullopt};

      std::optional<rmm::device_uvector<vertex_t>> vertex_list = std::make_optional(
        rmm::device_uvector<vertex_t>(offsets_->size_ - 1, handle_.get_stream()));

      cugraph::detail::sequence_fill(
        handle_.get_stream(), vertex_list->data(), vertex_list->size(), vertex_t{0});

      rmm::device_uvector<vertex_t> edgelist_srcs(0, handle_.get_stream());
      rmm::device_uvector<vertex_t> edgelist_dsts(indices_->size_, handle_.get_stream());

      edgelist_srcs = cugraph::c_api::expand_sparse_offsets(
        raft::device_span<edge_t const>{offsets_->as_type<edge_t>(), offsets_->size_},
        vertex_t{0},
        handle_.get_stream());
      raft::copy<vertex_t>(
        edgelist_dsts.data(), indices_->as_type<vertex_t>(), indices_->size_, handle_.get_stream());

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

      std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>
        edgelist_edge_tuple{};

      std::optional<rmm::device_uvector<edge_t>> edgelist_edge_ids =
        edge_ids_
          ? std::make_optional(rmm::device_uvector<edge_t>(edge_ids_->size_, handle_.get_stream()))
          : std::nullopt;

      if (edgelist_edge_ids) {
        raft::copy<edge_t>(edgelist_edge_ids->data(),
                           edge_ids_->as_type<edge_t>(),
                           edge_ids_->size_,
                           handle_.get_stream());
      }

      std::optional<rmm::device_uvector<edge_type_t>> edgelist_edge_types =
        edge_type_ids_ ? std::make_optional(rmm::device_uvector<edge_type_t>(edge_type_ids_->size_,
                                                                             handle_.get_stream()))
                       : std::nullopt;

      if (edgelist_edge_types) {
        raft::copy<edge_type_t>(edgelist_edge_types->data(),
                                edge_type_ids_->as_type<edge_type_t>(),
                                edge_type_ids_->size_,
                                handle_.get_stream());
      }

      std::optional<rmm::device_uvector<edge_time_t>> edgelist_edge_start_times{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> edgelist_edge_end_times{std::nullopt};

      auto graph = new cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle_);

      rmm::device_uvector<vertex_t>* number_map =
        new rmm::device_uvector<vertex_t>(0, handle_.get_stream());

      auto edge_weights = new cugraph::edge_property_t<edge_t, weight_t>(handle_);

      auto edge_ids = new cugraph::edge_property_t<edge_t, edge_t>(handle_);

      auto edge_types = new cugraph::edge_property_t<edge_t, edge_type_t>(handle_);

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
                                       edge_time_t,
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

      std::tie(*graph,
               new_edge_weights,
               new_edge_ids,
               new_edge_types,
               std::ignore,
               std::ignore,
               new_number_map) = cugraph::create_graph_from_edgelist<vertex_t,
                                                                     edge_t,
                                                                     weight_t,
                                                                     edge_type_t,
                                                                     edge_time_t,
                                                                     store_transposed,
                                                                     multi_gpu>(
        handle_,
        std::move(vertex_list),
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::move(edgelist_weights),
        std::move(edgelist_edge_ids),
        std::move(edgelist_edge_types),
        std::move(edgelist_edge_start_times),
        std::move(edgelist_edge_end_times),
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
      auto result = new cugraph::c_api::cugraph_graph_t{
        indices_->type_,
        offsets_->type_,
        weights_ ? weights_->type_ : cugraph_data_type_id_t::FLOAT32,
        edge_type_ids_ ? edge_type_ids_->type_ : cugraph_data_type_id_t::INT32,
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

struct destroy_graph_functor : public cugraph::c_api::abstract_functor {
  void* graph_;
  void* number_map_;
  void* edge_weights_;
  void* edge_ids_;
  void* edge_types_;

  destroy_graph_functor(
    void* graph, void* number_map, void* edge_weights, void* edge_ids, void* edge_types)
    : abstract_functor(),
      graph_(graph),
      number_map_(number_map),
      edge_weights_(edge_weights),
      edge_ids_(edge_ids),
      edge_types_(edge_types)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    auto internal_graph_pointer =
      reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(graph_);

    delete internal_graph_pointer;

    auto internal_number_map_pointer =
      reinterpret_cast<rmm::device_uvector<vertex_t>*>(number_map_);

    delete internal_number_map_pointer;

    auto internal_edge_weight_pointer =
      reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(edge_weights_);
    if (internal_edge_weight_pointer) { delete internal_edge_weight_pointer; }

    auto internal_edge_id_pointer =
      reinterpret_cast<cugraph::edge_property_t<edge_t, edge_t>*>(edge_ids_);
    if (internal_edge_id_pointer) { delete internal_edge_id_pointer; }

    auto internal_edge_type_pointer =
      reinterpret_cast<cugraph::edge_property_t<edge_t, edge_type_t>*>(edge_types_);
    if (internal_edge_type_pointer) { delete internal_edge_type_pointer; }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_graph_create_sg(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_device_array_view_t* edge_ids,
  const cugraph_type_erased_device_array_view_t* edge_type_ids,
  bool_t store_transposed,
  bool_t renumber,
  bool_t drop_self_loops,
  bool_t drop_multi_edges,
  bool_t symmetrize,
  bool_t do_expensive_check,
  cugraph_graph_t** graph,
  cugraph_error_t** error)
{
  constexpr bool multi_gpu = false;
  constexpr size_t int32_threshold{std::numeric_limits<int32_t>::max()};

  *graph = nullptr;
  *error = nullptr;

  auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto p_vertices =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(vertices);
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

  if (symmetrize == TRUE) {
    CAPI_EXPECTS((properties->is_symmetric == TRUE),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: The graph property must be symmetric if 'symmetrize' is "
                 "set to True.",
                 *error);
  }

  CAPI_EXPECTS(p_src->size_ == p_dst->size_,
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != dst size.",
               *error);

  CAPI_EXPECTS((p_vertices == nullptr) || (p_src->type_ == p_vertices->type_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src type != vertices type.",
               *error);

  CAPI_EXPECTS(p_src->type_ == p_dst->type_,
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src type != dst type.",
               *error);

  CAPI_EXPECTS((weights == nullptr) || (p_weights->size_ == p_src->size_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != weights size.",
               *error);

  if (p_src->type_ == cugraph_data_type_id_t::INT32)
    CAPI_EXPECTS(p_src->size_ < int32_threshold,
                 CUGRAPH_INVALID_INPUT,
                 "Number of edges won't fit in 32-bit integer, using 32-bit type",
                 *error);

  cugraph_data_type_id_t edge_type = p_src->type_;
  cugraph_data_type_id_t weight_type;

  if (weights != nullptr) {
    weight_type = p_weights->type_;
  } else {
    weight_type = cugraph_data_type_id_t::FLOAT32;
  }

  CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids->type_ == edge_type),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: Edge id type must match edge type",
               *error);

  CAPI_EXPECTS((edge_ids == nullptr) || (p_edge_ids->size_ == p_src->size_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != edge id prop size",
               *error);

  CAPI_EXPECTS((edge_type_ids == nullptr) || (p_edge_type_ids->size_ == p_src->size_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != edge type prop size",
               *error);

  cugraph_data_type_id_t edge_type_id_type = cugraph_data_type_id_t::INT32;
  if (edge_type_ids != nullptr) { edge_type_id_type = p_edge_type_ids->type_; }

  ::create_graph_functor functor(*p_handle->handle_,
                                 properties,
                                 p_vertices,
                                 p_src,
                                 p_dst,
                                 p_weights,
                                 p_edge_ids,
                                 p_edge_type_ids,
                                 renumber,
                                 drop_self_loops,
                                 drop_multi_edges,
                                 symmetrize,
                                 do_expensive_check,
                                 edge_type);

  try {
    cugraph::c_api::vertex_dispatcher(p_src->type_,
                                      edge_type,
                                      weight_type,
                                      edge_type_id_type,
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

cugraph_error_code_t cugraph_graph_create_sg_from_csr(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_view_t* offsets,
  const cugraph_type_erased_device_array_view_t* indices,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_device_array_view_t* edge_ids,
  const cugraph_type_erased_device_array_view_t* edge_type_ids,
  bool_t store_transposed,
  bool_t renumber,
  bool_t symmetrize,
  bool_t do_expensive_check,
  cugraph_graph_t** graph,
  cugraph_error_t** error)
{
  constexpr bool multi_gpu = false;

  *graph = nullptr;
  *error = nullptr;

  auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto p_offsets =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(offsets);
  auto p_indices =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(indices);
  auto p_weights =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(weights);
  auto p_edge_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(edge_ids);
  auto p_edge_type_ids =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(edge_type_ids);

  CAPI_EXPECTS((weights == nullptr) || (p_weights->size_ == p_indices->size_),
               CUGRAPH_INVALID_INPUT,
               "Invalid input arguments: src size != weights size.",
               *error);

  cugraph_data_type_id_t weight_type;

  if (weights != nullptr) {
    weight_type = p_weights->type_;
  } else {
    weight_type = cugraph_data_type_id_t::FLOAT32;
  }

  if (symmetrize == TRUE) {
    CAPI_EXPECTS((properties->is_symmetric == TRUE),
                 CUGRAPH_INVALID_INPUT,
                 "Invalid input arguments: The graph property must be symmetric if 'symmetrize' is "
                 "set to True.",
                 *error);
  }

  CAPI_EXPECTS(
    (edge_type_ids == nullptr && edge_ids == nullptr) ||
      (edge_type_ids != nullptr && edge_ids != nullptr),
    CUGRAPH_INVALID_INPUT,
    "Invalid input arguments: either none or both of edge ids and edge types must be provided.",
    *error);

  CAPI_EXPECTS(
    (edge_type_ids == nullptr && edge_ids == nullptr) || (p_edge_ids->type_ == p_offsets->type_),
    CUGRAPH_INVALID_INPUT,
    "Invalid input arguments: Edge id type must match edge type",
    *error);

  CAPI_EXPECTS(
    (edge_type_ids == nullptr && edge_ids == nullptr) ||
      (p_edge_ids->size_ == p_indices->size_ && p_edge_type_ids->size_ == p_indices->size_),
    CUGRAPH_INVALID_INPUT,
    "Invalid input arguments: src size != edge prop size",
    *error);

  ::create_graph_csr_functor functor(*p_handle->handle_,
                                     properties,
                                     p_offsets,
                                     p_indices,
                                     p_weights,
                                     p_edge_ids,
                                     p_edge_type_ids,
                                     renumber,
                                     symmetrize,
                                     do_expensive_check);

  try {
    cugraph::c_api::vertex_dispatcher(p_indices->type_,
                                      p_offsets->type_,
                                      weight_type,
                                      p_indices->type_,
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

extern "C" void cugraph_graph_free(cugraph_graph_t* ptr_graph)
{
  if (ptr_graph != NULL) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(ptr_graph);

    destroy_graph_functor functor(internal_pointer->graph_,
                                  internal_pointer->number_map_,
                                  internal_pointer->edge_weights_,
                                  internal_pointer->edge_ids_,
                                  internal_pointer->edge_types_);

    cugraph::c_api::vertex_dispatcher(internal_pointer->vertex_type_,
                                      internal_pointer->edge_type_,
                                      internal_pointer->weight_type_,
                                      internal_pointer->edge_type_id_type_,
                                      internal_pointer->store_transposed_,
                                      internal_pointer->multi_gpu_,
                                      functor);

    delete internal_pointer;
  }
}
