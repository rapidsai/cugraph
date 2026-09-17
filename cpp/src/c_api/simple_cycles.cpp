/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/error.hpp"
#include "c_api/graph.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/simple_cycles_result.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/components_algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>

#include <limits>
#include <optional>

namespace {

struct simple_cycles_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* seed_vertices_{nullptr};
  size_t length_bound_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_simple_cycles_result_t* result_{nullptr};

  simple_cycles_functor(cugraph_resource_handle_t const* handle,
                        cugraph_graph_t* graph,
                        cugraph_type_erased_device_array_view_t const* seed_vertices,
                        size_t length_bound,
                        bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      seed_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          seed_vertices)),
      length_bound_(length_bound),
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
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      if (length_bound_ > static_cast<size_t>(std::numeric_limits<vertex_t>::max())) {
        mark_error(CUGRAPH_INVALID_INPUT, "length_bound exceeds the maximum value for vertex_t");
        return;
      }
      auto const length_bound_v = static_cast<vertex_t>(length_bound_);

      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      std::optional<rmm::device_uvector<vertex_t>> seed_vertices{std::nullopt};

      if (seed_vertices_ != nullptr) {
        seed_vertices = rmm::device_uvector<vertex_t>{seed_vertices_->size_, handle_.get_stream()};
        raft::copy(seed_vertices->data(),
                   seed_vertices_->as_type<vertex_t>(),
                   seed_vertices->size(),
                   handle_.get_stream());

        if constexpr (multi_gpu) {
          std::tie(seed_vertices, std::ignore) =
            cugraph::shuffle_ext_vertices(handle_,
                                          std::move(*seed_vertices),
                                          std::vector<cugraph::arithmetic_device_uvector_t>{});
        }

        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          seed_vertices->data(),
          seed_vertices->size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);

        if (seed_vertices->size() > 0) {
          cugraph::sort(handle_.get_thrust_policy(), seed_vertices->begin(), seed_vertices->end());
        }
      }

      auto [cycle_vertices, cycle_offsets] = cugraph::simple_cycles<vertex_t, edge_t, multi_gpu>(
        handle_,
        graph_view,
        seed_vertices_ != nullptr ? std::make_optional(raft::device_span<vertex_t const>{
                                      seed_vertices->data(), seed_vertices->size()})
                                  : std::nullopt,
        length_bound_v,
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        cycle_vertices.data(),
        cycle_vertices.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      result_ = new cugraph::c_api::cugraph_simple_cycles_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(cycle_vertices,
                                                               graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(cycle_offsets, SIZE_T)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_simple_cycles(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* seed_vertices,
  size_t length_bound,
  bool_t do_expensive_check,
  cugraph_simple_cycles_result_t** result,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(
    length_bound > 0, CUGRAPH_INVALID_INPUT, "length_bound must be greater than 0", *error);

  simple_cycles_functor functor(handle, graph, seed_vertices, length_bound, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
