/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "c_api/capi_helper.hpp"
#include "c_api/graph.hpp"
#include "c_api/induced_subgraph_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <numeric>
#include <optional>

namespace {

struct extract_ego_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* source_vertices_;
  size_t radius_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_induced_subgraph_result_t* result_{};

  extract_ego_functor(::cugraph_resource_handle_t const* handle,
                      ::cugraph_graph_t* graph,
                      ::cugraph_type_erased_device_array_view_t const* source_vertices,
                      size_t radius,
                      bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      source_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          source_vertices)),
      radius_(radius),
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
      // extract ego expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> source_vertices(source_vertices_->size_, handle_.get_stream());
      raft::copy(source_vertices.data(),
                 source_vertices_->as_type<vertex_t>(),
                 source_vertices.size(),
                 handle_.get_stream());

      std::optional<rmm::device_uvector<size_t>> source_indices{std::nullopt};

      if constexpr (multi_gpu) {
        auto displacements = cugraph::host_scalar_allgather(
          handle_.get_comms(), source_vertices.size(), handle_.get_stream());
        std::exclusive_scan(
          displacements.begin(), displacements.end(), displacements.begin(), size_t{0});
        source_indices = rmm::device_uvector<size_t>(source_vertices.size(), handle_.get_stream());
        cugraph::detail::sequence_fill(handle_.get_stream(),
                                       (*source_indices).data(),
                                       (*source_indices).size(),
                                       displacements[handle_.get_comms().get_rank()]);

        std::tie(source_vertices, source_indices) = cugraph::shuffle_ext_vertex_value_pairs(
          handle_, std::move(source_vertices), std::move(*source_indices));
      }

      cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        source_vertices.data(),
        source_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        do_expensive_check_);

      auto [src, dst, wgt, edge_offsets] =
        cugraph::extract_ego<vertex_t, edge_t, weight_t, multi_gpu>(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          raft::device_span<vertex_t const>{source_vertices.data(), source_vertices.size()},
          radius_,
          do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        src.data(),
        src.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        dst.data(),
        dst.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      if constexpr (multi_gpu) {
        auto recvcounts = cugraph::host_scalar_allgather(
          handle_.get_comms(), (*source_indices).size(), handle_.get_stream());
        std::vector<size_t> displacements(recvcounts.size());
        std::exclusive_scan(recvcounts.begin(), recvcounts.end(), displacements.begin(), size_t{0});
        rmm::device_uvector<size_t> allgathered_indices(displacements.back() + recvcounts.back(),
                                                        handle_.get_stream());
        cugraph::device_allgatherv(
          handle_.get_comms(),
          (*source_indices).begin(),
          allgathered_indices.begin(),
          raft::host_span<size_t const>(recvcounts.data(), recvcounts.size()),
          raft::host_span<size_t const>(displacements.data(), displacements.size()),
          handle_.get_stream());
        source_indices = std::move(allgathered_indices);

        std::tie(edge_offsets, src, dst, wgt) =
          cugraph::c_api::detail::reorder_extracted_egonets<vertex_t, weight_t>(
            handle_,
            std::move(*source_indices),
            std::move(edge_offsets),
            std::move(src),
            std::move(dst),
            std::move(wgt));
      }

      result_ = new cugraph::c_api::cugraph_induced_subgraph_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(dst, graph_->vertex_type_),
        wgt ? new cugraph::c_api::cugraph_type_erased_device_array_t(*wgt, graph_->weight_type_)
            : NULL,
        NULL,
        NULL,
        new cugraph::c_api::cugraph_type_erased_device_array_t(edge_offsets,
                                                               cugraph_data_type_id_t::SIZE_T)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_extract_ego(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* source_vertices,
  size_t radius,
  bool_t do_expensive_check,
  cugraph_induced_subgraph_result_t** result,
  cugraph_error_t** error)
{
  extract_ego_functor functor(handle, graph, source_vertices, radius, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
