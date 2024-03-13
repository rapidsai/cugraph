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
#include "c_api/degrees_result.hpp"
#include "c_api/graph.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <thrust/gather.h>

#include <optional>

namespace {

struct degrees_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* source_vertices_;
  bool in_degrees_{false};
  bool out_degrees_{false};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_degrees_result_t* result_{};

  degrees_functor(cugraph_resource_handle_t const* handle,
                  cugraph_graph_t* graph,
                  ::cugraph_type_erased_device_array_view_t const* source_vertices,
                  bool in_degrees,
                  bool out_degrees,
                  bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      source_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          source_vertices)),
      in_degrees_{in_degrees},
      out_degrees_{out_degrees},
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
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      std::optional<rmm::device_uvector<edge_t>> in_degrees{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> out_degrees{std::nullopt};

      if (in_degrees_ && out_degrees_ && graph_view.is_symmetric()) {
        in_degrees = store_transposed ? graph_view.compute_in_degrees(handle_)
                                      : graph_view.compute_out_degrees(handle_);
        // out_degrees will be extracted from in_degrees in the result
      } else {
        if (in_degrees_) in_degrees = graph_view.compute_in_degrees(handle_);

        if (out_degrees_) out_degrees = graph_view.compute_out_degrees(handle_);
      }

      rmm::device_uvector<vertex_t> vertex_ids(0, handle_.get_stream());

      if (source_vertices_) {
        // FIXME: Would be more efficient if graph_view.compute_*_degrees could take a vertex
        //  subset
        vertex_ids.resize(source_vertices_->size_, handle_.get_stream());
        raft::copy(vertex_ids.data(),
                   source_vertices_->as_type<vertex_t>(),
                   vertex_ids.size(),
                   handle_.get_stream());

        if constexpr (multi_gpu) {
          vertex_ids = cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
            handle_, std::move(vertex_ids));
        }

        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          vertex_ids.data(),
          vertex_ids.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);

        auto vertex_partition = cugraph::vertex_partition_device_view_t<vertex_t, multi_gpu>(
          graph_view.local_vertex_partition_view());

        auto vertices_iter = thrust::make_transform_iterator(
          vertex_ids.begin(),
          cuda::proclaim_return_type<vertex_t>([vertex_partition] __device__(auto v) {
            return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
          }));

        if (in_degrees && out_degrees) {
          rmm::device_uvector<edge_t> tmp_in_degrees(vertex_ids.size(), handle_.get_stream());
          rmm::device_uvector<edge_t> tmp_out_degrees(vertex_ids.size(), handle_.get_stream());
          thrust::gather(
            handle_.get_thrust_policy(),
            vertices_iter,
            vertices_iter + vertex_ids.size(),
            thrust::make_zip_iterator(in_degrees->begin(), out_degrees->begin()),
            thrust::make_zip_iterator(tmp_in_degrees.begin(), tmp_out_degrees.begin()));
          in_degrees  = std::move(tmp_in_degrees);
          out_degrees = std::move(tmp_out_degrees);
        } else if (in_degrees) {
          rmm::device_uvector<edge_t> tmp_in_degrees(vertex_ids.size(), handle_.get_stream());
          thrust::gather(handle_.get_thrust_policy(),
                         vertices_iter,
                         vertices_iter + vertex_ids.size(),
                         in_degrees->begin(),
                         tmp_in_degrees.begin());
          in_degrees = std::move(tmp_in_degrees);
        } else {
          rmm::device_uvector<edge_t> tmp_out_degrees(vertex_ids.size(), handle_.get_stream());
          thrust::gather(handle_.get_thrust_policy(),
                         vertices_iter,
                         vertices_iter + vertex_ids.size(),
                         out_degrees->begin(),
                         tmp_out_degrees.begin());
          out_degrees = std::move(tmp_out_degrees);
        }

        cugraph::unrenumber_local_int_vertices<vertex_t>(
          handle_,
          vertex_ids.data(),
          vertex_ids.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);
      } else {
        vertex_ids.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
        raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());
      }

      result_ = new cugraph::c_api::cugraph_degrees_result_t{
        graph_view.is_symmetric(),
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        in_degrees
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*in_degrees, graph_->edge_type_)
          : nullptr,
        out_degrees
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*out_degrees, graph_->edge_type_)
          : nullptr};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_in_degrees(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* source_vertices,
  bool_t do_expensive_check,
  cugraph_degrees_result_t** result,
  cugraph_error_t** error)
{
  degrees_functor functor(handle, graph, source_vertices, true, false, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_out_degrees(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* source_vertices,
  bool_t do_expensive_check,
  cugraph_degrees_result_t** result,
  cugraph_error_t** error)
{
  degrees_functor functor(handle, graph, source_vertices, false, true, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_degrees(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* source_vertices,
  bool_t do_expensive_check,
  cugraph_degrees_result_t** result,
  cugraph_error_t** error)
{
  degrees_functor functor(handle, graph, source_vertices, true, true, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
