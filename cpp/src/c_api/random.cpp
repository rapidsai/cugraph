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
#include "c_api/random.hpp"

#include "c_api/abstract_functor.hpp"
#include "c_api/error.hpp"
#include "c_api/graph.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

namespace {

struct select_random_vertices_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t const* graph_{};
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  size_t num_vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_t* result_{};

  select_random_vertices_functor(cugraph_resource_handle_t const* handle,
                                 cugraph_graph_t const* graph,
                                 cugraph_rng_state_t* rng_state,
                                 size_t num_vertices)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t const*>(graph)),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      num_vertices_(num_vertices)
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
      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();
      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> local_vertices(0, handle_.get_stream());

      local_vertices = cugraph::select_random_vertices(
        handle_,
        graph_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        rng_state_->rng_state_,
        num_vertices_,
        false,
        false);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        local_vertices.data(),
        local_vertices.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        false);

      result_ = new cugraph::c_api::cugraph_type_erased_device_array_t(local_vertices,
                                                                       graph_->vertex_type_);
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_rng_state_create(const cugraph_resource_handle_t* handle,
                                                         uint64_t seed,
                                                         cugraph_rng_state_t** state,
                                                         cugraph_error_t** error)
{
  *state = nullptr;
  *error = nullptr;

  try {
    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);

    if (p_handle->handle_->comms_initialized()) {
      // need to verify that every seed is different
      auto seed_v = cugraph::host_scalar_allgather(
        p_handle->handle_->get_comms(), seed, p_handle->handle_->get_stream());
      std::sort(seed_v.begin(), seed_v.end());
      if (std::unique(seed_v.begin(), seed_v.end()) != seed_v.end()) {
        *error = reinterpret_cast<cugraph_error_t*>(
          new cugraph::c_api::cugraph_error_t{"seed must be different on each GPU"});
        return CUGRAPH_INVALID_INPUT;
      }
    }

    *state = reinterpret_cast<cugraph_rng_state_t*>(
      new cugraph::c_api::cugraph_rng_state_t{raft::random::RngState{seed}});
    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" void cugraph_rng_state_free(cugraph_rng_state_t* p)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(p);
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_select_random_vertices(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_t* graph,
  cugraph_rng_state_t* rng_state,
  size_t num_vertices,
  cugraph_type_erased_device_array_t** vertices,
  cugraph_error_t** error)
{
  select_random_vertices_functor functor(handle, graph, rng_state, num_vertices);

  return cugraph::c_api::run_algorithm(graph, functor, vertices, error);
}
