/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/simple_cycles_result.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/components_algorithms.h>

#include <cugraph/utilities/error.hpp>

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
      // Planned implementation (after cugraph::simple_cycles lands in libcugraph):
      // 1. If store_transposed, transpose graph storage (same as SCC/BFS).
      // 2. Copy seed_vertices (if non-NULL) to a mutable device buffer.
      // 3. Multi-GPU: shuffle_ext_vertices so each rank holds its partition's seeds.
      // 4. renumber_ext_vertices using graph number_map (external -> internal ids).
      // 5. thrust::sort (and optionally unique) internal seed ids for the C++ API.
      // 6. Map NULL seed_vertices -> std::nullopt; non-NULL -> raft::device_span.
      // 7. Call cugraph::simple_cycles(handle, graph_view, seeds, length_bound, check).
      // 8. unrenumber_int_vertices on the flat cycle_vertices output buffer.
      // 9. Wrap cycle_vertices and cycle_offsets in cugraph_type_erased_device_array_t.
      (void)handle_;
      (void)graph_;
      (void)seed_vertices_;
      (void)length_bound_;
      (void)do_expensive_check_;
      CUGRAPH_FAIL("cugraph_simple_cycles is not implemented");
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
  simple_cycles_functor functor(handle, graph, seed_vertices, length_bound, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
