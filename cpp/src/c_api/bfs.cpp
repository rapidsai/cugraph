/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/paths_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>

#include <limits>
#include <memory>

namespace cugraph {
namespace c_api {

struct bfs_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t* sources_;
  bool direction_optimizing_;
  size_t depth_limit_;
  bool compute_predecessors_;
  bool do_expensive_check_;
  cugraph_paths_result_t* result_{};

  bfs_functor(::cugraph_resource_handle_t const* handle,
              ::cugraph_graph_t* graph,
              ::cugraph_type_erased_device_array_view_t* sources,
              bool direction_optimizing,
              size_t depth_limit,
              bool compute_predecessors,
              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(sources)),
      direction_optimizing_(direction_optimizing),
      depth_limit_(depth_limit),
      compute_predecessors_(compute_predecessors),
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
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // BFS expects store_transposed == false
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

      rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                              handle_.get_stream());
      rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

      if (compute_predecessors_) {
        predecessors.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
      }

      rmm::device_uvector<vertex_t> sources(sources_->size_, handle_.get_stream());
      raft::copy(
        sources.data(), sources_->as_type<vertex_t>(), sources_->size_, handle_.get_stream());

      if constexpr (multi_gpu) {
        std::tie(sources, std::ignore) = shuffle_ext_vertices(
          handle_, std::move(sources), std::vector<cugraph::arithmetic_device_uvector_t>{});
      }

      //
      // Need to renumber sources
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 sources.data(),
                                                 sources.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 do_expensive_check_);

      size_t invalid_count = cugraph::detail::count_values(
        handle_,
        raft::device_span<vertex_t const>{sources.data(), sources.size()},
        cugraph::invalid_vertex_id<vertex_t>::value);

      if constexpr (multi_gpu) {
        invalid_count = cugraph::host_scalar_allreduce(
          handle_.get_comms(), invalid_count, raft::comms::op_t::SUM, handle_.get_stream());
      }

      if (invalid_count != 0) {
        mark_error(CUGRAPH_INVALID_INPUT, "Found invalid vertex in the input sources");
        return;
      }

      cugraph::bfs<vertex_t, edge_t, multi_gpu>(
        handle_,
        graph_view,
        distances.data(),
        compute_predecessors_ ? predecessors.data() : nullptr,
        sources.data(),
        sources.size(),
        direction_optimizing_,
        static_cast<vertex_t>(depth_limit_),
        do_expensive_check_);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      if (compute_predecessors_) {
        unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                     predecessors.data(),
                                                     predecessors.size(),
                                                     number_map->data(),
                                                     graph_view.vertex_partition_range_lasts(),
                                                     do_expensive_check_);
      }

      result_ = new cugraph_paths_result_t{
        new cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph_type_erased_device_array_t(distances, graph_->vertex_type_),
        new cugraph_type_erased_device_array_t(predecessors, graph_->vertex_type_)};
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_vertices(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_distances(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->distances_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_predecessors(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->predecessors_->view());
}

extern "C" void cugraph_paths_result_free(cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->distances_;
  delete internal_pointer->predecessors_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_bfs(const cugraph_resource_handle_t* handle,
                                            cugraph_graph_t* graph,
                                            cugraph_type_erased_device_array_view_t* sources,
                                            bool_t direction_optimizing,
                                            size_t depth_limit,
                                            bool_t compute_predecessors,
                                            bool_t do_expensive_check,
                                            cugraph_paths_result_t** result,
                                            cugraph_error_t** error)
{
  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources)
        ->type_,
    CUGRAPH_INVALID_INPUT,
    "vertex type of graph and sources must match",
    *error);

  cugraph::c_api::bfs_functor functor(handle,
                                      graph,
                                      sources,
                                      direction_optimizing,
                                      depth_limit,
                                      compute_predecessors,
                                      do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

namespace cugraph {
namespace c_api {

struct dawn_bfs_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t* sources_;
  size_t depth_limit_;
  bool do_expensive_check_;
  cugraph_paths_result_t* result_{};
  std::unique_ptr<cugraph_error_t> error_ = {std::make_unique<cugraph_error_t>("")};
  cugraph_error_code_t error_code_{CUGRAPH_SUCCESS};

  dawn_bfs_functor(::cugraph_resource_handle_t const* handle,
                   ::cugraph_graph_t* graph,
                   ::cugraph_type_erased_device_array_view_t* sources,
                   size_t depth_limit,
                   bool do_expensive_check)
    : handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(sources)),
      depth_limit_(depth_limit),
      do_expensive_check_(do_expensive_check)
  {
  }

  void mark_error(cugraph_error_code_t error_code, std::string const& error_message)
  {
    error_code_ = error_code;
    error_->error_message_ = error_message;
  }

  void operator()()
  {
    if (graph_->multi_gpu_) {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently supports only single-GPU graphs");
      return;
    }
    if (graph_->store_transposed_) {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently requires store_transposed=false");
      return;
    }
    if (graph_->renumber_) {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently requires renumber=false");
      return;
    }
    if (graph_->vertex_type_ == INT32 && graph_->edge_type_ == INT32) {
      run<int32_t, int32_t>();
    } else if (graph_->vertex_type_ == INT64 && graph_->edge_type_ == INT64) {
      run<int64_t, int64_t>();
    } else {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently supports only int32/int32 and int64/int64 graphs");
      return;
    }
  }

  template <typename vertex_t, typename edge_t>
  void run()
  {
    constexpr bool multi_gpu = false;
    auto graph =
      reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);
    auto graph_view = graph->view();

    if (sources_->size_ == 0) {
      mark_error(CUGRAPH_INVALID_INPUT, "DAWN BFS requires at least one source");
      return;
    }
    if (depth_limit_ != std::numeric_limits<size_t>::max() &&
        depth_limit_ > static_cast<size_t>(std::numeric_limits<vertex_t>::max())) {
      mark_error(CUGRAPH_INVALID_INPUT, "DAWN BFS depth_limit does not fit the vertex type");
      return;
    }

    rmm::device_uvector<vertex_t> sources(sources_->size_, handle_.get_stream());
    raft::copy(sources.data(), sources_->as_type<vertex_t>(), sources_->size_, handle_.get_stream());

    // DAWN's performance path assumes an unrenumbered graph with contiguous vertex IDs.
    // Source IDs are therefore already internal IDs and should not be remapped here.

    rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                            handle_.get_stream());
    rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

    auto depth_limit = depth_limit_ == std::numeric_limits<size_t>::max()
                         ? std::numeric_limits<vertex_t>::max()
                         : static_cast<vertex_t>(depth_limit_);

    cugraph::dawn_bfs<vertex_t, edge_t, multi_gpu>(handle_,
                                                   graph_view,
                                                   distances.data(),
                                                   sources.data(),
                                                   sources.size(),
                                                   depth_limit,
                                                   do_expensive_check_);

    // DAWN returns distances in contiguous unrenumbered vertex-id order.  Avoid materializing a
    // graph-sized vertex_ids array in this compatibility wrapper; performance benchmarks should
    // consume distances directly.
    rmm::device_uvector<vertex_t> vertex_ids(0, handle_.get_stream());

    result_ = new cugraph_paths_result_t{
      new cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
      new cugraph_type_erased_device_array_t(distances, graph_->vertex_type_),
      new cugraph_type_erased_device_array_t(predecessors, graph_->vertex_type_)};
  }
};

struct dawn_bfs_distances_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t* sources_;
  cugraph_type_erased_device_array_view_t* distances_;
  size_t depth_limit_;
  bool do_expensive_check_;
  std::unique_ptr<cugraph_error_t> error_ = {std::make_unique<cugraph_error_t>("")};
  cugraph_error_code_t error_code_{CUGRAPH_SUCCESS};

  dawn_bfs_distances_functor(::cugraph_resource_handle_t const* handle,
                             ::cugraph_graph_t* graph,
                             ::cugraph_type_erased_device_array_view_t* sources,
                             size_t depth_limit,
                             bool do_expensive_check,
                             ::cugraph_type_erased_device_array_view_t* distances)
    : handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(sources)),
      distances_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(distances)),
      depth_limit_(depth_limit),
      do_expensive_check_(do_expensive_check)
  {
  }

  void mark_error(cugraph_error_code_t error_code, std::string const& error_message)
  {
    error_code_ = error_code;
    error_->error_message_ = error_message;
  }

  void operator()()
  {
    if (graph_->multi_gpu_) {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently supports only single-GPU graphs");
      return;
    }
    if (graph_->store_transposed_) {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently requires store_transposed=false");
      return;
    }
    if (graph_->renumber_) {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently requires renumber=false");
      return;
    }
    if (graph_->vertex_type_ == INT32 && graph_->edge_type_ == INT32) {
      run<int32_t, int32_t>();
    } else if (graph_->vertex_type_ == INT64 && graph_->edge_type_ == INT64) {
      run<int64_t, int64_t>();
    } else {
      mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "DAWN BFS currently supports only int32/int32 and int64/int64 graphs");
    }
  }

  template <typename vertex_t, typename edge_t>
  void run()
  {
    constexpr bool multi_gpu = false;
    auto graph =
      reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);
    auto graph_view = graph->view();

    if (sources_->size_ == 0) {
      mark_error(CUGRAPH_INVALID_INPUT, "DAWN BFS requires at least one source");
      return;
    }
    if (distances_->size_ != graph_view.local_vertex_partition_range_size()) {
      mark_error(CUGRAPH_INVALID_INPUT,
                 "DAWN BFS distances output must be graph local vertex partition sized");
      return;
    }
    if (depth_limit_ != std::numeric_limits<size_t>::max() &&
        depth_limit_ > static_cast<size_t>(std::numeric_limits<vertex_t>::max())) {
      mark_error(CUGRAPH_INVALID_INPUT, "DAWN BFS depth_limit does not fit the vertex type");
      return;
    }

    rmm::device_uvector<vertex_t> sources(sources_->size_, handle_.get_stream());
    raft::copy(sources.data(), sources_->as_type<vertex_t>(), sources_->size_, handle_.get_stream());

    auto depth_limit = depth_limit_ == std::numeric_limits<size_t>::max()
                         ? std::numeric_limits<vertex_t>::max()
                         : static_cast<vertex_t>(depth_limit_);

    cugraph::dawn_bfs<vertex_t, edge_t, multi_gpu>(handle_,
                                                   graph_view,
                                                   distances_->as_type<vertex_t>(),
                                                   sources.data(),
                                                   sources.size(),
                                                   depth_limit,
                                                   do_expensive_check_);
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" cugraph_error_code_t cugraph_dawn_bfs(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  cugraph_type_erased_device_array_view_t* sources,
  size_t depth_limit,
  bool_t do_expensive_check,
  cugraph_paths_result_t** result,
  cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;

  if (reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ !=
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources)
        ->type_) {
    *error = reinterpret_cast<::cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"vertex type of graph and sources must match"});
    return CUGRAPH_INVALID_INPUT;
  }

  try {
    cugraph::c_api::dawn_bfs_functor functor(
      handle, graph, sources, depth_limit, do_expensive_check);
    functor();
    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<::cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }
    *result = reinterpret_cast<cugraph_paths_result_t*>(functor.result_);
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_dawn_bfs_distances(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  cugraph_type_erased_device_array_view_t* sources,
  size_t depth_limit,
  bool_t do_expensive_check,
  cugraph_type_erased_device_array_view_t* distances,
  cugraph_error_t** error)
{
  *error = nullptr;

  auto internal_graph = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph);
  auto internal_sources =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources);
  auto internal_distances =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(distances);

  if (internal_graph->vertex_type_ != internal_sources->type_ ||
      internal_graph->vertex_type_ != internal_distances->type_) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{
      "vertex type of graph, sources, and distances must match"});
    return CUGRAPH_INVALID_INPUT;
  }

  try {
    cugraph::c_api::dawn_bfs_distances_functor functor(
      handle, graph, sources, depth_limit, do_expensive_check, distances);
    functor();
    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<::cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}
