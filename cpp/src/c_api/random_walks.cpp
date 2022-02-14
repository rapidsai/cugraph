/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph_c/algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/visitors/generic_cascaded_dispatch.hpp>

#include <raft/handle.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_random_walk_result_t {
  bool result_compressed_;
  size_t max_path_length_;
  cugraph_type_erased_device_array_t* paths_;
  cugraph_type_erased_device_array_t* weights_;
  cugraph_type_erased_device_array_t* offsets_;
};

struct node2vec_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t const* sources_;
  size_t max_depth_;
  bool compress_result_;
  double p_;
  double q_;
  cugraph_random_walk_result_t* result_{};

  node2vec_functor(raft::handle_t const& handle,
                   cugraph_graph_t* graph,
                   cugraph_type_erased_device_array_view_t const* sources,
                   size_t max_depth,
                   bool compress_result,
                   double p,
                   double q)
    : abstract_functor(),
      handle_(handle),
      graph_(graph),
      sources_(sources),
      max_depth_(max_depth),
      compress_result_(compress_result),
      p_(p),
      q_(q)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else if constexpr (multi_gpu) {
      unsupported();
    } else {
      // node2vec expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> sources(sources_->size_, handle_.get_stream());
      raft::copy(
        sources.data(), sources_->as_type<vertex_t>(), sources.size(), handle_.get_stream());

      //
      // Need to renumber sources
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 sources.data(),
                                                 sources.size(),
                                                 number_map->data(),
                                                 graph_view.get_local_vertex_first(),
                                                 graph_view.get_local_vertex_last(),
                                                 false);

      // FIXME:  Forcing this to edge_t for now.  What should it really be?
      // Seems like it should be the smallest size that can accommodate
      // max_depth_ * sources_->size_
      auto [paths, weights, offsets] = cugraph::random_walks(
        handle_,
        graph_view,
        sources.data(),
        static_cast<edge_t>(sources.size()),
        static_cast<edge_t>(max_depth_),
        !compress_result_,
        // std::make_unique<sampling_params_t>(2, p_, q_, false));
        std::make_unique<sampling_params_t>(cugraph::sampling_strategy_t::NODE2VEC, p_, q_));

      result_ = new cugraph_random_walk_result_t{
        compress_result_,
        max_depth_,
        new cugraph_type_erased_device_array_t(paths, graph_->vertex_type_),
        new cugraph_type_erased_device_array_t(weights, graph_->weight_type_),
        new cugraph_type_erased_device_array_t(offsets, graph_->vertex_type_)};
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

cugraph_error_code_t cugraph_node2vec(const cugraph_resource_handle_t* handle,
                                      cugraph_graph_t* graph,
                                      const cugraph_type_erased_device_array_view_t* sources,
                                      size_t max_depth,
                                      bool_t compress_results,
                                      double p,
                                      double q,
                                      cugraph_random_walk_result_t** result,
                                      cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;

  try {
    auto p_handle = reinterpret_cast<raft::handle_t const*>(handle);
    auto p_graph  = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph);
    auto p_sources =
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources);

    cugraph::c_api::node2vec_functor functor(
      *p_handle, p_graph, p_sources, max_depth, compress_results, p, q);

    // FIXME:  This seems like a recurring pattern.  Can I encapsulate
    //    The vertex_dispatcher and error handling calls into a reusable function?
    //    After all, we're in C++ here.
    cugraph::dispatch::vertex_dispatcher(cugraph::c_api::dtypes_mapping[p_graph->vertex_type_],
                                         cugraph::c_api::dtypes_mapping[p_graph->edge_type_],
                                         cugraph::c_api::dtypes_mapping[p_graph->weight_type_],
                                         p_graph->store_transposed_,
                                         p_graph->multi_gpu_,
                                         functor);

    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }

    *result = reinterpret_cast<cugraph_random_walk_result_t*>(functor.result_);
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

size_t cugraph_random_walk_result_get_max_path_length(cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return internal_pointer->max_path_length_;
}

cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_paths(
  cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->paths_->view());
}

cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_weights(
  cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->weights_->view());
}

cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_offsets(
  cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->offsets_->view());
}

void cugraph_random_walk_result_free(cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  delete internal_pointer->paths_;
  delete internal_pointer->weights_;
  delete internal_pointer;
}
