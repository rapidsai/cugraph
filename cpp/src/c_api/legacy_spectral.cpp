/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <c_api/capi_helper.hpp>
#include <c_api/graph.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_clustering_result_t {
  cugraph_type_erased_device_array_t* vertices_{nullptr};
  cugraph_type_erased_device_array_t* clusters_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct balanced_cut_clustering_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  size_t n_clusters_;
  size_t n_eigenvectors_;
  double evs_tolerance_;
  int evs_max_iterations_;
  double k_means_tolerance_;
  int k_means_max_iterations_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_clustering_result_t* result_{};

  balanced_cut_clustering_functor(::cugraph_resource_handle_t const* handle,
                                  ::cugraph_graph_t* graph,
                                  size_t n_clusters,
                                  size_t n_eigenvectors,
                                  double evs_tolerance,
                                  int evs_max_iterations,
                                  double k_means_tolerance,
                                  int k_means_max_iterations,
                                  bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      n_clusters_(n_clusters),
      n_eigenvectors_(n_eigenvectors),
      evs_tolerance_(evs_tolerance),
      evs_max_iterations_(evs_max_iterations),
      k_means_tolerance_(k_means_tolerance),
      k_means_max_iterations_(k_means_max_iterations),
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
    } else if constexpr (multi_gpu) {
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      // balanced_cut expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
      if (edge_weights == nullptr) {
        tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
        cugraph::detail::scalar_fill(handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
      }

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        (edge_weights == nullptr)
          ? tmp_weights.data()
          : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

      rmm::device_uvector<vertex_t> clusters(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());

      cugraph::ext_raft::balancedCutClustering(legacy_graph_view,
                                               static_cast<vertex_t>(n_clusters_),
                                               static_cast<vertex_t>(n_eigenvectors_),
                                               static_cast<weight_t>(evs_tolerance_),
                                               evs_max_iterations_,
                                               static_cast<weight_t>(k_means_tolerance_),
                                               k_means_max_iterations_,
                                               clusters.data());

      rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());
      raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_clustering_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertices, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(clusters, graph_->vertex_type_)};
    }
  }
};

struct spectral_clustering_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  size_t n_clusters_;
  size_t n_eigenvectors_;
  double evs_tolerance_;
  int evs_max_iterations_;
  double k_means_tolerance_;
  int k_means_max_iterations_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_clustering_result_t* result_{};

  spectral_clustering_functor(::cugraph_resource_handle_t const* handle,
                              ::cugraph_graph_t* graph,
                              size_t n_clusters,
                              size_t n_eigenvectors,
                              double evs_tolerance,
                              int evs_max_iterations,
                              double k_means_tolerance,
                              int k_means_max_iterations,
                              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      n_clusters_(n_clusters),
      n_eigenvectors_(n_eigenvectors),
      evs_tolerance_(evs_tolerance),
      evs_max_iterations_(evs_max_iterations),
      k_means_tolerance_(k_means_tolerance),
      k_means_max_iterations_(k_means_max_iterations),
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
    } else if constexpr (multi_gpu) {
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      // spectral clustering expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
      if (edge_weights == nullptr) {
        tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
        cugraph::detail::scalar_fill(handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
      }

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        (edge_weights == nullptr)
          ? tmp_weights.data()
          : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

      rmm::device_uvector<vertex_t> clusters(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());

      cugraph::ext_raft::spectralModularityMaximization(legacy_graph_view,
                                                        static_cast<vertex_t>(n_clusters_),
                                                        static_cast<vertex_t>(n_eigenvectors_),
                                                        static_cast<weight_t>(evs_tolerance_),
                                                        evs_max_iterations_,
                                                        static_cast<weight_t>(k_means_tolerance_),
                                                        k_means_max_iterations_,
                                                        clusters.data());

      rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());
      raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_clustering_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertices, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(clusters, graph_->vertex_type_)};
    }
  }
};

struct analyze_clustering_ratio_cut_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  size_t n_clusters_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* clusters_{};
  double result_{};

  analyze_clustering_ratio_cut_functor(::cugraph_resource_handle_t const* handle,
                                       ::cugraph_graph_t* graph,
                                       size_t n_clusters,
                                       ::cugraph_type_erased_device_array_view_t const* vertices,
                                       ::cugraph_type_erased_device_array_view_t const* clusters)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      n_clusters_(n_clusters),
      vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(vertices)),
      clusters_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(clusters))
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
    } else if constexpr (multi_gpu) {
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      // analyze clustering expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
      if (edge_weights == nullptr) {
        tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
        cugraph::detail::scalar_fill(handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
      }

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        (edge_weights == nullptr)
          ? tmp_weights.data()
          : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

      weight_t score;

      if (cugraph::detail::is_equal(
            handle_,
            raft::device_span<vertex_t const>{vertices_->as_type<vertex_t const>(),
                                              vertices_->size_},
            raft::device_span<vertex_t const>{number_map->data(), number_map->size()})) {
        cugraph::ext_raft::analyzeClustering_ratio_cut(
          legacy_graph_view, n_clusters_, clusters_->as_type<vertex_t>(), &score);
      } else {
        rmm::device_uvector<vertex_t> tmp_v(vertices_->size_, handle_.get_stream());
        rmm::device_uvector<vertex_t> tmp_c(clusters_->size_, handle_.get_stream());

        raft::copy(
          tmp_v.data(), vertices_->as_type<vertex_t>(), vertices_->size_, handle_.get_stream());
        raft::copy(
          tmp_c.data(), clusters_->as_type<vertex_t>(), clusters_->size_, handle_.get_stream());

        cugraph::renumber_ext_vertices<vertex_t, false>(
          handle_,
          tmp_v.data(),
          tmp_v.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          false);

        cugraph::c_api::detail::sort_by_key(
          handle_,
          raft::device_span<vertex_t>{tmp_v.data(), tmp_v.size()},
          raft::device_span<vertex_t>{tmp_c.data(), tmp_c.size()});

        cugraph::ext_raft::analyzeClustering_ratio_cut(
          legacy_graph_view, n_clusters_, tmp_c.data(), &score);
      }

      result_ = static_cast<double>(score);
    }
  }
};

struct analyze_clustering_edge_cut_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  size_t n_clusters_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* clusters_{};
  double result_{};

  analyze_clustering_edge_cut_functor(::cugraph_resource_handle_t const* handle,
                                      ::cugraph_graph_t* graph,
                                      size_t n_clusters,
                                      ::cugraph_type_erased_device_array_view_t const* vertices,
                                      ::cugraph_type_erased_device_array_view_t const* clusters)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      n_clusters_(n_clusters),
      vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(vertices)),
      clusters_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(clusters))
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
    } else if constexpr (multi_gpu) {
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      // analyze clustering expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
      if (edge_weights == nullptr) {
        tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
        cugraph::detail::scalar_fill(handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
      }

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        (edge_weights == nullptr)
          ? tmp_weights.data()
          : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

      weight_t score;

      if (cugraph::detail::is_equal(
            handle_,
            raft::device_span<vertex_t const>{vertices_->as_type<vertex_t const>(),
                                              vertices_->size_},
            raft::device_span<vertex_t const>{number_map->data(), number_map->size()})) {
        cugraph::ext_raft::analyzeClustering_edge_cut(
          legacy_graph_view, n_clusters_, clusters_->as_type<vertex_t>(), &score);
      } else {
        rmm::device_uvector<vertex_t> tmp_v(vertices_->size_, handle_.get_stream());
        rmm::device_uvector<vertex_t> tmp_c(clusters_->size_, handle_.get_stream());

        raft::copy(
          tmp_v.data(), vertices_->as_type<vertex_t>(), vertices_->size_, handle_.get_stream());
        raft::copy(
          tmp_c.data(), clusters_->as_type<vertex_t>(), clusters_->size_, handle_.get_stream());

        cugraph::renumber_ext_vertices<vertex_t, false>(
          handle_,
          tmp_v.data(),
          tmp_v.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          false);

        cugraph::c_api::detail::sort_by_key(
          handle_,
          raft::device_span<vertex_t>{tmp_v.data(), tmp_v.size()},
          raft::device_span<vertex_t>{tmp_c.data(), tmp_c.size()});

        cugraph::ext_raft::analyzeClustering_edge_cut(
          legacy_graph_view, n_clusters_, tmp_c.data(), &score);
      }

      result_ = static_cast<double>(score);
    }
  }
};

struct analyze_clustering_modularity_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  size_t n_clusters_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* clusters_{};
  double result_{};

  analyze_clustering_modularity_functor(::cugraph_resource_handle_t const* handle,
                                        ::cugraph_graph_t* graph,
                                        size_t n_clusters,
                                        ::cugraph_type_erased_device_array_view_t const* vertices,
                                        ::cugraph_type_erased_device_array_view_t const* clusters)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      n_clusters_(n_clusters),
      vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(vertices)),
      clusters_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(clusters))
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
    } else if constexpr (multi_gpu) {
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      // analyze clustering expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
      if (edge_weights == nullptr) {
        tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
        cugraph::detail::scalar_fill(handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
      }

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        (edge_weights == nullptr)
          ? tmp_weights.data()
          : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

      weight_t score;

      if (cugraph::detail::is_equal(
            handle_,
            raft::device_span<vertex_t const>{vertices_->as_type<vertex_t const>(),
                                              vertices_->size_},
            raft::device_span<vertex_t const>{number_map->data(), number_map->size()})) {
        cugraph::ext_raft::analyzeClustering_modularity(
          legacy_graph_view, n_clusters_, clusters_->as_type<vertex_t>(), &score);
      } else {
        rmm::device_uvector<vertex_t> tmp_v(vertices_->size_, handle_.get_stream());
        rmm::device_uvector<vertex_t> tmp_c(clusters_->size_, handle_.get_stream());

        raft::copy(
          tmp_v.data(), vertices_->as_type<vertex_t>(), vertices_->size_, handle_.get_stream());
        raft::copy(
          tmp_c.data(), clusters_->as_type<vertex_t>(), clusters_->size_, handle_.get_stream());

        cugraph::renumber_ext_vertices<vertex_t, false>(
          handle_,
          tmp_v.data(),
          tmp_v.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          false);

        cugraph::c_api::detail::sort_by_key(
          handle_,
          raft::device_span<vertex_t>{tmp_v.data(), tmp_v.size()},
          raft::device_span<vertex_t>{tmp_c.data(), tmp_c.size()});

        cugraph::ext_raft::analyzeClustering_modularity(
          legacy_graph_view, n_clusters_, tmp_c.data(), &score);
      }

      result_ = static_cast<double>(score);
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_clustering_result_get_vertices(
  cugraph_clustering_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_clustering_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_clustering_result_get_clusters(
  cugraph_clustering_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_clustering_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->clusters_->view());
}

extern "C" void cugraph_clustering_result_free(cugraph_clustering_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_clustering_result_t*>(result);
  delete internal_pointer->vertices_;
  delete internal_pointer->clusters_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_balanced_cut_clustering(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  size_t n_eigenvectors,
  double evs_tolerance,
  int evs_max_iterations,
  double k_means_tolerance,
  int k_means_max_iterations,
  bool_t do_expensive_check,
  cugraph_clustering_result_t** result,
  cugraph_error_t** error)
{
  balanced_cut_clustering_functor functor(handle,
                                          graph,
                                          n_clusters,
                                          n_eigenvectors,
                                          evs_tolerance,
                                          evs_max_iterations,
                                          k_means_tolerance,
                                          k_means_max_iterations,
                                          do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
extern "C" cugraph_error_code_t cugraph_spectral_modularity_maximization(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  size_t n_eigenvectors,
  double evs_tolerance,
  int evs_max_iterations,
  double k_means_tolerance,
  int k_means_max_iterations,
  bool_t do_expensive_check,
  cugraph_clustering_result_t** result,
  cugraph_error_t** error)
{
  spectral_clustering_functor functor(handle,
                                      graph,
                                      n_clusters,
                                      n_eigenvectors,
                                      evs_tolerance,
                                      evs_max_iterations,
                                      k_means_tolerance,
                                      k_means_max_iterations,
                                      do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_analyze_clustering_modularity(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* clusters,
  double* score,
  cugraph_error_t** error)
{
  analyze_clustering_modularity_functor functor(handle, graph, n_clusters, vertices, clusters);

  return cugraph::c_api::run_algorithm(graph, functor, score, error);
}

extern "C" cugraph_error_code_t cugraph_analyze_clustering_edge_cut(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* clusters,
  double* score,
  cugraph_error_t** error)
{
  analyze_clustering_edge_cut_functor functor(handle, graph, n_clusters, vertices, clusters);

  return cugraph::c_api::run_algorithm(graph, functor, score, error);
}

extern "C" cugraph_error_code_t cugraph_analyze_clustering_ratio_cut(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* clusters,
  double* score,
  cugraph_error_t** error)
{
  analyze_clustering_ratio_cut_functor functor(handle, graph, n_clusters, vertices, clusters);

  return cugraph::c_api::run_algorithm(graph, functor, score, error);
}
