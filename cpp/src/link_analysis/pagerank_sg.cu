/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <link_analysis/pagerank_impl.cuh>

namespace cugraph {

// SG instantiation
template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                       std::optional<float const*> precomputed_vertex_out_weight_sums,
                       std::optional<int32_t const*> personalization_vertices,
                       std::optional<float const*> personalization_values,
                       std::optional<int32_t> personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                       std::optional<double const*> precomputed_vertex_out_weight_sums,
                       std::optional<int32_t const*> personalization_vertices,
                       std::optional<double const*> personalization_values,
                       std::optional<int32_t> personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int64_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                       std::optional<float const*> precomputed_vertex_out_weight_sums,
                       std::optional<int32_t const*> personalization_vertices,
                       std::optional<float const*> personalization_values,
                       std::optional<int32_t> personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int64_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                       std::optional<double const*> precomputed_vertex_out_weight_sums,
                       std::optional<int32_t const*> personalization_vertices,
                       std::optional<double const*> personalization_values,
                       std::optional<int32_t> personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                       std::optional<float const*> precomputed_vertex_out_weight_sums,
                       std::optional<int64_t const*> personalization_vertices,
                       std::optional<float const*> personalization_values,
                       std::optional<int64_t> personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                       std::optional<double const*> precomputed_vertex_out_weight_sums,
                       std::optional<int64_t const*> personalization_vertices,
                       std::optional<double const*> personalization_values,
                       std::optional<int64_t> personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

}  // namespace cugraph
