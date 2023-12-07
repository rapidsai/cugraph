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

#pragma once

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>
#include <cugraph/mtmg/edge_property.hpp>
#include <cugraph/mtmg/graph_view.hpp>
#include <cugraph/mtmg/handle.hpp>
#include <cugraph/mtmg/renumber_map.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Graph object for each GPU
 */
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
class graph_t : public detail::device_shared_wrapper_t<
                  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>> {
  using parent_t = detail::device_shared_wrapper_t<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>>;

 public:
  /**
   * @brief Create an MTMG graph view (read only)
   */
  auto view()
  {
    std::lock_guard<std::mutex> lock(parent_t::lock_);

    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> result;

    std::for_each(parent_t::objects_.begin(), parent_t::objects_.end(), [&result](auto& p) {
      result.set(p.first, std::move(p.second.view()));
    });

    return result;
  }
};

/**
 * @brief Create an MTMG graph from an edgelist
 *
 * @param[in]  handle             Resource handle
 * @param[in]  edgelist           Edgelist
 * @param[in]  graph_properties   Graph properties
 * @param[in]  renumber           If true, renumber graph (must be true for MG)
 * @param[out] graph              MTMG graph is stored here
 * @param[out] edge_weights       MTMG edge weights is stored here
 * @param[out] edge_ids           MTMG edge ids is stored here
 * @param[out] edge_types         MTMG edge types is stored here
 * @param[in]  renumber_map       MTMG renumber_map is stored here
 * @param[in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
void create_graph_from_edgelist(
  handle_t const& handle,
  cugraph::mtmg::edgelist_t<vertex_t, weight_t, edge_id_t, edge_type_t>& edgelist,
  graph_properties_t graph_properties,
  bool renumber,
  cugraph::mtmg::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>& graph,
  std::optional<cugraph::mtmg::edge_property_t<
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    weight_t>>& edge_weights,
  std::optional<cugraph::mtmg::edge_property_t<
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    edge_id_t>>& edge_ids,
  std::optional<cugraph::mtmg::edge_property_t<
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    edge_type_t>>& edge_types,
  std::optional<cugraph::mtmg::renumber_map_t<vertex_t>>& renumber_map,
  bool do_expensive_check = false)
{
  if (handle.get_thread_rank() > 0) return;

  CUGRAPH_EXPECTS(renumber_map.has_value() == renumber,
                  "Renumbering set to true, but no space for renumber map");

  auto& my_edgelist = edgelist.get(handle);

  CUGRAPH_EXPECTS(my_edgelist.get_src().size() > 0, "Cannot create graph without an edge list");
  CUGRAPH_EXPECTS(my_edgelist.get_src().size() == 1,
                  "Must consolidate edges into a single list before creating graph");

  auto [local_graph, local_edge_weights, local_edge_ids, local_edge_types, local_renumber_map] =
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_id_t,
                                        edge_type_t,
                                        store_transposed,
                                        multi_gpu>(
      handle.raft_handle(),
      std::nullopt,
      std::move(my_edgelist.get_src()[0]),
      std::move(my_edgelist.get_dst()[0]),
      my_edgelist.get_wgt() ? std::make_optional(std::move((*my_edgelist.get_wgt())[0]))
                            : std::nullopt,
      my_edgelist.get_edge_id() ? std::make_optional(std::move((*my_edgelist.get_edge_id())[0]))
                                : std::nullopt,
      my_edgelist.get_edge_type() ? std::make_optional(std::move((*my_edgelist.get_edge_type())[0]))
                                  : std::nullopt,
      graph_properties,
      renumber,
      do_expensive_check);

  graph.set(handle, std::move(local_graph));
  if (edge_weights) edge_weights->set(handle, std::move(*local_edge_weights));
  if (edge_ids) edge_ids->set(handle, std::move(*local_edge_ids));
  if (edge_types) edge_types->set(handle, std::move(*local_edge_types));
  if (renumber) renumber_map->set(handle, std::move(*local_renumber_map));
}

}  // namespace mtmg
}  // namespace cugraph
