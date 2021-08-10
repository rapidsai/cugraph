/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <utilities/test_utilities.hpp>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <cstdint>

namespace cugraph {
namespace test {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
generate_graph_from_rmat_params(raft::handle_t const& handle,
                                size_t scale,
                                size_t edge_factor,
                                double a,
                                double b,
                                double c,
                                uint64_t base_seed,
                                bool undirected,
                                bool scramble_vertex_ids,
                                bool test_weighted,
                                bool renumber,
                                std::vector<size_t> const& partition_ids,
                                size_t num_partitions)
{
  CUGRAPH_EXPECTS(!multi_gpu || renumber, "renumber should be true if multi_gpu is true.");
  CUGRAPH_EXPECTS(size_t{1} << scale <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
                  "vertex_t overflow.");
  CUGRAPH_EXPECTS(
    (size_t{1} << scale) * edge_factor <= static_cast<size_t>(std::numeric_limits<edge_t>::max()),
    " edge_t overflow.");

  vertex_t number_of_vertices = static_cast<vertex_t>(size_t{1} << scale);
  edge_t number_of_edges =
    static_cast<edge_t>(static_cast<size_t>(number_of_vertices) * edge_factor);

  std::vector<edge_t> partition_edge_counts(partition_ids.size());
  std::vector<vertex_t> partition_vertex_firsts(partition_ids.size());
  std::vector<vertex_t> partition_vertex_lasts(partition_ids.size());
  for (size_t i = 0; i < partition_ids.size(); ++i) {
    auto id = partition_ids[i];

    partition_edge_counts[i] = number_of_edges / num_partitions +
                               (id < number_of_edges % num_partitions ? edge_t{1} : edge_t{0});

    partition_vertex_firsts[i] = (number_of_vertices / num_partitions) * id;
    partition_vertex_lasts[i]  = (number_of_vertices / num_partitions) * (id + 1);
    if (id < number_of_vertices % num_partitions) {
      partition_vertex_firsts[i] += id;
      partition_vertex_lasts[i] += id + 1;
    } else {
      partition_vertex_firsts[i] += number_of_vertices % num_partitions;
      partition_vertex_lasts[i] += number_of_vertices % num_partitions;
    }
  }

  rmm::device_uvector<vertex_t> d_edgelist_rows(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_edgelist_cols(0, handle.get_stream());
  auto d_edgelist_weights =
    test_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
                  : std::nullopt;
  for (size_t i = 0; i < partition_ids.size(); ++i) {
    auto id = partition_ids[i];

    rmm::device_uvector<vertex_t> d_tmp_rows(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_tmp_cols(0, handle.get_stream());
    std::tie(i == 0 ? d_edgelist_rows : d_tmp_rows, i == 0 ? d_edgelist_cols : d_tmp_cols) =
      cugraph::generate_rmat_edgelist<vertex_t>(handle,
                                                scale,
                                                partition_edge_counts[i],
                                                a,
                                                b,
                                                c,
                                                base_seed + id,
                                                undirected ? true : false);

    std::optional<rmm::device_uvector<weight_t>> d_tmp_weights{std::nullopt};
    if (d_edgelist_weights) {
      if (i == 0) {
        (*d_edgelist_weights).resize(d_edgelist_rows.size(), handle.get_stream());
      } else {
        d_tmp_weights =
          std::make_optional<rmm::device_uvector<weight_t>>(d_tmp_rows.size(), handle.get_stream());
      }

      cugraph::detail::uniform_random_fill(
        handle.get_stream_view(),
        i == 0 ? (*d_edgelist_weights).data() : (*d_tmp_weights).data(),
        i == 0 ? (*d_edgelist_weights).size() : (*d_tmp_weights).size(),
        weight_t{0.0},
        weight_t{1.0},
        base_seed + num_partitions + id);
    }

    if (i > 0) {
      auto start_offset = d_edgelist_rows.size();
      d_edgelist_rows.resize(start_offset + d_tmp_rows.size(), handle.get_stream());
      d_edgelist_cols.resize(d_edgelist_rows.size(), handle.get_stream());
      raft::copy(d_edgelist_rows.begin() + start_offset,
                 d_tmp_rows.begin(),
                 d_tmp_rows.size(),
                 handle.get_stream());
      raft::copy(d_edgelist_cols.begin() + start_offset,
                 d_tmp_cols.begin(),
                 d_tmp_cols.size(),
                 handle.get_stream());
      if (d_edgelist_weights) {
        (*d_edgelist_weights).resize(d_edgelist_rows.size(), handle.get_stream());
        raft::copy(d_edgelist_weights->begin() + start_offset,
                   d_tmp_weights->begin(),
                   d_tmp_weights->size(),
                   handle.get_stream());
      }
    }
  }

  if (undirected) {
// FIXME: may need to undo this and handle symmetrization elsewhere once the new test graph
// generation API gets integrated
#if 1
    std::tie(d_edgelist_rows, d_edgelist_cols, d_edgelist_weights) =
      cugraph::symmetrize_edgelist<vertex_t, weight_t>(
        handle,
        std::move(d_edgelist_rows),
        std::move(d_edgelist_cols),
        test_weighted ? std::optional<rmm::device_uvector<weight_t>>(std::move(d_edgelist_weights))
                      : std::nullopt);
#endif
  }

  if (multi_gpu) {
    std::tie(d_edgelist_rows, d_edgelist_cols, d_edgelist_weights) =
      cugraph::detail::shuffle_edgelist_by_gpu_id(
        handle, d_edgelist_rows, d_edgelist_cols, d_edgelist_weights, store_transposed);
  }

  rmm::device_uvector<vertex_t> d_vertices(0, handle.get_stream());
  for (size_t i = 0; i < partition_ids.size(); ++i) {
    auto id = partition_ids[i];

    auto start_offset = d_vertices.size();
    d_vertices.resize(start_offset + (partition_vertex_lasts[i] - partition_vertex_firsts[i]),
                      handle.get_stream());
    cugraph::detail::sequence_fill(handle.get_stream_view(),
                                   d_vertices.begin() + start_offset,
                                   d_vertices.size() - start_offset,
                                   partition_vertex_firsts[i]);
  }

  if (multi_gpu) { d_vertices = cugraph::detail::shuffle_vertices_by_gpu_id(handle, d_vertices); }

  return cugraph::
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::optional<std::tuple<vertex_t const*, vertex_t>>{
        std::make_tuple(d_vertices.data(), static_cast<vertex_t>(d_vertices.size()))},
      std::move(d_edgelist_rows),
      std::move(d_edgelist_cols),
      std::move(d_edgelist_weights),
      cugraph::graph_properties_t{undirected, true},
      renumber);
}  // namespace test

// explicit instantiations

template std::tuple<cugraph::graph_t<int32_t, int32_t, float, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int32_t, float, false, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int32_t, float, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int32_t, float, true, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int32_t, double, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int32_t, double, false, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int32_t, double, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int32_t, double, true, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, float, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, float, false, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, float, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, float, true, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, double, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, double, false, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, double, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int32_t, int64_t, double, true, true>,
                    std::optional<rmm::device_uvector<int32_t>>>
generate_graph_from_rmat_params<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, float, false, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, float, false, true>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, float, true, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, float, true, true>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, double, false, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, double, false, true>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, double, true, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

template std::tuple<cugraph::graph_t<int64_t, int64_t, double, true, true>,
                    std::optional<rmm::device_uvector<int64_t>>>
generate_graph_from_rmat_params<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  size_t scale,
  size_t edge_factor,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool undirected,
  bool scramble_vertex_ids,
  bool test_weighted,
  bool renumber,
  std::vector<size_t> const& partition_ids,
  size_t num_partitions);

}  // namespace test
}  // namespace cugraph
