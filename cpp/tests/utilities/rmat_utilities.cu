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

#include <cugraph/experimental/detail/graph_utils.cuh>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/random/rng.cuh>

#include <thrust/sequence.h>

#include <cstdint>

namespace cugraph {
namespace test {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
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
  rmm::device_uvector<weight_t> d_edgelist_weights(0, handle.get_stream());
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

    rmm::device_uvector<weight_t> d_tmp_weights(0, handle.get_stream());
    if (test_weighted) {
      if (i == 0) {
        d_edgelist_weights.resize(d_edgelist_rows.size(), handle.get_stream());
      } else {
        d_tmp_weights.resize(d_tmp_rows.size(), handle.get_stream());
      }

      raft::random::Rng rng(base_seed + num_partitions + id);
      rng.uniform<weight_t, size_t>(i == 0 ? d_edgelist_weights.data() : d_tmp_weights.data(),
                                    i == 0 ? d_edgelist_weights.size() : d_tmp_weights.size(),
                                    weight_t{0.0},
                                    weight_t{1.0},
                                    handle.get_stream());
    }

    if (i > 0) {
      auto start_offset = d_edgelist_rows.size();
      d_edgelist_rows.resize(start_offset + d_tmp_rows.size(), handle.get_stream());
      d_edgelist_cols.resize(d_edgelist_rows.size(), handle.get_stream());
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   d_tmp_rows.begin(),
                   d_tmp_rows.end(),
                   d_edgelist_rows.begin() + start_offset);
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   d_tmp_cols.begin(),
                   d_tmp_cols.end(),
                   d_edgelist_cols.begin() + start_offset);
      if (test_weighted) {
        d_edgelist_weights.resize(d_edgelist_rows.size(), handle.get_stream());
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     d_tmp_weights.begin(),
                     d_tmp_weights.end(),
                     d_edgelist_weights.begin() + start_offset);
      }
    }
  }

  if (undirected) {
// FIXME: may need to undo this and handle symmetrization elsewhere once the new test graph
// generation API gets integrated
#if 1
    auto offset = d_edgelist_rows.size();
    d_edgelist_rows.resize(offset * 2, handle.get_stream());
    d_edgelist_cols.resize(d_edgelist_rows.size(), handle.get_stream());
    d_edgelist_weights.resize(test_weighted ? d_edgelist_rows.size() : size_t{0},
                              handle.get_stream());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 d_edgelist_cols.begin(),
                 d_edgelist_cols.begin() + offset,
                 d_edgelist_rows.begin() + offset);
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 d_edgelist_rows.begin(),
                 d_edgelist_rows.begin() + offset,
                 d_edgelist_cols.begin() + offset);
    if (test_weighted) {
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   d_edgelist_weights.begin(),
                   d_edgelist_weights.begin() + offset,
                   d_edgelist_weights.begin() + offset);
    }
#endif
  }

  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();

    rmm::device_uvector<vertex_t> d_rx_edgelist_rows(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_rx_edgelist_cols(0, handle.get_stream());
    rmm::device_uvector<weight_t> d_rx_edgelist_weights(0, handle.get_stream());
    if (test_weighted) {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(store_transposed ? d_edgelist_cols.begin() : d_edgelist_rows.begin(),
                           store_transposed ? d_edgelist_rows.begin() : d_edgelist_cols.begin(),
                           d_edgelist_weights.begin()));

      std::forward_as_tuple(std::tie(store_transposed ? d_rx_edgelist_cols : d_rx_edgelist_rows,
                                     store_transposed ? d_rx_edgelist_rows : d_rx_edgelist_cols,
                                     d_rx_edgelist_weights),
                            std::ignore) =
        cugraph::experimental::groupby_gpuid_and_shuffle_values(
          comm,  // handle.get_comms(),
          edge_first,
          edge_first + d_edgelist_rows.size(),
          [key_func =
             cugraph::experimental::detail::compute_gpu_id_from_edge_t<vertex_t>{
               comm_size, row_comm_size, col_comm_size}] __device__(auto val) {
            return key_func(thrust::get<0>(val), thrust::get<1>(val));
          },
          handle.get_stream());
    } else {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(store_transposed ? d_edgelist_cols.begin() : d_edgelist_rows.begin(),
                           store_transposed ? d_edgelist_rows.begin() : d_edgelist_cols.begin()));

      std::forward_as_tuple(std::tie(store_transposed ? d_rx_edgelist_cols : d_rx_edgelist_rows,
                                     store_transposed ? d_rx_edgelist_rows : d_rx_edgelist_cols),
                            std::ignore) =
        cugraph::experimental::groupby_gpuid_and_shuffle_values(
          comm,  // handle.get_comms(),
          edge_first,
          edge_first + d_edgelist_rows.size(),
          [key_func =
             cugraph::experimental::detail::compute_gpu_id_from_edge_t<vertex_t>{
               comm_size, row_comm_size, col_comm_size}] __device__(auto val) {
            return key_func(thrust::get<0>(val), thrust::get<1>(val));
          },
          handle.get_stream());
    }

    d_edgelist_rows    = std::move(d_rx_edgelist_rows);
    d_edgelist_cols    = std::move(d_rx_edgelist_cols);
    d_edgelist_weights = std::move(d_rx_edgelist_weights);
  }

  rmm::device_uvector<vertex_t> d_vertices(0, handle.get_stream());
  for (size_t i = 0; i < partition_ids.size(); ++i) {
    auto id = partition_ids[i];

    auto start_offset = d_vertices.size();
    d_vertices.resize(start_offset + (partition_vertex_lasts[i] - partition_vertex_firsts[i]),
                      handle.get_stream());
    thrust::sequence(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     d_vertices.begin() + start_offset,
                     d_vertices.end(),
                     partition_vertex_firsts[i]);
  }

  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    rmm::device_uvector<vertex_t> d_rx_vertices(0, handle.get_stream());
    std::tie(d_rx_vertices, std::ignore) = cugraph::experimental::groupby_gpuid_and_shuffle_values(
      comm,  // handle.get_comms(),
      d_vertices.begin(),
      d_vertices.end(),
      [key_func =
         cugraph::experimental::detail::compute_gpu_id_from_vertex_t<vertex_t>{
           comm_size}] __device__(auto val) { return key_func(val); },
      handle.get_stream());
    d_vertices = std::move(d_rx_vertices);
  }

  return cugraph::experimental::
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::optional<std::tuple<vertex_t const*, vertex_t>>{
        std::make_tuple(d_vertices.data(), static_cast<vertex_t>(d_vertices.size()))},
      std::move(d_edgelist_rows),
      std::move(d_edgelist_cols),
      std::move(d_edgelist_weights),
      cugraph::experimental::graph_properties_t{undirected, true, test_weighted},
      renumber);
}

// explicit instantiations

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, false, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, false, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, true, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, true, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, false, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, false, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, true, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, true, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, false, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, false, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, true, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, true, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, false, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, false, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, true, false>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, true, true>,
                    rmm::device_uvector<int32_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, false, false>,
                    rmm::device_uvector<int64_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, false, true>,
                    rmm::device_uvector<int64_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, true, false>,
                    rmm::device_uvector<int64_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, true, true>,
                    rmm::device_uvector<int64_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, false, false>,
                    rmm::device_uvector<int64_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, false, true>,
                    rmm::device_uvector<int64_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, true, false>,
                    rmm::device_uvector<int64_t>>
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

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, true, true>,
                    rmm::device_uvector<int64_t>>
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
