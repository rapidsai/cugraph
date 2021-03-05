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

#include <experimental/graph_generator.hpp>
#include <utilities/error.hpp>

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
                                uint64_t seed,
                                bool undirected,
                                bool scramble_vertex_ids,
                                bool test_weighted,
                                bool renumber)
{
  rmm::device_uvector<vertex_t> d_edgelist_rows(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_edgelist_cols(0, handle.get_stream());
  std::tie(d_edgelist_rows, d_edgelist_cols) =
    cugraph::experimental::generate_rmat_edgelist<vertex_t>(
      handle, scale, edge_factor, a, b, c, seed, undirected ? true : false, scramble_vertex_ids);
  if (undirected) {
    // FIXME: need to symmetrize
    CUGRAPH_FAIL("unimplemented.");
  }

  rmm::device_uvector<weight_t> d_edgelist_weights(test_weighted ? d_edgelist_rows.size() : 0,
                                                   handle.get_stream());
  if (test_weighted) {
    raft::random::Rng rng(seed + 1);
    rng.uniform<weight_t, size_t>(d_edgelist_weights.data(),
                                  d_edgelist_weights.size(),
                                  weight_t{0.0},
                                  weight_t{1.0},
                                  handle.get_stream());
  }

  rmm::device_uvector<vertex_t> d_vertices(static_cast<vertex_t>(size_t{1} << scale),
                                           handle.get_stream());
  thrust::sequence(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   d_vertices.begin(),
                   d_vertices.end(),
                   vertex_t{0});

  return generate_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle,
    std::move(d_vertices),
    std::move(d_edgelist_rows),
    std::move(d_edgelist_cols),
    std::move(d_edgelist_weights),
    false,
    test_weighted,
    renumber);
}

// explicit instantiations

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, false, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int32_t, float, false, false>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int32_t, float, false, true>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int32_t, float, true, false>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int32_t, float, true, true>(raft::handle_t const& handle,
                                                                     size_t scale,
                                                                     size_t edge_factor,
                                                                     double a,
                                                                     double b,
                                                                     double c,
                                                                     uint64_t seed,
                                                                     bool undirected,
                                                                     bool scramble_vertex_ids,
                                                                     bool test_weighted,
                                                                     bool renumber);

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
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int32_t, double, false, true>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int32_t, double, true, false>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int32_t, double, true, true>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, false, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int64_t, float, false, false>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int64_t, float, false, true>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int64_t, float, true, false>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int64_t, float, true, true>(raft::handle_t const& handle,
                                                                     size_t scale,
                                                                     size_t edge_factor,
                                                                     double a,
                                                                     double b,
                                                                     double c,
                                                                     uint64_t seed,
                                                                     bool undirected,
                                                                     bool scramble_vertex_ids,
                                                                     bool test_weighted,
                                                                     bool renumber);

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
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int64_t, double, false, true>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int64_t, double, true, false>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_rmat_params<int32_t, int64_t, double, true, true>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, false, false>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_rmat_params<int64_t, int64_t, float, false, false>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, false, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_rmat_params<int64_t, int64_t, float, false, true>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, true, false>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_rmat_params<int64_t, int64_t, float, true, false>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, true, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_rmat_params<int64_t, int64_t, float, true, true>(raft::handle_t const& handle,
                                                                     size_t scale,
                                                                     size_t edge_factor,
                                                                     double a,
                                                                     double b,
                                                                     double c,
                                                                     uint64_t seed,
                                                                     bool undirected,
                                                                     bool scramble_vertex_ids,
                                                                     bool test_weighted,
                                                                     bool renumber);

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
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, false, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_rmat_params<int64_t, int64_t, double, false, true>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, true, false>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_rmat_params<int64_t, int64_t, double, true, false>(raft::handle_t const& handle,
                                                                       size_t scale,
                                                                       size_t edge_factor,
                                                                       double a,
                                                                       double b,
                                                                       double c,
                                                                       uint64_t seed,
                                                                       bool undirected,
                                                                       bool scramble_vertex_ids,
                                                                       bool test_weighted,
                                                                       bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, true, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_rmat_params<int64_t, int64_t, double, true, true>(raft::handle_t const& handle,
                                                                      size_t scale,
                                                                      size_t edge_factor,
                                                                      double a,
                                                                      double b,
                                                                      double c,
                                                                      uint64_t seed,
                                                                      bool undirected,
                                                                      bool scramble_vertex_ids,
                                                                      bool test_weighted,
                                                                      bool renumber);

}  // namespace test
}  // namespace cugraph
