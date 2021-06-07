/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cugraph/experimental/graph.hpp>
#include <cugraph/graph.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

extern "C" {
#include "mmio.h"
}

namespace cugraph {
namespace test {

std::string getFileName(const std::string& s);

/// Read matrix properties from Matrix Market file
/** Matrix Market file is assumed to be a sparse matrix in coordinate
 *  format.
 *
 *  @param f File stream for Matrix Market file.
 *  @param tg Boolean indicating whether to convert matrix to general
 *  format (from symmetric, Hermitian, or skew symmetric format).
 *  @param t (Output) MM_typecode with matrix properties.
 *  @param m (Output) Number of matrix rows.
 *  @param n (Output) Number of matrix columns.
 *  @param nnz (Output) Number of non-zero matrix entries.
 *  @return Zero if properties were read successfully. Otherwise
 *  non-zero.
 */
template <typename IndexType_>
int mm_properties(FILE* f, int tg, MM_typecode* t, IndexType_* m, IndexType_* n, IndexType_* nnz);

/// Read Matrix Market file and convert to COO format matrix
/** Matrix Market file is assumed to be a sparse matrix in coordinate
 *  format.
 *
 *  @param f File stream for Matrix Market file.
 *  @param tg Boolean indicating whether to convert matrix to general
 *  format (from symmetric, Hermitian, or skew symmetric format).
 *  @param nnz Number of non-zero matrix entries.
 *  @param cooRowInd (Output) Row indices for COO matrix. Should have
 *  at least nnz entries.
 *  @param cooColInd (Output) Column indices for COO matrix. Should
 *  have at least nnz entries.
 *  @param cooRVal (Output) Real component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @param cooIVal (Output) Imaginary component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @return Zero if matrix was read successfully. Otherwise non-zero.
 */
template <typename IndexType_, typename ValueType_>
int mm_to_coo(FILE* f,
              int tg,
              IndexType_ nnz,
              IndexType_* cooRowInd,
              IndexType_* cooColInd,
              ValueType_* cooRVal,
              ValueType_* cooIVal);

// FIXME: A similar function could be useful for CSC format
//        There are functions above that operate coo -> csr and coo->csc
/**
 * @tparam
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<cugraph::GraphCSR<vertex_t, edge_t, weight_t>> generate_graph_csr_from_mm(
  bool& directed, std::string mm_file);

// Define RAPIDS_DATASET_ROOT_DIR using a preprocessor variable to
// allow for a build to override the default. This is useful for
// having different builds for specific default dataset locations.
#ifndef RAPIDS_DATASET_ROOT_DIR
#define RAPIDS_DATASET_ROOT_DIR "/datasets"
#endif

static const std::string& get_rapids_dataset_root_dir()
{
  static std::string rdrd("");
  // Env var always overrides the value of RAPIDS_DATASET_ROOT_DIR
  if (rdrd == "") {
    const char* envVar = std::getenv("RAPIDS_DATASET_ROOT_DIR");
    rdrd               = (envVar != NULL) ? envVar : RAPIDS_DATASET_ROOT_DIR;
  }
  return rdrd;
}

// returns a tuple of (rows, columns, weights, number_of_vertices, is_symmetric)
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           vertex_t,
           bool>
read_edgelist_from_matrix_market_file(raft::handle_t const& handle,
                                      std::string const& graph_file_full_path,
                                      bool test_weighted);

// renumber must be true if multi_gpu is true
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
read_graph_from_matrix_market_file(raft::handle_t const& handle,
                                   std::string const& graph_file_full_path,
                                   bool test_weighted,
                                   bool renumber);

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
                                bool renumber,
                                std::vector<size_t> const& partition_ids,
                                size_t num_partitions);

// alias for easy customization for debug purposes:
//
template <typename value_t>
using vector_test_t = rmm::device_uvector<value_t>;

template <typename vertex_t, typename edge_t, typename weight_t>
decltype(auto) make_graph(raft::handle_t const& handle,
                          std::vector<vertex_t> const& v_src,
                          std::vector<vertex_t> const& v_dst,
                          std::vector<weight_t> const& v_w,
                          vertex_t num_vertices,
                          edge_t num_edges,
                          bool is_weighted)
{
  using namespace cugraph::experimental;

  vector_test_t<vertex_t> d_src(num_edges, handle.get_stream());
  vector_test_t<vertex_t> d_dst(num_edges, handle.get_stream());
  vector_test_t<weight_t> d_weights(num_edges, handle.get_stream());

  raft::update_device(d_src.data(), v_src.data(), d_src.size(), handle.get_stream());
  raft::update_device(d_dst.data(), v_dst.data(), d_dst.size(), handle.get_stream());

  weight_t* ptr_d_weights{nullptr};
  if (is_weighted) {
    raft::update_device(d_weights.data(), v_w.data(), d_weights.size(), handle.get_stream());

    ptr_d_weights = d_weights.data();
  }

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    d_src.data(), d_dst.data(), ptr_d_weights, num_edges};

  graph_t<vertex_t, edge_t, weight_t, false, false> graph(
    handle, edgelist, num_vertices, graph_properties_t{false, false, is_weighted}, false);

  return graph;
}

// compares single GPU CSR graph data:
// (for testing / debugging);
// on first == false, second == brief description of what is different;
//
template <typename left_graph_t, typename right_graph_t>
std::pair<bool, std::string> compare_graphs(raft::handle_t const& handle,
                                            left_graph_t const& lgraph,
                                            right_graph_t const& rgraph)
{
  if constexpr (left_graph_t::is_multi_gpu && right_graph_t::is_multi_gpu) {
    // no support for comparing distributed graphs, yet:
    //
    CUGRAPH_FAIL("Unsupported graph type for comparison.");
    return std::make_pair(false, std::string("unsupported"));
  } else if constexpr (!std::is_same_v<left_graph_t, right_graph_t>) {
    return std::make_pair(false, std::string("type"));
  } else {
    // both graphs are single GPU:
    //
    using graph_t = left_graph_t;

    using vertex_t = typename graph_t::vertex_type;
    using edge_t   = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;

    size_t num_vertices = lgraph.get_number_of_vertices();
    size_t num_edges    = lgraph.get_number_of_edges();

    {
      size_t r_num_vertices = rgraph.get_number_of_vertices();
      size_t r_num_edges    = rgraph.get_number_of_edges();

      if (num_vertices != r_num_vertices) return std::make_pair(false, std::string("num_vertices"));

      if (num_edges != r_num_edges) return std::make_pair(false, std::string("num_edges"));
    }

    if (lgraph.is_symmetric() != rgraph.is_symmetric())
      return std::make_pair(false, std::string("symmetric"));

    if (lgraph.is_multigraph() != rgraph.is_multigraph())
      return std::make_pair(false, std::string("multigraph"));

    bool is_weighted = lgraph.is_weighted();
    if (is_weighted != rgraph.is_weighted()) return std::make_pair(false, std::string("weighted"));

    auto lgraph_view = lgraph.view();
    auto rgraph_view = rgraph.view();

    std::vector<edge_t> lv_ro(num_vertices + 1);
    std::vector<vertex_t> lv_ci(num_edges);

    raft::update_host(lv_ro.data(), lgraph_view.offsets(), num_vertices + 1, handle.get_stream());
    raft::update_host(lv_ci.data(), lgraph_view.indices(), num_edges, handle.get_stream());

    std::vector<edge_t> rv_ro(num_vertices + 1);
    std::vector<vertex_t> rv_ci(num_edges);

    raft::update_host(rv_ro.data(), rgraph_view.offsets(), num_vertices + 1, handle.get_stream());
    raft::update_host(rv_ci.data(), rgraph_view.indices(), num_edges, handle.get_stream());

    if (lv_ro != rv_ro) return std::make_pair(false, std::string("offsets"));

    if (lv_ci != rv_ci) return std::make_pair(false, std::string("indices"));

    if (is_weighted) {
      std::vector<weight_t> lv_vs(num_edges);
      raft::update_host(lv_vs.data(), lgraph_view.weights(), num_edges, handle.get_stream());

      std::vector<weight_t> rv_vs(num_edges);
      raft::update_host(rv_vs.data(), rgraph_view.weights(), num_edges, handle.get_stream());

      if (lv_vs != rv_vs) return std::make_pair(false, std::string("values"));
    }

    if (lgraph_view.get_local_adj_matrix_partition_segment_offsets(0) !=
        rgraph_view.get_local_adj_matrix_partition_segment_offsets(0))
      return std::make_pair(false, std::string("segment offsets"));

    return std::make_pair(true, std::string{});
  }
}

template <typename vertex_t>
bool renumbered_vectors_same(raft::handle_t const& handle,
                             std::vector<vertex_t> const& v1,
                             std::vector<vertex_t> const& v2)
{
  if (v1.size() != v2.size()) return false;

  std::map<vertex_t, vertex_t> map;

  auto iter = thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin()));

  std::for_each(iter, iter + v1.size(), [&map](auto pair) {
    vertex_t e1 = thrust::get<0>(pair);
    vertex_t e2 = thrust::get<1>(pair);

    map[e1] = e2;
  });

  auto error_count = std::count_if(iter, iter + v1.size(), [&map](auto pair) {
    vertex_t e1 = thrust::get<0>(pair);
    vertex_t e2 = thrust::get<1>(pair);

    return (map[e1] != e2);
  });

  return (error_count == 0);
}

template <typename vertex_t>
bool renumbered_vectors_same(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t> const& v1,
                             rmm::device_uvector<vertex_t> const& v2)
{
  if (v1.size() != v2.size()) return false;

  std::vector<vertex_t> h_v1(v1.size());
  std::vector<vertex_t> h_v2(v1.size());

  raft::update_host(h_v1.data(), v1.data(), v1.size(), handle.get_stream());
  raft::update_host(h_v2.data(), v2.data(), v2.size(), handle.get_stream());

  return renumbered_vectors_same(handle, h_v1, h_v2);
}

}  // namespace test
}  // namespace cugraph
