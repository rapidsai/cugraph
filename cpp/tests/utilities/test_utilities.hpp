/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/legacy/graph.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <numeric>
#include <optional>
#include <random>
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
std::unique_ptr<cugraph::legacy::GraphCSR<vertex_t, edge_t, weight_t>> generate_graph_csr_from_mm(
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
template <typename vertex_t, typename weight_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<vertex_t>,
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
std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
read_graph_from_matrix_market_file(raft::handle_t const& handle,
                                   std::string const& graph_file_full_path,
                                   bool test_weighted,
                                   bool renumber);

// alias for easy customization for debug purposes:
//
template <typename value_t>
using vector_test_t = rmm::device_uvector<value_t>;

template <typename vertex_t, typename edge_t, typename weight_t>
decltype(auto) make_graph(raft::handle_t const& handle,
                          std::vector<vertex_t> const& v_src,
                          std::vector<vertex_t> const& v_dst,
                          std::optional<std::vector<weight_t>> const& v_w,
                          vertex_t num_vertices,
                          edge_t num_edges)
{
  vector_test_t<vertex_t> d_src(num_edges, handle.get_stream());
  vector_test_t<vertex_t> d_dst(num_edges, handle.get_stream());
  auto d_w = v_w ? std::make_optional<vector_test_t<weight_t>>(num_edges, handle.get_stream())
                 : std::nullopt;

  raft::update_device(d_src.data(), v_src.data(), d_src.size(), handle.get_stream());
  raft::update_device(d_dst.data(), v_dst.data(), d_dst.size(), handle.get_stream());
  if (d_w) {
    raft::update_device((*d_w).data(), (*v_w).data(), (*d_w).size(), handle.get_stream());
  }

  cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
  std::tie(graph, std::ignore) =
    cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, false, false>(
      handle,
      std::nullopt,
      std::move(d_src),
      std::move(d_dst),
      std::move(d_w),
      cugraph::graph_properties_t{false, false},
      false);

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

    size_t num_vertices = lgraph.number_of_vertices();
    size_t num_edges    = lgraph.number_of_edges();

    {
      size_t r_num_vertices = rgraph.number_of_vertices();
      size_t r_num_edges    = rgraph.number_of_edges();

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

    raft::update_host(lv_ro.data(),
                      lgraph_view.local_edge_partition_view().offsets(),
                      num_vertices + 1,
                      handle.get_stream());
    raft::update_host(lv_ci.data(),
                      lgraph_view.local_edge_partition_view().indices(),
                      num_edges,
                      handle.get_stream());

    std::vector<edge_t> rv_ro(num_vertices + 1);
    std::vector<vertex_t> rv_ci(num_edges);

    raft::update_host(rv_ro.data(),
                      rgraph_view.local_edge_partition_view().offsets(),
                      num_vertices + 1,
                      handle.get_stream());
    raft::update_host(rv_ci.data(),
                      rgraph_view.local_edge_partition_view().indices(),
                      num_edges,
                      handle.get_stream());

    auto lv_vs = is_weighted ? std::make_optional<std::vector<weight_t>>(num_edges) : std::nullopt;
    auto rv_vs = is_weighted ? std::make_optional<std::vector<weight_t>>(num_edges) : std::nullopt;
    if (is_weighted) {
      raft::update_host((*lv_vs).data(),
                        *(lgraph_view.local_edge_partition_view().weights()),
                        num_edges,
                        handle.get_stream());

      raft::update_host((*rv_vs).data(),
                        *(rgraph_view.local_edge_partition_view().weights()),
                        num_edges,
                        handle.get_stream());
    }

    handle.sync_stream();

    if (lv_ro != rv_ro) return std::make_pair(false, std::string("offsets"));

    for (size_t i = 0; i < num_vertices; ++i) {
      auto first = lv_ro[i];
      auto last  = lv_ro[i + 1];
      if (is_weighted) {
        std::vector<std::tuple<vertex_t, weight_t>> lv_pairs(last - first);
        std::vector<std::tuple<vertex_t, weight_t>> rv_pairs(last - first);
        for (edge_t j = first; j < last; ++j) {
          lv_pairs[j - first] = std::make_tuple(lv_ci[j], (*lv_vs)[j]);
          rv_pairs[j - first] = std::make_tuple(rv_ci[j], (*rv_vs)[j]);
        }
        std::sort(lv_pairs.begin(), lv_pairs.end());
        std::sort(rv_pairs.begin(), rv_pairs.end());
        if (!std::equal(lv_pairs.begin(), lv_pairs.end(), rv_pairs.begin(), [](auto lhs, auto rhs) {
              return std::get<0>(lhs) == std::get<0>(rhs);
            }))
          return std::make_pair(false, std::string("indices"));
        if (!std::equal(lv_pairs.begin(), lv_pairs.end(), rv_pairs.begin(), [](auto lhs, auto rhs) {
              return std::get<1>(lhs) == std::get<1>(rhs);
            }))
          return std::make_pair(false, std::string("values"));
      } else {
        std::sort(lv_ci.begin() + first, lv_ci.begin() + last);
        std::sort(rv_ci.begin() + first, rv_ci.begin() + last);
        if (!std::equal(lv_ci.begin() + first, lv_ci.begin() + last, rv_ci.begin() + first))
          return std::make_pair(false, std::string("indices"));
      }
    }

    if (lgraph_view.local_edge_partition_segment_offsets(0) !=
        rgraph_view.local_edge_partition_segment_offsets(0))
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

template <typename T, typename L>
std::vector<T> to_host(raft::handle_t const& handle, T const* data, L size)
{
  std::vector<T> h_data(size);
  raft::update_host(h_data.data(), data, size, handle.get_stream());
  handle.sync_stream();
  return h_data;
}

template <typename vertex_t>
bool renumbered_vectors_same(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t> const& v1,
                             rmm::device_uvector<vertex_t> const& v2)
{
  if (v1.size() != v2.size()) return false;

  return renumbered_vectors_same(
    handle, to_host(handle, v1.data(), v1.size()), to_host(handle, v2.data(), v2.size()));
}

template <typename T, typename L>
std::vector<T> random_vector(L size, unsigned seed = 0)
{
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<T> dist(0.0, 1.0);
  std::vector<T> v(size);
  std::generate(v.begin(), v.end(), [&] { return dist(gen); });
  return v;
}

}  // namespace test
}  // namespace cugraph
