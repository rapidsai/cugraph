/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/core/span.hpp>
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

// returns a tuple of (rows, columns, weights, vertices, is_symmetric)
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<vertex_t>,
           bool>
read_edgelist_from_matrix_market_file(raft::handle_t const& handle,
                                      std::string const& graph_file_full_path,
                                      bool test_weighted,
                                      bool store_transposed,
                                      bool multi_gpu);

// renumber must be true if multi_gpu is true
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<
             cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                      weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
read_graph_from_matrix_market_file(raft::handle_t const& handle,
                                   std::string const& graph_file_full_path,
                                   bool test_weighted,
                                   bool renumber);

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           bool>
read_edgelist_from_csv_file(raft::handle_t const& handle,
                            std::string const& graph_file_full_path,
                            bool test_weighted,
                            bool store_transposed,
                            bool multi_gpu);

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

  cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
  std::optional<cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
    edge_weights{std::nullopt};
  std::tie(graph, edge_weights, std::ignore, std::ignore) =
    cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, false, false>(
      handle,
      std::nullopt,
      std::move(d_src),
      std::move(d_dst),
      std::move(d_w),
      std::nullopt,
      cugraph::graph_properties_t{false, false},
      false);

  return std::make_tuple(std::move(graph), std::move(edge_weights));
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
                      lgraph_view.local_edge_partition_view().offsets().data(),
                      num_vertices + 1,
                      handle.get_stream());
    raft::update_host(lv_ci.data(),
                      lgraph_view.local_edge_partition_view().indices().data(),
                      num_edges,
                      handle.get_stream());

    std::vector<edge_t> rv_ro(num_vertices + 1);
    std::vector<vertex_t> rv_ci(num_edges);

    raft::update_host(rv_ro.data(),
                      rgraph_view.local_edge_partition_view().offsets().data(),
                      num_vertices + 1,
                      handle.get_stream());
    raft::update_host(rv_ci.data(),
                      rgraph_view.local_edge_partition_view().indices().data(),
                      num_edges,
                      handle.get_stream());

    auto lv_vs = is_weighted ? std::make_optional<std::vector<weight_t>>(num_edges) : std::nullopt;
    auto rv_vs = is_weighted ? std::make_optional<std::vector<weight_t>>(num_edges) : std::nullopt;
    if (is_weighted) {
      raft::update_host((*lv_vs).data(),
                        (*(lgraph_view.local_edge_partition_view().weights())).data(),
                        num_edges,
                        handle.get_stream());

      raft::update_host((*rv_vs).data(),
                        (*(rgraph_view.local_edge_partition_view().weights())).data(),
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

// check the mapping between v0 & v1 is invertible (one to one and onto)
template <typename vertex_t>
bool check_invertible(raft::host_span<vertex_t const> v0, raft::host_span<vertex_t const> v1)
{
  if (v0.size() != v1.size()) return false;

  std::map<vertex_t, vertex_t> map{};

  for (size_t i = 0; i < v0.size(); ++i) {
    auto find_it = map.find(v0[i]);
    if (find_it == map.end()) {
      map[v0[i]] = v1[i];
    } else if (find_it->second != v1[i])
      return false;  // one value in v0 is mapped to multiple distinct values in v1, so v0 to v1 is
                     // not a function
  }

  std::map<vertex_t, vertex_t> inv_map{};
  for (auto it = map.begin(); it != map.end(); ++it) {
    auto find_it = inv_map.find(it->second);
    if (find_it == inv_map.end()) {
      inv_map[it->second] = it->first;
    } else
      return false;  // multiple distinct values in v0 are mapped to one value in v1
  }

  std::vector<vertex_t> inv_v1(v1.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    auto find_it = inv_map.find(v1[i]);
    if (find_it == inv_map.end()) return false;  // elements in v0 are mapped to only a subset of v1
    inv_v1[i] = find_it->second;
  }

  return std::equal(v0.begin(), v0.end(), inv_v1.begin());
}

template <typename T>
std::vector<T> to_host(raft::handle_t const& handle, raft::device_span<T const> data)
{
  std::vector<T> h_data(data.size());
  raft::update_host(h_data.data(), data.data(), data.size(), handle.get_stream());
  handle.sync_stream();
  return h_data;
}

template <typename T>
std::vector<T> to_host(raft::handle_t const& handle, rmm::device_uvector<T> const& data)
{
  std::vector<T> h_data(data.size());
  raft::update_host(h_data.data(), data.data(), data.size(), handle.get_stream());
  handle.sync_stream();
  return h_data;
}

template <typename T>
std::optional<std::vector<T>> to_host(raft::handle_t const& handle,
                                      std::optional<raft::device_span<T const>> data)
{
  std::optional<std::vector<T>> h_data{std::nullopt};
  if (data) {
    h_data = std::vector<T>((*data).size());
    raft::update_host((*h_data).data(), (*data).data(), (*data).size(), handle.get_stream());
    handle.sync_stream();
  }
  return h_data;
}

template <typename T>
std::optional<std::vector<T>> to_host(raft::handle_t const& handle,
                                      std::optional<rmm::device_uvector<T>> const& data)
{
  std::optional<std::vector<T>> h_data{std::nullopt};
  if (data) {
    h_data = std::vector<T>((*data).size());
    raft::update_host((*h_data).data(), (*data).data(), (*data).size(), handle.get_stream());
    handle.sync_stream();
  }
  return h_data;
}

template <typename T>
rmm::device_uvector<T> to_device(raft::handle_t const& handle, raft::host_span<T const> data)
{
  rmm::device_uvector<T> d_data(data.size(), handle.get_stream());
  raft::update_device(d_data.data(), data.data(), data.size(), handle.get_stream());
  handle.sync_stream();
  return d_data;
}

template <typename T>
rmm::device_uvector<T> to_device(raft::handle_t const& handle, std::vector<T> const& data)
{
  rmm::device_uvector<T> d_data(data.size(), handle.get_stream());
  raft::update_device(d_data.data(), data.data(), data.size(), handle.get_stream());
  handle.sync_stream();
  return d_data;
}

template <typename T>
std::optional<rmm::device_uvector<T>> to_device(raft::handle_t const& handle,
                                                std::optional<raft::host_span<T const>> data)
{
  std::optional<rmm::device_uvector<T>> d_data{std::nullopt};
  if (data) {
    d_data = rmm::device_uvector<T>(data->size(), handle.get_stream());
    raft::update_device(d_data->data(), data->data(), data->size(), handle.get_stream());
    handle.sync_stream();
  }
  return d_data;
}

template <typename T>
std::optional<rmm::device_uvector<T>> to_device(raft::handle_t const& handle,
                                                std::optional<std::vector<T>> const& data)
{
  std::optional<rmm::device_uvector<T>> d_data{std::nullopt};
  if (data) {
    d_data = rmm::device_uvector<T>(data->size(), handle.get_stream());
    raft::update_host(d_data->data(), data->data(), data->size(), handle.get_stream());
    handle.sync_stream();
  }
  return d_data;
}

template <typename vertex_t>
bool check_invertible(raft::handle_t const& handle,
                      raft::device_span<vertex_t const> v0,
                      raft::device_span<vertex_t const> v1)
{
  auto v0_copy = to_host(handle, v0);
  auto v1_copy = to_host(handle, v1);

  return check_invertible(raft::host_span<vertex_t const>(v0_copy.data(), v0_copy.size()),
                          raft::host_span<vertex_t const>(v1_copy.data(), v1_copy.size()));
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

// If multi-GPU, only the rank 0 GPU holds the valid data
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<vertex_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view);

// If multi-GPU, only the rank 0 GPU holds the valid data
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view);

// If multi-GPU, only the rank 0 GPU holds the valid data
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view);

// Only the rank 0 GPU holds the valid data
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, false>,
           std::optional<cugraph::edge_property_t<
             cugraph::graph_view_t<vertex_t, edge_t, store_transposed, false>,
             weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> number_map,
  bool renumber);

// Only the rank 0 GPU holds the valid data
template <typename vertex_t, typename value_t>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>, rmm::device_uvector<value_t>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<vertex_t const>>
    mg_renumber_map,  // std::nullopt if the MG graph is not renumbered
  std::tuple<vertex_t, vertex_t> mg_vertex_partition_range,
  std::optional<raft::device_span<vertex_t const>>
    sg_renumber_map,  // std::nullopt if the SG graph is not renumbered
  std::optional<raft::device_span<vertex_t const>>
    mg_vertices,      // std::nullopt if the entire local vertex partition range is assumed
  raft::device_span<value_t const> mg_values);

template <typename type_t>
struct nearly_equal {
  const type_t threshold_ratio;
  const type_t threshold_magnitude;

  bool operator()(type_t lhs, type_t rhs) const
  {
    return std::abs(lhs - rhs) <
           std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
  }
};

template <typename type_t>
struct device_nearly_equal {
  const type_t threshold_ratio;
  const type_t threshold_magnitude;

  bool __device__ operator()(type_t lhs, type_t rhs) const
  {
    return std::abs(lhs - rhs) <
           thrust::max(thrust::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
  }
};

}  // namespace test
}  // namespace cugraph
