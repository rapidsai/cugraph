/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "utilities/csv_file_utilities.hpp"
#include "utilities/matrix_market_file_utilities.hpp"
#include "utilities/misc_utilities.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/random/rng_state.hpp>

#include <memory>
#include <numeric>

namespace cugraph {
namespace test {

namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>>
concatenate_edge_chunks(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<vertex_t>>&& src_chunks,
  std::vector<rmm::device_uvector<vertex_t>>&& dst_chunks,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& weight_chunks,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edge_id_chunks,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edge_type_chunks,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edge_start_time_chunks,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edge_end_time_chunks)
{
  if (src_chunks.size() == 1) {
    return std::make_tuple(
      std::move(src_chunks[0]),
      std::move(dst_chunks[0]),
      weight_chunks
        ? std::make_optional<rmm::device_uvector<weight_t>>(std::move((*weight_chunks)[0]))
        : std::nullopt,
      edge_id_chunks
        ? std::make_optional<rmm::device_uvector<edge_t>>(std::move((*edge_id_chunks)[0]))
        : std::nullopt,
      edge_type_chunks
        ? std::make_optional<rmm::device_uvector<edge_type_t>>(std::move((*edge_type_chunks)[0]))
        : std::nullopt,
      edge_start_time_chunks ? std::make_optional<rmm::device_uvector<edge_time_t>>(
                                 std::move((*edge_start_time_chunks)[0]))
                             : std::nullopt,
      edge_end_time_chunks ? std::make_optional<rmm::device_uvector<edge_time_t>>(
                               std::move((*edge_end_time_chunks)[0]))
                           : std::nullopt);
  } else {
    size_t edge_count{0};
    for (size_t i = 0; i < src_chunks.size(); ++i) {
      edge_count += src_chunks[i].size();
    }

    rmm::device_uvector<vertex_t> srcs(edge_count, handle.get_stream());
    {
      size_t offset{0};
      for (size_t i = 0; i < src_chunks.size(); ++i) {
        raft::copy(
          srcs.data() + offset, src_chunks[i].data(), src_chunks[i].size(), handle.get_stream());
        offset += src_chunks[i].size();
        src_chunks[i].resize(0, handle.get_stream());
        src_chunks[i].shrink_to_fit(handle.get_stream());
      }
      src_chunks.clear();
    }

    rmm::device_uvector<vertex_t> dsts(edge_count, handle.get_stream());
    {
      size_t offset{0};
      for (size_t i = 0; i < dst_chunks.size(); ++i) {
        raft::copy(
          dsts.data() + offset, dst_chunks[i].data(), dst_chunks[i].size(), handle.get_stream());
        offset += dst_chunks[i].size();
        dst_chunks[i].resize(0, handle.get_stream());
        dst_chunks[i].shrink_to_fit(handle.get_stream());
      }
      dst_chunks.clear();
    }

    auto weights = weight_chunks ? std::make_optional<rmm::device_uvector<weight_t>>(
                                     edge_count, handle.get_stream())
                                 : std::nullopt;
    if (weights) {
      size_t offset{0};
      for (size_t i = 0; i < (*weight_chunks).size(); ++i) {
        raft::copy((*weights).data() + offset,
                   (*weight_chunks)[i].data(),
                   (*weight_chunks)[i].size(),
                   handle.get_stream());
        offset += (*weight_chunks)[i].size();
        (*weight_chunks)[i].resize(0, handle.get_stream());
        (*weight_chunks)[i].shrink_to_fit(handle.get_stream());
      }
      (*weight_chunks).clear();
    }

    auto edge_ids = edge_id_chunks ? std::make_optional<rmm::device_uvector<edge_t>>(
                                       edge_count, handle.get_stream())
                                   : std::nullopt;
    if (edge_ids) {
      size_t offset{0};
      for (size_t i = 0; i < (*edge_id_chunks).size(); ++i) {
        raft::copy((*edge_ids).data() + offset,
                   (*edge_id_chunks)[i].data(),
                   (*edge_id_chunks)[i].size(),
                   handle.get_stream());
        offset += (*edge_id_chunks)[i].size();
        (*edge_id_chunks)[i].resize(0, handle.get_stream());
        (*edge_id_chunks)[i].shrink_to_fit(handle.get_stream());
      }
      (*edge_id_chunks).clear();
    }

    auto edge_types = edge_type_chunks ? std::make_optional<rmm::device_uvector<edge_type_t>>(
                                           edge_count, handle.get_stream())
                                       : std::nullopt;
    if (edge_types) {
      size_t offset{0};
      for (size_t i = 0; i < (*edge_type_chunks).size(); ++i) {
        raft::copy((*edge_types).data() + offset,
                   (*edge_type_chunks)[i].data(),
                   (*edge_type_chunks)[i].size(),
                   handle.get_stream());
        offset += (*edge_type_chunks)[i].size();
        (*edge_type_chunks)[i].resize(0, handle.get_stream());
        (*edge_type_chunks)[i].shrink_to_fit(handle.get_stream());
      }
      (*edge_type_chunks).clear();
    }

    auto edge_start_times =
      edge_start_time_chunks
        ? std::make_optional<rmm::device_uvector<edge_time_t>>(edge_count, handle.get_stream())
        : std::nullopt;
    if (edge_start_times) {
      size_t offset{0};
      for (size_t i = 0; i < (*edge_start_time_chunks).size(); ++i) {
        raft::copy((*edge_start_times).data() + offset,
                   (*edge_start_time_chunks)[i].data(),
                   (*edge_start_time_chunks)[i].size(),
                   handle.get_stream());
        offset += (*edge_start_time_chunks)[i].size();
        (*edge_start_time_chunks)[i].resize(0, handle.get_stream());
        (*edge_start_time_chunks)[i].shrink_to_fit(handle.get_stream());
      }
      (*edge_start_time_chunks).clear();
    }

    auto edge_end_times =
      edge_end_time_chunks
        ? std::make_optional<rmm::device_uvector<edge_time_t>>(edge_count, handle.get_stream())
        : std::nullopt;
    if (edge_end_times) {
      size_t offset{0};
      for (size_t i = 0; i < (*edge_end_time_chunks).size(); ++i) {
        raft::copy((*edge_end_times).data() + offset,
                   (*edge_end_time_chunks)[i].data(),
                   (*edge_end_time_chunks)[i].size(),
                   handle.get_stream());
        offset += (*edge_end_time_chunks)[i].size();
        (*edge_end_time_chunks)[i].resize(0, handle.get_stream());
        (*edge_end_time_chunks)[i].shrink_to_fit(handle.get_stream());
      }
      (*edge_end_time_chunks).clear();
    }

    return std::make_tuple(std::move(srcs),
                           std::move(dsts),
                           std::move(weights),
                           std::move(edge_ids),
                           std::move(edge_types),
                           std::move(edge_start_times),
                           std::move(edge_end_times));
  }
}

class TranslateGraph_Usecase {
 public:
  TranslateGraph_Usecase() = delete;
  TranslateGraph_Usecase(size_t base_vertex_id = 0) : base_vertex_id_(base_vertex_id) {}

  template <typename vertex_t>
  void translate(raft::handle_t const& handle, rmm::device_uvector<vertex_t>& vertices) const
  {
    if (base_vertex_id_ > 0) {
      cugraph::test::translate_vertex_ids(handle, vertices, static_cast<vertex_t>(base_vertex_id_));
    }
  }

  template <typename vertex_t>
  void translate(raft::handle_t const& handle,
                 rmm::device_uvector<vertex_t>& srcs,
                 rmm::device_uvector<vertex_t>& dsts) const
  {
    if (base_vertex_id_ > 0) {
      cugraph::test::translate_vertex_ids(handle, srcs, static_cast<vertex_t>(base_vertex_id_));
      cugraph::test::translate_vertex_ids(handle, dsts, static_cast<vertex_t>(base_vertex_id_));
    }
  }

  size_t base_vertex_id_{};
};

}  // namespace detail

class File_Usecase : public detail::TranslateGraph_Usecase {
 public:
  File_Usecase() = delete;

  File_Usecase(std::string const& graph_file_path, size_t base_vertex_id = 0)
    : detail::TranslateGraph_Usecase(base_vertex_id)
  {
    set_filename(graph_file_path);
  }

  void set_filename(std::string const& graph_file_path)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path_ = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path_ = graph_file_path;
    }
  }

  template <typename vertex_t, typename weight_t>
  std::tuple<std::vector<rmm::device_uvector<vertex_t>>,
             std::vector<rmm::device_uvector<vertex_t>>,
             std::optional<std::vector<rmm::device_uvector<weight_t>>>,
             std::optional<rmm::device_uvector<vertex_t>>,
             bool>
  construct_edgelist(raft::handle_t const& handle,
                     bool test_weighted,
                     bool store_transposed,
                     bool multi_gpu,
                     bool shuffle                                                = true,
                     std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
                     std::optional<large_buffer_type_t> large_edge_buffer_type = std::nullopt) const
  {
    CUGRAPH_EXPECTS(
      !large_vertex_buffer_type || cugraph::large_buffer_manager::memory_buffer_initialized(),
      "Invalid input argument: large memory buffer is not initialized.");
    CUGRAPH_EXPECTS(
      !large_edge_buffer_type || cugraph::large_buffer_manager::memory_buffer_initialized(),
      "Invalid input argument: large memory buffer is not initialized.");

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{};
    std::optional<rmm::device_uvector<vertex_t>> vertices{};
    bool is_symmetric{};
    auto extension = graph_file_full_path_.substr(graph_file_full_path_.find_last_of(".") + 1);
    if (extension == "mtx") {
      std::tie(srcs, dsts, weights, vertices, is_symmetric) =
        read_edgelist_from_matrix_market_file<vertex_t, weight_t>(handle,
                                                                  graph_file_full_path_,
                                                                  test_weighted,
                                                                  store_transposed,
                                                                  multi_gpu,
                                                                  shuffle,
                                                                  large_vertex_buffer_type,
                                                                  large_edge_buffer_type);
    } else if (extension == "csv") {
      std::tie(srcs, dsts, weights, is_symmetric) =
        read_edgelist_from_csv_file<vertex_t, weight_t>(handle,
                                                        graph_file_full_path_,
                                                        test_weighted,
                                                        store_transposed,
                                                        multi_gpu,
                                                        shuffle,
                                                        large_vertex_buffer_type,
                                                        large_edge_buffer_type);
    }

    translate(handle, srcs, dsts);
    if (vertices) { translate(handle, *vertices); }

    std::vector<rmm::device_uvector<vertex_t>> edge_src_chunks{};
    edge_src_chunks.push_back(std::move(srcs));
    std::vector<rmm::device_uvector<vertex_t>> edge_dst_chunks{};
    edge_dst_chunks.push_back(std::move(dsts));
    std::optional<std::vector<rmm::device_uvector<weight_t>>> edge_weight_chunks{std::nullopt};
    if (weights) {
      edge_weight_chunks = std::vector<rmm::device_uvector<weight_t>>{};
      (*edge_weight_chunks).push_back(std::move(*weights));
    }
    return std::make_tuple(std::move(edge_src_chunks),
                           std::move(edge_dst_chunks),
                           std::move(edge_weight_chunks),
                           std::move(vertices),
                           is_symmetric);
  }

 private:
  std::string graph_file_full_path_{};
};

class Rmat_Usecase : public detail::TranslateGraph_Usecase {
 public:
  Rmat_Usecase() = delete;

  Rmat_Usecase(size_t scale,
               size_t edge_factor,
               double a,
               double b,
               double c,
               uint64_t base_seed,
               bool undirected,
               bool scramble_vertex_ids,
               size_t base_vertex_id = 0)
    : detail::TranslateGraph_Usecase(base_vertex_id),
      scale_(scale),
      edge_factor_(edge_factor),
      a_(a),
      b_(b),
      c_(c),
      base_seed_(base_seed),
      undirected_(undirected),
      scramble_vertex_ids_(scramble_vertex_ids)
  {
  }

  template <typename vertex_t, typename weight_t>
  std::tuple<std::vector<rmm::device_uvector<vertex_t>>,
             std::vector<rmm::device_uvector<vertex_t>>,
             std::optional<std::vector<rmm::device_uvector<weight_t>>>,
             std::optional<rmm::device_uvector<vertex_t>>,
             bool>
  construct_edgelist(raft::handle_t const& handle,
                     bool test_weighted,
                     bool store_transposed,
                     bool multi_gpu,
                     bool shuffle                                                = true,
                     std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
                     std::optional<large_buffer_type_t> large_edge_buffer_type = std::nullopt) const
  {
    CUGRAPH_EXPECTS(
      (size_t{1} << scale_) <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
      "Invalid template parameter: scale_ too large for vertex_t.");
    CUGRAPH_EXPECTS(
      !large_vertex_buffer_type || cugraph::large_buffer_manager::memory_buffer_initialized(),
      "Invalid input argument: large memory buffer is not initialized.");
    CUGRAPH_EXPECTS(
      !large_edge_buffer_type || cugraph::large_buffer_manager::memory_buffer_initialized(),
      "Invalid input argument: large memory buffer is not initialized.");

    // Generate in multi-partitions to limit peak memory usage (thrust::sort &
    // shuffle_edges requires a temporary buffer with the size of the original data). With the
    // current implementation, the temporary memory requirement is roughly 50% of the original data
    // with num_partitions_per_gpu = 2. If we use cuMemAddressReserve
    // (https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management), we
    // can reduce the temporary memory requirement to (1 / num_partitions) * (original data size)
    size_t constexpr num_partitions_per_gpu = 8;
    size_t num_partitions =
      num_partitions_per_gpu * static_cast<size_t>(multi_gpu ? handle.get_comms().get_size() : 1);

    // 1. calculate # edges to generate in each partition, and partition vertex ranges

    vertex_t number_of_vertices = static_cast<vertex_t>(size_t{1} << scale_);
    size_t number_of_edges =
      static_cast<size_t>(static_cast<size_t>(number_of_vertices) * edge_factor_);

    std::array<size_t, num_partitions_per_gpu> partition_edge_counts{};
    std::array<vertex_t, num_partitions_per_gpu> partition_vertex_firsts{};
    std::array<vertex_t, num_partitions_per_gpu> partition_vertex_lasts{};

    for (size_t i = 0; i < num_partitions_per_gpu; ++i) {
      auto id =
        (multi_gpu ? num_partitions_per_gpu * static_cast<size_t>(handle.get_comms().get_rank())
                   : size_t{0}) +
        i;

      partition_edge_counts[i] = number_of_edges / num_partitions +
                                 (id < number_of_edges % num_partitions ? size_t{1} : size_t{0});

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

    // 2. generate edges

    raft::random::RngState rng_state{
      base_seed_ + static_cast<uint64_t>(multi_gpu ? handle.get_comms().get_rank() : 0)};

    std::vector<rmm::device_uvector<vertex_t>> edge_src_chunks{};
    std::vector<rmm::device_uvector<vertex_t>> edge_dst_chunks{};
    auto edge_weight_chunks = test_weighted
                                ? std::make_optional<std::vector<rmm::device_uvector<weight_t>>>()
                                : std::nullopt;
    edge_src_chunks.reserve(num_partitions_per_gpu);
    edge_dst_chunks.reserve(num_partitions_per_gpu);
    if (edge_weight_chunks) { (*edge_weight_chunks).reserve(num_partitions_per_gpu); }
    for (size_t i = 0; i < num_partitions_per_gpu; ++i) {
      auto [tmp_src_v, tmp_dst_v] =
        cugraph::generate_rmat_edgelist<vertex_t>(handle,
                                                  rng_state,
                                                  scale_,
                                                  partition_edge_counts[i],
                                                  a_,
                                                  b_,
                                                  c_,
                                                  undirected_ ? true : false,
                                                  scramble_vertex_ids_,
                                                  large_edge_buffer_type);

      std::optional<rmm::device_uvector<weight_t>> tmp_weights_v{std::nullopt};
      if (edge_weight_chunks) {
        tmp_weights_v = large_edge_buffer_type
                          ? large_buffer_manager::allocate_memory_buffer<weight_t>(
                              tmp_src_v.size(), handle.get_stream())
                          : rmm::device_uvector<weight_t>(tmp_src_v.size(), handle.get_stream());

        cugraph::detail::uniform_random_fill(handle.get_stream(),
                                             tmp_weights_v->data(),
                                             tmp_weights_v->size(),
                                             weight_t{0.0},
                                             weight_t{1.0},
                                             rng_state);
      }

      translate(handle, tmp_src_v, tmp_dst_v);

      if (undirected_) {
        std::tie(tmp_src_v, tmp_dst_v, tmp_weights_v) =
          cugraph::symmetrize_edgelist_from_triangular<vertex_t, weight_t>(handle,
                                                                           std::move(tmp_src_v),
                                                                           std::move(tmp_dst_v),
                                                                           std::move(tmp_weights_v),
                                                                           false,
                                                                           large_edge_buffer_type);
      }

      if (multi_gpu && shuffle) {
        std::vector<cugraph::arithmetic_device_uvector_t> tmp_edge_properties{};
        if (tmp_weights_v) tmp_edge_properties.push_back(std::move(*tmp_weights_v));

        std::tie(tmp_src_v, tmp_dst_v, tmp_edge_properties, std::ignore) =
          cugraph::shuffle_ext_edges(handle,
                                     std::move(tmp_src_v),
                                     std::move(tmp_dst_v),
                                     std::move(tmp_edge_properties),
                                     store_transposed,
                                     large_edge_buffer_type);

        if (tmp_weights_v)
          *tmp_weights_v =
            std::move(std::get<rmm::device_uvector<weight_t>>(tmp_edge_properties[0]));
      }

      edge_src_chunks.push_back(std::move(tmp_src_v));
      edge_dst_chunks.push_back(std::move(tmp_dst_v));
      if (edge_weight_chunks) { (*edge_weight_chunks).push_back(std::move(*tmp_weights_v)); }
    }

    // 3. generate vertices

    size_t tot_vertex_counts{0};
    for (size_t i = 0; i < partition_vertex_firsts.size(); ++i) {
      tot_vertex_counts += partition_vertex_lasts[i] - partition_vertex_firsts[i];
    }
    auto vertex_v = large_vertex_buffer_type
                      ? large_buffer_manager::allocate_memory_buffer<vertex_t>(tot_vertex_counts,
                                                                               handle.get_stream())
                      : rmm::device_uvector<vertex_t>(tot_vertex_counts, handle.get_stream());
    size_t v_offset{0};
    for (size_t i = 0; i < partition_vertex_firsts.size(); ++i) {
      cugraph::detail::sequence_fill(handle.get_stream(),
                                     vertex_v.begin() + v_offset,
                                     partition_vertex_lasts[i] - partition_vertex_firsts[i],
                                     partition_vertex_firsts[i]);
      v_offset += partition_vertex_lasts[i] - partition_vertex_firsts[i];
    }
    if (scramble_vertex_ids_) {
      vertex_v = cugraph::scramble_vertex_ids(handle, std::move(vertex_v), scale_);
    }

    translate(handle, vertex_v);

    if (multi_gpu && shuffle) {
      std::tie(vertex_v, std::ignore) =
        cugraph::shuffle_ext_vertices(handle,
                                      std::move(vertex_v),
                                      std::vector<cugraph::arithmetic_device_uvector_t>{},
                                      large_vertex_buffer_type);
    }

    return std::make_tuple(std::move(edge_src_chunks),
                           std::move(edge_dst_chunks),
                           std::move(edge_weight_chunks),
                           std::move(vertex_v),
                           undirected_);
  }

  void set_scale(size_t scale) { scale_ = scale; }

  void set_edge_factor(size_t edge_factor) { edge_factor_ = edge_factor; }

  bool undirected() const { return undirected_; }

 private:
  size_t scale_{};
  size_t edge_factor_{};
  double a_{};
  double b_{};
  double c_{};
  uint64_t base_seed_{};
  bool undirected_{};
  bool scramble_vertex_ids_{};
  bool multi_gpu_usecase_{};
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu,
          typename input_usecase_t>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<cugraph::edge_property_t<edge_t, weight_t>>,
           std::optional<cugraph::edge_property_t<edge_t, edge_t>>,
           std::optional<cugraph::edge_property_t<edge_t, edge_type_t>>,
           std::optional<cugraph::edge_property_t<edge_t, edge_time_t>>,
           std::optional<cugraph::edge_property_t<edge_t, edge_time_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
construct_graph(
  raft::handle_t const& handle,
  input_usecase_t const& input_usecase,
  bool test_weighted,
  std::optional<std::function<rmm::device_uvector<edge_t>(raft::handle_t const&, size_t, size_t)>>
    edge_ids_functor,
  std::optional<
    std::function<rmm::device_uvector<edge_type_t>(raft::handle_t const&, size_t, size_t)>>
    edge_types_functor,
  std::optional<
    std::function<rmm::device_uvector<edge_time_t>(raft::handle_t const&, size_t, size_t)>>
    edge_start_times_functor,
  std::optional<
    std::function<rmm::device_uvector<edge_time_t>(raft::handle_t const&, size_t, size_t)>>
    edge_end_times_functor,
  bool renumber                                               = true,
  bool drop_self_loops                                        = false,
  bool drop_multi_edges                                       = false,
  std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
  std::optional<large_buffer_type_t> large_edge_buffer_type   = std::nullopt)
{
  auto [edge_src_chunks, edge_dst_chunks, edge_weight_chunks, d_vertices_v, is_symmetric] =
    input_usecase.template construct_edgelist<vertex_t, weight_t>(handle,
                                                                  test_weighted,
                                                                  store_transposed,
                                                                  multi_gpu,
                                                                  true /* shuffle */,
                                                                  large_vertex_buffer_type,
                                                                  large_edge_buffer_type);

  size_t num_edges{0};
  for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
    num_edges += edge_src_chunks[i].size();
  }
  CUGRAPH_EXPECTS(num_edges <= static_cast<size_t>(std::numeric_limits<edge_t>::max()),
                  "Invalid template parameter: edge_t overflow.");

  size_t base_offset{0};
  if constexpr (multi_gpu) {
    auto base_offsets = host_scalar_allgather(handle.get_comms(), num_edges, handle.get_stream());
    handle.sync_stream();
    std::exclusive_scan(base_offsets.begin(), base_offsets.end(), base_offsets.begin(), size_t{0});
    base_offset = base_offsets[handle.get_comms().get_rank()];
  }

  std::optional<std::vector<rmm::device_uvector<edge_t>>> edge_id_chunks{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_type_chunks{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edge_start_time_chunks{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edge_end_time_chunks{std::nullopt};

  if (edge_ids_functor) edge_id_chunks = std::vector<rmm::device_uvector<edge_t>>{};
  if (edge_types_functor) edge_type_chunks = std::vector<rmm::device_uvector<edge_type_t>>{};
  if (edge_start_times_functor)
    edge_start_time_chunks = std::vector<rmm::device_uvector<edge_time_t>>{};
  if (edge_end_times_functor)
    edge_end_time_chunks = std::vector<rmm::device_uvector<edge_time_t>>{};

  for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
    if (edge_ids_functor) {
      edge_id_chunks->push_back(
        (*edge_ids_functor)(handle, edge_src_chunks[i].size(), base_offset));
    }
    if (edge_types_functor) {
      edge_type_chunks->push_back(
        (*edge_types_functor)(handle, edge_src_chunks[i].size(), base_offset));
    }
    if (edge_start_times_functor) {
      edge_start_time_chunks->push_back(
        (*edge_start_times_functor)(handle, edge_src_chunks[i].size(), base_offset));
    }
    if (edge_end_times_functor) {
      edge_end_time_chunks->push_back(
        (*edge_end_times_functor)(handle, edge_src_chunks[i].size(), base_offset));
    }
  }

  if (drop_self_loops) {
    for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
      std::optional<rmm::device_uvector<weight_t>> tmp_weights{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> tmp_ids{std::nullopt};
      std::optional<rmm::device_uvector<edge_type_t>> tmp_types{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> tmp_start_times{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> tmp_end_times{std::nullopt};
      std::tie(edge_src_chunks[i],
               edge_dst_chunks[i],
               tmp_weights,
               tmp_ids,
               tmp_types,
               tmp_start_times,
               tmp_end_times) =
        cugraph::remove_self_loops<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          handle,
          std::move(edge_src_chunks[i]),
          std::move(edge_dst_chunks[i]),
          edge_weight_chunks
            ? std::make_optional<rmm::device_uvector<weight_t>>(std::move((*edge_weight_chunks)[i]))
            : std::nullopt,
          edge_id_chunks ? std::make_optional(std::move((*edge_id_chunks)[i])) : std::nullopt,
          edge_type_chunks ? std::make_optional(std::move((*edge_type_chunks)[i])) : std::nullopt,
          edge_start_time_chunks ? std::make_optional(std::move((*edge_start_time_chunks)[i]))
                                 : std::nullopt,
          edge_end_time_chunks ? std::make_optional(std::move((*edge_end_time_chunks)[i]))
                               : std::nullopt,
          large_edge_buffer_type);
      if (tmp_weights) { (*edge_weight_chunks)[i] = std::move(*tmp_weights); }
      if (tmp_ids) { (*edge_id_chunks)[i] = std::move(*tmp_ids); }
      if (tmp_types) { (*edge_type_chunks)[i] = std::move(*tmp_types); }
      if (tmp_start_times) { (*edge_start_time_chunks)[i] = std::move(*tmp_start_times); }
      if (tmp_end_times) { (*edge_end_time_chunks)[i] = std::move(*tmp_end_times); }
    }
  }

  if (drop_multi_edges) {
    auto [srcs, dsts, weights, edge_ids, edge_types, edge_start_times, edge_end_times] =
      detail::concatenate_edge_chunks(handle,
                                      std::move(edge_src_chunks),
                                      std::move(edge_dst_chunks),
                                      std::move(edge_weight_chunks),
                                      std::move(edge_id_chunks),
                                      std::move(edge_type_chunks),
                                      std::move(edge_start_time_chunks),
                                      std::move(edge_end_time_chunks));
    std::tie(srcs, dsts, weights, edge_ids, edge_types, edge_start_times, edge_end_times) =
      cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, int32_t>(
        handle,
        std::move(srcs),
        std::move(dsts),
        std::move(weights),
        std::move(edge_ids),
        std::move(edge_types),
        std::move(edge_start_times),
        std::move(edge_end_times),
        is_symmetric ? true /* keep minimum weight edges to maintain symmetry */ : false,
        large_edge_buffer_type);
    edge_src_chunks = std::vector<rmm::device_uvector<vertex_t>>{};
    edge_src_chunks.push_back(std::move(srcs));
    edge_dst_chunks = std::vector<rmm::device_uvector<vertex_t>>{};
    edge_dst_chunks.push_back(std::move(dsts));
    edge_weight_chunks = std::nullopt;
    if (weights) {
      edge_weight_chunks = std::vector<rmm::device_uvector<weight_t>>{};
      (*edge_weight_chunks).push_back(std::move(*weights));
    }
    if (edge_ids) {
      edge_id_chunks = std::vector<rmm::device_uvector<edge_t>>{};
      (*edge_id_chunks).push_back(std::move(*edge_ids));
    }
    if (edge_types) {
      edge_type_chunks = std::vector<rmm::device_uvector<edge_type_t>>{};
      (*edge_type_chunks).push_back(std::move(*edge_types));
    }
    if (edge_start_times) {
      edge_start_time_chunks = std::vector<rmm::device_uvector<edge_time_t>>{};
      (*edge_start_time_chunks).push_back(std::move(*edge_start_times));
    }
    if (edge_end_times) {
      edge_end_time_chunks = std::vector<rmm::device_uvector<edge_time_t>>{};
      (*edge_end_time_chunks).push_back(std::move(*edge_end_times));
    }
  }

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);
  std::optional<edge_property_t<edge_t, weight_t>> edge_weights{std::nullopt};
  std::optional<edge_property_t<edge_t, edge_t>> edge_ids{std::nullopt};
  std::optional<edge_property_t<edge_t, edge_type_t>> edge_types{std::nullopt};
  std::optional<edge_property_t<edge_t, edge_time_t>> edge_start_times{std::nullopt};
  std::optional<edge_property_t<edge_t, edge_time_t>> edge_end_times{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  if (edge_src_chunks.size() == 1) {
    std::tie(
      graph, edge_weights, edge_ids, edge_types, edge_start_times, edge_end_times, renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t,
                                          edge_t,
                                          weight_t,
                                          edge_type_t,
                                          edge_time_t,
                                          store_transposed,
                                          multi_gpu>(
        handle,
        std::move(d_vertices_v),
        std::move(edge_src_chunks[0]),
        std::move(edge_dst_chunks[0]),
        edge_weight_chunks ? std::make_optional(std::move((*edge_weight_chunks)[0])) : std::nullopt,
        edge_id_chunks ? std::make_optional(std::move((*edge_id_chunks)[0])) : std::nullopt,
        edge_type_chunks ? std::make_optional(std::move((*edge_type_chunks)[0])) : std::nullopt,
        edge_start_time_chunks ? std::make_optional(std::move((*edge_start_time_chunks)[0]))
                               : std::nullopt,
        edge_end_time_chunks ? std::make_optional(std::move((*edge_end_time_chunks)[0]))
                             : std::nullopt,
        cugraph::graph_properties_t{is_symmetric, drop_multi_edges ? false : true},
        renumber,
        large_vertex_buffer_type,
        large_edge_buffer_type);
  } else {
    std::tie(
      graph, edge_weights, edge_ids, edge_types, edge_start_times, edge_end_times, renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t,
                                          edge_t,
                                          weight_t,
                                          edge_type_t,
                                          edge_time_t,
                                          store_transposed,
                                          multi_gpu>(
        handle,
        std::move(d_vertices_v),
        std::move(edge_src_chunks),
        std::move(edge_dst_chunks),
        std::move(edge_weight_chunks),
        std::move(edge_id_chunks),
        std::move(edge_type_chunks),
        std::move(edge_start_time_chunks),
        std::move(edge_end_time_chunks),
        cugraph::graph_properties_t{is_symmetric, drop_multi_edges ? false : true},
        renumber,
        large_vertex_buffer_type,
        large_edge_buffer_type);
  }

  return std::make_tuple(std::move(graph),
                         std::move(edge_weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(edge_start_times),
                         std::move(edge_end_times),
                         std::move(renumber_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu,
          typename input_usecase_t>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<cugraph::edge_property_t<edge_t, weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
construct_graph(raft::handle_t const& handle,
                input_usecase_t const& input_usecase,
                bool test_weighted,
                bool renumber                                               = true,
                bool drop_self_loops                                        = false,
                bool drop_multi_edges                                       = false,
                std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
                std::optional<large_buffer_type_t> large_edge_buffer_type   = std::nullopt)
{
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);
  std::optional<cugraph::edge_property_t<edge_t, weight_t>> edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, edge_weights, std::ignore, std::ignore, std::ignore, std::ignore, renumber_map) =
    construct_graph<vertex_t,
                    edge_t,
                    weight_t,
                    int32_t,
                    int32_t,
                    store_transposed,
                    multi_gpu,
                    input_usecase_t>(handle,
                                     input_usecase,
                                     test_weighted,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     renumber,
                                     drop_self_loops,
                                     drop_multi_edges,
                                     large_vertex_buffer_type,
                                     large_edge_buffer_type);

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

}  // namespace test
}  // namespace cugraph
