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
#pragma once

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>

#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

namespace cugraph {
namespace test {

namespace detail {

class TranslateGraph_Usecase {
 public:
  TranslateGraph_Usecase() = delete;
  TranslateGraph_Usecase(size_t base_vertex_id = 0) : base_vertex_id_(base_vertex_id) {}

  template <typename vertex_t>
  void translate(raft::handle_t const& handle,
                 rmm::device_uvector<vertex_t>& d_src,
                 rmm::device_uvector<vertex_t>& d_dst) const
  {
    if (base_vertex_id_ > 0)
      cugraph::test::translate_vertex_ids(
        handle, d_src, d_dst, static_cast<vertex_t>(base_vertex_id_));
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
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path_ = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path_ = graph_file_path;
    }
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>,
             rmm::device_uvector<vertex_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    auto [d_src_v, d_dst_v, d_weights_v, d_vertices_v, num_vertices, is_symmetric] =
      read_edgelist_from_matrix_market_file<vertex_t, weight_t, store_transposed, multi_gpu>(
        handle, graph_file_full_path_, test_weighted);

    translate(handle, d_src_v, d_dst_v);

    return std::make_tuple(
      std::move(d_src_v),
      std::move(d_dst_v),
      std::move(d_weights_v),
      std::move(d_vertices_v),
      static_cast<vertex_t>(detail::TranslateGraph_Usecase::base_vertex_id_) + num_vertices,
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
               uint64_t seed,
               bool undirected,
               bool scramble_vertex_ids,
               size_t base_vertex_id  = 0,
               bool multi_gpu_usecase = false)
    : detail::TranslateGraph_Usecase(base_vertex_id),
      scale_(scale),
      edge_factor_(edge_factor),
      a_(a),
      b_(b),
      c_(c),
      seed_(seed),
      undirected_(undirected),
      scramble_vertex_ids_(scramble_vertex_ids),
      multi_gpu_usecase_(multi_gpu_usecase)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>,
             rmm::device_uvector<vertex_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    CUGRAPH_EXPECTS(
      (size_t{1} << scale_) <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
      "Invalid template parameter: scale_ too large for vertex_t.");
    CUGRAPH_EXPECTS(((size_t{1} << scale_) * edge_factor_) <=
                      static_cast<size_t>(std::numeric_limits<edge_t>::max()),
                    "Invalid template parameter: (scale_, edge_factor_) too large for edge_t");

    std::vector<size_t> partition_ids(1);
    size_t num_partitions;

    if (multi_gpu_usecase_) {
      auto& comm           = handle.get_comms();
      num_partitions       = comm.get_size();
      auto const comm_rank = comm.get_rank();

      partition_ids.resize(multi_gpu ? size_t{1} : static_cast<size_t>(num_partitions));

      std::iota(partition_ids.begin(),
                partition_ids.end(),
                multi_gpu ? static_cast<size_t>(comm_rank) : size_t{0});
    } else {
      num_partitions   = 1;
      partition_ids[0] = size_t{0};
    }

    vertex_t number_of_vertices = static_cast<vertex_t>(size_t{1} << scale_);
    edge_t number_of_edges =
      static_cast<edge_t>(static_cast<size_t>(number_of_vertices) * edge_factor_);

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

    rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());
    auto weights_v = test_weighted
                       ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
                       : std::nullopt;
    for (size_t i = 0; i < partition_ids.size(); ++i) {
      auto id = partition_ids[i];

      rmm::device_uvector<vertex_t> tmp_src_v(0, handle.get_stream());
      rmm::device_uvector<vertex_t> tmp_dst_v(0, handle.get_stream());
      std::tie(i == 0 ? src_v : tmp_src_v, i == 0 ? dst_v : tmp_dst_v) =
        cugraph::generate_rmat_edgelist<vertex_t>(handle,
                                                  scale_,
                                                  partition_edge_counts[i],
                                                  a_,
                                                  b_,
                                                  c_,
                                                  seed_ + id,
                                                  undirected_ ? true : false);

      std::optional<rmm::device_uvector<weight_t>> tmp_weights_v{std::nullopt};
      if (weights_v) {
        if (i == 0) {
          weights_v->resize(src_v.size(), handle.get_stream());
        } else {
          tmp_weights_v = std::make_optional<rmm::device_uvector<weight_t>>(tmp_src_v.size(),
                                                                            handle.get_stream());
        }

        cugraph::detail::uniform_random_fill(handle.get_stream_view(),
                                             i == 0 ? weights_v->data() : tmp_weights_v->data(),
                                             i == 0 ? weights_v->size() : tmp_weights_v->size(),
                                             weight_t{0.0},
                                             weight_t{1.0},
                                             seed_ + num_partitions + id);
      }

      if (i > 0) {
        auto start_offset = src_v.size();
        src_v.resize(start_offset + tmp_src_v.size(), handle.get_stream());
        dst_v.resize(start_offset + tmp_dst_v.size(), handle.get_stream());
        raft::copy(
          src_v.begin() + start_offset, tmp_src_v.begin(), tmp_src_v.size(), handle.get_stream());
        raft::copy(
          dst_v.begin() + start_offset, tmp_dst_v.begin(), tmp_dst_v.size(), handle.get_stream());

        if (weights_v) {
          weights_v->resize(start_offset + tmp_weights_v->size(), handle.get_stream());
          raft::copy(weights_v->begin() + start_offset,
                     tmp_weights_v->begin(),
                     tmp_weights_v->size(),
                     handle.get_stream());
        }
      }
    }

    translate(handle, src_v, dst_v);

    if (undirected_)
      std::tie(src_v, dst_v, weights_v) =
        cugraph::symmetrize_edgelist_from_triangular<vertex_t, weight_t>(
          handle, std::move(src_v), std::move(dst_v), std::move(weights_v));

    if (multi_gpu) {
      std::tie(store_transposed ? dst_v : src_v, store_transposed ? src_v : dst_v, weights_v) =
        cugraph::detail::shuffle_edgelist_by_gpu_id(
          handle,
          store_transposed ? std::move(dst_v) : std::move(src_v),
          store_transposed ? std::move(src_v) : std::move(dst_v),
          std::move(weights_v));
    }

    rmm::device_uvector<vertex_t> vertices_v(0, handle.get_stream());
    for (size_t i = 0; i < partition_ids.size(); ++i) {
      auto id = partition_ids[i];

      auto start_offset = vertices_v.size();
      vertices_v.resize(start_offset + (partition_vertex_lasts[i] - partition_vertex_firsts[i]),
                        handle.get_stream());
      cugraph::detail::sequence_fill(handle.get_stream_view(),
                                     vertices_v.begin() + start_offset,
                                     vertices_v.size() - start_offset,
                                     partition_vertex_firsts[i]);
    }

    if constexpr (multi_gpu) {
      vertices_v = cugraph::detail::shuffle_vertices_by_gpu_id(handle, std::move(vertices_v));
    }

    return std::make_tuple(
      std::move(src_v),
      std::move(dst_v),
      std::move(weights_v),
      std::move(vertices_v),
      static_cast<vertex_t>(detail::TranslateGraph_Usecase::base_vertex_id_) + number_of_vertices,
      undirected_);
  }

  void set_scale(size_t scale) { scale_ = scale; }

  void set_edge_factor(size_t edge_factor) { edge_factor_ = edge_factor; }

 private:
  size_t scale_{};
  size_t edge_factor_{};
  double a_{};
  double b_{};
  double c_{};
  uint64_t seed_{};
  bool undirected_{};
  bool scramble_vertex_ids_{};
  bool multi_gpu_usecase_{};
};

class PathGraph_Usecase {
 public:
  PathGraph_Usecase() = delete;

  PathGraph_Usecase(std::vector<std::tuple<size_t, size_t>> parms,
                    bool weighted = false,
                    bool scramble = false)
    : parms_(parms), weighted_(weighted)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>,
             rmm::device_uvector<vertex_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    constexpr bool symmetric{true};

    std::vector<std::tuple<vertex_t, vertex_t>> converted_parms(parms_.size());

    std::transform(parms_.begin(), parms_.end(), converted_parms.begin(), [](auto p) {
      return std::make_tuple(static_cast<vertex_t>(std::get<0>(p)),
                             static_cast<vertex_t>(std::get<1>(p)));
    });

    auto [src_v, dst_v] = cugraph::generate_path_graph_edgelist<vertex_t>(handle, converted_parms);
    std::tie(src_v, dst_v, std::ignore) =
      cugraph::symmetrize_edgelist_from_triangular<vertex_t, weight_t>(
        handle, std::move(src_v), std::move(dst_v), std::nullopt);

    rmm::device_uvector<vertex_t> d_vertices(num_vertices_, handle.get_stream());
    cugraph::detail::sequence_fill(
      handle.get_stream(), d_vertices.data(), num_vertices_, vertex_t{0});
    handle.get_stream_view().synchronize();

    return std::make_tuple(std::move(src_v),
                           std::move(dst_v),
                           test_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                             src_v.size(), handle.get_stream())
                                         : std::nullopt,
                           std::move(d_vertices),
                           num_vertices_,
                           symmetric);
  }

 private:
  std::vector<std::tuple<size_t, size_t>> parms_{};
  size_t num_vertices_{0};
  bool weighted_{false};
};

class Mesh2DGraph_Usecase {
 public:
  Mesh2DGraph_Usecase() = delete;

  Mesh2DGraph_Usecase(std::vector<std::tuple<size_t, size_t, size_t>> const& parms, bool weighted)
    : parms_(parms), weighted_(weighted)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>,
             rmm::device_uvector<vertex_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
  }

 private:
  std::vector<std::tuple<size_t, size_t, size_t>> parms_{};
  bool weighted_{false};
};

class Mesh3DGraph_Usecase {
 public:
  Mesh3DGraph_Usecase() = delete;

  Mesh3DGraph_Usecase(std::vector<std::tuple<size_t, size_t, size_t, size_t>> const& parms,
                      bool weighted)
    : parms_(parms), weighted_(weighted)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>,
             rmm::device_uvector<vertex_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const;

 private:
  std::vector<std::tuple<size_t, size_t, size_t, size_t>> parms_{};
  bool weighted_{false};
};

class CompleteGraph_Usecase {
 public:
  CompleteGraph_Usecase() = delete;

  CompleteGraph_Usecase(std::vector<std::tuple<size_t, size_t>> const& parms, bool weighted)
    : parms_(parms), weighted_(weighted)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>,
             rmm::device_uvector<vertex_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const;

 private:
  std::vector<std::tuple<size_t, size_t>> parms_{};
  bool weighted_{false};
};

namespace detail {

template <typename generator_tuple_t, size_t I, size_t N>
struct combined_construct_graph_tuple_impl {
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::vector<std::tuple<rmm::device_uvector<vertex_t>,
                         rmm::device_uvector<vertex_t>,
                         rmm::device_uvector<weight_t>,
                         vertex_t,
                         bool>>
  construct_edges(raft::handle_t const& handle,
                  bool test_weighted,
                  generator_tuple_t const& generator_tuple) const
  {
    return combined_construct_graph_tuple_impl<generator_tuple_t, I + 1, N>()
      .construct_edges(generator_tuple)
      .push_back(std::get<I>(generator_tuple).construct_edges(handle, test_weighted));
  }
};

template <typename generator_tuple_t, size_t I>
struct combined_construct_graph_tuple_impl<generator_tuple_t, I, I> {
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::vector<std::tuple<rmm::device_uvector<vertex_t>,
                         rmm::device_uvector<vertex_t>,
                         rmm::device_uvector<weight_t>,
                         vertex_t,
                         bool>>
  construct_edges(raft::handle_t const& handle,
                  bool test_weighted,
                  generator_tuple_t const& generator_tuple) const
  {
    return std::vector<std::tuple<rmm::device_uvector<vertex_t>,
                                  rmm::device_uvector<vertex_t>,
                                  rmm::device_uvector<weight_t>,
                                  vertex_t,
                                  bool>>();
  }
};

}  // namespace detail

template <typename generator_tuple_t>
class CombinedGenerator_Usecase {
  CombinedGenerator_Usecase() = delete;

  CombinedGenerator_Usecase(generator_tuple_t const& tuple) : generator_tuple_(tuple) {}

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>,
             rmm::device_uvector<vertex_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    size_t constexpr tuple_size{std::tuple_size<generator_tuple_t>::value};

    auto edge_tuple_vector =
      detail::combined_construct_graph_tuple_impl<generator_tuple_t, 0, tuple_size>()
        .construct_edges(handle, test_weighted, generator_tuple_);

    // Need to combine elements.  We have a vector of tuples, we want to combine
    //  the elements of each component of the tuple
    CUGRAPH_FAIL("not implemented");
  }

 private:
  generator_tuple_t const& generator_tuple_;
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu,
          typename input_usecase_t>
std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
construct_graph(raft::handle_t const& handle,
                input_usecase_t const& input_usecase,
                bool test_weighted,
                bool renumber         = true,
                bool drop_self_loops  = false,
                bool drop_multi_edges = false)
{
  auto [d_src_v, d_dst_v, d_weights_v, d_vertices_v, num_vertices, is_symmetric] =
    input_usecase
      .template construct_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
        handle, test_weighted);

  if (drop_self_loops) { remove_self_loops(handle, d_src_v, d_dst_v, d_weights_v); }

  if (drop_multi_edges) { sort_and_remove_multi_edges(handle, d_src_v, d_dst_v, d_weights_v); }

  return cugraph::
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::make_optional(std::move(d_vertices_v)),
      std::move(d_src_v),
      std::move(d_dst_v),
      std::move(d_weights_v),
      cugraph::graph_properties_t{is_symmetric, true},
      renumber);
}

}  // namespace test
}  // namespace cugraph
