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

#include <cugraph/experimental/graph_functions.hpp>
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
             rmm::device_uvector<weight_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    rmm::device_uvector<vertex_t> d_src_v(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dst_v(0, handle.get_stream());
    rmm::device_uvector<weight_t> d_weights_v(0, handle.get_stream());
    vertex_t num_vertices;
    bool is_symmetric;

    std::tie(d_src_v, d_dst_v, d_weights_v, num_vertices, is_symmetric) =
      read_edgelist_from_matrix_market_file<vertex_t, weight_t>(
        handle, graph_file_full_path_, test_weighted);

    translate(handle, d_src_v, d_dst_v);

#if 0
    if (multi_gpu) {
      std::tie(d_src_v, d_dst_v) = filter_edgelist_by_gpu(handle, d_src_v, d_dst_v);
    }
#endif

    return std::make_tuple(
      std::move(d_src_v),
      std::move(d_dst_v),
      std::move(d_weights_v),
      static_cast<vertex_t>(detail::TranslateGraph_Usecase::base_vertex_id_) + num_vertices,
      is_symmetric);
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const
  {
    rmm::device_uvector<vertex_t> d_src_v(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dst_v(0, handle.get_stream());
    rmm::device_uvector<weight_t> d_weights_v(0, handle.get_stream());
    vertex_t num_vertices;
    bool is_symmetric;

    std::tie(d_src_v, d_dst_v, d_weights_v, num_vertices, is_symmetric) =
      this->template construct_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
        handle, test_weighted);

    // TODO:  Consider calling construct_edgelist and creating
    //        a generic test function to take the edgelist and
    //        do the graph construction.
    //
    //        Would be more reusable across tests
    //
    return read_graph_from_matrix_market_file<vertex_t,
                                              edge_t,
                                              weight_t,
                                              store_transposed,
                                              multi_gpu>(
      handle, graph_file_full_path_, test_weighted, renumber);
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
             rmm::device_uvector<weight_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    // TODO: Tease through generate_graph_from_rmat_params
    //       to extract the edgelist part
    // Call cugraph::translate_vertex_ids(handle, d_src_v, d_dst_v, base_vertex_id_);

    CUGRAPH_FAIL("Not implemented");
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const
  {
    std::vector<size_t> partition_ids(1);
    size_t comm_size;

    if (multi_gpu_usecase_) {
      auto& comm           = handle.get_comms();
      comm_size            = comm.get_size();
      auto const comm_rank = comm.get_rank();

      partition_ids.resize(multi_gpu ? size_t{1} : static_cast<size_t>(comm_size));

      std::iota(partition_ids.begin(),
                partition_ids.end(),
                multi_gpu ? static_cast<size_t>(comm_rank) : size_t{0});
    } else {
      comm_size        = 1;
      partition_ids[0] = size_t{0};
    }

    // TODO: Need to offset by base_vertex_id_
    // static_cast<vertex_t>(base_vertex_id_));
    //    Consider using construct_edgelist like other options
    return generate_graph_from_rmat_params<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      scale_,
      edge_factor_,
      a_,
      b_,
      c_,
      seed_,
      undirected_,
      scramble_vertex_ids_,
      test_weighted,
      renumber,
      partition_ids,
      comm_size);
  }

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
             rmm::device_uvector<weight_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    rmm::device_uvector<weight_t> weights_v(0, handle.get_stream());

    constexpr bool symmetric{true};

    std::vector<std::tuple<vertex_t, vertex_t>> converted_parms(parms_.size());

    std::transform(parms_.begin(), parms_.end(), converted_parms.begin(), [](auto p) {
      return std::make_tuple(static_cast<vertex_t>(std::get<0>(p)),
                             static_cast<vertex_t>(std::get<1>(p)));
    });

    rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

    std::tie(src_v, dst_v) =
      cugraph::generate_path_graph_edgelist<vertex_t>(handle, converted_parms);
    std::tie(src_v, dst_v, std::ignore) = cugraph::symmetrize_edgelist<vertex_t, weight_t>(
      handle, std::move(src_v), std::move(dst_v), std::nullopt);

    if (test_weighted) {
      auto length = src_v.size();
      weights_v.resize(length, handle.get_stream());
    }

    return std::make_tuple(
      std::move(src_v), std::move(dst_v), std::move(weights_v), num_vertices_, symmetric);
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const
  {
    CUGRAPH_FAIL("not implemented");
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
             rmm::device_uvector<weight_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const;

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
             rmm::device_uvector<weight_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const;

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const;

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
             rmm::device_uvector<weight_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const;

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const;

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
             rmm::device_uvector<weight_t>,
             vertex_t,
             bool>
  construct_edgelist(raft::handle_t const& handle, bool test_weighted) const
  {
    size_t constexpr tuple_size{std::tuple_size<generator_tuple_t>::value};

    auto edge_tuple_vector =
      detail::combined_construct_graph_tuple_impl<generator_tuple_t, 0, tuple_size>()
        .construct_edges(handle, test_weighted, generator_tuple_);

    // Need to combine
    CUGRAPH_FAIL("not implemented");
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const
  {
    // Call construct_edgelist to get tuple of edge lists
    // return generate_graph_from_edgelist<...>(...)
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
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
construct_graph(raft::handle_t const& handle,
                input_usecase_t const& input_usecase,
                bool test_weighted,
                bool renumber = true)
{
  rmm::device_uvector<vertex_t> d_src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(0, handle.get_stream());
  rmm::device_uvector<weight_t> d_weights_v(0, handle.get_stream());
  vertex_t num_vertices{0};
  bool is_symmetric{false};

  std::tie(d_src_v, d_dst_v, d_weights_v, num_vertices, is_symmetric) =
    input_usecase
      .template construct_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
        handle, test_weighted);

  return cugraph::experimental::
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::nullopt,
      std::move(d_src_v),
      std::move(d_dst_v),
      std::move(d_weights_v),
      cugraph::experimental::graph_properties_t{is_symmetric, false, test_weighted},
      renumber);
}

}  // namespace test
}  // namespace cugraph
