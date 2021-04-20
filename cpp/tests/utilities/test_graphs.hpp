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

#include <utilities/test_utilities.hpp>

#include <graph_generators.hpp>

namespace cugraph {
namespace test {

class File_Usecase {
 public:
  File_Usecase() = delete;

  File_Usecase(std::string const& graph_file_path, size_t base_vertex_id = 0)
    : base_vertex_id_(base_vertex_id)
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
    auto edgelist = read_edgelist_from_matrix_market_file<vertex_t,
                                                          edge_t,
                                                          weight_t,
                                                          store_transposed,
                                                          multi_gpu>(
      handle, graph_file_full_path_, test_weighted);

    if (base_vertex_id_ > 0)
      cugraph::translate_vertex_ids(
        handle, std::get<0>(edgelist), std::get<1>(edgelist), base_vertex_id_);

    return edgelist;
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
  size_t base_vertex_id_{};
};

class Rmat_Usecase {
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
    : scale_(scale),
      edge_factor_(edge_factor),
      a_(a),
      b_(b),
      c_(c),
      seed_(seed),
      undirected_(undirected),
      scramble_vertex_ids_(scramble_vertex_ids),
      base_vertex_id_(base_vertex_id),
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
    //static_cast<vertex_t>(base_vertex_id_));
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
  size_t base_vertex_id_{};
  bool multi_gpu_usecase_{};
};

class RandomPathGraph_Usecase {
 public:
  RandomPathGraph_Usecase() = delete;

  RandomPathGraph_Usecase(size_t num_vertices, bool weighted = false, size_t base_vertex_id = 0)
    : num_vertices_(num_vertices), weighted_(weighted), base_vertex_id_(base_vertex_id)
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

    auto edgelist =
      cugraph::generate_random_path_graph_edgelist<vertex_t>(handle, num_vertices_, symmetric);

    if (base_vertex_id_ > 0)
      cugraph::translate_vertex_ids(
        handle, std::get<0>(edgelist), std::get<1>(edgelist), base_vertex_id_);

    if (test_weighted) {
      auto length = std::get<0>(edgelist).size();
      weights_v.resize(length, handle.get_stream());
    }

    return std::make_tuple(
      std::get<0>(edgelist), std::get<1>(edgelist), weights_v, num_vertices_, symmetric);
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
  size_t num_vertices_{0};
  bool weighted_{false};
  size_t base_vertex_id_{0};
};

class Mesh2DGraph_Usecase {
 public:
  Mesh2DGraph_Usecase() = delete;

  Mesh2DGraph_Usecase(size_t x, size_t y, size_t num_meshes, size_t base_vertex_id)
    : x_(x), y_(y), num_meshes_(num_meshes), base_vertex_id_(base_vertex_id)
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
  size_t x_{};
  size_t y_{};
  size_t num_meshes_{};
  size_t base_vertex_id_{0};
};

class Mesh3DGraph_Usecase {
 public:
  Mesh3DGraph_Usecase() = delete;

  Mesh3DGraph_Usecase(size_t x, size_t y, size_t z, size_t num_meshes, size_t base_vertex_id)
    : x_(x), y_(y), z_(z), num_meshes_(num_meshes), base_vertex_id_(base_vertex_id)
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
  size_t x_{};
  size_t y_{};
  size_t z_{};
  size_t num_meshes_{};
  size_t base_vertex_id_{0};
};

class CliqueGraph_Usecase {
 public:
  CliqueGraph_Usecase() = delete;

  CliqueGraph_Usecase(size_t clique_size, size_t num_cliques, size_t base_vertex_id)
    : clique_size_(clique_size), num_cliques_(num_cliques), base_vertex_id_(base_vertex_id)
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
  size_t clique_size_{};
  size_t num_cliques_{};
  size_t base_vertex_id_{};
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

}  // namespace test
}  // namespace cugraph
