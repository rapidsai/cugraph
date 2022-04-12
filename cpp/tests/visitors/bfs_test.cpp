/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

// Andrei Schaffer, aschaffer@nvidia.com
//

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

// visitor artifacts:
//
#include <cugraph/visitors/bfs_visitor.hpp>
#include <cugraph/visitors/erased_api.hpp>
#include <cugraph/visitors/erased_pack.hpp>
#include <cugraph/visitors/graph_envelope.hpp>
#include <cugraph/visitors/ret_terased.hpp>

#include <gtest/gtest.h>

#include <iterator>
#include <limits>
#include <vector>

template <typename vertex_t, typename edge_t>
void bfs_reference(edge_t* offsets,
                   vertex_t* indices,
                   vertex_t* distances,
                   vertex_t* predecessors,
                   vertex_t num_vertices,
                   vertex_t source,
                   vertex_t depth_limit = std::numeric_limits<vertex_t>::max())
{
  vertex_t depth{0};

  std::fill(distances, distances + num_vertices, std::numeric_limits<vertex_t>::max());
  std::fill(predecessors, predecessors + num_vertices, cugraph::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = depth;
  std::vector<vertex_t> cur_frontier_srcs{source};
  std::vector<vertex_t> new_frontier_srcs{};

  while (cur_frontier_srcs.size() > 0) {
    for (auto const row : cur_frontier_srcs) {
      auto nbr_offset_first = *(offsets + row);
      auto nbr_offset_last  = *(offsets + row + 1);
      for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
        auto nbr = *(indices + nbr_offset);
        if (*(distances + nbr) == std::numeric_limits<vertex_t>::max()) {
          *(distances + nbr)    = depth + 1;
          *(predecessors + nbr) = row;
          new_frontier_srcs.push_back(nbr);
        }
      }
    }
    std::swap(cur_frontier_srcs, new_frontier_srcs);
    new_frontier_srcs.clear();
    ++depth;
    if (depth >= depth_limit) { break; }
  }

  return;
}

typedef struct BFS_Usecase_t {
  std::string graph_file_full_path{};
  size_t source{false};

  BFS_Usecase_t(std::string const& graph_file_path, size_t source) : source(source)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} BFS_Usecase;

class Tests_BFS : public ::testing::TestWithParam<BFS_Usecase> {
 public:
  Tests_BFS() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(BFS_Usecase const& configuration)
  {
    using namespace cugraph::visitors;

    using weight_t = float;

    raft::handle_t handle{};

    bool test_weighted = false;

    // extract graph data from graph matrix file:
    //
    auto&& [d_src, d_dst, opt_d_w, opt_v, num_vertices, is_sym] =
      cugraph::test::read_edgelist_from_matrix_market_file<vertex_t, weight_t, false, false>(
        handle, configuration.graph_file_full_path, test_weighted);

    cugraph::graph_properties_t graph_props{is_sym, false};
    edge_t num_edges = d_dst.size();

    std::optional<weight_t const*> opt_ptr_w;
    if (opt_d_w.has_value()) { opt_ptr_w = opt_d_w->data(); }

    // to be filled:
    //
    cugraph::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_src.data(), d_dst.data(), opt_ptr_w, num_edges};
    bool sorted{false};
    bool check{false};

    cugraph::graph_meta_t<vertex_t, edge_t, false> meta{num_vertices, graph_props, std::nullopt};
    erased_pack_t ep_graph{&handle, &edgelist, &meta, &check};

    DTypes vertex_tid = reverse_dmap_t<vertex_t>::type_id;
    DTypes edge_tid   = reverse_dmap_t<edge_t>::type_id;
    DTypes weight_tid = reverse_dmap_t<weight_t>::type_id;
    bool st           = false;
    bool mg           = false;
    GTypes graph_tid  = GTypes::GRAPH_T;

    graph_envelope_t graph_envelope{vertex_tid, edge_tid, weight_tid, st, mg, graph_tid, ep_graph};

    auto const* p_graph =
      dynamic_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> const*>(
        graph_envelope.graph().get());

    auto graph_view = p_graph->view();

    std::vector<edge_t> h_offsets(graph_view.number_of_vertices() + 1);
    std::vector<vertex_t> h_indices(graph_view.number_of_edges());
    raft::update_host(h_offsets.data(),
                      graph_view.local_edge_partition_view().offsets(),
                      graph_view.number_of_vertices() + 1,
                      handle.get_stream());
    raft::update_host(h_indices.data(),
                      graph_view.local_edge_partition_view().indices(),
                      graph_view.number_of_edges(),
                      handle.get_stream());
    handle.sync_stream();

    ASSERT_TRUE(configuration.source >= 0 &&
                configuration.source <= graph_view.number_of_vertices())
      << "Starting source vertex value should be >= 0 and"
      << " less than the number of vertices in the graph.";

    std::vector<vertex_t> h_reference_distances(graph_view.number_of_vertices());
    std::vector<vertex_t> h_reference_predecessors(graph_view.number_of_vertices());

    bfs_reference(h_offsets.data(),
                  h_indices.data(),
                  h_reference_distances.data(),
                  h_reference_predecessors.data(),
                  graph_view.number_of_vertices(),
                  static_cast<vertex_t>(configuration.source),
                  std::numeric_limits<vertex_t>::max());

    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
    {
      // visitors version:
      //
      // in a context where dependent types are known,
      // type-erasing the graph is not necessary,
      // hence the `<alg>_wrapper()` is not necessary;
      //

      // packing visitor arguments = bfs algorithm arguments
      //
      vertex_t* p_d_dist   = d_distances.begin();
      vertex_t* p_d_predec = d_predecessors.begin();
      rmm::device_scalar<vertex_t> d_source(configuration.source, handle.get_stream());
      bool dir_opt{false};
      auto depth_l = std::numeric_limits<vertex_t>::max();
      bool check{false};
      size_t n_sources{1};
      erased_pack_t ep{&handle,
                       p_d_dist,
                       p_d_predec,
                       d_source.data(),
                       &n_sources,
                       &dir_opt,
                       &depth_l,
                       &check};  // args for bfs(),

      // several options to run the BFS algorithm:
      //
      // (1.) if a graph object already exists,
      //      we can use it to make the appropriate
      //      visitor:
      //
      // auto v_uniq_ptr = make_visitor(
      //   *p_graph,
      //   [](graph_envelope_t::visitor_factory_t const& vfact, erased_pack_t& parg) {
      //     return vfact.make_bfs_visitor(parg);
      //   },
      //   ep);
      // p_graph->apply(*v_uniq_ptr);

      // (2.) if a graph object already exists, alternatively we can
      //      explicitly instantiate the factory and call its make method:
      //
      // dependent_factory_t<vertex_t, edge_t, weight_t, false, false> visitor_factory{}; // okay
      // auto v_uniq_ptr = visitor_factory.make_bfs_visitor(ep); // okay
      // p_graph->apply(*v_uniq_ptr);

      // (3.) if only the `graph_envelope_t` object exists,
      //      we can invoke the algorithm via the wrapper:
      //
      return_t ret = cugraph::api::bfs(graph_envelope, ep);
    }

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<vertex_t> h_cugraph_distances(graph_view.number_of_vertices());
    std::vector<vertex_t> h_cugraph_predecessors(graph_view.number_of_vertices());

    raft::update_host(
      h_cugraph_distances.data(), d_distances.data(), d_distances.size(), handle.get_stream());
    raft::update_host(h_cugraph_predecessors.data(),
                      d_predecessors.data(),
                      d_predecessors.size(),
                      handle.get_stream());
    handle.sync_stream();

    ASSERT_TRUE(std::equal(
      h_reference_distances.begin(), h_reference_distances.end(), h_cugraph_distances.begin()))
      << "distances do not match with the reference values.";

    for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
      auto i = std::distance(h_cugraph_predecessors.begin(), it);
      if (*it == cugraph::invalid_vertex_id<vertex_t>::value) {
        ASSERT_TRUE(h_reference_predecessors[i] == *it)
          << "vertex reachability do not match with the reference.";
      } else {
        ASSERT_TRUE(h_reference_distances[*it] + 1 == h_reference_distances[i])
          << "distance to this vertex != distance to the predecessor vertex + 1.";
        bool found{false};
        for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
          if (h_indices[j] == i) {
            found = true;
            break;
          }
        }
        ASSERT_TRUE(found) << "no edge from the predecessor vertex to this vertex.";
      }
    }
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_BFS, CheckInt32Int32) { run_current_test<int32_t, int32_t>(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_BFS,
                        ::testing::Values(BFS_Usecase("test/datasets/karate.mtx", 0),
                                          BFS_Usecase("test/datasets/polbooks.mtx", 0),
                                          BFS_Usecase("test/datasets/netscience.mtx", 0),
                                          BFS_Usecase("test/datasets/netscience.mtx", 100),
                                          BFS_Usecase("test/datasets/wiki2003.mtx", 1000),
                                          BFS_Usecase("test/datasets/wiki-Talk.mtx", 1000)));

CUGRAPH_TEST_PROGRAM_MAIN()
