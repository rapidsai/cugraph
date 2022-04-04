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

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <utilities/../sampling/random_walks_utils.cuh>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

// visitor artifacts:
//
#include <cugraph/visitors/erased_api.hpp>
#include <cugraph/visitors/erased_pack.hpp>
#include <cugraph/visitors/graph_envelope.hpp>
#include <cugraph/visitors/ret_terased.hpp>
#include <cugraph/visitors/rw_visitor.hpp>

#include <gtest/gtest.h>

#include <iterator>
#include <limits>
#include <vector>

namespace {  // anonym.
template <typename vertex_t, typename index_t>
void fill_start(raft::handle_t const& handle,
                rmm::device_uvector<vertex_t>& d_start,
                index_t num_vertices)
{
  index_t num_paths = d_start.size();

  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<index_t>(0),
                    thrust::make_counting_iterator<index_t>(num_paths),

                    d_start.begin(),
                    [num_vertices] __device__(auto indx) { return indx % num_vertices; });
}
}  // namespace

struct RandomWalks_Usecase {
  std::string graph_file_full_path{};
  bool test_weighted{false};

  RandomWalks_Usecase(std::string const& graph_file_path, bool test_weighted)
    : test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
};

class Tests_RW_Visitor : public ::testing::TestWithParam<RandomWalks_Usecase> {
 public:
  Tests_RW_Visitor() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(RandomWalks_Usecase const& configuration)
  {
    using namespace cugraph::visitors;
    using index_t = edge_t;

#ifdef _USE_UNERASED_RW_RET_
    using algo_ret_t = std::tuple<rmm::device_uvector<vertex_t>,
                                  rmm::device_uvector<weight_t>,
                                  rmm::device_uvector<index_t>>;
#else
    using algo_ret_t = std::tuple<rmm::device_buffer, rmm::device_buffer, rmm::device_buffer>;
#endif

    using ptr_params_t = std::unique_ptr<cugraph::sampling_params_t>;

    raft::handle_t handle{};

    // extract graph data from graph matrix file:
    //
    auto&& [d_src, d_dst, opt_d_w, d_verts_ignore, num_vertices, is_sym] =
      cugraph::test::read_edgelist_from_matrix_market_file<vertex_t, weight_t, false, false>(
        handle, configuration.graph_file_full_path, configuration.test_weighted);

    cugraph::graph_properties_t graph_props{is_sym, false};
    edge_t num_edges = d_dst.size();

    std::optional<weight_t const*> opt_ptr_w;
    if (opt_d_w.has_value()) { opt_ptr_w = opt_d_w->data(); }

    // to be filled:
    //
    cugraph::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_src.data(), d_dst.data(), opt_ptr_w, num_edges};

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

    index_t num_paths{10};
    index_t max_depth{10};
    rmm::device_uvector<vertex_t> d_start(num_paths, handle.get_stream());

    fill_start(handle, d_start, graph_view.number_of_vertices());

    // visitors machinery:
    //
    // in a context where dependent types are known,
    // type-erasing the graph is not necessary,
    // hence the `<alg>_wrapper()` is not necessary;
    //

    // packing visitor arguments = random walks algorithm arguments
    //
    vertex_t* p_d_start = d_start.data();
    bool use_padding{false};
    auto sampling = std::make_unique<cugraph::sampling_params_t>(0);
    erased_pack_t ep{&handle,
                     p_d_start,
                     &num_paths,
                     &max_depth,
                     &use_padding,
                     &sampling};  // type-erased args for random_walks()

    // several options to run the <algorithm>:
    // (<algorithm> \in {bfs, random_walks, etc.})
    //
    // (1.) if a graph object already exists,
    //      we can use it to make the appropriate
    //      visitor:
    //
    // auto v_uniq_ptr = make_visitor(
    //   *p_graph,
    //   [](graph_envelope_t::visitor_factory_t const& vfact, erased_pack_t& parg) {
    //     return vfact.make_<algorithm>_visitor(parg);
    //   },
    //   ep);
    // p_graph->apply(*v_uniq_ptr);

    // (2.) if a graph object already exists, alternatively we can
    //      explicitly instantiate the factory and call its make method:
    //
    // dependent_factory_t<vertex_t, edge_t, weight_t, false, false> visitor_factory{};
    // auto v_uniq_ptr = visitor_factory.make_<algorithm>_visitor(ep)
    // p_graph->apply(*v_uniq_ptr);

    // (3.) if only the `graph_envelope_t` object exists,
    //      we can invoke the algorithm via the wrapper:
    //
    return_t ret = cugraph::api::random_walks(graph_envelope, ep);

    // unpack type-erased result:
    //
    auto&& ret_tuple = ret.get<algo_ret_t>();

    // check results:
    //
#ifdef _USE_UNERASED_RW_RET_
    bool test_all_paths = cugraph::test::host_check_rw_paths(
      handle, graph_view, std::get<0>(ret_tuple), std::get<1>(ret_tuple), std::get<2>(ret_tuple));
#else
    rmm::device_buffer const& d_buf_vertices = std::get<0>(ret_tuple);
    size_t num_path_vertices                 = d_buf_vertices.size() / sizeof(vertex_t);

    rmm::device_buffer const& d_buf_weights = std::get<1>(ret_tuple);
    size_t num_path_edges                   = d_buf_weights.size() / sizeof(weight_t);

    rmm::device_buffer const& d_buf_sizes = std::get<2>(ret_tuple);
    size_t path_sizes                     = d_buf_sizes.size() / sizeof(index_t);

    bool test_all_paths =
      cugraph::test::host_check_rw_paths(handle,
                                         graph_view,
                                         static_cast<vertex_t const*>(d_buf_vertices.data()),
                                         num_path_vertices,
                                         static_cast<weight_t const*>(d_buf_weights.data()),
                                         num_path_edges,
                                         static_cast<index_t const*>(d_buf_sizes.data()),
                                         path_sizes,
                                         0);
#endif

    ASSERT_TRUE(test_all_paths);
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_RW_Visitor, CheckInt32Int32) { run_current_test<int32_t, int32_t, float>(GetParam()); }

INSTANTIATE_TEST_CASE_P(visitor_test,
                        Tests_RW_Visitor,
                        ::testing::Values(RandomWalks_Usecase("test/datasets/karate.mtx", true),
                                          RandomWalks_Usecase("test/datasets/web-Google.mtx",
                                                              true)));

CUGRAPH_TEST_PROGRAM_MAIN()
