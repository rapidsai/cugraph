/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <vector>

#include <rmm/thrust_rmm_allocator.h>
#include <algorithms.hpp>
#include <graph.hpp>

#include <utilities/base_fixture.hpp>
template <typename vertex_t, typename edge_t, typename weight_t>
struct EL_host {
  vertex_t v;
  std::vector<vertex_t> src;
  std::vector<vertex_t> dst;
  std::vector<weight_t> weights;
};

namespace cugraph {
namespace ego_test {

template <typename vertex_t, typename edge_t, typename weight_t>
class EGONETTest : public ::testing::TestWithParam<EL_host<vertex_t, edge_t, weight_t>> {
 protected:
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             rmm::device_uvector<weight_t>>
  egonet_test()
  {
    rmm::device_vector<vertex_t> d_edgelist_src(edgelist_h.src);
    rmm::device_vector<vertex_t> d_edgelist_dst(edgelist_h.dst);
    rmm::device_vector<weight_t> d_edgelist_weights(edgelist_h.weights);

    cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_edgelist_src.data().get(),
      d_edgelist_dst.data().get(),
      d_edgelist_weights.data().get(),
      static_cast<edge_t>(d_edgelist_src.size())};

    auto G = cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false>(
      handle, edgelist, v, cugraph::experimental::graph_properties_t{false, false}, false, true);

    vertex_t source = 0;
    vertex_t radius = 3;

    auto result = cugraph::experimental::extract_ego(handle, G.view(), source, radius);
    // raft::print_device_vector("Final EgoNet Src: ", result.src_indices.data(),
    // result.number_of_edges, std::cout); raft::print_device_vector("Final EgoNet Dst: ",
    // result.dst_indices.data(), result.number_of_edges, std::cout); std::cout <<
    // "number_of_EgoNet_edges: " << result.number_of_edges << std::endl;
    return result;
  }

  void SetUp() override
  {
    edgelist_h = ::testing::TestWithParam<EL_host<vertex_t, edge_t, weight_t>>::GetParam();
    v          = edgelist_h.v;
    e          = edgelist_h.src.size();
  }

  void TearDown() override {}

 protected:
  EL_host<vertex_t, edge_t, weight_t> edgelist_h;
  vertex_t v;
  edge_t e;
  raft::handle_t handle;
};

const std::vector<EL_host<int32_t, int32_t, float>> el_in_h = {
  // single iteration
  {8, {0, 0, 0, 1, 1, 2, 2, 3}, {1, 2, 3, 0, 3, 0, 0, 1}, {2, 3, 4, 2, 1, 3, 4, 1}},

  //  multiple iterations and cycles
  {20,
   {0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6},
   {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {5.0f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 5.0f, 2.0f, 6.0f, 8.0f,
    1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},
  // negative weights
  {20,
   {0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6},
   {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {-5.0f, -9.0f,  -1.0f, 4.0f,  -8.0f, -7.0f, -5.0f, -2.0f, -6.0f, -8.0f,
    -1.0f, -10.0f, -9.0f, -2.0f, -1.0f, -1.0f, -6.0f, 4.0f,  -7.0f, -10.0f}},

  // equal weights
  {20,
   {0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6},
   {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2,
    0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1}},

  // self loop
  {20,
   {0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6},
   {0, 4, 5, 6, 3, 6, 2, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {0.5f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 0.5f, 2.0f, 6.0f, 8.0f,
    1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},

  //  disconnected
  {16,
   {0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6},
   {2, 4, 5, 3, 6, 0, 4, 5, 1, 6, 0, 2, 0, 2, 1, 3},
   {5.0f,
    9.0f,
    1.0f,
    8.0f,
    7.0f,
    5.0f,
    2.0f,
    6.0f,
    8.0f,
    10.0f,
    9.0f,
    2.0f,
    1.0f,
    6.0f,
    7.0f,
    10.0f}},

  //  singletons
  {16,
   {0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 6, 6, 7, 7, 8, 8},
   {2, 8, 7, 3, 8, 0, 8, 7, 1, 8, 0, 2, 0, 2, 1, 3},
   {5.0f,
    9.0f,
    1.0f,
    8.0f,
    7.0f,
    5.0f,
    2.0f,
    6.0f,
    8.0f,
    10.0f,
    9.0f,
    2.0f,
    1.0f,
    6.0f,
    7.0f,
    10.0f}}};

typedef EGONETTest<int32_t, int32_t, float> EGONETTest1;
TEST_P(EGONETTest1, happytests)
{
  auto gpu_result = egonet_test();

  // do assertions here
  // EXPECT_LE(gpu_result.n_edges, edgelist_h.size());
}

INSTANTIATE_TEST_CASE_P(EGONETTests, EGONETTest1, ::testing::ValuesIn(el_in_h));

}  // namespace ego_test
}  // namespace cugraph
