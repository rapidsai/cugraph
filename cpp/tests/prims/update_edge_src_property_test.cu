/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include "prims/extract_transform_e.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

typedef struct UpdateEdgeSrcProperty_Usecase_t {
  bool check_correctness{true};
} UpdateEdgeSrcProperty_Usecase;

template <typename input_usecase_t>
class Tests_UpdateEdgeSrcProperty
  : public ::testing::TestWithParam<std::tuple<UpdateEdgeSrcProperty_Usecase, input_usecase_t>> {
 public:
  Tests_UpdateEdgeSrcProperty() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(UpdateEdgeSrcProperty_Usecase const& update_edge_src_property_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("UpdateEdgeSrcProperty");
    }

    raft::random::RngState rng_state(0);
    auto vertices = cugraph::select_random_vertices<vertex_t, vertex_t, store_transposed, false>(
      handle,
      graph_view,
      std::nullopt,
      rng_state,
      std::min(size_t{10}, static_cast<size_t>(graph_view.number_of_vertices())),
      false,
      true);

    rmm::device_uvector<int> property_vector(vertices.size(), handle.get_stream());
    cugraph::detail::sequence_fill(
      handle.get_stream(), property_vector.data(), property_vector.size(), int{1});

    raft::print_device_vector("vertices", vertices.data(), vertices.size(), std::cout);
    raft::print_device_vector(
      "property_vector", property_vector.data(), property_vector.size(), std::cout);

    cugraph::edge_src_property_t<edge_t, int> edge_src_property(handle, graph_view);

    cugraph::fill_edge_src_property(handle, graph_view, edge_src_property.mutable_view(), int{0});
    cugraph::update_edge_src_property(handle,
                                      graph_view,
                                      vertices.begin(),
                                      vertices.end(),
                                      property_vector.begin(),
                                      edge_src_property.mutable_view(),
                                      true);

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<int> props(0, handle.get_stream());

    std::tie(srcs, props) =
      cugraph::extract_transform_e(handle,
                                   graph_view,
                                   edge_src_property.view(),
                                   cugraph::detail::edge_endpoint_dummy_property_view_t{},
                                   cugraph::edge_dummy_property_view_t{},
                                   cuda::proclaim_return_type<cuda::std::tuple<vertex_t, int>>(
                                     [] __device__(vertex_t src, auto, int prop, auto, auto) {
                                       return cuda::std::make_tuple(src, prop);
                                     }));

    raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
    raft::print_device_vector("props", props.data(), props.size(), std::cout);

    // Alternative implementation

    rmm::device_uvector<int> property_vector2(graph_view.local_vertex_partition_range_size(),
                                              handle.get_stream());
    cugraph::detail::scalar_fill(handle, property_vector2.data(), property_vector2.size(), int{0});

    thrust::scatter(handle.get_thrust_policy(),
                    property_vector.begin(),
                    property_vector.end(),
                    vertices.begin(),
                    property_vector2.begin());

    update_edge_src_property(
      handle, graph_view, property_vector2.begin(), edge_src_property.mutable_view());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    std::tie(srcs, props) =
      cugraph::extract_transform_e(handle,
                                   graph_view,
                                   edge_src_property.view(),
                                   cugraph::detail::edge_endpoint_dummy_property_view_t{},
                                   cugraph::edge_dummy_property_view_t{},
                                   cuda::proclaim_return_type<cuda::std::tuple<vertex_t, int>>(
                                     [] __device__(vertex_t src, auto, int prop, auto, auto) {
                                       return cuda::std::make_tuple(src, prop);
                                     }));

    raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
    raft::print_device_vector("props", props.data(), props.size(), std::cout);

    if (update_edge_src_property_usecase.check_correctness) {}
  }
};

using Tests_UpdateEdgeSrcProperty_File = Tests_UpdateEdgeSrcProperty<cugraph::test::File_Usecase>;

TEST_P(Tests_UpdateEdgeSrcProperty_File, CheckInt32Int32FloatUpdateEdgeSrcPropertyFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_UpdateEdgeSrcProperty_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(UpdateEdgeSrcProperty_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

CUGRAPH_TEST_PROGRAM_MAIN()
