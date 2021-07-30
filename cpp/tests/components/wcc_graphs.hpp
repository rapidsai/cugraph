/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <cugraph/graph.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace test {

class LineGraph_Usecase {
 public:
  LineGraph_Usecase() = delete;

  LineGraph_Usecase(size_t num_vertices) : num_vertices_(num_vertices) {}

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
             std::optional<rmm::device_uvector<vertex_t>>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const;

 private:
  size_t num_vertices_{0};
};

}  // namespace test
}  // namespace cugraph
