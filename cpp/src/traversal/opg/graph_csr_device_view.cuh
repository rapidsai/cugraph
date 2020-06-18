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

#ifndef GRAPH_CSR_DEVICE_VIEW_CUH
#define GRAPH_CSR_DEVICE_VIEW_CUH

#include <graph.hpp>

namespace detail {

namespace opg {

template <typename VT, typename ET, typename WT>
class GraphCSRDeviceView {
  ET *offsets_{nullptr};    ///< CSR offsets
  VT *indices_{nullptr};    ///< CSR indices
  WT *edge_data_{nullptr};  ///< edge_data
  VT number_of_vertices_{0};
  ET number_of_edges_{0};

  public:
  GraphCSRDeviceView(cugraph::experimental::GraphCSRView<VT, ET, WT> &graph) :
    offsets_(graph.offsets),
    indices_(graph.indices),
    edge_data_(graph.edge_data),
    number_of_vertices_(graph.number_of_vertices),
    number_of_edges_(graph.number_of_edges) {}

  __device__
  ET * offsets(void) { return offsets_; }

  __device__
  VT * indices(void) { return indices_; }

  __device__
  WT * edge_data(void) { return edge_data_; }

  __device__
  VT number_of_vertices(void) { return number_of_vertices_; }

  __device__
  ET number_of_edges(void) { return number_of_edges_; }

};

}//namespace opg

}//namespace detail

#endif //GRAPH_CSR_DEVICE_VIEW_CUH
