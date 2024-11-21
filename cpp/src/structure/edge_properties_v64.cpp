/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "cugraph/device_vector.hpp"
#include "cugraph/utilities/error.hpp"
#include "edge_properties_impl.hpp"

#include <cugraph/graph.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <optional>

namespace cugraph {

template edge_properties_t::edge_properties_t(
  graph_view_t<int64_t, int64_t, false, false> const& graph_view);

template edge_properties_t::edge_properties_t(
  graph_view_t<int64_t, int64_t, false, true> const& graph_view);

template edge_properties_t::edge_properties_t(
  graph_view_t<int64_t, int64_t, true, false> const& graph_view);

template edge_properties_t::edge_properties_t(
  graph_view_t<int64_t, int64_t, true, true> const& graph_view);

template void edge_properties_t::add_property(size_t idx,
                                              std::vector<rmm::device_uvector<int64_t>>&& buffers);

template edge_property_view_t<int64_t, int64_t const*> edge_properties_t::view<int64_t, int64_t>(
  size_t idx) const;

template edge_property_view_t<int64_t, int64_t*> edge_properties_t::mutable_view<int64_t, int64_t>(
  size_t idx);

}  // namespace cugraph
