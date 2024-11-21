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

namespace detail {

edge_property_impl_t::edge_property_impl_t(raft::handle_t const& handle,
                                           cugraph_data_type_id_t data_type,
                                           std::vector<size_t> const& edge_counts)
  : dtype_(data_type)
{
  vectors_.reserve(edge_counts.size());
  std::for_each(edge_counts.begin(),
                edge_counts.end(),
                [&handle, data_type, &vectors = vectors_](size_t n_elements) {
                  vectors.push_back(cugraph::device_vector_t(handle, data_type, n_elements));
                });
}

edge_property_impl_t::edge_property_impl_t(cugraph_data_type_id_t data_type,
                                           std::vector<cugraph::device_vector_t>&& vectors)
  : dtype_(data_type), vectors_(std::move(vectors))
{
}

template edge_property_impl_t::edge_property_impl_t(
  std::vector<rmm::device_uvector<int32_t>>&& buffers, std::vector<size_t> const& edge_counts);

}  // namespace detail

void edge_properties_t::add_property(raft::handle_t const& handle,
                                     size_t idx,
                                     cugraph_data_type_id_t data_type)
{
  if (idx < properties_.size()) {
    CUGRAPH_EXPECTS(
      !is_defined(idx), "Cannot replace an existing property (%lu), use clear_property first", idx);
  } else {
    properties_.reserve(idx + 1);
    while (properties_.size() < (idx + 1)) {
      properties_.push_back(std::nullopt);
    }
  }

  properties_[idx] = detail::edge_property_impl_t(handle, data_type, edge_counts_);
}

void edge_properties_t::add_property(size_t idx, std::vector<cugraph::device_vector_t>&& vectors)
{
  if (idx < properties_.size()) {
    CUGRAPH_EXPECTS(
      !is_defined(idx), "Cannot replace an existing property (%lu), use clear_property first", idx);
  } else {
    properties_.reserve(idx + 1);
    while (properties_.size() < (idx + 1)) {
      properties_.push_back(std::nullopt);
    }
  }

  properties_[idx] = detail::edge_property_impl_t(vectors[0].type(), std::move(vectors));
}

void edge_properties_t::clear_property(size_t idx)
{
  if (idx < properties_.size()) { properties_[idx] = std::nullopt; }
}

void edge_properties_t::clear_all_properties()
{
  std::fill(properties_.begin(), properties_.end(), std::nullopt);
}

bool edge_properties_t::is_defined(size_t idx)
{
  return (idx < properties_.size() && properties_[idx].has_value());
}

cugraph_data_type_id_t edge_properties_t::data_type(size_t idx)
{
  return idx < properties_.size()
           ? (properties_[idx].has_value() ? properties_[idx]->data_type() : NTYPES)
           : NTYPES;
}

std::vector<size_t> edge_properties_t::defined() const
{
  std::vector<size_t> result;
  result.reserve(std::count_if(
    properties_.begin(), properties_.end(), [](auto const& p) { return p.has_value(); }));

  std::copy_if(thrust::make_counting_iterator<size_t>(0),
               thrust::make_counting_iterator(properties_.size()),
               result.begin(),
               [&properties = properties_](auto idx) { return properties[idx].has_value(); });

  return result;
}

template edge_properties_t::edge_properties_t(
  graph_view_t<int32_t, int32_t, false, false> const& graph_view);

template edge_properties_t::edge_properties_t(
  graph_view_t<int32_t, int32_t, false, true> const& graph_view);

template edge_properties_t::edge_properties_t(
  graph_view_t<int32_t, int32_t, true, false> const& graph_view);

template edge_properties_t::edge_properties_t(
  graph_view_t<int32_t, int32_t, true, true> const& graph_view);

template void edge_properties_t::add_property(size_t idx,
                                              std::vector<rmm::device_uvector<int32_t>>&& buffers);

template edge_property_view_t<int32_t, int32_t const*> edge_properties_t::view<int32_t, int32_t>(
  size_t idx) const;

template edge_property_view_t<int32_t, int32_t*> edge_properties_t::mutable_view<int32_t, int32_t>(
  size_t idx);

}  // namespace cugraph
