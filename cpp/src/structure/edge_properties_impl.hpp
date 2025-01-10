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

#pragma once

#include "cugraph/device_vector.hpp"
#include "cugraph/edge_property.hpp"
#include "cugraph/utilities/cugraph_data_type_id.hpp"

#include <cugraph/edge_properties.hpp>

#include <thrust/tabulate.h>

#include <algorithm>

namespace cugraph {

namespace detail {

template <typename value_type>
edge_property_impl_t::edge_property_impl_t(std::vector<rmm::device_uvector<value_type>>&& buffers,
                                           std::vector<size_t> const& edge_counts)
{
  CUGRAPH_EXPECTS(buffers.size() == edge_counts.size(),
                  "Mismatch in buffer size (%lu) and edge counts size (%lu)",
                  buffers.size(),
                  edge_counts.size());
  std::for_each(thrust::make_zip_iterator(buffers.begin(), edge_counts.begin()),
                thrust::make_zip_iterator(buffers.end(), edge_counts.end()),
                [](auto const& t) {
                  CUGRAPH_EXPECTS(thrust::get<0>(t).size() == thrust::get<1>(t),
                                  "Buffer size (%lu) and edge counts (%lu) should match",
                                  thrust::get<0>(t).size(),
                                  thrust::get<1>(t));
                });

  dtype_ = type_to_id<value_type>();
  vectors_.reserve(edge_counts.size());
  std::for_each(buffers.begin(), buffers.end(), [&vectors = vectors_](auto& buffer) {
    vectors.push_back(cugraph::device_vector_t(std::move(buffer)));
  });
}

template <typename edge_t, typename value_t>
edge_property_view_t<edge_t, value_t const*> edge_property_impl_t::view(
  std::vector<size_t> const& edge_counts) const
{
  CUGRAPH_EXPECTS(dtype_ == type_to_id<value_t>(),
                  "Requesting invalid type for property, requesting type %s, property of type %s",
                  type_to_name(type_to_id<value_t>()).c_str(),
                  type_to_name(dtype_).c_str());

  std::vector<edge_t> counts(edge_counts.size());
  std::vector<value_t const*> value_iterators(edge_counts.size());

  std::transform(edge_counts.begin(), edge_counts.end(), counts.begin(), [](size_t c) {
    return static_cast<edge_t>(c);
  });
  std::transform(
    vectors_.begin(), vectors_.end(), value_iterators.begin(), [](device_vector_t const& buff) {
      return buff.begin<value_t>();
    });

  // FIXME: Could edge_property_view_t just use size_t instead of edge_t?
  //  It's a small amount of host memory.
  return edge_property_view_t<edge_t, value_t const*>{};
}

template <typename edge_t, typename value_t>
edge_property_view_t<edge_t, value_t*> edge_property_impl_t::mutable_view(
  std::vector<size_t> const& edge_counts)
{
  CUGRAPH_EXPECTS(dtype_ == type_to_id<value_t>(),
                  "Requesting invalid type for property, requesting type %s, property of type %s",
                  type_to_name(type_to_id<value_t>()).c_str(),
                  type_to_name(dtype_).c_str());

  std::vector<edge_t> counts(edge_counts.size());
  std::vector<value_t*> value_iterators(edge_counts.size());

  std::transform(edge_counts.begin(), edge_counts.end(), counts.begin(), [](size_t c) {
    return static_cast<edge_t>(c);
  });
  std::transform(
    vectors_.begin(), vectors_.end(), value_iterators.begin(), [](device_vector_t& buff) {
      return buff.begin<value_t>();
    });

  // FIXME: Could edge_property_view_t just use size_t instead of edge_t?
  //  It's a small amount of host memory.
  return edge_property_view_t<edge_t, value_t*>{};
}

}  // namespace detail

template <typename GraphViewType>
edge_properties_t::edge_properties_t(GraphViewType const& graph_view)
{
  edge_counts_.resize(graph_view.number_of_local_edge_partitions());

  thrust::tabulate(edge_counts_.begin(), edge_counts_.end(), [&graph_view](size_t i) {
    return static_cast<size_t>(graph_view.local_edge_partition_view(i).number_of_edges());
  });
}

template <typename value_type>
void edge_properties_t::add_property(size_t idx,
                                     std::vector<rmm::device_uvector<value_type>>&& buffers)
{
  if (idx < properties_.size()) {
    CUGRAPH_EXPECTS(
      !is_defined(idx), "Cannot replace an existing property (%lu), use clear_property first", idx);
  } else {
    // properties_.resize(idx + 1, std::nullopt);
    properties_.reserve(idx + 1);
    while (properties_.size() < (idx + 1)) {
      properties_.push_back(std::nullopt);
    }
  }

  properties_[idx] = detail::edge_property_impl_t(std::move(buffers), edge_counts_);
}

template <typename edge_t, typename value_t>
edge_property_view_t<edge_t, value_t const*> edge_properties_t::view(size_t idx) const
{
  CUGRAPH_EXPECTS(idx < properties_.size(),
                  "idx %lu out of range (only %lu properties)",
                  idx,
                  properties_.size());
  CUGRAPH_EXPECTS(properties_[idx].has_value(), "property %lu is not defined", idx);
  return properties_[idx]->view<edge_t, value_t>(edge_counts_);
}

template <typename edge_t, typename value_t>
edge_property_view_t<edge_t, value_t*> edge_properties_t::mutable_view(size_t idx)
{
  CUGRAPH_EXPECTS(idx < properties_.size(),
                  "idx %lu out of range (only %lu properties)",
                  idx,
                  properties_.size());
  CUGRAPH_EXPECTS(properties_[idx].has_value(), "property %lu is not defined", idx);
  return properties_[idx]->mutable_view<edge_t, value_t>(edge_counts_);
}

}  // namespace cugraph
