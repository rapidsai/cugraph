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

#include <cugraph_c/types.h>

#include <cugraph/device_vector.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/utilities/cugraph_data_type_id.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {

namespace detail {

class edge_property_impl_t {
 public:
  edge_property_impl_t(raft::handle_t const& handle,
                       cugraph_data_type_id_t data_type,
                       std::vector<size_t> const& edge_counts);

  edge_property_impl_t(cugraph_data_type_id_t data_type,
                       std::vector<cugraph::device_vector_t>&& vectors);

  template <typename value_type>
  edge_property_impl_t(std::vector<rmm::device_uvector<value_type>>&& vectors,
                       std::vector<size_t> const& edge_counts);

  template <typename edge_t, typename value_t>
  edge_property_view_t<edge_t, value_t const*> view(std::vector<size_t> const& edge_counts) const;

  template <typename edge_t, typename value_t>
  edge_property_view_t<edge_t, value_t*> mutable_view(std::vector<size_t> const& edge_counts);

  cugraph_data_type_id_t data_type() const { return dtype_; }

 private:
  cugraph_data_type_id_t dtype_{NTYPES};
  std::vector<cugraph::device_vector_t> vectors_{};
};

}  // namespace detail

/**
 * Class for containing a collection of edge properties.  Semantic interpretation of
 * the properties is for the caller to interpret.
 *
 * Edge properties are labeled as in a vector, from 0 to n-1.  It is up to the caller to
 * handle proper usage of the properties.
 */
class edge_properties_t {
 public:
  /**
   * Constructor initializing properties from a graph view
   *
   * @tparam GraphViewType   type for the graph view
   * @param  graph_view      Graph view object
   */
  template <typename GraphViewType>
  edge_properties_t(GraphViewType const& graph_view);

  /**
   * Adds an empty property of the specified type.
   *
   * Fails if the property for @p idx is already defined
   *
   * @param  handle    Handle for resources
   * @param  idx       Index of which property to add
   * @param  data_type Data type of the property to add
   */
  void add_property(raft::handle_t const& handle, size_t idx, cugraph_data_type_id_t data_type);

  /**
   * Adds a property of the specified type initialized with the provided values
   *
   * Fails if the property for @p idx is already defined
   *
   * @tparam value_type  Type of the property
   * @param  idx         Index of which property to add
   * @param  buffers     Initial value of the property
   */
  template <typename value_type>
  void add_property(size_t idx, std::vector<rmm::device_uvector<value_type>>&& buffers);

  /**
   * Adds a property initialized with the provided values
   *
   * Fails if the property for @p idx is already defined
   *
   * @param  idx        Index of which property to add
   * @param  vectors    Type erased device vectors for the initial value
   */
  void add_property(size_t idx, std::vector<cugraph::device_vector_t>&& vectors);
  /**
   * Clears the specified property, releasing any allocated memory
   *
   * @param  idx     Index of which property to clear
   */
  void clear_property(size_t idx);

  /**
   * clears all properties, releasing any allocate memory
   */
  void clear_all_properties();

  /**
   * Returns true if property @p idx is defined
   */
  bool is_defined(size_t idx);

  /**
   * Returns data type of property @p idx
   */
  cugraph_data_type_id_t data_type(size_t idx);

  /**
   * Returns a read-only edge property view of the property using the provided types
   *
   * Throws exception if idx does not refer to a defined property.
   * Throws exception if value_t does not match the property type
   *
   * @tparam edge_t     Typename for the edge
   * @tparam value_t    Typename for the property
   * @param  idx        Index of the property
   *
   * @return a read-only view for accessing the property
   */
  template <typename edge_t, typename value_t>
  edge_property_view_t<edge_t, value_t const*> view(size_t idx) const;

  /**
   * Returns a read-write edge property view of the property using the provided types
   *
   * Throws exception if idx does not refer to a defined property.
   * Throws exception if value_t does not match the property type
   *
   * @tparam edge_t     Typename for the edge
   * @tparam value_t    Typename for the property
   * @param  idx        Index of the property
   *
   * @return a read-write view for accessing the property
   */
  template <typename edge_t, typename value_t>
  edge_property_view_t<edge_t, value_t*> mutable_view(size_t idx);

  /**
   * Return list of defined properties
   *
   * @return vector of defined property indexes
   */
  std::vector<size_t> defined() const;

 private:
  std::vector<std::optional<detail::edge_property_impl_t>> properties_{};
  std::vector<size_t> edge_counts_{};
};

}  // namespace cugraph
