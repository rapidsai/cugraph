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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/error.h>
#include <cugraph_c/graph_functions.h>

#include <cugraph/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

namespace {

template <typename vertex_t>
cugraph_error_code_t renumber_arbitrary_edgelist(
  raft::handle_t const& handle,
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* renumber_map,
  cugraph::c_api::cugraph_type_erased_device_array_view_t* srcs,
  cugraph::c_api::cugraph_type_erased_device_array_view_t* dsts)
{
  rmm::device_uvector<vertex_t> vertices(2 * srcs->size_, handle.get_stream());

  thrust::copy_n(
    handle.get_thrust_policy(), srcs->as_type<vertex_t>(), srcs->size_, vertices.data());
  thrust::copy_n(handle.get_thrust_policy(),
                 dsts->as_type<vertex_t>(),
                 dsts->size_,
                 vertices.data() + srcs->size_);

  thrust::sort(handle.get_thrust_policy(), vertices.begin(), vertices.end());
  vertices.resize(cuda::std::distance(
                    vertices.begin(),
                    thrust::unique(handle.get_thrust_policy(), vertices.begin(), vertices.end())),
                  handle.get_stream());

  vertices.shrink_to_fit(handle.get_stream());
  rmm::device_uvector<vertex_t> ids(vertices.size(), handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               ids.begin(),
               ids.end(),
               cugraph::invalid_vertex_id<vertex_t>::value);

  raft::device_span<vertex_t const> vertices_span{vertices.data(), vertices.size()};
  raft::device_span<vertex_t> ids_span{ids.data(), ids.size()};

  // Read chunk of renumber_map in a loop, updating base offset to compute vertex id
  // FIXME: Compute this as a function of free memory?  Or some value that keeps a
  // particular GPU saturated?
  size_t chunk_size = size_t{1} << 20;

  rmm::device_uvector<vertex_t> renumber_chunk(chunk_size, handle.get_stream());

  for (size_t chunk_base_offset = 0; chunk_base_offset < renumber_map->size_;
       chunk_base_offset += chunk_size) {
    size_t size = std::min(chunk_size, renumber_map->size_ - chunk_base_offset);
    if (size < chunk_size) renumber_chunk.resize(size, handle.get_stream());

    raft::update_device(renumber_chunk.data(),
                        renumber_map->as_type<vertex_t>() + chunk_base_offset,
                        size,
                        handle.get_stream());

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(renumber_chunk.size()),
      [chunk_base_offset,
       renumber_chunk_span =
         raft::device_span<vertex_t const>{renumber_chunk.data(), renumber_chunk.size()},
       vertices_span,
       ids_span] __device__(size_t idx) {
        auto pos = thrust::lower_bound(
          thrust::seq, vertices_span.begin(), vertices_span.end(), renumber_chunk_span[idx]);
        if ((pos != vertices_span.end()) && (*pos == renumber_chunk_span[idx])) {
          ids_span[cuda::std::distance(vertices_span.begin(), pos)] =
            static_cast<vertex_t>(chunk_base_offset + idx);
        }
      });
  }

  CUGRAPH_EXPECTS(thrust::count(handle.get_thrust_policy(),
                                ids.begin(),
                                ids.end(),
                                cugraph::invalid_vertex_id<vertex_t>::value) == 0,
                  "some vertices were not renumbered");

  thrust::transform(
    handle.get_thrust_policy(),
    srcs->as_type<vertex_t>(),
    srcs->as_type<vertex_t>() + srcs->size_,
    srcs->as_type<vertex_t>(),
    [vertices_span, ids_span] __device__(vertex_t v) {
      return ids_span[cuda::std::distance(
        vertices_span.begin(),
        thrust::lower_bound(thrust::seq, vertices_span.begin(), vertices_span.end(), v))];
    });

  thrust::transform(
    handle.get_thrust_policy(),
    dsts->as_type<vertex_t>(),
    dsts->as_type<vertex_t>() + srcs->size_,
    dsts->as_type<vertex_t>(),
    [vertices_span, ids_span] __device__(vertex_t v) {
      return ids_span[cuda::std::distance(
        vertices_span.begin(),
        thrust::lower_bound(thrust::seq, vertices_span.begin(), vertices_span.end(), v))];
    });

  return CUGRAPH_SUCCESS;
}

}  // namespace

extern "C" cugraph_error_code_t cugraph_renumber_arbitrary_edgelist(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_host_array_view_t* renumber_map,
  cugraph_type_erased_device_array_view_t* srcs,
  cugraph_type_erased_device_array_view_t* dsts,
  cugraph_error_t** error)
{
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* h_renumber_map =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(renumber_map);
  cugraph::c_api::cugraph_type_erased_device_array_view_t* d_srcs =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(srcs);
  cugraph::c_api::cugraph_type_erased_device_array_view_t* d_dsts =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(dsts);

  CAPI_EXPECTS(h_renumber_map->type_ == d_srcs->type_,
               CUGRAPH_INVALID_INPUT,
               "type of renumber map and src vertices must match",
               *error);

  CAPI_EXPECTS(h_renumber_map->type_ == d_dsts->type_,
               CUGRAPH_INVALID_INPUT,
               "type of renumber map and dst vertices must match",
               *error);

  CAPI_EXPECTS(
    d_srcs->size_ == d_dsts->size_, CUGRAPH_INVALID_INPUT, "src and dst sizes must match", *error);

  *error = nullptr;

  try {
    switch (h_renumber_map->type_) {
      case cugraph_data_type_id_t::INT32: {
        return renumber_arbitrary_edgelist<int32_t>(
          *reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_,
          h_renumber_map,
          d_srcs,
          d_dsts);
      } break;
      case cugraph_data_type_id_t::INT64: {
        return renumber_arbitrary_edgelist<int64_t>(
          *reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_,
          h_renumber_map,
          d_srcs,
          d_dsts);
      } break;
      default: {
        std::stringstream ss;
        ss << "ERROR: Unsupported data type enum:" << static_cast<int>(h_renumber_map->type_);
        *error =
          reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ss.str().c_str()});
        return CUGRAPH_INVALID_INPUT;
      }
    }
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}
