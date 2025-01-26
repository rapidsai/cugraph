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
#include "cugraph/graph.hpp"
#include "cugraph/utilities/error.hpp"
#include "cugraph_c/error.h"
#include "thrust/binary_search.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/zip_iterator.h"

#include <cugraph_c/graph_functions.h>

namespace {

template <typename vertex_t>
cugraph_error_code_t renumber_arbitrary_edgelist(
  raft::handle_t const& handle,
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* renumber_map,
  cugraph::c_api::cugraph_type_erased_device_array_view_t* srcs,
  cugraph::c_api::cugraph_type_erased_device_array_view_t* dsts)
{
  // Create a sorted representation of each vertex id and where it exists in the input array
  rmm::device_uvector<vertex_t> srcs_v(srcs->size_, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts_v(dsts->size_, handle.get_stream());
  rmm::device_uvector<size_t> srcs_pos(srcs->size_, handle.get_stream());
  rmm::device_uvector<size_t> dsts_pos(dsts->size_, handle.get_stream());

  thrust::copy_n(handle.get_thrust_policy(), srcs->as_type<vertex_t>(), srcs->size_, srcs_v.data());
  thrust::copy_n(handle.get_thrust_policy(), dsts->as_type<vertex_t>(), dsts->size_, dsts_v.data());
  thrust::sequence(handle.get_thrust_policy(), srcs_pos.begin(), srcs_pos.end(), size_t{0});
  thrust::sequence(handle.get_thrust_policy(), dsts_pos.begin(), dsts_pos.end(), size_t{0});

  thrust::sort(handle.get_thrust_policy(),
               thrust::make_zip_iterator(srcs_v.begin(), srcs_pos.begin()),
               thrust::make_zip_iterator(srcs_v.end(), srcs_pos.end()));

  thrust::sort(handle.get_thrust_policy(),
               thrust::make_zip_iterator(dsts_v.begin(), dsts_pos.begin()),
               thrust::make_zip_iterator(dsts_v.end(), dsts_pos.end()));

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

    rmm::device_uvector<vertex_t> renumbered_values(srcs_v.size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(),
                 renumbered_values.begin(),
                 renumbered_values.end(),
                 cugraph::invalid_vertex_id<vertex_t>::value);

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(renumber_chunk.size()),
      [chunk_base_offset,
       renumber_chunk_span =
         raft::device_span<vertex_t const>{renumber_chunk.data(), renumber_chunk.size()},
       srcs_span        = raft::device_span<vertex_t>{srcs_v.data(), srcs_v.size()},
       srcs_pos_span    = raft::device_span<size_t const>{srcs_pos.data(), srcs_pos.size()},
       dsts_span        = raft::device_span<vertex_t>{dsts_v.data(), dsts_v.size()},
       dsts_pos_span    = raft::device_span<size_t const>{dsts_pos.data(), dsts_pos.size()},
       output_srcs_span = raft::device_span<vertex_t>{srcs->as_type<vertex_t>(), srcs->size_},
       output_dsts_span = raft::device_span<vertex_t>{dsts->as_type<vertex_t>(),
                                                      dsts->size_}] __device__(size_t idx) {
        vertex_t old_vertex_id = renumber_chunk_span[idx];
        vertex_t new_vertex_id = static_cast<vertex_t>(chunk_base_offset + idx);

        auto begin_iter =
          thrust::lower_bound(thrust::seq, srcs_span.begin(), srcs_span.end(), old_vertex_id);
        if (begin_iter != srcs_span.end()) {
          auto end_iter =
            thrust::upper_bound(thrust::seq, srcs_span.begin(), srcs_span.end(), old_vertex_id);

          while (begin_iter != end_iter) {
            size_t offset = thrust::distance(srcs_span.begin(), begin_iter);
            output_srcs_span[srcs_pos_span[offset]] = new_vertex_id;
            srcs_span[offset]                       = cugraph::invalid_vertex_id<vertex_t>();
            ++begin_iter;
          }
        }

        begin_iter =
          thrust::lower_bound(thrust::seq, dsts_span.begin(), dsts_span.end(), old_vertex_id);
        if (begin_iter != dsts_span.end()) {
          auto end_iter =
            thrust::upper_bound(thrust::seq, dsts_span.begin(), dsts_span.end(), old_vertex_id);

          while (begin_iter != end_iter) {
            size_t offset = thrust::distance(dsts_span.begin(), begin_iter);
            output_dsts_span[dsts_pos_span[offset]] = new_vertex_id;
            dsts_span[offset]                       = cugraph::invalid_vertex_id<vertex_t>();
            ++begin_iter;
          }
        }
      });

    srcs_v.resize(thrust::distance(
                    thrust::make_zip_iterator(srcs_v.begin(), srcs_pos.begin()),
                    thrust::remove_if(handle.get_thrust_policy(),
                                      thrust::make_zip_iterator(srcs_v.begin(), srcs_pos.begin()),
                                      thrust::make_zip_iterator(srcs_v.end(), srcs_pos.end()),
                                      [] __device__(auto t) {
                                        return thrust::get<0>(t) ==
                                               cugraph::invalid_vertex_id<vertex_t>();
                                      })),
                  handle.get_stream());
    srcs_pos.resize(srcs_v.size(), handle.get_stream());

    dsts_v.resize(thrust::distance(
                    thrust::make_zip_iterator(dsts_v.begin(), dsts_pos.begin()),
                    thrust::remove_if(handle.get_thrust_policy(),
                                      thrust::make_zip_iterator(dsts_v.begin(), dsts_pos.begin()),
                                      thrust::make_zip_iterator(dsts_v.end(), dsts_pos.end()),
                                      [] __device__(auto t) {
                                        return thrust::get<0>(t) ==
                                               cugraph::invalid_vertex_id<vertex_t>();
                                      })),
                  handle.get_stream());
    dsts_pos.resize(dsts_v.size(), handle.get_stream());
  }

  CUGRAPH_EXPECTS(srcs_v.size() == 0, "some src vertices were not renumbered");
  CUGRAPH_EXPECTS(dsts_v.size() == 0, "some dst vertices were not renumbered");

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
