/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/variant/edge_properties.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  raft::device_span<vertex_t> edgelist_majors,
  raft::device_span<vertex_t> edgelist_minors,
  raft::host_span<cugraph::variant::device_spans_t> edgelist_properties,
  bool groupby_and_count_local_partition_by_minor)
{
  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto const comm_rank       = comm.get_rank();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto const major_comm_rank = major_comm.get_rank();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();
  auto const minor_comm_rank = minor_comm.get_rank();

  size_t element_size = sizeof(vertex_t) * 2;

  if (edgelist_properties.size() == 1) {
    element_size += cugraph::variant::variant_type_dispatch(edgelist_properties[0],
                                                            cugraph::variant::variant_size{});
  } else if (edgelist_properties.size() > 1) {
    element_size += sizeof(size_t);
  }

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
          // group-by by default, and thrust::sort requires temporary buffer comparable to the input
          // data size)
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  auto pair_first =
    thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));

  rmm::device_uvector<size_t> result(0, handle.get_stream());

  auto local_edge_partition_include_minor_op =
    [major_comm_size,
     local_edge_partition_id_key_func =
       cugraph::detail::compute_local_edge_partition_id_from_ext_edge_endpoints_t<vertex_t>{
         comm_size, major_comm_size, minor_comm_size},
     vertex_partition_id_key_func =
       cugraph::detail::compute_vertex_partition_id_from_ext_vertex_t<vertex_t>{
         comm_size}] __device__(auto pair) {
      auto local_edge_partition_id = local_edge_partition_id_key_func(pair);
      auto vertex_partition_id     = vertex_partition_id_key_func(thrust::get<1>(pair));
      return (local_edge_partition_id * major_comm_size) +
             ((vertex_partition_id) % major_comm_size);
    };

  auto local_edge_partition_op =
    [key_func =
       cugraph::detail::compute_local_edge_partition_id_from_ext_edge_endpoints_t<vertex_t>{
         comm_size, major_comm_size, minor_comm_size}] __device__(auto pair) {
      return key_func(pair);
    };

  if (edgelist_properties.size() == 0) {
    if (groupby_and_count_local_partition_by_minor) {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + edgelist_majors.size(),
                                          local_edge_partition_include_minor_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    } else {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + edgelist_majors.size(),
                                          local_edge_partition_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    }
  } else if (edgelist_properties.size() == 1) {
    result = cugraph::variant::variant_type_dispatch(
      edgelist_properties[0],
      [&handle,
       &pair_first,
       size = edgelist_majors.size(),
       &local_edge_partition_include_minor_op,
       local_edge_partition_op,
       comm_size,
       mem_frugal_threshold,
       groupby_and_count_local_partition_by_minor](auto& prop) {
        if (groupby_and_count_local_partition_by_minor) {
          return cugraph::groupby_and_count(pair_first,
                                            pair_first + size,
                                            prop.begin(),
                                            local_edge_partition_include_minor_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
        } else {
          return cugraph::groupby_and_count(pair_first,
                                            pair_first + size,
                                            prop.begin(),
                                            local_edge_partition_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
        }
      });
  } else {
    rmm::device_uvector<size_t> property_position(edgelist_majors.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), size_t{0});

    if (groupby_and_count_local_partition_by_minor) {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + edgelist_majors.size(),
                                          property_position.begin(),
                                          local_edge_partition_include_minor_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    } else {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + edgelist_majors.size(),
                                          property_position.begin(),
                                          local_edge_partition_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    }

    std::for_each(edgelist_properties.begin(),
                  edgelist_properties.end(),
                  [&property_position, &handle](auto& property) {
                    cugraph::variant::variant_type_dispatch(
                      property, [&handle, &property_position](auto& prop) {
                        using T = typename std::remove_reference<decltype(prop)>::type::value_type;
                        rmm::device_uvector<T> tmp(prop.size(), handle.get_stream());

                        thrust::gather(handle.get_thrust_policy(),
                                       property_position.begin(),
                                       property_position.end(),
                                       prop.begin(),
                                       tmp.begin());

                        thrust::copy(
                          handle.get_thrust_policy(), tmp.begin(), tmp.end(), prop.begin());
                      });
                  });
  }

  return result;
}

}  // namespace detail
}  // namespace cugraph
