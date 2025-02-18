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

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_edgelist_weights,
  std::optional<rmm::device_uvector<edge_t>>& d_edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>& d_edgelist_edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>& d_edgelist_edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>& d_edgelist_edge_end_times,
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

  int edge_property_count = 0;
  size_t element_size     = sizeof(vertex_t) * 2;

  if (d_edgelist_weights) {
    ++edge_property_count;
    element_size += sizeof(weight_t);
  }

  if (d_edgelist_edge_ids) {
    ++edge_property_count;
    element_size += sizeof(edge_t);
  }
  if (d_edgelist_edge_types) {
    ++edge_property_count;
    element_size += sizeof(edge_type_t);
  }
  if (d_edgelist_edge_start_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }
  if (d_edgelist_edge_end_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }

  if (edge_property_count > 1) { element_size = sizeof(vertex_t) * 2 + sizeof(size_t); }

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
          // group-by by default, and thrust::sort requires temporary buffer comparable to the input
          // data size)
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(d_edgelist_majors.begin(), d_edgelist_minors.begin()));

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

  if (edge_property_count == 0) {
    if (groupby_and_count_local_partition_by_minor) {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + d_edgelist_majors.size(),
                                          local_edge_partition_include_minor_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    } else {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + d_edgelist_majors.size(),
                                          local_edge_partition_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    }

  } else if (edge_property_count == 1) {
    if (d_edgelist_weights) {
      if (groupby_and_count_local_partition_by_minor) {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_weights->begin(),
                                            local_edge_partition_include_minor_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      } else {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_weights->begin(),
                                            local_edge_partition_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      }
    } else if (d_edgelist_edge_ids) {
      if (groupby_and_count_local_partition_by_minor) {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_ids->begin(),
                                            local_edge_partition_include_minor_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      } else {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_ids->begin(),
                                            local_edge_partition_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      }
    } else if (d_edgelist_edge_types) {
      if (groupby_and_count_local_partition_by_minor) {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_types->begin(),
                                            local_edge_partition_include_minor_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      } else {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_types->begin(),
                                            local_edge_partition_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      }
    } else if (d_edgelist_edge_start_times) {
      if (groupby_and_count_local_partition_by_minor) {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_start_times->begin(),
                                            local_edge_partition_include_minor_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      } else {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_start_times->begin(),
                                            local_edge_partition_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      }
    } else if (d_edgelist_edge_end_times) {
      if (groupby_and_count_local_partition_by_minor) {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_end_times->begin(),
                                            local_edge_partition_include_minor_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      } else {
        result = cugraph::groupby_and_count(pair_first,
                                            pair_first + d_edgelist_majors.size(),
                                            d_edgelist_edge_end_times->begin(),
                                            local_edge_partition_op,
                                            comm_size,
                                            mem_frugal_threshold,
                                            handle.get_stream());
      }
    }
  } else {
    rmm::device_uvector<edge_t> property_position(d_edgelist_majors.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), edge_t{0});

    if (groupby_and_count_local_partition_by_minor) {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + d_edgelist_majors.size(),
                                          property_position.begin(),
                                          local_edge_partition_include_minor_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    } else {
      result = cugraph::groupby_and_count(pair_first,
                                          pair_first + d_edgelist_majors.size(),
                                          property_position.begin(),
                                          local_edge_partition_op,
                                          comm_size,
                                          mem_frugal_threshold,
                                          handle.get_stream());
    }

    if (d_edgelist_weights) {
      rmm::device_uvector<weight_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     d_edgelist_weights->begin(),
                     tmp.begin());

      d_edgelist_weights = std::move(tmp);
    }

    if (d_edgelist_edge_ids) {
      rmm::device_uvector<edge_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     d_edgelist_edge_ids->begin(),
                     tmp.begin());

      d_edgelist_edge_ids = std::move(tmp);
    }

    if (d_edgelist_edge_types) {
      rmm::device_uvector<edge_type_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     d_edgelist_edge_types->begin(),
                     tmp.begin());

      d_edgelist_edge_types = std::move(tmp);
    }

    if (d_edgelist_edge_start_times) {
      rmm::device_uvector<edge_time_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     d_edgelist_edge_start_times->begin(),
                     tmp.begin());

      d_edgelist_edge_start_times = std::move(tmp);
    }

    if (d_edgelist_edge_end_times) {
      rmm::device_uvector<edge_time_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     d_edgelist_edge_end_times->begin(),
                     tmp.begin());

      d_edgelist_edge_end_times = std::move(tmp);
    }
  }

  return result;
}

}  // namespace detail
}  // namespace cugraph
