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

#include "prims/kv_store.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>

#include <optional>

namespace cugraph {
namespace test {

template <typename vertex_t>
class nbr_unrenumber_cache_t {
 public:
  nbr_unrenumber_cache_t(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>&& sorted_unique_nbrs,
                         rmm::device_uvector<vertex_t>&& unrenumbered_sorted_unique_nbrs,
                         vertex_t number_of_vertices,
                         vertex_t invalid_vertex)
    : sorted_unique_nbr_segment_offsets_(0, handle.get_stream()),
      sorted_unique_nbr_lower_uint16s_(0, handle.get_stream()),
      unrenumbered_sorted_unique_nbrs_(std::move(unrenumbered_sorted_unique_nbrs)),
      number_of_vertices_(number_of_vertices),
      invalid_vertex_(invalid_vertex)
  {
    init_key_cache(handle, std::move(sorted_unique_nbrs), number_of_vertices);
  }

  void init_key_cache(raft::handle_t const& handle,
                      rmm::device_uvector<vertex_t>&& sorted_unique_nbrs,
                      vertex_t number_of_vertices)
  {
    auto num_segments =
      (static_cast<size_t>(number_of_vertices) + static_cast<size_t>((1 << 16) - 1)) >> 16;
    sorted_unique_nbr_segment_offsets_.resize(num_segments + 1, handle.get_stream());
    auto search_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [] __device__(size_t segment_id) { return static_cast<vertex_t>(segment_id << 16); }));
    thrust::lower_bound(handle.get_thrust_policy(),
                        sorted_unique_nbrs.begin(),
                        sorted_unique_nbrs.end(),
                        search_first,
                        search_first + num_segments,
                        sorted_unique_nbr_segment_offsets_.begin());
    auto array_size = sorted_unique_nbrs.size();
    sorted_unique_nbr_segment_offsets_.set_element_async(
      num_segments, array_size, handle.get_stream());
    sorted_unique_nbr_lower_uint16s_.resize(array_size, handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      sorted_unique_nbrs.begin(),
                      sorted_unique_nbrs.end(),
                      sorted_unique_nbr_lower_uint16s_.begin(),
                      cuda::proclaim_return_type<uint16_t>([] __device__(vertex_t v) {
                        return static_cast<uint16_t>(static_cast<uint64_t>(v));
                      }));
    sorted_unique_nbrs.resize(0, handle.get_stream());
    sorted_unique_nbrs.shrink_to_fit(handle.get_stream());
  }

  void unrenumber(raft::handle_t const& handle, raft::device_span<vertex_t> nbrs)
  {
    thrust::transform(
      handle.get_thrust_policy(),
      nbrs.begin(),
      nbrs.end(),
      nbrs.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [sorted_unique_nbr_segment_offsets = raft::device_span<size_t const>(
           sorted_unique_nbr_segment_offsets_.data(), sorted_unique_nbr_segment_offsets_.size()),
         sorted_unique_nbr_lower_uint16s = raft::device_span<uint16_t const>(
           sorted_unique_nbr_lower_uint16s_.data(), sorted_unique_nbr_lower_uint16s_.size()),
         unrenumbered_sorted_unique_nbrs = raft::device_span<vertex_t const>(
           unrenumbered_sorted_unique_nbrs_.data(), unrenumbered_sorted_unique_nbrs_.size()),
         invalid_vertex = invalid_vertex_] __device__(auto nbr) {
          if (nbr == invalid_vertex) {
            return invalid_vertex;
          } else {
            auto first = sorted_unique_nbr_lower_uint16s.begin() +
                         sorted_unique_nbr_segment_offsets[nbr >> 16];
            auto last = sorted_unique_nbr_lower_uint16s.begin() +
                        sorted_unique_nbr_segment_offsets[(nbr >> 16) + 1];
            auto it = thrust::lower_bound(
              thrust::seq, first, last, static_cast<uint16_t>(static_cast<uint64_t>(nbr)));
            assert((it != last) && (*it == static_cast<uint16_t>(static_cast<uint64_t>(nbr))));
            return unrenumbered_sorted_unique_nbrs[cuda::std::distance(
              sorted_unique_nbr_lower_uint16s.begin(), it)];
          }
        }));
  }

 private:
  rmm::device_uvector<size_t> sorted_unique_nbr_segment_offsets_;
  rmm::device_uvector<uint16_t> sorted_unique_nbr_lower_uint16s_;
  rmm::device_uvector<vertex_t> unrenumbered_sorted_unique_nbrs_;
  vertex_t number_of_vertices_{};
  vertex_t invalid_vertex_{};
};

template <typename vertex_t, typename edge_t>
nbr_unrenumber_cache_t<vertex_t> build_nbr_unrenumber_cache(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_graph_view,
  raft::device_span<vertex_t const> renumber_map,
  vertex_t invalid_vertex,
  std::optional<rmm::host_device_async_resource_ref> pinned_host_mr)
{
#if 1
  pinned_host_mr = std::nullopt;
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "START build_unrenumber_cache range_size="
            << mg_graph_view.local_vertex_partition_range_size() << std::endl;
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
#endif
  auto& comm = handle.get_comms();

  auto h_vertex_partition_range_lasts = mg_graph_view.vertex_partition_range_lasts();
  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
    h_vertex_partition_range_lasts.size(), handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      h_vertex_partition_range_lasts.data(),
                      h_vertex_partition_range_lasts.size(),
                      handle.get_stream());

  constexpr size_t num_k_hop_rounds = 16;  // to cut peak memory usage

#if 1
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1 = std::chrono::steady_clock::now();
#endif
  auto nbrs = pinned_host_mr
                ? rmm::device_uvector<vertex_t>(0, handle.get_stream(), *pinned_host_mr)
                : rmm::device_uvector<vertex_t>(0, handle.get_stream());
  for (size_t r = 0; r < num_k_hop_rounds; ++r) {
    auto range_size = mg_graph_view.local_vertex_partition_range_size();
    auto num_seeds  = range_size / num_k_hop_rounds + (r < (range_size % num_k_hop_rounds) ? 1 : 0);
    rmm::device_uvector<vertex_t> seeds(num_seeds, handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      seeds.begin(),
      seeds.end(),
      cuda::proclaim_return_type<vertex_t>(
        [r, v_first = mg_graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
          return v_first + static_cast<vertex_t>(r + i * num_k_hop_rounds);
        }));

    if (cugraph::host_scalar_allreduce(
          comm, seeds.size(), raft::comms::op_t::SUM, handle.get_stream()) == size_t{0}) {
      continue;
    }

    rmm::device_uvector<vertex_t> this_round_nbrs(0, handle.get_stream());
    std::tie(std::ignore, this_round_nbrs) =
      cugraph::k_hop_nbrs(handle,
                          mg_graph_view,
                          raft::device_span<vertex_t const>(seeds.data(), seeds.size()),
                          size_t{1});
    thrust::sort(handle.get_thrust_policy(), this_round_nbrs.begin(), this_round_nbrs.end());
    this_round_nbrs.resize(
      cuda::std::distance(
        this_round_nbrs.begin(),
        thrust::unique(handle.get_thrust_policy(), this_round_nbrs.begin(), this_round_nbrs.end())),
      handle.get_stream());
    auto old_size = nbrs.size();
    nbrs.resize(old_size + this_round_nbrs.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 this_round_nbrs.begin(),
                 this_round_nbrs.end(),
                 nbrs.begin() + old_size);
  }

#if 1
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time2 = std::chrono::steady_clock::now();
#endif
  thrust::sort(handle.get_thrust_policy(), nbrs.begin(), nbrs.end());
  nbrs.resize(cuda::std::distance(
                nbrs.begin(), thrust::unique(handle.get_thrust_policy(), nbrs.begin(), nbrs.end())),
              handle.get_stream());
  nbrs.shrink_to_fit(handle.get_stream());
#if 1
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time3 = std::chrono::steady_clock::now();
#endif
  auto unrenumbered_nbrs =
    pinned_host_mr
      ? rmm::device_uvector<vertex_t>(nbrs.size(), handle.get_stream(), *pinned_host_mr)
      : rmm::device_uvector<vertex_t>(nbrs.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), nbrs.begin(), nbrs.end(), unrenumbered_nbrs.begin());
  cugraph::unrenumber_int_vertices<vertex_t, true>(handle,
                                                   unrenumbered_nbrs.data(),
                                                   unrenumbered_nbrs.size(),
                                                   renumber_map.data(),
                                                   mg_graph_view.vertex_partition_range_lasts());
#if 1
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time4                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::chrono::duration<double> dur1 = time2 - time1;
  std::chrono::duration<double> dur2 = time3 - time2;
  std::chrono::duration<double> dur3 = time4 - time3;
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "END build_unrenumber_cache nbrs.size()=" << nbrs.size() << " dur=(" << dur0.count()
            << "," << dur1.count() << "," << dur2.count() << "," << dur3.count() << ")"
            << std::endl;
#endif

  return nbr_unrenumber_cache_t(handle,
                                std::move(nbrs),
                                std::move(unrenumbered_nbrs),
                                mg_graph_view.number_of_vertices(),
                                invalid_vertex);
}

}  // namespace test
}  // namespace cugraph
