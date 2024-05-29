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
#include "detail/graph_partition_utils.cuh"
#include "prims/kv_store.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/src_dst_lookup_container.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/handle.hpp>

namespace cugraph {

template <typename edge_id_t, typename edge_type_t, typename value_t>
template <typename _edge_id_t, typename _edge_type_t, typename _value_t>
struct lookup_container_t<edge_id_t, edge_type_t, value_t>::lookup_container_impl {
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);

  static_assert(std::is_same_v<edge_type_t, _edge_type_t>);
  static_assert(std::is_same_v<edge_id_t, _edge_id_t>);
  static_assert(std::is_same_v<value_t, _value_t>);

  ~lookup_container_impl() {}
  lookup_container_impl() {}
  lookup_container_impl(raft::handle_t const& handle,
                        std::vector<edge_type_t> types,
                        std::vector<edge_id_t> type_counts)
  {
    auto invalid_key   = cugraph::invalid_vertex_id<edge_id_t>::value;
    auto invalid_value = invalid_of_thrust_tuple_of_integral<value_t>();

    edge_type_to_kv_store = container_t{};
    edge_type_to_kv_store.reserve(types.size());

    for (size_t idx = 0; idx < types.size(); idx++) {
      auto typ              = types[idx];
      size_t store_capacity = type_counts[idx];

      edge_type_to_kv_store.insert(
        {typ, std::move(store_t(store_capacity, invalid_key, invalid_value, handle.get_stream()))});
    }
  }

  void insert(raft::handle_t const& handle,
              edge_type_t type,
              raft::device_span<edge_id_t const> edge_ids_to_insert,
              decltype(cugraph::allocate_dataframe_buffer<value_t>(
                0, rmm::cuda_stream_view{}))&& values_to_insert)
  {
    auto itr = edge_type_to_kv_store.find(type);

    if (itr != edge_type_to_kv_store.end()) {
      assert(itr->first == type);

      itr->second.insert(edge_ids_to_insert.begin(),
                         edge_ids_to_insert.end(),
                         cugraph::get_dataframe_buffer_begin(values_to_insert),
                         handle.get_stream());

    } else {
      assert(false);
    }
  }

  dataframe_buffer_type_t<value_t> src_dst_from_edge_id_and_type(
    raft::handle_t const& handle,
    raft::device_span<edge_id_t const> edge_ids_to_lookup,
    edge_type_t edge_type_to_lookup,
    bool multi_gpu) const
  {
    using store_t = typename container_t::mapped_type;
    const store_t* kv_store_object{nullptr};

    auto value_buffer = cugraph::allocate_dataframe_buffer<value_t>(0, handle.get_stream());
    auto itr          = edge_type_to_kv_store.find(edge_type_to_lookup);

    if (itr != edge_type_to_kv_store.end()) {
      assert(edge_type_to_lookup == itr->first);
      kv_store_object = &(itr->second);

    } else {
      cugraph::resize_dataframe_buffer(
        value_buffer, edge_ids_to_lookup.size(), handle.get_stream());

      auto invalid_value = invalid_of_thrust_tuple_of_integral<value_t>();

      thrust::fill(handle.get_thrust_policy(),
                   cugraph::get_dataframe_buffer_begin(value_buffer),
                   cugraph::get_dataframe_buffer_end(value_buffer),
                   invalid_value);

      // FIXME: we are assuming value_t is tuple of size 2
      return std::make_tuple(std::move(std::get<0>(value_buffer)),
                             std::move(std::get<1>(value_buffer)));
    }

    if (multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto key_func = cugraph::detail::compute_gpu_id_from_ext_vertex_t<edge_id_t>{
        comm_size, major_comm_size, minor_comm_size};

      value_buffer = cugraph::collect_values_for_keys(handle,
                                                      kv_store_object->view(),
                                                      edge_ids_to_lookup.begin(),
                                                      edge_ids_to_lookup.end(),
                                                      key_func);

    } else {
      cugraph::resize_dataframe_buffer(
        value_buffer, edge_ids_to_lookup.size(), handle.get_stream());

      kv_store_object->view().find(edge_ids_to_lookup.begin(),
                                   edge_ids_to_lookup.end(),
                                   cugraph::get_dataframe_buffer_begin(value_buffer),
                                   handle.get_stream());
    }

    // FIXME: we are assuming value_t is tuple of size 2
    return std::make_tuple(std::move(std::get<0>(value_buffer)),
                           std::move(std::get<1>(value_buffer)));
  }

  dataframe_buffer_type_t<value_t> src_dst_from_edge_id_and_type(
    raft::handle_t const& handle,
    raft::device_span<edge_id_t const> edge_ids_to_lookup,
    raft::device_span<edge_type_t const> edge_types_to_lookup,
    bool multi_gpu) const
  {
    static_assert(std::is_integral_v<edge_id_t>);
    static_assert(std::is_integral_v<edge_type_t>);

    assert(edge_ids_to_lookup.size() == edge_types_to_lookup.size());

    rmm::device_uvector<edge_id_t> tmp_edge_ids_to_lookup(edge_ids_to_lookup.size(),
                                                          handle.get_stream());

    rmm::device_uvector<edge_type_t> tmp_edge_types_to_lookup(edge_types_to_lookup.size(),
                                                              handle.get_stream());

    rmm::device_uvector<edge_id_t> original_idxs(edge_ids_to_lookup.size(), handle.get_stream());

    thrust::sequence(
      handle.get_thrust_policy(), original_idxs.begin(), original_idxs.end(), edge_id_t{0});

    thrust::copy(handle.get_thrust_policy(),
                 edge_ids_to_lookup.begin(),
                 edge_ids_to_lookup.end(),
                 tmp_edge_ids_to_lookup.begin());

    thrust::copy(handle.get_thrust_policy(),
                 edge_types_to_lookup.begin(),
                 edge_types_to_lookup.end(),
                 tmp_edge_types_to_lookup.begin());

    thrust::sort_by_key(handle.get_thrust_policy(),
                        tmp_edge_types_to_lookup.begin(),
                        tmp_edge_types_to_lookup.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(tmp_edge_ids_to_lookup.begin(),
                                                                     original_idxs.begin())));

    auto nr_uniqe_edge_types_to_lookup = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(tmp_edge_types_to_lookup.size()),
      detail::is_first_in_run_t<edge_type_t const*>{tmp_edge_types_to_lookup.data()});

    rmm::device_uvector<edge_type_t> unique_types(nr_uniqe_edge_types_to_lookup,
                                                  handle.get_stream());
    rmm::device_uvector<edge_id_t> type_offsets(nr_uniqe_edge_types_to_lookup + 1,
                                                handle.get_stream());

    thrust::fill(
      handle.get_thrust_policy(), type_offsets.begin(), type_offsets.end(), edge_id_t{0});

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          tmp_edge_types_to_lookup.begin(),
                          tmp_edge_types_to_lookup.end(),
                          thrust::make_constant_iterator(size_t{1}),
                          unique_types.begin(),
                          type_offsets.begin());

    thrust::exclusive_scan(handle.get_thrust_policy(),
                           type_offsets.begin(),
                           type_offsets.end(),
                           type_offsets.begin(),
                           size_t{0});

    std::vector<edge_type_t> h_unique_types(unique_types.size());
    std::vector<edge_id_t> h_type_offsets(type_offsets.size());

    raft::update_host(
      h_unique_types.data(), unique_types.data(), unique_types.size(), handle.get_stream());

    raft::update_host(
      h_type_offsets.data(), type_offsets.data(), type_offsets.size(), handle.get_stream());

    std::unordered_map<edge_type_t, int> typ_to_local_idx_map{};

    handle.sync_stream();

    for (size_t idx = 0; idx < h_unique_types.size(); idx++) {
      typ_to_local_idx_map[h_unique_types[idx]] = idx;
    }

    auto output_value_buffer =
      cugraph::allocate_dataframe_buffer<value_t>(edge_ids_to_lookup.size(), handle.get_stream());
    auto invalid_value = invalid_of_thrust_tuple_of_integral<value_t>();
    thrust::fill(handle.get_thrust_policy(),
                 cugraph::get_dataframe_buffer_begin(output_value_buffer),
                 cugraph::get_dataframe_buffer_begin(output_value_buffer),
                 invalid_value);

    if (multi_gpu) {
      auto& comm     = handle.get_comms();
      auto rx_counts = host_scalar_allgather(comm, unique_types.size(), handle.get_stream());
      std::vector<size_t> rx_displacements(rx_counts.size());
      std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
      rmm::device_uvector<edge_type_t> rx_unique_types(rx_displacements.back() + rx_counts.back(),
                                                       handle.get_stream());

      device_allgatherv(comm,
                        unique_types.begin(),
                        rx_unique_types.begin(),
                        rx_counts,
                        rx_displacements,
                        handle.get_stream());
      unique_types = std::move(rx_unique_types);

      thrust::sort(handle.get_thrust_policy(), unique_types.begin(), unique_types.end());

      unique_types.resize(
        thrust::distance(
          unique_types.begin(),
          thrust::unique(handle.get_thrust_policy(), unique_types.begin(), unique_types.end())),
        handle.get_stream());
    }

    h_unique_types.resize(unique_types.size());
    raft::update_host(
      h_unique_types.data(), unique_types.data(), unique_types.size(), handle.get_stream());

    for (size_t idx = 0; idx < h_unique_types.size(); idx++) {
      auto typ = h_unique_types[idx];

      auto tmp_edge_ids_begin = tmp_edge_ids_to_lookup.begin();
      size_t span_size        = 0;

      if (typ_to_local_idx_map.find(typ) != typ_to_local_idx_map.end()) {
        auto local_idx     = typ_to_local_idx_map[typ];
        tmp_edge_ids_begin = tmp_edge_ids_to_lookup.begin() + h_type_offsets[local_idx];
        span_size          = h_type_offsets[local_idx + 1] - h_type_offsets[local_idx];
      }

      auto value_buffer_typ = src_dst_from_edge_id_and_type(
        handle, raft::device_span<edge_id_t const>{tmp_edge_ids_begin, span_size}, typ, multi_gpu);

      if (typ_to_local_idx_map.find(typ) != typ_to_local_idx_map.end()) {
        thrust::copy(handle.get_thrust_policy(),
                     cugraph::get_dataframe_buffer_begin(value_buffer_typ),
                     cugraph::get_dataframe_buffer_end(value_buffer_typ),
                     cugraph::get_dataframe_buffer_begin(output_value_buffer) +
                       h_type_offsets[typ_to_local_idx_map[typ]]);
      }
    }

    thrust::sort_by_key(handle.get_thrust_policy(),
                        original_idxs.begin(),
                        original_idxs.end(),
                        cugraph::get_dataframe_buffer_begin(output_value_buffer));

    // FIXME: we are assuming value_t is tuple of size 2
    return std::make_tuple(std::move(std::get<0>(output_value_buffer)),
                           std::move(std::get<1>(output_value_buffer)));
  }

 private:
  using container_t =
    std::unordered_map<edge_type_t,
                       cugraph::kv_store_t<edge_id_t, value_t, false /*use_binary_search*/>>;
  using store_t = typename container_t::mapped_type;
  container_t edge_type_to_kv_store;
};

template <typename edge_id_t, typename edge_type_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, value_t>::~lookup_container_t()
{
  pimpl.reset();
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, value_t>::lookup_container_t()
  : pimpl{std::make_unique<lookup_container_impl<edge_id_t, edge_type_t, value_t>>()}
{
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, value_t>::lookup_container_t(
  raft::handle_t const& handle, std::vector<edge_type_t> types, std::vector<edge_id_t> type_counts)
  : pimpl{std::make_unique<lookup_container_impl<edge_id_t, edge_type_t, value_t>>(
      handle, types, type_counts)}
{
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, value_t>::lookup_container_t(const lookup_container_t&)
{
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
void lookup_container_t<edge_id_t, edge_type_t, value_t>::insert(
  raft::handle_t const& handle,
  edge_type_t type,
  raft::device_span<edge_id_t const> edge_ids_to_insert,
  decltype(cugraph::allocate_dataframe_buffer<value_t>(0,
                                                       rmm::cuda_stream_view{}))&& values_to_insert)
{
  pimpl->insert(handle, type, edge_ids_to_insert, std::move(values_to_insert));
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
dataframe_buffer_type_t<value_t>
lookup_container_t<edge_id_t, edge_type_t, value_t>::src_dst_from_edge_id_and_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup,
  bool multi_gpu) const
{
  return pimpl->src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_type_to_lookup, multi_gpu);
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
dataframe_buffer_type_t<value_t>
lookup_container_t<edge_id_t, edge_type_t, value_t>::src_dst_from_edge_id_and_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup,
  bool multi_gpu) const
{
  return pimpl->src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_types_to_lookup, multi_gpu);
}

namespace detail {

template <typename GraphViewType,
          typename EdgeIdInputWrapper,
          typename EdgeTypeInputWrapper,
          typename EdgeTypeAndIdToSrcDstLookupContainerType>
EdgeTypeAndIdToSrcDstLookupContainerType build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeIdInputWrapper edge_id_view,
  EdgeTypeInputWrapper edge_type_view)
{
  static_assert(!std::is_same_v<typename EdgeIdInputWrapper::value_type, thrust::nullopt_t>,
                "Can not create edge id lookup table without edge ids");

  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using edge_type_t = typename EdgeTypeInputWrapper::value_type;
  using edge_id_t   = typename EdgeIdInputWrapper::value_type;
  using value_t     = typename EdgeTypeAndIdToSrcDstLookupContainerType::value_type;

  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_same_v<value_t, thrust::tuple<vertex_t, vertex_t>>);

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type, edge_type_t>,
    "edge_type_t must match with EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type");

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type, edge_id_t>,
    "edge_id_t must match with typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type");

  std::vector<edge_type_t> h_types_to_this_gpu{};
  std::vector<edge_t> h_freq_of_types_to_this_gpu{};
  std::unordered_map<edge_type_t, edge_t> edge_type_to_count_map{};

  for (size_t local_ep_idx = 0; local_ep_idx < graph_view.number_of_local_edge_partitions();
       ++local_ep_idx) {
    //
    //  Copy edge ids and types
    //

    auto number_of_edges_partition =
      graph_view.local_edge_partition_view(local_ep_idx).number_of_edges();
    auto number_of_active_edges_partition = number_of_edges_partition;

    if (graph_view.has_edge_mask()) {
      number_of_active_edges_partition =
        detail::count_set_bits(handle,
                               (*(graph_view.edge_mask_view())).value_firsts()[local_ep_idx],
                               number_of_edges_partition);
    }

    [[maybe_unused]] auto edgelist_ids =
      rmm::device_uvector<edge_t>(number_of_active_edges_partition, handle.get_stream());

    auto edgelist_types =
      rmm::device_uvector<edge_type_t>(number_of_active_edges_partition, handle.get_stream());

    auto edge_partition_mask_view =
      graph_view.has_edge_mask()
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *(graph_view.edge_mask_view()), local_ep_idx)
        : std::nullopt;

    if (edge_partition_mask_view) {
      if constexpr (multi_gpu) {
        detail::copy_if_mask_set(
          handle,
          edge_id_view.value_firsts()[local_ep_idx],
          edge_id_view.value_firsts()[local_ep_idx] + number_of_edges_partition,
          (*edge_partition_mask_view).value_first(),
          edgelist_ids.begin());
      }
      detail::copy_if_mask_set(
        handle,
        edge_type_view.value_firsts()[local_ep_idx],
        edge_type_view.value_firsts()[local_ep_idx] + number_of_edges_partition,
        (*edge_partition_mask_view).value_first(),
        edgelist_types.begin());

    } else {
      if constexpr (multi_gpu) {
        thrust::copy(handle.get_thrust_policy(),
                     edge_id_view.value_firsts()[local_ep_idx],
                     edge_id_view.value_firsts()[local_ep_idx] + number_of_edges_partition,
                     edgelist_ids.begin());
      }
      thrust::copy(handle.get_thrust_policy(),
                   edge_type_view.value_firsts()[local_ep_idx],
                   edge_type_view.value_firsts()[local_ep_idx] + number_of_edges_partition,
                   edgelist_types.begin());
    }

    std::vector<int> h_unique_gpu_ids{};
    std::vector<edge_t> h_gpu_offsets{};

    if constexpr (multi_gpu) {
      //
      // Count number of edge ids mapped to each GPU
      //

      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      // destination gpu id from edge id. NB: edgelist_ids will holds gpu ids after the following
      // thrust::transform
      thrust::transform(handle.get_thrust_policy(),
                        edgelist_ids.begin(),
                        edgelist_ids.end(),
                        edgelist_ids.begin(),
                        [key_func = cugraph::detail::compute_gpu_id_from_ext_vertex_t<edge_t>{
                           comm_size, major_comm_size, minor_comm_size}] __device__(auto eid) {
                          return key_func(eid);
                        });

      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_ids.begin(), edgelist_types.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_ids.end(), edgelist_types.end())));

      auto nr_unique_gpu_ids =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(edgelist_ids.size()),
                         detail::is_first_in_run_t<edge_t const*>{edgelist_ids.data()});

      rmm::device_uvector<int> unique_gpu_ids(nr_unique_gpu_ids, handle.get_stream());
      rmm::device_uvector<edge_t> gpu_offsets(nr_unique_gpu_ids + 1, handle.get_stream());

      thrust::fill(handle.get_thrust_policy(), gpu_offsets.begin(), gpu_offsets.end(), edge_t{0});

      thrust::reduce_by_key(handle.get_thrust_policy(),
                            edgelist_ids.begin(),
                            edgelist_ids.end(),
                            thrust::make_constant_iterator(size_t{1}),
                            unique_gpu_ids.begin(),
                            gpu_offsets.begin());

      thrust::exclusive_scan(handle.get_thrust_policy(),
                             gpu_offsets.begin(),
                             gpu_offsets.end(),
                             gpu_offsets.begin(),
                             size_t{0});

      h_unique_gpu_ids.resize(unique_gpu_ids.size());
      h_gpu_offsets.resize(gpu_offsets.size());
      raft::update_host(
        h_unique_gpu_ids.data(), unique_gpu_ids.data(), unique_gpu_ids.size(), handle.get_stream());
      raft::update_host(
        h_gpu_offsets.data(), gpu_offsets.data(), gpu_offsets.size(), handle.get_stream());

    } else {
      thrust::sort(handle.get_thrust_policy(), edgelist_types.begin(), edgelist_types.end());

      h_unique_gpu_ids.resize(size_t{1});
      h_unique_gpu_ids[0] = 0;

      h_gpu_offsets.resize(h_unique_gpu_ids.size() + 1);
      h_gpu_offsets[0] = 0;
      h_gpu_offsets[1] = edgelist_types.size();
    }

    //
    // For edge ids mapped to each gpu, count number of unique types and elements per type.
    // cub::DeviceSegmentedReduce(ByKey) ???
    //

    [[maybe_unused]] std::vector<int> h_gpu_ids_partition{};
    std::vector<edge_type_t> h_types_partition{};
    std::vector<edge_t> h_type_freqs_partition{};

    rmm::device_uvector<edge_type_t> unique_types_segment(0, handle.get_stream());
    rmm::device_uvector<edge_t> type_freqs_segment(0, handle.get_stream());

    for (size_t idx = 0; idx < h_unique_gpu_ids.size(); ++idx) {
      auto gpu_id = h_unique_gpu_ids[idx];

      auto nr_uniqe_types_segment = thrust::count_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(
          static_cast<size_t>(h_gpu_offsets[idx + 1] - h_gpu_offsets[idx])),
        detail::is_first_in_run_t<edge_type_t const*>{edgelist_types.data() + h_gpu_offsets[idx]});

      unique_types_segment.resize(nr_uniqe_types_segment, handle.get_stream());
      type_freqs_segment.resize(nr_uniqe_types_segment, handle.get_stream());

      thrust::reduce_by_key(handle.get_thrust_policy(),
                            edgelist_types.begin() + h_gpu_offsets[idx],
                            edgelist_types.begin() + h_gpu_offsets[idx + 1],
                            thrust::make_constant_iterator(size_t{1}),
                            unique_types_segment.begin(),
                            type_freqs_segment.begin());

      std::vector<edge_type_t> h_unique_types_segment(nr_uniqe_types_segment);
      std::vector<edge_t> h_type_freqs_segment(nr_uniqe_types_segment);

      raft::update_host(h_unique_types_segment.data(),
                        unique_types_segment.data(),
                        unique_types_segment.size(),
                        handle.get_stream());
      raft::update_host(h_type_freqs_segment.data(),
                        type_freqs_segment.data(),
                        type_freqs_segment.size(),
                        handle.get_stream());

      if constexpr (multi_gpu) {
        h_gpu_ids_partition.insert(h_gpu_ids_partition.end(), nr_uniqe_types_segment, gpu_id);
      }

      h_types_partition.insert(
        h_types_partition.end(), h_unique_types_segment.begin(), h_unique_types_segment.end());
      h_type_freqs_partition.insert(
        h_type_freqs_partition.end(), h_type_freqs_segment.begin(), h_type_freqs_segment.end());
    }

    [[maybe_unused]] rmm::device_uvector<int> gpu_ids_partition(h_gpu_ids_partition.size(),
                                                                handle.get_stream());
    rmm::device_uvector<edge_type_t> types_partition(h_types_partition.size(), handle.get_stream());
    rmm::device_uvector<edge_t> type_freqs_partition(h_type_freqs_partition.size(),
                                                     handle.get_stream());

    if constexpr (multi_gpu) {
      raft::update_device(gpu_ids_partition.data(),
                          h_gpu_ids_partition.data(),
                          h_gpu_ids_partition.size(),
                          handle.get_stream());
    }
    raft::update_device(types_partition.data(),
                        h_types_partition.data(),
                        h_types_partition.size(),
                        handle.get_stream());
    raft::update_device(type_freqs_partition.data(),
                        h_type_freqs_partition.data(),
                        h_type_freqs_partition.size(),
                        handle.get_stream());

    if constexpr (multi_gpu) {
      // Shuffle to the proper GPUs
      std::forward_as_tuple(std::tie(gpu_ids_partition, types_partition, type_freqs_partition),
                            std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          thrust::make_zip_iterator(thrust::make_tuple(
            gpu_ids_partition.begin(), types_partition.begin(), type_freqs_partition.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(
            gpu_ids_partition.end(), types_partition.end(), type_freqs_partition.end())),
          [] __device__(auto val) { return thrust::get<0>(val); },
          handle.get_stream());

      thrust::for_each(
        handle.get_thrust_policy(),
        gpu_ids_partition.begin(),
        gpu_ids_partition.end(),
        [rank = handle.get_comms().get_rank()] __device__(auto val) { assert(val == rank); });
    }

    thrust::sort_by_key(handle.get_thrust_policy(),
                        types_partition.begin(),
                        types_partition.end(),
                        type_freqs_partition.begin());

    auto nr_uniqe_types_partition =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(types_partition.size()),
                       detail::is_first_in_run_t<edge_type_t const*>{types_partition.data()});

    rmm::device_uvector<edge_type_t> unique_types_partition(nr_uniqe_types_partition,
                                                            handle.get_stream());
    rmm::device_uvector<edge_t> unique_type_freqs_partition(nr_uniqe_types_partition,
                                                            handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          types_partition.begin(),
                          types_partition.end(),
                          type_freqs_partition.begin(),
                          unique_types_partition.begin(),
                          unique_type_freqs_partition.begin());

    std::vector<edge_type_t> h_unique_types_partition(nr_uniqe_types_partition);
    std::vector<edge_t> h_unique_type_freqs_partition(nr_uniqe_types_partition);

    raft::update_host(h_unique_types_partition.data(),
                      unique_types_partition.data(),
                      unique_types_partition.size(),
                      handle.get_stream());

    raft::update_host(h_unique_type_freqs_partition.data(),
                      unique_type_freqs_partition.data(),
                      unique_type_freqs_partition.size(),
                      handle.get_stream());

    for (edge_type_t idx = 0; idx < nr_uniqe_types_partition; idx++) {
      auto typ  = h_unique_types_partition[idx];
      auto freq = h_unique_type_freqs_partition[idx];
      if (edge_type_to_count_map.find(typ) != edge_type_to_count_map.end()) {
        edge_type_to_count_map[typ] += freq;
      } else {
        edge_type_to_count_map[typ] = freq;
      }
    }

    h_types_to_this_gpu.insert(
      h_types_to_this_gpu.end(), h_unique_types_partition.begin(), h_unique_types_partition.end());

    h_freq_of_types_to_this_gpu.insert(h_freq_of_types_to_this_gpu.end(),
                                       h_unique_type_freqs_partition.begin(),
                                       h_unique_type_freqs_partition.end());
  }

  //
  // Find global unique types and their frequencies
  //
  rmm::device_uvector<edge_type_t> types_to_this_gpu(h_types_to_this_gpu.size(),
                                                     handle.get_stream());

  rmm::device_uvector<edge_t> freq_of_types_to_this_gpu(h_freq_of_types_to_this_gpu.size(),
                                                        handle.get_stream());

  raft::update_device(types_to_this_gpu.data(),
                      h_types_to_this_gpu.data(),
                      h_types_to_this_gpu.size(),
                      handle.get_stream());

  raft::update_device(freq_of_types_to_this_gpu.data(),
                      h_freq_of_types_to_this_gpu.data(),
                      h_freq_of_types_to_this_gpu.size(),
                      handle.get_stream());

  thrust::sort_by_key(handle.get_thrust_policy(),
                      types_to_this_gpu.begin(),
                      types_to_this_gpu.end(),
                      freq_of_types_to_this_gpu.begin());

  auto nr_unique_types_this_gpu =
    thrust::count_if(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(types_to_this_gpu.size()),
                     detail::is_first_in_run_t<edge_type_t const*>{types_to_this_gpu.data()});

  rmm::device_uvector<edge_type_t> unique_types_to_this_gpu(nr_unique_types_this_gpu,
                                                            handle.get_stream());
  rmm::device_uvector<edge_t> freq_of_unique_types_this_gpu(nr_unique_types_this_gpu,
                                                            handle.get_stream());

  thrust::reduce_by_key(handle.get_thrust_policy(),
                        types_to_this_gpu.begin(),
                        types_to_this_gpu.end(),
                        freq_of_types_to_this_gpu.begin(),
                        unique_types_to_this_gpu.begin(),
                        freq_of_unique_types_this_gpu.begin());

  rmm::device_uvector<edge_type_t> global_unique_types(nr_unique_types_this_gpu,
                                                       handle.get_stream());

  thrust::transform(handle.get_thrust_policy(),
                    unique_types_to_this_gpu.begin(),
                    unique_types_to_this_gpu.end(),
                    global_unique_types.begin(),
                    [] __device__(auto val) { return int{val}; });

  auto nr_unique_types_global = nr_unique_types_this_gpu;

  if constexpr (multi_gpu) {
    global_unique_types = cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
      handle, std::move(global_unique_types));

    thrust::sort(
      handle.get_thrust_policy(), global_unique_types.begin(), global_unique_types.end());

    auto nr_unique_elements = static_cast<size_t>(thrust::distance(
      global_unique_types.begin(),
      thrust::unique(
        handle.get_thrust_policy(), global_unique_types.begin(), global_unique_types.end())));

    global_unique_types.resize(nr_unique_elements, handle.get_stream());

    nr_unique_types_global = host_scalar_allreduce(
      handle.get_comms(), nr_unique_elements, raft::comms::op_t::SUM, handle.get_stream());

    global_unique_types = detail::device_allgatherv(
      handle,
      handle.get_comms(),
      raft::device_span<int const>{global_unique_types.data(), global_unique_types.size()});

    assert(global_unique_types.size() == nr_unique_types_global);
  }

  std::vector<edge_type_t> h_unique_types_global(nr_unique_types_global);

  raft::update_host(h_unique_types_global.data(),
                    global_unique_types.data(),
                    global_unique_types.size(),
                    handle.get_stream());

  //
  // Create search container with appropriate capacity
  //

  std::vector<edge_id_t> local_freqs(h_unique_types_global.size(), 0);

  for (size_t idx = 0; idx < h_unique_types_global.size(); idx++) {
    auto typ              = h_unique_types_global[idx];
    auto search_itr       = edge_type_to_count_map.find(typ);
    size_t store_capacity = (search_itr != edge_type_to_count_map.end()) ? search_itr->second : 0;
    local_freqs[idx]      = store_capacity;
  }

  auto search_container =
    EdgeTypeAndIdToSrcDstLookupContainerType(handle, h_unique_types_global, local_freqs);

  //
  // Populate the search container
  //

  for (size_t local_ep_idx = 0; local_ep_idx < graph_view.number_of_local_edge_partitions();
       ++local_ep_idx) {
    //
    // decompress one edge_partition at a time
    //

    auto number_of_local_edges =
      graph_view.local_edge_partition_view(local_ep_idx).number_of_edges();

    if (graph_view.has_edge_mask()) {
      number_of_local_edges =
        detail::count_set_bits(handle,
                               (*(graph_view.edge_mask_view())).value_firsts()[local_ep_idx],
                               number_of_local_edges);
    }

    rmm::device_uvector<vertex_t> edgelist_majors(number_of_local_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
    auto edgelist_ids = rmm::device_uvector<edge_t>(edgelist_majors.size(), handle.get_stream());
    auto edgelist_types =
      rmm::device_uvector<edge_type_t>(edgelist_majors.size(), handle.get_stream());

    detail::decompress_edge_partition_to_edgelist<vertex_t, edge_t, float, edge_type_t, multi_gpu>(
      handle,
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
        graph_view.local_edge_partition_view(local_ep_idx)),
      std::nullopt,
      std::make_optional<detail::edge_partition_edge_property_device_view_t<edge_t, edge_t const*>>(
        edge_id_view, local_ep_idx),
      std::make_optional<
        detail::edge_partition_edge_property_device_view_t<edge_t, edge_type_t const*>>(
        edge_type_view, local_ep_idx),
      graph_view.has_edge_mask()
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *(graph_view.edge_mask_view()), local_ep_idx)
        : std::nullopt,
      raft::device_span<vertex_t>(edgelist_majors.data(), number_of_local_edges),
      raft::device_span<vertex_t>(edgelist_minors.data(), number_of_local_edges),
      std::nullopt,
      std::make_optional<raft::device_span<edge_t>>(edgelist_ids.data(), number_of_local_edges),
      std::make_optional<raft::device_span<edge_type_t>>(edgelist_types.data(),
                                                         number_of_local_edges),
      graph_view.local_edge_partition_segment_offsets(local_ep_idx));

    //
    // Shuffle to the right GPUs using edge ids as keys
    //

    if constexpr (multi_gpu) {
      auto const comm_size = handle.get_comms().get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      // Shuffle to the proper GPUs
      std::forward_as_tuple(
        std::tie(edgelist_majors, edgelist_minors, edgelist_ids, edgelist_types), std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors.begin(),
                                                       edgelist_minors.begin(),
                                                       edgelist_ids.begin(),
                                                       edgelist_types.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors.end(),
                                                       edgelist_minors.end(),
                                                       edgelist_ids.end(),
                                                       edgelist_types.end())),
          [key_func =
             cugraph::detail::compute_gpu_id_from_ext_vertex_t<edge_t>{
               comm_size,
               major_comm_size,
               minor_comm_size}] __device__(auto val) { return key_func(thrust::get<2>(val)); },
          handle.get_stream());
    }

    //
    // Sort by edge types and insert to type specific kv_store_t object
    //

    auto itr_to_triple = thrust::make_zip_iterator(
      edgelist_majors.begin(), edgelist_minors.begin(), edgelist_ids.begin());

    thrust::sort_by_key(
      handle.get_thrust_policy(), edgelist_types.begin(), edgelist_types.end(), itr_to_triple);

    auto nr_uniqe_edge_types_partition =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(edgelist_types.size()),
                       detail::is_first_in_run_t<edge_type_t const*>{edgelist_types.data()});

    rmm::device_uvector<edge_type_t> unique_types(nr_uniqe_edge_types_partition,
                                                  handle.get_stream());
    rmm::device_uvector<edge_t> type_offsets(nr_uniqe_edge_types_partition + 1,
                                             handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), type_offsets.begin(), type_offsets.end(), edge_t{0});

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          edgelist_types.begin(),
                          edgelist_types.end(),
                          thrust::make_constant_iterator(size_t{1}),
                          unique_types.begin(),
                          type_offsets.begin());

    thrust::exclusive_scan(handle.get_thrust_policy(),
                           type_offsets.begin(),
                           type_offsets.end(),
                           type_offsets.begin(),
                           size_t{0});

    std::vector<edge_type_t> h_unique_types(unique_types.size());
    std::vector<edge_t> h_type_offsets(type_offsets.size());

    raft::update_host(
      h_unique_types.data(), unique_types.data(), unique_types.size(), handle.get_stream());

    raft::update_host(
      h_type_offsets.data(), type_offsets.data(), type_offsets.size(), handle.get_stream());

    for (size_t idx = 0; idx < h_unique_types.size(); idx++) {
      auto typ                   = h_unique_types[idx];
      auto nr_elements_to_insert = (h_type_offsets[idx + 1] - h_type_offsets[idx]);

      auto values_to_insert =
        cugraph::allocate_dataframe_buffer<value_t>(nr_elements_to_insert, handle.get_stream());

      auto zip_itr = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));

      // FIXME: Can we avoid copying
      thrust::copy(handle.get_thrust_policy(),
                   zip_itr + h_type_offsets[idx],
                   zip_itr + h_type_offsets[idx] + nr_elements_to_insert,
                   cugraph::get_dataframe_buffer_begin(values_to_insert));

      static_assert(std::is_same_v<
                    typename thrust::iterator_traits<decltype(cugraph::get_dataframe_buffer_begin(
                      values_to_insert))>::value_type,
                    value_t>);

      search_container.insert(handle,
                              typ,
                              raft::device_span<edge_t>(edgelist_ids.begin() + h_type_offsets[idx],
                                                        nr_elements_to_insert),
                              std::move(values_to_insert));
    }
  }

  return search_container;
}

template <typename vertex_t,
          typename edge_id_t,
          typename edge_type_t,
          typename EdgeTypeAndIdToSrcDstLookupContainerType,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_for_edge_ids_of_single_type(
  raft::handle_t const& handle,
  EdgeTypeAndIdToSrcDstLookupContainerType const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup)
{
  using value_t = typename EdgeTypeAndIdToSrcDstLookupContainerType::value_type;
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(std::is_same_v<value_t, thrust::tuple<vertex_t, vertex_t>>);

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type, edge_id_t>,
    "edge_id_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type");
  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type, edge_type_t>,
    "edge_type_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type ");

  auto value_buffer = search_container.src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_type_to_lookup, multi_gpu);

  return std::make_tuple(std::move(std::get<0>(value_buffer)),
                         std::move(std::get<1>(value_buffer)));
}

template <typename vertex_t,
          typename edge_id_t,
          typename edge_type_t,
          typename EdgeTypeAndIdToSrcDstLookupContainerType,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_for_edge_ids_and_types(
  raft::handle_t const& handle,
  EdgeTypeAndIdToSrcDstLookupContainerType const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup)
{
  using value_t = typename EdgeTypeAndIdToSrcDstLookupContainerType::value_type;
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(std::is_same_v<value_t, thrust::tuple<vertex_t, vertex_t>>);

  assert(edge_ids_to_lookup.size() == edge_types_to_lookup.size());

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type, edge_id_t>,
    "edge_id_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type");
  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type, edge_type_t>,
    "edge_type_t must match EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type ");

  auto value_buffer = search_container.src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_types_to_lookup, multi_gpu);

  return std::make_tuple(std::move(std::get<0>(value_buffer)),
                         std::move(std::get<1>(value_buffer)));
}
}  // namespace detail

template <typename vertex_t, typename edge_id_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_for_edge_ids_of_single_type(
  raft::handle_t const& handle,
  lookup_container_t<edge_id_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>> const&
    search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup)
{
  using m_t = lookup_container_t<edge_id_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>>;
  return detail::
    lookup_endpoints_for_edge_ids_of_single_type<vertex_t, edge_id_t, edge_type_t, m_t, multi_gpu>(
      handle, search_container, edge_ids_to_lookup, edge_type_to_lookup);
}

template <typename vertex_t, typename edge_id_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_for_edge_ids_and_types(
  raft::handle_t const& handle,
  lookup_container_t<edge_id_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>> const&
    search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup)
{
  using m_t = lookup_container_t<edge_id_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>>;
  return detail::
    lookup_endpoints_for_edge_ids_and_types<vertex_t, edge_id_t, edge_type_t, m_t, multi_gpu>(
      handle, search_container, edge_ids_to_lookup, edge_types_to_lookup);
}

template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
lookup_container_t<edge_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>>
build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, edge_t const*> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view)
{
  using graph_view_t = graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  using return_t     = lookup_container_t<edge_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>>;

  return detail::build_edge_id_and_type_to_src_dst_lookup_map<graph_view_t,
                                                              decltype(edge_id_view),
                                                              decltype(edge_type_view),
                                                              return_t>(
    handle, graph_view, edge_id_view, edge_type_view);
}
}  // namespace cugraph
