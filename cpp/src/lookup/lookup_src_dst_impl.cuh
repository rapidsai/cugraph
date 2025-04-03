/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "prims/extract_transform_e.cuh"
#include "prims/kv_store.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/src_dst_lookup_container.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/handle.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>

namespace cugraph {

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
template <typename _edge_id_t, typename _edge_type_t, typename _vertex_t, typename _value_t>
struct lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_impl {
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
    auto invalid_vertex_id = cugraph::invalid_vertex_id<edge_id_t>::value;
    auto invalid_value = thrust::tuple<vertex_t, vertex_t>(invalid_vertex_id, invalid_vertex_id);

    edge_type_to_kv_store = container_t{};
    edge_type_to_kv_store.reserve(types.size());

    for (size_t idx = 0; idx < types.size(); idx++) {
      auto typ = types[idx];
      assert(typ != empty_type_);
      size_t store_capacity = type_counts[idx];

      edge_type_to_kv_store.insert(
        {typ, store_t(store_capacity, invalid_vertex_id, invalid_value, handle.get_stream())});
    }

    edge_type_to_kv_store.insert(
      {empty_type_, store_t(0, invalid_vertex_id, invalid_value, handle.get_stream())});
  }

  void insert(raft::handle_t const& handle,
              edge_type_t type,
              raft::device_span<edge_id_t const> edge_ids_to_insert,
              dataframe_buffer_type_t<value_t>&& values_to_insert)
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
      kv_store_object = &(edge_type_to_kv_store.find(empty_type_)->second);
    }

    if (multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      value_buffer = cugraph::collect_values_for_keys(
        comm,
        kv_store_object->view(),
        edge_ids_to_lookup.begin(),
        edge_ids_to_lookup.end(),
        cugraph::detail::compute_gpu_id_from_ext_edge_id_t<edge_id_t>{
          comm_size, major_comm_size, minor_comm_size},
        handle.get_stream());
    } else {
      cugraph::resize_dataframe_buffer(
        value_buffer, edge_ids_to_lookup.size(), handle.get_stream());

      kv_store_object->view().find(edge_ids_to_lookup.begin(),
                                   edge_ids_to_lookup.end(),
                                   cugraph::get_dataframe_buffer_begin(value_buffer),
                                   handle.get_stream());
    }

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
    rmm::device_uvector<edge_id_t> type_offsets(nr_uniqe_edge_types_to_lookup + size_t{1},
                                                handle.get_stream());

    thrust::copy_if(handle.get_thrust_policy(),
                    tmp_edge_types_to_lookup.begin(),
                    tmp_edge_types_to_lookup.end(),
                    thrust::make_counting_iterator(size_t{0}),
                    unique_types.begin(),
                    detail::is_first_in_run_t<edge_type_t const*>{tmp_edge_types_to_lookup.data()});

    type_offsets.set_element_to_zero_async(0, handle.get_stream());
    thrust::upper_bound(handle.get_thrust_policy(),
                        tmp_edge_types_to_lookup.begin(),
                        tmp_edge_types_to_lookup.end(),
                        unique_types.begin(),
                        unique_types.end(),
                        type_offsets.begin() + 1);

    std::vector<edge_type_t> h_unique_types(unique_types.size());
    std::vector<edge_id_t> h_type_offsets(type_offsets.size());

    raft::update_host(
      h_unique_types.data(), unique_types.data(), unique_types.size(), handle.get_stream());

    raft::update_host(
      h_type_offsets.data(), type_offsets.data(), type_offsets.size(), handle.get_stream());

    handle.sync_stream();

    std::unordered_map<edge_type_t, int> typ_to_local_idx_map{};
    for (size_t idx = 0; idx < h_unique_types.size(); idx++) {
      typ_to_local_idx_map[h_unique_types[idx]] = idx;
    }

    auto output_value_buffer =
      cugraph::allocate_dataframe_buffer<value_t>(edge_ids_to_lookup.size(), handle.get_stream());
    if (multi_gpu) {
      auto& comm     = handle.get_comms();
      auto rx_counts = host_scalar_allgather(comm, unique_types.size(), handle.get_stream());
      std::vector<size_t> rx_displacements(rx_counts.size());
      std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
      rmm::device_uvector<edge_type_t> rx_unique_types(rx_displacements.back() + rx_counts.back(),
                                                       handle.get_stream());

      device_allgatherv(
        comm,
        unique_types.begin(),
        rx_unique_types.begin(),
        raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
        raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
        handle.get_stream());
      unique_types = std::move(rx_unique_types);

      thrust::sort(handle.get_thrust_policy(), unique_types.begin(), unique_types.end());

      unique_types.resize(
        cuda::std::distance(
          unique_types.begin(),
          thrust::unique(handle.get_thrust_policy(), unique_types.begin(), unique_types.end())),
        handle.get_stream());
    }

    h_unique_types.resize(unique_types.size());
    raft::update_host(
      h_unique_types.data(), unique_types.data(), unique_types.size(), handle.get_stream());

    handle.sync_stream();

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

      thrust::copy(handle.get_thrust_policy(),
                   cugraph::get_dataframe_buffer_begin(value_buffer_typ),
                   cugraph::get_dataframe_buffer_end(value_buffer_typ),
                   cugraph::get_dataframe_buffer_begin(output_value_buffer) +
                     h_type_offsets[typ_to_local_idx_map[typ]]);
    }

    thrust::sort_by_key(handle.get_thrust_policy(),
                        original_idxs.begin(),
                        original_idxs.end(),
                        cugraph::get_dataframe_buffer_begin(output_value_buffer));

    return std::make_tuple(std::move(std::get<0>(output_value_buffer)),
                           std::move(std::get<1>(output_value_buffer)));
  }

 private:
  using container_t =
    std::unordered_map<edge_type_t,
                       cugraph::kv_store_t<edge_id_t, value_t, false /*use_binary_search*/>>;
  using store_t = typename container_t::mapped_type;
  container_t edge_type_to_kv_store;
  edge_type_t empty_type_ = std::numeric_limits<edge_type_t>::max() - 1;
};

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::~lookup_container_t()
{
  pimpl.reset();
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_t()
  : pimpl{std::make_unique<lookup_container_impl<edge_id_t, edge_type_t, vertex_t, value_t>>()}
{
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_t(
  raft::handle_t const& handle, std::vector<edge_type_t> types, std::vector<edge_id_t> type_counts)
  : pimpl{std::make_unique<lookup_container_impl<edge_id_t, edge_type_t, vertex_t, value_t>>(
      handle, types, type_counts)}
{
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_container_t(
  lookup_container_t&& other)
  : pimpl{std::move(other.pimpl)}
{
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>&
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::operator=(lookup_container_t&& other)
{
  pimpl = std::move(other.pimpl);
  return *this;
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
void lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::insert(
  raft::handle_t const& handle,
  edge_type_t type,
  raft::device_span<edge_id_t const> edge_ids_to_insert,
  dataframe_buffer_type_t<value_t>&& values_to_insert)
{
  pimpl->insert(handle, type, edge_ids_to_insert, std::move(values_to_insert));
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
dataframe_buffer_type_t<value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_from_edge_ids_and_single_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup,
  bool multi_gpu) const
{
  return pimpl->src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_type_to_lookup, multi_gpu);
}

template <typename edge_id_t, typename edge_type_t, typename vertex_t, typename value_t>
dataframe_buffer_type_t<value_t>
lookup_container_t<edge_id_t, edge_type_t, vertex_t, value_t>::lookup_from_edge_ids_and_types(
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
  static_assert(!std::is_same_v<typename EdgeIdInputWrapper::value_type, cuda::std::nullopt_t>,
                "Can not create edge id lookup table without edge ids");

  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using edge_type_t = typename EdgeTypeInputWrapper::value_type;
  using edge_id_t   = typename EdgeIdInputWrapper::value_type;
  using value_t     = typename EdgeTypeAndIdToSrcDstLookupContainerType::value_type;

  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_same_v<edge_t, edge_id_t>);
  static_assert(std::is_same_v<value_t, thrust::tuple<vertex_t, vertex_t>>);

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type, edge_type_t>,
    "edge_type_t must match with EdgeTypeAndIdToSrcDstLookupContainerType::edge_type_type");

  static_assert(
    std::is_same_v<typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type, edge_id_t>,
    "edge_id_t must match with typename EdgeTypeAndIdToSrcDstLookupContainerType::edge_id_type");

  rmm::device_uvector<edge_type_t> unique_types(0, handle.get_stream());
  rmm::device_uvector<edge_t> unique_type_counts(0, handle.get_stream());

  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    auto [gpu_ids, edge_types] =
      cugraph::extract_transform_e(
        handle,
        graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        view_concat(edge_id_view, edge_type_view),
        cuda::proclaim_return_type<thrust::tuple<int, edge_type_t>>(
          [key_func =
             cugraph::detail::compute_gpu_id_from_ext_edge_id_t<edge_t>{
               comm_size,
               major_comm_size,
               minor_comm_size}] __device__(auto,
                                            auto,
                                            cuda::std::nullopt_t,
                                            cuda::std::nullopt_t,
                                            thrust::tuple<edge_t, edge_type_t> id_and_type) {
            return thrust::make_tuple(key_func(thrust::get<0>(id_and_type)),
                                      thrust::get<1>(id_and_type));
          }));

    auto type_and_gpu_id_pair_begin =
      thrust::make_zip_iterator(thrust::make_tuple(edge_types.begin(), gpu_ids.begin()));

    thrust::sort(handle.get_thrust_policy(),
                 type_and_gpu_id_pair_begin,
                 type_and_gpu_id_pair_begin + edge_types.size());

    auto nr_unique_paris = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(edge_types.size()),
      detail::is_first_in_run_t<decltype(type_and_gpu_id_pair_begin)>{type_and_gpu_id_pair_begin});

    auto unique_pairs_buffer = cugraph::allocate_dataframe_buffer<
      typename thrust::iterator_traits<decltype(type_and_gpu_id_pair_begin)>::value_type>(
      nr_unique_paris, handle.get_stream());

    rmm::device_uvector<edge_t> unique_pair_counts(nr_unique_paris, handle.get_stream());

    {
      rmm::device_uvector<edge_t> unique_pair_end_offsets(nr_unique_paris, handle.get_stream());

      thrust::copy_if(handle.get_thrust_policy(),
                      type_and_gpu_id_pair_begin,
                      type_and_gpu_id_pair_begin + edge_types.size(),
                      thrust::make_counting_iterator(size_t{0}),
                      cugraph::get_dataframe_buffer_begin(unique_pairs_buffer),
                      detail::is_first_in_run_t<decltype(type_and_gpu_id_pair_begin)>{
                        type_and_gpu_id_pair_begin});

      thrust::upper_bound(handle.get_thrust_policy(),
                          type_and_gpu_id_pair_begin,
                          type_and_gpu_id_pair_begin + edge_types.size(),
                          cugraph::get_dataframe_buffer_begin(unique_pairs_buffer),
                          cugraph::get_dataframe_buffer_end(unique_pairs_buffer),
                          unique_pair_end_offsets.begin());

      thrust::adjacent_difference(handle.get_thrust_policy(),
                                  unique_pair_end_offsets.begin(),
                                  unique_pair_end_offsets.end(),
                                  unique_pair_counts.begin());
    }

    edge_types.resize(0, handle.get_stream());
    gpu_ids.resize(0, handle.get_stream());
    edge_types.shrink_to_fit(handle.get_stream());
    gpu_ids.shrink_to_fit(handle.get_stream());

    std::forward_as_tuple(
      std::tie(std::get<0>(unique_pairs_buffer), std::ignore, unique_pair_counts), std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        handle.get_comms(),
        thrust::make_zip_iterator(thrust::make_tuple(std::get<0>(unique_pairs_buffer).begin(),
                                                     std::get<1>(unique_pairs_buffer).begin(),
                                                     unique_pair_counts.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(std::get<0>(unique_pairs_buffer).end(),
                                                     std::get<1>(unique_pairs_buffer).end(),
                                                     unique_pair_counts.end())),
        [] __device__(auto val) { return thrust::get<1>(val); },
        handle.get_stream());

    //
    // Count local #elments for all the types mapped to this GPU
    //

    thrust::sort_by_key(handle.get_thrust_policy(),
                        std::get<0>(unique_pairs_buffer).begin(),
                        std::get<0>(unique_pairs_buffer).end(),
                        unique_pair_counts.begin());

    auto nr_unique_types = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(std::get<0>(unique_pairs_buffer).size()),
      detail::is_first_in_run_t<edge_type_t const*>{std::get<0>(unique_pairs_buffer).data()});

    unique_types.resize(static_cast<size_t>(nr_unique_types), handle.get_stream());
    unique_type_counts.resize(static_cast<size_t>(nr_unique_types), handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          std::get<0>(unique_pairs_buffer).begin(),
                          std::get<0>(unique_pairs_buffer).end(),
                          unique_pair_counts.begin(),
                          unique_types.begin(),
                          unique_type_counts.begin());

  } else {
    auto edge_types = cugraph::extract_transform_e(
      handle,
      graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_type_view,
      cuda::proclaim_return_type<edge_type_t>(
        [] __device__(auto, auto, cuda::std::nullopt_t, cuda::std::nullopt_t, edge_type_t et) {
          return et;
        }));

    thrust::sort(handle.get_thrust_policy(), edge_types.begin(), edge_types.end());

    auto nr_unique_types =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(edge_types.size()),
                       detail::is_first_in_run_t<edge_type_t const*>{edge_types.data()});

    unique_types.resize(static_cast<size_t>(nr_unique_types), handle.get_stream());
    unique_type_counts.resize(static_cast<size_t>(nr_unique_types), handle.get_stream());

    {
      rmm::device_uvector<edge_t> unique_type_end_offsets(nr_unique_types, handle.get_stream());
      thrust::copy_if(handle.get_thrust_policy(),
                      edge_types.begin(),
                      edge_types.end(),
                      thrust::make_counting_iterator(size_t{0}),
                      unique_types.begin(),
                      detail::is_first_in_run_t<edge_type_t const*>{edge_types.data()});

      thrust::upper_bound(handle.get_thrust_policy(),
                          edge_types.begin(),
                          edge_types.end(),
                          unique_types.begin(),
                          unique_types.end(),
                          unique_type_end_offsets.begin());

      thrust::adjacent_difference(handle.get_thrust_policy(),
                                  unique_type_end_offsets.begin(),
                                  unique_type_end_offsets.end(),
                                  unique_type_counts.begin());
    }
  }

  std::vector<edge_type_t> h_unique_types(unique_types.size());
  std::vector<edge_t> h_unique_type_counts(unique_types.size());

  raft::update_host(
    h_unique_types.data(), unique_types.data(), unique_types.size(), handle.get_stream());

  raft::update_host(h_unique_type_counts.data(),
                    unique_type_counts.data(),
                    unique_type_counts.size(),
                    handle.get_stream());

  handle.sync_stream();

  auto search_container =
    EdgeTypeAndIdToSrcDstLookupContainerType(handle, h_unique_types, h_unique_type_counts);

  //
  // Populate the search container
  //

  for (size_t local_ep_idx = 0; local_ep_idx < graph_view.number_of_local_edge_partitions();
       ++local_ep_idx) {
    //
    // decompress one edge_partition at a time
    //

    auto edge_partition = edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
      graph_view.local_edge_partition_view(local_ep_idx));

    auto edge_partition_mask_view =
      graph_view.has_edge_mask()
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *(graph_view.edge_mask_view()), local_ep_idx)
        : std::nullopt;

    auto number_of_local_edges = edge_partition.number_of_edges();
    if (graph_view.has_edge_mask()) {
      number_of_local_edges = edge_partition.compute_number_of_edges_with_mask(
        (*edge_partition_mask_view).value_first(),
        thrust::make_counting_iterator(edge_partition.major_range_first()),
        thrust::make_counting_iterator(edge_partition.major_range_last()),
        handle.get_stream());
    }

    rmm::device_uvector<vertex_t> edgelist_majors(number_of_local_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
    auto edgelist_ids = rmm::device_uvector<edge_t>(edgelist_majors.size(), handle.get_stream());
    auto edgelist_types =
      rmm::device_uvector<edge_type_t>(edgelist_majors.size(), handle.get_stream());

    detail::decompress_edge_partition_to_edgelist<vertex_t, edge_t, float, edge_type_t, multi_gpu>(
      handle,
      edge_partition,
      std::nullopt,
      std::make_optional<detail::edge_partition_edge_property_device_view_t<edge_t, edge_t const*>>(
        edge_id_view, local_ep_idx),
      std::make_optional<
        detail::edge_partition_edge_property_device_view_t<edge_t, edge_type_t const*>>(
        edge_type_view, local_ep_idx),
      edge_partition_mask_view,
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
             cugraph::detail::compute_gpu_id_from_ext_edge_id_t<edge_t>{
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

    thrust::copy_if(handle.get_thrust_policy(),
                    edgelist_types.begin(),
                    edgelist_types.end(),
                    thrust::make_counting_iterator(size_t{0}),
                    unique_types.begin(),
                    detail::is_first_in_run_t<edge_type_t const*>{edgelist_types.data()});

    type_offsets.set_element_to_zero_async(0, handle.get_stream());

    thrust::upper_bound(handle.get_thrust_policy(),
                        edgelist_types.begin(),
                        edgelist_types.end(),
                        unique_types.begin(),
                        unique_types.end(),
                        type_offsets.begin() + 1);

    std::vector<edge_type_t> h_unique_types(unique_types.size());
    std::vector<edge_t> h_type_offsets(type_offsets.size());

    raft::update_host(
      h_unique_types.data(), unique_types.data(), unique_types.size(), handle.get_stream());

    raft::update_host(
      h_type_offsets.data(), type_offsets.data(), type_offsets.size(), handle.get_stream());
    handle.sync_stream();

    for (size_t idx = 0; idx < h_unique_types.size(); idx++) {
      auto typ                   = h_unique_types[idx];
      auto nr_elements_to_insert = (h_type_offsets[idx + 1] - h_type_offsets[idx]);

      auto values_to_insert =
        cugraph::allocate_dataframe_buffer<value_t>(nr_elements_to_insert, handle.get_stream());

      auto zip_itr = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));

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
lookup_endpoints_from_edge_ids_and_single_type(
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

  auto value_buffer = search_container.lookup_from_edge_ids_and_single_type(
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
lookup_endpoints_from_edge_ids_and_types(
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

  auto value_buffer = search_container.lookup_from_edge_ids_and_types(
    handle, edge_ids_to_lookup, edge_types_to_lookup, multi_gpu);

  return std::make_tuple(std::move(std::get<0>(value_buffer)),
                         std::move(std::get<1>(value_buffer)));
}
}  // namespace detail

template <typename vertex_t, typename edge_id_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_single_type(
  raft::handle_t const& handle,
  lookup_container_t<edge_id_t, edge_type_t, vertex_t> const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup)
{
  using m_t = lookup_container_t<edge_id_t, edge_type_t, vertex_t>;
  return detail::lookup_endpoints_from_edge_ids_and_single_type<vertex_t,
                                                                edge_id_t,
                                                                edge_type_t,
                                                                m_t,
                                                                multi_gpu>(
    handle, search_container, edge_ids_to_lookup, edge_type_to_lookup);
}

template <typename vertex_t, typename edge_id_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_types(
  raft::handle_t const& handle,
  lookup_container_t<edge_id_t, edge_type_t, vertex_t> const& search_container,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup)
{
  using m_t = lookup_container_t<edge_id_t, edge_type_t, vertex_t>;
  return detail::
    lookup_endpoints_from_edge_ids_and_types<vertex_t, edge_id_t, edge_type_t, m_t, multi_gpu>(
      handle, search_container, edge_ids_to_lookup, edge_types_to_lookup);
}

template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
lookup_container_t<edge_t, edge_type_t, vertex_t> build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, edge_t const*> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view)
{
  using graph_view_t = graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  using return_t     = lookup_container_t<edge_t, edge_type_t, vertex_t>;

  return detail::build_edge_id_and_type_to_src_dst_lookup_map<graph_view_t,
                                                              decltype(edge_id_view),
                                                              decltype(edge_type_view),
                                                              return_t>(
    handle, graph_view, edge_id_view, edge_type_view);
}
}  // namespace cugraph
