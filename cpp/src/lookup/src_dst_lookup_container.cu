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
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/src_dst_lookup_container.hpp>

namespace cugraph {

template <typename edge_id_t, typename edge_type_t, typename value_t>
template <typename _edge_id_t, typename _edge_type_t, typename _value_t>
struct search_container_t<edge_id_t, edge_type_t, value_t>::search_container_impl {
 private:
  using container_t =
    std::unordered_map<edge_type_t,
                       cugraph::kv_store_t<edge_id_t, value_t, false /*use_binary_search*/>>;
  using store_t = typename container_t::mapped_type;
  container_t edge_type_to_kv_store;

 public:
  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);

  static_assert(std::is_same_v<edge_type_t, _edge_type_t>);
  static_assert(std::is_same_v<edge_id_t, _edge_id_t>);
  static_assert(std::is_same_v<value_t, _value_t>);

  ~search_container_impl() {}
  search_container_impl() {}
  search_container_impl(raft::handle_t const& handle,
                        std::vector<edge_type_t> typs,
                        std::vector<edge_id_t> typ_freqs)
  {
    auto invalid_key   = cugraph::invalid_vertex_id<edge_id_t>::value;
    auto invalid_value = invalid_of_thrust_tuple_of_integral<value_t>();

    edge_type_to_kv_store = container_t{};
    edge_type_to_kv_store.reserve(typs.size());

    for (size_t idx = 0; idx < typs.size(); idx++) {
      auto typ              = typs[idx];
      size_t store_capacity = typ_freqs[idx];

      edge_type_to_kv_store.insert(
        {typ, std::move(store_t(store_capacity, invalid_key, invalid_value, handle.get_stream()))});
    }
  }

  void print() const
  {
    for (const auto& [key, map] : edge_type_to_kv_store) {
      std::cout << "key: " << key << " size: " << map.size() << std::endl;
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

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  src_dst_from_edge_id_and_type(raft::handle_t const& handle,
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
      // Mark with invalid here
      thrust::copy(handle.get_thrust_policy(),
                   thrust::make_constant_iterator(invalid_value),
                   thrust::make_constant_iterator(invalid_value) + edge_ids_to_lookup.size(),
                   cugraph::get_dataframe_buffer_begin(value_buffer));

      return std::make_optional(std::move(value_buffer));
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

    return std::make_optional(std::move(value_buffer));
  }

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  src_dst_from_edge_id_and_type(raft::handle_t const& handle,
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

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

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

      auto start_ptr   = tmp_edge_ids_to_lookup.begin();
      size_t span_size = 0;

      if (typ_to_local_idx_map.find(typ) != typ_to_local_idx_map.end()) {
        auto xx = typ_to_local_idx_map[typ];

        start_ptr = tmp_edge_ids_to_lookup.begin() + h_type_offsets[xx];
        span_size = h_type_offsets[xx + 1] - h_type_offsets[xx];
      }

      auto optional_value_buffer = src_dst_from_edge_id_and_type(
        handle, raft::device_span<edge_id_t const>{start_ptr, span_size}, typ, multi_gpu);

      if (optional_value_buffer.has_value()) {
        if (typ_to_local_idx_map.find(typ) != typ_to_local_idx_map.end()) {
          thrust::copy(handle.get_thrust_policy(),
                       cugraph::get_dataframe_buffer_begin((*optional_value_buffer)),
                       cugraph::get_dataframe_buffer_end((*optional_value_buffer)),
                       cugraph::get_dataframe_buffer_begin(output_value_buffer) +
                         h_type_offsets[typ_to_local_idx_map[typ]]);
        }
      }
    }

    thrust::sort_by_key(handle.get_thrust_policy(),
                        original_idxs.begin(),
                        original_idxs.end(),
                        cugraph::get_dataframe_buffer_begin(output_value_buffer));

    return std::make_optional(std::move(output_value_buffer));
  }
};

template <typename edge_id_t, typename edge_type_t, typename value_t>
search_container_t<edge_id_t, edge_type_t, value_t>::~search_container_t()
{
  pimpl.reset();
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
search_container_t<edge_id_t, edge_type_t, value_t>::search_container_t()
  : pimpl{std::make_unique<search_container_impl<edge_id_t, edge_type_t, value_t>>()}
{
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
search_container_t<edge_id_t, edge_type_t, value_t>::search_container_t(
  raft::handle_t const& handle, std::vector<edge_type_t> typs, std::vector<edge_id_t> typ_freqs)
  : pimpl{std::make_unique<search_container_impl<edge_id_t, edge_type_t, value_t>>(
      handle, typs, typ_freqs)}
{
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
search_container_t<edge_id_t, edge_type_t, value_t>::search_container_t(const search_container_t&)
{
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
void search_container_t<edge_id_t, edge_type_t, value_t>::insert(
  raft::handle_t const& handle,
  edge_type_t type,
  raft::device_span<edge_id_t const> edge_ids_to_insert,
  decltype(cugraph::allocate_dataframe_buffer<value_t>(0,
                                                       rmm::cuda_stream_view{}))&& values_to_insert)
{
  pimpl->insert(handle, type, edge_ids_to_insert, std::move(values_to_insert));
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
search_container_t<edge_id_t, edge_type_t, value_t>::src_dst_from_edge_id_and_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup,
  bool multi_gpu) const
{
  return pimpl->src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_type_to_lookup, multi_gpu);
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
search_container_t<edge_id_t, edge_type_t, value_t>::src_dst_from_edge_id_and_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup,
  bool multi_gpu) const
{
  return pimpl->src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_types_to_lookup, multi_gpu);
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
void search_container_t<edge_id_t, edge_type_t, value_t>::print() const
{
  pimpl->print();
}

template class search_container_t<int32_t, int32_t, thrust::tuple<int32_t, int32_t>>;
template class search_container_t<int64_t, int32_t, thrust::tuple<int32_t, int32_t>>;
template class search_container_t<int64_t, int32_t, thrust::tuple<int64_t, int64_t>>;

}  // namespace cugraph
