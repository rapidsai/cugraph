
#include "detail/graph_partition_utils.cuh"
#include "prims/kv_store.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/lookup_container.hpp>

namespace cugraph {

// ------- Don't remove----
// template <typename edge_type_t, typename edge_id_t, typename value_t>
// struct impl {
//   using container_t =
//     std::unordered_map<edge_type_t,
//                        cugraph::kv_store_t<edge_id_t, value_t, false /*use_binary_search*/>>;

//   static_assert(std::is_arithmetic_v<edge_type_t>);
//   static_assert(std::is_arithmetic_v<edge_id_t>);
//   static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

//   void insert(edge_type_t et, edge_id_t ei, value_t v) { std::cout << "From impl insert\n"; }
// };
//---------------------

template <typename edge_type_t, typename edge_id_t, typename value_t>
template <typename _edge_type_t, typename _edge_id_t, typename _value_t>
struct search_container_t<edge_type_t, edge_id_t, value_t>::impl {
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

  ~impl() {}
  impl() {}
  impl(raft::handle_t const& handle,
       std::vector<edge_type_t> types,
       std::vector<edge_id_t> type_freqs)
  {
    auto invalid_key   = cugraph::invalid_vertex_id<edge_id_t>::value;
    auto invalid_value = invalid_of_thrust_tuple_of_integral<value_t>();

    // constexpr(std::is_integral<value_t>::value) ? (cugraph::invalid_vertex_id<value_t>::value)
    edge_type_to_kv_store = container_t{};
    edge_type_to_kv_store.reserve(types.size());

    // constexpr auto ts = thrust::tuple_size<value_t>::value;
    // std::cout << "ts: " << ts << ", iv: (" << thrust::get<0>(invalid_value) << ", "
    //           << thrust::get<1>(invalid_value) << ")\n";

    for (size_t idx = 0; idx < types.size(); idx++) {
      auto typ              = types[idx];
      size_t store_capacity = type_freqs[idx];

      edge_type_to_kv_store.insert(
        {typ, std::move(store_t(store_capacity, invalid_key, invalid_value, handle.get_stream()))});

      // auto iv = edge_type_to_kv_store.find(typ)->second.invalid_value();

      // std::cout << "invalid -(" << thrust::get<0>(iv) << ", " << thrust::get<1>(iv) << ")-\n";

      // if (edge_type_to_kv_store.find(typ)->second.invalid_value() ==
      //     thrust::tuple<int32_t, int32_t>(-1, -1)) {
      //   std::cout << "Right kind of Invalid \n";
      // } else {
      //   std::cout << "Something is wrong with invalid value \n";
      // }
    }
  }

  void print()
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

      // auto const comm_rank = handle.get_comms().get_rank();
      // std::cout << "** rank:  " << comm_rank << " inserting  type: " << type << "\n";
      // auto srcs = raft::device_span<typename thrust::tuple_element<0, value_t>::type const>(
      //   std::get<0>(values_to_insert).data(), std::get<0>(values_to_insert).size());

      // auto dsts = raft::device_span<typename thrust::tuple_element<1, value_t>::type const>(
      //   std::get<1>(values_to_insert).data(), std::get<1>(values_to_insert).size());

      // auto srcs_title = std::string("isrcs_").append(std::to_string(comm_rank));
      // RAFT_CUDA_TRY(cudaDeviceSynchronize());
      // raft::print_device_vector(srcs_title.c_str(), srcs.begin(), srcs.size(), std::cout);

      // auto dsts_title = std::string("idsts_").append(std::to_string(comm_rank));
      // RAFT_CUDA_TRY(cudaDeviceSynchronize());
      // raft::print_device_vector(dsts_title.c_str(), dsts.begin(), dsts.size(), std::cout);

      itr->second.insert(edge_ids_to_insert.begin(),
                         edge_ids_to_insert.end(),
                         cugraph::get_dataframe_buffer_begin(values_to_insert),
                         handle.get_stream());
    } else {
      assert(false);
    }
  }

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  lookup_src_dst_from_edge_id_and_type(raft::handle_t const& handle,
                                       raft::device_span<edge_id_t const> edge_ids_to_lookup,
                                       edge_type_t edge_type_to_lookup,
                                       bool multi_gpu) const
  {
    using store_t = typename container_t::mapped_type;
    const store_t* kv_store_object{nullptr};

    auto value_buffer = cugraph::allocate_dataframe_buffer<value_t>(0, handle.get_stream());

    auto itr = edge_type_to_kv_store.find(edge_type_to_lookup);
    if (itr != edge_type_to_kv_store.end()) {
      assert(edge_type_to_lookup == itr->first);
      kv_store_object = &(itr->second);
    } else {
      return std::optional<decltype(value_buffer)>{std::nullopt};
    }

    // auto invalid_key = cugraph::invalid_vertex_id<edge_id_t>::value;
    // auto invalid_value = invalid_of_thrust_tuple_of_integral<value_t>();
    // Mark with invalid here
    // thrust::copy(handle.get_thrust_policy(),
    //              thrust::make_constant_iterator(invalid_value),
    //              thrust::make_constant_iterator(invalid_value) + edge_ids_to_lookup.size(),
    //              cugraph::get_dataframe_buffer_begin(value_buffer));

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

    // auto const comm_rank = multi_gpu ? handle.get_comms().get_rank() : 0;

    // auto srcs = raft::device_span<typename thrust::tuple_element<0, value_t>::type const>(
    //   std::get<0>(value_buffer).data(), std::get<0>(value_buffer).size());

    // auto dsts = raft::device_span<typename thrust::tuple_element<1, value_t>::type const>(
    //   std::get<1>(value_buffer).data(), std::get<1>(value_buffer).size());

    // auto srcs_title = std::string("*srcs_").append(std::to_string(comm_rank));
    // RAFT_CUDA_TRY(cudaDeviceSynchronize());
    // raft::print_device_vector(srcs_title.c_str(), srcs.begin(), srcs.size(), std::cout);

    // auto dsts_title = std::string("*dsts_").append(std::to_string(comm_rank));
    // RAFT_CUDA_TRY(cudaDeviceSynchronize());
    // raft::print_device_vector(dsts_title.c_str(), dsts.begin(), dsts.size(), std::cout);
    return std::make_optional(std::move(value_buffer));
  }

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  lookup_src_dst_from_edge_id_and_type(raft::handle_t const& handle,
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

    // auto const comm_rank = handle.get_comms().get_rank();

    // auto ts_ttile = std::string("ltys_").append(std::to_string(comm_rank));
    // RAFT_CUDA_TRY(cudaDeviceSynchronize());

    // RAFT_CUDA_TRY(cudaDeviceSynchronize());

    // raft::print_device_vector(ts_ttile.c_str(),
    //                           tmp_edge_types_to_lookup.begin(),
    //                           tmp_edge_types_to_lookup.size(),
    //                           std::cout);

    // raft::print_device_vector(
    //   ts_ttile.c_str(), edge_types_to_lookup.begin(), edge_types_to_lookup.size(), std::cout);

    // auto lids_title = std::string("lids_").append(std::to_string(comm_rank));
    // RAFT_CUDA_TRY(cudaDeviceSynchronize());

    // raft::print_device_vector(
    //   lids_title.c_str(), tmp_edge_ids_to_lookup.begin(), tmp_edge_ids_to_lookup.size(),
    //   std::cout);

    // RAFT_CUDA_TRY(cudaDeviceSynchronize());

    // raft::print_device_vector(
    //   lids_title.c_str(), edge_ids_to_lookup.begin(), edge_ids_to_lookup.size(), std::cout);

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

    auto output_value_buffer =
      cugraph::allocate_dataframe_buffer<value_t>(edge_ids_to_lookup.size(), handle.get_stream());

    for (size_t idx = 0; idx < h_unique_types.size(); idx++) {
      auto typ = h_unique_types[idx];

      // std::cout << " rank:  " << comm_rank << " searching type: " << typ << "\n";

      // auto lp_title = std::string("lp_")
      //                   .append(std::to_string(comm_rank))
      //                   .append("_t_")
      //                   .append(std::to_string(typ));
      // RAFT_CUDA_TRY(cudaDeviceSynchronize());

      // auto ptr = tmp_edge_ids_to_lookup.begin() + h_type_offsets[idx];
      // raft::print_device_vector(
      //   lp_title.c_str(), ptr, (h_type_offsets[idx + 1] - h_type_offsets[idx]), std::cout);

      auto optional_value_buffer = lookup_src_dst_from_edge_id_and_type(
        handle,
        raft::device_span<edge_id_t const>{
          tmp_edge_ids_to_lookup.begin() + h_type_offsets[idx],
          tmp_edge_ids_to_lookup.begin() + h_type_offsets[idx + 1]},
        typ,
        multi_gpu);

      // if (optional_value_buffer.has_value()) {
      //   auto srcs = raft::device_span<typename thrust::tuple_element<0, value_t>::type const>(
      //     std::get<0>((*optional_value_buffer)).data(),
      //     std::get<0>((*optional_value_buffer)).size());

      //   auto dsts = raft::device_span<typename thrust::tuple_element<1, value_t>::type const>(
      //     std::get<1>((*optional_value_buffer)).data(),
      //     std::get<1>((*optional_value_buffer)).size());

      //   auto srcs_title = std::string("fsrcs_").append(std::to_string(comm_rank));
      //   RAFT_CUDA_TRY(cudaDeviceSynchronize());
      //   raft::print_device_vector(srcs_title.c_str(), srcs.begin(), srcs.size(), std::cout);

      //   auto dsts_title = std::string("sdsts_").append(std::to_string(comm_rank));
      //   RAFT_CUDA_TRY(cudaDeviceSynchronize());
      //   raft::print_device_vector(dsts_title.c_str(), dsts.begin(), dsts.size(), std::cout);
      // } else {
      //   std::cout << comm_rank << ": Not found" << std::endl;
      // }

      if (optional_value_buffer.has_value()) {
        thrust::copy(
          handle.get_thrust_policy(),
          cugraph::get_dataframe_buffer_begin((*optional_value_buffer)),
          cugraph::get_dataframe_buffer_end((*optional_value_buffer)),
          cugraph::get_dataframe_buffer_begin(output_value_buffer) + h_type_offsets[idx]);
      }
    }

    thrust::sort_by_key(handle.get_thrust_policy(),
                        original_idxs.begin(),
                        original_idxs.end(),
                        cugraph::get_dataframe_buffer_begin(output_value_buffer));

    return std::make_optional(std::move(output_value_buffer));
  }
};

template <typename edge_type_t, typename edge_id_t, typename value_t>
search_container_t<edge_type_t, edge_id_t, value_t>::~search_container_t()
{
  pimpl.reset();
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
search_container_t<edge_type_t, edge_id_t, value_t>::search_container_t()
  : pimpl{std::make_unique<impl<edge_type_t, edge_id_t, value_t>>()}
{
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
search_container_t<edge_type_t, edge_id_t, value_t>::search_container_t(
  raft::handle_t const& handle, std::vector<edge_type_t> types, std::vector<edge_id_t> type_freqs)
  : pimpl{std::make_unique<impl<edge_type_t, edge_id_t, value_t>>(handle, types, type_freqs)}
{
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
search_container_t<edge_type_t, edge_id_t, value_t>::search_container_t(const search_container_t&)
{
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
void search_container_t<edge_type_t, edge_id_t, value_t>::insert(
  raft::handle_t const& handle,
  edge_type_t type,
  raft::device_span<edge_id_t const> edge_ids_to_insert,
  decltype(cugraph::allocate_dataframe_buffer<value_t>(0,
                                                       rmm::cuda_stream_view{}))&& values_to_insert)
{
  pimpl->print();
  pimpl->insert(handle, type, edge_ids_to_insert, std::move(values_to_insert));
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
search_container_t<edge_type_t, edge_id_t, value_t>::lookup_src_dst_from_edge_id_and_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup,
  bool multi_gpu) const
{
  return pimpl->lookup_src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_type_to_lookup, multi_gpu);
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
search_container_t<edge_type_t, edge_id_t, value_t>::lookup_src_dst_from_edge_id_and_type(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup,
  bool multi_gpu) const
{
  return pimpl->lookup_src_dst_from_edge_id_and_type(
    handle, edge_ids_to_lookup, edge_types_to_lookup, multi_gpu);
}

template class search_container_t<int32_t, int32_t, thrust::tuple<int32_t, int32_t>>;

}  // namespace cugraph