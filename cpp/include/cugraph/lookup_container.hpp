#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <unordered_map>

namespace cugraph {

// ------- Don't remove----
// template <typename edge_type_t, typename edge_id_t, typename value_t>
// struct impl;
//---------------------

namespace detail {

template <typename TupleType, std::size_t... Is>
constexpr TupleType invalid_of_thrust_tuple_of_integral(std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    cugraph::invalid_idx<typename thrust::tuple_element<Is, TupleType>::type>::value...);
}
}  // namespace detail

template <typename TupleType>
constexpr TupleType invalid_of_thrust_tuple_of_integral()
{
  return detail::invalid_of_thrust_tuple_of_integral<TupleType>(
    std::make_index_sequence<thrust::tuple_size<TupleType>::value>());
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
class search_container_t {
  template <typename _edge_type_t, typename _edge_id_t, typename _value_t>
  struct impl;
  std::unique_ptr<impl<edge_type_t, edge_id_t, value_t>> pimpl;

 public:
  using edge_type_type = edge_type_t;
  using edge_id_type   = edge_id_t;
  using value_type     = value_t;

  static_assert(std::is_arithmetic_v<edge_type_t>);
  static_assert(std::is_arithmetic_v<edge_id_t>);
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

  ~search_container_t();
  search_container_t();
  search_container_t(raft::handle_t const& handle,
                     std::vector<edge_type_t> types,
                     std::vector<edge_id_t> type_counts);
  search_container_t(const search_container_t&);

  void insert(edge_type_t, edge_id_t, value_t);

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  lookup_src_dst_from_edge_id_and_type(raft::handle_t const& handle,
                                       raft::device_span<edge_id_t const> edge_ids_to_lookup,
                                       edge_type_t edge_type_to_lookup,
                                       bool multi_gpu) const;

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  lookup_src_dst_from_edge_id_and_type(raft::handle_t const& handle,
                                       raft::device_span<edge_id_t const> edge_ids_to_lookup,
                                       raft::device_span<edge_type_t const> edge_types_to_lookup,
                                       bool multi_gpu) const;
};

}  // namespace cugraph
