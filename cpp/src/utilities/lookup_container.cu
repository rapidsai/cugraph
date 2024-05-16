
#include "prims/kv_store.cuh"

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
struct edge_type_and_id_search_container_t_2<edge_type_t, edge_id_t, value_t>::impl {
  static_assert(std::is_same_v<edge_type_t, _edge_type_t>);
  static_assert(std::is_same_v<edge_id_t, _edge_id_t>);
  static_assert(std::is_same_v<value_t, _value_t>);
  using container_t =
    std::unordered_map<edge_type_t,
                       cugraph::kv_store_t<edge_id_t, value_t, false /*use_binary_search*/>>;

  static_assert(std::is_arithmetic_v<edge_type_t>);
  static_assert(std::is_arithmetic_v<edge_id_t>);
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

  void insert(edge_type_t et, edge_id_t ei, value_t v) { std::cout << "From impl insert\n"; }
};

template <typename edge_type_t, typename edge_id_t, typename value_t>
edge_type_and_id_search_container_t_2<edge_type_t, edge_id_t, value_t>::
  edge_type_and_id_search_container_t_2()
  : pimpl{std::make_unique<impl<edge_type_t, edge_id_t, value_t>>()}
{
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
edge_type_and_id_search_container_t_2<edge_type_t, edge_id_t, value_t>::
  edge_type_and_id_search_container_t_2(const edge_type_and_id_search_container_t_2&)
{
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
edge_type_and_id_search_container_t_2<edge_type_t, edge_id_t, value_t>::
  ~edge_type_and_id_search_container_t_2()
{
}

template <typename edge_type_t, typename edge_id_t, typename value_t>
void edge_type_and_id_search_container_t_2<edge_type_t, edge_id_t, value_t>::insert(edge_type_t et,
                                                                                    edge_id_t ei,
                                                                                    value_t v)
{
  pimpl->insert(et, ei, v);
}

template class edge_type_and_id_search_container_t_2<int32_t, int32_t, int32_t>;

}  // namespace cugraph