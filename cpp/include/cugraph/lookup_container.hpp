#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <unordered_map>

namespace cugraph {

// ------- Don't remove----
// template <typename edge_type_t, typename edge_id_t, typename value_t>
// struct impl;
//---------------------

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
  search_container_t(const search_container_t&);

  void insert(edge_type_t, edge_id_t, value_t);
};

}  // namespace cugraph
