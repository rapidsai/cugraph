

#include <community/detail/mis_impl.cuh>

namespace cugraph {
namespace detail {
rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, float, false, false> const& decision_graph_view);

rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, double, false, false> const& decision_graph_view);

rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, float, false, false> const& decision_graph_view);

rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, double, false, false> const& decision_graph_view);

rmm::device_uvector<int64_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, float, false, false> const& decision_graph_view);

rmm::device_uvector<int64_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, double, false, false> const& decision_graph_view);

}  // namespace detail
}  // namespace cugraph
