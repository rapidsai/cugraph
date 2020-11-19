#pragma once

namespace cugraph {
namespace experimental {

template class graph_view_t<int32_t, int32_t, float, true, true, void>;
template class graph_view_t<int32_t, int32_t, float, true, false, void>;
template class graph_view_t<int32_t, int32_t, float, false, true, void>;
template class graph_view_t<int32_t, int32_t, float, false, false, void>;
template class graph_view_t<int32_t, int32_t, double, true, true, void>;
template class graph_view_t<int32_t, int32_t, double, true, false, void>;
template class graph_view_t<int32_t, int32_t, double, false, true, void>;
template class graph_view_t<int32_t, int32_t, double, false, false, void>;
template class graph_view_t<int32_t, int64_t, float, true, true, void>;
template class graph_view_t<int32_t, int64_t, float, true, false, void>;
template class graph_view_t<int32_t, int64_t, float, false, true, void>;
template class graph_view_t<int32_t, int64_t, float, false, false, void>;
template class graph_view_t<int32_t, int64_t, double, true, true, void>;
template class graph_view_t<int32_t, int64_t, double, true, false, void>;
template class graph_view_t<int32_t, int64_t, double, false, true, void>;
template class graph_view_t<int32_t, int64_t, double, false, false, void>;
template class graph_view_t<int64_t, int64_t, float, true, true, void>;
template class graph_view_t<int64_t, int64_t, float, true, false, void>;
template class graph_view_t<int64_t, int64_t, float, false, true, void>;
template class graph_view_t<int64_t, int64_t, float, false, false, void>;
template class graph_view_t<int64_t, int64_t, double, true, true, void>;
template class graph_view_t<int64_t, int64_t, double, true, false, void>;
template class graph_view_t<int64_t, int64_t, double, false, true, void>;
template class graph_view_t<int64_t, int64_t, double, false, false, void>;

}  // namespace experimental
}  // namespace cugraph
