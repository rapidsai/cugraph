/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utilities/property_generator_utilities_impl.cuh"

#include <cuda/std/tuple>

namespace cugraph {
namespace test {

template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, true>, bool>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, true>, int32_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, true>, int64_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, true>,
                         cuda::std::tuple<int, float>>;

template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, true>, bool>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, true>, int32_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, true>, int64_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, true>,
                         cuda::std::tuple<int, float>>;

template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, true>, bool>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, true>, int32_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, true>, int64_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, true>,
                         cuda::std::tuple<int, float>>;

template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, true>, bool>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, true>, int32_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, true>, int64_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, true>,
                         cuda::std::tuple<int, float>>;

}  // namespace test
}  // namespace cugraph
