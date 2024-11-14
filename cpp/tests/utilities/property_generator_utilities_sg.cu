/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include "utilities/property_generator_utilities_impl.cuh"

namespace cugraph {
namespace test {

template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, false>, bool>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, false>, int32_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, false>, int64_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, false, false>,
                         thrust::tuple<int, float>>;

template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, false>, bool>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, false>, int32_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, false>, int64_t>;
template struct generate<cugraph::graph_view_t<int32_t, int32_t, true, false>,
                         thrust::tuple<int, float>>;

template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, false>, bool>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, false>, int32_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, false>, int64_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, false, false>,
                         thrust::tuple<int, float>>;

template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, false>, bool>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, false>, int32_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, false>, int64_t>;
template struct generate<cugraph::graph_view_t<int64_t, int64_t, true, false>,
                         thrust::tuple<int, float>>;

}  // namespace test
}  // namespace cugraph
