/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
