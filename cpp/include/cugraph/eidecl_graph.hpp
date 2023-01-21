/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
extern template class graph_t<int32_t, int32_t, true, true, void>;
extern template class graph_t<int32_t, int32_t, true, false, void>;
extern template class graph_t<int32_t, int32_t, false, true, void>;
extern template class graph_t<int32_t, int32_t, false, false, void>;
extern template class graph_t<int32_t, int64_t, true, true, void>;
extern template class graph_t<int32_t, int64_t, true, false, void>;
extern template class graph_t<int32_t, int64_t, false, true, void>;
extern template class graph_t<int32_t, int64_t, false, false, void>;
extern template class graph_t<int64_t, int32_t, true, true, void>;
extern template class graph_t<int64_t, int32_t, true, false, void>;
extern template class graph_t<int64_t, int32_t, false, true, void>;
extern template class graph_t<int64_t, int32_t, false, false, void>;
extern template class graph_t<int64_t, int64_t, true, true, void>;
extern template class graph_t<int64_t, int64_t, true, false, void>;
extern template class graph_t<int64_t, int64_t, false, true, void>;
extern template class graph_t<int64_t, int64_t, false, false, void>;
extern template class graph_view_t<int32_t, int32_t, true, true, void>;
extern template class graph_view_t<int32_t, int32_t, true, false, void>;
extern template class graph_view_t<int32_t, int32_t, false, true, void>;
extern template class graph_view_t<int32_t, int32_t, false, false, void>;
extern template class graph_view_t<int32_t, int64_t, true, true, void>;
extern template class graph_view_t<int32_t, int64_t, true, false, void>;
extern template class graph_view_t<int32_t, int64_t, false, true, void>;
extern template class graph_view_t<int32_t, int64_t, false, false, void>;
extern template class graph_view_t<int64_t, int32_t, true, true, void>;
extern template class graph_view_t<int64_t, int32_t, true, false, void>;
extern template class graph_view_t<int64_t, int32_t, false, true, void>;
extern template class graph_view_t<int64_t, int32_t, false, false, void>;
extern template class graph_view_t<int64_t, int64_t, true, true, void>;
extern template class graph_view_t<int64_t, int64_t, true, false, void>;
extern template class graph_view_t<int64_t, int64_t, false, true, void>;
extern template class graph_view_t<int64_t, int64_t, false, false, void>;
}  // namespace cugraph
