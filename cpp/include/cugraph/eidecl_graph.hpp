/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
