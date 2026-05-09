/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

namespace CUGRAPH_EXPORT cugraph {
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
}  // namespace CUGRAPH_EXPORT cugraph
