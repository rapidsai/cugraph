/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cugraph {
namespace legacy {
template class GraphViewBase<int32_t, int32_t, float>;
template class GraphViewBase<int32_t, int32_t, double>;
template class GraphViewBase<int32_t, int64_t, float>;
template class GraphViewBase<int32_t, int64_t, double>;
template class GraphViewBase<int64_t, int64_t, float>;
template class GraphViewBase<int64_t, int64_t, double>;
template class GraphCompressedSparseBaseView<int32_t, int32_t, float>;
template class GraphCompressedSparseBaseView<int32_t, int32_t, double>;
template class GraphCompressedSparseBaseView<int32_t, int64_t, float>;
template class GraphCompressedSparseBaseView<int32_t, int64_t, double>;
template class GraphCompressedSparseBaseView<int64_t, int64_t, float>;
template class GraphCompressedSparseBaseView<int64_t, int64_t, double>;
template class GraphCompressedSparseBase<int32_t, int32_t, float>;
template class GraphCompressedSparseBase<int32_t, int32_t, double>;
template class GraphCompressedSparseBase<int32_t, int64_t, float>;
template class GraphCompressedSparseBase<int32_t, int64_t, double>;
template class GraphCompressedSparseBase<int64_t, int64_t, float>;
template class GraphCompressedSparseBase<int64_t, int64_t, double>;
template class GraphCOOView<int32_t, int32_t, float>;
template class GraphCOOView<int32_t, int32_t, double>;
template class GraphCOOView<int32_t, int64_t, float>;
template class GraphCOOView<int32_t, int64_t, double>;
template class GraphCOOView<int64_t, int64_t, float>;
template class GraphCOOView<int64_t, int64_t, double>;
template class GraphCSRView<int32_t, int32_t, float>;
template class GraphCSRView<int32_t, int32_t, double>;
template class GraphCSRView<int32_t, int64_t, float>;
template class GraphCSRView<int32_t, int64_t, double>;
template class GraphCSRView<int64_t, int64_t, float>;
template class GraphCSRView<int64_t, int64_t, double>;
template class GraphCOO<int32_t, int32_t, float>;
template class GraphCOO<int32_t, int32_t, double>;
template class GraphCOO<int32_t, int64_t, float>;
template class GraphCOO<int32_t, int64_t, double>;
template class GraphCOO<int64_t, int64_t, float>;
template class GraphCOO<int64_t, int64_t, double>;
template class GraphCSR<int32_t, int32_t, float>;
template class GraphCSR<int32_t, int32_t, double>;
template class GraphCSR<int32_t, int64_t, float>;
template class GraphCSR<int32_t, int64_t, double>;
template class GraphCSR<int64_t, int64_t, float>;
template class GraphCSR<int64_t, int64_t, double>;
}  // namespace legacy
}  // namespace cugraph
