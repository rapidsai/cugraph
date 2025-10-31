/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cugraph {
namespace legacy {
extern template class GraphViewBase<int32_t, int32_t, float>;
extern template class GraphViewBase<int32_t, int32_t, double>;
extern template class GraphViewBase<int32_t, int64_t, float>;
extern template class GraphViewBase<int32_t, int64_t, double>;
extern template class GraphViewBase<int64_t, int32_t, float>;
extern template class GraphViewBase<int64_t, int32_t, double>;
extern template class GraphViewBase<int64_t, int64_t, float>;
extern template class GraphViewBase<int64_t, int64_t, double>;
extern template class GraphCompressedSparseBaseView<int32_t, int32_t, float>;
extern template class GraphCompressedSparseBaseView<int32_t, int32_t, double>;
extern template class GraphCompressedSparseBaseView<int32_t, int64_t, float>;
extern template class GraphCompressedSparseBaseView<int32_t, int64_t, double>;
extern template class GraphCompressedSparseBaseView<int64_t, int32_t, float>;
extern template class GraphCompressedSparseBaseView<int64_t, int32_t, double>;
extern template class GraphCompressedSparseBaseView<int64_t, int64_t, float>;
extern template class GraphCompressedSparseBaseView<int64_t, int64_t, double>;
extern template class GraphCompressedSparseBase<int32_t, int32_t, float>;
extern template class GraphCompressedSparseBase<int32_t, int32_t, double>;
extern template class GraphCompressedSparseBase<int32_t, int64_t, float>;
extern template class GraphCompressedSparseBase<int32_t, int64_t, double>;
extern template class GraphCompressedSparseBase<int64_t, int32_t, float>;
extern template class GraphCompressedSparseBase<int64_t, int32_t, double>;
extern template class GraphCompressedSparseBase<int64_t, int64_t, float>;
extern template class GraphCompressedSparseBase<int64_t, int64_t, double>;
extern template class GraphCOOView<int32_t, int32_t, float>;
extern template class GraphCOOView<int32_t, int32_t, double>;
extern template class GraphCOOView<int32_t, int64_t, float>;
extern template class GraphCOOView<int32_t, int64_t, double>;
extern template class GraphCOOView<int64_t, int32_t, float>;
extern template class GraphCOOView<int64_t, int32_t, double>;
extern template class GraphCOOView<int64_t, int64_t, float>;
extern template class GraphCOOView<int64_t, int64_t, double>;
extern template class GraphCSRView<int32_t, int32_t, float>;
extern template class GraphCSRView<int32_t, int32_t, double>;
extern template class GraphCSRView<int32_t, int64_t, float>;
extern template class GraphCSRView<int32_t, int64_t, double>;
extern template class GraphCSRView<int64_t, int32_t, float>;
extern template class GraphCSRView<int64_t, int32_t, double>;
extern template class GraphCSRView<int64_t, int64_t, float>;
extern template class GraphCSRView<int64_t, int64_t, double>;
extern template class GraphCOO<int32_t, int32_t, float>;
extern template class GraphCOO<int32_t, int32_t, double>;
extern template class GraphCOO<int32_t, int64_t, float>;
extern template class GraphCOO<int32_t, int64_t, double>;
extern template class GraphCOO<int64_t, int32_t, float>;
extern template class GraphCOO<int64_t, int32_t, double>;
extern template class GraphCOO<int64_t, int64_t, float>;
extern template class GraphCOO<int64_t, int64_t, double>;
extern template class GraphCSR<int32_t, int32_t, float>;
extern template class GraphCSR<int32_t, int32_t, double>;
extern template class GraphCSR<int32_t, int64_t, float>;
extern template class GraphCSR<int32_t, int64_t, double>;
extern template class GraphCSR<int64_t, int32_t, float>;
extern template class GraphCSR<int64_t, int32_t, double>;
extern template class GraphCSR<int64_t, int64_t, float>;
extern template class GraphCSR<int64_t, int64_t, double>;
}  // namespace legacy
}  // namespace cugraph
