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
template class GraphCSCView<int32_t, int32_t, float>;
template class GraphCSCView<int32_t, int32_t, double>;
template class GraphCSCView<int32_t, int64_t, float>;
template class GraphCSCView<int32_t, int64_t, double>;
template class GraphCSCView<int64_t, int64_t, float>;
template class GraphCSCView<int64_t, int64_t, double>;
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
template class GraphCSC<int32_t, int32_t, float>;
template class GraphCSC<int32_t, int32_t, double>;
template class GraphCSC<int32_t, int64_t, float>;
template class GraphCSC<int32_t, int64_t, double>;
template class GraphCSC<int64_t, int64_t, float>;
template class GraphCSC<int64_t, int64_t, double>;
}  // namespace cugraph
