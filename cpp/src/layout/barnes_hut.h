/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "bh_kernels.h"
#include "utils.h"

namespace cugraph {
namespace ForceAtlas2 {

template <typename IndexType, typename ValueType>
void barnes_hut(const IndexType *row, const IndexType *col,
        const ValueType *val, const int nnz, float *x_pos,
        float *y_pos, int max_iter, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float barnes_hut_theta, float scaling_ratio,
        bool strong_gravity_mode, float gravity) {
    return;
}

} // namespace ForceAtlas2
} // namespace cugraph
