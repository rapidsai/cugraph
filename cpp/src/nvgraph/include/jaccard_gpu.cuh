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
// Jaccard symilarity edge weights
// Author: Alexandre Fender afender@nvidia.com and Maxim Naumov.

#pragma once

namespace nvlouvain 
{
template <bool weighted, typename T> 
int jaccard(int n, int e, int *csrPtr, int *csrInd, T * csrVal, T *v, T *work, T gamma, T *weight_i, T *weight_s, T *weight_j);
}
