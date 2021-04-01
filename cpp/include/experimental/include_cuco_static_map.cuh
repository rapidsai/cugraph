/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

// "FIXME": remove the guards below and references to CUCO_STATIC_MAP_DEFINED
//
// cuco/static_map.cuh depends on features not supported on or before Pascal.
//
// If we build for sm_60 or before, the inclusion of cuco/static_map.cuh wil
// result in compilation errors.
//
// If we're Pascal or before we do nothing here and will suppress including
// some code below.  If we are later than Pascal we define CUCO_STATIC_MAP_DEFINED
// which will result in the full implementation being pulled in.
//
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#else
#define CUCO_STATIC_MAP_DEFINED
#include <cuco/static_map.cuh>
#endif
