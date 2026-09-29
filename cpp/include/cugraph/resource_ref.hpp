/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/memory_resource>

namespace cugraph {

/**
 * @brief Stream-ordered reference to a device-accessible memory resource
 */
using device_resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;

}  // namespace cugraph
