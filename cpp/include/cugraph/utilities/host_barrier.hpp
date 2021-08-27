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

#include <raft/handle.hpp>

namespace cugraph {

// FIXME: a temporary hack till UCC is integrated into RAFT (so we can use UCC barrier for DASK and
// MPI barrier for MPI)
void host_barrier(raft::comms::comms_t const& comm, rmm::cuda_stream_view stream_view);

}  // namespace cugraph
