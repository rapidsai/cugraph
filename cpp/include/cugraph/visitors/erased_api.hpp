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

// Andrei Schaffer, aschaffer@nvidia.com
//

/**
 * @brief Set of type-erased wrappers, following the (almost) universal general signature:
 * graph_envelope reference; erased_pack_t pack of erased arguments, that the caller is responsible
 * to set correctly (FIXME: handshake protocol must be put in place); return set;
 */

#pragma once

namespace cugraph {
namespace api {

using namespace cugraph::visitors;

/**
 * @brief Type-erased BFS wrapper.
 *
 * @param[in] g graph_envelope reference;
 * @param[in] ep erased_pack_t pack of erased arguments, that the caller is responsible to set
 * correctly (FIXME: handshake protocol must be put in place);
 * @return return set;
 */
return_t bfs(graph_envelope_t const& g, erased_pack_t& ep);

// TODO: more to follow...

}  // namespace api
}  // namespace cugraph
