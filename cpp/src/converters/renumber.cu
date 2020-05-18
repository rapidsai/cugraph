/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include "renumber.cuh"

namespace cugraph {

template <typename VT, typename ET>
std::unique_ptr<rmm::device_buffer> renumber_vertices(
                                                      ET number_of_edges,
                                                      VT const *src,
                                                      VT const *dst,
                                                      VT *src_renumbered,
                                                      VT *dst_renumbered,
                                                      ET *map_size,
                                                      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())

{
  //
  //  For now, let's just specify a default value of the hash size.
  //  This should be configurable.
  //
  //  FIXME:  cudf has a hash table implementation (moving to cuCollections)
  //          that is dynamic.  We should use it instead, it will be faster
  //          and dynamically adjust to data sizes.
  //
  int hash_size = 8191;

  return cugraph::detail::renumber_vertices(number_of_edges,
                                            src,
                                            dst,
                                            src_renumbered,
                                            dst_renumbered,
                                            map_size,
                                            cugraph::detail::HashFunctionObjectInt(hash_size),
                                            thrust::less<int32_t>(),
                                            mr);
}

template std::unique_ptr<rmm::device_buffer> renumber_vertices(int32_t, int32_t const *, int32_t const *, int32_t *, int32_t *, int32_t *, rmm::mr::device_memory_resource *);

}  // namespace cugraph
