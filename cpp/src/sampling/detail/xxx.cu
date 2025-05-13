/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <cuda/std/optional>
#include <thrust/tuple.h>

template <typename vertex_t, typename EdgeProperties, typename label_t>
struct format_gather_edges_return_t {
  using return_type = std::conditional_t<
    std::is_same_v<label_t, cuda::std::nullopt_t>,
    std::conditional_t<std::is_same_v<EdgeProperties, cuda::std::nullopt_t>,
                       thrust::tuple<vertex_t, vertex_t>,
                       std::conditional_t<cugraph::is_thrust_tuple<EdgeProperties>::value,
                                          decltype(cugraph::thrust_tuple_cat(
                                            thrust::tuple<vertex_t, vertex_t>{}, EdgeProperties{})),
                                          thrust::tuple<vertex_t, vertex_t, EdgeProperties>>>,
    std::conditional_t<
      std::is_same_v<EdgeProperties, cuda::std::nullopt_t>,
      thrust::tuple<vertex_t, vertex_t, label_t>,
      std::conditional_t<cugraph::is_thrust_tuple<EdgeProperties>::value,
                         decltype(cugraph::thrust_tuple_cat(thrust::tuple<vertex_t, vertex_t>{},
                                                            thrust::tuple<vertex_t, vertex_t>{},
                                                            EdgeProperties{},
                                                            thrust::tuple<label_t>{})),
                         thrust::tuple<vertex_t, vertex_t, EdgeProperties, label_t>>>>;
};

int main(int argc, char** argv)
{
  format_gather_edges_return_t<int32_t, int32_t, int32_t> x;
  format_gather_edges_return_t<int32_t, cuda::std::nullopt_t, int32_t> y;
  format_gather_edges_return_t<int32_t, thrust::tuple<int32_t, int32_t>, int32_t> z;
}

// nvcc -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -I/home/coder/cugraph/cpp/include
// -I/home/coder/cugraph/cpp/build/_deps/cccl-src/thrust/thrust/cmake/../..
// -I/home/coder/cugraph/cpp/build/_deps/cccl-src/libcudacxx/lib/cmake/libcudacxx/../../../include
// -I/home/coder/cugraph/cpp/build/_deps/cccl-src/cub/cub/cmake/../..
// -I/home/coder/cugraph/cpp/build/_deps/cuco-src/include -isystem
// /home/coder/.conda/envs/rapids/include -isystem
// /home/coder/.conda/envs/rapids/targets/x86_64-linux/include -Wno-deprecated-gpu-targets -O3
// -DNDEBUG -std=c++17 "--generate-code=arch=compute_70,code=[sm_70]" -Xcompiler=-fPIC
// --expt-extended-lambda --expt-relaxed-constexpr -Werror=cross-execution-space-call
// -Wno-deprecated-declarations -Xptxas=--disable-warnings
// -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable -x cu xxx.cu
