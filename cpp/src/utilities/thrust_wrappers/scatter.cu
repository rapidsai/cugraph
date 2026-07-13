/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/thrust_wrappers/scatter.hpp.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/thrust_wrappers/scatter.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/scatter.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

template <typename InputIterator, typename OutputIterator, typename MapType>
void scatter_impl(rmm::exec_policy const& policy,
                  InputIterator input_first,
                  InputIterator input_last,
                  MapType const* map_first,
                  OutputIterator output_first)
{
  thrust::scatter(policy, input_first, input_last, map_first, output_first);
}

template <typename InputIterator, typename OutputIterator, typename MapType>
void scatter_impl(rmm::exec_policy_nosync const& policy,
                  InputIterator input_first,
                  InputIterator input_last,
                  MapType const* map_first,
                  OutputIterator output_first)
{
  thrust::scatter(policy, input_first, input_last, map_first, output_first);
}

#define CUGRAPH_SCATTER_SCALAR_MAP_INST(ScalarType, MapType)                          \
  template CUGRAPH_EXPORT void scatter_impl<ScalarType const*, ScalarType*, MapType>( \
    rmm::exec_policy const& policy,                                                   \
    ScalarType const* input_first,                                                    \
    ScalarType const* input_last,                                                     \
    MapType const* map_first,                                                         \
    ScalarType* output_first);                                                        \
  template CUGRAPH_EXPORT void scatter_impl<ScalarType const*, ScalarType*, MapType>( \
    rmm::exec_policy_nosync const& policy,                                            \
    ScalarType const* input_first,                                                    \
    ScalarType const* input_last,                                                     \
    MapType const* map_first,                                                         \
    ScalarType* output_first);                                                        \
  template CUGRAPH_EXPORT void scatter_impl<ScalarType*, ScalarType*, MapType>(       \
    rmm::exec_policy const& policy,                                                   \
    ScalarType* input_first,                                                          \
    ScalarType* input_last,                                                           \
    MapType const* map_first,                                                         \
    ScalarType* output_first);                                                        \
  template CUGRAPH_EXPORT void scatter_impl<ScalarType*, ScalarType*, MapType>(       \
    rmm::exec_policy_nosync const& policy,                                            \
    ScalarType* input_first,                                                          \
    ScalarType* input_last,                                                           \
    MapType const* map_first,                                                         \
    ScalarType* output_first)

#define CUGRAPH_SCATTER_SCALAR_INST(ScalarType)              \
  CUGRAPH_SCATTER_SCALAR_MAP_INST(ScalarType, std::size_t);  \
  CUGRAPH_SCATTER_SCALAR_MAP_INST(ScalarType, std::int32_t); \
  CUGRAPH_SCATTER_SCALAR_MAP_INST(ScalarType, std::int64_t)

CUGRAPH_SCATTER_SCALAR_INST(std::int32_t);
CUGRAPH_SCATTER_SCALAR_INST(std::int64_t);
CUGRAPH_SCATTER_SCALAR_INST(float);
CUGRAPH_SCATTER_SCALAR_INST(double);
CUGRAPH_SCATTER_SCALAR_INST(std::size_t);

#undef CUGRAPH_SCATTER_SCALAR_INST
#undef CUGRAPH_SCATTER_SCALAR_MAP_INST

template <typename MapType>
using shift_left_transform_map_iterator_const_t =
  cuda::transform_iterator<shift_left_t<MapType>, MapType const*>;

template <typename MapType>
using shift_left_transform_map_iterator_mutable_t =
  cuda::transform_iterator<shift_left_t<MapType>, MapType*>;

template <typename InputIterator, typename OutputIterator, typename MapIterator>
void scatter_shift_left_impl(rmm::exec_policy const& policy,
                             InputIterator input_first,
                             InputIterator input_last,
                             MapIterator map_first,
                             OutputIterator output_first)
{
  thrust::scatter(policy, input_first, input_last, map_first, output_first);
}

template <typename InputIterator, typename OutputIterator, typename MapIterator>
void scatter_shift_left_impl(rmm::exec_policy_nosync const& policy,
                             InputIterator input_first,
                             InputIterator input_last,
                             MapIterator map_first,
                             OutputIterator output_first)
{
  thrust::scatter(policy, input_first, input_last, map_first, output_first);
}

#define CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, MapType)          \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType const*,                                     \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy const& policy,                                              \
    ScalarType const* input_first,                                               \
    ScalarType const* input_last,                                                \
    shift_left_transform_map_iterator_const_t<MapType> map_first,                \
    ScalarType* output_first);                                                   \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType const*,                                     \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy_nosync const& policy,                                       \
    ScalarType const* input_first,                                               \
    ScalarType const* input_last,                                                \
    shift_left_transform_map_iterator_const_t<MapType> map_first,                \
    ScalarType* output_first);                                                   \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType const*,                                     \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy const& policy,                                              \
    ScalarType const* input_first,                                               \
    ScalarType const* input_last,                                                \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,              \
    ScalarType* output_first);                                                   \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType const*,                                     \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy_nosync const& policy,                                       \
    ScalarType const* input_first,                                               \
    ScalarType const* input_last,                                                \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,              \
    ScalarType* output_first);                                                   \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType*,                                           \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy const& policy,                                              \
    ScalarType* input_first,                                                     \
    ScalarType* input_last,                                                      \
    shift_left_transform_map_iterator_const_t<MapType> map_first,                \
    ScalarType* output_first);                                                   \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType*,                                           \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy_nosync const& policy,                                       \
    ScalarType* input_first,                                                     \
    ScalarType* input_last,                                                      \
    shift_left_transform_map_iterator_const_t<MapType> map_first,                \
    ScalarType* output_first);                                                   \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType*,                                           \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy const& policy,                                              \
    ScalarType* input_first,                                                     \
    ScalarType* input_last,                                                      \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,              \
    ScalarType* output_first);                                                   \
  template CUGRAPH_EXPORT void                                                   \
  scatter_shift_left_impl<ScalarType*,                                           \
                          ScalarType*,                                           \
                          shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy_nosync const& policy,                                       \
    ScalarType* input_first,                                                     \
    ScalarType* input_last,                                                      \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,              \
    ScalarType* output_first)

#define CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_INST(ScalarType)              \
  CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, std::size_t);  \
  CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, std::int32_t); \
  CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, std::int64_t)

CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_INST(std::int32_t);
CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_INST(std::int64_t);
CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_INST(float);
CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_INST(double);
CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_INST(std::size_t);

#undef CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_INST
#undef CUGRAPH_SCATTER_SHIFT_LEFT_SCALAR_MAP_INST

}  // namespace detail
}  // namespace cugraph
