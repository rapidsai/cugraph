/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph::gather in thrust_wrappers/gather.hpp.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/gather.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

template <typename InputIterator, typename OutputIterator, typename MapType>
OutputIterator gather_impl(rmm::exec_policy const& policy,
                           MapType const* map_first,
                           MapType const* map_last,
                           InputIterator input_first,
                           OutputIterator output_first)
{
  return thrust::gather(policy, map_first, map_last, input_first, output_first);
}

template <typename InputIterator, typename OutputIterator, typename MapType>
OutputIterator gather_impl(rmm::exec_policy_nosync const& policy,
                           MapType const* map_first,
                           MapType const* map_last,
                           InputIterator input_first,
                           OutputIterator output_first)
{
  return thrust::gather(policy, map_first, map_last, input_first, output_first);
}

#define CUGRAPH_GATHER_SCALAR_MAP_INST(ScalarType, MapType)                                 \
  template CUGRAPH_EXPORT ScalarType* gather_impl<ScalarType const*, ScalarType*, MapType>( \
    rmm::exec_policy const& policy,                                                         \
    MapType const* map_first,                                                               \
    MapType const* map_last,                                                                \
    ScalarType const* input_first,                                                          \
    ScalarType* output_first);                                                              \
  template CUGRAPH_EXPORT ScalarType* gather_impl<ScalarType const*, ScalarType*, MapType>( \
    rmm::exec_policy_nosync const& policy,                                                  \
    MapType const* map_first,                                                               \
    MapType const* map_last,                                                                \
    ScalarType const* input_first,                                                          \
    ScalarType* output_first);                                                              \
  template CUGRAPH_EXPORT ScalarType* gather_impl<ScalarType*, ScalarType*, MapType>(       \
    rmm::exec_policy const& policy,                                                         \
    MapType const* map_first,                                                               \
    MapType const* map_last,                                                                \
    ScalarType* input_first,                                                                \
    ScalarType* output_first);                                                              \
  template CUGRAPH_EXPORT ScalarType* gather_impl<ScalarType*, ScalarType*, MapType>(       \
    rmm::exec_policy_nosync const& policy,                                                  \
    MapType const* map_first,                                                               \
    MapType const* map_last,                                                                \
    ScalarType* input_first,                                                                \
    ScalarType* output_first)

#define CUGRAPH_GATHER_SCALAR_INST(ScalarType)              \
  CUGRAPH_GATHER_SCALAR_MAP_INST(ScalarType, std::size_t);  \
  CUGRAPH_GATHER_SCALAR_MAP_INST(ScalarType, std::int32_t); \
  CUGRAPH_GATHER_SCALAR_MAP_INST(ScalarType, std::int64_t)

CUGRAPH_GATHER_SCALAR_INST(std::int32_t);
CUGRAPH_GATHER_SCALAR_INST(std::int64_t);
CUGRAPH_GATHER_SCALAR_INST(float);
CUGRAPH_GATHER_SCALAR_INST(double);
CUGRAPH_GATHER_SCALAR_INST(std::size_t);

#undef CUGRAPH_GATHER_SCALAR_INST
#undef CUGRAPH_GATHER_SCALAR_MAP_INST

template <typename MapType>
using shift_left_transform_map_iterator_const_t =
  cuda::transform_iterator<shift_left_t<MapType>, MapType const*>;

template <typename MapType>
using shift_left_transform_map_iterator_mutable_t =
  cuda::transform_iterator<shift_left_t<MapType>, MapType*>;

template <typename InputIterator, typename OutputIterator, typename MapIterator>
OutputIterator gather_shift_left_impl(rmm::exec_policy const& policy,
                                      MapIterator map_first,
                                      MapIterator map_last,
                                      InputIterator input_first,
                                      OutputIterator output_first)
{
  return thrust::gather(policy, map_first, map_last, input_first, output_first);
}

template <typename InputIterator, typename OutputIterator, typename MapIterator>
OutputIterator gather_shift_left_impl(rmm::exec_policy_nosync const& policy,
                                      MapIterator map_first,
                                      MapIterator map_last,
                                      InputIterator input_first,
                                      OutputIterator output_first)
{
  return thrust::gather(policy, map_first, map_last, input_first, output_first);
}

#define CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, MapType)          \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType const*,                                     \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy const& policy,                                             \
    shift_left_transform_map_iterator_const_t<MapType> map_first,               \
    shift_left_transform_map_iterator_const_t<MapType> map_last,                \
    ScalarType const* input_first,                                              \
    ScalarType* output_first);                                                  \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType const*,                                     \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy_nosync const& policy,                                      \
    shift_left_transform_map_iterator_const_t<MapType> map_first,               \
    shift_left_transform_map_iterator_const_t<MapType> map_last,                \
    ScalarType const* input_first,                                              \
    ScalarType* output_first);                                                  \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType const*,                                     \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy const& policy,                                             \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,             \
    shift_left_transform_map_iterator_mutable_t<MapType> map_last,              \
    ScalarType const* input_first,                                              \
    ScalarType* output_first);                                                  \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType const*,                                     \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy_nosync const& policy,                                      \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,             \
    shift_left_transform_map_iterator_mutable_t<MapType> map_last,              \
    ScalarType const* input_first,                                              \
    ScalarType* output_first);                                                  \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType*,                                           \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy const& policy,                                             \
    shift_left_transform_map_iterator_const_t<MapType> map_first,               \
    shift_left_transform_map_iterator_const_t<MapType> map_last,                \
    ScalarType* input_first,                                                    \
    ScalarType* output_first);                                                  \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType*,                                           \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_const_t<MapType>>(   \
    rmm::exec_policy_nosync const& policy,                                      \
    shift_left_transform_map_iterator_const_t<MapType> map_first,               \
    shift_left_transform_map_iterator_const_t<MapType> map_last,                \
    ScalarType* input_first,                                                    \
    ScalarType* output_first);                                                  \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType*,                                           \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy const& policy,                                             \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,             \
    shift_left_transform_map_iterator_mutable_t<MapType> map_last,              \
    ScalarType* input_first,                                                    \
    ScalarType* output_first);                                                  \
  template CUGRAPH_EXPORT ScalarType*                                           \
  gather_shift_left_impl<ScalarType*,                                           \
                         ScalarType*,                                           \
                         shift_left_transform_map_iterator_mutable_t<MapType>>( \
    rmm::exec_policy_nosync const& policy,                                      \
    shift_left_transform_map_iterator_mutable_t<MapType> map_first,             \
    shift_left_transform_map_iterator_mutable_t<MapType> map_last,              \
    ScalarType* input_first,                                                    \
    ScalarType* output_first)

#define CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_INST(ScalarType)              \
  CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, std::size_t);  \
  CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, std::int32_t); \
  CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_MAP_INST(ScalarType, std::int64_t)

CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_INST(std::int32_t);
CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_INST(std::int64_t);
CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_INST(float);
CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_INST(double);
CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_INST(std::size_t);

#undef CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_INST
#undef CUGRAPH_GATHER_SHIFT_LEFT_SCALAR_MAP_INST

}  // namespace detail
}  // namespace cugraph
