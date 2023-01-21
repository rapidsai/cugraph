/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {

namespace detail {

template <typename T>
T* iter_to_raw_ptr(T* ptr)
{
  return ptr;
}

template <typename T>
T* iter_to_raw_ptr(thrust::device_ptr<T> ptr)
{
  return thrust::raw_pointer_cast(ptr);
}

template <typename T>
auto iter_to_raw_ptr(thrust::detail::normal_iterator<thrust::device_ptr<T>> iter)
{
  return thrust::raw_pointer_cast(iter.base());
}

template <typename InputIterator, typename OutputValueType>
std::enable_if_t<std::is_same<OutputValueType, thrust::detail::any_assign>::value, void>
device_isend_impl(raft::comms::comms_t const& comm,
                  InputIterator input_first,
                  size_t count,
                  int dst,
                  int tag,
                  raft::comms::request_t* request)
{
  // no-op
}

template <typename InputIterator, typename OutputValueType>
std::enable_if_t<std::is_arithmetic<OutputValueType>::value, void> device_isend_impl(
  raft::comms::comms_t const& comm,
  InputIterator input_first,
  size_t count,
  int dst,
  int tag,
  raft::comms::request_t* request)
{
  static_assert(
    std::is_same<typename std::iterator_traits<InputIterator>::value_type, OutputValueType>::value);
  comm.isend(iter_to_raw_ptr(input_first), count, dst, tag, request);
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_isend_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           size_t count,
           int dst,
           int base_tag,
           raft::comms::request_t* requests) const
  {
    using output_value_t = typename thrust::
      tuple_element<I, typename std::iterator_traits<OutputIterator>::value_type>::type;
    auto tuple_element_input_first = thrust::get<I>(input_first.get_iterator_tuple());
    device_isend_impl<decltype(tuple_element_input_first), output_value_t>(
      comm, tuple_element_input_first, count, dst, static_cast<int>(base_tag + I), requests + I);
    device_isend_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, count, dst, base_tag, requests);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_isend_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           size_t count,
           int dst,
           int base_tag,
           raft::comms::request_t* requests) const
  {
  }
};

template <typename InputValueType, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_irecv_impl(raft::comms::comms_t const& comm,
                  OutputIterator output_first,
                  size_t count,
                  int src,
                  int tag,
                  raft::comms::request_t* request)
{
  // no-op
}

template <typename InputValueType, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_irecv_impl(raft::comms::comms_t const& comm,
                  OutputIterator output_first,
                  size_t count,
                  int src,
                  int tag,
                  raft::comms::request_t* request)
{
  static_assert(

    std::is_same<InputValueType, typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.irecv(iter_to_raw_ptr(output_first), count, src, tag, request);
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_irecv_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           OutputIterator output_first,
           size_t count,
           int src,
           int base_tag,
           raft::comms::request_t* requests) const
  {
    using input_value_t = typename thrust::
      tuple_element<I, typename std::iterator_traits<InputIterator>::value_type>::type;
    auto tuple_element_output_first = thrust::get<I>(output_first.get_iterator_tuple());
    device_irecv_impl<input_value_t, decltype(tuple_element_output_first)>(
      comm, tuple_element_output_first, count, src, static_cast<int>(base_tag + I), requests + I);
    device_irecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, output_first, count, src, base_tag, requests);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_irecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           OutputIterator output_first,
           size_t count,
           int src,
           int base_tag,
           raft::comms::request_t* requests) const
  {
  }
};

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_sendrecv_impl(raft::comms::comms_t const& comm,
                     InputIterator input_first,
                     size_t tx_count,
                     int dst,
                     OutputIterator output_first,
                     size_t rx_count,
                     int src,
                     rmm::cuda_stream_view stream_view)
{
  // no-op
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_sendrecv_impl(raft::comms::comms_t const& comm,
                     InputIterator input_first,
                     size_t tx_count,
                     int dst,
                     OutputIterator output_first,
                     size_t rx_count,
                     int src,
                     rmm::cuda_stream_view stream_view)
{
  using value_type = typename std::iterator_traits<InputIterator>::value_type;
  static_assert(
    std::is_same<typename std::iterator_traits<OutputIterator>::value_type, value_type>::value);
  comm.device_sendrecv(iter_to_raw_ptr(input_first),
                       tx_count,
                       dst,
                       iter_to_raw_ptr(output_first),
                       rx_count,
                       src,
                       stream_view.value());
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_sendrecv_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           size_t tx_count,
           int dst,
           OutputIterator output_first,
           size_t rx_count,
           int src,
           rmm::cuda_stream_view stream_view) const
  {
    using output_value_t = typename thrust::
      tuple_element<I, typename std::iterator_traits<OutputIterator>::value_type>::type;
    auto tuple_element_input_first  = thrust::get<I>(input_first.get_iterator_tuple());
    auto tuple_element_output_first = thrust::get<I>(output_first.get_iterator_tuple());
    device_sendrecv_impl<decltype(tuple_element_input_first), decltype(tuple_element_output_first)>(
      comm,
      tuple_element_input_first,
      tx_count,
      dst,
      tuple_element_output_first,
      rx_count,
      src,
      stream_view.value());
    device_sendrecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, tx_count, dst, output_first, rx_count, src, stream_view);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_sendrecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           size_t tx_count,
           int dst,
           OutputIterator output_first,
           size_t rx_count,
           int src,
           rmm::cuda_stream_view stream_view) const
  {
  }
};

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_multicast_sendrecv_impl(raft::comms::comms_t const& comm,
                               InputIterator input_first,
                               std::vector<size_t> const& tx_counts,
                               std::vector<size_t> const& tx_offsets,
                               std::vector<int> const& tx_dst_ranks,
                               OutputIterator output_first,
                               std::vector<size_t> const& rx_counts,
                               std::vector<size_t> const& rx_offsets,
                               std::vector<int> const& rx_src_ranks,
                               rmm::cuda_stream_view stream_view)
{
  // no-op
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_multicast_sendrecv_impl(raft::comms::comms_t const& comm,
                               InputIterator input_first,
                               std::vector<size_t> const& tx_counts,
                               std::vector<size_t> const& tx_offsets,
                               std::vector<int> const& tx_dst_ranks,
                               OutputIterator output_first,
                               std::vector<size_t> const& rx_counts,
                               std::vector<size_t> const& rx_offsets,
                               std::vector<int> const& rx_src_ranks,
                               rmm::cuda_stream_view stream_view)
{
  using value_type = typename std::iterator_traits<InputIterator>::value_type;
  static_assert(
    std::is_same<typename std::iterator_traits<OutputIterator>::value_type, value_type>::value);
  comm.device_multicast_sendrecv(iter_to_raw_ptr(input_first),
                                 tx_counts,
                                 tx_offsets,
                                 tx_dst_ranks,
                                 iter_to_raw_ptr(output_first),
                                 rx_counts,
                                 rx_offsets,
                                 rx_src_ranks,
                                 stream_view.value());
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_multicast_sendrecv_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           std::vector<size_t> const& tx_counts,
           std::vector<size_t> const& tx_offsets,
           std::vector<int> const& tx_dst_ranks,
           OutputIterator output_first,
           std::vector<size_t> const& rx_counts,
           std::vector<size_t> const& rx_offsets,
           std::vector<int> const& rx_src_ranks,
           rmm::cuda_stream_view stream_view) const
  {
    using output_value_t = typename thrust::
      tuple_element<I, typename std::iterator_traits<OutputIterator>::value_type>::type;
    auto tuple_element_input_first  = thrust::get<I>(input_first.get_iterator_tuple());
    auto tuple_element_output_first = thrust::get<I>(output_first.get_iterator_tuple());
    device_multicast_sendrecv_impl<decltype(tuple_element_input_first),
                                   decltype(tuple_element_output_first)>(comm,
                                                                         tuple_element_input_first,
                                                                         tx_counts,
                                                                         tx_offsets,
                                                                         tx_dst_ranks,
                                                                         tuple_element_output_first,
                                                                         rx_counts,
                                                                         rx_offsets,
                                                                         rx_src_ranks,
                                                                         stream_view);
    device_multicast_sendrecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>()
      .run(comm,
           input_first,
           tx_counts,
           tx_offsets,
           tx_dst_ranks,
           output_first,
           rx_counts,
           rx_offsets,
           rx_src_ranks,
           stream_view);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_multicast_sendrecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           std::vector<size_t> const& tx_counts,
           std::vector<size_t> const& tx_offsets,
           std::vector<int> const& tx_dst_ranks,
           OutputIterator output_first,
           std::vector<size_t> const& rx_counts,
           std::vector<size_t> const& rx_offsets,
           std::vector<int> const& rx_src_ranks,
           rmm::cuda_stream_view stream_view) const
  {
  }
};

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_bcast_impl(raft::comms::comms_t const& comm,
                  InputIterator input_first,
                  OutputIterator output_first,
                  size_t count,
                  int root,
                  rmm::cuda_stream_view stream_view)
{
  // no-op
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_bcast_impl(raft::comms::comms_t const& comm,
                  InputIterator input_first,
                  OutputIterator output_first,
                  size_t count,
                  int root,
                  rmm::cuda_stream_view stream_view)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.bcast(
    iter_to_raw_ptr(input_first), iter_to_raw_ptr(output_first), count, root, stream_view.value());
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_bcast_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           int root,
           rmm::cuda_stream_view stream_view) const
  {
    device_bcast_impl(comm,
                      thrust::get<I>(input_first.get_iterator_tuple()),
                      thrust::get<I>(output_first.get_iterator_tuple()),
                      count,
                      root,
                      stream_view);
    device_bcast_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, output_first, count, root, stream_view);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_bcast_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           int root,
           rmm::cuda_stream_view stream_view) const
  {
  }
};

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_allreduce_impl(raft::comms::comms_t const& comm,
                      InputIterator input_first,
                      OutputIterator output_first,
                      size_t count,
                      raft::comms::op_t op,
                      rmm::cuda_stream_view stream_view)
{
  // no-op
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_allreduce_impl(raft::comms::comms_t const& comm,
                      InputIterator input_first,
                      OutputIterator output_first,
                      size_t count,
                      raft::comms::op_t op,
                      rmm::cuda_stream_view stream_view)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.allreduce(
    iter_to_raw_ptr(input_first), iter_to_raw_ptr(output_first), count, op, stream_view.value());
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_allreduce_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           raft::comms::op_t op,
           rmm::cuda_stream_view stream_view) const
  {
    device_allreduce_impl(comm,
                          thrust::get<I>(input_first.get_iterator_tuple()),
                          thrust::get<I>(output_first.get_iterator_tuple()),
                          count,
                          op,
                          stream_view);
    device_allreduce_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, output_first, count, op, stream_view);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_allreduce_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           raft::comms::op_t op,
           rmm::cuda_stream_view stream_view) const
  {
  }
};

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_reduce_impl(raft::comms::comms_t const& comm,
                   InputIterator input_first,
                   OutputIterator output_first,
                   size_t count,
                   raft::comms::op_t op,
                   int root,
                   rmm::cuda_stream_view stream_view)
{
  // no-op
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_reduce_impl(raft::comms::comms_t const& comm,
                   InputIterator input_first,
                   OutputIterator output_first,
                   size_t count,
                   raft::comms::op_t op,
                   int root,
                   rmm::cuda_stream_view stream_view)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.reduce(iter_to_raw_ptr(input_first),
              iter_to_raw_ptr(output_first),
              count,
              op,
              root,
              stream_view.value());
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_reduce_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           raft::comms::op_t op,
           int root,
           rmm::cuda_stream_view stream_view) const
  {
    device_reduce_impl(comm,
                       thrust::get<I>(input_first.get_iterator_tuple()),
                       thrust::get<I>(output_first.get_iterator_tuple()),
                       count,
                       op,
                       root,
                       stream_view);
    device_reduce_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, output_first, count, op, root, stream_view);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_reduce_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           raft::comms::op_t op,
           int root,
           rmm::cuda_stream_view stream_view) const
  {
  }
};

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_allgatherv_impl(raft::comms::comms_t const& comm,
                       InputIterator input_first,
                       OutputIterator output_first,
                       std::vector<size_t> const& recvcounts,
                       std::vector<size_t> const& displacements,
                       rmm::cuda_stream_view stream_view)
{
  // no-op
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_allgatherv_impl(raft::comms::comms_t const& comm,
                       InputIterator input_first,
                       OutputIterator output_first,
                       std::vector<size_t> const& recvcounts,
                       std::vector<size_t> const& displacements,
                       rmm::cuda_stream_view stream_view)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.allgatherv(iter_to_raw_ptr(input_first),
                  iter_to_raw_ptr(output_first),
                  recvcounts.data(),
                  displacements.data(),
                  stream_view.value());
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_allgatherv_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           std::vector<size_t> const& recvcounts,
           std::vector<size_t> const& displacements,
           rmm::cuda_stream_view stream_view) const
  {
    device_allgatherv_impl(comm,
                           thrust::get<I>(input_first.get_iterator_tuple()),
                           thrust::get<I>(output_first.get_iterator_tuple()),
                           recvcounts,
                           displacements,
                           stream_view);
    device_allgatherv_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, output_first, recvcounts, displacements, stream_view);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_allgatherv_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           std::vector<size_t> const& recvcounts,
           std::vector<size_t> const& displacements,
           rmm::cuda_stream_view stream_view) const
  {
  }
};

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<thrust::detail::is_discard_iterator<OutputIterator>::value, void>
device_gatherv_impl(raft::comms::comms_t const& comm,
                    InputIterator input_first,
                    OutputIterator output_first,
                    size_t sendcount,
                    std::vector<size_t> const& recvcounts,
                    std::vector<size_t> const& displacements,
                    int root,
                    rmm::cuda_stream_view stream_view)
{
  // no-op
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_gatherv_impl(raft::comms::comms_t const& comm,
                    InputIterator input_first,
                    OutputIterator output_first,
                    size_t sendcount,
                    std::vector<size_t> const& recvcounts,
                    std::vector<size_t> const& displacements,
                    int root,
                    rmm::cuda_stream_view stream_view)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.gatherv(iter_to_raw_ptr(input_first),
               iter_to_raw_ptr(output_first),
               sendcount,
               recvcounts.data(),
               displacements.data(),
               root,
               stream_view.value());
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_gatherv_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t sendcount,
           std::vector<size_t> const& recvcounts,
           std::vector<size_t> const& displacements,
           int root,
           rmm::cuda_stream_view stream_view) const
  {
    device_gatherv_impl(comm,
                        thrust::get<I>(input_first.get_iterator_tuple()),
                        thrust::get<I>(output_first.get_iterator_tuple()),
                        sendcount,
                        recvcounts,
                        displacements,
                        root,
                        stream_view);
    device_gatherv_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, output_first, sendcount, recvcounts, displacements, root, stream_view);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_gatherv_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t sendcount,
           std::vector<size_t> const& recvcounts,
           std::vector<size_t> const& displacements,
           int root,
           rmm::cuda_stream_view stream_view) const
  {
  }
};

}  // namespace detail

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_isend(raft::comms::comms_t const& comm,
             InputIterator input_first,
             size_t count,
             int dst,
             int base_tag /* actual tag = base tag */,
             raft::comms::request_t* requests)
{
  detail::device_isend_impl<InputIterator,
                            typename std::iterator_traits<OutputIterator>::value_type>(
    comm, input_first, count, dst, base_tag, requests);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_isend(raft::comms::comms_t const& comm,
             InputIterator input_first,
             size_t count,
             int dst,
             int base_tag /* actual tag = base_tag + tuple index */,
             raft::comms::request_t* requests)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::
    device_isend_tuple_iterator_element_impl<InputIterator, OutputIterator, size_t{0}, tuple_size>()
      .run(comm, input_first, count, dst, base_tag, requests);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_irecv(raft::comms::comms_t const& comm,
             OutputIterator output_first,
             size_t count,
             int src,
             int base_tag /* actual tag = base tag */,
             raft::comms::request_t* requests)
{
  detail::device_irecv_impl<typename std::iterator_traits<InputIterator>::value_type,
                            OutputIterator>(comm, output_first, count, src, base_tag, requests);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_irecv(raft::comms::comms_t const& comm,
             OutputIterator output_first,
             size_t count,
             int src,
             int base_tag /* actual tag = base_tag + tuple index */,
             raft::comms::request_t* requests)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::
    device_irecv_tuple_iterator_element_impl<InputIterator, OutputIterator, size_t{0}, tuple_size>()
      .run(comm, output_first, count, src, base_tag, requests);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_sendrecv(raft::comms::comms_t const& comm,
                InputIterator input_first,
                size_t tx_count,
                int dst,
                OutputIterator output_first,
                size_t rx_count,
                int src,
                rmm::cuda_stream_view stream_view)
{
  detail::device_sendrecv_impl<InputIterator, OutputIterator>(
    comm, input_first, tx_count, dst, output_first, rx_count, src, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_sendrecv(raft::comms::comms_t const& comm,
                InputIterator input_first,
                size_t tx_count,
                int dst,
                OutputIterator output_first,
                size_t rx_count,
                int src,
                rmm::cuda_stream_view stream_view)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  // FIXME: NCCL 2.7 supports only one ncclSend and one ncclRecv for a source rank and destination
  // rank inside ncclGroupStart/ncclGroupEnd, so we cannot place this inside
  // ncclGroupStart/ncclGroupEnd, this restriction will be lifted in NCCL 2.8
  detail::device_sendrecv_tuple_iterator_element_impl<InputIterator,
                                                      OutputIterator,
                                                      size_t{0},
                                                      tuple_size>()
    .run(comm, input_first, tx_count, dst, output_first, rx_count, src, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_multicast_sendrecv(raft::comms::comms_t const& comm,
                          InputIterator input_first,
                          std::vector<size_t> const& tx_counts,
                          std::vector<size_t> const& tx_offsets,
                          std::vector<int> const& tx_dst_ranks,
                          OutputIterator output_first,
                          std::vector<size_t> const& rx_counts,
                          std::vector<size_t> const& rx_offsets,
                          std::vector<int> const& rx_src_ranks,
                          rmm::cuda_stream_view stream_view)
{
  detail::device_multicast_sendrecv_impl<InputIterator, OutputIterator>(comm,
                                                                        input_first,
                                                                        tx_counts,
                                                                        tx_offsets,
                                                                        tx_dst_ranks,
                                                                        output_first,
                                                                        rx_counts,
                                                                        rx_offsets,
                                                                        rx_src_ranks,
                                                                        stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_multicast_sendrecv(raft::comms::comms_t const& comm,
                          InputIterator input_first,
                          std::vector<size_t> const& tx_counts,
                          std::vector<size_t> const& tx_offsets,
                          std::vector<int> const& tx_dst_ranks,
                          OutputIterator output_first,
                          std::vector<size_t> const& rx_counts,
                          std::vector<size_t> const& rx_offsets,
                          std::vector<int> const& rx_src_ranks,
                          rmm::cuda_stream_view stream_view)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  // FIXME: NCCL 2.7 supports only one ncclSend and one ncclRecv for a source rank and destination
  // rank inside ncclGroupStart/ncclGroupEnd, so we cannot place this inside
  // ncclGroupStart/ncclGroupEnd, this restriction will be lifted in NCCL 2.8
  detail::device_multicast_sendrecv_tuple_iterator_element_impl<InputIterator,
                                                                OutputIterator,
                                                                size_t{0},
                                                                tuple_size>()
    .run(comm,
         input_first,
         tx_counts,
         tx_offsets,
         tx_dst_ranks,
         output_first,
         rx_counts,
         rx_offsets,
         rx_src_ranks,
         stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_bcast(raft::comms::comms_t const& comm,
             InputIterator input_first,
             OutputIterator output_first,
             size_t count,
             int root,
             rmm::cuda_stream_view stream_view)
{
  detail::device_bcast_impl(comm, input_first, output_first, count, root, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_bcast(raft::comms::comms_t const& comm,
             InputIterator input_first,
             OutputIterator output_first,
             size_t count,
             int root,
             rmm::cuda_stream_view stream_view)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::
    device_bcast_tuple_iterator_element_impl<InputIterator, OutputIterator, size_t{0}, tuple_size>()
      .run(comm, input_first, output_first, count, root, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_allreduce(raft::comms::comms_t const& comm,
                 InputIterator input_first,
                 OutputIterator output_first,
                 size_t count,
                 raft::comms::op_t op,
                 rmm::cuda_stream_view stream_view)
{
  detail::device_allreduce_impl(comm, input_first, output_first, count, op, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_allreduce(raft::comms::comms_t const& comm,
                 InputIterator input_first,
                 OutputIterator output_first,
                 size_t count,
                 raft::comms::op_t op,
                 rmm::cuda_stream_view stream_view)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::device_allreduce_tuple_iterator_element_impl<InputIterator,
                                                       OutputIterator,
                                                       size_t{0},
                                                       tuple_size>()
    .run(comm, input_first, output_first, count, op, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_reduce(raft::comms::comms_t const& comm,
              InputIterator input_first,
              OutputIterator output_first,
              size_t count,
              raft::comms::op_t op,
              int root,
              rmm::cuda_stream_view stream_view)
{
  detail::device_reduce_impl(comm, input_first, output_first, count, op, root, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_reduce(raft::comms::comms_t const& comm,
              InputIterator input_first,
              OutputIterator output_first,
              size_t count,
              raft::comms::op_t op,
              int root,
              rmm::cuda_stream_view stream_view)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::device_reduce_tuple_iterator_element_impl<InputIterator,
                                                    OutputIterator,
                                                    size_t{0},
                                                    tuple_size>()
    .run(comm, input_first, output_first, count, op, root, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_allgatherv(raft::comms::comms_t const& comm,
                  InputIterator input_first,
                  OutputIterator output_first,
                  std::vector<size_t> const& recvcounts,
                  std::vector<size_t> const& displacements,
                  rmm::cuda_stream_view stream_view)
{
  detail::device_allgatherv_impl(
    comm, input_first, output_first, recvcounts, displacements, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_allgatherv(raft::comms::comms_t const& comm,
                  InputIterator input_first,
                  OutputIterator output_first,
                  std::vector<size_t> const& recvcounts,
                  std::vector<size_t> const& displacements,
                  rmm::cuda_stream_view stream_view)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::device_allgatherv_tuple_iterator_element_impl<InputIterator,
                                                        OutputIterator,
                                                        size_t{0},
                                                        tuple_size>()
    .run(comm, input_first, output_first, recvcounts, displacements, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  std::is_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value,
  void>
device_gatherv(raft::comms::comms_t const& comm,
               InputIterator input_first,
               OutputIterator output_first,
               size_t sendcount,
               std::vector<size_t> const& recvcounts,
               std::vector<size_t> const& displacements,
               int root,
               rmm::cuda_stream_view stream_view)
{
  detail::device_gatherv_impl(
    comm, input_first, output_first, sendcount, recvcounts, displacements, root, stream_view);
}

template <typename InputIterator, typename OutputIterator>
std::enable_if_t<
  is_thrust_tuple_of_arithmetic<typename std::iterator_traits<InputIterator>::value_type>::value &&
    is_thrust_tuple<typename std::iterator_traits<OutputIterator>::value_type>::value,
  void>
device_gatherv(raft::comms::comms_t const& comm,
               InputIterator input_first,
               OutputIterator output_first,
               size_t sendcount,
               std::vector<size_t> const& recvcounts,
               std::vector<size_t> const& displacements,
               int root,
               rmm::cuda_stream_view stream_view)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::device_gatherv_tuple_iterator_element_impl<InputIterator,
                                                     OutputIterator,
                                                     size_t{0},
                                                     tuple_size>()
    .run(comm, input_first, output_first, sendcount, recvcounts, displacements, root, stream_view);
}

inline void device_group_start(raft::comms::comms_t const& comm) { comm.group_start(); }

inline void device_group_end(raft::comms::comms_t const& comm) { comm.group_end(); }

}  // namespace cugraph
