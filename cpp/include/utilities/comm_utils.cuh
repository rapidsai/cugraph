/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <utilities/thrust_tuple_utils.cuh>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/detail/normal_iterator.h>

#include <numeric>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

template <typename TupleType, size_t I, size_t N>
struct update_vector_of_tuple_scalar_elements_from_tuple_impl {
  void update(std::vector<int64_t>& tuple_scalar_elements, TupleType const& tuple) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_scalar_elements.data() + I);
    *ptr     = thrust::get<I>(tuple);
    update_vector_of_tuple_scalar_elements_from_tuple_impl<TupleType, I + 1, N>().update(
      tuple_scalar_elements, tuple);
  }
};

template <typename TupleType, size_t I>
struct update_vector_of_tuple_scalar_elements_from_tuple_impl<TupleType, I, I> {
  void update(std::vector<int64_t>& tuple_scalar_elements, TupleType const& tuple) const { return; }
};

template <typename TupleType, size_t I, size_t N>
struct update_tuple_from_vector_of_tuple_scalar_elements_impl {
  void update(TupleType& tuple, std::vector<int64_t> const& tuple_scalar_elements) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr              = reinterpret_cast<element_t const*>(tuple_scalar_elements.data() + I);
    thrust::get<I>(tuple) = *ptr;
    update_tuple_from_vector_of_tuple_scalar_elements_impl<TupleType, I + 1, N>().update(
      tuple, tuple_scalar_elements);
  }
};

template <typename TupleType, size_t I>
struct update_tuple_from_vector_of_tuple_scalar_elements_impl<TupleType, I, I> {
  void update(TupleType& tuple, std::vector<int64_t> const& tuple_scalar_elements) const { return; }
};

template <typename TupleType, size_t I, size_t N>
struct host_allreduce_tuple_scalar_element_impl {
  void run(raft::comms::comms_t const& comm,
           rmm::device_uvector<int64_t>& tuple_scalar_elements,
           cudaStream_t stream) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_scalar_elements.data() + I);
    comm.allreduce(ptr, ptr, 1, raft::comms::op_t::SUM, stream);
    host_allreduce_tuple_scalar_element_impl<TupleType, I + 1, N>().run(
      comm, tuple_scalar_elements, stream);
  }
};

template <typename TupleType, size_t I>
struct host_allreduce_tuple_scalar_element_impl<TupleType, I, I> {
  void run(raft::comms::comms_t const& comm,
           rmm::device_uvector<int64_t>& tuple_scalar_elements,
           cudaStream_t stream) const
  {
  }
};

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
                     cudaStream_t stream)
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
                     cudaStream_t stream)
{
  using value_type = typename std::iterator_traits<InputIterator>::value_type;
  static_assert(
    std::is_same<typename std::iterator_traits<OutputIterator>::value_type, value_type>::value);
  // ncclSend/ncclRecv pair needs to be located inside ncclGroupStart/ncclGroupEnd to avoid deadlock
  ncclGroupStart();
  ncclSend(iter_to_raw_ptr(input_first),
           tx_count * sizeof(value_type),
           ncclUint8,
           dst,
           comm.get_nccl_comm(),
           stream);
  ncclRecv(iter_to_raw_ptr(output_first),
           rx_count * sizeof(value_type),
           ncclUint8,
           src,
           comm.get_nccl_comm(),
           stream);
  ncclGroupEnd();
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
           cudaStream_t stream) const
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
      stream);
    device_sendrecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, tx_count, dst, output_first, rx_count, src, stream);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_sendrecv_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           size_t count,
           int dst,
           int base_tag,
           raft::comms::request_t* requests) const
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
                               cudaStream_t stream)
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
                               cudaStream_t stream)
{
  using value_type = typename std::iterator_traits<InputIterator>::value_type;
  static_assert(
    std::is_same<typename std::iterator_traits<OutputIterator>::value_type, value_type>::value);
  // ncclSend/ncclRecv pair needs to be located inside ncclGroupStart/ncclGroupEnd to avoid deadlock
  ncclGroupStart();
  for (size_t i = 0; i < tx_counts.size(); ++i) {
    ncclSend(iter_to_raw_ptr(input_first + tx_offsets[i]),
             tx_counts[i] * sizeof(value_type),
             ncclUint8,
             tx_dst_ranks[i],
             comm.get_nccl_comm(),
             stream);
  }
  for (size_t i = 0; i < rx_counts.size(); ++i) {
    ncclRecv(iter_to_raw_ptr(output_first + rx_offsets[i]),
             rx_counts[i] * sizeof(value_type),
             ncclUint8,
             rx_src_ranks[i],
             comm.get_nccl_comm(),
             stream);
  }
  ncclGroupEnd();
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
           cudaStream_t stream) const
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
                                                                         stream);
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
           stream);
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
           cudaStream_t stream) const
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
                  cudaStream_t stream)
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
                  cudaStream_t stream)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  if (comm.get_rank() == root) {
    comm.bcast(iter_to_raw_ptr(input_first), count, root, stream);
  } else {
    comm.bcast(iter_to_raw_ptr(output_first), count, root, stream);
  }
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_bcast_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           int root,
           cudaStream_t stream) const
  {
    device_bcast_impl(comm,
                      thrust::get<I>(input_first.get_iterator_tuple()),
                      thrust::get<I>(output_first.get_iterator_tuple()),
                      count,
                      root,
                      stream);
    device_bcast_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>(
      comm, input_first, output_first, count, root, stream);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_bcast_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           int root,
           cudaStream_t stream) const
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
                   cudaStream_t stream)
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
                   cudaStream_t stream)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.reduce(iter_to_raw_ptr(input_first), iter_to_raw_ptr(output_first), count, op, root, stream);
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_reduce_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           size_t count,
           raft::comms::op_t op,
           int root,
           cudaStream_t stream) const
  {
    device_reduce_impl(comm,
                       thrust::get<I>(input_first.get_iterator_tuple()),
                       thrust::get<I>(output_first.get_iterator_tuple()),
                       count,
                       op,
                       root,
                       stream);
    device_reduce_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>(
      comm, input_first, output_first, count, op, root, stream);
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
           cudaStream_t stream) const
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
                       cudaStream_t stream)
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
                       cudaStream_t stream)
{
  static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type,
                             typename std::iterator_traits<OutputIterator>::value_type>::value);
  comm.allgatherv(iter_to_raw_ptr(input_first),
                  iter_to_raw_ptr(output_first),
                  recvcounts.data(),
                  displacements.data(),
                  stream);
}

template <typename InputIterator, typename OutputIterator, size_t I, size_t N>
struct device_allgatherv_tuple_iterator_element_impl {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           std::vector<size_t> const& recvcounts,
           std::vector<size_t> const& displacements,
           cudaStream_t stream) const
  {
    device_allgatherv_impl(comm,
                           thrust::get<I>(input_first.get_iterator_tuple()),
                           thrust::get<I>(output_first.get_iterator_tuple()),
                           recvcounts,
                           displacements,
                           stream);
    device_allgatherv_tuple_iterator_element_impl<InputIterator, OutputIterator, I + 1, N>().run(
      comm, input_first, output_first, recvcounts, displacements, stream);
  }
};

template <typename InputIterator, typename OutputIterator, size_t I>
struct device_allgatherv_tuple_iterator_element_impl<InputIterator, OutputIterator, I, I> {
  void run(raft::comms::comms_t const& comm,
           InputIterator input_first,
           OutputIterator output_first,
           std::vector<size_t> const& recvcounts,
           std::vector<size_t> const& displacements,
           cudaStream_t stream) const
  {
  }
};

template <typename TupleType, size_t I>
auto allocate_comm_buffer_tuple_element_impl(size_t buffer_size, cudaStream_t stream)
{
  using element_t = typename thrust::tuple_element<I, TupleType>::type;
  return rmm::device_uvector<element_t>(buffer_size, stream);
}

template <typename TupleType, size_t... Is>
auto allocate_comm_buffer_tuple_impl(std::index_sequence<Is...>,
                                     size_t buffer_size,
                                     cudaStream_t stream)
{
  return thrust::make_tuple(
    allocate_comm_buffer_tuple_element_impl<TupleType, Is>(buffer_size, stream)...);
}

template <typename TupleType, size_t I, typename BufferType>
auto get_comm_buffer_begin_tuple_element_impl(BufferType& buffer)
{
  using element_t = typename thrust::tuple_element<I, TupleType>::type;
  return thrust::get<I>(buffer).begin();
}

template <typename TupleType, size_t... Is, typename BufferType>
auto get_comm_buffer_begin_tuple_impl(std::index_sequence<Is...>, BufferType& buffer)
{
  return thrust::make_tuple(get_comm_buffer_begin_tuple_element_impl<TupleType, Is>(buffer)...);
}

}  // namespace detail

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_allreduce(
  raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  rmm::device_uvector<T> d_input(1, stream);
  raft::update_device(d_input.data(), &input, 1, stream);
  comm.allreduce(d_input.data(), d_input.data(), 1, raft::comms::op_t::SUM, stream);
  T h_input{};
  raft::update_host(&h_input, d_input.data(), 1, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  return h_input;
}

template <typename T>
std::enable_if_t<cugraph::experimental::is_thrust_tuple_of_arithmetic<T>::value, T>
host_scalar_allreduce(raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  std::vector<int64_t> h_tuple_scalar_elements(tuple_size);
  rmm::device_uvector<int64_t> d_tuple_scalar_elements(tuple_size, stream);
  T ret{};

  detail::update_vector_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_tuple_scalar_elements, input);
  raft::update_device(
    d_tuple_scalar_elements.data(), h_tuple_scalar_elements.data(), tuple_size, stream);
  detail::host_allreduce_tuple_scalar_element_impl<T, size_t{0}, tuple_size>().run(
    comm, d_tuple_scalar_elements, stream);
  raft::update_host(
    h_tuple_scalar_elements.data(), d_tuple_scalar_elements.data(), tuple_size, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  detail::update_tuple_from_vector_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>().update(
    ret, h_tuple_scalar_elements);

  return ret;
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_bcast(
  raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  rmm::device_uvector<T> d_input(1, stream);
  if (comm.get_rank() == root) { raft::update_device(d_input.data(), &input, 1, stream); }
  comm.bcast(d_input.data(), 1, root, stream);
  auto h_input = input;
  if (comm.get_rank() != root) { raft::update_host(&h_input, d_input.data(), 1, stream); }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  return h_input;
}

template <typename T>
std::enable_if_t<cugraph::experimental::is_thrust_tuple_of_arithmetic<T>::value, T>
host_scalar_bcast(raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  std::vector<int64_t> h_tuple_scalar_elements(tuple_size);
  rmm::device_uvector<int64_t> d_tuple_scalar_elements(tuple_size, stream);
  auto ret = input;

  if (comm.get_rank() == root) {
    detail::update_vector_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>()
      .update(h_tuple_scalar_elements, input);
    raft::update_device(
      d_tuple_scalar_elements.data(), h_tuple_scalar_elements.data(), tuple_size, stream);
  }
  comm.bcast(d_tuple_scalar_elements.data(), d_tuple_scalar_elements.size(), root, stream);
  if (comm.get_rank() != root) {
    raft::update_host(
      h_tuple_scalar_elements.data(), d_tuple_scalar_elements.data(), tuple_size, stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  if (comm.get_rank() != root) {
    detail::update_tuple_from_vector_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>()
      .update(ret, h_tuple_scalar_elements);
  }

  return ret;
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, std::vector<T>> host_scalar_allgather(
  raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  std::vector<size_t> rx_counts(comm.get_size(), size_t{1});
  std::vector<size_t> displacements(rx_counts.size(), size_t{0});
  std::iota(displacements.begin(), displacements.end(), size_t{0});
  rmm::device_uvector<T> d_outputs(rx_counts.size(), stream);
  raft::update_device(d_outputs.data() + comm.get_rank(), &input, 1, stream);
  comm.allgatherv(d_outputs.data() + comm.get_rank(),
                  d_outputs.data(),
                  rx_counts.data(),
                  displacements.data(),
                  stream);
  std::vector<T> h_outputs(rx_counts.size(), size_t{0});
  raft::update_host(h_outputs.data(), d_outputs.data(), rx_counts.size(), stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  return h_outputs;
}

template <typename T>
std::enable_if_t<cugraph::experimental::is_thrust_tuple_of_arithmetic<T>::value, std::vector<T>>
host_scalar_allgather(raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  std::vector<size_t> rx_counts(comm.get_size(), tuple_size);
  std::vector<size_t> displacements(rx_counts.size(), size_t{0});
  for (size_t i = 0; i < displacements.size(); ++i) { displacements[i] = i * tuple_size; }
  std::vector<int64_t> h_tuple_scalar_elements(tuple_size);
  rmm::device_uvector<int64_t> d_allgathered_tuple_scalar_elements(comm.get_size() * tuple_size,
                                                                   stream);

  detail::update_vector_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_tuple_scalar_elements, input);
  raft::update_device(d_allgathered_tuple_scalar_elements.data() + comm.get_rank() * tuple_size,
                      h_tuple_scalar_elements.data(),
                      tuple_size,
                      stream);
  comm.allgatherv(d_allgathered_tuple_scalar_elements.data() + comm.get_rank() * tuple_size,
                  d_allgathered_tuple_scalar_elements.data(),
                  rx_counts.data(),
                  displacements.data(),
                  stream);
  std::vector<int64_t> h_allgathered_tuple_scalar_elements(comm.get_size() * tuple_size);
  raft::update_host(h_allgathered_tuple_scalar_elements.data(),
                    d_allgathered_tuple_scalar_elements.data(),
                    comm.get_size() * tuple_size,
                    stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  std::vector<T> ret(comm.get_size());
  for (size_t i = 0; i < ret.size(); ++i) {
    std::vector<int64_t> h_tuple_scalar_elements(
      h_allgathered_tuple_scalar_elements.data() + i * tuple_size,
      h_allgathered_tuple_scalar_elements.data() + (i + 1) * tuple_size);
    detail::update_tuple_from_vector_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>()
      .update(ret[i], h_tuple_scalar_elements);
  }

  return ret;
}

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
                cudaStream_t stream)
{
  detail::device_sendrecv_impl<InputIterator, OutputIterator>(
    comm, input_first, tx_count, dst, output_first, rx_count, src, stream);
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
                cudaStream_t stream)
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
    .run(comm, input_first, tx_count, dst, output_first, rx_count, src, stream);
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
                          cudaStream_t stream)
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
                                                                        stream);
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
                          cudaStream_t stream)
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
         stream);
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
             cudaStream_t stream)
{
  detail::device_bcast_impl(comm, input_first, output_first, count, root, stream);
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
             cudaStream_t stream)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::
    device_bcast_tuple_iterator_element_impl<InputIterator, OutputIterator, size_t{0}, tuple_size>(
      comm, input_first, output_first, count, root, stream);
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
              cudaStream_t stream)
{
  detail::device_reduce_impl(comm, input_first, output_first, count, op, root, stream);
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
              cudaStream_t stream)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value ==
    thrust::tuple_size<typename thrust::iterator_traits<OutputIterator>::value_type>::value);

  size_t constexpr tuple_size =
    thrust::tuple_size<typename thrust::iterator_traits<InputIterator>::value_type>::value;

  detail::
    device_reduce_tuple_iterator_element_impl<InputIterator, OutputIterator, size_t{0}, tuple_size>(
      comm, input_first, output_first, count, op, root, stream);
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
                  cudaStream_t stream)
{
  detail::device_allgatherv_impl(
    comm, input_first, output_first, recvcounts, displacements, stream);
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
                  cudaStream_t stream)
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
    .run(comm, input_first, output_first, recvcounts, displacements, stream);
}

template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto allocate_comm_buffer(size_t buffer_size, cudaStream_t stream)
{
  return rmm::device_uvector<T>(buffer_size, stream);
}

template <typename T, typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto allocate_comm_buffer(size_t buffer_size, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return detail::allocate_comm_buffer_tuple_impl<T>(
    std::make_index_sequence<tuple_size>(), buffer_size, stream);
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto get_comm_buffer_begin(BufferType& buffer)
{
  return buffer.begin();
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto get_comm_buffer_begin(BufferType& buffer)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return thrust::make_zip_iterator(
    detail::get_comm_buffer_begin_tuple_impl<T>(std::make_index_sequence<tuple_size>(), buffer));
}

}  // namespace experimental
}  // namespace cugraph
