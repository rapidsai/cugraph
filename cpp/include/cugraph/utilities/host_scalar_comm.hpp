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

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

#include <numeric>
#include <type_traits>

namespace cugraph {

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
           raft::comms::op_t op,
           cudaStream_t stream) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_scalar_elements.data() + I);
    comm.allreduce(ptr, ptr, 1, op, stream);
    host_allreduce_tuple_scalar_element_impl<TupleType, I + 1, N>().run(
      comm, tuple_scalar_elements, op, stream);
  }
};

template <typename TupleType, size_t I>
struct host_allreduce_tuple_scalar_element_impl<TupleType, I, I> {
  void run(raft::comms::comms_t const& comm,
           rmm::device_uvector<int64_t>& tuple_scalar_elements,
           raft::comms::op_t op,
           cudaStream_t stream) const
  {
  }
};

template <typename TupleType, size_t I, size_t N>
struct host_reduce_tuple_scalar_element_impl {
  void run(raft::comms::comms_t const& comm,
           rmm::device_uvector<int64_t>& tuple_scalar_elements,
           raft::comms::op_t op,
           int root,
           cudaStream_t stream) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_scalar_elements.data() + I);
    comm.reduce(ptr, ptr, 1, op, root, stream);
    host_reduce_tuple_scalar_element_impl<TupleType, I + 1, N>().run(
      comm, tuple_scalar_elements, op, root, stream);
  }
};

template <typename TupleType, size_t I>
struct host_reduce_tuple_scalar_element_impl<TupleType, I, I> {
  void run(raft::comms::comms_t const& comm,
           rmm::device_uvector<int64_t>& tuple_scalar_elements,
           raft::comms::op_t op,
           int root,
           cudaStream_t stream) const
  {
  }
};

}  // namespace detail

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_allreduce(
  raft::comms::comms_t const& comm, T input, raft::comms::op_t op, cudaStream_t stream)
{
  rmm::device_uvector<T> d_input(1, stream);
  raft::update_device(d_input.data(), &input, 1, stream);
  comm.allreduce(d_input.data(), d_input.data(), 1, op, stream);
  T h_input{};
  raft::update_host(&h_input, d_input.data(), 1, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  return h_input;
}

template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, T> host_scalar_allreduce(
  raft::comms::comms_t const& comm, T input, raft::comms::op_t op, cudaStream_t stream)
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
    comm, d_tuple_scalar_elements, op, stream);
  raft::update_host(
    h_tuple_scalar_elements.data(), d_tuple_scalar_elements.data(), tuple_size, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  detail::update_tuple_from_vector_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>().update(
    ret, h_tuple_scalar_elements);

  return ret;
}

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_reduce(
  raft::comms::comms_t const& comm, T input, raft::comms::op_t op, int root, cudaStream_t stream)
{
  rmm::device_uvector<T> d_input(1, stream);
  raft::update_device(d_input.data(), &input, 1, stream);
  comm.reduce(d_input.data(), d_input.data(), 1, op, stream);
  T h_input{};
  if (comm.get_rank() == root) { raft::update_host(&h_input, d_input.data(), 1, stream); }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  return h_input;
}

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, T> host_scalar_reduce(
  raft::comms::comms_t const& comm, T input, raft::comms::op_t op, int root, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  std::vector<int64_t> h_tuple_scalar_elements(tuple_size);
  rmm::device_uvector<int64_t> d_tuple_scalar_elements(tuple_size, stream);
  T ret{};

  detail::update_vector_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_tuple_scalar_elements, input);
  raft::update_device(
    d_tuple_scalar_elements.data(), h_tuple_scalar_elements.data(), tuple_size, stream);
  detail::host_reduce_tuple_scalar_element_impl<T, size_t{0}, tuple_size>().run(
    comm, d_tuple_scalar_elements, op, root, stream);
  if (comm.get_rank() == root) {
    raft::update_host(
      h_tuple_scalar_elements.data(), d_tuple_scalar_elements.data(), tuple_size, stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  if (comm.get_rank() == root) {
    detail::update_tuple_from_vector_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>()
      .update(ret, h_tuple_scalar_elements);
  }

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
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, T> host_scalar_bcast(
  raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
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
  // FIXME: better use allgather
  comm.allgatherv(d_outputs.data() + comm.get_rank(),
                  d_outputs.data(),
                  rx_counts.data(),
                  displacements.data(),
                  stream);
  std::vector<T> h_outputs(rx_counts.size());
  raft::update_host(h_outputs.data(), d_outputs.data(), rx_counts.size(), stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  return h_outputs;
}

template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, std::vector<T>>
host_scalar_allgather(raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  std::vector<size_t> rx_counts(comm.get_size(), tuple_size);
  std::vector<size_t> displacements(rx_counts.size(), size_t{0});
  for (size_t i = 0; i < displacements.size(); ++i) {
    displacements[i] = i * tuple_size;
  }
  std::vector<int64_t> h_tuple_scalar_elements(tuple_size);
  rmm::device_uvector<int64_t> d_allgathered_tuple_scalar_elements(comm.get_size() * tuple_size,
                                                                   stream);

  detail::update_vector_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_tuple_scalar_elements, input);
  raft::update_device(d_allgathered_tuple_scalar_elements.data() + comm.get_rank() * tuple_size,
                      h_tuple_scalar_elements.data(),
                      tuple_size,
                      stream);
  // FIXME: better use allgather
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

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, std::vector<T>> host_scalar_gather(
  raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  rmm::device_uvector<T> d_outputs(comm.get_rank() == root ? comm.get_size() : int{1}, stream);
  raft::update_device(
    comm.get_rank() == root ? d_outputs.data() + comm.get_rank() : d_outputs.data(),
    &input,
    1,
    stream);
  comm.gather(comm.get_rank() == root ? d_outputs.data() + comm.get_rank() : d_outputs.data(),
              d_outputs.data(),
              size_t{1},
              root,
              stream);
  std::vector<T> h_outputs(comm.get_rank() == root ? comm.get_size() : 0);
  if (comm.get_rank() == root) {
    raft::update_host(h_outputs.data(), d_outputs.data(), comm.get_size(), stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  return h_outputs;
}

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, std::vector<T>>
host_scalar_gather(raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  std::vector<int64_t> h_tuple_scalar_elements(tuple_size);
  rmm::device_uvector<int64_t> d_gathered_tuple_scalar_elements(
    comm.get_rank() == root ? comm.get_size() * tuple_size : tuple_size, stream);

  detail::update_vector_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_tuple_scalar_elements, input);
  raft::update_device(comm.get_rank() == root
                        ? d_gathered_tuple_scalar_elements.data() + comm.get_rank() * tuple_size
                        : d_gathered_tuple_scalar_elements.data(),
                      h_tuple_scalar_elements.data(),
                      tuple_size,
                      stream);
  comm.gather(comm.get_rank() == root
                ? d_gathered_tuple_scalar_elements.data() + comm.get_rank() * tuple_size
                : d_gathered_tuple_scalar_elements.data(),
              d_gathered_tuple_scalar_elements.data(),
              tuple_size,
              root,
              stream);
  std::vector<int64_t> h_gathered_tuple_scalar_elements(
    comm.get_rank() == root ? comm.get_size() * tuple_size : size_t{0});
  if (comm.get_rank() == root) {
    raft::update_host(h_gathered_tuple_scalar_elements.data(),
                      d_gathered_tuple_scalar_elements.data(),
                      comm.get_size() * tuple_size,
                      stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  std::vector<T> ret(comm.get_size());
  if (comm.get_rank() == root) {
    for (size_t i = 0; i < ret.size(); ++i) {
      std::vector<int64_t> h_tuple_scalar_elements(
        h_gathered_tuple_scalar_elements.data() + i * tuple_size,
        h_gathered_tuple_scalar_elements.data() + (i + 1) * tuple_size);
      detail::update_tuple_from_vector_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>()
        .update(ret[i], h_tuple_scalar_elements);
    }
  }

  return ret;
}

}  // namespace cugraph
