/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/host_staging_buffer_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/tuple>

#include <numeric>
#include <type_traits>
#include <variant>

namespace cugraph {

namespace detail {

template <typename TupleType, size_t I, size_t N>
struct update_array_of_tuple_scalar_elements_from_tuple_impl {
  void update(int64_t* tuple_scalar_elements, TupleType const& tuple) const
  {
    using element_t = typename cuda::std::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_scalar_elements + I);
    *ptr     = cuda::std::get<I>(tuple);
    update_array_of_tuple_scalar_elements_from_tuple_impl<TupleType, I + 1, N>().update(
      tuple_scalar_elements, tuple);
  }
};

template <typename TupleType, size_t I>
struct update_array_of_tuple_scalar_elements_from_tuple_impl<TupleType, I, I> {
  void update(int64_t* tuple_scalar_elements, TupleType const& tuple) const { return; }
};

template <typename TupleType, size_t I, size_t N>
struct update_tuple_from_array_of_tuple_scalar_elements_impl {
  void update(TupleType& tuple, int64_t const* tuple_scalar_elements) const
  {
    using element_t = typename cuda::std::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr                 = reinterpret_cast<element_t const*>(tuple_scalar_elements + I);
    cuda::std::get<I>(tuple) = *ptr;
    update_tuple_from_array_of_tuple_scalar_elements_impl<TupleType, I + 1, N>().update(
      tuple, tuple_scalar_elements);
  }
};

template <typename TupleType, size_t I>
struct update_tuple_from_array_of_tuple_scalar_elements_impl<TupleType, I, I> {
  void update(TupleType& tuple, int64_t const* tuple_scalar_elements) const { return; }
};

template <typename TupleType, size_t I, size_t N>
struct host_allreduce_tuple_scalar_element_impl {
  void run(raft::comms::comms_t const& comm,
           int64_t* tuple_scalar_elements,
           raft::comms::op_t op,
           cudaStream_t stream) const
  {
    using element_t = typename cuda::std::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_scalar_elements + I);
    comm.allreduce(ptr, ptr, 1, op, stream);
    host_allreduce_tuple_scalar_element_impl<TupleType, I + 1, N>().run(
      comm, tuple_scalar_elements, op, stream);
  }
};

template <typename TupleType, size_t I>
struct host_allreduce_tuple_scalar_element_impl<TupleType, I, I> {
  void run(raft::comms::comms_t const& comm,
           int64_t* tuple_scalar_elements,
           raft::comms::op_t op,
           cudaStream_t stream) const
  {
  }
};

template <typename TupleType, size_t I, size_t N>
struct host_reduce_tuple_scalar_element_impl {
  void run(raft::comms::comms_t const& comm,
           int64_t* tuple_scalar_elements,
           raft::comms::op_t op,
           int root,
           cudaStream_t stream) const
  {
    using element_t = typename cuda::std::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_scalar_elements + I);
    comm.reduce(ptr, ptr, 1, op, root, stream);
    host_reduce_tuple_scalar_element_impl<TupleType, I + 1, N>().run(
      comm, tuple_scalar_elements, op, root, stream);
  }
};

template <typename TupleType, size_t I>
struct host_reduce_tuple_scalar_element_impl<TupleType, I, I> {
  void run(raft::comms::comms_t const& comm,
           int64_t* tuple_scalar_elements,
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
  std::variant<std::vector<T>, rmm::device_uvector<T>> h_tmp_buffer{};
  raft::host_span<T> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<T>(1, stream);
  } else {
    h_tmp_buffer = std::vector<T>(1);
  }
  h_tmp_buffer_view = std::visit(
    [](auto& buffer) { return raft::host_span<T>(buffer.data(), buffer.size()); }, h_tmp_buffer);
  rmm::device_uvector<T> d_tmp_buffer(1, stream);
  T* h_staging_buffer = h_tmp_buffer_view.data();
  T* d_staging_buffer = d_tmp_buffer.data();
  h_staging_buffer[0] = input;
  raft::update_device(d_staging_buffer, h_staging_buffer, 1, stream);
  comm.allreduce(d_staging_buffer, d_staging_buffer, 1, op, stream);
  raft::update_host(h_staging_buffer, d_staging_buffer, 1, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  auto h_input = h_staging_buffer[0];
  return h_input;
}

template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, T> host_scalar_allreduce(
  raft::comms::comms_t const& comm, T input, raft::comms::op_t op, cudaStream_t stream)
{
  size_t constexpr tuple_size = cuda::std::tuple_size<T>::value;

  std::variant<std::vector<int64_t>, rmm::device_uvector<int64_t>> h_tmp_buffer{};
  raft::host_span<int64_t> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer =
      host_staging_buffer_manager::allocate_staging_buffer<int64_t>(tuple_size, stream);
  } else {
    h_tmp_buffer = std::vector<int64_t>(tuple_size);
  }
  h_tmp_buffer_view =
    std::visit([](auto& buffer) { return raft::host_span<int64_t>(buffer.data(), buffer.size()); },
               h_tmp_buffer);
  rmm::device_uvector<int64_t> d_tmp_buffer(tuple_size, stream);
  int64_t* h_staging_buffer = h_tmp_buffer_view.data();
  int64_t* d_staging_buffer = d_tmp_buffer.data();
  detail::update_array_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_staging_buffer, input);
  raft::update_device(d_staging_buffer, h_staging_buffer, tuple_size, stream);
  detail::host_allreduce_tuple_scalar_element_impl<T, size_t{0}, tuple_size>().run(
    comm, d_staging_buffer, op, stream);
  raft::update_host(h_staging_buffer, d_staging_buffer, tuple_size, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  T ret{};
  detail::update_tuple_from_array_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>().update(
    ret, h_staging_buffer);
  return ret;
}

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_reduce(
  raft::comms::comms_t const& comm, T input, raft::comms::op_t op, int root, cudaStream_t stream)
{
  std::variant<std::vector<T>, rmm::device_uvector<T>> h_tmp_buffer{};
  raft::host_span<T> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<T>(1, stream);
  } else {
    h_tmp_buffer = std::vector<T>(1);
  }
  h_tmp_buffer_view = std::visit(
    [](auto& buffer) { return raft::host_span<T>(buffer.data(), buffer.size()); }, h_tmp_buffer);
  rmm::device_uvector<T> d_tmp_buffer(1, stream);
  T* h_staging_buffer = h_tmp_buffer_view.data();
  T* d_staging_buffer = d_tmp_buffer.data();
  h_staging_buffer[0] = input;
  raft::update_device(d_staging_buffer, h_staging_buffer, 1, stream);
  comm.reduce(d_staging_buffer, d_staging_buffer, 1, op, stream);
  if (comm.get_rank() == root) { raft::update_host(h_staging_buffer, d_staging_buffer, 1, stream); }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  T h_input{};
  if (comm.get_rank() == root) { h_input = h_staging_buffer[0]; }
  return h_input;
}

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, T> host_scalar_reduce(
  raft::comms::comms_t const& comm, T input, raft::comms::op_t op, int root, cudaStream_t stream)
{
  size_t constexpr tuple_size = cuda::std::tuple_size<T>::value;

  std::variant<std::vector<int64_t>, rmm::device_uvector<int64_t>> h_tmp_buffer{};
  raft::host_span<int64_t> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer =
      host_staging_buffer_manager::allocate_staging_buffer<int64_t>(tuple_size, stream);
  } else {
    h_tmp_buffer = std::vector<int64_t>(tuple_size);
  }
  h_tmp_buffer_view =
    std::visit([](auto& buffer) { return raft::host_span<int64_t>(buffer.data(), buffer.size()); },
               h_tmp_buffer);
  rmm::device_uvector<int64_t> d_tmp_buffer(tuple_size, stream);
  int64_t* h_staging_buffer = h_tmp_buffer_view.data();
  int64_t* d_staging_buffer = d_tmp_buffer.data();
  detail::update_array_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_staging_buffer, input);
  raft::update_device(d_staging_buffer, h_staging_buffer, tuple_size, stream);
  detail::host_reduce_tuple_scalar_element_impl<T, size_t{0}, tuple_size>().run(
    comm, d_staging_buffer, op, root, stream);
  if (comm.get_rank() == root) {
    raft::update_host(h_staging_buffer, d_staging_buffer, tuple_size, stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  T ret{};
  if (comm.get_rank() == root) {
    detail::update_tuple_from_array_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>()
      .update(ret, h_staging_buffer);
  }
  return ret;
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_bcast(
  raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  std::variant<std::vector<T>, rmm::device_uvector<T>> h_tmp_buffer{};
  raft::host_span<T> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<T>(1, stream);
  } else {
    h_tmp_buffer = std::vector<T>(1);
  }
  h_tmp_buffer_view = std::visit(
    [](auto& buffer) { return raft::host_span<T>(buffer.data(), buffer.size()); }, h_tmp_buffer);
  rmm::device_uvector<T> d_tmp_buffer(1, stream);
  T* h_staging_buffer = h_tmp_buffer_view.data();
  T* d_staging_buffer = d_tmp_buffer.data();
  if (comm.get_rank() == root) {
    h_staging_buffer[0] = input;
    raft::update_device(d_staging_buffer, h_staging_buffer, 1, stream);
  }
  comm.bcast(d_staging_buffer, 1, root, stream);
  if (comm.get_rank() != root) { raft::update_host(h_staging_buffer, d_staging_buffer, 1, stream); }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  auto h_input = h_staging_buffer[0];
  return h_input;
}

template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, T> host_scalar_bcast(
  raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  size_t constexpr tuple_size = cuda::std::tuple_size<T>::value;

  std::variant<std::vector<int64_t>, rmm::device_uvector<int64_t>> h_tmp_buffer{};
  raft::host_span<int64_t> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer =
      host_staging_buffer_manager::allocate_staging_buffer<int64_t>(tuple_size, stream);
  } else {
    h_tmp_buffer = std::vector<int64_t>(tuple_size);
  }
  h_tmp_buffer_view =
    std::visit([](auto& buffer) { return raft::host_span<int64_t>(buffer.data(), buffer.size()); },
               h_tmp_buffer);
  rmm::device_uvector<int64_t> d_tmp_buffer(tuple_size, stream);
  int64_t* h_staging_buffer = h_tmp_buffer_view.data();
  int64_t* d_staging_buffer = d_tmp_buffer.data();
  if (comm.get_rank() == root) {
    detail::update_array_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>()
      .update(h_staging_buffer, input);
    raft::update_device(d_staging_buffer, h_staging_buffer, tuple_size, stream);
  }
  comm.bcast(d_staging_buffer, tuple_size, root, stream);
  if (comm.get_rank() != root) {
    raft::update_host(h_staging_buffer, d_staging_buffer, tuple_size, stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  T ret{};
  detail::update_tuple_from_array_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>().update(
    ret, h_staging_buffer);
  return ret;
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, std::vector<T>> host_scalar_allgather(
  raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  std::variant<std::vector<T>, rmm::device_uvector<T>> h_tmp_buffer{};
  raft::host_span<T> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<T>(comm.get_size(), stream);
  } else {
    h_tmp_buffer = std::vector<T>(comm.get_size());
  }
  h_tmp_buffer_view = std::visit(
    [](auto& buffer) { return raft::host_span<T>(buffer.data(), buffer.size()); }, h_tmp_buffer);
  rmm::device_uvector<T> d_tmp_buffer(comm.get_size(), stream);
  T* h_staging_buffer               = h_tmp_buffer_view.data();
  T* d_staging_buffer               = d_tmp_buffer.data();
  h_staging_buffer[comm.get_rank()] = input;
  raft::update_device(
    d_staging_buffer + comm.get_rank(), h_staging_buffer + comm.get_rank(), 1, stream);
  comm.allgather(d_staging_buffer + comm.get_rank(), d_staging_buffer, size_t{1}, stream);
  raft::update_host(h_staging_buffer, d_staging_buffer, comm.get_size(), stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  std::vector<T> h_outputs(h_staging_buffer, h_staging_buffer + comm.get_size());
  return h_outputs;
}

template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, std::vector<T>>
host_scalar_allgather(raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  size_t constexpr tuple_size = cuda::std::tuple_size<T>::value;

  std::variant<std::vector<int64_t>, rmm::device_uvector<int64_t>> h_tmp_buffer{};
  raft::host_span<int64_t> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<int64_t>(
      comm.get_size() * tuple_size, stream);
  } else {
    h_tmp_buffer = std::vector<int64_t>(comm.get_size() * tuple_size);
  }
  h_tmp_buffer_view =
    std::visit([](auto& buffer) { return raft::host_span<int64_t>(buffer.data(), buffer.size()); },
               h_tmp_buffer);
  rmm::device_uvector<int64_t> d_tmp_buffer(comm.get_size() * tuple_size, stream);
  int64_t* h_staging_buffer = h_tmp_buffer_view.data();
  int64_t* d_staging_buffer = d_tmp_buffer.data();
  detail::update_array_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_staging_buffer + comm.get_rank() * tuple_size, input);
  raft::update_device(d_staging_buffer + comm.get_rank() * tuple_size,
                      h_staging_buffer + comm.get_rank() * tuple_size,
                      tuple_size,
                      stream);
  comm.allgather(
    d_staging_buffer + comm.get_rank() * tuple_size, d_staging_buffer, tuple_size, stream);
  raft::update_host(h_staging_buffer, d_staging_buffer, comm.get_size() * tuple_size, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  std::vector<T> ret(comm.get_size());
  for (size_t i = 0; i < ret.size(); ++i) {
    detail::update_tuple_from_array_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>()
      .update(ret[i], h_staging_buffer + i * tuple_size);
  }
  return ret;
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_scatter(
  raft::comms::comms_t const& comm,
  std::vector<T> const& inputs,  // relevant only in root
  int root,
  cudaStream_t stream)
{
  CUGRAPH_EXPECTS(
    ((comm.get_rank() == root) && (inputs.size() == static_cast<size_t>(comm.get_size()))) ||
      ((comm.get_rank() != root) && (inputs.size() == 0)),
    "inputs.size() should match with comm.get_size() in root and should be 0 otherwise.");

  std::variant<std::vector<T>, rmm::device_uvector<T>> h_tmp_buffer{};
  raft::host_span<T> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<T>(comm.get_size(), stream);
  } else {
    h_tmp_buffer = std::vector<T>(comm.get_size());
  }
  h_tmp_buffer_view = std::visit(
    [](auto& buffer) { return raft::host_span<T>(buffer.data(), buffer.size()); }, h_tmp_buffer);
  rmm::device_uvector<T> d_tmp_buffer(comm.get_size(), stream);
  T* h_staging_buffer = h_tmp_buffer_view.data();
  T* d_staging_buffer = d_tmp_buffer.data();
  if (comm.get_rank() == root) {
    std::copy(inputs.begin(), inputs.end(), h_staging_buffer);
    raft::update_device(d_staging_buffer, h_staging_buffer, comm.get_size(), stream);
  }
  comm.bcast(d_staging_buffer, comm.get_size(), root, stream);
  if (comm.get_rank() != root) {
    raft::update_host(
      h_staging_buffer + comm.get_rank(), d_staging_buffer + comm.get_rank(), 1, stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  auto h_output = h_staging_buffer[comm.get_rank()];
  return h_output;
}

template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, T> host_scalar_scatter(
  raft::comms::comms_t const& comm,
  std::vector<T> const& inputs,  // relevant only in root
  int root,
  cudaStream_t stream)
{
  size_t constexpr tuple_size = cuda::std::tuple_size<T>::value;
  CUGRAPH_EXPECTS(
    ((comm.get_rank() == root) && (inputs.size() == static_cast<size_t>(comm.get_size()))) ||
      ((comm.get_rank() != root) && (inputs.size() == 0)),
    "inputs.size() should match with comm.get_size() in root and should be 0 otherwise.");

  std::variant<std::vector<int64_t>, rmm::device_uvector<int64_t>> h_tmp_buffer{};
  raft::host_span<int64_t> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<int64_t>(
      comm.get_size() * tuple_size, stream);
  } else {
    h_tmp_buffer = std::vector<int64_t>(comm.get_size() * tuple_size);
  }
  h_tmp_buffer_view =
    std::visit([](auto& buffer) { return raft::host_span<int64_t>(buffer.data(), buffer.size()); },
               h_tmp_buffer);
  rmm::device_uvector<int64_t> d_tmp_buffer(comm.get_size() * tuple_size, stream);
  int64_t* h_staging_buffer = h_tmp_buffer_view.data();
  int64_t* d_staging_buffer = d_tmp_buffer.data();
  if (comm.get_rank() == root) {
    for (int i = 0; i < comm.get_size(); ++i) {
      detail::update_array_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>()
        .update(h_staging_buffer + i * tuple_size, inputs[i]);
    }
    raft::update_device(d_staging_buffer, h_staging_buffer, comm.get_size() * tuple_size, stream);
  }
  comm.bcast(d_staging_buffer, comm.get_size() * tuple_size, root, stream);
  if (comm.get_rank() != root) {
    raft::update_host(h_staging_buffer + comm.get_rank() * tuple_size,
                      d_staging_buffer + comm.get_rank() * tuple_size,
                      tuple_size,
                      stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  T ret{};
  detail::update_tuple_from_array_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>().update(
    ret, h_staging_buffer + comm.get_rank() * tuple_size);

  return ret;
}

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, std::vector<T>> host_scalar_gather(
  raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  std::variant<std::vector<T>, rmm::device_uvector<T>> h_tmp_buffer{};
  raft::host_span<T> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<T>(comm.get_size(), stream);
  } else {
    h_tmp_buffer = std::vector<T>(comm.get_size());
  }
  h_tmp_buffer_view = std::visit(
    [](auto& buffer) { return raft::host_span<T>(buffer.data(), buffer.size()); }, h_tmp_buffer);
  rmm::device_uvector<T> d_tmp_buffer(comm.get_size(), stream);
  T* h_staging_buffer               = h_tmp_buffer_view.data();
  T* d_staging_buffer               = d_tmp_buffer.data();
  h_staging_buffer[comm.get_rank()] = input;
  raft::update_device(
    d_staging_buffer + comm.get_rank(), h_staging_buffer + comm.get_rank(), 1, stream);
  comm.gather(d_staging_buffer + comm.get_rank(), d_staging_buffer, size_t{1}, root, stream);
  if (comm.get_rank() == root) {
    raft::update_host(h_staging_buffer, d_staging_buffer, comm.get_size(), stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  std::vector<T> h_outputs{};
  if (comm.get_rank() == root) {
    h_outputs = std::vector<T>(h_staging_buffer, h_staging_buffer + comm.get_size());
  }
  return h_outputs;
}

// Return value is valid only in root (return value may better be std::optional in C++17 or later)
template <typename T>
std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<T>::value, std::vector<T>>
host_scalar_gather(raft::comms::comms_t const& comm, T input, int root, cudaStream_t stream)
{
  size_t constexpr tuple_size = cuda::std::tuple_size<T>::value;

  std::variant<std::vector<int64_t>, rmm::device_uvector<int64_t>> h_tmp_buffer{};
  raft::host_span<int64_t> h_tmp_buffer_view{};
  if (host_staging_buffer_manager::initialized()) {
    h_tmp_buffer = host_staging_buffer_manager::allocate_staging_buffer<int64_t>(
      comm.get_size() * tuple_size, stream);
  } else {
    h_tmp_buffer = std::vector<int64_t>(comm.get_size() * tuple_size);
  }
  h_tmp_buffer_view =
    std::visit([](auto& buffer) { return raft::host_span<int64_t>(buffer.data(), buffer.size()); },
               h_tmp_buffer);
  rmm::device_uvector<int64_t> d_tmp_buffer(comm.get_size() * tuple_size, stream);
  int64_t* h_staging_buffer = h_tmp_buffer_view.data();
  int64_t* d_staging_buffer = d_tmp_buffer.data();
  detail::update_array_of_tuple_scalar_elements_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_staging_buffer + comm.get_rank() * tuple_size, input);
  raft::update_device(d_staging_buffer + comm.get_rank() * tuple_size,
                      h_staging_buffer + comm.get_rank() * tuple_size,
                      tuple_size,
                      stream);
  comm.gather(
    d_staging_buffer + comm.get_rank() * tuple_size, d_staging_buffer, tuple_size, root, stream);
  if (comm.get_rank() == root) {
    raft::update_host(h_staging_buffer, d_staging_buffer, comm.get_size() * tuple_size, stream);
  }
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  std::vector<T> ret(comm.get_size());
  if (comm.get_rank() == root) {
    for (size_t i = 0; i < ret.size(); ++i) {
      detail::update_tuple_from_array_of_tuple_scalar_elements_impl<T, size_t{0}, tuple_size>()
        .update(ret[i], h_staging_buffer + i * tuple_size);
    }
  }

  return ret;
}

}  // namespace cugraph
