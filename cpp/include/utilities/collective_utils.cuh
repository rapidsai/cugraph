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

#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

template <typename TupleType, size_t I, size_t N>
struct update_tuple_elements_vector_from_tuple_impl {
  void update(std::vector<int64_t>& tuple_elements, TupleType const& tuple) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_elements.data() + I);
    *ptr     = thrust::get<I>(tuple);
    update_tuple_elements_vector_from_tuple_impl<TupleType, I + 1, N>().update(tuple_elements,
                                                                               tuple);
  }
};

template <typename TupleType, size_t I>
struct update_tuple_elements_vector_from_tuple_impl<TupleType, I, I> {
  void update(std::vector<int64_t>& tuple_elements, TupleType const& tuple) const { return; }
};

template <typename TupleType, size_t I, size_t N>
struct update_tuple_from_tuple_elements_vector_impl {
  void update(TupleType& tuple, std::vector<int64_t> const& tuple_elements) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr              = reinterpret_cast<element_t const*>(tuple_elements.data() + I);
    thrust::get<I>(tuple) = *ptr;
    update_tuple_from_tuple_elements_vector_impl<TupleType, I + 1, N>().update(tuple,
                                                                               tuple_elements);
  }
};

template <typename TupleType, size_t I>
struct update_tuple_from_tuple_elements_vector_impl<TupleType, I, I> {
  void update(TupleType& tuple, std::vector<int64_t> const& tuple_elements) const { return; }
};

template <typename TupleType, size_t I, size_t N>
struct allreduce_tuple_element_impl {
  void run(raft::comms::comms_t const& comm,
           rmm::device_uvector<int64_t>& tuple_elements,
           cudaStream_t stream) const
  {
    using element_t = typename thrust::tuple_element<I, TupleType>::type;
    static_assert(sizeof(element_t) <= sizeof(int64_t));
    auto ptr = reinterpret_cast<element_t*>(tuple_elements.data() + I);
    comm.allreduce(ptr, ptr, 1, raft::comms::op_t::SUM, stream);
  }
};

template <typename TupleType, size_t I>
struct allreduce_tuple_element_impl<TupleType, I, I> {
  void run(raft::comms::comms_t const& comm,
           rmm::device_uvector<int64_t>& tuple_elements,
           cudaStream_t stream) const
  {
    return;
  }
};

}  // namespace detail

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, T> host_scalar_allreduce(
  raft::comms::comms_t const& comm, T input, cudaStream_t stream)
{
  rmm::device_uvector<T> d_input(1, stream);
  T h_input{};
  raft::update_device(d_input.data(), &input, 1, stream);
  comm.allreduce(d_input.data(), d_input.data(), 1, raft::comms::op_t::SUM, stream);
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
  std::vector<int64_t> h_tuple_elements(tuple_size);
  rmm::device_uvector<int64_t> d_tuple_elements(tuple_size, stream);
  T ret{};

  detail::update_tuple_elements_vector_from_tuple_impl<T, size_t{0}, tuple_size>().update(
    h_tuple_elements, input);
  raft::update_device(d_tuple_elements.data(), h_tuple_elements.data(), tuple_size, stream);
  // FIXME: these broadcasts can be placed between ncclGroupStart() and ncclGroupEnd()
  detail::allreduce_tuple_element_impl<T, size_t{0}, tuple_size>(comm, d_tuple_elements, stream);
  raft::update_host(h_tuple_elements.data(), d_tuple_elements.data(), tuple_size, stream);
  auto status = comm.sync_stream(stream);
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");
  detail::update_tuple_from_tuple_elements_vector_impl<T, size_t{0}, tuple_size>().update(
    ret, h_tuple_elements);

  return ret;
}

}  // namespace experimental
}  // namespace cugraph