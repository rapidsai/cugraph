/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device_core.hpp"

#include <cstdint>
#include <tuple>
#include <type_traits>

namespace cugraph::ops::utils {

// The following struct must be used to transmit the size and alignment of
// a field to the shared memory helpers below.
// By default, the alignment is just like the alignment of the original data type.
template <typename DataT, int32_t ALIGN = 0>
struct field_type {
  using data_t                       = DataT;
  static constexpr int32_t BYTES     = static_cast<int32_t>(sizeof(DataT));
  static constexpr int32_t ALIGNMENT = ALIGN > 0 ? ALIGN : alignof(DataT);
};

// Imagine we have 2 fields of data in shared memory, one for ints, one for doubles.
// The intended usage of the following class in simple cases is as follows:
// 1. specify the type somewhere for both host and kernel code:
//    using special_smem_name_t = smem_helper< 0, 0, field_type<int>, field_type<double> >;
//    /* can be simplified to the following: */
//    using special_smem_name_t = smem_simple_t< int, double >;
// 2. in host code, get the size of shared memory:
//    int32_t smem_sizes[] = {n_ints, n_doubles};
//    /* note: sizes are always in number of elements, not bytes */
//    /*       sizes always have type `int32_t` */
//    auto size = special_smem_name_t::get_size(sizes);
// 3. in device code, call the empty constructor:
//    special_smem_name_t helper {};
//    int* s_ints;
//    double* s_doubles;
//    int32_t smem_sizes[] = {n_ints, n_doubles};
//    helper.set_ptrs(sizes, s_ints, s_doubles);
//
// For more complicated use cases, it is often useful to create a struct overloading
// operator[] and passing that to the `get_size` or `set_ptrs` helpers.
// The struct can also be used to directly pass the size information from
// host code (launch) to the kernel, avoiding duplication of calculating sizes.
// Be aware that this overload must have a `__host__ __device__` signature.
// Here is an example struct for the above use case:
// struct sizes_t {
//   int32_t n_ints, n_doubles;
//   __host__ __device__ sizes_t() = delete;
//   __host__ __device__ sizes_t(int32_t _n_ints, int32_t _n_doubles) :
//     n_ints(_n_ints), n_doubles(_n_doubles) {}
//
//   /* you may also just return int32_t here instead of const int32_t& */
//   __host__ __device__ const int32_t& operator[](int idx) const
//   {
//     return idx == 0 ? n_ints : n_doubles;
//   }
// };
//
// The ALIGN_INIT template parameter is important for correctness:
// By default (ALIGN_INIT=0), we assume that all alignments are powers of 2,
// and we set ALIGN_INIT to the max alignment of the fields. If you want more
// control, you can set it yourself, but we always assume that it is a multiple
// of all alignment values of the fields.
//
// The N_UNIT_FIELDS template parameters allows specifying sub-spaces
// for a given number of "units" (often warps) such that the first
// `N_UNIT_FIELDS` fields are reserved sub-spaces per unit.
// In this case, the `get_size` and `set_ptrs` methods are modified such that
// you have to specify the number of units, and for `set_ptrs` the unit ID
// as well.
// This is useful for reserving exclusive shared memory per warp for example.
// Each unit (warp) will have its sub-space (containing the `N_UNIT_FIELDS`
// fields) aligned to the initial alignment as described above.
template <int32_t ALIGN_INIT, int N_UNIT_FIELDS, typename... FieldsT>
class smem_helper {
 public:
  static constexpr size_t N_ARGS = sizeof...(FieldsT);

 protected:
  static_assert(N_ARGS > 0, "smem_helper: must have at least one field type");
  static_assert(N_UNIT_FIELDS >= 0, "smem_helper: #unit fields must be non-negative");
  static_assert(N_UNIT_FIELDS <= N_ARGS,
                "smem_helper: #unit fields must be smaller than #field types");
  // following static assertion for FieldsT to not be scalar types is based on
  // https://stackoverflow.com/a/28253503/4134127
  template <bool...>
  struct bool_pack;
  template <bool... BOOLS>
  using all_true_t = std::is_same<bool_pack<true, BOOLS...>, bool_pack<BOOLS..., true>>;
  static_assert(all_true_t<!std::is_scalar<FieldsT>::value...>::value,
                "smem_helper: the given field template types must be of type `field_type` and "
                "cannot be scalars");

  template <int IDX>
  __host__ __device__ static constexpr typename std::enable_if<(IDX < N_ARGS), int32_t>::type
  max_align()
  {
    using f_t = typename std::tuple_element<IDX, std::tuple<FieldsT...>>::type;
    static_assert(f_t::ALIGNMENT > 0, "field alignments must be greater than 0");
    return max_align<IDX + 1>() > f_t::ALIGNMENT ? max_align<IDX + 1>() : f_t::ALIGNMENT;
  }
  template <int IDX>
  __host__ __device__ static constexpr typename std::enable_if<(IDX >= N_ARGS), int32_t>::type
  max_align()
  {
    return -1;
  }

  // this is assumed to be a multiple of all alignments
  static constexpr int32_t ALIGN_BASE = ALIGN_INIT > 0 ? ALIGN_INIT : max_align<0>();

  // here we exploit that the base pointer must be aligned to 16 bytes.
  // if 16 is a multiple of ALIGN_BASE, that means we don't have any overhead.
  // if ALIGN_BASE is a multiple of 16, it means that we need at most
  // ALIGN_BASE - 16 extra bytes, otherwise it's ALIGN_BASE - 1
  static constexpr int32_t SIZE_OVERHEAD = 16 % ALIGN_BASE == 0   ? 0
                                           : ALIGN_BASE % 16 == 0 ? ALIGN_BASE - 16
                                                                  : ALIGN_BASE - 1;

 public:
  // cannot easily use "= default" here for host-only code
  // NOLINTNEXTLINE(modernize-use-equals-default)
  __host__ __device__ smem_helper()
  {
#if defined(__CUDA_ARCH__)
    // must be aligned to 16 bytes on all supported architectures
    // (don't have a reference for this at the moment!)
    extern __shared__ uint8_t smem[];
    // align manually to `ALIGN_BASE`: this avoids the `__align(X)__` attribute
    // which can cause issues if this is used in the same compilation unit
    // with different types / alignments.
    // In any case, the compiler/hardware cannot do a better job at providing
    // an aligned pointer than we can do manually.
    auto smem_aligned = align_to(reinterpret_cast<uintptr_t>(smem), uintptr_t(ALIGN_BASE));
    base_ptr_         = reinterpret_cast<uint8_t*>(smem_aligned);
#endif
  }

  template <typename SizeT, int N = N_UNIT_FIELDS>
  __host__ __device__ static inline typename std::enable_if<(N <= 0), int32_t>::type get_size(
    const SizeT& sizes)
  {
    auto current_total = 0;  // base pointer must be aligned to ALIGN_BASE
    size_helper<1>(current_total, sizes);
    return SIZE_OVERHEAD + current_total;
  }

  template <typename SizeT, int N = N_UNIT_FIELDS>
  __host__ __device__ static inline typename std::enable_if<(N > 0), int32_t>::type get_size(
    const int32_t n_units, const SizeT& sizes)
  {
    auto current_total = 0;  // base pointer must be aligned to all alignments
    unit_size_helper<1>(current_total, sizes);
    // since the unit size is aligned to ALIGN_BASE, every base pointer for
    // each unit as well as the base pointer after all units is aligned to
    // ALIGN_BASE: since that is a multiple of all alignments, we can safely
    // continue adding the sizes afterwards
    auto unit_size = align_to(current_total, ALIGN_BASE);
    current_total  = 0;  // base pointer must be aligned to all alignments
    size_helper<N + 1>(current_total, sizes);
    return SIZE_OVERHEAD + unit_size * n_units + current_total;
  }

  template <typename SizeT, int N = N_UNIT_FIELDS>
  __device__ inline typename std::enable_if<(N <= 0)>::type set_ptrs(
    const SizeT& sizes, typename FieldsT::data_t*&... ptrs) const
  {
    return ptrs_helper<1>(0, 0, 0, 0, sizes, ptrs...);
  }

  template <typename SizeT, int N = N_UNIT_FIELDS>
  __device__ inline typename std::enable_if<(N > 0)>::type set_ptrs(
    const int32_t& unit_id,
    const int32_t& n_units,
    const SizeT& sizes,
    typename FieldsT::data_t*&... ptrs) const
  {
    auto current_total = 0;  // base pointer must be aligned to all alignments
    unit_size_helper<1>(current_total, sizes);
    // see explanation in `get_size` for what aligning to ALIGN_BASE means
    auto unit_size = align_to(current_total, ALIGN_BASE);
    return ptrs_helper<1>(0, unit_id, unit_size, n_units, sizes, ptrs...);
  }

 protected:
  template <int NEXT, typename SizeT>
  __host__ __device__ static inline void single_size(int32_t& current_total, const SizeT& sizes)
  {
    using next_field_t = typename std::tuple_element<(NEXT < N_ARGS ? NEXT : N_ARGS - 1),
                                                     std::tuple<FieldsT...>>::type;
    using this_field_t = typename std::tuple_element<(NEXT < N_ARGS ? NEXT - 1 : N_ARGS - 1),
                                                     std::tuple<FieldsT...>>::type;
    static constexpr int32_t ALIGN =
      NEXT == N_UNIT_FIELDS || NEXT >= N_ARGS ? 1 : next_field_t::ALIGNMENT;
    current_total = align_to(current_total + sizes[NEXT - 1] * this_field_t::BYTES, ALIGN);
  }

  // parentheses in `enable_if` here are used to help the parser understand "<>"
  template <int NEXT, typename SizeT>
  __host__ __device__ static inline typename std::enable_if<(NEXT <= N_ARGS)>::type size_helper(
    int32_t& current_total, const SizeT& sizes)
  {
    single_size<NEXT>(current_total, sizes);
    size_helper<NEXT + 1>(current_total, sizes);
  }
  template <int NEXT, typename SizeT>
  __host__ __device__ static inline typename std::enable_if<(NEXT > N_ARGS)>::type size_helper(
    int32_t& /* current_total */, const SizeT& /* sizes */)
  {
  }

  template <int NEXT, typename SizeT>
  __host__ __device__ static inline typename std::enable_if<(NEXT <= N_UNIT_FIELDS)>::type
  unit_size_helper(int32_t& current_total, const SizeT& sizes)
  {
    single_size<NEXT>(current_total, sizes);
    unit_size_helper<NEXT + 1>(current_total, sizes);
  }
  template <int NEXT, typename SizeT>
  __host__ __device__ static inline typename std::enable_if<(NEXT > N_UNIT_FIELDS)>::type
  unit_size_helper(int32_t& /* current_total */, const SizeT& /* sizes */)
  {
  }

  template <int /* NEXT */, typename SizeT>
  __device__ inline void ptrs_helper(const int32_t& /* offset */,
                                     const int32_t& /* unit_id */,
                                     const int32_t& /* unit_size */,
                                     const int32_t& /* n_units */,
                                     const SizeT& /* sizes */) const
  {
  }
  template <int NEXT, typename SizeT, typename PtrT, typename... PtrsT>
  __device__ inline void ptrs_helper(const int32_t& offset,
                                     const int32_t& unit_id,
                                     const int32_t& unit_size,
                                     const int32_t& n_units,
                                     const SizeT& sizes,
                                     PtrT*& ptr,
                                     PtrsT*&... ptrs) const
  {
    // see `get_size`: base_ptr_ + u_off is always aligned to all alignments
    // (whether for each individual unit or after all units)
    auto u_off          = NEXT <= N_UNIT_FIELDS ? unit_id * unit_size : n_units * unit_size;
    ptr                 = reinterpret_cast<PtrT*>(base_ptr_ + (u_off + offset));
    int32_t next_offset = offset;
    if (NEXT == N_UNIT_FIELDS)
      next_offset = 0;  // pointer after all unit fields is aligned to all alignments
    else
      single_size<NEXT>(next_offset, sizes);
    ptrs_helper<NEXT + 1>(next_offset, unit_id, unit_size, n_units, sizes, ptrs...);
  }

  uint8_t* base_ptr_{nullptr};
};

template <typename... DataT>
using smem_simple_t = smem_helper<0, 0, field_type<DataT>...>;

template <int N_UNIT_FIELDS, typename... DataT>
using smem_unit_simple_t = smem_helper<0, N_UNIT_FIELDS, field_type<DataT>...>;

}  // namespace cugraph::ops::utils
