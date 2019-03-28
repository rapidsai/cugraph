/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#include <cusp/detail/config.h>

#include <cusp/complex.h>

#include <thrust/functional.h>

namespace cusp
{
namespace detail
{

template <typename BinaryFunction>
struct base_functor
  : public thrust::unary_function<typename BinaryFunction::first_argument_type, typename BinaryFunction::result_type>
{
    public:

    typedef typename BinaryFunction::first_argument_type T;
    typedef typename BinaryFunction::result_type         result_type;

    T value;
    BinaryFunction op;

    __host__ __device__
    base_functor(const T value) : value(value) {}

    __host__ __device__
    base_functor operator=(const base_functor& base)
    {
        value = base.value;
        op    = base.op;
        return *this;
    }

    __host__ __device__
    result_type operator()(const T x)
    {
        return op(x, value);
    }
};

template<typename IndexType>
struct coo_tuple_comp_functor
{
    template<typename Tuple1, typename Tuple2>
    __host__ __device__
    bool operator()(const Tuple1& t1, const Tuple2& t2) const
    {
        const IndexType i1 = thrust::get<0>(t1);
        const IndexType j1 = thrust::get<1>(t1);
        const IndexType i2 = thrust::get<0>(t2);
        const IndexType j2 = thrust::get<1>(t2);

        return (i1 < i2) || ((i1 == i2) && (j1 < j2));
    }
};

template <typename BinaryFunction>
struct combine_tuple_base_functor
        : public thrust::unary_function<
        thrust::tuple<typename BinaryFunction::first_argument_type,
        typename BinaryFunction::second_argument_type>,
        typename BinaryFunction::result_type>
{
    BinaryFunction op;

    template<typename Tuple>
    __host__ __device__
    typename BinaryFunction::result_type
    operator()(const Tuple& t)
    {
        return op(thrust::get<0>(t),thrust::get<1>(t));
    }
};

template <typename IndexType>
struct occupied_diagonal_functor
{
    typedef IndexType result_type;

    const   IndexType num_rows;

    occupied_diagonal_functor(const IndexType num_rows)
        : num_rows(num_rows) {}

    template <typename Tuple>
    __host__ __device__
    IndexType operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);

        return j - i + num_rows;
    }
};

struct speed_threshold_functor
{
    size_t num_rows;
    float  relative_speed;
    size_t breakeven_threshold;

    speed_threshold_functor(const size_t num_rows, const float relative_speed, const size_t breakeven_threshold)
        : num_rows(num_rows),
          relative_speed(relative_speed),
          breakeven_threshold(breakeven_threshold)
    {}

    template <typename IndexType>
    __host__ __device__
    bool operator()(const IndexType rows) const
    {
        return relative_speed * (num_rows-rows) < num_rows || (size_t) (num_rows-rows) < breakeven_threshold;
    }
};

template <typename IndexType>
struct diagonal_index_functor : public thrust::unary_function<IndexType,IndexType>
{
    const IndexType pitch;

    diagonal_index_functor(const IndexType pitch)
        : pitch(pitch) {}

    template <typename Tuple>
    __host__ __device__
    IndexType operator()(const Tuple& t) const
    {
        const IndexType row  = thrust::get<0>(t);
        const IndexType diag = thrust::get<1>(t);

        return (diag * pitch) + row;
    }
};

template <typename IndexType>
struct is_valid_ell_index_functor
{
    const IndexType num_rows;

    is_valid_ell_index_functor(const IndexType num_rows)
        : num_rows(num_rows) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);

        return i < num_rows && j != IndexType(-1);
    }
};

template <typename IndexType, typename ValueType>
struct is_valid_coo_index_functor
{
    const IndexType num_rows;
    const IndexType num_cols;

    is_valid_coo_index_functor(const IndexType num_rows, const IndexType num_cols)
        : num_rows(num_rows), num_cols(num_cols) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);
        const ValueType value = thrust::get<2>(t);

        return ( i > IndexType(-1) && i < num_rows ) &&
               ( j > IndexType(-1) && j < num_cols ) &&
               ( value != ValueType(0) ) ;
    }
};

} // end namespace detail
} // end namespace cusp

