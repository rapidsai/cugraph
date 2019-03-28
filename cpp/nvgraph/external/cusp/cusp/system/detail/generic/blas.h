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

#pragma once

#include <cusp/array1d.h>
#include <cusp/complex.h>
#include <cusp/exception.h>
#include <cusp/functional.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

#include <thrust/iterator/transform_iterator.h>

#include <cmath>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{
namespace blas
{

template <typename T>
struct SCAL
{
    T alpha;

    SCAL(T _alpha)
        : alpha(_alpha) {}

    template <typename T2>
    __host__ __device__
    void operator()(T2& x)
    {
        x = T(alpha) * x;
    }
};


template <typename T>
struct AXPY
{
    T alpha;

    AXPY(T _alpha)
        : alpha(_alpha) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<1>(t) = alpha * thrust::get<0>(t) +
                            thrust::get<1>(t);
    }
};

template <typename T1, typename T2>
struct AXPBY
{
    T1 alpha;
    T2 beta;

    AXPBY(T1 _alpha, T2 _beta)
        : alpha(_alpha), beta(_beta) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<2>(t) = alpha * thrust::get<0>(t) +
                            beta  * thrust::get<1>(t);
    }
};

template <typename T1,typename T2,typename T3>
struct AXPBYPCZ
{
    T1 alpha;
    T2 beta;
    T3 gamma;

    AXPBYPCZ(T1 _alpha, T2 _beta, T3 _gamma)
        : alpha(_alpha), beta(_beta), gamma(_gamma) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<3>(t) = alpha * thrust::get<0>(t) +
                            beta  * thrust::get<1>(t) +
                            gamma * thrust::get<2>(t);
    }
};

template <typename T>
struct XMY : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& x, const T& y)
    {
        return x * y;
    }
};

template<typename T>
struct AMAX : public thrust::binary_function<T,T,bool>
{
    __host__ __device__
    bool operator()(const T& lhs, const T& rhs)
    {
        return cusp::abs(lhs) < cusp::abs(rhs);
    }
};

template <typename DerivedPolicy,
          typename Array>
int amax(thrust::execution_policy<DerivedPolicy>& exec,
         const Array& x)
{
    typedef typename Array::value_type                    ValueType;

#if THRUST_VERSION >= 100800
    typedef typename Array::const_iterator                Iterator;
    typedef cusp::abs_functor<ValueType>                  UnaryOp;
    typedef thrust::transform_iterator<UnaryOp, Iterator> TransformIterator;

    TransformIterator iter(x.begin(), UnaryOp());
    int index = thrust::max_element(exec, iter, iter + x.size()) - iter;
#else
    int index = thrust::max_element(exec, x.begin(), x.end(), AMAX<ValueType>()) - x.begin();
#endif

    return index;
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
asum(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type   ValueType;
    typedef typename cusp::norm_type<ValueType>::type NormType;

    cusp::abs_functor<ValueType> unary_op;
    thrust::plus<NormType>       binary_op;

    NormType init = 0;

    return thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(thrust::execution_policy<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    typedef typename Array1::value_type ValueType;

    cusp::assert_same_dimensions(x, y);

    size_t N = x.size();

    thrust::for_each(exec,
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())) + N,
                     AXPY<ValueType>(alpha));
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(thrust::execution_policy<DerivedPolicy> &exec,
           const Array1& x,
           const Array2& y,
                 Array3& z,
           const ScalarType1 alpha,
           const ScalarType2 beta)
{
    typedef typename Array1::value_type ValueType;

    cusp::assert_same_dimensions(x, y, z);

    size_t N = x.size();

    thrust::for_each(exec,
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())) + N,
                     AXPBY<ValueType,ValueType>(alpha, beta));
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3,
          typename Array4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(thrust::execution_policy<DerivedPolicy> &exec,
              const Array1& x,
              const Array2& y,
              const Array3& z,
                    Array4& output,
              const ScalarType1 alpha,
              const ScalarType2 beta,
              const ScalarType3 gamma)
{
    typedef typename Array1::value_type ValueType;

    cusp::assert_same_dimensions(x, y, z, output);

    size_t N = x.size();

    thrust::for_each(exec,
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())) + N,
                     AXPBYPCZ<ValueType,ValueType,ValueType>(alpha, beta, gamma));
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3>
void xmy(thrust::execution_policy<DerivedPolicy> &exec,
         const Array1& x,
         const Array2& y,
               Array3& z)
{
    typedef typename Array3::value_type ValueType;

    cusp::assert_same_dimensions(x, y, z);

    thrust::transform(exec, x.begin(), x.end(), y.begin(), z.begin(), XMY<ValueType>());
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y)
{
    cusp::assert_same_dimensions(x, y);

    thrust::copy(exec, x.begin(), x.end(), y.begin());
}

// TODO properly harmonize heterogenous types
template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(thrust::execution_policy<DerivedPolicy>& exec,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array1::value_type OutputType;

    cusp::assert_same_dimensions(x, y);

    return thrust::inner_product(exec, x.begin(), x.end(), y.begin(), OutputType(0));
}

// TODO properly harmonize heterogenous types
template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dotc(thrust::execution_policy<DerivedPolicy>& exec,
     const Array1& x,
     const Array2& y)
{
    typedef typename Array1::value_type OutputType;

    cusp::assert_same_dimensions(x, y);

    return thrust::inner_product(exec,
                                 thrust::make_transform_iterator(x.begin(), cusp::conj_functor<OutputType>()),
                                 thrust::make_transform_iterator(x.end(),   cusp::conj_functor<OutputType>()),
                                 y.begin(),
                                 OutputType(0));
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void fill(thrust::execution_policy<DerivedPolicy>& exec,
          Array& x,
          const ScalarType alpha)
{
    thrust::fill(exec, x.begin(), x.end(), alpha);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm2(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type   ValueType;
    typedef typename cusp::norm_type<ValueType>::type NormType;

    cusp::abs_squared_functor<ValueType> unary_op;
    thrust::plus<NormType>               binary_op;

    NormType init = 0;

    return std::sqrt(thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op));
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void scal(thrust::execution_policy<DerivedPolicy>& exec,
          Array& x,
          const ScalarType alpha)
{
    thrust::for_each(exec, x.begin(), x.end(), SCAL<ScalarType>(alpha));
}

template<typename DerivedPolicy,
         typename Array2d,
         typename Array1d1,
         typename Array1d2,
         typename ScalarType1,
         typename ScalarType2>
void gemv(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d&  A,
          const Array1d1& x,
                Array1d2& y,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    throw cusp::not_implemented_exception("CUSP GEMV not implemented");
}

template <typename DerivedPolicy,
          typename Array1d1,
          typename Array1d2,
          typename Array2d1,
          typename ScalarType>
void ger(thrust::execution_policy<DerivedPolicy> &exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A,
         const ScalarType alpha)
{
    throw cusp::not_implemented_exception("CUSP GER not implemented");
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2,
          typename ScalarType1,
          typename ScalarType2>
void symv(thrust::execution_policy<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    throw cusp::not_implemented_exception("CUSP SYMV not implemented");
}

template <typename DerivedPolicy,
          typename Array1d,
          typename Array2d,
          typename ScalarType>
void syr(thrust::execution_policy<DerivedPolicy> &exec,
         const Array1d& x,
               Array2d& A,
         const ScalarType alpha)
{
    throw cusp::not_implemented_exception("CUSP SYR not implemented");
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trmv(thrust::execution_policy<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x)
{
    throw cusp::not_implemented_exception("CUSP TRMV not implemented");
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trsv(thrust::execution_policy<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x)
{
    throw cusp::not_implemented_exception("CUSP TRSV not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3,
         typename ScalarType1,
         typename ScalarType2>
void gemm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    throw cusp::not_implemented_exception("CUSP GEMM not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3,
         typename ScalarType1,
         typename ScalarType2>
void symm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    throw cusp::not_implemented_exception("CUSP SYMM not implemented");
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename ScalarType1,
          typename ScalarType2>
void syrk(thrust::execution_policy<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    throw cusp::not_implemented_exception("CUSP SYRK not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3,
         typename ScalarType1,
         typename ScalarType2>
void syr2k(thrust::execution_policy<DerivedPolicy>& exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C,
           const ScalarType1 alpha,
           const ScalarType2 beta)
{
    throw cusp::not_implemented_exception("CUSP SYR2K not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename ScalarType>
void trmm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType alpha)
{
    throw cusp::not_implemented_exception("CUSP TRMM not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename ScalarType>
void trsm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B,
          const ScalarType alpha)
{
    throw cusp::not_implemented_exception("CUSP TRSM not implemented");
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrm1(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    return cusp::blas::asum(exec, x);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::norm_type<typename Array::value_type>::type
nrmmax(thrust::execution_policy<DerivedPolicy>& exec,
       const Array& x)
{
    typedef typename Array::value_type ValueType;

    int index = cusp::blas::amax(exec, x);
    ValueType val = *(x.begin() + index);

    return cusp::abs(val);
}

} // end namespace blas
} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

