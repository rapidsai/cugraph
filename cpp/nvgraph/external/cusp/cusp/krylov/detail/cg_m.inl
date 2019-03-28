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


#include <cusp/array1d.h>
#include <cusp/complex.h>
#include <cusp/linear_operator.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>

#include <cusp/blas/blas.h>

#include <cusp/detail/temporary_array.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

#include <thrust/iterator/transform_iterator.h>

/*
 * The point of these routines is to solve systems of the type
 *
 * (A+\sigma Id)x = b
 *
 * for a number of different \sigma, iteratively, for sparse A, without
 * additional matrix-vector multiplication.
 *
 * The idea comes from the following paper:
 *     Krylov space solvers for shifted linear systems
 *     B. Jegerlehner
 *     http://arxiv.org/abs/hep-lat/9612014
 *
 * This implementation was contributed by Greg van Anders.
 *
 */

namespace cusp
{
namespace krylov
{
namespace cg_detail
{

// structs in this namespace do things that are somewhat blas-like, but
// are not usual blas operations (e.g. they aren't all linear in all arguments)
//
// except for KERNEL_VCOPY all of these structs perform operations that
// are specific to CG-M
namespace detail_m
{
// computes new \zeta, \beta
template <typename ScalarType>
struct KERNEL_ZB
{
    ScalarType beta_m1;
    ScalarType beta_0;
    ScalarType alpha_0;

    KERNEL_ZB(ScalarType _beta_m1, ScalarType _beta_0, ScalarType _alpha_0)
        : beta_m1(_beta_m1), beta_0(_beta_0), alpha_0(_alpha_0)
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        typedef typename cusp::norm_type<ScalarType>::type NormType;

        // compute \zeta_1^\sigma
        ScalarType z1, b0, z0=thrust::get<2>(t), zm1 = thrust::get<3>(t),
                           sigma = thrust::get<4>(t);
        z1 = z0*zm1*beta_m1/(beta_0*alpha_0*(zm1-z0)
                             +beta_m1*zm1*(ScalarType(1)-beta_0*sigma));
        b0 = beta_0*z1/z0;
        if ( cusp::abs(z1) < NormType(1e-30) )
            z1 = ScalarType(1e-18);
        thrust::get<0>(t) = z1;
        thrust::get<1>(t) = b0;
    }
};

// computes new alpha
template <typename ScalarType>
struct KERNEL_A
{
    ScalarType beta_0;
    ScalarType alpha_0;

    // note: only the ratio alpha_0/beta_0 enters in the computation, it might
    // be better just to pass this ratio
    KERNEL_A(ScalarType _beta_0, ScalarType _alpha_0)
        : beta_0(_beta_0), alpha_0(_alpha_0)
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // compute \alpha_0^\sigma
        thrust::get<0>(t)=alpha_0/beta_0*thrust::get<2>(t)*thrust::get<3>(t)/
                          thrust::get<1>(t);
    }
};

// computes new x
template <typename ScalarType>
struct KERNEL_XP
{
    int N;
    const ScalarType *alpha_0_s;
    const ScalarType *beta_0_s;
    const ScalarType *z_1_s;
    const ScalarType *r_0;

    KERNEL_XP(int _N, const ScalarType *_alpha_0_s, const ScalarType *_beta_0_s,
              const ScalarType *_z_1_s, const ScalarType *_r_0) :
        N(_N), alpha_0_s(_alpha_0_s),
        beta_0_s(_beta_0_s), z_1_s(_z_1_s), r_0(_r_0) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // return the transformed result
        ScalarType x = thrust::get<0>(t);
        ScalarType p_0 = thrust::get<1>(t);
        int index = thrust::get<2>(t);

        int N_s = index / N;
        int N_i = index % N;

        x = x-beta_0_s[N_s]*p_0;
        p_0 = z_1_s[N_s]*r_0[N_i]+alpha_0_s[N_s]*p_0;

        thrust::get<0>(t) = x;
        thrust::get<1>(t) = p_0;
    }
};

// like blas::copy, but copies the same array many times into a larger array
template <typename ScalarType>
struct KERNEL_VCOPY : thrust::unary_function<int, ScalarType>
{
    int N_t;
    const ScalarType *source;

    KERNEL_VCOPY(int _N_t, const ScalarType *_source) :
        N_t(_N_t), source(_source)
    {}

    __host__ __device__
    ScalarType operator()(int index)
    {
        unsigned int N   = index % N_t;
        return source[N];
    }

};

struct KERNEL_DCOPY
{
    KERNEL_DCOPY() {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<2>(t)=thrust::get<1>(t);
        thrust::get<1>(t)=thrust::get<0>(t);
    }
};

template <typename T>
struct XPAY : public thrust::binary_function<T,T,T>
{
    T alpha;

    XPAY(T _alpha) : alpha(_alpha) {}

    __host__ __device__
    T operator()(T x, T y)
    {
        return x + alpha * y;
    }
};

} // end namespace detail_m

// Methods in this namespace are all routines that involve using
// thrust::for_each to perform some transformations on arrays of data.
//
// Except for vectorize_copy, these are specific to CG-M.
//
// Each has a version that takes Array inputs, and another that takes iterators
// as input. The CG-M routine only explicitly refers version with Arrays as
// arguments. The Array version calls the iterator version which uses
// a struct from cusp::krylov::detail_m.
namespace trans_m
{
// compute \zeta_1^\sigma, \beta_0^\sigma using iterators
// uses detail_m::KERNEL_ZB
template <typename InputIterator1, typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1, typename OutputIterator2,
         typename ScalarType>
void compute_zb_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
                  InputIterator2 z_m1_s_b, InputIterator3 sig_b,
                  OutputIterator1 z_1_s_b, OutputIterator2 b_0_s_b,
                  ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0)
{
    size_t N = z_0_s_e - z_0_s_b;
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(z_1_s_b,b_0_s_b,z_0_s_b,z_m1_s_b,sig_b)),
        thrust::make_zip_iterator(thrust::make_tuple(z_1_s_b,b_0_s_b,z_0_s_b,z_m1_s_b,sig_b))+N,
        cusp::krylov::cg_detail::detail_m::KERNEL_ZB<ScalarType>(beta_m1,beta_0,alpha_0)
    );
}

// compute \zeta_1^\sigma, \beta_0^\sigma using arrays
template <typename Array1, typename Array2, typename Array3,
         typename Array4, typename Array5, typename ScalarType>
void compute_zb_m(const Array1& z_0_s, const Array2& z_m1_s,
                  const Array3& sig, Array4& z_1_s, Array5& b_0_s,
                  ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0)
{
    // sanity checks
    cusp::assert_same_dimensions(z_0_s,z_m1_s,z_1_s);
    cusp::assert_same_dimensions(z_1_s,b_0_s,sig);

    // compute
    cusp::krylov::cg_detail::trans_m::compute_zb_m(z_0_s.begin(),z_0_s.end(),
                                        z_m1_s.begin(),sig.begin(),z_1_s.begin(),b_0_s.begin(),
                                        beta_m1,beta_0,alpha_0);

}

// compute \alpha_0^\sigma, and swap \zeta_i^\sigma using iterators
// uses detail_m::KERNEL_A
template <typename InputIterator1, typename InputIterator2,
         typename InputIterator3, typename OutputIterator,
         typename ScalarType>
void compute_a_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
                 InputIterator2 z_1_s_b, InputIterator3 beta_0_s_b,
                 OutputIterator alpha_0_s_b,
                 ScalarType beta_0, ScalarType alpha_0)
{
    size_t N = z_0_s_e - z_0_s_b;
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(alpha_0_s_b,z_0_s_b,z_1_s_b,beta_0_s_b)),
        thrust::make_zip_iterator(thrust::make_tuple(alpha_0_s_b,z_0_s_b,z_1_s_b,beta_0_s_b))+N,
        cusp::krylov::cg_detail::detail_m::KERNEL_A<ScalarType>(beta_0,alpha_0));
}

// compute \alpha_0^\sigma, and swap \zeta_i^\sigma using arrays
template <typename Array1, typename Array2, typename Array3,
         typename Array4, typename ScalarType>
void compute_a_m(const Array1& z_0_s, const Array2& z_1_s,
                 const Array3& beta_0_s, Array4& alpha_0_s,
                 ScalarType beta_0, ScalarType alpha_0)
{
    // sanity checks
    cusp::assert_same_dimensions(z_0_s,z_1_s);
    cusp::assert_same_dimensions(z_0_s,alpha_0_s,beta_0_s);

    // compute
    cusp::krylov::cg_detail::trans_m::compute_a_m(z_0_s.begin(), z_0_s.end(),
                                       z_1_s.begin(), beta_0_s.begin(), alpha_0_s.begin(),
                                       beta_0, alpha_0);
}

// compute x^\sigma, p^\sigma
// this is currently done by calling two different kernels... this is likely
// not optimal
// uses detail_m::KERNEL_XP
template <typename Array1, typename Array2, typename Array3,
         typename Array4, typename Array5, typename Array6>
void compute_xp_m(const Array1& alpha_0_s, const Array2& z_1_s,
                  const Array3& beta_0_s, const Array4& r_0,
                  Array5& x_0_s, Array6& p_0_s)
{
    // sanity check
    cusp::assert_same_dimensions(alpha_0_s, z_1_s, beta_0_s);
    cusp::assert_same_dimensions(x_0_s, p_0_s);
    size_t N = r_0.end() - r_0.begin();
    size_t N_s = alpha_0_s.end() - alpha_0_s.begin();
    size_t N_t = x_0_s.end() - x_0_s.begin();
    assert (N_t == N*N_s);

    // counting iterators to pass to thrust::transform
    thrust::counting_iterator<int> counter(0);

    // get raw pointers for passing to kernels
    typedef typename Array1::value_type   ScalarType;
    const ScalarType *raw_ptr_alpha_0_s = thrust::raw_pointer_cast(&alpha_0_s[0]);
    const ScalarType *raw_ptr_z_1_s     = thrust::raw_pointer_cast(&z_1_s[0]);
    const ScalarType *raw_ptr_beta_0_s  = thrust::raw_pointer_cast(&beta_0_s[0]);
    const ScalarType *raw_ptr_r_0       = thrust::raw_pointer_cast(&r_0[0]);

    // compute new x,p
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(x_0_s.begin(), p_0_s.begin(),counter)),
        thrust::make_zip_iterator(thrust::make_tuple(x_0_s.begin(), p_0_s.begin(),counter))+N_t,
        cusp::krylov::cg_detail::detail_m::KERNEL_XP<ScalarType>(N, raw_ptr_alpha_0_s, raw_ptr_beta_0_s, raw_ptr_z_1_s, raw_ptr_r_0));
}

template <typename Array1, typename Array2, typename Array3>
void doublecopy(const Array1& s, Array2& sd, Array3& d)
{
    // sanity check
    cusp::assert_same_dimensions(s, sd, d);
    size_t N = s.end() - s.begin();

    // recycle
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(s.begin(), sd.begin(), d.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(s.begin(), sd.begin(), d.begin())) + N,
        cusp::krylov::cg_detail::detail_m::KERNEL_DCOPY());
}

// multiple copy of array to another array
// this is just a vectorization of blas::copy
// uses detail_m::KERNEL_VCOPY
template <typename Array1, typename Array2>
void vectorize_copy(const Array1& source, Array2& dest)
{
    // sanity check
    size_t N = source.end() - source.begin();
    size_t N_t = dest.end() - dest.begin();
    assert ( N_t%N == 0 );

    // counting iterators to pass to thrust::transform
    thrust::counting_iterator<int> counter(0);

    // pointer to data
    typedef typename Array1::value_type   ScalarType;
    const ScalarType *raw_ptr_source = thrust::raw_pointer_cast(source.data());

    // compute
    thrust::transform(counter, counter + N_t, dest.begin(),
                      cusp::krylov::cg_detail::detail_m::KERNEL_VCOPY<ScalarType>(N, raw_ptr_source));

}

template <typename ForwardIterator1,
         typename ForwardIterator2,
         typename ScalarType>
void xpay(ForwardIterator1 first1,
          ForwardIterator1 last1,
          ForwardIterator2 first2,
          ScalarType alpha)
{
    thrust::transform(first1, last1, first2, first2, detail_m::XPAY<ScalarType>(alpha));
}

template <typename Array1,
         typename Array2,
         typename ScalarType>
void xpay(const Array1& x,
          Array2& y,
          ScalarType alpha)
{
    cusp::assert_same_dimensions(x, y);
    cusp::krylov::cg_detail::trans_m::xpay(x.begin(), x.end(), y.begin(), alpha);
}

} // end namespace trans_m

// CG-M routine that takes a user specified monitor
template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3,
          typename Monitor>
void cg_m(thrust::execution_policy<DerivedPolicy> &exec,
          const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma,
                Monitor& monitor)
{
    //
    // This bit is initialization of the solver.
    //

    // shorthand for typenames
    typedef typename LinearOperator::value_type        ValueType;

    // sanity checking
    const size_t N = A.num_rows;
    const size_t N_t = x.end() - x.begin();
    const size_t test = b.end() - b.begin();
    const size_t N_s = sigma.end() - sigma.begin();

    assert(A.num_rows == A.num_cols);
    assert(N_t == N*N_s);
    assert(N == test);

    // p has data used in computing the soln.
    cusp::detail::temporary_array<ValueType, DerivedPolicy> p_0_s(exec, N_t);

    // stores residuals
    cusp::detail::temporary_array<ValueType, DerivedPolicy> r_0(exec, N);
    // used in iterates
    cusp::detail::temporary_array<ValueType, DerivedPolicy> p_0(exec, N);

    // stores parameters used in the iteration
    cusp::detail::temporary_array<ValueType, DerivedPolicy> z_m1_s(exec, N_s, ValueType(1));
    cusp::detail::temporary_array<ValueType, DerivedPolicy> z_0_s(exec, N_s, ValueType(1));
    cusp::detail::temporary_array<ValueType, DerivedPolicy> z_1_s(exec, N_s);

    cusp::detail::temporary_array<ValueType, DerivedPolicy> alpha_0_s(exec, N_s, ValueType(0));
    cusp::detail::temporary_array<ValueType, DerivedPolicy> beta_0_s(exec, N_s);

    // stores parameters used in the iteration for the undeformed system
    ValueType beta_m1, beta_0(ValueType(1));
    ValueType alpha_0(ValueType(0));
    //ValueType alpha_0_inv;

    // stores the value of the matrix-vector product we have to compute
    cusp::detail::temporary_array<ValueType, DerivedPolicy> Ap(exec, N);

    // stores the value of the inner product (p,Ap)
    ValueType pAp;

    // store the values of (r_i,r_i) and (r_{i+1},r_{i+1})
    ValueType rsq_0, rsq_1;

    // set up the initial conditions for the iteration
    cusp::blas::copy(exec, b, r_0);
    rsq_1 = cusp::blas::dotc(exec, r_0, r_0);

    // set up the intitial guess
    //  cusp::blas::fill(x.begin(),x.end(),ValueType(0));
    cusp::blas::fill(exec, x, ValueType(0));

    // set up initial value of p_0 and p_0^\sigma
    cusp::krylov::cg_detail::trans_m::vectorize_copy(b, p_0_s);
    cusp::blas::copy(exec, b, p_0);

    //
    // Initialization is done. Solve iteratively
    //
    while (!monitor.finished(exec, r_0))
    {
        // recycle iterates
        rsq_0 = rsq_1;
        beta_m1 = beta_0;

        // compute the matrix-vector product Ap
        cusp::multiply(exec, A, p_0, Ap);

        // compute the inner product (p,Ap)
        pAp = cusp::blas::dotc(exec, p_0, Ap);

        // compute \beta_0
        beta_0 = -rsq_0/pAp;

        // compute the new residual
        cusp::blas::axpy(exec, Ap, r_0, beta_0);

        // compute \zeta_1^\sigma, \beta_0^\sigma
        cusp::krylov::cg_detail::trans_m::compute_zb_m(z_0_s, z_m1_s, sigma, z_1_s, beta_0_s,
                                            beta_m1, beta_0, alpha_0);

        // compute \alpha_0
        rsq_1 = cusp::blas::dotc(exec, r_0, r_0);
        alpha_0 = rsq_1 / rsq_0;
        cusp::krylov::cg_detail::trans_m::xpay(r_0, p_0, alpha_0);

        // calculate \alpha_0^\sigma
        cusp::krylov::cg_detail::trans_m::compute_a_m(z_0_s, z_1_s, beta_0_s,
                                                      alpha_0_s, beta_0, alpha_0);

        // compute x_0^\sigma, p_0^\sigma
        cusp::krylov::cg_detail::trans_m::compute_xp_m(alpha_0_s, z_1_s, beta_0_s, r_0,
                                                       x, p_0_s);

        // recycle \zeta_i^\sigma
        cusp::krylov::cg_detail::trans_m::doublecopy(z_1_s, z_0_s, z_m1_s);

        ++monitor;

    }// finished iteration

} // end cg_m

} // end cg_detail namespace

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3,
          typename Monitor>
void cg_m(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma,
                Monitor& monitor)
{
    using cusp::krylov::cg_detail::cg_m;

    return cg_m(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, b, sigma, monitor);
}

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3,
          typename Monitor>
void cg_m(const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma,
                Monitor& monitor)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename VectorType1::memory_space    System2;
    typedef typename VectorType2::memory_space    System3;
    typedef typename VectorType3::memory_space    System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;

    return cusp::krylov::cg_m(select_system(system1,system2,system3,system4), A, x, b, sigma, monitor);
}

// CG-M routine that uses the default monitor to determine completion
template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3>
void cg_m(const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::monitor<ValueType> monitor(b);

    return cusp::krylov::cg_m(A, x, b, sigma, monitor);
}

} // end namespace krylov
} // end namespace cusp

