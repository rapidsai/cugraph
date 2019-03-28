/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <cusp/detail/format.h>

#include <cusp/functional.h>

#include <cusp/system/detail/generic/multiply/generalized_spmv.h>
#include <cusp/system/detail/generic/multiply/generalized_spgemm.h>
#include <cusp/system/detail/generic/multiply/permute.h>
#include <cusp/system/detail/generic/multiply/spgemm.h>
#include <cusp/system/detail/generic/multiply/spmv.h>

#include <thrust/functional.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

using namespace thrust::detail;
//__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_operator_exec_impl, operator())

__THRUST_DEFINE_HAS_MEMBER_FUNCTION(has_member_operator_exec_imploperator, operator())

struct yes {};
struct no { yes m[2]; };

template <typename T, typename Signature>
struct has_member_operator_exec_impl
{
private:
	//struct yes {};
	//struct no { yes m[2]; };
	struct derived : public T
	{
		using T::operator();
		no operator()(...) const;
	};

	typedef typename thrust::detail::is_call_possible_detail::clone_constness<T, derived>::type derived_type;

	template<typename U, typename Result>
	struct return_value_check
	{
		static yes deduce(Result);
		static no deduce(...);
		static no deduce(no);
		static no deduce(thrust::detail::is_call_possible_detail::void_exp_result<T>);
	};

	template<typename U>
	struct return_value_check<U, void>
	{
		static yes deduce(...);
		static no deduce(no);
	};

	template<bool has_the_member_of_interest, typename F>
	struct impl
	{
		static const bool value = false;
	};

	template<typename Result, typename Arg>
	struct impl<true, Result(Arg)>
	{
		static typename add_reference<derived_type>::type test_me;
		static typename add_reference<Arg>::type          arg;

		static const bool value =
			sizeof(
				return_value_check<T, Result>::deduce(
				(test_me.operator()(arg), thrust::detail::is_call_possible_detail::void_exp_result<T>())
				)
				) == sizeof(yes);
	};

	template<typename Result, typename Arg1, typename Arg2>
	struct impl<true, Result(Arg1, Arg2)>
	{
		static typename add_reference<derived_type>::type test_me;
		static typename add_reference<Arg1>::type         arg1;
		static typename add_reference<Arg2>::type         arg2;

		static const bool value =
			sizeof(
				return_value_check<T, Result>::deduce(
				(test_me.operator()(arg1, arg2), thrust::detail::is_call_possible_detail::void_exp_result<T>())
				)
				) == sizeof(yes);
	};

	template<typename Result, typename Arg1, typename Arg2, typename Arg3>
	struct impl<true, Result(Arg1, Arg2, Arg3)>
	{
		static typename add_reference<derived_type>::type test_me;
		static typename add_reference<Arg1>::type         arg1;
		static typename add_reference<Arg2>::type         arg2;
		static typename add_reference<Arg3>::type         arg3;

		static const bool value =
			sizeof(
				return_value_check<T, Result>::deduce(
				(test_me.operator()(arg1, arg2, arg3), thrust::detail::is_call_possible_detail::void_exp_result<T>())
				)
				) == sizeof(yes);

	};

	template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
	struct impl<true, Result(Arg1, Arg2, Arg3, Arg4)>
	{
		static typename add_reference<derived_type>::type test_me;
		static typename add_reference<Arg1>::type         arg1;
		static typename add_reference<Arg2>::type         arg2;
		static typename add_reference<Arg3>::type         arg3;
		static typename add_reference<Arg4>::type         arg4;

		static const bool value =
			sizeof(
				return_value_check<T, Result>::deduce(
				(test_me.operator()(arg1, arg2, arg3, arg4), thrust::detail::is_call_possible_detail::void_exp_result<T>())
				)
				) == sizeof(yes);
	};

public:
	static const bool value = impl<has_member_operator_exec_imploperator<T, Signature>::value, Signature>::value;
	typedef thrust::detail::integral_constant<bool, value> type;
};

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
struct has_member_operator_exec
: has_member_operator_exec_impl<LinearOperator, void(thrust::execution_policy<DerivedPolicy>&,const MatrixOrVector1&,MatrixOrVector2&)>
{};

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
typename enable_if<
thrust::detail::and_<
  has_member_operator_exec<DerivedPolicy,LinearOperator,MatrixOrVector1,MatrixOrVector2>,
  thrust::detail::is_convertible<typename LinearOperator::format,cusp::unknown_format>
  >::value
>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
               MatrixOrVector2& C)
{
    const_cast<LinearOperator&>(A)(exec, B, C);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
typename enable_if<
thrust::detail::and_<
  thrust::detail::not_<has_member_operator_exec<DerivedPolicy,LinearOperator,MatrixOrVector1,MatrixOrVector2> >,
  thrust::detail::is_convertible<typename LinearOperator::format,cusp::unknown_format>
  >::value
>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
               MatrixOrVector2& C)
{
    const_cast<LinearOperator&>(A)(B, C);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
typename thrust::detail::disable_if_convertible<typename LinearOperator::format,cusp::unknown_format>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
               MatrixOrVector2& C)
{
    typedef typename LinearOperator::value_type ValueType;

    cusp::constant_functor<ValueType> initialize(0);
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType> reduce;

    cusp::multiply(exec, A, B, C, initialize, combine, reduce);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
typename thrust::detail::disable_if_convertible<UnaryFunction,cusp::known_format>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
         MatrixOrVector2& C,
         UnaryFunction   initialize,
         BinaryFunction1 combine,
         BinaryFunction2 reduce)
{
    typedef typename LinearOperator::format  Format1;
    typedef typename MatrixOrVector1::format Format2;
    typedef typename MatrixOrVector2::format Format3;

    Format1 format1;
    Format2 format2;
    Format3 format3;

    multiply(thrust::detail::derived_cast(exec), A, B, C, initialize, combine, reduce, format1, format2, format3);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void generalized_spgemm(thrust::execution_policy<DerivedPolicy> &exec,
                        const LinearOperator&  A,
                        const MatrixOrVector1& B,
                        MatrixOrVector2& C,
                        UnaryFunction   initialize,
                        BinaryFunction1 combine,
                        BinaryFunction2 reduce)
{
    typedef typename LinearOperator::format  Format1;
    typedef typename MatrixOrVector1::format Format2;
    typedef typename MatrixOrVector2::format Format3;

    Format1 format1;
    Format2 format2;
    Format3 format3;

    generalized_spgemm(exec, A, B, C, initialize, combine, reduce, format1, format2, format3);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename Vector1,
         typename Vector2,
         typename Vector3,
         typename BinaryFunction1,
         typename BinaryFunction2>
void generalized_spmv(thrust::execution_policy<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce)
{
    typedef typename LinearOperator::format  Format1;
    typedef typename Vector1::format         Format2;
    typedef typename Vector2::format         Format3;
    typedef typename Vector3::format         Format4;

    Format1 format1;
    Format2 format2;
    Format3 format3;
    Format4 format4;

    generalized_spmv(exec, A, x, y, z, combine, reduce, format1, format2, format3, format4);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

