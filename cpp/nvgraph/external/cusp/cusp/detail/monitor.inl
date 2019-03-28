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

#include <cusp/blas/blas.h>

#include <limits>
#include <iostream>
#include <iomanip>

namespace cusp
{

template <typename ValueType>
template <typename VectorType>
monitor<ValueType>
::monitor(const VectorType& b, size_t iteration_limit, Real relative_tolerance, Real absolute_tolerance, bool verbose)
    : b_norm(cusp::blas::nrm2(b)),
      r_norm(std::numeric_limits<Real>::max()),
      iteration_limit_(iteration_limit),
      iteration_count_(0),
      relative_tolerance_(relative_tolerance),
      absolute_tolerance_(absolute_tolerance),
      verbose(verbose)
{
    if(verbose)
    {
        std::cout << "Solver will continue until ";
        std::cout << "residual norm " << relative_tolerance << " or reaching ";
        std::cout << iteration_limit << " iterations " << std::endl;
        std::cout << "  Iteration Number  | Residual Norm" << std::endl;
    }

    residuals.reserve(iteration_limit);
}

template <typename ValueType>
void
monitor<ValueType>
::operator++(void)
{
    ++iteration_count_;
}

template <typename ValueType>
bool
monitor<ValueType>
::converged(void) const
{
    return residual_norm() <= tolerance();
}

template <typename ValueType>
typename monitor<ValueType>::Real
monitor<ValueType>
::residual_norm(void) const
{
    return r_norm;
}

template <typename ValueType>
size_t
monitor<ValueType>
::iteration_count(void) const
{
    return iteration_count_;
}

template <typename ValueType>
size_t
monitor<ValueType>
::iteration_limit(void) const
{
    return iteration_limit_;
}

template <typename ValueType>
typename monitor<ValueType>::Real
monitor<ValueType>
::relative_tolerance(void) const
{
    return relative_tolerance_;
}

template <typename ValueType>
typename monitor<ValueType>::Real
monitor<ValueType>
::absolute_tolerance(void) const
{
    return absolute_tolerance_;
}

template <typename ValueType>
typename monitor<ValueType>::Real
monitor<ValueType>
::tolerance(void) const
{
    return absolute_tolerance() + relative_tolerance() * b_norm;
}

template <typename ValueType>
void
monitor<ValueType>
::set_verbose(bool verbose_)
{
    verbose = verbose_;
}

template <typename ValueType>
bool
monitor<ValueType>
::is_verbose(void)
{
    return verbose;
}

template <typename ValueType>
template <typename Vector>
void
monitor<ValueType>
::reset(const Vector& b)
{
    b_norm = cusp::blas::nrm2(b);
    r_norm = std::numeric_limits<Real>::max();
    iteration_count_ = 0;
    residuals.resize(0);
}

template <typename ValueType>
void
monitor<ValueType>
::print(void)
{
    if(iteration_count() == 0)
    {
        std::cout << "Monitor configured with " << tolerance() << " tolerance ";
        std::cout << "and iteration limit " << iteration_limit() << std::endl;
        return;
    }

    // report solver results
    if (converged())
    {
        std::cout << "Solver converged to " << tolerance() << " tolerance";
    }
    else if(iteration_count() >= iteration_limit())
    {
        std::cout << "Solver reached iteration limit " << iteration_limit() << " before converging";
    }
    else
    {
        throw cusp::runtime_exception("Monitor is in inconsistent state.");
    }

    std::cout << " to (" << residual_norm() << " final residual)" << std::endl;

    std::cout << "Ran " << iteration_count();
    std::cout << " iterations with a final residual of ";
    std::cout << r_norm << std::endl;

    std::cout << "geometric convergence factor : " << geometric_rate() << std::endl;
    std::cout << "immediate convergence factor : " << immediate_rate() << std::endl;
    std::cout << "average convergence factor   : " << average_rate() << std::endl;
}

template <typename ValueType>
template <typename DerivedPolicy, typename Vector>
bool monitor<ValueType>
::finished(thrust::execution_policy<DerivedPolicy> &exec,
           const Vector& r)
{
    r_norm = cusp::blas::nrm2(exec, r);
    residuals.push_back(r_norm);

    if(verbose)
    {
        std::cout << "       "  << std::setw(10) << iteration_count();
        std::cout << "       "  << std::setw(10) << std::scientific << residual_norm() << std::endl;
    }

    if (converged())
    {
        if(verbose) std::cout << "Successfully converged after " << iteration_count() << " iterations." << std::endl;
        return true;
    }
    else if (iteration_count() >= iteration_limit())
    {
        if(verbose) std::cout << "Failed to converge after " << iteration_count() << " iterations." << std::endl;
        return true;
    }
    else
    {
        return false;
    }
}

template <typename ValueType>
template <typename Vector>
bool monitor<ValueType>
::finished(const Vector& r)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Vector::memory_space System;

    System system;

    return finished(select_system(system), r);
}

template <typename ValueType>
typename monitor<ValueType>::Real
monitor<ValueType>
::immediate_rate(void)
{
    size_t num = residuals.size();
    return residuals[num-1] / residuals[num-2];
}

template <typename ValueType>
typename monitor<ValueType>::Real
monitor<ValueType>
::geometric_rate(void)
{
    size_t num = residuals.size();
    return std::pow(residuals[num-1] / residuals[0], Real(1.0)/num);
}

template <typename ValueType>
typename monitor<ValueType>::Real
monitor<ValueType>
::average_rate(void)
{
    size_t num = residuals.size();
    cusp::array1d<Real,cusp::host_memory> avg_vec(num-1);
    thrust::transform(residuals.begin() + 1, residuals.end(), residuals.begin(), avg_vec.begin(), thrust::divides<Real>());
    Real sum = thrust::reduce(avg_vec.begin(), avg_vec.end(), Real(0), thrust::plus<Real>());
    return sum / Real(avg_vec.size());
}
} // end namespace cusp

