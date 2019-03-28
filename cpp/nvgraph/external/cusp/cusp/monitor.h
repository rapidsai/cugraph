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

/*! \file monitor.h
 *  \brief Monitor iterative solver convergence
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/blas/blas.h>
#include <cusp/complex.h>

#include <limits>
#include <iostream>
#include <iomanip>

namespace cusp
{
/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup monitors Monitors
 *  \ingroup iterative_solvers
 *  \brief Configurable convergence monitors for iterative solvers
 *  \{
 */

/**
 * \brief Implements standard convergence criteria and reporting for iterative solvers.
 *
 * \tparam ValueType scalar type used in the solver (e.g. \c float or \c cusp::complex<double>).
 *
 * \par Overview
 *  The \p monitor terminates iteration when the residual norm
 *  satisfies the condition
 *       ||b - A x|| <= absolute_tolerance + relative_tolerance * ||b||
 *  or when the iteration limit is reached.
 *  Classes to monitor iterative solver progress, check for convergence, etc.
 *  Follows the implementation of Iteration in the ITL:
 *  \see http://www.osl.iu.edu/research/itl/doc/Iteration.html
 *
 * \par Example
 *  The following code snippet demonstrates how to configure
 *  the \p monitor and use it with an iterative solver.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/krylov/cg.h>
 *  #include <cusp/gallery/poisson.h>
 *
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, float, cusp::device_memory> A;
 *
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 10, 10);
 *
 *      // allocate storage for solution (x) and right hand side (b)
 *      cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *      cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *      // set stopping criteria:
 *      //  iteration_limit    = 100
 *      //  relative_tolerance = 1e-6
 *      cusp::monitor<float> monitor(b, 100, 1e-6);
 *
 *      // solve the linear system A x = b
 *      cusp::krylov::cg(A, x, b, monitor);
 *
 *      // report solver results
 *      if (monitor.converged())
 *      {
 *          std::cout << "Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
 *          std::cout << " after " << monitor.iteration_count() << " iterations" << std::endl;
 *      }
 *      else
 *      {
 *          std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
 *          std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl;
 *      }
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename ValueType>
class monitor
{
public:
    typedef typename cusp::norm_type<ValueType>::type Real;

    /**
     *  \brief Constructs a \p monitor for a given right-hand-side \p b
     *
     *  \tparam VectorType Type of initial vector
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *  \param verbose Controls printing status updates during execution
     */
    template <typename VectorType>
    monitor(const VectorType& b,
            const size_t iteration_limit = 500,
            const Real relative_tolerance = 1e-5,
            const Real absolute_tolerance = 0,
            const bool verbose = false);

    /**
     * \brief Increments the iteration count
     */
    void operator++(void);

    /**
     * \brief Indicates whether the last tested residual satisfies the convergence tolerance
     *
     * \return Boolean convergence indicator
     */
    bool converged(void) const;

    /**
     * \brief Euclidean norm of last residual
     *
     * \return The residual norm
     */
    Real residual_norm(void) const;

    /**
     * \brief Returns the number of iterations that the monitor has executed
     *
     * \return Number of iterations
     */
    size_t iteration_count(void) const;

    /**
     * \brief Returns the maximum number of iterations
     *
     * \return Maximum number of allowed iterations
     */
    size_t iteration_limit(void) const;

    /**
     * \brief Returns the relative tolerance
     *
     * \return absolute_tolerance set for monitor
     */
    Real relative_tolerance(void) const;

    /**
     * \brief Returns the absolute tolerance
     *
     * \return absolute_tolerance set for monitor
     */
    Real absolute_tolerance(void) const;

    /**
     *  \brief Return the tolerance equal to absolute_tolerance() + relative_tolerance() * ||b||
     *
     * \return tolerance set for monitor
     */
    Real tolerance(void) const;

    /**
     *  \brief Applies convergence criteria to determine whether iteration is finished
     *
     *  \tparam Vector vector
     *  \param r residual vector of the linear system (r = b - A x)
     */
    template <typename Vector>
    bool finished(const Vector& r);

    template <typename DerivedPolicy, typename Vector>
    bool finished(thrust::execution_policy<DerivedPolicy> &exec, const Vector& r);

    /**
     *  \brief Sets the verbosity level of the monitor
     *
     *  \param verbose_ If \c true print convergence messages during
     *  iterations.
     */
    void set_verbose(bool verbose_ = true);

    /**
     *  \brief Gets the verbosity level of the monitor
     *
     *  \return verbosity of this monitor.
     */
    bool is_verbose(void);

    /**
     *  \brief Resets the monitor using same convergence criteria
     *
     *  \tparam Vector vector
     *  \param b initial right hand side
     */
    template <typename Vector>
    void reset(const Vector& b);

    /**
     *  \brief Prints the number of iterations and convergence history information.
     */
    void print(void);

    /**
     *  \brief Returns the immedidate convergence rate.
     *
     *  \return immediate convergence rate.
     */
    Real immediate_rate(void);

    /**
     *  \brief Returns the geometric convergence rate.
     *
     *  \return geometric convergence rate.
     */
    Real geometric_rate(void);

    /**
     *  \brief Returns the average convergence rate.
     *
     *  \return average convergence rate.
     */
    Real average_rate(void);

    /*
     * Array holding the residuals per iteration
     */
    cusp::array1d<Real,cusp::host_memory> residuals;

private:

    /*! \cond */
    Real b_norm;
    Real r_norm;
    size_t iteration_limit_;
    size_t iteration_count_;
    Real relative_tolerance_;
    Real absolute_tolerance_;
    bool verbose;
    /*! \endcond */
};

// TODO : Remove convergence, default, and verbose monitors in v0.6.0
template <typename ValueType>
class convergence_monitor : public monitor<ValueType>
{
private:

    typedef monitor<ValueType> Parent;

public:

    typedef typename Parent::Real Real;

    /**
     *  \brief Constructs a \p convergence_monitor for a given right-hand-side \p b
     *
     *  \tparam VectorType Type of initial vector
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *  \param verbose Controls printing status updates during execution
     *
     *  \deprecated As of v0.4.0 monitors have been unified. Use monitor
     *  instead.
     */
    template <typename VectorType>
    CUSP_DEPRECATED
    convergence_monitor(const VectorType& b,
                        const size_t iteration_limit = 500,
                        const Real relative_tolerance = 1e-5,
                        const Real absolute_tolerance = 0)
    : Parent(b, iteration_limit, relative_tolerance, absolute_tolerance, false) {}
};

template <typename ValueType>
class default_monitor : public monitor<ValueType>
{
private:

    typedef monitor<ValueType> Parent;

public:

    typedef typename Parent::Real Real;

    /**
     *  \brief Constructs a \p default_monitor for a given right-hand-side \p b
     *
     *  \tparam VectorType Type of initial vector
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *  \param verbose Controls printing status updates during execution
     *
     *  \deprecated As of v0.4.0 monitors have been unified. Use monitor
     *  instead.
     */
    template <typename VectorType>
    CUSP_DEPRECATED
    default_monitor(const VectorType& b,
                    const size_t iteration_limit = 500,
                    const Real relative_tolerance = 1e-5,
                    const Real absolute_tolerance = 0)
    : Parent(b, iteration_limit, relative_tolerance, absolute_tolerance, false) {}
};

template <typename ValueType>
class verbose_monitor : public monitor<ValueType>
{
private:

    typedef monitor<ValueType> Parent;

public:

    typedef typename Parent::Real Real;

    /**
     *  \brief Constructs a \p verbose_monitor for a given right-hand-side \p b
     *
     *  \tparam VectorType Type of initial vector
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *  \param verbose Controls printing status updates during execution
     *
     *  \deprecated As of v0.4.0 monitors have been unified. Use monitor
     *  instead.
     */
    template <typename VectorType>
    CUSP_DEPRECATED
    verbose_monitor(const VectorType& b,
                    const size_t iteration_limit = 500,
                    const Real relative_tolerance = 1e-5,
                    const Real absolute_tolerance = 0)
    : Parent(b, iteration_limit, relative_tolerance, absolute_tolerance, true) {}
};

} // end namespace cusp

#include <cusp/detail/monitor.inl>
