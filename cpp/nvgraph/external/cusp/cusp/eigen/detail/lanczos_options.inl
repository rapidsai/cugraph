/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <limits>
#include <iostream>

#include <cusp/eigen/lanczos_options.h>

namespace cusp
{
namespace eigen
{

template<typename ValueType>
lanczos_options<ValueType>
::lanczos_options(const lanczos_options<ValueType>& opts)
{
    computeEigVecs        = opts.computeEigVecs;
    verbose               = opts.verbose;

    minIter               = opts.minIter;
    maxIter               = opts.maxIter;
    extraIter             = opts.extraIter;
    stride                = opts.stride;
    defaultMinIterFactor  = opts.defaultMinIterFactor;
    defaultMaxIterFactor  = opts.defaultMaxIterFactor;

    eigLowCut             = opts.eigLowCut;
    eigHighCut            = opts.eigHighCut;
    memoryExpansionFactor = opts.memoryExpansionFactor;
    doubleReorthGamma     = opts.doubleReorthGamma;
    localReorthGamma      = opts.localReorthGamma;
    tol                   = opts.tol;

    eigPart               = opts.eigPart;
    reorth                = opts.reorth;
}

template<typename ValueType>
void
lanczos_options<ValueType>
::print(void)
{
    std::string spectrumNames[3];
    spectrumNames[SA] = "Smallest";
    spectrumNames[LA] = "Largest";
    spectrumNames[BE] = "Both Ends";

    std::string reorthNames[2];
    reorthNames[Full]    = "Full";
    reorthNames[None]    = "None";

    std::cout << "Lanczos( target range : " << spectrumNames[eigPart]
              << ", reorthogonalization strategy : " << reorthNames[reorth]
              << " )" <<std::endl;
    std::cout << "\tMinimum Iterations      : " << minIter << std::endl;
    std::cout << "\tMaximum Iterations      : " << maxIter << std::endl;
    std::cout << "\tExtra Iterations        : " << extraIter << std::endl;
    std::cout << "\tMemory Expansion Factor : " << memoryExpansionFactor << std::endl;
    std::cout << "\tDefault MinIter Factor  : " << defaultMinIterFactor << std::endl;
    std::cout << "\tDefault MaxIter Factor  : " << defaultMaxIterFactor << std::endl;

    std::cout << "\tLow Eigenvalue Cut      : " << eigLowCut  << std::endl;
    std::cout << "\tHigh Eigenvalue Cut     : " << eigHighCut << std::endl;
    std::cout << "\tConvergence Stride      : " << stride  << std::endl;
    std::cout << "\tConvergence Tolerance   : " << tol << std::endl;
    std::cout << "\tdouble reorthogonalization gamma : " << doubleReorthGamma << std::endl;
    std::cout << "\tlocal reorthogonalization gamma  : " << localReorthGamma << std::endl;
    std::cout << std::endl;
}

} // end namespace eigen
} // end namespace cusp
