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

/*! \file lanczos_options.h
 *  \brief Lanczos options
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>

#include <limits>
#include <iostream>

namespace cusp
{
namespace eigen
{

typedef enum
{
    SA,
    LA,
    BE,
} SpectrumPart;

typedef enum
{
    None,
    Full,
} ReorthStrategy;

template<typename ValueType>
class lanczos_options
{
public:

    bool computeEigVecs;
    bool verbose;

    size_t minIter;
    size_t maxIter;
    size_t extraIter;
    size_t stride;

    ReorthStrategy reorth;
    SpectrumPart eigPart;

    ValueType memoryExpansionFactor;
    ValueType tol;
    ValueType doubleReorthGamma;
    ValueType localReorthGamma;

    size_t defaultMinIterFactor;
    size_t defaultMaxIterFactor;

    ValueType eigLowCut;
    ValueType eigHighCut;

    lanczos_options() :
        computeEigVecs(false), verbose(false), minIter(0), maxIter(0), extraIter(10),
        stride(10), reorth(None), eigPart(LA), memoryExpansionFactor(1.2), tol(1e-4),
        doubleReorthGamma(1.0/std::sqrt(2.0)), localReorthGamma(1.0/std::sqrt(2.0)),
        defaultMinIterFactor(5), defaultMaxIterFactor(50),
        eigLowCut(std::numeric_limits<ValueType>::infinity()),
        eigHighCut(-std::numeric_limits<ValueType>::infinity())
    {}

    lanczos_options(const lanczos_options<ValueType>& opts);

    void print(void);
};

} // end namespace eigen
} // end namespace cusp

#include <cusp/eigen/detail/lanczos_options.inl>
