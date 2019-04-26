/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "nvgraph_error.hxx"

#define CHECK_STATUS(...)                                                      \
    do {                                                                       \
        if (__VA_ARGS__) {                                                     \
            FatalError(#__VA_ARGS__, NVGRAPH_ERR_UNKNOWN);                        \
        }                                                                      \
    } while (0)

#define CHECK_NVGRAPH(...)                                                        \
    do {                                                                       \
        NVGRAPH_ERROR e = __VA_ARGS__;                                            \
        if (e != NVGRAPH_OK) {                                                    \
            FatalError(#__VA_ARGS__, e)                                        \
        }                                                                      \
    } while (0)

#ifdef DEBUG
#define COUT() (std::cout)
#define CERR() (std::cerr)
#define WARNING(message)                                                       \
    do {                                                                       \
        std::stringstream ss;                                                  \
        ss << "Warning (" << __FILE__ << ":" << __LINE__ << "): " << message;  \
        CERR() << ss.str() << std::endl;                                       \
    } while (0)
#else // DEBUG
#define WARNING(message)
#endif
