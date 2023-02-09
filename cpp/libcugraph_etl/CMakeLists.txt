#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
include(../../fetch_rapids.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUGRAPH_ETL)

project(CUGRAPH_ETL VERSION 23.02.00 LANGUAGES C CXX CUDA)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA compiler version must be at least 11.0")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()

# Write the version header
rapids_cmake_write_version_file(include/cugraph_etl/version_config.hpp)

################################################################################
# - build type -----------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

################################################################################
# - User Options  --------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuGraph shared libraries" ON)
option(BUILD_CUGRAPH_ETL_MG_TESTS "Build cuGraph multigpu algorithm tests" OFF)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)

################################################################################
# - compiler options -----------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_RUNTIME)
  set(_ctk_static_suffix "_static")
endif()

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET    cugraph_etl-exports
    INSTALL_EXPORT_SET  cugraph_etl-exports
    )

set(CUGRAPH_ETL_CXX_FLAGS "")
set(CUGRAPH_ETL_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_ETL_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND CUGRAPH_ETL_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_ETL_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
list(APPEND CUGRAPH_ETL_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND CUGRAPH_ETL_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND CUGRAPH_ETL_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND CUGRAPH_ETL_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()

################################################################################
# - find CPM based dependencies  -----------------------------------------------

rapids_cpm_init()

include(cmake/thirdparty/get_cugraph.cmake)
include(cmake/thirdparty/get_cudf.cmake)

################################################################################
# - ETL library --------------------------------------------------------------

add_library(cugraph_etl
            src/renumbering.cu
           )
add_library(cugraph::cugraph_etl ALIAS cugraph_etl)

set_target_properties(cugraph_etl
               PROPERTIES BUILD_RPATH              "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cugraph_etl
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_ETL_CXX_FLAGS}>"
)

target_compile_options(cugraph_etl
            PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_ETL_CUDA_FLAGS}>")
################################################################################
# - ETL include paths --------------------------------------------------------
target_include_directories(cugraph_etl
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

################################################################################
# - ETL link libraries -------------------------------------------------------
target_link_libraries(cugraph_etl
        PUBLIC
                CUDA::cublas${_ctk_static_suffix}
                CUDA::curand${_ctk_static_suffix}
                CUDA::cusolver${_ctk_static_suffix}
                CUDA::cusparse${_ctk_static_suffix}
        PRIVATE
                cugraph::cugraph
                cudf::cudf
)

################################################################################
# - install targets ------------------------------------------------------------
rapids_cmake_install_lib_dir( lib_dir )
include(CPack)

install(TARGETS cugraph_etl
        DESTINATION ${lib_dir}
        EXPORT cugraph_etl-exports
)

install(DIRECTORY include/cugraph_etl/
        DESTINATION include/cugraph_etl)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cugraph_etl/version_config.hpp
        DESTINATION include/cugraph_etl)

################################################################################
# - install export -------------------------------------------------------------
set(doc_string
[=[

cuGraph_etl library is a collection of GPU accelerated ETL functions specifically for
cuGraph.

]=])

rapids_export(INSTALL cugraph_etl
    EXPORT_SET cugraph_etl-exports
    GLOBAL_TARGETS cugraph_etl
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    )

################################################################################
# - build export ---------------------------------------------------------------
rapids_export(BUILD cugraph
    EXPORT_SET cugraph_etl-exports
    GLOBAL_TARGETS cugraph cugraph_c cugraph_etl
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    )
