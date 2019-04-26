#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
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
# Download and unpack googletest at configure time

set(GTEST_CMAKE_ARGS " -Dgtest_build_samples=ON"	
	                     " -DCMAKE_VERBOSE_MAKEFILE=ON")
	
if(CMAKE_CXX11_ABI)
    message(STATUS "GTEST: Enabling the GLIBCXX11 ABI")
else()
    message(STATUS "GTEST: Disabling the GLIBCXX11 ABI")
    list(APPEND GTEST_CMAKE_ARGS " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
    list(APPEND GTEST_CMAKE_ARGS " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
endif(CMAKE_CXX11_ABI)

configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/GoogleTest.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-download/CMakeLists.txt)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-download/
)

if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-download/
)

if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Locate the Google Test package.
# Requires that you build with:
#   -DGTEST_ROOT:PATH=/path/to/googletest_install_dir
set(GTEST_ROOT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-install/)
message(STATUS "GTEST_ROOT: " ${GTEST_ROOT})

link_directories(${GTEST_ROOT}/lib/)
# FIXME: lib64 also needs to be added for CentOS-7. This might be a code smell - investigate.
link_directories(${GTEST_ROOT}/lib64/)
