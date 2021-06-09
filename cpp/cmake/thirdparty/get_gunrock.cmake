#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

function(find_and_configure_gunrock VERSION)

    if(NOT TARGET gunrock)
        set(GUNROCK_GENCODE_SM60 OFF)
        set(GUNROCK_GENCODE_SM61 OFF)
        set(GUNROCK_GENCODE_SM70 OFF)
        set(GUNROCK_GENCODE_SM72 OFF)
        set(GUNROCK_GENCODE_SM75 OFF)
        set(GUNROCK_GENCODE_SM80 OFF)

        foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REPLACE "-real" "" arch ${arch})
            set(GUNROCK_GENCODE_SM${arch} "ON")
        endforeach()

        # FIXME: gunrock is still using ExternalProject instead of CPM, as version 1.2
        # doesn't work with CPM

        include(ExternalProject)

        set(GUNROCK_DIR ${CMAKE_CURRENT_BINARY_DIR}/gunrock)
        ExternalProject_Add(gunrock_ext
          GIT_REPOSITORY    https://github.com/gunrock/gunrock.git
          GIT_TAG           v${VERSION}
          PREFIX            ${GUNROCK_DIR}
          CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                            -DGUNROCK_BUILD_SHARED_LIBS=OFF
                            -DGUNROCK_BUILD_TESTS=OFF
                            -DCUDA_AUTODETECT_GENCODE=OFF
                            -DGUNROCK_GENCODE_SM60=${GUNROCK_GENCODE_SM60}
                            -DGUNROCK_GENCODE_SM61=${GUNROCK_GENCODE_SM61}
                            -DGUNROCK_GENCODE_SM70=${GUNROCK_GENCODE_SM70}
                            -DGUNROCK_GENCODE_SM72=${GUNROCK_GENCODE_SM72}
                            -DGUNROCK_GENCODE_SM75=${GUNROCK_GENCODE_SM75}
                            -DGUNROCK_GENCODE_SM80=${GUNROCK_GENCODE_SM80}
          BUILD_BYPRODUCTS  ${GUNROCK_DIR}/src/gunrock_ext-build/lib/libgunrock.a
          INSTALL_COMMAND   ""
        )

        add_library(gunrock STATIC IMPORTED)
        add_dependencies(gunrock gunrock_ext)
        set_property(TARGET gunrock PROPERTY IMPORTED_LOCATION "${GUNROCK_DIR}/src/gunrock_ext-build/lib/libgunrock.a")
        target_include_directories(gunrock INTERFACE "${GUNROCK_DIR}/src/gunrock_ext")
    endif()
endfunction()


find_and_configure_gunrock(1.2)
