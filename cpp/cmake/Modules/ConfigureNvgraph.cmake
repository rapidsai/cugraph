set(NVGRAPH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph")

set(NVGRAPH_CMAKE_ARGS "")
                     #" -DNVGRAPH_build_samples=ON" 
                     #" -DCMAKE_VERBOSE_MAKEFILE=ON")

if(NOT CMAKE_CXX11_ABI)
    message(STATUS "NVGRAPH: Disabling the GLIBCXX11 ABI")
    list(APPEND NVGRAPH_CMAKE_ARGS " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
    list(APPEND NVGRAPH_CMAKE_ARGS " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
elseif(CMAKE_CXX11_ABI)
    message(STATUS "NVGRAPH: Enabling the GLIBCXX11 ABI")
    list(APPEND NVGRAPH_CMAKE_ARGS " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1")
    list(APPEND NVGRAPH_CMAKE_ARGS " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1")
endif(NOT CMAKE_CXX11_ABI)

#configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/Nvgraph.CMakeLists.txt.cmake"
#               "${NVGRAPH_ROOT}/cpp/CMakeLists.txt")

file(MAKE_DIRECTORY "${NVGRAPH_ROOT}/cpp/build")
#file(MAKE_DIRECTORY "${NVGRAPH_ROOT}/install")

execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .. -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DNVGRAPH_LIGHT=${NVGRAPH_LIGHT}
                RESULT_VARIABLE NVGRAPH_CONFIG
                WORKING_DIRECTORY ${NVGRAPH_ROOT}/cpp/build)

if(NVGRAPH_CONFIG)
    message(FATAL_ERROR "Configuring nvgraph failed: " ${NVGRAPH_CONFIG})
endif(NVGRAPH_CONFIG)

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "NVGRAPH BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "NVGRAPH BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "NVGRAPH BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

execute_process(COMMAND ${CMAKE_COMMAND} --build . -- ${PARALLEL_BUILD}
                RESULT_VARIABLE NVGRAPH_BUILD
                WORKING_DIRECTORY ${NVGRAPH_ROOT}/cpp/build)
if(NVGRAPH_BUILD)
    message(FATAL_ERROR "Building nvgraph failed: " ${NVGRAPH_BUILD})
endif(NVGRAPH_BUILD)

execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install 
                RESULT_VARIABLE NVGRAPH_BUILD
                WORKING_DIRECTORY ${NVGRAPH_ROOT}/cpp/build)

if(NVGRAPH_BUILD)
    message(FATAL_ERROR "Installing nvgraph failed: " ${NVGRAPH_BUILD})
endif(NVGRAPH_BUILD)

message(STATUS "nvgraph installed under: " ${CMAKE_INSTALL_PREFIX})
set(NVGRAPH_INCLUDE "${CMAKE_INSTALL_PREFIX}/include/nvgraph.h ${CMAKE_INSTALL_PREFIX}/include/test_opt_utils.cuh")
set(NVGRAPH_LIBRARY "${CMAKE_INSTALL_PREFIX}/lib/libnvgraph_rapids.so")
set(NVGRAPH_FOUND TRUE)
