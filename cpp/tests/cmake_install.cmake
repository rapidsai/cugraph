# Install script for directory: /work/cugraph/cpp/tests

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_GENERATORS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_GENERATORS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_GENERATORS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/GRAPH_GENERATORS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_GENERATORS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_GENERATORS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_GENERATORS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_GENERATORS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ERDOS_RENYI_GENERATOR_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ERDOS_RENYI_GENERATOR_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ERDOS_RENYI_GENERATOR_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/ERDOS_RENYI_GENERATOR_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ERDOS_RENYI_GENERATOR_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ERDOS_RENYI_GENERATOR_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ERDOS_RENYI_GENERATOR_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ERDOS_RENYI_GENERATOR_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BETWEENNESS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BETWEENNESS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BETWEENNESS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/LEGACY_BETWEENNESS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BETWEENNESS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BETWEENNESS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BETWEENNESS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BETWEENNESS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_EDGE_BETWEENNESS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_EDGE_BETWEENNESS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_EDGE_BETWEENNESS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/LEGACY_EDGE_BETWEENNESS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_EDGE_BETWEENNESS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_EDGE_BETWEENNESS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_EDGE_BETWEENNESS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_EDGE_BETWEENNESS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BFS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BFS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/LEGACY_BFS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BFS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BFS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_BFS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LOUVAIN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LOUVAIN_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LOUVAIN_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/LOUVAIN_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LOUVAIN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LOUVAIN_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LOUVAIN_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LOUVAIN_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEIDEN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEIDEN_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEIDEN_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/LEIDEN_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEIDEN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEIDEN_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEIDEN_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEIDEN_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ECG_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ECG_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ECG_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/ECG_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ECG_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ECG_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ECG_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/ECG_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BALANCED_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BALANCED_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BALANCED_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/BALANCED_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BALANCED_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BALANCED_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BALANCED_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BALANCED_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EGO_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EGO_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EGO_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/EGO_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EGO_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EGO_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EGO_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EGO_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_FA2_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_FA2_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_FA2_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/LEGACY_FA2_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_FA2_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_FA2_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_FA2_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/LEGACY_FA2_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CONNECT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CONNECT_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CONNECT_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CONNECT_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CONNECT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CONNECT_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CONNECT_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CONNECT_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SCC_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SCC_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SCC_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/SCC_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SCC_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SCC_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SCC_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SCC_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HUNGARIAN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HUNGARIAN_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HUNGARIAN_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/HUNGARIAN_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HUNGARIAN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HUNGARIAN_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HUNGARIAN_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HUNGARIAN_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MST_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MST_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MST_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/MST_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MST_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MST_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MST_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MST_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/STREAM_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/STREAM_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/STREAM_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/STREAM_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/STREAM_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/STREAM_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/STREAM_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/STREAM_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GENERATE_RMAT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GENERATE_RMAT_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GENERATE_RMAT_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/GENERATE_RMAT_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GENERATE_RMAT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GENERATE_RMAT_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GENERATE_RMAT_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GENERATE_RMAT_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_MASK_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_MASK_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_MASK_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/GRAPH_MASK_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_MASK_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_MASK_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_MASK_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/GRAPH_MASK_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SYMMETRIZE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SYMMETRIZE_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SYMMETRIZE_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/SYMMETRIZE_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SYMMETRIZE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SYMMETRIZE_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SYMMETRIZE_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SYMMETRIZE_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/TRANSPOSE_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_STORAGE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_STORAGE_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_STORAGE_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/TRANSPOSE_STORAGE_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_STORAGE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_STORAGE_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_STORAGE_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRANSPOSE_STORAGE_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEIGHT_SUM_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEIGHT_SUM_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEIGHT_SUM_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/WEIGHT_SUM_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEIGHT_SUM_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEIGHT_SUM_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEIGHT_SUM_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEIGHT_SUM_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/DEGREE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/DEGREE_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/DEGREE_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/DEGREE_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/DEGREE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/DEGREE_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/DEGREE_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/DEGREE_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COARSEN_GRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COARSEN_GRAPH_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COARSEN_GRAPH_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/COARSEN_GRAPH_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COARSEN_GRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COARSEN_GRAPH_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COARSEN_GRAPH_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/COARSEN_GRAPH_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/INDUCED_SUBGRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/INDUCED_SUBGRAPH_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/INDUCED_SUBGRAPH_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/INDUCED_SUBGRAPH_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/INDUCED_SUBGRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/INDUCED_SUBGRAPH_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/INDUCED_SUBGRAPH_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/INDUCED_SUBGRAPH_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BFS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BFS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/BFS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BFS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BFS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BFS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EXTRACT_BFS_PATHS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EXTRACT_BFS_PATHS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EXTRACT_BFS_PATHS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/EXTRACT_BFS_PATHS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EXTRACT_BFS_PATHS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EXTRACT_BFS_PATHS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EXTRACT_BFS_PATHS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EXTRACT_BFS_PATHS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MSBFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MSBFS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MSBFS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/MSBFS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MSBFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MSBFS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MSBFS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/MSBFS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SSSP_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SSSP_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SSSP_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/SSSP_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SSSP_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SSSP_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SSSP_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SSSP_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HITS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HITS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HITS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/HITS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HITS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HITS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HITS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/HITS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/PAGERANK_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/PAGERANK_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/PAGERANK_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/PAGERANK_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/PAGERANK_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/PAGERANK_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/PAGERANK_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/PAGERANK_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/KATZ_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/KATZ_CENTRALITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/KATZ_CENTRALITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/KATZ_CENTRALITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/KATZ_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/KATZ_CENTRALITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/KATZ_CENTRALITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/KATZ_CENTRALITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EIGENVECTOR_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EIGENVECTOR_CENTRALITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EIGENVECTOR_CENTRALITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/EIGENVECTOR_CENTRALITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EIGENVECTOR_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EIGENVECTOR_CENTRALITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EIGENVECTOR_CENTRALITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EIGENVECTOR_CENTRALITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BETWEENNESS_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BETWEENNESS_CENTRALITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BETWEENNESS_CENTRALITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/BETWEENNESS_CENTRALITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BETWEENNESS_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BETWEENNESS_CENTRALITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BETWEENNESS_CENTRALITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/BETWEENNESS_CENTRALITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EDGE_BETWEENNESS_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EDGE_BETWEENNESS_CENTRALITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EDGE_BETWEENNESS_CENTRALITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/EDGE_BETWEENNESS_CENTRALITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EDGE_BETWEENNESS_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EDGE_BETWEENNESS_CENTRALITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EDGE_BETWEENNESS_CENTRALITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/EDGE_BETWEENNESS_CENTRALITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEAKLY_CONNECTED_COMPONENTS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEAKLY_CONNECTED_COMPONENTS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEAKLY_CONNECTED_COMPONENTS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/WEAKLY_CONNECTED_COMPONENTS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEAKLY_CONNECTED_COMPONENTS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEAKLY_CONNECTED_COMPONENTS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEAKLY_CONNECTED_COMPONENTS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/WEAKLY_CONNECTED_COMPONENTS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SIMILARITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SIMILARITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SIMILARITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/SIMILARITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SIMILARITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SIMILARITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SIMILARITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/SIMILARITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RANDOM_WALKS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RANDOM_WALKS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RANDOM_WALKS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/RANDOM_WALKS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RANDOM_WALKS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RANDOM_WALKS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RANDOM_WALKS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RANDOM_WALKS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/UNIFORM_NEIGHBOR_SAMPLING_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/UNIFORM_NEIGHBOR_SAMPLING_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/UNIFORM_NEIGHBOR_SAMPLING_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/UNIFORM_NEIGHBOR_SAMPLING_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/UNIFORM_NEIGHBOR_SAMPLING_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/UNIFORM_NEIGHBOR_SAMPLING_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/UNIFORM_NEIGHBOR_SAMPLING_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/UNIFORM_NEIGHBOR_SAMPLING_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RENUMBERING_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RENUMBERING_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RENUMBERING_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/RENUMBERING_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RENUMBERING_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RENUMBERING_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RENUMBERING_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/RENUMBERING_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CORE_NUMBER_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CORE_NUMBER_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CORE_NUMBER_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CORE_NUMBER_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CORE_NUMBER_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CORE_NUMBER_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CORE_NUMBER_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/CORE_NUMBER_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_CORE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_CORE_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_CORE_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/K_CORE_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_CORE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_CORE_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_CORE_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_CORE_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRIANGLE_COUNT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRIANGLE_COUNT_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRIANGLE_COUNT_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/TRIANGLE_COUNT_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRIANGLE_COUNT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRIANGLE_COUNT_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRIANGLE_COUNT_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/TRIANGLE_COUNT_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_HOP_NBRS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_HOP_NBRS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_HOP_NBRS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/K_HOP_NBRS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_HOP_NBRS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_HOP_NBRS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_HOP_NBRS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph/K_HOP_NBRS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CREATE_GRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CREATE_GRAPH_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CREATE_GRAPH_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_CREATE_GRAPH_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CREATE_GRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CREATE_GRAPH_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CREATE_GRAPH_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CREATE_GRAPH_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_PAGERANK_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_PAGERANK_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_PAGERANK_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_PAGERANK_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_PAGERANK_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_PAGERANK_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_PAGERANK_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_PAGERANK_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_KATZ_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_KATZ_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_KATZ_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_KATZ_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_KATZ_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_KATZ_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_KATZ_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_KATZ_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EIGENVECTOR_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EIGENVECTOR_CENTRALITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EIGENVECTOR_CENTRALITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_EIGENVECTOR_CENTRALITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EIGENVECTOR_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EIGENVECTOR_CENTRALITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EIGENVECTOR_CENTRALITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EIGENVECTOR_CENTRALITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BETWEENNESS_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BETWEENNESS_CENTRALITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BETWEENNESS_CENTRALITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_BETWEENNESS_CENTRALITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BETWEENNESS_CENTRALITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BETWEENNESS_CENTRALITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BETWEENNESS_CENTRALITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BETWEENNESS_CENTRALITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_HITS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_HITS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_HITS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_HITS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_HITS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_HITS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_HITS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_HITS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BFS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BFS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_BFS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BFS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BFS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BFS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_BFS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SSSP_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SSSP_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SSSP_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_SSSP_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SSSP_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SSSP_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SSSP_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SSSP_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EXTRACT_PATHS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EXTRACT_PATHS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EXTRACT_PATHS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_EXTRACT_PATHS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EXTRACT_PATHS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EXTRACT_PATHS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EXTRACT_PATHS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EXTRACT_PATHS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_NODE2VEC_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_NODE2VEC_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_NODE2VEC_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_NODE2VEC_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_NODE2VEC_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_NODE2VEC_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_NODE2VEC_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_NODE2VEC_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_RANDOM_WALKS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_RANDOM_WALKS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_RANDOM_WALKS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_RANDOM_WALKS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_RANDOM_WALKS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_RANDOM_WALKS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_RANDOM_WALKS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_RANDOM_WALKS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TRIANGLE_COUNT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TRIANGLE_COUNT_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TRIANGLE_COUNT_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_TRIANGLE_COUNT_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TRIANGLE_COUNT_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TRIANGLE_COUNT_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TRIANGLE_COUNT_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TRIANGLE_COUNT_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_LOUVAIN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_LOUVAIN_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_LOUVAIN_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_LOUVAIN_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_LOUVAIN_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_LOUVAIN_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_LOUVAIN_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_LOUVAIN_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CORE_NUMBER_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CORE_NUMBER_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CORE_NUMBER_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_CORE_NUMBER_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CORE_NUMBER_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CORE_NUMBER_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CORE_NUMBER_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_CORE_NUMBER_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SIMILARITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SIMILARITY_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SIMILARITY_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_SIMILARITY_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SIMILARITY_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SIMILARITY_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SIMILARITY_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_SIMILARITY_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_K_CORE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_K_CORE_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_K_CORE_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_K_CORE_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_K_CORE_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_K_CORE_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_K_CORE_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_K_CORE_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_INDUCED_SUBGRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_INDUCED_SUBGRAPH_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_INDUCED_SUBGRAPH_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_INDUCED_SUBGRAPH_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_INDUCED_SUBGRAPH_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_INDUCED_SUBGRAPH_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_INDUCED_SUBGRAPH_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_INDUCED_SUBGRAPH_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EGONET_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EGONET_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EGONET_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_EGONET_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EGONET_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EGONET_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EGONET_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_EGONET_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TWO_HOP_NEIGHBORS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TWO_HOP_NEIGHBORS_TEST")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TWO_HOP_NEIGHBORS_TEST"
         RPATH "$ORIGIN/../../../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c" TYPE EXECUTABLE FILES "/work/cugraph/cpp/tests/CAPI_TWO_HOP_NEIGHBORS_TEST")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TWO_HOP_NEIGHBORS_TEST" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TWO_HOP_NEIGHBORS_TEST")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TWO_HOP_NEIGHBORS_TEST"
         OLD_RPATH "::::::::::::::::::::"
         NEW_RPATH "$ORIGIN/../../../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/conda/envs/rapids/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gtests/libcugraph_c/CAPI_TWO_HOP_NEIGHBORS_TEST")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/work/cugraph/cpp/tests/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
