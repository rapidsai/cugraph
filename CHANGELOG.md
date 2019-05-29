# cuGraph 0.8.0 (Date TBD)

## New Features
- PR #287 SNMG power iteration step1 
- PR #297 SNMG degree calculation 
- PR #302 SNMG CSR Pagerank (cuda/C++)

## Improvements
- PR #291 nvGraph is updated to use RMM instead of directly invoking cnmem functions.
- PR #286 Reorganized cugraph source directory
- PR #306 Integrated nvgraph to libcugraph.so (libnvgraph_rapids.so will not be built anymore).
- PR #306 Updated python test files to run pytest with all four RMM configurations.


## Bug Fixes
- PR #283 Automerge fix
- PR #291 Fixed a RMM memory allocation failure due to duplicate copies of cnmem.o
- PR #291 Fixed a cub CsrMV call error when RMM pool allocator is used.
- PR #306 Fixed cmake warnings due to library conflicts.

# cuGraph 0.7.0 (Date TBD)

## New Features
- PR #195 Added Graph.get_two_hop_neighbors() method
- PR #195 Updated Jaccard and Weighted Jaccard to accept lists of vertex pairs to compute for
- PR #202 Added methods to compute the overlap coefficient and weighted overlap coefficient
- PR #230 SNMG SPMV and helpers functions
- PR #210 Expose degree calculation kernel via python API
- PR #220 Added bindings for Nvgraph triangle counting
- PR #234 Added bindings for renumbering, modify renumbering to use RMM
- PR #246 Added bindings for subgraph extraction
- PR #250 Add local build script to mimic gpuCI
- PR #261 Add docs build script to cuGraph
- PR #301 Added build.sh script, updated CI scripts and documentation

## Improvements
- PR #157 Removed cudatoolkit dependency in setup.py
- PR #185 Update docs version
- PR #194 Open source nvgraph in cugraph repository #194
- PR #190 Added a copy option in graph creation
- PR #196 Fix typos in readme intro
- PR #207 mtx2csv script
- PR #203 Added small datasets directly in the repo
- PR #215 Simplified get_rapids_dataset_root_dir(), set a default value for the root dir
- PR #233 Added csv datasets and edited test to use cudf for reading graphs
- PR #247 Added some documentation for renumbering
- PR #252 cpp test upgrades for more convenient testing on large input
- PR #264 Add cudatoolkit conda dependency
- PR #267 Use latest release version in update-version CI script
- PR #270 Updated the README.md and CONTRIBUTING.md files
- PR #281 Updated README with algorithm list


## Bug Fixes
- PR #256 Add pip to the install, clean up conda instructions
- PR #253 Add rmm to conda configuration
- PR #226 Bump cudf dependencies to 0.7
- PR #169 Disable terminal output in sssp
- PR #191 Fix double upload bug
- PR #181 Fixed crash/rmm free error when edge values provided
- PR #193 Fixed segfault when egde values not provided
- PR #190 Fixed a memory reference counting error between cudf & cugraph
- PR #190 Fixed a language level warning (cython)
- PR #214 Removed throw exception from dtor in TC
- PR #211 Remove hardcoded dataset paths, replace with build var that can be overridden with an env var
- PR #206 Updated versions in conda envs
- PR #218 Update c_graph.pyx
- PR #224 Update erroneous comments in overlap_wrapper.pyx, woverlap_wrapper.pyx, test_louvain.py, and spectral_clustering.pyx
- PR #220 Fixed bugs in Nvgraph triangle counting
- PR #232 Fixed memory leaks in managing cudf columns.
- PR #236 Fixed issue with v0.7 nightly yml environment file.  Also updated the README to remove pip
- PR #239 Added a check to prevent a cugraph object to store two different graphs.
- PR #244 Fixed issue with nvgraph's subgraph extraction if the first vertex in the vertex list is not incident on an edge in the extracted graph
- PR #249 Fix oudated cuDF version in gpu/build.shi
- PR #262 Removed networkx conda dependency for both build and runtime
- PR #271 Removed nvgraph conda dependency
- PR #276 Removed libgdf_cffi import from bindings
- PR #288 Add boost as a conda dependency

# cuGraph 0.6.0 (22 Mar 2019)

## New Features

- PR #73 Weighted Jaccard bindings
- PR #41 RMAT graph bindings
- PR #43 Louvain binings
- PR #44 SSSP bindings
- PR #47 BSF bindings
- PR #53 New Repo structure
- PR #67 RMM Integration with rmm as as submodule
- PR #82 Spectral Clustering bindings
- PR #82 Clustering metrics binding
- PR #85 Helper functions on python Graph object
- PR #106 Add gpu/build.sh file for gpuCI

## Improvements

- PR #50 Reorganize directory structure to match cuDF
- PR #85 Deleted setup.py and setup.cfg which had been replaced
- PR #95 Code clean up
- PR #96 Relocated mmio.c and mmio.h (external files) to thirdparty/mmio
- PR #97 Updated python tests to speed them up
- PR #100 Added testing for returned vertex and edge identifiers
- PR #105 Updated python code to follow PEP8 (fixed flake8 complaints)
- PR #121 Cleaned up READEME file
- PR #130 Update conda build recipes
- PR #144 Documentation for top level functions

## Bug Fixes

- PR #48 ABI Fixes
- PR #72 Bug fix for segfault issue getting transpose from adjacency list
- PR #105 Bug fix for memory leaks and python test failures
- PR #110 Bug fix for segfault calling Louvain with only edge list
- PR #115 Fixes for changes in cudf 0.6, pick up RMM from cudf instead of thirdpary
- PR #116 Added netscience.mtx dataset to datasets.tar.gz
- PR #120 Bug fix for segfault calling spectral clustering with only edge list
- PR #123 Fixed weighted Jaccard to assume the input weights are given as a cudf.Series
- PR #152 Fix conda package version string
- PR #160 Added additional link directory to support building on CentOS-7
- PR #221 Moved two_hop_neighbors.cuh to src folder to prevent it being installed
- PR #223 Fixed compiler warning in cpp/src/cugraph.cu
- PR #284 Commented out unit test code that fails due to a cudf bug


# cuGraph 0.5.0 (28 Jan 2019)
