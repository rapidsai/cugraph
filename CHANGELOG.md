# cuGraph 0.14.0 (Date TBD)

## New Features
- PR #822 Added new functions in python graph class, similar to networkx

## Improvements
- PR #764 Updated sssp and bfs with GraphCSR, removed gdf_column, added nullptr weights test for sssp
- PR #765 Remove gdf_column from connected components
- PR #780 Remove gdf_column from cuhornet features
- PR #781 Fix compiler argument syntax for ccache
- PR #782 Use Cython's `new_build_ext` (if available)
- PR #788 Added options and config file to enable codecov
- PR #793 Fix legacy cudf imports/cimports
- PR #802 Removed use of gdf_column from db code
- PR #798 Edit return graph type in algorithms return graphs
- PR #803 Enable Ninja build
- PR #804 Cythonize in parallel
- PR #807 Updating the Python docs
- PR #820 OPG infra and all-gather smoke test
- PR #829 Updated README and CONTRIBUTIOIN docs
- PR #836 Remove SNMG code
- PR #831 Updated Notebook - Added K-Truss, ECG, and Betweenness Centrality
- PR #832 Removed RMM ALLOC from db subtree
- PR #833 Update graph functions to use new Graph class
- PR #834 Updated local gpuci build
- PR #845 Add .clang-format & format all files

## Bug Fixes
- PR #763 Update RAPIDS conda dependencies to v0.14
- PR #795 Fix some documentation
- PR #800 Fix bfs error in optimization path
- PR #825 Fix outdated CONTRIBUTING.md
- PR #827 Fix indexing CI errors due to cudf updates
- PR #844 Fixing tests, converting __getitem__ calls to .iloc


# cuGraph 0.13.0 (Date TBD)

## New Features
- PR #736 cuHornet KTruss integration
- PR #735 Integration gunrock's betweenness centrality
- PR #760 cuHornet Weighted KTruss

## Improvements
- PR #688 Cleanup datasets after testing on gpuCI
- PR #694 Replace the expensive cudaGetDeviceProperties call in triangle counting with cheaper cudaDeviceGetAttribute calls
- PR #701 Add option to filter datasets and tests when run from CI
- PR #715 Added new YML file for CUDA 10.2
- PR #719 Updated docs to remove CUDA 9.2 and add CUDA 10.2
- PR #720 Updated error messages
- PR #722 Refactor graph to remove gdf_column
- PR #723 Added notebook testing to gpuCI gpu build
- PR #734 Updated view_edge_list for Graph, added unrenumbering test, fixed column access issues
- PR #738 Move tests directory up a level
- PR #739 Updated Notebooks
- PR #740 added utility to extract paths from SSSP/BFS results
- PR #742 Rremove gdf column from jaccard
- PR #741 Added documentation for running and adding new benchmarks and shell script to automate
- PR #747 updated viewing of graph, datatypecasting and two hop neighbor unrenumbering for multi column
- PR #766 benchmark script improvements/refactorings: separate ETL steps, averaging, cleanup

## Bug Fixes
- PR #697 Updated versions in conda environments.
- PR #692 Add check after opening golden result files in C++ Katz Centrality tests.
- PR #702 Add libcypher include path to target_include_directories
- PR #716 Fixed bug due to disappearing get_column_data_ptr function in cudf
- PR #726 Fixed SSSP notebook issues in last cell
- PR #728 Temporary fix for dask attribute error issue
- PR #733 Fixed multi-column renumbering issues with indexes
- PR #746 Dask + Distributed 2.12.0+
- PR #753 ECG Error
- PR #758 Fix for graph comparison failure
- PR #761 Added flag to not treat deprecation warnings as errors, for now
- PR #771 Added unrenumbering in wcc and scc. Updated tests to compare vertices of largest component
- PR #774 Raise TypeError if a DiGraph is used with spectral*Clustering()

# cuGraph 0.12.0 (04 Feb 2020)

## New Features
- PR #628 Add (Di)Graph constructor from Multi(Di)Graph
- PR #630 Added ECG clustering
- PR #636 Added Multi-column renumbering support

## Improvements
- PR #640 remove gdf_column in sssp
- PR #629 get rid of gdf_column in pagerank
- PR #641 Add codeowners
- PR #646 Skipping all tests in test_bfs_bsp.py since SG BFS is not formally supported
- PR #652 Remove gdf_column in BFS
- PR #660 enable auto renumbering
- PR #664 Added support for Louvain early termination.
- PR #667 Drop `cython` from run requirements in conda recipe
- PR #666 Incorporate multicolumn renumbering in python graph class for Multi(Di)Graph
- PR #685 Avoid deep copy in index reset

## Bug Fixes
- PR #634 renumber vertex ids passed in analytics
- PR #649 Change variable names in wjaccard and woverlap to avoid exception
- PR #651 fix cudf error in katz wrapper and test nstart
- PR #663 Replaced use of cudf._lib.gdf_dtype_from_value based on cudf refactoring
- PR #670 Use cudf pandas version
- PR #672 fix snmg pagerank based on cudf Buffer changes
- PR #681 fix column length mismatch cudf issue
- PR #684 Deprecated cudf calls
- PR #686 Balanced cut fix
- PR #689 Check graph input type, disable Multi(Di)Graph, add cugraph.from_cudf_edgelist


# cuGraph 0.11.0 (11 Dec 2019)

## New Features
- PR #588 Python graph class and related changes
- PR #630 Adds ECG clustering functionality

## Improvements
- PR #569 Added exceptions
- PR #554 Upgraded namespace so that cugraph can be used for the API.
- PR #564 Update cudf type aliases
- PR #562 Remove pyarrow dependency so we inherit the one cudf uses
- PR #576 Remove adj list conversion automation from c++
- PR #587 API upgrade
- PR #585 Remove BUILD_ABI references from CI scripts
- PR #591 Adding initial GPU metrics to benchmark utils
- PR #599 Pregel BFS
- PR #601 add test for type conversion, edit createGraph_nvgraph
- PR #614 Remove unused CUDA conda labels
- PR #616 Remove c_ prefix
- PR #618 Updated Docs
- PR #619 Transition guide

## Bug Fixes
- PR #570 Temporarily disabling 2 DB tests
- PR #573 Fix pagerank test and symmetrize for cudf 0.11
- PR #574 dev env update
- PR #580 Changed hardcoded test output file to a generated tempfile file name
- PR #595 Updates to use the new RMM Python reinitialize() API
- PR #625 use destination instead of target when adding edgelist

# cuGraph 0.10.0 (16 Oct 2019)


## New Features
- PR #469 Symmetrize a COO
- PR #477 Add cuHornet as a submodule
- PR #483 Katz Centrality
- PR #524 Integrated libcypher-parser conda package into project.
- PR #493 Added C++ findMatches operator for OpenCypher query.
- PR #527 Add testing with asymmetric graph (where appropriate)
- PR #520 KCore and CoreNumber
- PR #496 Gunrock submodule + SM prelimis.
- PR #575 Added updated benchmark files that use new func wrapper pattern and asvdb

## Improvements
- PR #466 Add file splitting test; Update to reduce dask overhead
- PR #468 Remove unnecessary print statement
- PR #464 Limit initial RMM pool allocator size to 128mb so pytest can run in parallel
- PR #474 Add csv file writing, lazy compute - snmg pagerank
- PR #481 Run bfs on unweighted graphs when calling sssp
- PR #491 Use YYMMDD tag in nightly build
- PR #487 Add woverlap test, add namespace in snmg COO2CSR
- PR #531 Use new rmm python package

## Bug Fixes
- PR #458 Fix potential race condition in SSSP
- PR #471 Remove nvidia driver installation from ci/cpu/build.sh
- PR #473 Re-sync cugraph with cudf (cudf renamed the bindings directory to _lib).
- PR #480 Fixed DASK CI build script
- PR #478 Remove requirements and setup for pi
- PR #495 Fixed cuhornet and cmake for Turing cards
- PR #489 Handle negative vertex ids in renumber
- PR #519 Removed deprecated cusparse calls
- PR #522 Added the conda dev env file for 10.1
- PR #525 Update build scripts and YYMMDD tagging for nightly builds
- PR #548 Added missing cores documentation
- PR #556 Fixed recursive remote options for submodules
- PR #559 Added RMM init check so RMM free APIs are not called if not initialized


# cuGraph 0.9.0 (21 Aug 2019)

## New Features
- PR #361 Prototypes for cusort functions
- PR #357 Pagerank cpp API
- PR #366 Adds graph.degrees() function returning both in and out degree.
- PR #380 First implemention of cusort - SNMG key/value sorting
- PR #416 OpenCypher: Added C++ implementation of db_object class and assorted other classes
- PR #411 Integrate dask-cugraph in cugraph
- PR #411 Integrate dask-cugraph in cugraph #411
- PR #418 Update cusort to handle SNMG key-only sorting
- PR #423 Add Strongly Connected Components (GEMM); Weakly CC updates;
- PR #437 Streamline CUDA_REL environment variable
- PR #449 Fix local build generated file ownerships
- PR #454 Initial version of updated script to run benchmarks


## Improvements
- PR #353 Change snmg python wrapper in accordance to cpp api
- PR #362 Restructured python/cython directories and files.
- PR #365 Updates for setting device and vertex ids for snmg pagerank
- PR #383 Exposed MG pagerank solver parameters
- PR #399 Example Prototype of Strongly Connected Components using primitives
- PR #419 Version test
- PR #420 drop duplicates, remove print, compute/wait read_csv in pagerank.py
- PR #439 More efficient computation of number of vertices from edge list
- PR #445 Update view_edge_list, view_adj_list, and view_transposed_adj_list to return edge weights.
- PR #450 Add a multi-GPU section in cuGraph documentation.

## Bug Fixes
- PR #368 Bump cudf dependency versions for cugraph conda packages
- PR #354 Fixed bug in building a debug version
- PR #360 Fixed bug in snmg coo2csr causing intermittent test failures.
- PR #364 Fixed bug building or installing cugraph when conda isn't installed
- PR #375 Added a function to initialize gdf columns in cugraph #375
- PR #378 cugraph was unable to import device_of_gpu_pointer
- PR #384 Fixed bug in snmg coo2csr causing error in dask-cugraph tests.
- PR #382 Disabled vertex id check to allow Azure deployment
- PR #410 Fixed overflow error in SNMG COO2CSR
- PR #395 run omp_ge_num_threads in a parallel context
- PR #412 Fixed formatting issues in cuGraph documentation.
- PR #413 Updated python build instructions.
- PR #414 Add weights to wjaccrd.py
- PR #436 Fix Skip Test Functionality
- PR #438 Fix versions of packages in build script and conda yml
- PR #441 Import cudf_cpp.pxd instead of duplicating cudf definitions.
- PR #441 Removed redundant definitions of python dictionaries and functions.
- PR #442 Updated versions in conda environments.
- PR #442 Added except + to cython bindings to C(++) functions.
- PR #443 Fix accuracy loss issue for snmg pagerank
- PR #444 Fix warnings in strongly connected components
- PR #446 Fix permission for source (-x) and script (+x) files.
- PR #448 Import filter_unreachable
- PR #453 Re-sync cugraph with cudf (dependencies, type conversion & scatter functions).
- PR #463 Remove numba dependency and use the one from cudf

# cuGraph 0.8.0 (27 June 2019)

## New Features
- PR #287 SNMG power iteration step1
- PR #297 SNMG degree calculation
- PR #300 Personalized Page Rank
- PR #302 SNMG CSR Pagerank (cuda/C++)
- PR #315 Weakly Connected Components adapted from cuML (cuda/C++)
- PR #323 Add test skipping function to build.sh
- PR #308 SNMG python wrapper for pagerank
- PR #321 Added graph initialization functions for NetworkX compatibility.
- PR #332 Added C++ support for strings in renumbering function
- PR #325 Implement SSSP with predecessors (cuda/C++)
- PR #331 Python bindings and test for Weakly Connected Components.
- PR #339 SNMG COO2CSR (cuda/C++)
- PR #341 SSSP with predecessors (python) and function for filtering unreachable nodes in the traversal
- PR #348 Updated README for release

## Improvements
- PR #291 nvGraph is updated to use RMM instead of directly invoking cnmem functions.
- PR #286 Reorganized cugraph source directory
- PR #306 Integrated nvgraph to libcugraph.so (libnvgraph_rapids.so will not be built anymore).
- PR #306 Updated python test files to run pytest with all four RMM configurations.
- PR #321 Added check routines for input graph data vertex IDs and offsets (cugraph currently supports only 32-bit integers).
- PR #333 Various general improvements at the library level

## Bug Fixes
- PR #283 Automerge fix
- PR #291 Fixed a RMM memory allocation failure due to duplicate copies of cnmem.o
- PR #291 Fixed a cub CsrMV call error when RMM pool allocator is used.
- PR #306 Fixed cmake warnings due to library conflicts.
- PR #311 Fixed bug in SNMG degree causing failure for three gpus
- PR #309 Update conda build recipes
- PR #314 Added datasets to gitignore
- PR #322 Updates to accommodate new cudf include file locations
- PR #324 Fixed crash in WeakCC for larger graph and added adj matrix symmetry check
- PR #327 Implemented a temporary fix for the build failure due to gunrock updates.
- PR #345 Updated CMakeLists.txt to apply RUNPATH to transitive dependencies.
- PR #350 Configure Sphinx to render params correctly
- PR #359 Updates to remove libboost_system as a runtime dependency on libcugraph.so


# cuGraph 0.7.0 (10 May 2019)

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
