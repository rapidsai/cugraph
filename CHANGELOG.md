# cuGraph 0.18.0 (Date TBD)

## New Features

## Improvements

## Bug Fixes

# cuGraph 0.17.0 (Date TBD)

## New Features
- PR #1276 MST
- PR #1245 Add functions to add pandas and numpy compatibility
- PR #1260 Add katz_centrality mnmg wrapper
- PR #1264 CuPy sparse matrix input support for WCC, SCC, SSSP, and BFS
- PR #1265 Implement Hungarian Algorithm
- PR #1274 Add generic from_edgelist() and from_adjlist() APIs
- PR #1279 Add self loop check variable in graph
- PR #1277 SciPy sparse matrix input support for WCC, SCC, SSSP, and BFS
- PR #1278 Add support for shortest_path_length and fix graph vertex checks

## Improvements
- PR #1227 Pin cmake policies to cmake 3.17 version
- PR #1267 Compile time improvements via Explicit Instantiation Declarations.
- PR #1269 Removed old db code that was not being used
- PR #1271 Add extra check to make SG Louvain deterministic
- PR #1273 Update Force Atlas 2 notebook, wrapper and coding style
- PR #1289 Update api.rst for MST
- PR #1281 Update README
- PR #1293: Updating RAFT to latest

## Bug Fixes
- PR #1237 update tests for assymetric graphs, enable personalization pagerank
- PR #1242 Calling gunrock cmake using explicit -D options, re-enabling C++ tests
- PR #1246 Use latest Gunrock, update HITS implementation
- PR #1250 Updated cuco commit hash to latest as of 2020-10-30 and removed unneeded GIT_SHALLOW param
- PR #1251 Changed the MG context testing class to use updated parameters passed in from the individual tests
- PR #1253 MG test fixes: updated additional comms.initialize() calls, fixed dask DataFrame comparisons
- PR #1270 Raise exception for p2p, disable bottom up approach for bfs
- PR #1275 Force local artifact conda install
- PR #1285 Move codecov upload to gpu build script
- PR #1290 Update weights check in bc and graph prims wrappers
- PR #1299 Update doc and notebook
- PR #1304 Enable all GPU archs for test builds

# cuGraph 0.16.0 (21 Oct 2020)

## New Features
- PR #1098 Add new graph classes to support 2D partitioning
- PR #1124 Sub-communicator initialization for 2D partitioning support
- PR #838 Add pattern accelerator API functions and pattern accelerator API based implementations of PageRank, Katz Centrality, BFS, and SSSP
- PR #1147 Added support for NetworkX graphs as input type
- PR #1157 Louvain API update to use graph_container_t
- PR #1151 MNMG extension for pattern accelerator based PageRank, Katz Centrality, BFS, and SSSP implementations (C++ part)
- PR #1163 Integrated 2D shuffling and Louvain updates
- PR #1178 Refactored cython graph factory code to scale to additional data types
- PR #1175 Integrated 2D pagerank python/cython infra
- PR #1177 Integrated 2D bfs and sssp python/cython infra
- PR #1172 MNMG Louvain implementation

## Improvements
- PR 1081 MNMG Renumbering - sort partitions by degree
- PR 1115 Replace deprecated rmm::mr::get_default_resource with rmm::mr::get_current_device_resource
- PR #1133 added python 2D shuffling
- PR 1129 Refactored test to use common dataset and added additional doc pages
- PR 1135 SG Updates to Louvain et. al.
- PR 1132 Upgrade Thrust to latest commit
- PR #1129 Refactored test to use common dataset and added additional doc pages
- PR #1145 Simple edge list generator
- PR #1144 updated documentation and APIs
- PR #1139 MNMG Louvain Python updates, Cython cleanup
- PR #1156 Add aarch64 gencode support
- PR #1149 Parquet read and concat within workers
- PR #1152 graph container cleanup, added arg for instantiating legacy types and switch statements to factory function
- PR #1164 MG symmetrize and conda env updates
- PR #1162 enhanced networkx testing
- PR #1169 Added RAPIDS cpp packages to cugraph dev env
- PR #1165 updated remaining algorithms to be NetworkX compatible
- PR #1176 Update ci/local/README.md
- PR #1184 BLD getting latest tags
- PR #1222 Added min CUDA version check to MG Louvain
- PR #1217 NetworkX Transition doc
- PR #1223 Update mnmg docs
- PR #1230 Improve gpuCI scripts

## Bug Fixes
- PR #1131 Show style checker errors with set +e
- PR #1150 Update RAFT git tag
- PR #1155 Remove RMM library dependency and CXX11 ABI handling
- PR #1158 Pass size_t* & size_t* instead of size_t[] & int[] for raft allgatherv's input parameters recvcounts & displs
- PR #1168 Disabled MG tests on single GPU
- PR #1166 Fix misspelling of function calls in asserts causing debug build to fail
- PR #1180 BLD Adopt RAFT model for cuhornet dependency
- PR #1181 Fix notebook error handling in CI
- PR #1199 BUG segfault in python test suite
- PR #1186 BLD Installing raft headers under cugraph
- PR #1192 Fix benchmark notes and documentation issues in graph.py
- PR #1196 Move subcomms init outside of individual algorithm functions
- PR #1198 Remove deprecated call to from_gpu_matrix
- PR #1174 Fix bugs in MNMG pattern accelerators and pattern accelerator based implementations of MNMG PageRank, BFS, and SSSP
- PR #1233 Temporarily disabling C++ tests for 0.16
- PR #1240 Require `ucx-proc=*=gpu`
- PR #1241 Fix a bug in personalized PageRank with the new graph primitives API.
- PR #1249 Fix upload script syntax

# cuGraph 0.15.0 (26 Aug 2020)

## New Features
- PR #940 Add MG Batch BC
- PR #937 Add wrapper for gunrock HITS algorithm
- PR #939 Updated Notebooks to include new features and benchmarks
- PR #944 MG pagerank (dask)
- PR #947 MG pagerank (CUDA)
- PR #826 Bipartite Graph python API
- PR #963 Renumbering refactor, add multi GPU support
- PR #964 MG BFS (CUDA)
- PR #990 MG Consolidation
- PR #993 Add persistent Handle for Comms
- PR #979 Add hypergraph implementation to convert DataFrames into Graphs
- PR #1010 MG BFS (dask)
- PR #1018 MG personalized pagerank
- PR #1047 Updated select tests to use new dataset list that includes asymmetric directed graph
- PR #1090 Add experimental Leiden function
- PR #1077 Updated/added copyright notices, added copyright CI check from cuml
- PR #1100 Add support for new build process (Project Flash)
- PR #1093 New benchmarking notebook

## Improvements
- PR #898 Add Edge Betweenness Centrality, and endpoints to BC
- PR #913 Eliminate `rmm.device_array` usage
- PR #903 Add short commit hash to conda package
- PR #920 modify bfs test, update graph number_of_edges, update storage of transposedAdjList in Graph
- PR #933 Update mg_degree to use raft, add python tests
- PR #930 rename test_utils.h to utilities/test_utils.hpp and remove thrust dependency
- PR #934 Update conda dev environment.yml dependencies to 0.15
- PR #942 Removed references to deprecated RMM headers.
- PR #941 Regression python/cudf fix
- PR #945 Simplified benchmark --no-rmm-reinit option, updated default options
- PR #946 Install meta packages for dependencies
- PR #952 Updated get_test_data.sh to also (optionally) download and install datasets for benchmark runs
- PR #953 fix setting RAFT_DIR from the RAFT_PATH env var
- PR #954 Update cuGraph error handling to use RAFT
- PR #968 Add build script for CI benchmark integration
- PR #959 Add support for uint32_t and int64_t types for BFS (cpp side)
- PR #962 Update dask pagerank
- PR #975 Upgrade GitHub template
- PR #976 Fix error in Graph.edges(), update cuDF rename() calls
- PR #977 Update force_atlas2 to call on_train_end after iterating
- PR #980 Replace nvgraph Spectral Clustering (SC) functionality with RAFT SC
- PR #987 Move graph out of experimental namespace
- PR #984 Removing codecov until we figure out how to interpret failures that block CI
- PR #985 Add raft handle to BFS, BC and edge BC
- PR #991 Update conda upload versions for new supported CUDA/Python
- PR #988 Add clang and clang tools to the conda env
- PR #997 Update setup.cfg to run pytests under cugraph tests directory only
- PR #1007 Add tolerance support to MG Pagerank and fix
- PR #1009 Update benchmarks script to include requirements used
- PR #1014 Fix benchmarks script variable name
- PR #1021 Update cuGraph to use RAFT CUDA utilities
- PR #1019 Remove deprecated CUDA library calls
- PR #1024 Updated condata environment YML files
- PR #1026 update chunksize for mnmg, remove files and unused code
- PR #1028 Update benchmarks script to use ASV_LABEL
- PR #1030 MG directory org and documentation
- PR #1020 Updated Louvain to honor max_level, ECG now calls Louvain for 1 level, then full run.
- PR #1031 MG notebook
- PR #1034 Expose resolution (gamma) parameter in Louvain
- PR #1037 Centralize test main function and replace usage of deprecated `cnmem_memory_resource`
- PR #1041 Use S3 bucket directly for benchmark plugin
- PR #1056 Fix MG BFS performance
- PR #1062 Compute max_vertex_id in mnmg local data computation
- PR #1068 Remove unused thirdparty code
- PR #1105 Update `master` references to `main`

## Bug Fixes
- PR #936 Update Force Atlas 2 doc and wrapper
- PR #938 Quote conda installs to avoid bash interpretation
- PR #966 Fix build error (debug mode)
- PR #983 Fix offset calculation in COO to CSR
- PR #989: Fix issue with incorrect docker image being used in local build script
- PR #992 Fix unrenumber of predecessor
- PR #1008 Fix for cudf updates disabling iteration of Series/Columns/Index
- PR #1012 Fix Local build script README
- PR #1017 Fix more mg bugs
- PR #1022 Fix support for using a cudf.DataFrame with a MG graph
- PR #1025: Explicitly skip raft test folder for pytest 6.0.0
- PR #1027 Fix documentation
- PR #1033 Fix reparition error in big datasets, updated coroutine, fixed warnings
- PR #1036 Fixed benchmarks for new renumbering API, updated comments, added quick test-only benchmark run to CI
- PR #1040 Fix spectral clustering renumbering issue
- PR #1057 Updated raft dependency to pull fixes on cusparse selection in CUDA 11
- PR #1066 Update cugunrock to not build for unsupported CUDA architectures
- PR #1069 Fixed CUDA 11 Pagerank crash, by replacing CUB's SpMV with raft's.
- PR #1083 Fix NBs to run in nightly test run, update renumbering text, cleanup
- PR #1087 Updated benchmarks README to better describe how to get plugin, added rapids-pytest-benchmark plugin to conda dev environments
- PR #1101 Removed unnecessary device-to-host copy which caused a performance regression
- PR #1106 Added new release.ipynb to notebook test skip list
- PR #1125 Patch Thrust to workaround `CUDA_CUB_RET_IF_FAIL` macro clearing CUDA errors


# cuGraph 0.14.0 (03 Jun 2020)

## New Features
- PR #756 Add Force Atlas 2 layout
- PR #822 Added new functions in python graph class, similar to networkx
- PR #840 MG degree
- PR #875 UVM notebook
- PR #881 Raft integration infrastructure

## Improvements
- PR #917 Remove gunrock option from Betweenness Centrality
- PR #764 Updated sssp and bfs with GraphCSR, removed gdf_column, added nullptr weights test for sssp
- PR #765 Remove gdf_column from connected components
- PR #780 Remove gdf_column from cuhornet features
- PR #781 Fix compiler argument syntax for ccache
- PR #782 Use Cython's `new_build_ext` (if available)
- PR #788 Added options and config file to enable codecov
- PR #793 Fix legacy cudf imports/cimports
- PR #798 Edit return graph type in algorithms return graphs
- PR #799 Refactored graph class with RAII
- PR #802 Removed use of gdf_column from db code
- PR #803 Enable Ninja build
- PR #804 Cythonize in parallel
- PR #807 Updating the Python docs
- PR #817 Add native Betweenness Centrality with sources subset
- PR #818 Initial version of new "benchmarks" folder
- PR #820 MG infra and all-gather smoke test
- PR #823 Remove gdf column from nvgraph
- PR #829 Updated README and CONTRIBUTIOIN docs
- PR #831 Updated Notebook - Added K-Truss, ECG, and Betweenness Centrality
- PR #832 Removed RMM ALLOC from db subtree
- PR #833 Update graph functions to use new Graph class
- PR #834 Updated local gpuci build
- PR #836 Remove SNMG code
- PR #845 Add .clang-format & format all files
- PR #859 Updated main docs
- PR #862 Katz Centrality : Auto calculation of alpha parameter if set to none
- PR #865 Added C++ docs
- PR #866 Use RAII graph class in KTruss
- PR #867 Updates to support the latest flake8 version
- PR #874 Update setup.py to use custom clean command
- PR #876 Add BFS C++ tests
- PR #878 Updated build script
- PR #887 Updates test to common datasets
- PR #879 Add docs build script to repository
- PR #880 Remove remaining gdf_column references
- PR #882 Add Force Atlas 2 to benchmarks
- PR #891 A few gdf_column stragglers
- PR #893 Add external_repositories dir and raft symlink to .gitignore
- PR #897 Remove RMM ALLOC calls
- PR #899 Update include paths to remove deleted cudf headers
- PR #906 Update Louvain notebook
- PR #948 Move doc customization scripts to Jenkins

## Bug Fixes
- PR #927 Update scikit learn dependency
- PR #916 Fix CI error on Force Atlas 2 test
- PR #763 Update RAPIDS conda dependencies to v0.14
- PR #795 Fix some documentation
- PR #800 Fix bfs error in optimization path
- PR #825 Fix outdated CONTRIBUTING.md
- PR #827 Fix indexing CI errors due to cudf updates
- PR #844 Fixing tests, converting __getitem__ calls to .iloc
- PR #851 Removed RMM from tests
- PR #852 Fix BFS Notebook
- PR #855 Missed a file in the original SNMG PR
- PR #860 Fix all Notebooks
- PR #870 Fix Louvain
- PR #889 Added missing conftest.py file to benchmarks dir
- PR #896 mg dask infrastructure fixes
- PR #907 Fix bfs directed missing vertices
- PR #911 Env and changelog update
- PR #923 Updated pagerank with @afender 's temp fix for double-free crash
- PR #928 Fix scikit learn test install to work with libgcc-ng 7.3
- PR 935 Merge
- PR #956 Use new gpuCI image in local build script


# cuGraph 0.13.0 (31 Mar 2020)

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
