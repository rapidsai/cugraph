# cuGraph 23.02.00 (9 Feb 2023)

## 🚨 Breaking Changes

- Pin `dask` and `distributed` for release ([#3232](https://github.com/rapidsai/cugraph/pull/3232)) [@galipremsagar](https://github.com/galipremsagar)
- Replace PropertyGraph in cugraph-PyG with FeatureStore ([#3159](https://github.com/rapidsai/cugraph/pull/3159)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Remove CGS from cuGraph-PyG ([#3155](https://github.com/rapidsai/cugraph/pull/3155)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Update cugraph_dgl to use the new FeatureStore ([#3143](https://github.com/rapidsai/cugraph/pull/3143)) [@VibhuJawa](https://github.com/VibhuJawa)
- Implement New Sampling API in Python ([#3082](https://github.com/rapidsai/cugraph/pull/3082)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Adds parameterized benchmarks for `uniform_neighbor_sampling`, updates `benchmarks` dir for future additions ([#3048](https://github.com/rapidsai/cugraph/pull/3048)) [@rlratzel](https://github.com/rlratzel)

## 🐛 Bug Fixes

- Import handle from core ([#3190](https://github.com/rapidsai/cugraph/pull/3190)) [@vyasr](https://github.com/vyasr)
- Pin gcc to 9.x. ([#3174](https://github.com/rapidsai/cugraph/pull/3174)) [@vyasr](https://github.com/vyasr)
- Fixes devices vector alloc to fix seg fault, removes unused RAFT code in PLC, re-enables full CI testing ([#3167](https://github.com/rapidsai/cugraph/pull/3167)) [@rlratzel](https://github.com/rlratzel)
- TEMPORARILY allows python and notebook tests that return exit code 139 to pass. ([#3132](https://github.com/rapidsai/cugraph/pull/3132)) [@rlratzel](https://github.com/rlratzel)
- Bug fix in the C++ CSV file reader (used in C++ testing only). ([#3055](https://github.com/rapidsai/cugraph/pull/3055)) [@seunghwak](https://github.com/seunghwak)

## 📖 Documentation

- Create a notebook comparing nx and cuGraph using synthetic data ([#3135](https://github.com/rapidsai/cugraph/pull/3135)) [@acostadon](https://github.com/acostadon)
- Add API&#39;s for dgl, pyg, cugraph service (server and client) to sphinx ([#3075](https://github.com/rapidsai/cugraph/pull/3075)) [@acostadon](https://github.com/acostadon)
- redo cuGraph main docs ([#3060](https://github.com/rapidsai/cugraph/pull/3060)) [@acostadon](https://github.com/acostadon)

## 🚀 New Features

- Bulk Loading Support for cuGraph-PyG ([#3170](https://github.com/rapidsai/cugraph/pull/3170)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Feature storage ([#3139](https://github.com/rapidsai/cugraph/pull/3139)) [@VibhuJawa](https://github.com/VibhuJawa)
- Add `RelGraphConv`, `GATConv` and `SAGEConv` models to `cugraph_dgl` ([#3131](https://github.com/rapidsai/cugraph/pull/3131)) [@tingyu66](https://github.com/tingyu66)
- Created notebook for running louvain algorithm on a Multi-GPU Property Graph ([#3130](https://github.com/rapidsai/cugraph/pull/3130)) [@acostadon](https://github.com/acostadon)
- cugraph_dgl benchmarks ([#3092](https://github.com/rapidsai/cugraph/pull/3092)) [@VibhuJawa](https://github.com/VibhuJawa)
- Add DGL benchmarks ([#3089](https://github.com/rapidsai/cugraph/pull/3089)) [@VibhuJawa](https://github.com/VibhuJawa)
- Add cugraph+UCX build instructions ([#3088](https://github.com/rapidsai/cugraph/pull/3088)) [@VibhuJawa](https://github.com/VibhuJawa)
- Implement New Sampling API in Python ([#3082](https://github.com/rapidsai/cugraph/pull/3082)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Update per_v_transform_reduce_incoming|outgoing_e to take a reduction operator. ([#2975](https://github.com/rapidsai/cugraph/pull/2975)) [@seunghwak](https://github.com/seunghwak)

## 🛠️ Improvements

- Pin `dask` and `distributed` for release ([#3232](https://github.com/rapidsai/cugraph/pull/3232)) [@galipremsagar](https://github.com/galipremsagar)
- Update shared workflow branches ([#3231](https://github.com/rapidsai/cugraph/pull/3231)) [@ajschmidt8](https://github.com/ajschmidt8)
- Updates dependency to latest DGL ([#3211](https://github.com/rapidsai/cugraph/pull/3211)) [@rlratzel](https://github.com/rlratzel)
- Make graph objects accessible across multiple clients ([#3192](https://github.com/rapidsai/cugraph/pull/3192)) [@VibhuJawa](https://github.com/VibhuJawa)
- Drop extraneous columns that were appearing in MGPropertyGraph ([#3191](https://github.com/rapidsai/cugraph/pull/3191)) [@eriknw](https://github.com/eriknw)
- Enable using cugraph uniform sampling in multi client environments ([#3184](https://github.com/rapidsai/cugraph/pull/3184)) [@VibhuJawa](https://github.com/VibhuJawa)
- DGL Dataloader ([#3181](https://github.com/rapidsai/cugraph/pull/3181)) [@VibhuJawa](https://github.com/VibhuJawa)
- Update cuhornet to fix `using namespace rmm;`. ([#3171](https://github.com/rapidsai/cugraph/pull/3171)) [@bdice](https://github.com/bdice)
- add type annotations to `cugraph_dgl` nn modules ([#3166](https://github.com/rapidsai/cugraph/pull/3166)) [@tingyu66](https://github.com/tingyu66)
- Replace Raft header ([#3162](https://github.com/rapidsai/cugraph/pull/3162)) [@lowener](https://github.com/lowener)
- Update to support NetworkX 3.0 (and handle other deprecations) ([#3161](https://github.com/rapidsai/cugraph/pull/3161)) [@eriknw](https://github.com/eriknw)
- Replace PropertyGraph in cugraph-PyG with FeatureStore ([#3159](https://github.com/rapidsai/cugraph/pull/3159)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Adding density algorithm and test ([#3156](https://github.com/rapidsai/cugraph/pull/3156)) [@BradReesWork](https://github.com/BradReesWork)
- Remove CGS from cuGraph-PyG ([#3155](https://github.com/rapidsai/cugraph/pull/3155)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Update cugraph_dgl to use the new FeatureStore ([#3143](https://github.com/rapidsai/cugraph/pull/3143)) [@VibhuJawa](https://github.com/VibhuJawa)
- Fix documentation author ([#3128](https://github.com/rapidsai/cugraph/pull/3128)) [@bdice](https://github.com/bdice)
- build.sh switch to use `RAPIDS` magic value ([#3127](https://github.com/rapidsai/cugraph/pull/3127)) [@robertmaynard](https://github.com/robertmaynard)
- Drop DiGraph ([#3126](https://github.com/rapidsai/cugraph/pull/3126)) [@BradReesWork](https://github.com/BradReesWork)
- MGPropertyGraph: fix OOM when renumbering by type ([#3123](https://github.com/rapidsai/cugraph/pull/3123)) [@eriknw](https://github.com/eriknw)
- Build CUDA 11.8 and Python 3.10 Packages ([#3120](https://github.com/rapidsai/cugraph/pull/3120)) [@bdice](https://github.com/bdice)
- Updates README for cugraph-service to provide an up-to-date quickstart ([#3119](https://github.com/rapidsai/cugraph/pull/3119)) [@rlratzel](https://github.com/rlratzel)
- Speed Improvements for cuGraph-PyG (Short Circuit, Use Type Indices) ([#3101](https://github.com/rapidsai/cugraph/pull/3101)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Update workflows for nightly tests ([#3098](https://github.com/rapidsai/cugraph/pull/3098)) [@ajschmidt8](https://github.com/ajschmidt8)
- GH Actions Notebook Testing Fixes ([#3097](https://github.com/rapidsai/cugraph/pull/3097)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build pip wheels alongside conda CI ([#3096](https://github.com/rapidsai/cugraph/pull/3096)) [@sevagh](https://github.com/sevagh)
- Add notebooks testing to GH Actions PR Workflow ([#3095](https://github.com/rapidsai/cugraph/pull/3095)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix C++ Bugs in Graph Creation with Edge Properties ([#3093](https://github.com/rapidsai/cugraph/pull/3093)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Update `cugraph` recipes ([#3091](https://github.com/rapidsai/cugraph/pull/3091)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix tests for MG property graph ([#3090](https://github.com/rapidsai/cugraph/pull/3090)) [@eriknw](https://github.com/eriknw)
- Adds initial cugraph-service client scaling benchmark, refactorings, performance config updates ([#3087](https://github.com/rapidsai/cugraph/pull/3087)) [@rlratzel](https://github.com/rlratzel)
- Optimize pg.get_x_data APIs ([#3086](https://github.com/rapidsai/cugraph/pull/3086)) [@VibhuJawa](https://github.com/VibhuJawa)
- Add GitHub Actions Workflows ([#3076](https://github.com/rapidsai/cugraph/pull/3076)) [@bdice](https://github.com/bdice)
- Updates conda versioning to install correct dependencies, changes CI script to better track deps from individual build installs ([#3066](https://github.com/rapidsai/cugraph/pull/3066)) [@seunghwak](https://github.com/seunghwak)
- Use pre-commit for CI style checks. ([#3062](https://github.com/rapidsai/cugraph/pull/3062)) [@bdice](https://github.com/bdice)
- Sampling primitive performance optimization. ([#3061](https://github.com/rapidsai/cugraph/pull/3061)) [@seunghwak](https://github.com/seunghwak)
- Replace clock_gettime with std::chrono::steady_clock ([#3049](https://github.com/rapidsai/cugraph/pull/3049)) [@seunghwak](https://github.com/seunghwak)
- Adds parameterized benchmarks for `uniform_neighbor_sampling`, updates `benchmarks` dir for future additions ([#3048](https://github.com/rapidsai/cugraph/pull/3048)) [@rlratzel](https://github.com/rlratzel)
- Add dependencies.yaml for rapids-dependency-file-generator ([#3042](https://github.com/rapidsai/cugraph/pull/3042)) [@ChuckHastings](https://github.com/ChuckHastings)
- Unpin `dask` and `distributed` for development ([#3036](https://github.com/rapidsai/cugraph/pull/3036)) [@galipremsagar](https://github.com/galipremsagar)
- Forward merge 22.12 into 23.02 ([#3033](https://github.com/rapidsai/cugraph/pull/3033)) [@vyasr](https://github.com/vyasr)
- Optimize pg.add_data for vector properties ([#3022](https://github.com/rapidsai/cugraph/pull/3022)) [@VibhuJawa](https://github.com/VibhuJawa)
- Adds better reporting of server subprocess errors during testing ([#3012](https://github.com/rapidsai/cugraph/pull/3012)) [@rlratzel](https://github.com/rlratzel)
- Update cugraph_dgl to use vector_properties ([#3000](https://github.com/rapidsai/cugraph/pull/3000)) [@VibhuJawa](https://github.com/VibhuJawa)
- Fix MG C++ Jaccard/Overlap/Sorensen coefficients tests. ([#2999](https://github.com/rapidsai/cugraph/pull/2999)) [@seunghwak](https://github.com/seunghwak)
- Update Uniform Neighborhood Sampling API ([#2997](https://github.com/rapidsai/cugraph/pull/2997)) [@ChuckHastings](https://github.com/ChuckHastings)
- Use Vertex ID Offsets in CuGraphStorage ([#2996](https://github.com/rapidsai/cugraph/pull/2996)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Replace deprecated raft headers ([#2978](https://github.com/rapidsai/cugraph/pull/2978)) [@lowener](https://github.com/lowener)

# cuGraph 22.12.00 (8 Dec 2022)

## 🚨 Breaking Changes

- remove all algorithms from cython.cu ([#2955](https://github.com/rapidsai/cugraph/pull/2955)) [@ChuckHastings](https://github.com/ChuckHastings)
- PyG Monorepo Refactor ([#2905](https://github.com/rapidsai/cugraph/pull/2905)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Fix PyG Loaders by properly supporting `multi_get_tensor` ([#2860](https://github.com/rapidsai/cugraph/pull/2860)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Adds arbitrary server extension support to cugraph-service ([#2850](https://github.com/rapidsai/cugraph/pull/2850)) [@rlratzel](https://github.com/rlratzel)
- Separate edge weights from graph objects and update primitives to support general edge properties. ([#2843](https://github.com/rapidsai/cugraph/pull/2843)) [@seunghwak](https://github.com/seunghwak)
- Move weight-related graph_t and graph_view_t member functions to standalone functions ([#2841](https://github.com/rapidsai/cugraph/pull/2841)) [@seunghwak](https://github.com/seunghwak)
- Avoid directly calling graph constructor (as code cleanup before edge property support in primitives) ([#2834](https://github.com/rapidsai/cugraph/pull/2834)) [@seunghwak](https://github.com/seunghwak)
- Split Sampler from Graph Store to Support New PyG Sampling API ([#2803](https://github.com/rapidsai/cugraph/pull/2803)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Code cleanup (remove dead code and move legacy files to the legacy directory) ([#2798](https://github.com/rapidsai/cugraph/pull/2798)) [@seunghwak](https://github.com/seunghwak)
- remove graph broadcast and serialization object, not used ([#2783](https://github.com/rapidsai/cugraph/pull/2783)) [@ChuckHastings](https://github.com/ChuckHastings)
- Multi-GPU induced subgraph tests code ([#2602](https://github.com/rapidsai/cugraph/pull/2602)) [@yang-hu-nv](https://github.com/yang-hu-nv)

## 🐛 Bug Fixes

- Always build without isolation ([#3052](https://github.com/rapidsai/cugraph/pull/3052)) [@vyasr](https://github.com/vyasr)
- Makes `cugraph-pyg` an optional depenency for `cugraph-service` tests ([#3051](https://github.com/rapidsai/cugraph/pull/3051)) [@rlratzel](https://github.com/rlratzel)
- Fix cugraph_c target name in Python builds ([#3045](https://github.com/rapidsai/cugraph/pull/3045)) [@vyasr](https://github.com/vyasr)
- Initialize CUDA architectures for all Python cugraph builds ([#3041](https://github.com/rapidsai/cugraph/pull/3041)) [@vyasr](https://github.com/vyasr)
- Update the python API to create a PLC graph from a CSR ([#3027](https://github.com/rapidsai/cugraph/pull/3027)) [@jnke2016](https://github.com/jnke2016)
- Updates experimental warning wrapper and PropertyGraph docs for correct experimental namespace name ([#3007](https://github.com/rapidsai/cugraph/pull/3007)) [@rlratzel](https://github.com/rlratzel)
- Fix cluster startup script ([#2977](https://github.com/rapidsai/cugraph/pull/2977)) [@VibhuJawa](https://github.com/VibhuJawa)
- Don&#39;t use CMake 3.25.0 as it has a FindCUDAToolkit show stopping bug ([#2957](https://github.com/rapidsai/cugraph/pull/2957)) [@robertmaynard](https://github.com/robertmaynard)
- Fix build script to install dask main ([#2943](https://github.com/rapidsai/cugraph/pull/2943)) [@galipremsagar](https://github.com/galipremsagar)
- Fixes options added to build.sh for building without cugraph-ops that were dropped in a merge mistake. ([#2935](https://github.com/rapidsai/cugraph/pull/2935)) [@rlratzel](https://github.com/rlratzel)
- Update dgl dependency to dglcuda=11.6 ([#2929](https://github.com/rapidsai/cugraph/pull/2929)) [@VibhuJawa](https://github.com/VibhuJawa)
- Adds option to build.sh to build without cugraphops, updates docs ([#2904](https://github.com/rapidsai/cugraph/pull/2904)) [@rlratzel](https://github.com/rlratzel)
- Fix bug in how is_symmetric is set when transposing storage ([#2898](https://github.com/rapidsai/cugraph/pull/2898)) [@ChuckHastings](https://github.com/ChuckHastings)
- Correct build failures when doing a local build ([#2895](https://github.com/rapidsai/cugraph/pull/2895)) [@robertmaynard](https://github.com/robertmaynard)
- Update `cuda-python` dependency to 11.7.1 ([#2865](https://github.com/rapidsai/cugraph/pull/2865)) [@galipremsagar](https://github.com/galipremsagar)
- Add package to the list of dependencies ([#2858](https://github.com/rapidsai/cugraph/pull/2858)) [@jnke2016](https://github.com/jnke2016)
- Add parameter checks to BFS and SSSP in C API ([#2844](https://github.com/rapidsai/cugraph/pull/2844)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix uniform neighborhood sampling memory leak ([#2835](https://github.com/rapidsai/cugraph/pull/2835)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix out of index errors encountered with  sampling on out of index samples ([#2825](https://github.com/rapidsai/cugraph/pull/2825)) [@VibhuJawa](https://github.com/VibhuJawa)
- Fix MG tests bugs ([#2819](https://github.com/rapidsai/cugraph/pull/2819)) [@jnke2016](https://github.com/jnke2016)
- Fix  MNMG failures in mg_dgl_extensions ([#2786](https://github.com/rapidsai/cugraph/pull/2786)) [@VibhuJawa](https://github.com/VibhuJawa)
- Bug fix when -1 is used as a valid external vertex ID ([#2776](https://github.com/rapidsai/cugraph/pull/2776)) [@seunghwak](https://github.com/seunghwak)

## 📖 Documentation

- Update dgl-cuda conda installation instructions ([#2972](https://github.com/rapidsai/cugraph/pull/2972)) [@VibhuJawa](https://github.com/VibhuJawa)
- cuGraph Readme pages and Documentation API structure refactoring ([#2894](https://github.com/rapidsai/cugraph/pull/2894)) [@acostadon](https://github.com/acostadon)
- Create a page on why we do not support cascading ([#2842](https://github.com/rapidsai/cugraph/pull/2842)) [@BradReesWork](https://github.com/BradReesWork)
- Add ProperyGraph to doc generation and update docstrings ([#2826](https://github.com/rapidsai/cugraph/pull/2826)) [@acostadon](https://github.com/acostadon)
- Updated Release Notebook for changes in latest cuGraph release ([#2800](https://github.com/rapidsai/cugraph/pull/2800)) [@acostadon](https://github.com/acostadon)

## 🚀 New Features

- Add wheel builds ([#2964](https://github.com/rapidsai/cugraph/pull/2964)) [@vyasr](https://github.com/vyasr)
- Reenable copy_prs ([#2959](https://github.com/rapidsai/cugraph/pull/2959)) [@vyasr](https://github.com/vyasr)
- Provide option to keep original vertex/edge IDs when renumbering ([#2951](https://github.com/rapidsai/cugraph/pull/2951)) [@eriknw](https://github.com/eriknw)
- Support cuGraph-Service in cuGraph-PyG ([#2946](https://github.com/rapidsai/cugraph/pull/2946)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Add  conda yml for`cugraph+torch+DGL` dev ([#2919](https://github.com/rapidsai/cugraph/pull/2919)) [@VibhuJawa](https://github.com/VibhuJawa)
- Bring up cugraph_dgl_repo ([#2896](https://github.com/rapidsai/cugraph/pull/2896)) [@VibhuJawa](https://github.com/VibhuJawa)
- Adds setup.py files and conda recipes for cugraph-service ([#2862](https://github.com/rapidsai/cugraph/pull/2862)) [@BradReesWork](https://github.com/BradReesWork)
- Add remote storage support ([#2859](https://github.com/rapidsai/cugraph/pull/2859)) [@VibhuJawa](https://github.com/VibhuJawa)
- Separate edge weights from graph objects and update primitives to support general edge properties. ([#2843](https://github.com/rapidsai/cugraph/pull/2843)) [@seunghwak](https://github.com/seunghwak)
- GitHub Action adding issues/prs to project board ([#2837](https://github.com/rapidsai/cugraph/pull/2837)) [@jarmak-nv](https://github.com/jarmak-nv)
- Replacing markdown issue templates with yml forms ([#2836](https://github.com/rapidsai/cugraph/pull/2836)) [@jarmak-nv](https://github.com/jarmak-nv)
- Cugraph-Service Remote Graphs and Algorithm Dispatch ([#2832](https://github.com/rapidsai/cugraph/pull/2832)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Remote Graph Wrappers for cuGraph-Service ([#2821](https://github.com/rapidsai/cugraph/pull/2821)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Updte transform_reduce_e_by_src|dst_key to take a custom reduction op ([#2813](https://github.com/rapidsai/cugraph/pull/2813)) [@seunghwak](https://github.com/seunghwak)
- C++ minimal CSV reader ([#2791](https://github.com/rapidsai/cugraph/pull/2791)) [@seunghwak](https://github.com/seunghwak)
- K-hop neighbors ([#2782](https://github.com/rapidsai/cugraph/pull/2782)) [@seunghwak](https://github.com/seunghwak)

## 🛠️ Improvements

- Update dask-cuda version and disable wheel builds in CI ([#3009](https://github.com/rapidsai/cugraph/pull/3009)) [@vyasr](https://github.com/vyasr)
- Branch 22.12 merge 22.10 ([#3008](https://github.com/rapidsai/cugraph/pull/3008)) [@rlratzel](https://github.com/rlratzel)
- Shuffle the vertex pair ([#3002](https://github.com/rapidsai/cugraph/pull/3002)) [@jnke2016](https://github.com/jnke2016)
- remove all algorithms from cython.cu ([#2955](https://github.com/rapidsai/cugraph/pull/2955)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update gitignore to Exclude Egg Files ([#2948](https://github.com/rapidsai/cugraph/pull/2948)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Pin `dask` and `distributed` for release ([#2940](https://github.com/rapidsai/cugraph/pull/2940)) [@galipremsagar](https://github.com/galipremsagar)
- Make dgl, pytorch  optional imports for cugraph_dgl package ([#2936](https://github.com/rapidsai/cugraph/pull/2936)) [@VibhuJawa](https://github.com/VibhuJawa)
- Implement k core ([#2933](https://github.com/rapidsai/cugraph/pull/2933)) [@ChuckHastings](https://github.com/ChuckHastings)
- CuGraph-Service Asyncio Fix ([#2932](https://github.com/rapidsai/cugraph/pull/2932)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Debug MG egonet issues ([#2926](https://github.com/rapidsai/cugraph/pull/2926)) [@ChuckHastings](https://github.com/ChuckHastings)
- Optimize  `PG.add_data` ([#2924](https://github.com/rapidsai/cugraph/pull/2924)) [@VibhuJawa](https://github.com/VibhuJawa)
- Implement C API Similarity ([#2923](https://github.com/rapidsai/cugraph/pull/2923)) [@ChuckHastings](https://github.com/ChuckHastings)
- Adds `cugraph-dgl` conda package, updates CI scripts to build and upload it ([#2921](https://github.com/rapidsai/cugraph/pull/2921)) [@rlratzel](https://github.com/rlratzel)
- key, value store abstraction ([#2920](https://github.com/rapidsai/cugraph/pull/2920)) [@seunghwak](https://github.com/seunghwak)
- Implement two_hop_neighbors C API ([#2915](https://github.com/rapidsai/cugraph/pull/2915)) [@ChuckHastings](https://github.com/ChuckHastings)
- PyG Monorepo Refactor ([#2905](https://github.com/rapidsai/cugraph/pull/2905)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Update cugraph to support building for Ada and Hopper ([#2889](https://github.com/rapidsai/cugraph/pull/2889)) [@robertmaynard](https://github.com/robertmaynard)
- Optimize dask.uniform_neighbor_sample ([#2887](https://github.com/rapidsai/cugraph/pull/2887)) [@VibhuJawa](https://github.com/VibhuJawa)
- Add vector properties ([#2882](https://github.com/rapidsai/cugraph/pull/2882)) [@eriknw](https://github.com/eriknw)
- Add view_concat for edge_minor_property_view_t and update transform_reduce_e_by_dst_key to support reduce_op on tuple types ([#2879](https://github.com/rapidsai/cugraph/pull/2879)) [@naimnv](https://github.com/naimnv)
- Update egonet implementation ([#2874](https://github.com/rapidsai/cugraph/pull/2874)) [@jnke2016](https://github.com/jnke2016)
- Use new rapids-cmake functionality for rpath handling. ([#2868](https://github.com/rapidsai/cugraph/pull/2868)) [@vyasr](https://github.com/vyasr)
- Update python WCC to leverage the CAPI ([#2866](https://github.com/rapidsai/cugraph/pull/2866)) [@jnke2016](https://github.com/jnke2016)
- Define and implement C/C++ for MNMG Egonet ([#2864](https://github.com/rapidsai/cugraph/pull/2864)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update uniform random walks implementation ([#2861](https://github.com/rapidsai/cugraph/pull/2861)) [@jnke2016](https://github.com/jnke2016)
- Fix PyG Loaders by properly supporting `multi_get_tensor` ([#2860](https://github.com/rapidsai/cugraph/pull/2860)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- CAPI create graph from CSR ([#2856](https://github.com/rapidsai/cugraph/pull/2856)) [@ChuckHastings](https://github.com/ChuckHastings)
- Remove pg dependency from cugraph store.py ([#2855](https://github.com/rapidsai/cugraph/pull/2855)) [@VibhuJawa](https://github.com/VibhuJawa)
- Define C API and implement induced subgraph ([#2854](https://github.com/rapidsai/cugraph/pull/2854)) [@ChuckHastings](https://github.com/ChuckHastings)
- Adds arbitrary server extension support to cugraph-service ([#2850](https://github.com/rapidsai/cugraph/pull/2850)) [@rlratzel](https://github.com/rlratzel)
- Remove stale labeler ([#2849](https://github.com/rapidsai/cugraph/pull/2849)) [@raydouglass](https://github.com/raydouglass)
- Ensure correct data type ([#2847](https://github.com/rapidsai/cugraph/pull/2847)) [@jnke2016](https://github.com/jnke2016)
- Move weight-related graph_t and graph_view_t member functions to standalone functions ([#2841](https://github.com/rapidsai/cugraph/pull/2841)) [@seunghwak](https://github.com/seunghwak)
- Move &#39;graph_store.py&#39; under dgl_extensions ([#2839](https://github.com/rapidsai/cugraph/pull/2839)) [@VibhuJawa](https://github.com/VibhuJawa)
- Avoid directly calling graph constructor (as code cleanup before edge property support in primitives) ([#2834](https://github.com/rapidsai/cugraph/pull/2834)) [@seunghwak](https://github.com/seunghwak)
- removed docs from cugraph build defaults and updated docs clean ([#2831](https://github.com/rapidsai/cugraph/pull/2831)) [@acostadon](https://github.com/acostadon)
- Define API for Betweenness Centrality ([#2823](https://github.com/rapidsai/cugraph/pull/2823)) [@ChuckHastings](https://github.com/ChuckHastings)
- Adds `.git-blame-ignore-revs` for recent .py files reformatting by `black` ([#2809](https://github.com/rapidsai/cugraph/pull/2809)) [@rlratzel](https://github.com/rlratzel)
- Delete dead code in cython.cu ([#2807](https://github.com/rapidsai/cugraph/pull/2807)) [@seunghwak](https://github.com/seunghwak)
- Persist more in MGPropertyGraph ([#2805](https://github.com/rapidsai/cugraph/pull/2805)) [@eriknw](https://github.com/eriknw)
- Fix concat with different index dtypes in SG PropertyGraph ([#2804](https://github.com/rapidsai/cugraph/pull/2804)) [@eriknw](https://github.com/eriknw)
- Split Sampler from Graph Store to Support New PyG Sampling API ([#2803](https://github.com/rapidsai/cugraph/pull/2803)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- added a passthrough for storing transposed ([#2799](https://github.com/rapidsai/cugraph/pull/2799)) [@BradReesWork](https://github.com/BradReesWork)
- Code cleanup (remove dead code and move legacy files to the legacy directory) ([#2798](https://github.com/rapidsai/cugraph/pull/2798)) [@seunghwak](https://github.com/seunghwak)
- PG: join new vertex data by vertex ids ([#2796](https://github.com/rapidsai/cugraph/pull/2796)) [@eriknw](https://github.com/eriknw)
- Allow passing a dict in feat_name for add_edge_data and add_node_data ([#2795](https://github.com/rapidsai/cugraph/pull/2795)) [@VibhuJawa](https://github.com/VibhuJawa)
- remove graph broadcast and serialization object, not used ([#2783](https://github.com/rapidsai/cugraph/pull/2783)) [@ChuckHastings](https://github.com/ChuckHastings)
- Format Python code with black ([#2778](https://github.com/rapidsai/cugraph/pull/2778)) [@eriknw](https://github.com/eriknw)
- remove unused mechanism for calling Louvain ([#2777](https://github.com/rapidsai/cugraph/pull/2777)) [@ChuckHastings](https://github.com/ChuckHastings)
- Unpin `dask` and `distributed` for development ([#2772](https://github.com/rapidsai/cugraph/pull/2772)) [@galipremsagar](https://github.com/galipremsagar)
- Fix auto-merger ([#2771](https://github.com/rapidsai/cugraph/pull/2771)) [@galipremsagar](https://github.com/galipremsagar)
- Fix library version in yml files ([#2764](https://github.com/rapidsai/cugraph/pull/2764)) [@galipremsagar](https://github.com/galipremsagar)
- Refactor k-core ([#2731](https://github.com/rapidsai/cugraph/pull/2731)) [@jnke2016](https://github.com/jnke2016)
- Adds API option to `uniform_neighbor_sample()` and UCX-Py infrastructure to allow for a client-side device to directly receive results ([#2715](https://github.com/rapidsai/cugraph/pull/2715)) [@rlratzel](https://github.com/rlratzel)
- Add or Update Similarity algorithms ([#2704](https://github.com/rapidsai/cugraph/pull/2704)) [@jnke2016](https://github.com/jnke2016)
- Define a C API for data masking ([#2630](https://github.com/rapidsai/cugraph/pull/2630)) [@ChuckHastings](https://github.com/ChuckHastings)
- Multi-GPU induced subgraph tests code ([#2602](https://github.com/rapidsai/cugraph/pull/2602)) [@yang-hu-nv](https://github.com/yang-hu-nv)

# cuGraph 22.10.00 (12 Oct 2022)

## 🚨 Breaking Changes

- Add `is_multigraph` to PG and change `has_duplicate_edges` to use types ([#2708](https://github.com/rapidsai/cugraph/pull/2708)) [@eriknw](https://github.com/eriknw)
- Enable PLC algos to leverage the PLC graph ([#2682](https://github.com/rapidsai/cugraph/pull/2682)) [@jnke2016](https://github.com/jnke2016)
- Reduce cuGraph Sampling Overhead for PyG ([#2653](https://github.com/rapidsai/cugraph/pull/2653)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Code cleanup ([#2617](https://github.com/rapidsai/cugraph/pull/2617)) [@seunghwak](https://github.com/seunghwak)
- Update vertex_frontier_t to take unsorted (tagged-)vertex list with possible duplicates ([#2584](https://github.com/rapidsai/cugraph/pull/2584)) [@seunghwak](https://github.com/seunghwak)
- CuGraph+PyG Wrappers and Loaders ([#2567](https://github.com/rapidsai/cugraph/pull/2567)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Rename multiple .cuh (.cu) files to .hpp (.cpp) ([#2501](https://github.com/rapidsai/cugraph/pull/2501)) [@seunghwak](https://github.com/seunghwak)

## 🐛 Bug Fixes

- Properly Distribute Start Vertices for MG Uniform Neighbor Sample ([#2765](https://github.com/rapidsai/cugraph/pull/2765)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Removes unneeded test dependency on cugraph from pylibcugraph tests ([#2738](https://github.com/rapidsai/cugraph/pull/2738)) [@rlratzel](https://github.com/rlratzel)
- Add modularity to return result for louvain ([#2706](https://github.com/rapidsai/cugraph/pull/2706)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fixes bug in `NumberMap` preventing use of string vertex IDs for MG graphs ([#2688](https://github.com/rapidsai/cugraph/pull/2688)) [@rlratzel](https://github.com/rlratzel)
- Release all inactive futures ([#2659](https://github.com/rapidsai/cugraph/pull/2659)) [@jnke2016](https://github.com/jnke2016)
- Fix MG PLC algos intermittent hang ([#2607](https://github.com/rapidsai/cugraph/pull/2607)) [@jnke2016](https://github.com/jnke2016)
- Fix MG Louvain C API test ([#2588](https://github.com/rapidsai/cugraph/pull/2588)) [@ChuckHastings](https://github.com/ChuckHastings)

## 📖 Documentation

- Adding new classes to api docs ([#2754](https://github.com/rapidsai/cugraph/pull/2754)) [@acostadon](https://github.com/acostadon)
- Removed reference to hard limit of 2 billion vertices for dask cugraph ([#2680](https://github.com/rapidsai/cugraph/pull/2680)) [@acostadon](https://github.com/acostadon)
- updated list of conferences ([#2672](https://github.com/rapidsai/cugraph/pull/2672)) [@BradReesWork](https://github.com/BradReesWork)
- Refactor Sampling, Structure and Traversal Notebooks ([#2628](https://github.com/rapidsai/cugraph/pull/2628)) [@acostadon](https://github.com/acostadon)

## 🚀 New Features

- Implement a vertex pair intersection primitive ([#2728](https://github.com/rapidsai/cugraph/pull/2728)) [@seunghwak](https://github.com/seunghwak)
- Implement a random selection primitive ([#2703](https://github.com/rapidsai/cugraph/pull/2703)) [@seunghwak](https://github.com/seunghwak)
- adds mechanism to skip notebook directories for different run types ([#2693](https://github.com/rapidsai/cugraph/pull/2693)) [@acostadon](https://github.com/acostadon)
- Create graph with edge property values ([#2660](https://github.com/rapidsai/cugraph/pull/2660)) [@seunghwak](https://github.com/seunghwak)
- Reduce cuGraph Sampling Overhead for PyG ([#2653](https://github.com/rapidsai/cugraph/pull/2653)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Primitive to support gathering one hop neighbors ([#2623](https://github.com/rapidsai/cugraph/pull/2623)) [@seunghwak](https://github.com/seunghwak)
- Define a selection primtive API ([#2586](https://github.com/rapidsai/cugraph/pull/2586)) [@seunghwak](https://github.com/seunghwak)
- Leiden C++ API ([#2569](https://github.com/rapidsai/cugraph/pull/2569)) [@naimnv](https://github.com/naimnv)
- CuGraph+PyG Wrappers and Loaders ([#2567](https://github.com/rapidsai/cugraph/pull/2567)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- create a graph with additional edge properties ([#2521](https://github.com/rapidsai/cugraph/pull/2521)) [@seunghwak](https://github.com/seunghwak)

## 🛠️ Improvements

- Add missing entries in `update-version.sh` ([#2763](https://github.com/rapidsai/cugraph/pull/2763)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` and `distributed` for release ([#2758](https://github.com/rapidsai/cugraph/pull/2758)) [@galipremsagar](https://github.com/galipremsagar)
- Allow users to provide their own edge IDS to PropertyGraph ([#2757](https://github.com/rapidsai/cugraph/pull/2757)) [@eriknw](https://github.com/eriknw)
- Raise a warning for certain algorithms ([#2756](https://github.com/rapidsai/cugraph/pull/2756)) [@jnke2016](https://github.com/jnke2016)
- Fix cuGraph compile-time warnings. ([#2755](https://github.com/rapidsai/cugraph/pull/2755)) [@seunghwak](https://github.com/seunghwak)
- Use new sampling primitives ([#2751](https://github.com/rapidsai/cugraph/pull/2751)) [@ChuckHastings](https://github.com/ChuckHastings)
- C++ implementation for unweighted Jaccard/Sorensen/Overlap ([#2750](https://github.com/rapidsai/cugraph/pull/2750)) [@ChuckHastings](https://github.com/ChuckHastings)
- suppress expansion of unused raft spectral templates ([#2739](https://github.com/rapidsai/cugraph/pull/2739)) [@cjnolet](https://github.com/cjnolet)
- Update unit tests to leverage the datasets API ([#2733](https://github.com/rapidsai/cugraph/pull/2733)) [@jnke2016](https://github.com/jnke2016)
- Update raft import ([#2729](https://github.com/rapidsai/cugraph/pull/2729)) [@jnke2016](https://github.com/jnke2016)
- Document that minimum required CMake version is now 3.23.1 ([#2725](https://github.com/rapidsai/cugraph/pull/2725)) [@robertmaynard](https://github.com/robertmaynard)
- fix Comms import ([#2717](https://github.com/rapidsai/cugraph/pull/2717)) [@BradReesWork](https://github.com/BradReesWork)
- added tests for triangle count on unweighted graphs and graphs with int64 vertex types ([#2716](https://github.com/rapidsai/cugraph/pull/2716)) [@acostadon](https://github.com/acostadon)
- Define k-core API and tests ([#2712](https://github.com/rapidsai/cugraph/pull/2712)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add `is_multigraph` to PG and change `has_duplicate_edges` to use types ([#2708](https://github.com/rapidsai/cugraph/pull/2708)) [@eriknw](https://github.com/eriknw)
- Refactor louvain ([#2705](https://github.com/rapidsai/cugraph/pull/2705)) [@jnke2016](https://github.com/jnke2016)
- new notebook for loading mag240m ([#2701](https://github.com/rapidsai/cugraph/pull/2701)) [@BradReesWork](https://github.com/BradReesWork)
- PG allow get_vertex_data to accept single type or id ([#2698](https://github.com/rapidsai/cugraph/pull/2698)) [@eriknw](https://github.com/eriknw)
- Renumber PG to be contiguous per type ([#2697](https://github.com/rapidsai/cugraph/pull/2697)) [@eriknw](https://github.com/eriknw)
- Added `SamplingResult` cdef class to return cupy &quot;views&quot; for PLC sampling algos instead of copying result data ([#2684](https://github.com/rapidsai/cugraph/pull/2684)) [@rlratzel](https://github.com/rlratzel)
- Enable PLC algos to leverage the PLC graph ([#2682](https://github.com/rapidsai/cugraph/pull/2682)) [@jnke2016](https://github.com/jnke2016)
- `graph_mask_t` and separating raft includes for `host_span` and `device_span` ([#2679](https://github.com/rapidsai/cugraph/pull/2679)) [@cjnolet](https://github.com/cjnolet)
- Promote triangle count from experimental ([#2671](https://github.com/rapidsai/cugraph/pull/2671)) [@jnke2016](https://github.com/jnke2016)
- Small fix to the MG PyG Test to Account for Current Sampling Behavior ([#2666](https://github.com/rapidsai/cugraph/pull/2666)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Move GaaS sources, tests, docs, scripts from the rapidsai/GaaS repo to the cugraph repo ([#2661](https://github.com/rapidsai/cugraph/pull/2661)) [@rlratzel](https://github.com/rlratzel)
- C, Pylibcugraph, and Python API Updates for Edge Types ([#2629](https://github.com/rapidsai/cugraph/pull/2629)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Add coverage for uniform neighbor sampling ([#2625](https://github.com/rapidsai/cugraph/pull/2625)) [@jnke2016](https://github.com/jnke2016)
- Define C and C++ APIs for Jaccard/Sorensen/Overlap ([#2624](https://github.com/rapidsai/cugraph/pull/2624)) [@ChuckHastings](https://github.com/ChuckHastings)
- Code cleanup ([#2617](https://github.com/rapidsai/cugraph/pull/2617)) [@seunghwak](https://github.com/seunghwak)
- Branch 22.10 merge 22.08 ([#2599](https://github.com/rapidsai/cugraph/pull/2599)) [@rlratzel](https://github.com/rlratzel)
- Restructure Louvain to be more like other algorithms ([#2594](https://github.com/rapidsai/cugraph/pull/2594)) [@ChuckHastings](https://github.com/ChuckHastings)
- Hetrograph and dask_cudf support ([#2592](https://github.com/rapidsai/cugraph/pull/2592)) [@VibhuJawa](https://github.com/VibhuJawa)
- remove pagerank from cython.cu ([#2587](https://github.com/rapidsai/cugraph/pull/2587)) [@ChuckHastings](https://github.com/ChuckHastings)
- MG uniform random walk implementation ([#2585](https://github.com/rapidsai/cugraph/pull/2585)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update vertex_frontier_t to take unsorted (tagged-)vertex list with possible duplicates ([#2584](https://github.com/rapidsai/cugraph/pull/2584)) [@seunghwak](https://github.com/seunghwak)
- Use edge_ids directly in uniform sampling call to prevent cost of edge_id lookup ([#2550](https://github.com/rapidsai/cugraph/pull/2550)) [@VibhuJawa](https://github.com/VibhuJawa)
- PropertyGraph set index to vertex and edge ids ([#2523](https://github.com/rapidsai/cugraph/pull/2523)) [@eriknw](https://github.com/eriknw)
- Use rapids-cmake 22.10 best practice for RAPIDS.cmake location ([#2518](https://github.com/rapidsai/cugraph/pull/2518)) [@robertmaynard](https://github.com/robertmaynard)
- Unpin `dask` and `distributed` for development ([#2517](https://github.com/rapidsai/cugraph/pull/2517)) [@galipremsagar](https://github.com/galipremsagar)
- Use category dtype for type in PropertyGraph ([#2510](https://github.com/rapidsai/cugraph/pull/2510)) [@eriknw](https://github.com/eriknw)
- Split edge_partition_src_dst_property.cuh to .hpp and .cuh files. ([#2503](https://github.com/rapidsai/cugraph/pull/2503)) [@seunghwak](https://github.com/seunghwak)
- Rename multiple .cuh (.cu) files to .hpp (.cpp) ([#2501](https://github.com/rapidsai/cugraph/pull/2501)) [@seunghwak](https://github.com/seunghwak)
- Fix Forward-Merger Conflicts ([#2474](https://github.com/rapidsai/cugraph/pull/2474)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add tests for reading edge and vertex data from single input in PG, implementation to follow. ([#2154](https://github.com/rapidsai/cugraph/pull/2154)) [@rlratzel](https://github.com/rlratzel)

# cuGraph 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Change default return type `PropertyGraph.extract_subgraph() -&gt; cugraph.Graph(directed=True)` ([#2460](https://github.com/rapidsai/cugraph/pull/2460)) [@eriknw](https://github.com/eriknw)
- cuGraph code cleanup ([#2431](https://github.com/rapidsai/cugraph/pull/2431)) [@seunghwak](https://github.com/seunghwak)
- Clean up public api ([#2398](https://github.com/rapidsai/cugraph/pull/2398)) [@ChuckHastings](https://github.com/ChuckHastings)
- Delete old nbr sampling software ([#2371](https://github.com/rapidsai/cugraph/pull/2371)) [@ChuckHastings](https://github.com/ChuckHastings)
- Remove GraphCSC/GraphCSCView object, no longer used ([#2354](https://github.com/rapidsai/cugraph/pull/2354)) [@ChuckHastings](https://github.com/ChuckHastings)
- Replace raw pointers with device_span in induced subgraph ([#2348](https://github.com/rapidsai/cugraph/pull/2348)) [@yang-hu-nv](https://github.com/yang-hu-nv)
- Clean up some unused code in the C API (and beyond) ([#2339](https://github.com/rapidsai/cugraph/pull/2339)) [@ChuckHastings](https://github.com/ChuckHastings)
- Performance-optimize storing edge partition source/destination properties in (key, value) pairs ([#2328](https://github.com/rapidsai/cugraph/pull/2328)) [@seunghwak](https://github.com/seunghwak)
- Remove legacy katz ([#2324](https://github.com/rapidsai/cugraph/pull/2324)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🐛 Bug Fixes

- Fix PropertyGraph MG tests ([#2511](https://github.com/rapidsai/cugraph/pull/2511)) [@eriknw](https://github.com/eriknw)
- Update `k_core.py` to Check for Graph Direction ([#2507](https://github.com/rapidsai/cugraph/pull/2507)) [@oorliu](https://github.com/oorliu)
- fix non-deterministic bug in uniform neighborhood sampling ([#2477](https://github.com/rapidsai/cugraph/pull/2477)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix typos in Python CMakeLists CUDA arch file ([#2475](https://github.com/rapidsai/cugraph/pull/2475)) [@vyasr](https://github.com/vyasr)
- Updated imports to be compatible with latest version of cupy ([#2473](https://github.com/rapidsai/cugraph/pull/2473)) [@rlratzel](https://github.com/rlratzel)
- Fix pandas SettingWithCopyWarning, which really shouldn&#39;t be ignored. ([#2447](https://github.com/rapidsai/cugraph/pull/2447)) [@eriknw](https://github.com/eriknw)
- fix handling of fanout == -1 ([#2435](https://github.com/rapidsai/cugraph/pull/2435)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add options to `extract_subgraph()` to bypass renumbering and adding edge_data, exclude internal `_WEIGHT_` column from `edge_property_names`, added `num_vertices_with_properties` attr ([#2419](https://github.com/rapidsai/cugraph/pull/2419)) [@rlratzel](https://github.com/rlratzel)
- Remove the comms import from cugraph&#39;s init file ([#2402](https://github.com/rapidsai/cugraph/pull/2402)) [@jnke2016](https://github.com/jnke2016)
- Bug fix (providing invalid sentinel value for cuCollection). ([#2382](https://github.com/rapidsai/cugraph/pull/2382)) [@seunghwak](https://github.com/seunghwak)
- add debug print for betweenness centrality, fix typo ([#2369](https://github.com/rapidsai/cugraph/pull/2369)) [@jnke2016](https://github.com/jnke2016)
- Bug fix for decompressing partial edge list and using (key, value) pairs for major properties. ([#2366](https://github.com/rapidsai/cugraph/pull/2366)) [@seunghwak](https://github.com/seunghwak)
- Fix Fanout -1 ([#2358](https://github.com/rapidsai/cugraph/pull/2358)) [@VibhuJawa](https://github.com/VibhuJawa)
- Update sampling primitive again, fix hypersparse computations ([#2353](https://github.com/rapidsai/cugraph/pull/2353)) [@ChuckHastings](https://github.com/ChuckHastings)
- added test cases and verified that algorithm works for undirected graphs ([#2349](https://github.com/rapidsai/cugraph/pull/2349)) [@acostadon](https://github.com/acostadon)
- Fix sampling bug ([#2343](https://github.com/rapidsai/cugraph/pull/2343)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix triangle count ([#2325](https://github.com/rapidsai/cugraph/pull/2325)) [@ChuckHastings](https://github.com/ChuckHastings)

## 📖 Documentation

- Defer loading of `custom.js` ([#2506](https://github.com/rapidsai/cugraph/pull/2506)) [@galipremsagar](https://github.com/galipremsagar)
- Centralize common `css` &amp; `js` code in docs ([#2472](https://github.com/rapidsai/cugraph/pull/2472)) [@galipremsagar](https://github.com/galipremsagar)
- Fix issues with day &amp; night modes in python docs ([#2471](https://github.com/rapidsai/cugraph/pull/2471)) [@galipremsagar](https://github.com/galipremsagar)
- Use Datasets API to Update Docstring Examples ([#2441](https://github.com/rapidsai/cugraph/pull/2441)) [@oorliu](https://github.com/oorliu)
- README updates ([#2395](https://github.com/rapidsai/cugraph/pull/2395)) [@BradReesWork](https://github.com/BradReesWork)
- Switch `language` from `None` to `&quot;en&quot;` in docs build ([#2368](https://github.com/rapidsai/cugraph/pull/2368)) [@galipremsagar](https://github.com/galipremsagar)
- Doxygen improvements to improve documentation of C API ([#2355](https://github.com/rapidsai/cugraph/pull/2355)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update multi-GPU example to include data generation ([#2345](https://github.com/rapidsai/cugraph/pull/2345)) [@charlesbluca](https://github.com/charlesbluca)

## 🚀 New Features

- Cost Matrix first version ([#2377](https://github.com/rapidsai/cugraph/pull/2377)) [@acostadon](https://github.com/acostadon)

## 🛠️ Improvements

- Pin `dask` &amp; `distributed` for release ([#2478](https://github.com/rapidsai/cugraph/pull/2478)) [@galipremsagar](https://github.com/galipremsagar)
- Update PageRank to leverage pylibcugraph ([#2467](https://github.com/rapidsai/cugraph/pull/2467)) [@jnke2016](https://github.com/jnke2016)
- Change default return type `PropertyGraph.extract_subgraph() -&gt; cugraph.Graph(directed=True)` ([#2460](https://github.com/rapidsai/cugraph/pull/2460)) [@eriknw](https://github.com/eriknw)
- Updates to Link Notebooks ([#2456](https://github.com/rapidsai/cugraph/pull/2456)) [@acostadon](https://github.com/acostadon)
- Only build cugraphmgtestutil when requested ([#2454](https://github.com/rapidsai/cugraph/pull/2454)) [@robertmaynard](https://github.com/robertmaynard)
- Datasets API Update: Add Extra Params and Improve Testing ([#2453](https://github.com/rapidsai/cugraph/pull/2453)) [@oorliu](https://github.com/oorliu)
- Uniform neighbor sample ([#2450](https://github.com/rapidsai/cugraph/pull/2450)) [@VibhuJawa](https://github.com/VibhuJawa)
- Don&#39;t store redundant columns in PropertyGraph Dataframes ([#2449](https://github.com/rapidsai/cugraph/pull/2449)) [@eriknw](https://github.com/eriknw)
- Changes to Cores, components and layout notebooks ([#2448](https://github.com/rapidsai/cugraph/pull/2448)) [@acostadon](https://github.com/acostadon)
- Added `get_vertex_data()` and `get_edge_data()` to SG/MG PropertyGraph ([#2444](https://github.com/rapidsai/cugraph/pull/2444)) [@rlratzel](https://github.com/rlratzel)
- Remove OpenMP dependencies from CMake ([#2443](https://github.com/rapidsai/cugraph/pull/2443)) [@seunghwak](https://github.com/seunghwak)
- Use Datasets API to Update Notebook Examples ([#2440](https://github.com/rapidsai/cugraph/pull/2440)) [@oorliu](https://github.com/oorliu)
- Refactor MG C++ tests (handle initialization) ([#2439](https://github.com/rapidsai/cugraph/pull/2439)) [@seunghwak](https://github.com/seunghwak)
- Branch 22.08 merge 22.06 ([#2436](https://github.com/rapidsai/cugraph/pull/2436)) [@rlratzel](https://github.com/rlratzel)
- Add get_num_vertices and get_num_edges methods to PropertyGraph. ([#2434](https://github.com/rapidsai/cugraph/pull/2434)) [@eriknw](https://github.com/eriknw)
- Make cuco a private dependency and leverage rapids-cmake ([#2432](https://github.com/rapidsai/cugraph/pull/2432)) [@vyasr](https://github.com/vyasr)
- cuGraph code cleanup ([#2431](https://github.com/rapidsai/cugraph/pull/2431)) [@seunghwak](https://github.com/seunghwak)
- Add core number to the python API ([#2414](https://github.com/rapidsai/cugraph/pull/2414)) [@jnke2016](https://github.com/jnke2016)
- Enable concurrent broadcasts in update_edge_partition_minor_property() ([#2413](https://github.com/rapidsai/cugraph/pull/2413)) [@seunghwak](https://github.com/seunghwak)
- Optimize has_duplicate_edges ([#2409](https://github.com/rapidsai/cugraph/pull/2409)) [@VibhuJawa](https://github.com/VibhuJawa)
- Define API for MG random walk ([#2407](https://github.com/rapidsai/cugraph/pull/2407)) [@ChuckHastings](https://github.com/ChuckHastings)
- Support building without cugraph-ops ([#2405](https://github.com/rapidsai/cugraph/pull/2405)) [@ChuckHastings](https://github.com/ChuckHastings)
- Clean up public api ([#2398](https://github.com/rapidsai/cugraph/pull/2398)) [@ChuckHastings](https://github.com/ChuckHastings)
- Community notebook updates structure/testing/improvement ([#2397](https://github.com/rapidsai/cugraph/pull/2397)) [@acostadon](https://github.com/acostadon)
- Run relevant CI tests based on what&#39;s changed in the ChangeList ([#2396](https://github.com/rapidsai/cugraph/pull/2396)) [@anandhkb](https://github.com/anandhkb)
- Update `Graph` to store a Pylibcugraph Graph (SG/MG Graph) ([#2394](https://github.com/rapidsai/cugraph/pull/2394)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Moving Centrality notebooks to new structure and updating/testing ([#2388](https://github.com/rapidsai/cugraph/pull/2388)) [@acostadon](https://github.com/acostadon)
- Add conda compilers to env file ([#2384](https://github.com/rapidsai/cugraph/pull/2384)) [@vyasr](https://github.com/vyasr)
- Add get_node_storage and get_edge_storage to CuGraphStorage ([#2381](https://github.com/rapidsai/cugraph/pull/2381)) [@VibhuJawa](https://github.com/VibhuJawa)
- Pin max version of `cuda-python` to `11.7.0` ([#2380](https://github.com/rapidsai/cugraph/pull/2380)) [@Ethyling](https://github.com/Ethyling)
- Update cugraph python build ([#2378](https://github.com/rapidsai/cugraph/pull/2378)) [@jnke2016](https://github.com/jnke2016)
- Delete old nbr sampling software ([#2371](https://github.com/rapidsai/cugraph/pull/2371)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add datasets API to import graph data from configuration/metadata files ([#2367](https://github.com/rapidsai/cugraph/pull/2367)) [@betochimas](https://github.com/betochimas)
- Skip reduction for zero (in|out-)degree vertices. ([#2365](https://github.com/rapidsai/cugraph/pull/2365)) [@seunghwak](https://github.com/seunghwak)
- Update Python version support. ([#2363](https://github.com/rapidsai/cugraph/pull/2363)) [@bdice](https://github.com/bdice)
- Branch 22.08 merge 22.06 ([#2362](https://github.com/rapidsai/cugraph/pull/2362)) [@rlratzel](https://github.com/rlratzel)
- Support raft updating to new version of cuco ([#2360](https://github.com/rapidsai/cugraph/pull/2360)) [@ChuckHastings](https://github.com/ChuckHastings)
- Branch 22.08 merge 22.06 ([#2359](https://github.com/rapidsai/cugraph/pull/2359)) [@rlratzel](https://github.com/rlratzel)
- Remove topology header ([#2357](https://github.com/rapidsai/cugraph/pull/2357)) [@ChuckHastings](https://github.com/ChuckHastings)
- Switch back to PC generator ([#2356](https://github.com/rapidsai/cugraph/pull/2356)) [@ChuckHastings](https://github.com/ChuckHastings)
- Remove GraphCSC/GraphCSCView object, no longer used ([#2354](https://github.com/rapidsai/cugraph/pull/2354)) [@ChuckHastings](https://github.com/ChuckHastings)
- Resolve Forward merging of branch-22.06 into branch-22.08 ([#2350](https://github.com/rapidsai/cugraph/pull/2350)) [@jnke2016](https://github.com/jnke2016)
- Replace raw pointers with device_span in induced subgraph ([#2348](https://github.com/rapidsai/cugraph/pull/2348)) [@yang-hu-nv](https://github.com/yang-hu-nv)
- Some legacy BFS cleanup ([#2347](https://github.com/rapidsai/cugraph/pull/2347)) [@ChuckHastings](https://github.com/ChuckHastings)
- Remove legacy sssp implementation ([#2344](https://github.com/rapidsai/cugraph/pull/2344)) [@ChuckHastings](https://github.com/ChuckHastings)
- Unpin `dask` &amp; `distributed` for development ([#2342](https://github.com/rapidsai/cugraph/pull/2342)) [@galipremsagar](https://github.com/galipremsagar)
- Release notebook: Nx Generators &amp; Adding Perf_counter ([#2341](https://github.com/rapidsai/cugraph/pull/2341)) [@oorliu](https://github.com/oorliu)
- Clean up some unused code in the C API (and beyond) ([#2339](https://github.com/rapidsai/cugraph/pull/2339)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add core number to the C API ([#2338](https://github.com/rapidsai/cugraph/pull/2338)) [@betochimas](https://github.com/betochimas)
- Update the list of algos to benchmark ([#2337](https://github.com/rapidsai/cugraph/pull/2337)) [@jnke2016](https://github.com/jnke2016)
- Default GPU_COUNT to 1 in cmake file ([#2336](https://github.com/rapidsai/cugraph/pull/2336)) [@ChuckHastings](https://github.com/ChuckHastings)
- DOC Fix for Renumber-2.ipynb ([#2335](https://github.com/rapidsai/cugraph/pull/2335)) [@oorliu](https://github.com/oorliu)
- Resolve conflicts for merge from branch-22.06 to branch-22.08 ([#2334](https://github.com/rapidsai/cugraph/pull/2334)) [@rlratzel](https://github.com/rlratzel)
- update versions to 22.08 ([#2332](https://github.com/rapidsai/cugraph/pull/2332)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix experimental labels ([#2331](https://github.com/rapidsai/cugraph/pull/2331)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Performance-optimize storing edge partition source/destination properties in (key, value) pairs ([#2328](https://github.com/rapidsai/cugraph/pull/2328)) [@seunghwak](https://github.com/seunghwak)
- Remove legacy katz ([#2324](https://github.com/rapidsai/cugraph/pull/2324)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add missing Thrust includes ([#2310](https://github.com/rapidsai/cugraph/pull/2310)) [@bdice](https://github.com/bdice)

# cuGraph 22.06.00 (7 Jun 2022)

## 🚨 Breaking Changes

- Fix uniform neighborhood sampling remove duplicates ([#2301](https://github.com/rapidsai/cugraph/pull/2301)) [@ChuckHastings](https://github.com/ChuckHastings)
- Split update_v_frontier_from_outgoing_e to two simpler primitives ([#2290](https://github.com/rapidsai/cugraph/pull/2290)) [@seunghwak](https://github.com/seunghwak)
- Refactor MG neighborhood sampling and add SG implementation ([#2285](https://github.com/rapidsai/cugraph/pull/2285)) [@jnke2016](https://github.com/jnke2016)
- Resolve inconsistencies in reduction support in primitives ([#2257](https://github.com/rapidsai/cugraph/pull/2257)) [@seunghwak](https://github.com/seunghwak)
- Revert SG Katz API&#39;s signature to previous &lt;22.04 version ([#2242](https://github.com/rapidsai/cugraph/pull/2242)) [@betochimas](https://github.com/betochimas)
- Rename primitive functions. ([#2234](https://github.com/rapidsai/cugraph/pull/2234)) [@seunghwak](https://github.com/seunghwak)
- Graph primitives API updates ([#2220](https://github.com/rapidsai/cugraph/pull/2220)) [@seunghwak](https://github.com/seunghwak)
- Add Katz Centrality to pylibcugraph, refactor Katz Centrality for cugraph ([#2201](https://github.com/rapidsai/cugraph/pull/2201)) [@betochimas](https://github.com/betochimas)
- Update graph/graph primitives API to consistently use vertex/edge centric terminologies instead of matrix centric terminolgies ([#2187](https://github.com/rapidsai/cugraph/pull/2187)) [@seunghwak](https://github.com/seunghwak)
- Define C API for eigenvector centrality ([#2180](https://github.com/rapidsai/cugraph/pull/2180)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🐛 Bug Fixes

- fix sampling handling of dscr region ([#2321](https://github.com/rapidsai/cugraph/pull/2321)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add test to reproduce issue with double weights, fix issue (graph cre… ([#2305](https://github.com/rapidsai/cugraph/pull/2305)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix MG BFS through C API ([#2291](https://github.com/rapidsai/cugraph/pull/2291)) [@ChuckHastings](https://github.com/ChuckHastings)
- fixes BUG 2275 ([#2279](https://github.com/rapidsai/cugraph/pull/2279)) [@BradReesWork](https://github.com/BradReesWork)
- Refactored SG `hits` and MG `katz_centrality` ([#2276](https://github.com/rapidsai/cugraph/pull/2276)) [@betochimas](https://github.com/betochimas)
- Multi-GPU reduce_v &amp; transform_reduce_v bug fix. ([#2269](https://github.com/rapidsai/cugraph/pull/2269)) [@seunghwak](https://github.com/seunghwak)
- Update BFS and SSSP to check start/source vertex for validity ([#2268](https://github.com/rapidsai/cugraph/pull/2268)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Update some clustering algos to only support undirected graphs ([#2267](https://github.com/rapidsai/cugraph/pull/2267)) [@jnke2016](https://github.com/jnke2016)
- Resolves maximum spanning tree bug when using Edgelist instead of Adjlist ([#2256](https://github.com/rapidsai/cugraph/pull/2256)) [@betochimas](https://github.com/betochimas)
- cudf moved the default_hash into the cudf::detail namespace ([#2244](https://github.com/rapidsai/cugraph/pull/2244)) [@ChuckHastings](https://github.com/ChuckHastings)
- Allow `cugraph` to be imported in an SG env for SG algorithms ([#2241](https://github.com/rapidsai/cugraph/pull/2241)) [@betochimas](https://github.com/betochimas)
- Address some MNMG issues in cython.cu ([#2224](https://github.com/rapidsai/cugraph/pull/2224)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix error from two conflicting merges ([#2219](https://github.com/rapidsai/cugraph/pull/2219)) [@ChuckHastings](https://github.com/ChuckHastings)
- Branch 22.06 MNMG bug work and support for Undirected Graphs ([#2215](https://github.com/rapidsai/cugraph/pull/2215)) [@acostadon](https://github.com/acostadon)
- Branch 22.06 merge 22.04 ([#2190](https://github.com/rapidsai/cugraph/pull/2190)) [@rlratzel](https://github.com/rlratzel)

## 📖 Documentation

- Fix BFS Docstring ([#2318](https://github.com/rapidsai/cugraph/pull/2318)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- small typo ([#2250](https://github.com/rapidsai/cugraph/pull/2250)) [@hoosierEE](https://github.com/hoosierEE)
- Updating issue template and missing docs ([#2211](https://github.com/rapidsai/cugraph/pull/2211)) [@BradReesWork](https://github.com/BradReesWork)
- Python code cleanup across docs, wrappers, testing ([#2194](https://github.com/rapidsai/cugraph/pull/2194)) [@betochimas](https://github.com/betochimas)

## 🚀 New Features

- Multi GPU Property Graph with basic creation support ([#2286](https://github.com/rapidsai/cugraph/pull/2286)) [@acostadon](https://github.com/acostadon)
- Triangle Counting ([#2253](https://github.com/rapidsai/cugraph/pull/2253)) [@seunghwak](https://github.com/seunghwak)
- Triangle Counts C++ API ([#2233](https://github.com/rapidsai/cugraph/pull/2233)) [@seunghwak](https://github.com/seunghwak)
- Define C API for eigenvector centrality ([#2180](https://github.com/rapidsai/cugraph/pull/2180)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🛠️ Improvements

- Pin `dask` and `distributed` for release ([#2317](https://github.com/rapidsai/cugraph/pull/2317)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` &amp; `distributed` for release ([#2312](https://github.com/rapidsai/cugraph/pull/2312)) [@galipremsagar](https://github.com/galipremsagar)
- Triangle counting C API implementation ([#2302](https://github.com/rapidsai/cugraph/pull/2302)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix uniform neighborhood sampling remove duplicates ([#2301](https://github.com/rapidsai/cugraph/pull/2301)) [@ChuckHastings](https://github.com/ChuckHastings)
- Migrate SG and MG SSSP to pylibcugraph ([#2295](https://github.com/rapidsai/cugraph/pull/2295)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Add Louvain to the C API ([#2292](https://github.com/rapidsai/cugraph/pull/2292)) [@ChuckHastings](https://github.com/ChuckHastings)
- Split update_v_frontier_from_outgoing_e to two simpler primitives ([#2290](https://github.com/rapidsai/cugraph/pull/2290)) [@seunghwak](https://github.com/seunghwak)
- Add and test mechanism for creating graph with edge index as weight ([#2288](https://github.com/rapidsai/cugraph/pull/2288)) [@ChuckHastings](https://github.com/ChuckHastings)
- Implement eigenvector centrality ([#2287](https://github.com/rapidsai/cugraph/pull/2287)) [@ChuckHastings](https://github.com/ChuckHastings)
- Refactor MG neighborhood sampling and add SG implementation ([#2285](https://github.com/rapidsai/cugraph/pull/2285)) [@jnke2016](https://github.com/jnke2016)
- Migrate SG and MG BFS to pylibcugraph ([#2284](https://github.com/rapidsai/cugraph/pull/2284)) [@alexbarghi-nv](https://github.com/alexbarghi-nv)
- Optimize Sampling for graph_store ([#2283](https://github.com/rapidsai/cugraph/pull/2283)) [@VibhuJawa](https://github.com/VibhuJawa)
- Refactor mg symmetrize tests ([#2278](https://github.com/rapidsai/cugraph/pull/2278)) [@jnke2016](https://github.com/jnke2016)
- Add do_expensive_check to graph primitives ([#2274](https://github.com/rapidsai/cugraph/pull/2274)) [@seunghwak](https://github.com/seunghwak)
- add bindings for triangle counting ([#2273](https://github.com/rapidsai/cugraph/pull/2273)) [@jnke2016](https://github.com/jnke2016)
- Define triangle_count C API ([#2271](https://github.com/rapidsai/cugraph/pull/2271)) [@ChuckHastings](https://github.com/ChuckHastings)
- Revert old pattern of SG cugraph testing for CI purposes ([#2262](https://github.com/rapidsai/cugraph/pull/2262)) [@betochimas](https://github.com/betochimas)
- Branch 22.06 bug fixes + update imports ([#2261](https://github.com/rapidsai/cugraph/pull/2261)) [@betochimas](https://github.com/betochimas)
- Raft RNG updated API ([#2260](https://github.com/rapidsai/cugraph/pull/2260)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Add Degree Centrality to cugraph ([#2259](https://github.com/rapidsai/cugraph/pull/2259)) [@betochimas](https://github.com/betochimas)
- Refactor Uniform Neighborhood Sampling ([#2258](https://github.com/rapidsai/cugraph/pull/2258)) [@ChuckHastings](https://github.com/ChuckHastings)
- Resolve inconsistencies in reduction support in primitives ([#2257](https://github.com/rapidsai/cugraph/pull/2257)) [@seunghwak](https://github.com/seunghwak)
- Add Eigenvector Centrality to pylibcugraph, cugraph APIs ([#2255](https://github.com/rapidsai/cugraph/pull/2255)) [@betochimas](https://github.com/betochimas)
- Add MG Hits and MG Neighborhood_sampling to benchmarks ([#2254](https://github.com/rapidsai/cugraph/pull/2254)) [@jnke2016](https://github.com/jnke2016)
- Undirected graph support for MG graphs ([#2247](https://github.com/rapidsai/cugraph/pull/2247)) [@jnke2016](https://github.com/jnke2016)
- Branch 22.06 bugs ([#2245](https://github.com/rapidsai/cugraph/pull/2245)) [@BradReesWork](https://github.com/BradReesWork)
- Revert SG Katz API&#39;s signature to previous &lt;22.04 version ([#2242](https://github.com/rapidsai/cugraph/pull/2242)) [@betochimas](https://github.com/betochimas)
- add API for the new uniform neighborhood sampling ([#2236](https://github.com/rapidsai/cugraph/pull/2236)) [@ChuckHastings](https://github.com/ChuckHastings)
- Reverting raft pinned tag ([#2235](https://github.com/rapidsai/cugraph/pull/2235)) [@cjnolet](https://github.com/cjnolet)
- Rename primitive functions. ([#2234](https://github.com/rapidsai/cugraph/pull/2234)) [@seunghwak](https://github.com/seunghwak)
- Moves pylibcugraph APIS from 22.04 and earlier out of `experimental` namespace ([#2232](https://github.com/rapidsai/cugraph/pull/2232)) [@betochimas](https://github.com/betochimas)
- Use conda to build python packages during GPU tests ([#2230](https://github.com/rapidsai/cugraph/pull/2230)) [@Ethyling](https://github.com/Ethyling)
- Fix typos in documentation ([#2225](https://github.com/rapidsai/cugraph/pull/2225)) [@seunghwak](https://github.com/seunghwak)
- Update CMake pinning to allow newer CMake versions. ([#2221](https://github.com/rapidsai/cugraph/pull/2221)) [@vyasr](https://github.com/vyasr)
- Graph primitives API updates ([#2220](https://github.com/rapidsai/cugraph/pull/2220)) [@seunghwak](https://github.com/seunghwak)
- Enable MG support for small datasets ([#2216](https://github.com/rapidsai/cugraph/pull/2216)) [@jnke2016](https://github.com/jnke2016)
- Unpin `dask` &amp; `distributed` for devlopment ([#2214](https://github.com/rapidsai/cugraph/pull/2214)) [@galipremsagar](https://github.com/galipremsagar)
- updated MG Test code to not use DiGraph ([#2213](https://github.com/rapidsai/cugraph/pull/2213)) [@BradReesWork](https://github.com/BradReesWork)
- renaming detail space functions ([#2212](https://github.com/rapidsai/cugraph/pull/2212)) [@seunghwak](https://github.com/seunghwak)
- Make diagram and caption consistent in Pagerank.ipynb ([#2207](https://github.com/rapidsai/cugraph/pull/2207)) [@charlesbluca](https://github.com/charlesbluca)
- Add Katz Centrality to pylibcugraph, refactor Katz Centrality for cugraph ([#2201](https://github.com/rapidsai/cugraph/pull/2201)) [@betochimas](https://github.com/betochimas)
- Resolve Forward merging of branch-22.04 into branch-22.06 ([#2197](https://github.com/rapidsai/cugraph/pull/2197)) [@jnke2016](https://github.com/jnke2016)
- Add Katz Centrality to the C API ([#2192](https://github.com/rapidsai/cugraph/pull/2192)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update graph/graph primitives API to consistently use vertex/edge centric terminologies instead of matrix centric terminolgies ([#2187](https://github.com/rapidsai/cugraph/pull/2187)) [@seunghwak](https://github.com/seunghwak)
- Labeling algorithm updates for C API ([#2185](https://github.com/rapidsai/cugraph/pull/2185)) [@ChuckHastings](https://github.com/ChuckHastings)
- Added GraphStore Function ([#2183](https://github.com/rapidsai/cugraph/pull/2183)) [@wangxiaoyunNV](https://github.com/wangxiaoyunNV)
- Enable building static libs ([#2179](https://github.com/rapidsai/cugraph/pull/2179)) [@trxcllnt](https://github.com/trxcllnt)
- Fix merge conflicts ([#2155](https://github.com/rapidsai/cugraph/pull/2155)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove unused code (gunrock HITS) ([#2152](https://github.com/rapidsai/cugraph/pull/2152)) [@seunghwak](https://github.com/seunghwak)
- Turn off cuco dependency in RAFT. Re-establish explicit `cuco` and `libcuxx` cmake dependencies ([#2132](https://github.com/rapidsai/cugraph/pull/2132)) [@cjnolet](https://github.com/cjnolet)
- Consolidate C++ conda recipes and add `libcugraph-tests` package ([#2124](https://github.com/rapidsai/cugraph/pull/2124)) [@Ethyling](https://github.com/Ethyling)
- Use conda compilers ([#2101](https://github.com/rapidsai/cugraph/pull/2101)) [@Ethyling](https://github.com/Ethyling)
- Use mamba to build packages ([#2051](https://github.com/rapidsai/cugraph/pull/2051)) [@Ethyling](https://github.com/Ethyling)

# cuGraph 22.04.00 (6 Apr 2022)

## 🚨 Breaking Changes

- Remove major/minor from renumber_edgelist public functions. ([#2116](https://github.com/rapidsai/cugraph/pull/2116)) [@seunghwak](https://github.com/seunghwak)
- Add MG support to the C API ([#2110](https://github.com/rapidsai/cugraph/pull/2110)) [@ChuckHastings](https://github.com/ChuckHastings)
- Graph prmitives API update ([#2100](https://github.com/rapidsai/cugraph/pull/2100)) [@seunghwak](https://github.com/seunghwak)
- Reduce peak memory requirement in graph creation (part 1/2) ([#2070](https://github.com/rapidsai/cugraph/pull/2070)) [@seunghwak](https://github.com/seunghwak)

## 🐛 Bug Fixes

- Pin cmake in conda recipe to &lt;3.23 ([#2176](https://github.com/rapidsai/cugraph/pull/2176)) [@dantegd](https://github.com/dantegd)
- Remove unused cython code referencing RAFT APIs that are no longer present ([#2125](https://github.com/rapidsai/cugraph/pull/2125)) [@rlratzel](https://github.com/rlratzel)
- Add pylibcugraph as a run dep to the cugraph conda package ([#2121](https://github.com/rapidsai/cugraph/pull/2121)) [@rlratzel](https://github.com/rlratzel)
- update_frontier_v_push_if_out_nbr C++ test bug fix ([#2097](https://github.com/rapidsai/cugraph/pull/2097)) [@seunghwak](https://github.com/seunghwak)
- extract_if_e bug fix. ([#2096](https://github.com/rapidsai/cugraph/pull/2096)) [@seunghwak](https://github.com/seunghwak)
- Fix bug Random Walk in array sizes ([#2089](https://github.com/rapidsai/cugraph/pull/2089)) [@ChuckHastings](https://github.com/ChuckHastings)
- Coarsening symmetric graphs leads to slightly asymmetric edge weights ([#2080](https://github.com/rapidsai/cugraph/pull/2080)) [@seunghwak](https://github.com/seunghwak)
- Skips ktruss docstring example for CUDA version 11.4 ([#2074](https://github.com/rapidsai/cugraph/pull/2074)) [@betochimas](https://github.com/betochimas)
- Branch 22.04 merge 22.02 ([#2072](https://github.com/rapidsai/cugraph/pull/2072)) [@rlratzel](https://github.com/rlratzel)
- MG Louvain C++ test R-mat usecase parameters ([#2061](https://github.com/rapidsai/cugraph/pull/2061)) [@seunghwak](https://github.com/seunghwak)
- Updates to enable NumberMap to generate unique src/dst column names ([#2050](https://github.com/rapidsai/cugraph/pull/2050)) [@rlratzel](https://github.com/rlratzel)
- Allow class types to be properly represented in the `experimental_warning_wrapper()` return value ([#2048](https://github.com/rapidsai/cugraph/pull/2048)) [@rlratzel](https://github.com/rlratzel)
- Improve MG graph creation ([#2044](https://github.com/rapidsai/cugraph/pull/2044)) [@seunghwak](https://github.com/seunghwak)

## 📖 Documentation

- 22.04 Update docs ([#2171](https://github.com/rapidsai/cugraph/pull/2171)) [@BradReesWork](https://github.com/BradReesWork)
- Corrected image in Hits notebook so right node was highlighted. Issue 2079 ([#2106](https://github.com/rapidsai/cugraph/pull/2106)) [@acostadon](https://github.com/acostadon)
- API Doc Namespace Edits + SimpleGraphImpl methods ([#2086](https://github.com/rapidsai/cugraph/pull/2086)) [@betochimas](https://github.com/betochimas)

## 🚀 New Features

- Gather one hop neighbors ([#2117](https://github.com/rapidsai/cugraph/pull/2117)) [@kaatish](https://github.com/kaatish)
- Define the uniform neighbor sampling C API ([#2112](https://github.com/rapidsai/cugraph/pull/2112)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add `node2vec` wrapper to cugraph ([#2093](https://github.com/rapidsai/cugraph/pull/2093)) [@betochimas](https://github.com/betochimas)
- Add `node2vec` wrappers to pylibcugraph ([#2085](https://github.com/rapidsai/cugraph/pull/2085)) [@betochimas](https://github.com/betochimas)
- Multi gpu sample edges utilities ([#2064](https://github.com/rapidsai/cugraph/pull/2064)) [@kaatish](https://github.com/kaatish)
- add libcugraphops as a dependency of cugraph ([#2019](https://github.com/rapidsai/cugraph/pull/2019)) [@MatthiasKohl](https://github.com/MatthiasKohl)

## 🛠️ Improvements

- Updated random_walk_benchmark notebook for API change in cudf ([#2164](https://github.com/rapidsai/cugraph/pull/2164)) [@mmccarty](https://github.com/mmccarty)
- Neighborhood sampling C API implementation ([#2156](https://github.com/rapidsai/cugraph/pull/2156)) [@ChuckHastings](https://github.com/ChuckHastings)
- Enhancement on uniform random sampling of indices near zero. ([#2153](https://github.com/rapidsai/cugraph/pull/2153)) [@aschaffer](https://github.com/aschaffer)
- Temporarily disable new `ops-bot` functionality ([#2151](https://github.com/rapidsai/cugraph/pull/2151)) [@ajschmidt8](https://github.com/ajschmidt8)
- HITS C API implementation ([#2150](https://github.com/rapidsai/cugraph/pull/2150)) [@ChuckHastings](https://github.com/ChuckHastings)
- Use `rapids_find_package` to get `cugraph-ops` ([#2148](https://github.com/rapidsai/cugraph/pull/2148)) [@trxcllnt](https://github.com/trxcllnt)
- Pin `dask` and `distributed` versions ([#2147](https://github.com/rapidsai/cugraph/pull/2147)) [@galipremsagar](https://github.com/galipremsagar)
- Pin gtest/gmock to 1.10.0 in dev envs ([#2127](https://github.com/rapidsai/cugraph/pull/2127)) [@trxcllnt](https://github.com/trxcllnt)
- Add HITS to the C API ([#2123](https://github.com/rapidsai/cugraph/pull/2123)) [@ChuckHastings](https://github.com/ChuckHastings)
- node2vec Python wrapper API changes and refactoring, with improved testing coverage ([#2120](https://github.com/rapidsai/cugraph/pull/2120)) [@betochimas](https://github.com/betochimas)
- Add MG neighborhood sampling to pylibcugraph &amp; cugraph APIs ([#2118](https://github.com/rapidsai/cugraph/pull/2118)) [@betochimas](https://github.com/betochimas)
- Remove major/minor from renumber_edgelist public functions. ([#2116](https://github.com/rapidsai/cugraph/pull/2116)) [@seunghwak](https://github.com/seunghwak)
- Upgrade `dask` and `distributed` ([#2115](https://github.com/rapidsai/cugraph/pull/2115)) [@galipremsagar](https://github.com/galipremsagar)
- Remove references to gmock ([#2114](https://github.com/rapidsai/cugraph/pull/2114)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add `.github/ops-bot.yaml` config file ([#2111](https://github.com/rapidsai/cugraph/pull/2111)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add MG support to the C API ([#2110](https://github.com/rapidsai/cugraph/pull/2110)) [@ChuckHastings](https://github.com/ChuckHastings)
- Graph prmitives API update ([#2100](https://github.com/rapidsai/cugraph/pull/2100)) [@seunghwak](https://github.com/seunghwak)
- Nx compatibility based on making Graph subclass and calling Cugraph algos ([#2099](https://github.com/rapidsai/cugraph/pull/2099)) [@acostadon](https://github.com/acostadon)
- Fix cugraph-ops header names ([#2095](https://github.com/rapidsai/cugraph/pull/2095)) [@kaatish](https://github.com/kaatish)
- Updating a few headers that have been renamed in raft ([#2090](https://github.com/rapidsai/cugraph/pull/2090)) [@cjnolet](https://github.com/cjnolet)
- Add MG wrapper for HITS ([#2088](https://github.com/rapidsai/cugraph/pull/2088)) [@jnke2016](https://github.com/jnke2016)
- Automatically clone raft when the raft pinned tag changes ([#2087](https://github.com/rapidsai/cugraph/pull/2087)) [@cjnolet](https://github.com/cjnolet)
- updated release performance notebook to also measure using Nx as imput ([#2083](https://github.com/rapidsai/cugraph/pull/2083)) [@BradReesWork](https://github.com/BradReesWork)
- Reduce peak memory requirement in graph creation (part 2/2) ([#2081](https://github.com/rapidsai/cugraph/pull/2081)) [@seunghwak](https://github.com/seunghwak)
- C API code cleanup ([#2077](https://github.com/rapidsai/cugraph/pull/2077)) [@ChuckHastings](https://github.com/ChuckHastings)
- Remove usage of RAFT memory management ([#2076](https://github.com/rapidsai/cugraph/pull/2076)) [@viclafargue](https://github.com/viclafargue)
- MNMG Neighborhood Sampling ([#2073](https://github.com/rapidsai/cugraph/pull/2073)) [@aschaffer](https://github.com/aschaffer)
- Allow PropertyGraph `default_edge_weight` to be used to add an edge weight value on extracted Graphs even when a weight property wasn&#39;t specified ([#2071](https://github.com/rapidsai/cugraph/pull/2071)) [@rlratzel](https://github.com/rlratzel)
- Reduce peak memory requirement in graph creation (part 1/2) ([#2070](https://github.com/rapidsai/cugraph/pull/2070)) [@seunghwak](https://github.com/seunghwak)
- add node2vec C API implementation ([#2069](https://github.com/rapidsai/cugraph/pull/2069)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fixing cugraph for RAFT spectral/lap API changes ([#2067](https://github.com/rapidsai/cugraph/pull/2067)) [@cjnolet](https://github.com/cjnolet)
- remove unused spmv functions ([#2066](https://github.com/rapidsai/cugraph/pull/2066)) [@ChuckHastings](https://github.com/ChuckHastings)
- Improve MG Louvain scalability ([#2062](https://github.com/rapidsai/cugraph/pull/2062)) [@seunghwak](https://github.com/seunghwak)
- Added `pylibcugraph` utility for setting up return array values ([#2060](https://github.com/rapidsai/cugraph/pull/2060)) [@rlratzel](https://github.com/rlratzel)
- Add node2vec to C API - API PR ([#2059](https://github.com/rapidsai/cugraph/pull/2059)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add CMake `install` rules for tests ([#2057](https://github.com/rapidsai/cugraph/pull/2057)) [@ajschmidt8](https://github.com/ajschmidt8)
- PropertyGraph updates: added features for DGL, improved `extract_subgraph()` and `num_vertices` performance ([#2056](https://github.com/rapidsai/cugraph/pull/2056)) [@rlratzel](https://github.com/rlratzel)
- Update C++ SG and MG Louvain tests to support Rmat and benchmark tests ([#2054](https://github.com/rapidsai/cugraph/pull/2054)) [@ChuckHastings](https://github.com/ChuckHastings)
- Unpin max `dask` and `distributed` versions ([#2053](https://github.com/rapidsai/cugraph/pull/2053)) [@galipremsagar](https://github.com/galipremsagar)
- Removal of remaining DiGraph Python mentions ([#2049](https://github.com/rapidsai/cugraph/pull/2049)) [@betochimas](https://github.com/betochimas)
- Dgl graph store ([#2046](https://github.com/rapidsai/cugraph/pull/2046)) [@BradReesWork](https://github.com/BradReesWork)
- replace `ccache` with `sccache` ([#2045](https://github.com/rapidsai/cugraph/pull/2045)) [@AyodeAwe](https://github.com/AyodeAwe)
- Fix Merge Conflicts for `2024` ([#2040](https://github.com/rapidsai/cugraph/pull/2040)) [@ajschmidt8](https://github.com/ajschmidt8)
- Improve MG PageRank scalability ([#2038](https://github.com/rapidsai/cugraph/pull/2038)) [@seunghwak](https://github.com/seunghwak)
- Created initial list of simple Graph creation tests for nx compatibility ([#2035](https://github.com/rapidsai/cugraph/pull/2035)) [@acostadon](https://github.com/acostadon)
- neighbor sampling in COO/CSR format ([#1982](https://github.com/rapidsai/cugraph/pull/1982)) [@MatthiasKohl](https://github.com/MatthiasKohl)

# cuGraph 22.02.00 (2 Feb 2022)

## 🐛 Bug Fixes

- Always upload libcugraph ([#2041](https://github.com/rapidsai/cugraph/pull/2041)) [@raydouglass](https://github.com/raydouglass)
- Fix Louvain hang in multi-GPU testing ([#2028](https://github.com/rapidsai/cugraph/pull/2028)) [@seunghwak](https://github.com/seunghwak)
- fix bug when calculating the number of vertices ([#1992](https://github.com/rapidsai/cugraph/pull/1992)) [@jnke2016](https://github.com/jnke2016)
- update cuda 11.5 configuration to use clang format 11.1.0 ([#1990](https://github.com/rapidsai/cugraph/pull/1990)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update version in libcugraph_etl CMakeLists.txt to 22.02.00 to match libcugraph ([#1966](https://github.com/rapidsai/cugraph/pull/1966)) [@rlratzel](https://github.com/rlratzel)

## 📖 Documentation

- Initial automated doctest, all current examples now pass, other documentation edits ([#2014](https://github.com/rapidsai/cugraph/pull/2014)) [@betochimas](https://github.com/betochimas)
- Fix README example ([#1981](https://github.com/rapidsai/cugraph/pull/1981)) [@gitbuda](https://github.com/gitbuda)

## 🚀 New Features

- Add SSSP API, test and implementation ([#2016](https://github.com/rapidsai/cugraph/pull/2016)) [@ChuckHastings](https://github.com/ChuckHastings)
- Propose extract_bfs_paths C API ([#1955](https://github.com/rapidsai/cugraph/pull/1955)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🛠️ Improvements

- Do not build CUDA libs in Python jobs ([#2039](https://github.com/rapidsai/cugraph/pull/2039)) [@Ethyling](https://github.com/Ethyling)
- updated for release 22.02 ([#2034](https://github.com/rapidsai/cugraph/pull/2034)) [@BradReesWork](https://github.com/BradReesWork)
- Fix raft git ref ([#2032](https://github.com/rapidsai/cugraph/pull/2032)) [@Ethyling](https://github.com/Ethyling)
- Pin `dask` &amp; `distributed` ([#2031](https://github.com/rapidsai/cugraph/pull/2031)) [@galipremsagar](https://github.com/galipremsagar)
- Fix build script ([#2029](https://github.com/rapidsai/cugraph/pull/2029)) [@Ethyling](https://github.com/Ethyling)
- Prepare upload scripts for Python 3.7 removal ([#2027](https://github.com/rapidsai/cugraph/pull/2027)) [@Ethyling](https://github.com/Ethyling)
- Python API updates to enable explicit control of internal `graph_t` creation and deletion ([#2023](https://github.com/rapidsai/cugraph/pull/2023)) [@rlratzel](https://github.com/rlratzel)
- Updated build.sh help text and test execution steps in SOURCEBUILD.md ([#2020](https://github.com/rapidsai/cugraph/pull/2020)) [@acostadon](https://github.com/acostadon)
- Removed unused CI files ([#2017](https://github.com/rapidsai/cugraph/pull/2017)) [@rlratzel](https://github.com/rlratzel)
- Unpin `dask` and `distributed` ([#2010](https://github.com/rapidsai/cugraph/pull/2010)) [@galipremsagar](https://github.com/galipremsagar)
- Fix call to `getDeviceAttribute` following API change in RMM. ([#2008](https://github.com/rapidsai/cugraph/pull/2008)) [@shwina](https://github.com/shwina)
- drop fa2 cpu code ([#2007](https://github.com/rapidsai/cugraph/pull/2007)) [@BradReesWork](https://github.com/BradReesWork)
- Branch 22.02 merge 21.12 ([#2002](https://github.com/rapidsai/cugraph/pull/2002)) [@rlratzel](https://github.com/rlratzel)
- Update references to CHECK_CUDA, CUDA_CHECK and CUDA_TRY to use new RAFT_ names ([#2000](https://github.com/rapidsai/cugraph/pull/2000)) [@ChuckHastings](https://github.com/ChuckHastings)
- Initial PropertyGraph implementation and tests ([#1999](https://github.com/rapidsai/cugraph/pull/1999)) [@rlratzel](https://github.com/rlratzel)
- Fix optional and cstddef includes ([#1998](https://github.com/rapidsai/cugraph/pull/1998)) [@gitbuda](https://github.com/gitbuda)
- Add optimized 2x string column renumbering code ([#1996](https://github.com/rapidsai/cugraph/pull/1996)) [@chirayuG-nvidia](https://github.com/chirayuG-nvidia)
- Pass RMM memory allocator to cuco ([#1994](https://github.com/rapidsai/cugraph/pull/1994)) [@seunghwak](https://github.com/seunghwak)
- Add missing imports tests ([#1993](https://github.com/rapidsai/cugraph/pull/1993)) [@Ethyling](https://github.com/Ethyling)
- Update ucx-py version on release using rvc ([#1991](https://github.com/rapidsai/cugraph/pull/1991)) [@Ethyling](https://github.com/Ethyling)
- make C++ tests run faster (fewer tests) ([#1989](https://github.com/rapidsai/cugraph/pull/1989)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update the update_frontier_v_push_if_out_nbr primitive &amp; BFS performance ([#1988](https://github.com/rapidsai/cugraph/pull/1988)) [@seunghwak](https://github.com/seunghwak)
- Remove `IncludeCategories` from `.clang-format` ([#1987](https://github.com/rapidsai/cugraph/pull/1987)) [@codereport](https://github.com/codereport)
- Update frontier v push if out nbr prim test ([#1985](https://github.com/rapidsai/cugraph/pull/1985)) [@kaatish](https://github.com/kaatish)
- Pass stream to cuco::static_map ([#1984](https://github.com/rapidsai/cugraph/pull/1984)) [@seunghwak](https://github.com/seunghwak)
- Shutdown the connected scheduler and workers ([#1980](https://github.com/rapidsai/cugraph/pull/1980)) [@jnke2016](https://github.com/jnke2016)
- Use CUB 1.15.0&#39;s new segmented sort ([#1977](https://github.com/rapidsai/cugraph/pull/1977)) [@seunghwak](https://github.com/seunghwak)
- Improve consistency in C++ test case names and add R-mat tests to graph coarsening ([#1976](https://github.com/rapidsai/cugraph/pull/1976)) [@seunghwak](https://github.com/seunghwak)
- 22.02 dep fix ([#1974](https://github.com/rapidsai/cugraph/pull/1974)) [@BradReesWork](https://github.com/BradReesWork)
- Extract paths C API implementation ([#1973](https://github.com/rapidsai/cugraph/pull/1973)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add rmat tests to Louvain C++ unit tests ([#1971](https://github.com/rapidsai/cugraph/pull/1971)) [@ChuckHastings](https://github.com/ChuckHastings)
- Branch 22.02 merge 21.12 ([#1965](https://github.com/rapidsai/cugraph/pull/1965)) [@rlratzel](https://github.com/rlratzel)
- Update to UCX-Py 0.24 ([#1962](https://github.com/rapidsai/cugraph/pull/1962)) [@pentschev](https://github.com/pentschev)
- add rmm pool option for SNMG runs ([#1957](https://github.com/rapidsai/cugraph/pull/1957)) [@jnke2016](https://github.com/jnke2016)
- Branch 22.02 merge 21.12 ([#1953](https://github.com/rapidsai/cugraph/pull/1953)) [@rlratzel](https://github.com/rlratzel)
- Update probability params for RMAT call to match Graph500 ([#1952](https://github.com/rapidsai/cugraph/pull/1952)) [@rlratzel](https://github.com/rlratzel)
- Fix the difference in 2D partitioning of GPUs in python and C++ ([#1950](https://github.com/rapidsai/cugraph/pull/1950)) [@seunghwak](https://github.com/seunghwak)
- Raft Handle Updates to cuGraph ([#1894](https://github.com/rapidsai/cugraph/pull/1894)) [@divyegala](https://github.com/divyegala)
- Remove FAISS dependency, inherit other common dependencies from raft ([#1863](https://github.com/rapidsai/cugraph/pull/1863)) [@trxcllnt](https://github.com/trxcllnt)

# cuGraph 21.12.00 (9 Dec 2021)

## 🚨 Breaking Changes

- Disable HITS and setup 11.5 env ([#1930](https://github.com/rapidsai/cugraph/pull/1930)) [@BradReesWork](https://github.com/BradReesWork)

## 🐛 Bug Fixes

- Updates to `libcugraph_etl` conda recipe for CUDA Enhanced Compatibility ([#1968](https://github.com/rapidsai/cugraph/pull/1968)) [@rlratzel](https://github.com/rlratzel)
- Enforce renumbering for MNMG algos ([#1943](https://github.com/rapidsai/cugraph/pull/1943)) [@jnke2016](https://github.com/jnke2016)
- Bug fix in the R-mat generator ([#1929](https://github.com/rapidsai/cugraph/pull/1929)) [@seunghwak](https://github.com/seunghwak)
- Updates to support correct comparisons of cuDF Series with different names ([#1928](https://github.com/rapidsai/cugraph/pull/1928)) [@rlratzel](https://github.com/rlratzel)
- Updated error message and using a proper TypeError exception when an invalid MultiGraph is passed in ([#1925](https://github.com/rapidsai/cugraph/pull/1925)) [@rlratzel](https://github.com/rlratzel)
- Update calls to cuDF Series ctors, bug fix to `cugraph.subgraph()` for handling non-renumbered Graphs ([#1901](https://github.com/rapidsai/cugraph/pull/1901)) [@rlratzel](https://github.com/rlratzel)
- Fix MG test bug ([#1897](https://github.com/rapidsai/cugraph/pull/1897)) [@seunghwak](https://github.com/seunghwak)
- Temporary workaround for CI issues with 11.0 ([#1883](https://github.com/rapidsai/cugraph/pull/1883)) [@ChuckHastings](https://github.com/ChuckHastings)
- Ensuring dask workers are using local space ([#1879](https://github.com/rapidsai/cugraph/pull/1879)) [@jnke2016](https://github.com/jnke2016)
- Disable WCC test until we get get on an A100 to debug on ([#1870](https://github.com/rapidsai/cugraph/pull/1870)) [@ChuckHastings](https://github.com/ChuckHastings)

## 📖 Documentation

- Enable crosslink to rmm ([#1918](https://github.com/rapidsai/cugraph/pull/1918)) [@AyodeAwe](https://github.com/AyodeAwe)

## 🚀 New Features

- C API Create Graph Implementation ([#1940](https://github.com/rapidsai/cugraph/pull/1940)) [@ChuckHastings](https://github.com/ChuckHastings)
- Count self-loops and multi-edges ([#1939](https://github.com/rapidsai/cugraph/pull/1939)) [@seunghwak](https://github.com/seunghwak)
- Add a new graph primitive to filter edges (extract_if_e) ([#1938](https://github.com/rapidsai/cugraph/pull/1938)) [@seunghwak](https://github.com/seunghwak)
- Add options to drop self-loops &amp; multi_edges in C++ test graph generation ([#1934](https://github.com/rapidsai/cugraph/pull/1934)) [@seunghwak](https://github.com/seunghwak)
- K-core implementation for undirected graphs ([#1933](https://github.com/rapidsai/cugraph/pull/1933)) [@seunghwak](https://github.com/seunghwak)
- K-core decomposition API update ([#1924](https://github.com/rapidsai/cugraph/pull/1924)) [@seunghwak](https://github.com/seunghwak)
- Transpose ([#1834](https://github.com/rapidsai/cugraph/pull/1834)) [@seunghwak](https://github.com/seunghwak)
- Symmetrize ([#1833](https://github.com/rapidsai/cugraph/pull/1833)) [@seunghwak](https://github.com/seunghwak)

## 🛠️ Improvements

- Fix Changelog Merge Conflicts for `branch-21.12` ([#1960](https://github.com/rapidsai/cugraph/pull/1960)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin max `dask` &amp; `distributed` to `2021.11.2` ([#1958](https://github.com/rapidsai/cugraph/pull/1958)) [@galipremsagar](https://github.com/galipremsagar)
- Explicitly install cusolver version with the correct ABI version ([#1954](https://github.com/rapidsai/cugraph/pull/1954)) [@robertmaynard](https://github.com/robertmaynard)
- Upgrade `clang` to `11.1.0` ([#1949](https://github.com/rapidsai/cugraph/pull/1949)) [@galipremsagar](https://github.com/galipremsagar)
- cugraph bring in the same cuco as raft and cudf ([#1945](https://github.com/rapidsai/cugraph/pull/1945)) [@robertmaynard](https://github.com/robertmaynard)
- Re-enable HITS in the python API using the new primitive-based implementation ([#1941](https://github.com/rapidsai/cugraph/pull/1941)) [@rlratzel](https://github.com/rlratzel)
- Accounting for raft::random detail changes ([#1937](https://github.com/rapidsai/cugraph/pull/1937)) [@divyegala](https://github.com/divyegala)
- Use collections.abc.Sequence instead of deprecated collections.Sequence. ([#1932](https://github.com/rapidsai/cugraph/pull/1932)) [@bdice](https://github.com/bdice)
- Update rapids-cmake to 21.12 ([#1931](https://github.com/rapidsai/cugraph/pull/1931)) [@dantegd](https://github.com/dantegd)
- Disable HITS and setup 11.5 env ([#1930](https://github.com/rapidsai/cugraph/pull/1930)) [@BradReesWork](https://github.com/BradReesWork)
- add new demo notebook for louvain ([#1927](https://github.com/rapidsai/cugraph/pull/1927)) [@ChuckHastings](https://github.com/ChuckHastings)
- Ensure empty shuffled columns have the appropriate dtype ([#1926](https://github.com/rapidsai/cugraph/pull/1926)) [@jnke2016](https://github.com/jnke2016)
- improved Nx conversion performance ([#1921](https://github.com/rapidsai/cugraph/pull/1921)) [@BradReesWork](https://github.com/BradReesWork)
- Fix metadata mismatch ([#1920](https://github.com/rapidsai/cugraph/pull/1920)) [@jnke2016](https://github.com/jnke2016)
- Additional improvements to support (key, value) pairs when E/V is small and P is large ([#1919](https://github.com/rapidsai/cugraph/pull/1919)) [@seunghwak](https://github.com/seunghwak)
- Remove unnecessary host barrier synchronization ([#1917](https://github.com/rapidsai/cugraph/pull/1917)) [@seunghwak](https://github.com/seunghwak)
- Reduce MNMG memory requirements ([#1916](https://github.com/rapidsai/cugraph/pull/1916)) [@seunghwak](https://github.com/seunghwak)
- Added separate helpers for moving buffers to either cudf column and series objects ([#1915](https://github.com/rapidsai/cugraph/pull/1915)) [@rlratzel](https://github.com/rlratzel)
- C API for creating a graph ([#1907](https://github.com/rapidsai/cugraph/pull/1907)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add raft ops for reduce_v and transform_reduce_v ([#1902](https://github.com/rapidsai/cugraph/pull/1902)) [@kaatish](https://github.com/kaatish)
- Store benchmark results in json files ([#1900](https://github.com/rapidsai/cugraph/pull/1900)) [@jnke2016](https://github.com/jnke2016)
- HITS primitive based implementation ([#1898](https://github.com/rapidsai/cugraph/pull/1898)) [@kaatish](https://github.com/kaatish)
- Update to UCX-Py 0.23 ([#1895](https://github.com/rapidsai/cugraph/pull/1895)) [@Ethyling](https://github.com/Ethyling)
- Updating WCC/SCC notebook ([#1893](https://github.com/rapidsai/cugraph/pull/1893)) [@BradReesWork](https://github.com/BradReesWork)
- Update input argument check for graph_t constructor and remove expensive input argument check for graph_view_t ([#1890](https://github.com/rapidsai/cugraph/pull/1890)) [@seunghwak](https://github.com/seunghwak)
- Update `conda` recipes for Enhanced Compatibility effort ([#1889](https://github.com/rapidsai/cugraph/pull/1889)) [@ajschmidt8](https://github.com/ajschmidt8)
- Minor code clean-up ([#1888](https://github.com/rapidsai/cugraph/pull/1888)) [@seunghwak](https://github.com/seunghwak)
- Sort local neighbors in the graph adjacency list. ([#1886](https://github.com/rapidsai/cugraph/pull/1886)) [@seunghwak](https://github.com/seunghwak)
- initial creation of libcugraph_etl.so ([#1885](https://github.com/rapidsai/cugraph/pull/1885)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fixing Nx and Graph/DiGraph issues ([#1882](https://github.com/rapidsai/cugraph/pull/1882)) [@BradReesWork](https://github.com/BradReesWork)
- Remove unnecessary explicit template instantiation ([#1878](https://github.com/rapidsai/cugraph/pull/1878)) [@seunghwak](https://github.com/seunghwak)
- node2vec Sampling Implementation ([#1875](https://github.com/rapidsai/cugraph/pull/1875)) [@aschaffer](https://github.com/aschaffer)
- update docstring and examples ([#1866](https://github.com/rapidsai/cugraph/pull/1866)) [@jnke2016](https://github.com/jnke2016)
- Copy v transform reduce out test ([#1856](https://github.com/rapidsai/cugraph/pull/1856)) [@kaatish](https://github.com/kaatish)
- Unpin `dask` &amp; `distributed` ([#1849](https://github.com/rapidsai/cugraph/pull/1849)) [@galipremsagar](https://github.com/galipremsagar)
- Fix automerger for `branch-21.12` ([#1848](https://github.com/rapidsai/cugraph/pull/1848)) [@galipremsagar](https://github.com/galipremsagar)
- Extract BFS paths SG implementation ([#1838](https://github.com/rapidsai/cugraph/pull/1838)) [@ChuckHastings](https://github.com/ChuckHastings)
- Initial cuGraph C API - biased RW, C tests, script updates, cmake files, C library helpers ([#1799](https://github.com/rapidsai/cugraph/pull/1799)) [@aschaffer](https://github.com/aschaffer)

# cuGraph 21.10.00 (7 Oct 2021)

## 🚨 Breaking Changes

- remove tsp implementation from 21.10 ([#1812](https://github.com/rapidsai/cugraph/pull/1812)) [@ChuckHastings](https://github.com/ChuckHastings)
- multi seeds BFS with one seed per component ([#1591](https://github.com/rapidsai/cugraph/pull/1591)) [@afender](https://github.com/afender)

## 🐛 Bug Fixes

- make_zip_iterator should be on a make_tuple ([#1857](https://github.com/rapidsai/cugraph/pull/1857)) [@ChuckHastings](https://github.com/ChuckHastings)
- Removed NetworkX requirement for type checks, fixed docstring, added new docstrings, import cleanups ([#1853](https://github.com/rapidsai/cugraph/pull/1853)) [@rlratzel](https://github.com/rlratzel)
- Temporarily disable input argument checks for a currently disabled feature ([#1840](https://github.com/rapidsai/cugraph/pull/1840)) [@seunghwak](https://github.com/seunghwak)
- Changed value of the expensive check param to `false` in `populate_graph_container` ([#1839](https://github.com/rapidsai/cugraph/pull/1839)) [@rlratzel](https://github.com/rlratzel)
- Accommodate cudf change to is_string_dtype method ([#1827](https://github.com/rapidsai/cugraph/pull/1827)) [@ChuckHastings](https://github.com/ChuckHastings)
- Changed code to disable `k_truss` on CUDA 11.4 differently ([#1811](https://github.com/rapidsai/cugraph/pull/1811)) [@rlratzel](https://github.com/rlratzel)
- Clean-up artifacts from the multi-source BFS PR ([#1591) (#1804](https://github.com/rapidsai/cugraph/pull/1591) (#1804)) [@seunghwak](https://github.com/seunghwak)
- MG WCC bug fix ([#1802](https://github.com/rapidsai/cugraph/pull/1802)) [@seunghwak](https://github.com/seunghwak)
- Fix MG Louvain test compile errors ([#1797](https://github.com/rapidsai/cugraph/pull/1797)) [@seunghwak](https://github.com/seunghwak)
- force_atlas2 to support nx hypercube_graph ([#1779](https://github.com/rapidsai/cugraph/pull/1779)) [@jnke2016](https://github.com/jnke2016)
- Bug louvain reverted fix ([#1766](https://github.com/rapidsai/cugraph/pull/1766)) [@ChuckHastings](https://github.com/ChuckHastings)
- Bug dask cudf personalization ([#1764](https://github.com/rapidsai/cugraph/pull/1764)) [@Iroy30](https://github.com/Iroy30)

## 📖 Documentation

- updated to new doc theme ([#1793](https://github.com/rapidsai/cugraph/pull/1793)) [@BradReesWork](https://github.com/BradReesWork)
- Change python docs to pydata theme ([#1785](https://github.com/rapidsai/cugraph/pull/1785)) [@galipremsagar](https://github.com/galipremsagar)
- Initial doc update for running the python E2E benchmarks in a MNMG environment. ([#1781](https://github.com/rapidsai/cugraph/pull/1781)) [@rlratzel](https://github.com/rlratzel)

## 🚀 New Features

- C++ benchmarking for additional algorithms ([#1762](https://github.com/rapidsai/cugraph/pull/1762)) [@seunghwak](https://github.com/seunghwak)

## 🛠️ Improvements

- Updating cuco to latest ([#1859](https://github.com/rapidsai/cugraph/pull/1859)) [@BradReesWork](https://github.com/BradReesWork)
- fix benchmark exit status ([#1850](https://github.com/rapidsai/cugraph/pull/1850)) [@jnke2016](https://github.com/jnke2016)
- add try/catch for python-louvain ([#1842](https://github.com/rapidsai/cugraph/pull/1842)) [@BradReesWork](https://github.com/BradReesWork)
- Pin max dask and distributed versions to 2021.09.1 ([#1841](https://github.com/rapidsai/cugraph/pull/1841)) [@galipremsagar](https://github.com/galipremsagar)
- add compiler version checks to cmake to fail early ([#1836](https://github.com/rapidsai/cugraph/pull/1836)) [@ChuckHastings](https://github.com/ChuckHastings)
- Make sure we keep the rapids-cmake and cugraph cal version in sync ([#1830](https://github.com/rapidsai/cugraph/pull/1830)) [@robertmaynard](https://github.com/robertmaynard)
- Remove obsolete file ([#1829](https://github.com/rapidsai/cugraph/pull/1829)) [@ChuckHastings](https://github.com/ChuckHastings)
- Improve memory scaling for low average vertex degree graphs &amp; many GPUs ([#1823](https://github.com/rapidsai/cugraph/pull/1823)) [@seunghwak](https://github.com/seunghwak)
- Added the reduction op input parameter to host_scalar_(all)reduce utility functions. ([#1822](https://github.com/rapidsai/cugraph/pull/1822)) [@seunghwak](https://github.com/seunghwak)
- Count if e test ([#1821](https://github.com/rapidsai/cugraph/pull/1821)) [@kaatish](https://github.com/kaatish)
- Added Sorensen algorithm to Python API ([#1820](https://github.com/rapidsai/cugraph/pull/1820)) [@jnke2016](https://github.com/jnke2016)
- Updated to enforce only supported dtypes, changed to use legacy connected_components API ([#1817](https://github.com/rapidsai/cugraph/pull/1817)) [@rlratzel](https://github.com/rlratzel)
- Group return values of renumber_edgelist and input parameters of graph_t &amp; graph_view_t constructors. ([#1816](https://github.com/rapidsai/cugraph/pull/1816)) [@seunghwak](https://github.com/seunghwak)
- remove tsp implementation from 21.10 ([#1812](https://github.com/rapidsai/cugraph/pull/1812)) [@ChuckHastings](https://github.com/ChuckHastings)
- Changed pylibcugraph connected_components APIs to use duck typing for CAI inputs, added doc placeholders ([#1810](https://github.com/rapidsai/cugraph/pull/1810)) [@rlratzel](https://github.com/rlratzel)
- Add new new raft symlink path to .gitignore ([#1808](https://github.com/rapidsai/cugraph/pull/1808)) [@trxcllnt](https://github.com/trxcllnt)
- Initial version of `pylibcugraph` conda package and CI build script updates ([#1806](https://github.com/rapidsai/cugraph/pull/1806)) [@rlratzel](https://github.com/rlratzel)
- Also building cpp MG tests as part of conda/CI libcugraph builds ([#1805](https://github.com/rapidsai/cugraph/pull/1805)) [@rlratzel](https://github.com/rlratzel)
- Split many files to separate SG from MG template instantiations ([#1803](https://github.com/rapidsai/cugraph/pull/1803)) [@ChuckHastings](https://github.com/ChuckHastings)
- Graph primitives memory scaling improvements for low average vertex degree graphs and many GPUs (Part 1) ([#1801](https://github.com/rapidsai/cugraph/pull/1801)) [@seunghwak](https://github.com/seunghwak)
- Pylibcugraph connected components ([#1800](https://github.com/rapidsai/cugraph/pull/1800)) [@Iroy30](https://github.com/Iroy30)
- Transform Reduce E test ([#1798](https://github.com/rapidsai/cugraph/pull/1798)) [@kaatish](https://github.com/kaatish)
- Update with rapids cmake new features ([#1790](https://github.com/rapidsai/cugraph/pull/1790)) [@robertmaynard](https://github.com/robertmaynard)
- Update thrust/RMM deprecated calls ([#1789](https://github.com/rapidsai/cugraph/pull/1789)) [@dantegd](https://github.com/dantegd)
- Update UCX-Py to 0.22 ([#1788](https://github.com/rapidsai/cugraph/pull/1788)) [@pentschev](https://github.com/pentschev)
- Initial version of `pylibcugraph` source tree and build script updates ([#1787](https://github.com/rapidsai/cugraph/pull/1787)) [@rlratzel](https://github.com/rlratzel)
- Fix Forward-Merge Conflicts ([#1786](https://github.com/rapidsai/cugraph/pull/1786)) [@ajschmidt8](https://github.com/ajschmidt8)
- add conda environment for CUDA 11.4 ([#1784](https://github.com/rapidsai/cugraph/pull/1784)) [@seunghwak](https://github.com/seunghwak)
- Temporarily pin RMM while refactor removes deprecated calls ([#1775](https://github.com/rapidsai/cugraph/pull/1775)) [@dantegd](https://github.com/dantegd)
- MNMG memory footprint improvement for low average vertex degree graphs (part 2) ([#1774](https://github.com/rapidsai/cugraph/pull/1774)) [@seunghwak](https://github.com/seunghwak)
- Fix unused variables/parameters warnings ([#1772](https://github.com/rapidsai/cugraph/pull/1772)) [@seunghwak](https://github.com/seunghwak)
- MNMG memory footprint improvement for low average vertex degree graphs (part 1) ([#1769](https://github.com/rapidsai/cugraph/pull/1769)) [@seunghwak](https://github.com/seunghwak)
- Transform reduce v test ([#1768](https://github.com/rapidsai/cugraph/pull/1768)) [@kaatish](https://github.com/kaatish)
- Move experimental source files and a few implementation headers ([#1763](https://github.com/rapidsai/cugraph/pull/1763)) [@ChuckHastings](https://github.com/ChuckHastings)
- updating notebooks ([#1761](https://github.com/rapidsai/cugraph/pull/1761)) [@BradReesWork](https://github.com/BradReesWork)
- consolidate tests to use the fixture dask_client ([#1758](https://github.com/rapidsai/cugraph/pull/1758)) [@jnke2016](https://github.com/jnke2016)
- Move all new graph objects out of experimental namespace ([#1757](https://github.com/rapidsai/cugraph/pull/1757)) [@ChuckHastings](https://github.com/ChuckHastings)
- C++ benchmarking for MG PageRank ([#1755](https://github.com/rapidsai/cugraph/pull/1755)) [@seunghwak](https://github.com/seunghwak)
- Move legacy implementations into legacy directories ([#1752](https://github.com/rapidsai/cugraph/pull/1752)) [@ChuckHastings](https://github.com/ChuckHastings)
- Remove hardcoded Pagerank dtype ([#1751](https://github.com/rapidsai/cugraph/pull/1751)) [@jnke2016](https://github.com/jnke2016)
- Add python end to end benchmark and create new directories ([#1750](https://github.com/rapidsai/cugraph/pull/1750)) [@jnke2016](https://github.com/jnke2016)
- Modify MNMG louvain to support an empty vertex partition ([#1744](https://github.com/rapidsai/cugraph/pull/1744)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fea renumbering test ([#1742](https://github.com/rapidsai/cugraph/pull/1742)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix auto-merger for Branch 21.10 coming from 21.08 ([#1740](https://github.com/rapidsai/cugraph/pull/1740)) [@galipremsagar](https://github.com/galipremsagar)
- Use the new RAPIDS.cmake to fetch rapids-cmake ([#1734](https://github.com/rapidsai/cugraph/pull/1734)) [@robertmaynard](https://github.com/robertmaynard)
- Biased Random Walks for GNN ([#1732](https://github.com/rapidsai/cugraph/pull/1732)) [@aschaffer](https://github.com/aschaffer)
- Updated MG python tests to run in single and multi-node environments ([#1731](https://github.com/rapidsai/cugraph/pull/1731)) [@rlratzel](https://github.com/rlratzel)
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#1720](https://github.com/rapidsai/cugraph/pull/1720)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Apply modifications to account for RAFT changes ([#1707](https://github.com/rapidsai/cugraph/pull/1707)) [@viclafargue](https://github.com/viclafargue)
- multi seeds BFS with one seed per component ([#1591](https://github.com/rapidsai/cugraph/pull/1591)) [@afender](https://github.com/afender)

# cuGraph 21.08.00 (4 Aug 2021)

## 🚨 Breaking Changes

- Removed depricated code ([#1705](https://github.com/rapidsai/cugraph/pull/1705)) [@BradReesWork](https://github.com/BradReesWork)
- Delete legacy renumbering implementation ([#1681](https://github.com/rapidsai/cugraph/pull/1681)) [@ChuckHastings](https://github.com/ChuckHastings)
- Migrate old graph to legacy directory/namespace ([#1675](https://github.com/rapidsai/cugraph/pull/1675)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🐛 Bug Fixes

- Changed cuco cmake function to return early if cuco has already been added as a target ([#1746](https://github.com/rapidsai/cugraph/pull/1746)) [@rlratzel](https://github.com/rlratzel)
- revert cuco to latest dev branch, issues should be fixed ([#1721](https://github.com/rapidsai/cugraph/pull/1721)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix `conda` uploads ([#1712](https://github.com/rapidsai/cugraph/pull/1712)) [@ajschmidt8](https://github.com/ajschmidt8)
- Updated for CUDA-specific py packages ([#1709](https://github.com/rapidsai/cugraph/pull/1709)) [@rlratzel](https://github.com/rlratzel)
- Use `library_dirs` for cython linking, link cudatoolkit libs, allow setting UCX install location ([#1698](https://github.com/rapidsai/cugraph/pull/1698)) [@trxcllnt](https://github.com/trxcllnt)
- Fix the Louvain failure with 64 bit vertex IDs ([#1696](https://github.com/rapidsai/cugraph/pull/1696)) [@seunghwak](https://github.com/seunghwak)
- Use nested include in destination of install headers to avoid docker permission issues ([#1656](https://github.com/rapidsai/cugraph/pull/1656)) [@dantegd](https://github.com/dantegd)
- Added accidentally-removed cpp-mgtests target back to the valid args list ([#1652](https://github.com/rapidsai/cugraph/pull/1652)) [@rlratzel](https://github.com/rlratzel)
- Update UCX-Py version to 0.21 ([#1650](https://github.com/rapidsai/cugraph/pull/1650)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Docs for RMAT ([#1735](https://github.com/rapidsai/cugraph/pull/1735)) [@BradReesWork](https://github.com/BradReesWork)
- Doc updates ([#1719](https://github.com/rapidsai/cugraph/pull/1719)) [@BradReesWork](https://github.com/BradReesWork)

## 🚀 New Features

- Fea cleanup stream part1 ([#1653](https://github.com/rapidsai/cugraph/pull/1653)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🛠️ Improvements

- Pinning cuco to a specific commit hash for release ([#1741](https://github.com/rapidsai/cugraph/pull/1741)) [@rlratzel](https://github.com/rlratzel)
- Pin max version for `dask` &amp; `distributed` ([#1736](https://github.com/rapidsai/cugraph/pull/1736)) [@galipremsagar](https://github.com/galipremsagar)
- Fix libfaiss dependency to not expressly depend on conda-forge ([#1728](https://github.com/rapidsai/cugraph/pull/1728)) [@Ethyling](https://github.com/Ethyling)
- Fix MG_test bug ([#1718](https://github.com/rapidsai/cugraph/pull/1718)) [@jnke2016](https://github.com/jnke2016)
- Cascaded dispatch for type-erased API ([#1711](https://github.com/rapidsai/cugraph/pull/1711)) [@aschaffer](https://github.com/aschaffer)
- ReduceV test ([#1710](https://github.com/rapidsai/cugraph/pull/1710)) [@kaatish](https://github.com/kaatish)
- Removed depricated code ([#1705](https://github.com/rapidsai/cugraph/pull/1705)) [@BradReesWork](https://github.com/BradReesWork)
- Delete unused/out-dated primitives ([#1704](https://github.com/rapidsai/cugraph/pull/1704)) [@seunghwak](https://github.com/seunghwak)
- Update primitives to support DCSR (DCSC) segments (Part 2/2) ([#1703](https://github.com/rapidsai/cugraph/pull/1703)) [@seunghwak](https://github.com/seunghwak)
- Fea speedup compile ([#1702](https://github.com/rapidsai/cugraph/pull/1702)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update `conda` environment name for CI ([#1699](https://github.com/rapidsai/cugraph/pull/1699)) [@ajschmidt8](https://github.com/ajschmidt8)
- Count if test ([#1697](https://github.com/rapidsai/cugraph/pull/1697)) [@kaatish](https://github.com/kaatish)
- replace cudf assert_eq ([#1693](https://github.com/rapidsai/cugraph/pull/1693)) [@jnke2016](https://github.com/jnke2016)
- Fix int64 vertex_t ([#1691](https://github.com/rapidsai/cugraph/pull/1691)) [@Iroy30](https://github.com/Iroy30)
- Update primitives to support DCSR (DCSC) segments (Part 1) ([#1690](https://github.com/rapidsai/cugraph/pull/1690)) [@seunghwak](https://github.com/seunghwak)
- remove hardcoded dtype ([#1689](https://github.com/rapidsai/cugraph/pull/1689)) [@Iroy30](https://github.com/Iroy30)
- Updating Clang Version to 11.0.0 ([#1688](https://github.com/rapidsai/cugraph/pull/1688)) [@codereport](https://github.com/codereport)
- `CHECK_CUDA` macros in debug builds ([#1687](https://github.com/rapidsai/cugraph/pull/1687)) [@trxcllnt](https://github.com/trxcllnt)
- fixing symmetrize_ddf ([#1686](https://github.com/rapidsai/cugraph/pull/1686)) [@jnke2016](https://github.com/jnke2016)
- Improve Random Walks performance ([#1685](https://github.com/rapidsai/cugraph/pull/1685)) [@aschaffer](https://github.com/aschaffer)
- Use the 21.08 branch of rapids-cmake as rmm requires it ([#1683](https://github.com/rapidsai/cugraph/pull/1683)) [@robertmaynard](https://github.com/robertmaynard)
- Delete legacy renumbering implementation ([#1681](https://github.com/rapidsai/cugraph/pull/1681)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix vertex partition offsets ([#1680](https://github.com/rapidsai/cugraph/pull/1680)) [@Iroy30](https://github.com/Iroy30)
- Ues std::optional (or thrust::optional) for optional parameters &amp; first part of DCSR (DCSC) implementation. ([#1676](https://github.com/rapidsai/cugraph/pull/1676)) [@seunghwak](https://github.com/seunghwak)
- Migrate old graph to legacy directory/namespace ([#1675](https://github.com/rapidsai/cugraph/pull/1675)) [@ChuckHastings](https://github.com/ChuckHastings)
- Expose epsilon parameter (precision) through python layer ([#1674](https://github.com/rapidsai/cugraph/pull/1674)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fea hungarian expose precision ([#1673](https://github.com/rapidsai/cugraph/pull/1673)) [@ChuckHastings](https://github.com/ChuckHastings)
- Branch 21.08 merge 21.06 ([#1672](https://github.com/rapidsai/cugraph/pull/1672)) [@BradReesWork](https://github.com/BradReesWork)
- Update pins to Dask/Distributed &gt;= 2021.6.0 ([#1666](https://github.com/rapidsai/cugraph/pull/1666)) [@pentschev](https://github.com/pentschev)
- Fix conflicts in `1643` ([#1651](https://github.com/rapidsai/cugraph/pull/1651)) [@ajschmidt8](https://github.com/ajschmidt8)
- Rename include/cugraph/patterns to include/cugraph/prims ([#1644](https://github.com/rapidsai/cugraph/pull/1644)) [@seunghwak](https://github.com/seunghwak)
- Fix merge conflicts in 1631 ([#1639](https://github.com/rapidsai/cugraph/pull/1639)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update to changed `rmm::device_scalar` API ([#1637](https://github.com/rapidsai/cugraph/pull/1637)) [@harrism](https://github.com/harrism)
- Fix merge conflicts ([#1614](https://github.com/rapidsai/cugraph/pull/1614)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuGraph 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- Delete CUDA_ARCHITECTURES=OFF ([#1638](https://github.com/rapidsai/cugraph/pull/1638)) [@seunghwak](https://github.com/seunghwak)
- transform_reduce_e bug fixes ([#1633](https://github.com/rapidsai/cugraph/pull/1633)) [@ChuckHastings](https://github.com/ChuckHastings)
- Correct install path for include folder to avoid double nesting ([#1630](https://github.com/rapidsai/cugraph/pull/1630)) [@dantegd](https://github.com/dantegd)
- Remove thread local thrust::sort (thrust::sort with the execution policy thrust::seq) from copy_v_transform_reduce_key_aggregated_out_nbr ([#1627](https://github.com/rapidsai/cugraph/pull/1627)) [@seunghwak](https://github.com/seunghwak)

## 🚀 New Features

- SG &amp; MG Weakly Connected Components ([#1604](https://github.com/rapidsai/cugraph/pull/1604)) [@seunghwak](https://github.com/seunghwak)

## 🛠️ Improvements

- Remove Pascal guard and test cuGraph use of cuco::static_map on Pascal ([#1640](https://github.com/rapidsai/cugraph/pull/1640)) [@seunghwak](https://github.com/seunghwak)
- Upgraded recipe and dev envs to NCCL 2.9.9 ([#1636](https://github.com/rapidsai/cugraph/pull/1636)) [@rlratzel](https://github.com/rlratzel)
- Use UCX-Py 0.20 ([#1634](https://github.com/rapidsai/cugraph/pull/1634)) [@jakirkham](https://github.com/jakirkham)
- Updated dependencies for CalVer ([#1629](https://github.com/rapidsai/cugraph/pull/1629)) [@rlratzel](https://github.com/rlratzel)
- MG WCC improvements ([#1628](https://github.com/rapidsai/cugraph/pull/1628)) [@seunghwak](https://github.com/seunghwak)
- Initialize force_atlas2 `old_forces` device_uvector, use new `rmm::exec_policy` ([#1625](https://github.com/rapidsai/cugraph/pull/1625)) [@trxcllnt](https://github.com/trxcllnt)
- Fix developer guide examples for device_buffer ([#1619](https://github.com/rapidsai/cugraph/pull/1619)) [@harrism](https://github.com/harrism)
- Pass rmm memory allocator to cuco::static_map ([#1617](https://github.com/rapidsai/cugraph/pull/1617)) [@seunghwak](https://github.com/seunghwak)
- Undo disabling MG C++ testing outputs for non-root processes ([#1615](https://github.com/rapidsai/cugraph/pull/1615)) [@seunghwak](https://github.com/seunghwak)
- WCC bindings ([#1612](https://github.com/rapidsai/cugraph/pull/1612)) [@Iroy30](https://github.com/Iroy30)
- address &#39;ValueError: Series contains NULL values&#39; from from_cudf_edge… ([#1610](https://github.com/rapidsai/cugraph/pull/1610)) [@mattf](https://github.com/mattf)
- Fea rmm device buffer change ([#1609](https://github.com/rapidsai/cugraph/pull/1609)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update `CHANGELOG.md` links for calver ([#1608](https://github.com/rapidsai/cugraph/pull/1608)) [@ajschmidt8](https://github.com/ajschmidt8)
- Handle int64 in force atlas wrapper and update to uvector ([#1607](https://github.com/rapidsai/cugraph/pull/1607)) [@hlinsen](https://github.com/hlinsen)
- Update docs build script ([#1606](https://github.com/rapidsai/cugraph/pull/1606)) [@ajschmidt8](https://github.com/ajschmidt8)
- WCC performance/memory footprint optimization ([#1605](https://github.com/rapidsai/cugraph/pull/1605)) [@seunghwak](https://github.com/seunghwak)
- adding test graphs - part 2 ([#1603](https://github.com/rapidsai/cugraph/pull/1603)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update the Random Walk binding ([#1599](https://github.com/rapidsai/cugraph/pull/1599)) [@Iroy30](https://github.com/Iroy30)
- Add mnmg out degree ([#1592](https://github.com/rapidsai/cugraph/pull/1592)) [@Iroy30](https://github.com/Iroy30)
- Update `cugraph` to with newest CMake features, including CPM for dependencies ([#1585](https://github.com/rapidsai/cugraph/pull/1585)) [@robertmaynard](https://github.com/robertmaynard)
- Implement Graph Batching functionality ([#1580](https://github.com/rapidsai/cugraph/pull/1580)) [@aschaffer](https://github.com/aschaffer)
- add multi-column support in algorithms - part 2 ([#1571](https://github.com/rapidsai/cugraph/pull/1571)) [@Iroy30](https://github.com/Iroy30)

# cuGraph 0.19.0 (21 Apr 2021)

## 🐛 Bug Fixes

- Fixed copyright date and format ([#1526](https://github.com//rapidsai/cugraph/pull/1526)) [@rlratzel](https://github.com/rlratzel)
- fix mg_renumber non-deterministic errors ([#1523](https://github.com//rapidsai/cugraph/pull/1523)) [@Iroy30](https://github.com/Iroy30)
- Updated NetworkX version to 2.5.1 ([#1510](https://github.com//rapidsai/cugraph/pull/1510)) [@rlratzel](https://github.com/rlratzel)
- pascal renumbering fix ([#1505](https://github.com//rapidsai/cugraph/pull/1505)) [@Iroy30](https://github.com/Iroy30)
- Fix MNMG test failures and skip tests that are not supported on Pascal ([#1498](https://github.com//rapidsai/cugraph/pull/1498)) [@jnke2016](https://github.com/jnke2016)
- Revert &quot;Update conda recipes pinning of repo dependencies&quot; ([#1493](https://github.com//rapidsai/cugraph/pull/1493)) [@raydouglass](https://github.com/raydouglass)
- Update conda recipes pinning of repo dependencies ([#1485](https://github.com//rapidsai/cugraph/pull/1485)) [@mike-wendt](https://github.com/mike-wendt)
- Update to make notebook_list.py compatible with numba 0.53 ([#1455](https://github.com//rapidsai/cugraph/pull/1455)) [@rlratzel](https://github.com/rlratzel)
- Fix bugs in copy_v_transform_reduce_key_aggregated_out_nbr &amp; groupby_gpuid_and_shuffle ([#1434](https://github.com//rapidsai/cugraph/pull/1434)) [@seunghwak](https://github.com/seunghwak)
- update default path of setup to use the new directory paths in build … ([#1425](https://github.com//rapidsai/cugraph/pull/1425)) [@ChuckHastings](https://github.com/ChuckHastings)

## 📖 Documentation

- Create C++ documentation ([#1489](https://github.com//rapidsai/cugraph/pull/1489)) [@ChuckHastings](https://github.com/ChuckHastings)
- Create cuGraph developers guide ([#1431](https://github.com//rapidsai/cugraph/pull/1431)) [@ChuckHastings](https://github.com/ChuckHastings)
- Add boost 1.0 license file. ([#1401](https://github.com//rapidsai/cugraph/pull/1401)) [@seunghwak](https://github.com/seunghwak)

## 🚀 New Features

- Implement C/CUDA RandomWalks functionality ([#1439](https://github.com//rapidsai/cugraph/pull/1439)) [@aschaffer](https://github.com/aschaffer)
- Add R-mat generator ([#1411](https://github.com//rapidsai/cugraph/pull/1411)) [@seunghwak](https://github.com/seunghwak)

## 🛠️ Improvements

- Random Walks - Python Bindings ([#1516](https://github.com//rapidsai/cugraph/pull/1516)) [@jnke2016](https://github.com/jnke2016)
- Updating RAFT tag ([#1509](https://github.com//rapidsai/cugraph/pull/1509)) [@afender](https://github.com/afender)
- Clean up nullptr cuda_stream_view arguments ([#1504](https://github.com//rapidsai/cugraph/pull/1504)) [@hlinsen](https://github.com/hlinsen)
- Reduce the size of the cugraph libraries ([#1503](https://github.com//rapidsai/cugraph/pull/1503)) [@robertmaynard](https://github.com/robertmaynard)
- Add indirection and replace algorithms with new renumbering ([#1484](https://github.com//rapidsai/cugraph/pull/1484)) [@Iroy30](https://github.com/Iroy30)
- Multiple graph generator with power law distribution on sizes ([#1483](https://github.com//rapidsai/cugraph/pull/1483)) [@afender](https://github.com/afender)
- TSP solver bug fix ([#1480](https://github.com//rapidsai/cugraph/pull/1480)) [@hlinsen](https://github.com/hlinsen)
- Added cmake function and .hpp template for generating version_config.hpp file. ([#1476](https://github.com//rapidsai/cugraph/pull/1476)) [@rlratzel](https://github.com/rlratzel)
- Fix for bug in SCC on self-loops ([#1475](https://github.com//rapidsai/cugraph/pull/1475)) [@aschaffer](https://github.com/aschaffer)
- MS BFS python APIs + EgoNet updates ([#1469](https://github.com//rapidsai/cugraph/pull/1469)) [@afender](https://github.com/afender)
- Removed unused dependencies from libcugraph recipe, moved non-test script code from test script to gpu build script ([#1468](https://github.com//rapidsai/cugraph/pull/1468)) [@rlratzel](https://github.com/rlratzel)
- Remove literals passed to `device_uvector::set_element_async` ([#1453](https://github.com//rapidsai/cugraph/pull/1453)) [@harrism](https://github.com/harrism)
- ENH Change conda build directories to work with ccache ([#1452](https://github.com//rapidsai/cugraph/pull/1452)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Updating docs ([#1448](https://github.com//rapidsai/cugraph/pull/1448)) [@BradReesWork](https://github.com/BradReesWork)
- Improve graph primitives performance on graphs with widely varying vertex degrees ([#1447](https://github.com//rapidsai/cugraph/pull/1447)) [@seunghwak](https://github.com/seunghwak)
- Update Changelog Link ([#1446](https://github.com//rapidsai/cugraph/pull/1446)) [@ajschmidt8](https://github.com/ajschmidt8)
- Updated NCCL to version 2.8.4 ([#1445](https://github.com//rapidsai/cugraph/pull/1445)) [@BradReesWork](https://github.com/BradReesWork)
- Update FAISS to 1.7.0 ([#1444](https://github.com//rapidsai/cugraph/pull/1444)) [@BradReesWork](https://github.com/BradReesWork)
- Update graph partitioning scheme ([#1443](https://github.com//rapidsai/cugraph/pull/1443)) [@seunghwak](https://github.com/seunghwak)
- Add additional datasets to improve coverage ([#1441](https://github.com//rapidsai/cugraph/pull/1441)) [@jnke2016](https://github.com/jnke2016)
- Update C++ MG PageRank and SG PageRank, Katz Centrality, BFS, and SSSP to use the new R-mat graph generator ([#1438](https://github.com//rapidsai/cugraph/pull/1438)) [@seunghwak](https://github.com/seunghwak)
- Remove raft handle duplication ([#1436](https://github.com//rapidsai/cugraph/pull/1436)) [@Iroy30](https://github.com/Iroy30)
- Streams infra + support in egonet ([#1435](https://github.com//rapidsai/cugraph/pull/1435)) [@afender](https://github.com/afender)
- Prepare Changelog for Automation ([#1433](https://github.com//rapidsai/cugraph/pull/1433)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update 0.18 changelog entry ([#1429](https://github.com//rapidsai/cugraph/pull/1429)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update and Test Renumber bindings ([#1427](https://github.com//rapidsai/cugraph/pull/1427)) [@Iroy30](https://github.com/Iroy30)
- Update Louvain to use new graph primitives and pattern accelerators ([#1423](https://github.com//rapidsai/cugraph/pull/1423)) [@ChuckHastings](https://github.com/ChuckHastings)
- Replace rmm::device_vector &amp; thrust::host_vector with rmm::device_uvector &amp; std::vector, respectively. ([#1421](https://github.com//rapidsai/cugraph/pull/1421)) [@seunghwak](https://github.com/seunghwak)
- Update C++ MG PageRank test ([#1419](https://github.com//rapidsai/cugraph/pull/1419)) [@seunghwak](https://github.com/seunghwak)
- ENH Build with `cmake --build` &amp; Pass ccache variables to conda recipe &amp; use Ninja in CI ([#1415](https://github.com//rapidsai/cugraph/pull/1415)) [@Ethyling](https://github.com/Ethyling)
- Adding new primitives: copy_v_transform_reduce_key_aggregated_out_nbr &amp; transform_reduce_by_adj_matrix_row|col_key_e bug fixes ([#1399](https://github.com//rapidsai/cugraph/pull/1399)) [@seunghwak](https://github.com/seunghwak)
- Add new primitives: compute_in|out_degrees, compute_in|out_weight_sums to graph_view_t ([#1394](https://github.com//rapidsai/cugraph/pull/1394)) [@seunghwak](https://github.com/seunghwak)
- Rename sort_and_shuffle to groupby_gpuid_and_shuffle ([#1392](https://github.com//rapidsai/cugraph/pull/1392)) [@seunghwak](https://github.com/seunghwak)
- Matching updates for RAFT comms updates (device_sendrecv, device_multicast_sendrecv, gather, gatherv) ([#1391](https://github.com//rapidsai/cugraph/pull/1391)) [@seunghwak](https://github.com/seunghwak)
- Fix forward-merge conflicts for #1370 ([#1377](https://github.com//rapidsai/cugraph/pull/1377)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add utility function for computing a secondary cost for BFS and SSSP output ([#1376](https://github.com//rapidsai/cugraph/pull/1376)) [@hlinsen](https://github.com/hlinsen)

# cuGraph 0.18.0 (24 Feb 2021)

## Bug Fixes 🐛

- Fixed TSP returned routes (#1412) @hlinsen
- Updated CI scripts to use a different error handling convention, updated LD_LIBRARY_PATH for project flash runs (#1386) @rlratzel
- Bug fixes for MNMG coarsen_graph, renumber_edgelist, relabel (#1364) @seunghwak
- Set a specific known working commit hash for gunrock instead of &quot;dev&quot; (#1336) @rlratzel
- Updated git utils used by copyright.py for compatibility with current CI env (#1325) @rlratzel
- Fix MNMG Louvain tests on Pascal architecture (#1322) @ChuckHastings
- FIX Set bash trap after PATH is updated (#1321) @dillon-cullinan
- Fix graph nodes function and renumbering from series (#1319) @Iroy30
- Fix Branch 0.18 merge 0.17 (#1314) @BradReesWork
- Fix EXPERIMENTAL_LOUVAIN_TEST on Pascal (#1312) @ChuckHastings
- Updated cuxfilter to 0.18, removed datashader indirect dependency in conda dev .yml files (#1311) @rlratzel
- Update SG PageRank C++ tests (#1307) @seunghwak

## Documentation 📖

- Enabled MultiGraph class and tests, updated SOURCEBUILD.md to include the latest build.sh options (#1351) @rlratzel

## New Features 🚀

- New EgoNet extractor (#1365) @afender
- Implement induced subgraph extraction primitive (SG C++) (#1354) @seunghwak

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#1413) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#1408) @Ethyling
- update subgraph tests and remove legacy pagerank (#1378) @Iroy30
- Update the conda environments and README file (#1369) @BradReesWork
- Prepare Changelog for Automation (#1368) @ajschmidt8
- Update CMakeLists.txt files for consistency with RAPIDS and to support cugraph as an external project and other tech debt removal (#1367) @rlratzel
- Use new coarsen_graph primitive in Louvain (#1362) @ChuckHastings
- Added initial infrastructure for MG C++ testing and a Pagerank MG test using it (#1361) @rlratzel
- Add SG TSP (#1360) @hlinsen
- Build a Dendrogram class, adapt Louvain/Leiden/ECG to use it (#1359) @ChuckHastings
- Auto-label PRs based on their content (#1358) @jolorunyomi
- Implement MNMG Renumber (#1355) @aschaffer
- Enabling pytest code coverage output by default (#1352) @jnke2016
- Added configuration for new cugraph-doc-codeowners review group (#1344) @rlratzel
- API update to match RAFT PR #120 (#1343) @drobison00
- Pin gunrock to v1.2 for version 0.18 (#1342) @ChuckHastings
- Fix #1340 - Use generic from_edgelist() methods (#1341) @miguelusque
- Using RAPIDS_DATASET_ROOT_DIR env var in place of absolute path to datasets in tests (#1337) @jnke2016
- Expose dense implementation of Hungarian algorithm (#1333) @ChuckHastings
- SG Pagerank transition (#1332) @Iroy30
- improving error checking and docs (#1327) @BradReesWork
- Fix MNMG cleanup exceptions (#1326) @Iroy30
- Create labeler.yml (#1318) @jolorunyomi
- Updates to support nightly MG test automation (#1308) @rlratzel
- Add C++ graph functions (coarsen_grpah, renumber_edgelist, relabel) and primitvies (transform_reduce_by_adj_matrix_row_key, transform_reduce_by_adj_matrix_col_key, copy_v_transform_reduce_key_aggregated_out_nbr) (#1257) @seunghwak

# cuGraph 0.17.0 (10 Dec 2020)
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
- PR #1280 Add Multi(Di)Graph support

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
