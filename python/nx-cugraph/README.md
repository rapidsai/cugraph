# nx-cugraph

## Description
[RAPIDS](https://rapids.ai) nx-cugraph is a [backend to NetworkX](https://networkx.org/documentation/stable/reference/classes/index.html#backends)
with minimal dependencies (`networkx`, `cupy`, and `pylibcugraph`) to run graph algorithms on the GPU.

### Contribute

Follow instructions for [contributing to cugraph](https://github.com/rapidsai/cugraph/blob/branch-23.10/readme_pages/CONTRIBUTING.md)
and [building from source](https://docs.rapids.ai/api/cugraph/stable/installation/source_build/), then build nx-cugraph in develop (i.e., editable) mode:
```
$ ./build.sh nx-cugraph --pydevelop
```

### Run tests

Run nx-cugraph tests from `cugraph/python/nx-cugraph` directory:
```
$ pytest
```
Run nx-cugraph benchmarks:
```
$ pytest --bench
```
Run networkx tests (requires networkx version 3.2):
```
$ ./run_nx_tests.sh
```
Additional arguments may be passed to pytest such as:
```
$ ./run_nx_tests.sh -x --sw -k betweenness
```
