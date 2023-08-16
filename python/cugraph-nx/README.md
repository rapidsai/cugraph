# cugraph-nx

## Description
[RAPIDS](https://rapids.ai) cugraph-nx is a [backend to NetworkX](https://networkx.org/documentation/stable/reference/classes/index.html#backends)
with minimal dependencies (`networkx`, `cupy`, and `pylibcugraph`) to run graph algorithms on the GPU.

_Nightly conda packages and pip wheels coming soon._

### Contribute

Follow instructions for [contributing to cugraph](https://github.com/rapidsai/cugraph/blob/branch-23.10/readme_pages/CONTRIBUTING.md)
and [building from source](https://docs.rapids.ai/api/cugraph/stable/installation/source_build/), then build cugraph-nx in develop (i.e., editable) mode:
```
$ ./build.sh cugraph-nx --pydevelop
```

### Run tests

Run cugraph-nx tests from `cugraph/python/cugraph-nx` directory:
```
$ pytest
```
Run cugraph-nx benchmarks:
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
