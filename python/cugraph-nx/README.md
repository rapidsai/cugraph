# cugraph_nx

## Description
[RAPIDS](https://rapids.ai) cugraph-nx is an experimental [backend to NetworkX](https://networkx.org/documentation/stable/reference/classes/index.html#backends)
with minimal dependencies (`networkx`, `cupy`, and `pylibcugraph`).
It is a work in progress, and cugraph should typically be used instead.

### Run tests
```
./run_nx_tests.sh
```
Additional arguments may be passed to pytest such as:
```
./run_nx_tests.sh -x --sw -k connected
```
