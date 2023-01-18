<h1 align="center";>/
  <br>
  <img src="../img/cugraph_logo_2.png" alt="cuGraph" width="300">
</h1>
<h1 align="left";>
  <br>
CuGraph pylibcugraph
</h1>

Part of [RAPIDS](https://rapids.ai) cuGraph, pylibcugraph is a wrapper around the cuGraph C API. It is aimed more at integrators instead of algorithm writers or end users like Data Scientists. Most of the cuGraph python API uses pylibcugraph to efficiently run algorithms by removing much of the overhead of the python-centric implementation, relying more on cython instead. Pylibcugraph is intended for applications that require a tighter integration with cuGraph at the Python layer with fewer dependencies.

Here is an example of calling the Louvain algorithm using pylibcugraph directly.

```
import pylibcugraph, cupy, numpy
srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
dsts = cupy.asarray([1, 2, 0], dtype=numpy.int32)
weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
resource_handle = pylibcugraph.ResourceHandle()
graph_props = pylibcugraph.GraphProperties(is_symmetric=True, is_multigraph=False)
G = pylibcugraph.SGGraph(
                        resource_handle, graph_props, srcs, dsts, weights,
                        store_transposed=True, renumber=False, do_expensive_check=False)
(vertices, clusters, modularity) = pylibcugraph.louvain(resource_handle, G, 100, 1., False)
```
