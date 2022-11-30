# cuGraph – Python


cuGraph is a Python package that encapsulate and hides the complexity of the lower layer C/CUDA code.  Additionally, the software is focused on providing an easy and familiar API 



## cuGraph Notice

Vertex IDs are expected to be contiguous integers starting from 0.  If your data doesn't match that restriction, we have a solution.  cuGraph provides the renumber function, which is by default automatically called when data is added to a graph.  Input vertex IDs for the renumber function can be any type, can be non-contiguous, can be multiple columns, and can start from an arbitrary number. The renumber function maps the provided input vertex IDs to either 32- or 64-bit contiguous integers starting from 0. 

Additionally, when using the auto-renumbering feature, vertices are automatically un-renumbered in results.

cuGraph is constantly being updated and improved. Please see the [Transition Guide](TRANSITIONGUIDE.md) if errors are encountered with newer versions

## Graph Sizes and GPU Memory Size
The amount of memory required is dependent on the graph structure and the analytics being executed.  As a simple rule of thumb, the amount of GPU memory should be about twice the size of the data size.  That gives overhead for the CSV reader and other transform functions.  There are ways around the rule but using smaller data chunks.

|       Size        | Recommended GPU Memory |
|-------------------|------------------------|
| 500 million edges |  32 GB                 |
| 250 million edges |  16 GB                 |

The use of managed memory for oversubscription can also be used to exceed the above memory limitations.  See the recent blog on _Tackling Large Graphs with RAPIDS cuGraph and CUDA Unified Memory on GPUs_:  https://medium.com/rapids-ai/tackling-large-graphs-with-rapids-cugraph-and-unified-virtual-memory-b5b69a065d4