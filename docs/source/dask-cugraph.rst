~~~~~~~~~~~~~~~~~~~~~~
Multi-GPU with cuGraph
~~~~~~~~~~~~~~~~~~~~~~

cuGraph supports multi-GPU leveraging `Dask <https://dask.org>`_. Dask is a flexible library for parallel computing in Python which makes scaling out your workflow smooth and simple. cuGraph also uses other Dask-based RAPIDS projects such as `dask-cuda <https://github.com/rapidsai/dask-cuda>`_. The maximum graph size is currently limited to 2 Billion vertices (to be waived in the next versions).

Distributed graph analytics
===========================

The current solution is able to scale across multiple GPUs on multiple machines. Distributing the graph and computation lets you analyze datasets far larger than a single GPU’s memory.

With cuGraph and Dask, whether you’re using a single NVIDIA GPU or multiple nodes, your RAPIDS workflow will run smoothly, intelligently distributing the workload across the available resources.

If your graph comfortably fits in memory on a single GPU, you would want to use the single-GPU version of cuGraph. If you want to distribute your workflow across multiple GPUs and have more data than you can fit in memory on a single GPU, you would want to use cuGraph's multi-GPU features.


Distributed Graph Algorithms
----------------------------

.. automodule:: cugraph.dask.link_analysis.pagerank
    :members: pagerank
    :undoc-members: 

.. automodule:: cugraph.dask.traversal.bfs
    :members: bfs
    :undoc-members: 


Helper functions 
----------------

.. automodule:: cugraph.comms.comms
    :members: initialize
    :undoc-members:

.. automodule:: cugraph.comms.comms
    :members: destroy
    :undoc-members:

.. automodule:: cugraph.dask.common.read_utils
    :members: get_chunksize
    :undoc-members:

Consolidation
=============

cuGraph can transparently interpret the Dask cuDF Dataframe as a regular Dataframe when loading the edge list. This is particularly helpful for workflows extracting a single GPU sized edge list from a distributed dataset. From there any existing single GPU feature will just work on this input.

For instance, consolidation allows leveraging Dask cuDF CSV reader to load file(s) on multiple GPUs and consolidate this input to a single GPU graph. Reading is often the time and memory bottleneck, with this feature users can call the Multi-GPU version of the reader without changing anything else. 

Batch Processing
================

cuGraph can leverage multi GPUs to increase processing speed for graphs that fit on a single GPU, providing faster analytics on such graphs.
You will be able to use the Graph the same way as you used to in a Single GPU environment, but analytics that support batch processing will automatically use the GPUs available to the dask client.
For example, Betweenness Centrality scores can be slow to obtain depending on the number of vertices used in the approximation. Thank to Multi GPUs Batch Processing,
you can create Single GPU graph as you would regularly do it using cuDF CSV reader, enable Batch analytics on it, and obtain scores much faster as each GPU will handle a sub-set of the sources.
In order to use Batch Analytics you need to set up a Dask Cluster and Client in addition to the cuGraph communicator, then you can simply call `enable_batch()` on you graph, and algorithms supporting batch processing will use multiple GPUs.

Algorithms supporting Batch Processing
--------------------------------------
.. automodule:: cugraph.centrality
    :members: betweenness_centrality
    :undoc-members:
    :noindex:

.. automodule:: cugraph.centrality
    :members: edge_betweenness_centrality
    :undoc-members:
    :noindex:
