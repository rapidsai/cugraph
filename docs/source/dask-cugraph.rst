~~~~~~~~~~~~~~~~~~~~~~
Multi-GPU with cuGraph
~~~~~~~~~~~~~~~~~~~~~~

cuGraph supports multi-GPU leveraging `Dask <https://dask.org>`_. Dask is a flexible library for parallel computing in Python which makes scaling out your workflow smooth and simple. cuGraph also uses other Dask-based RAPIDS projects such as `dask-cuda <https://github.com/rapidsai/dask-cuda>`_.

Consolidation
=============

cuGraph can transparently interpret the Dask cuDF Dataframe as a regular Dataframe when loading the edge list. This is particularly helpful for workflows extracting a single GPU sized edge list from a distributed dataset. From there any existing single GPU feature will just work on this input.

For instance, consolidation allows leveraging Dask cuDF CSV reader to load file(s) on multiple GPUs and consolidate this input to a single GPU graph. Reading is often the time and memory bottleneck, with this feature users can call the Multi-GPU version of the reader without changing anything else. 

Distributed graph analytics
===========================

The current solution is able to scale across multiple GPUs on multiple machines. Distributing the graph and computation lets you analyze datasets far larger than a single GPU’s memory.

With cuGraph and Dask, whether you’re using a single NVIDIA GPU or using all 16 NVIDIA V100 GPUs on a DGX-2, your RAPIDS workflow will run smoothly, intelligently distributing the workload across the available resources.

If your graph comfortably fits in memory on a single GPU, you would want to use the single-GPU version of cuGraph. If you want to distribute your workflow across multiple GPUs and have more data than you can fit in memory on a single GPU, you would want to use cuGraph's multi-GPU features.

Distributed Graph Analytics Support
-----------------------------------

Pagerank

.. automodule:: cugraph.dask.pagerank.pagerank
    :members: pagerank
    :undoc-members: pagerank
