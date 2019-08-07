~~~~~~~~~~~~~~~~~~~
Multi-GPU with Dask
~~~~~~~~~~~~~~~~~~~

cuGraph is a single-GPU library. For Multi-GPU cuGraph solutions we use Dask (https://dask.org), which is able to scale cuGraph across multiple GPUs on a single machine, or in future releases, multiple GPUs across many machines in a cluster.

Supported Graph Analytics
=========================

Pagerank
--------

.. automodule:: cugraph.dask.pagerank.pagerank
    :members: pagerank
    :undoc-members: pagerank
