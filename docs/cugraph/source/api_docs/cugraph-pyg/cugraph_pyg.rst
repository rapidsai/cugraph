~~~~~~~~~~~~~~~~~~~~~~~~~
cugraph-pyg API Reference
~~~~~~~~~~~~~~~~~~~~~~~~~

cugraph-pyg

.. currentmodule:: cugraph_pyg

Graph Storage
-------------
.. autosummary::
   :toctree: ../api/cugraph-pyg/

   cugraph_pyg.data.dask_graph_store.DaskGraphStore
   cugraph_pyg.data.graph_store.GraphStore

Feature Storage
---------------
.. autosummary::
   :toctree: ../api/cugraph-pyg/

   cugraph_pyg.data.feature_store.TensorDictFeatureStore
   cugraph_pyg.data.feature_store.WholeFeatureStore

Data Loaders
------------
.. autosummary::
   :toctree: ../api/cugraph-pyg/

   cugraph_pyg.loader.dask_node_loader.DaskNeighborLoader
   cugraph_pyg.loader.dask_node_loader.BulkSampleLoader
   cugraph_pyg.loader.node_loader.NodeLoader
   cugraph_pyg.loader.neighbor_loader.NeighborLoader

Samplers
--------
.. autosummary::
   :toctree: ../api/cugraph-pyg/

   cugraph_pyg.sampler.sampler.BaseSampler
   cugraph_pyg.sampler.sampler.SampleReader
   cugraph_pyg.sampler.sampler.HomogeneousSampleReader
   cugraph_pyg.sampler.sampler.SampleIterator
