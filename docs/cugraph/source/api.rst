~~~~~~~~~~~~~~~~~~~~~
cuGraph API Reference
~~~~~~~~~~~~~~~~~~~~~



Structure
=========

Graph
-----

.. autoclass:: cugraph.structure.graph_classes.Graph
    :members:
    :undoc-members:


Symmetrize
----------

.. automodule:: cugraph.structure.symmetrize
    :members:
    :undoc-members:

Conversion from Other Formats
-----------------------------

.. automodule:: cugraph.structure.convert_matrix
    :members:
    :undoc-members:


Centrality
==========

Betweenness Centrality
----------------------

.. automodule:: cugraph.centrality.betweenness_centrality
    :members:
    :undoc-members:

Katz Centrality
---------------

.. automodule:: cugraph.centrality.katz_centrality
    :members:
    :undoc-members:


Katz Centrality (MG)
--------------------

.. automodule:: cugraph.dask.centrality.katz_centrality
    :members:
    :undoc-members:

Community
=========

EgoNet
------------------------------------

.. automodule:: cugraph.community.egonet
	:members:
	:undoc-members:

Ensemble clustering for graphs (ECG)
------------------------------------

.. automodule:: cugraph.community.ecg
	:members:
	:undoc-members:

K-Truss
-------

.. automodule:: cugraph.community.ktruss_subgraph
    :members:
    :undoc-members:

Leiden
-------

.. automodule:: cugraph.community.leiden
    :members:
    :undoc-members:

Louvain
-------

.. automodule:: cugraph.community.louvain
    :members:
    :undoc-members:

Louvain (MG)
------------

.. automodule:: cugraph.dask.community.louvain
    :members:
    :undoc-members:


Spectral Clustering
-------------------

.. automodule:: cugraph.community.spectral_clustering
    :members:
    :undoc-members:

Subgraph Extraction
-------------------

.. automodule:: cugraph.community.subgraph_extraction
    :members:
    :undoc-members:

Triangle Counting
-----------------

.. automodule:: cugraph.community.triangle_count
    :members:
    :undoc-members:


Components
==========

Connected Components
--------------------

.. automodule:: cugraph.components.connectivity
    :members:
    :undoc-members:

Connected Components (MG)
--------------------

.. automodule:: cugraph.dask.components.connectivity
    :members:
    :undoc-members:

Cores
=====

Core Number
-----------

.. automodule:: cugraph.cores.core_number
    :members:
    :undoc-members:

K-Core
------

.. automodule:: cugraph.cores.k_core
    :members:
    :undoc-members:


Layout
======

Force Atlas 2
-------------

.. automodule:: cugraph.layout.force_atlas2
    :members:
    :undoc-members:


Linear Assignment
=================

Hungarian
-------------

.. automodule:: cugraph.linear_assignment.hungarian
    :members:
    :undoc-members:


Link Analysis
=============

HITS
---------

.. automodule:: cugraph.link_analysis.hits
    :members:
    :undoc-members:

Pagerank
---------

.. automodule:: cugraph.link_analysis.pagerank
    :members:
    :undoc-members:

Pagerank (MG)
-------------

.. automodule:: cugraph.dask.link_analysis.pagerank
    :members: pagerank
    :undoc-members:


Link Prediction
===============

Jaccard Coefficient
-------------------

.. automodule:: cugraph.link_prediction.jaccard
    :members:
    :undoc-members:

.. automodule:: cugraph.link_prediction.wjaccard
    :members:
    :undoc-members:

Overlap Coefficient
-------------------

.. automodule:: cugraph.link_prediction.overlap
    :members:
    :undoc-members:

.. automodule:: cugraph.link_prediction.woverlap
    :members:
    :undoc-members:


Sampling
========

Random Walks
------------

.. automodule:: cugraph.sampling.random_walks
    :members:
    :undoc-members:


Traversal
=========

Breadth-first-search
--------------------

.. automodule:: cugraph.traversal.bfs
    :members:
    :undoc-members:

Breadth-first-search (MG)
-------------------------

.. automodule:: cugraph.dask.traversal.bfs
    :members:
    :undoc-members:

Single-source-shortest-path
---------------------------

.. automodule:: cugraph.traversal.sssp
    :members:
    :undoc-members:

Single-source-shortest-path (MG)
--------------------------------

.. automodule:: cugraph.dask.traversal.sssp
    :members:
    :undoc-members:

Traveling-salesperson-problem
-----------------------------

.. automodule:: cugraph.traversal.traveling_salesperson
    :members:
    :undoc-members:


Tree
=========

Minimum Spanning Tree
---------------------

.. automodule:: cugraph.tree.minimum_spanning_tree
    :members: minimum_spanning_tree
    :undoc-members:

Maximum Spanning Tree
---------------------

.. automodule:: cugraph.tree.minimum_spanning_tree
    :members: maximum_spanning_tree
    :undoc-members:
    :noindex:


Generator
=========

RMAT
---------------------

.. automodule:: cugraph.generators
    :members: rmat
    :undoc-members:


DASK MG Helper functions
===========================

.. automodule:: cugraph.comms.comms
    :members: initialize, destroy
    :undoc-members:
    :member-order: bysource

.. automodule:: cugraph.dask.common.read_utils
    :members: get_chunksize
    :undoc-members:
