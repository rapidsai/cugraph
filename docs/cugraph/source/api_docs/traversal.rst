=========
Traversal
=========
.. currentmodule:: cugraph



Breadth-first-search
--------------------
.. autosummary::
   :toctree: api/

   cugraph.traversal.bfs.bfs
   cugraph.traversal.bfs.bfs_edges

Breadth-first-search (MG)
-------------------------
.. autosummary::
   :toctree: api/

   cugraph.dask.traversal.bfs.bfs
   cugraph.dask.traversal.bfs.call_bfs


Single-source-shortest-path
---------------------------
.. autosummary::
   :toctree: api/

   cugraph.traversal.sssp.filter_unreachable
   cugraph.traversal.sssp.shortest_path
   cugraph.traversal.sssp.shortest_path_length
   cugraph.traversal.sssp.sssp

Single-source-shortest-path (MG)
--------------------------------
.. autosummary::
   :toctree: api/

   cugraph.dask.traversal.sssp.call_sssp
   cugraph.dask.traversal.sssp.sssp


Traveling-salesperson-problem
-----------------------------
.. autosummary::
   :toctree: api/

   cugraph.traversal.traveling_salesperson.traveling_salesperson
