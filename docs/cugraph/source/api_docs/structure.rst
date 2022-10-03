=============
Graph Classes
=============
.. currentmodule:: cugraph

Constructors
------------
.. autosummary::
   :toctree: api/

   Graph
   MultiGraph
   BiPartiteGraph



Adding Data
-----------
.. autosummary::
   :toctree: api/


   Graph.from_cudf_adjlist
   Graph.from_cudf_edgelist
   Graph.from_dask_cudf_edgelist
   Graph.from_pandas_adjacency
   Graph.from_pandas_edgelist
   Graph.from_numpy_array
   Graph.from_numpy_matrix
   Graph.add_internal_vertex_id
   Graph.add_nodes_from
   Graph.clear
   Graph.unrenumber

Checks
------
.. autosummary::
   :toctree: api/

   Graph.has_isolated_vertices
   Graph.is_bipartite
   Graph.is_directed
   Graph.is_multigraph
   Graph.is_multipartite
   Graph.is_renumbered
   Graph.is_weighted
   Graph.lookup_internal_vertex_id
   Graph.to_directed
   Graph.to_undirected


Symmetrize
----------
.. autosummary::
   :toctree: api/

   cugraph.symmetrize
   cugraph.symmetrize_ddf
   cugraph.symmetrize_df


Conversion from Other Formats
-----------------------------
.. autosummary::
   :toctree: api/

   cugraph.from_adjlist
   cugraph.from_cudf_edgelist
   cugraph.from_edgelist
   cugraph.from_numpy_array
   cugraph.from_numpy_matrix
   cugraph.from_pandas_adjacency
   cugraph.from_pandas_edgelist
   cugraph.to_numpy_array
   cugraph.to_numpy_matrix
   cugraph.to_pandas_adjacency
   cugraph.to_pandas_edgelist

NumberMap
-----------------------------
.. autosummary::
   :toctree: api/

   cugraph.structure.NumberMap
   cugraph.structure.NumberMap.MultiGPU
   cugraph.structure.NumberMap.SingleGPU
   cugraph.structure.NumberMap.from_internal_vertex_id
   cugraph.structure.NumberMap.to_internal_vertex_id
   cugraph.structure.NumberMap.add_internal_vertex_id
   cugraph.structure.NumberMap.compute_vals
   cugraph.structure.NumberMap.compute_vals_types
   cugraph.structure.NumberMap.generate_unused_column_name
   cugraph.structure.NumberMap.renumber
   cugraph.structure.NumberMap.renumber_and_segment
   cugraph.structure.NumberMap.set_renumbered_col_names
   cugraph.structure.NumberMap.unrenumber
   cugraph.structure.NumberMap.vertex_column_size

Other
-----------------------------
.. autosummary::
   :toctree: api/

   cugraph.hypergraph
   cugraph.structure.shuffle
