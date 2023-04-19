==========================
PyTorch Autograd Wrappers
==========================

.. currentmodule:: pylibcugraphops

Simple Neighborhood Aggregator (SAGEConv)
-----------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   pytorch.operators.agg_concat_n2n

Graph Attention (GATConv/GATv2Conv)
-----------------------------------
.. autosummary::
   :toctree: ../api/ops/

   pytorch.operators.mha_gat_n2n
   pytorch.operators.mha_gat_v2_n2n

Heterogenous Aggregator using Basis Decomposition (RGCNConv)
------------------------------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   pytorch.operators.agg_hg_basis_n2n_post


Update Edges: Concatenation or Sum of Edge and Node Features
------------------------------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   pytorch.operators.update_efeat_bipartite_e2e
   pytorch.operators.update_efeat_static_e2e
