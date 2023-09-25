========================
Operators on Full Graphs
========================

.. currentmodule:: pylibcugraphops

Simple Neighborhood Aggregator (SAGEConv)
-----------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.agg_simple_fg_n2n_fwd
   operators.agg_simple_fg_n2n_bwd
   operators.agg_simple_fg_e2n_fwd
   operators.agg_simple_fg_e2n_bwd
   operators.agg_simple_fg_n2n_e2n_fwd
   operators.agg_simple_fg_n2n_e2n_bwd

   operators.agg_concat_fg_n2n_fwd
   operators.agg_concat_fg_n2n_bwd
   operators.agg_concat_fg_e2n_fwd
   operators.agg_concat_fg_e2n_bwd
   operators.agg_concat_fg_n2n_e2n_fwd
   operators.agg_concat_fg_n2n_e2n_bwd

Weighted Neighborhood Aggregation
---------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.agg_weighted_fg_n2n_fwd
   operators.agg_weighted_fg_n2n_bwd
   operators.agg_concat_weighted_fg_n2n_fwd
   operators.agg_concat_weighted_fg_n2n_bwd

Heterogenous Aggregator using Basis Decomposition (RGCNConv)
------------------------------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.agg_hg_basis_fg_n2n_post_fwd
   operators.agg_hg_basis_fg_n2n_post_bwd

Graph Attention (GATConv/GATv2Conv)
-----------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.mha_gat_fg_n2n_fwd
   operators.mha_gat_fg_n2n_bwd
   operators.mha_gat_fg_n2n_efeat_fwd
   operators.mha_gat_fg_n2n_efeat_bwd

   operators.mha_gat_v2_fg_n2n_fwd
   operators.mha_gat_v2_fg_n2n_bwd
   operators.mha_gat_v2_fg_n2n_efeat_fwd
   operators.mha_gat_v2_fg_n2n_efeat_bwd

Transformer-like Graph Attention (TransformerConv)
--------------------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.mha_gat_v2_fg_n2n_fwd
   operators.mha_gat_v2_fg_n2n_bwd
   operators.mha_gat_v2_fg_n2n_efeat_fwd
   operators.mha_gat_v2_fg_n2n_efeat_bwd

Directional Message-Passing (DMPNN)
-----------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.agg_dmpnn_fg_e2e_fwd
   operators.agg_dmpnn_fg_e2e_bwd

Graph Pooling
-------------
.. autosummary::
   :toctree: ../api/ops/

   operators.pool_fg_n2s_fwd
   operators.pool_fg_n2s_bwd
