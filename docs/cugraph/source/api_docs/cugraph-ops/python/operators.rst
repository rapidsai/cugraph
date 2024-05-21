=============================
Operators for Message-Passing
=============================

.. currentmodule:: pylibcugraphops

Simple Neighborhood Aggregator (SAGEConv)
-----------------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.agg_simple_n2n_fwd
   operators.agg_simple_n2n_bwd
   operators.agg_simple_e2n_fwd
   operators.agg_simple_e2n_bwd
   operators.agg_simple_n2n_e2n_fwd
   operators.agg_simple_n2n_e2n_bwd

   operators.agg_concat_n2n_fwd
   operators.agg_concat_n2n_bwd
   operators.agg_concat_e2n_fwd
   operators.agg_concat_e2n_bwd
   operators.agg_concat_n2n_e2n_fwd
   operators.agg_concat_n2n_e2n_bwd


Weighted Neighborhood Aggregation
---------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.agg_weighted_n2n_fwd
   operators.agg_weighted_n2n_bwd
   operators.agg_concat_weighted_n2n_fwd
   operators.agg_concat_weighted_n2n_bwd

Heterogenous Aggregator using Basis Decomposition (RGCNConv)
------------------------------------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.agg_hg_basis_n2n_post_fwd
   operators.agg_hg_basis_n2n_post_bwd

Graph Attention (GATConv/GATv2Conv)
-----------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.mha_gat_n2n_fwd_bf16_fp32
   operators.mha_gat_n2n_fwd_fp16_fp32
   operators.mha_gat_n2n_fwd_fp32_fp32
   operators.mha_gat_n2n_bwd_bf16_bf16_bf16_fp32
   operators.mha_gat_n2n_bwd_bf16_bf16_fp32_fp32
   operators.mha_gat_n2n_bwd_bf16_fp32_fp32_fp32
   operators.mha_gat_n2n_bwd_fp16_fp16_fp16_fp32
   operators.mha_gat_n2n_bwd_fp16_fp16_fp32_fp32
   operators.mha_gat_n2n_bwd_fp16_fp32_fp32_fp32
   operators.mha_gat_n2n_bwd_fp32_fp32_fp32_fp32
   operators.mha_gat_n2n_efeat_fwd_bf16_fp32
   operators.mha_gat_n2n_efeat_fwd_fp16_fp32
   operators.mha_gat_n2n_efeat_fwd_fp32_fp32
   operators.mha_gat_n2n_efeat_bwd_bf16_bf16_bf16_fp32
   operators.mha_gat_n2n_efeat_bwd_bf16_bf16_fp32_fp32
   operators.mha_gat_n2n_efeat_bwd_bf16_fp32_fp32_fp32
   operators.mha_gat_n2n_efeat_bwd_fp16_fp16_fp16_fp32
   operators.mha_gat_n2n_efeat_bwd_fp16_fp16_fp32_fp32
   operators.mha_gat_n2n_efeat_bwd_fp16_fp32_fp32_fp32
   operators.mha_gat_n2n_efeat_bwd_fp32_fp32_fp32_fp32

   operators.mha_gat_v2_n2n_fwd
   operators.mha_gat_v2_n2n_bwd
   operators.mha_gat_v2_n2n_efeat_fwd
   operators.mha_gat_v2_n2n_efeat_bwd

Transformer-like Graph Attention (TransformerConv)
--------------------------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.mha_gat_v2_n2n_fwd
   operators.mha_gat_v2_n2n_bwd
   operators.mha_gat_v2_n2n_efeat_fwd
   operators.mha_gat_v2_n2n_efeat_bwd

Directional Message-Passing (DMPNN)
-----------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.agg_dmpnn_e2e_fwd
   operators.agg_dmpnn_e2e_bwd

Update Edges: Concatenation or Sum of Edge and Node Features
------------------------------------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.update_efeat_e2e_concat_fwd
   operators.update_efeat_e2e_concat_bwd

   operators.update_efeat_e2e_sum_fwd
   operators.update_efeat_e2e_sum_bwd

   operators.update_efeat_e2e_concat_fwd
   operators.update_efeat_e2e_concat_bwd

   operators.update_efeat_e2e_sum_fwd
   operators.update_efeat_e2e_sum_bwd
