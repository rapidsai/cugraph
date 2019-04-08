#!/bin/bash

# ****************** Edit this *************************
#*******************************************************
#Path to graphMat binary data
gm_data_prefix="/home-2/afender/GraphMat-master/data"
#Path to graphMat binary 
gm_bin_prefix="/home-2/afender/GraphMat-master/bin"
#Number of core to use in graphMat
export OMP_NUM_THREADS=24
# ******************************************************
#*******************************************************
#  NOTE 
#twitter_graphMat.bin and live_journal_graphMat.bin are assumed to be in "gm_data_prefix" directory
#*******************************************************

# Requiered export according to the doc
export KMP_AFFINITY=scatter

#Pagerank runs
numactl -i all $gm_bin_prefix/PageRank $gm_data_prefix/twitter.graphmat.bin
numactl -i all $gm_bin_prefix/PageRank $gm_data_prefix/soc-LiveJournal1.graphmat.bin

# SSSP runs 
# Warning: vertices seems to have 1-based indices (nvGraph use 0-base)
numactl -i all $gm_bin_prefix/SSSP $gm_data_prefix/twitter.graphmat.bin 1
numactl -i all $gm_bin_prefix/SSSP $gm_data_prefix/soc-LiveJournal1.graphmat.bin 1