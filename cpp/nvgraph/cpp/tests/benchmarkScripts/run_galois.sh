#!/bin/bash

# ****************** Edit this *************************
# Path to local workspace containing p4matrices:2024 sync //matrices/p4matrices/graphs/...
nvg_data_prefix="/home/afender/src/matrices/p4matrices/graphs"

#Path to galois 
galois_root="/home/afender/soft/galois-2.3.0/build/default"
# *****************************************************
export OMP_NUM_THREADS=24

declare -a arr=(
 #Small mtx just for debug
 #"$nvg_data_prefix/small/small.mtx"
 "$nvg_data_prefix/soc-liveJournal/soc-LiveJournal1.mtx"
 "$nvg_data_prefix/Twitter/twitter.mtx"
)

## now loop through the above array
for i in "${arr[@]}"
do
   echo "Pagerank"
   echo "$i" 
   time $galois_root/tools/graph-convert/graph-convert -mtx2gr -edgeType=float32 -print-all-options $i $i.galois
   time $galois_root/tools/graph-convert/graph-convert -gr2tgr -edgeType=float32 -print-all-options $i.galois $i_T.galois
   time $galois_root/apps/pagerank/app-pagerank $i.galois -graphTranspose="$i_T.galois" -t=$OMP_NUM_THREADS
   echo 
done
echo 
for i in "${arr[@]}"
do
   echo "SSSP"
   echo "$i" 
   time $galois_root/apps/sssp/app-sssp $i.galois -startNode=0 -t=$OMP_NUM_THREADS
   echo 
done
echo 
