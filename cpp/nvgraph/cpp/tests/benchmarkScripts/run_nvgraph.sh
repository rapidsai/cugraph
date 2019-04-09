#!/bin/bash

# ****************** Edit this *************************
#Path to nvgraph bin graphs
# From p4matrices:2024 sync //matrices/p4matrices/graphs/...
nvg_data_prefix="/home/afender/src/matrices/p4matrices/graphs"

#Path to nvgraph 
# nvg_bin_prefix should contain a release build of nvgraph's ToT (from p4sw //sw/gpgpu/nvgraph/...)
# and nvgraph_benchmark executable which is build along with nvgraph's tests
nvg_bin_prefix="/home/afender/src/sw/sw/gpgpu/bin/x86_64_Linux_release"
# *****************************************************

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$nvg_bin_prefix
export PATH=$PATH:$nvg_bin_prefix

declare -a arr=(
 "$nvg_data_prefix/webbase1M/webbase-1M_T.mtx.bin"
 "$nvg_data_prefix/liveJournal/ljournal-2008_T.mtx.bin" 
 "$nvg_data_prefix/webGoogle/web-Google_T.mtx.bin"
 "$nvg_data_prefix/citPatents/cit-Patents_T.mtx.bin"
 "$nvg_data_prefix/webBerkStan/web-BerkStan_T.mtx.bin"
 "$nvg_data_prefix/WikiTalk/wiki-Talk_T.mtx.bin"
 "$nvg_data_prefix/soc-liveJournal/soc-LiveJournal1_T.mtx.bin"
 # Warning  : Twitter case works only on GPU with more than 12 GB of memory
 "$nvg_data_prefix/Twitter/twitter.bin"
 #Just for debug
 #"$nvg_data_prefix/small/small.bin"
)


## now loop through the above array
for i in "${arr[@]}"
do
   echo "Pagerank"
   echo "$i" 
   echo "single precision"
   $nvg_bin_prefix/nvgraph_benchmark --pagerank "$i" 0.85 500 1E-6 --float --repeats 10
   echo 
   #echo "Pagerank"
   #echo "$i" 
   #echo "double precision"
   #$nvg_bin_prefix/nvgraph_benchmark --pagerank "$i" 0.85 500 1E-6 --double --repeats 10
   #echo 
done
echo 
for i in "${arr[@]}"
do
   echo "SSSP"
   echo "$i" 
   echo "single precision"
   $nvg_bin_prefix/nvgraph_benchmark --sssp "$i" 0 --float --repeats 10
   echo 
   #echo "SSSP"
   #echo "$i" 
   #echo "double precision"
   #$nvg_bin_prefix/nvgraph_benchmark --sssp "$i" 0 --double --repeats 10
   #echo 
done
echo 
for i in "${arr[@]}"
do
   echo "Widest Path"
   echo "$i" 
   echo "single precision"
   $nvg_bin_prefix/nvgraph_benchmark --widest "$i" 0 --float --repeats 10
   echo 
   #echo "Widest Path"
   #echo "$i" 
   #echo "double precision"
   #$nvg_bin_prefix/nvgraph_benchmark --widest "$i" 0 --double --repeats 10
   #echo 
done
echo 
