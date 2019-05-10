#!/bin/bash

# ****************** Edit this *************************

#Path to nvgraph bin graphs
# From p4matrices:2024 sync //matrices/p4matrices/dimacs10/...
nvg_data_prefix="/home/mnaumov/cuda_matrices/p4matrices/dimacs10"
#nvg_data_prefix="/home/afender/modularity/mat"

#Path to nvgraph 
# nvg_bin_prefix should contain a release build of nvgraph's ToT (from p4sw //sw/gpgpu/nvgraph/...)
# and nvgraph_benchmark executable which is build along with nvgraph's tests
nvg_bin_prefix="/home/afender/modularity/sw/gpgpu/bin/x86_64_Linux_release"

# *****************************************************

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$nvg_bin_prefix
export PATH=$PATH:$nvg_bin_prefix

declare -a dataset=(
"$nvg_data_prefix/preferentialAttachment.mtx"
"$nvg_data_prefix/caidaRouterLevel.mtx"
"$nvg_data_prefix/coAuthorsDBLP.mtx"
"$nvg_data_prefix/citationCiteseer.mtx"
"$nvg_data_prefix/coPapersDBLP.mtx"
"$nvg_data_prefix/coPapersCiteseer.mtx"
"/home/afender/modularity/as-Skitter.mtx"
"/home/afender/modularity/hollywood-2009.mtx"
#"$nvg_data_prefix/data.mtx"
#"/home/afender/modularity/karate.mtx"
#"$nvg_data_prefix/road_central.mtx"
#"$nvg_data_prefix/road_usa.mtx"
#"$nvg_data_prefix/rgg_n_2_23_s0.mtx"
)

#One particular number of cluster
for i in "${dataset[@]}"
do
   $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" 7 7 --double --repeats 4
done
echo 
for i in "${dataset[@]}"
do
   $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" 7 7 --float --repeats 4
done
echo 

#Spreadsheet 1
#declare -ia clusters=(2 3 5 7 11 17 19 23 29 31 37 41 43 47 53)
#for i in "${dataset[@]}"
#do
#  for j in "${clusters[@]}"
#  do
#     if [  $j -lt 10 ]
#     then
#        $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" $j $j --double --repeats 4
#     else
#        $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" $j 7 --double --repeats 4
#     fi
#  done
#  echo
#done
#echo

#Spreadsheet 3 (same as 1 in single precision)
#declare -ia clusters=(2 3 5 7 11 17 19 23 29 31 37 41 43 47 53)
#for i in "${dataset[@]}"
#do
#  for j in "${clusters[@]}"
#  do
#     if [  $j -lt 10 ]
#     then
#        $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" $j $j --foat --repeats 4
#     else
#        $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" $j 7 --foat --repeats 4
#     fi
#  done
#  echo
#done

#run only best case according to Spreadsheet 1
#$nvg_bin_prefix/nvgraph_benchmark --modularity "$nvg_data_prefix/preferentialAttachment.mtx" 7 7 --double --repeats 4
#$nvg_bin_prefix/nvgraph_benchmark --modularity "$nvg_data_prefix/caidaRouterLevel.mtx" 11 7 --double --repeats 4
#$nvg_bin_prefix/nvgraph_benchmark --modularity "$nvg_data_prefix/coAuthorsDBLP.mtx" 7 7 --double --repeats 4
#$nvg_bin_prefix/nvgraph_benchmark --modularity "$nvg_data_prefix/citationCiteseer.mtx" 17 7 --double --repeats 4
#$nvg_bin_prefix/nvgraph_benchmark --modularity "$nvg_data_prefix/coPapersDBLP.mtx" 73 7 --double --repeats 4
#$nvg_bin_prefix/nvgraph_benchmark --modularity "$nvg_data_prefix/coPapersCiteseer.mtx" 53 7 --double --repeats 4
#$nvg_bin_prefix/nvgraph_benchmark --modularity "/home/afender/modularity/as-Skitter.mtx" 7 7 --double --repeats 4
#$nvg_bin_prefix/nvgraph_benchmark --modularity "/home/afender/modularity/hollywood-2009.mtx" 11 7 --double --repeats 4

#Variation of the number of  clusters  and number of eigenpairs, independently on synthetic matrix
#for (( i = 2; i <= 8; i++ )) 
#do
#   for (( j = $i ; j <= 32; j++ ))
#   do
#         $nvg_bin_prefix/nvgraph_benchmark --modularity "/home/afender/modularity/karate_5_block_dia.mtx" $j $i --double --repeats 3
#   done
#   echo
#done
#echo 

#profiles
#nvprof --profile-from-start off --export-profile coPapersDBLP.mtx_23clusters_3ev_32b.bin /home/afender/modularity/sw/gpgpu/bin/x86_64_Linux_release/nvgraph_benchmark --modularity "/home/mnaumov/cuda_matrices/p4matrices/dimacs10/coPapersDBLP.mtx" 23 3 --double --repeats 3
# /home/mnaumov/cuda_toolkit/cuda-linux64-mixed-rel-nightly/bin/nvprof --profile-from-start off --export-profile eigensolver_coPapersDBLP.mtx_4clusters_4ev_32b.bin /home/afender/modularity/sw/gpgpu/bin/x86_64_Linux_release/nvgraph_benchmark --modularity "/home/mnaumov/cuda_matrices/p4matrices/dimacs10/coPapersDBLP.mtx" 4 4 --double --repeats 1
# /home/mnaumov/cuda_toolkit/cuda-linux64-mixed-rel-nightly/bin/nvprof --profile-from-start off --export-profile total_coPapersDBLP.mtx_4clusters_4ev_32b.bin /home/afender/modularity/sw/gpgpu/bin/x86_64_Linux_release/nvgraph_benchmark --modularity "/home/mnaumov/cuda_matrices/p4matrices/dimacs10/coPapersDBLP.mtx" 4 4 --double --repeats 1

#small matrices
#declare -a dataset_small=(
#"$nvg_data_prefix/karate.mtx"
#"$nvg_data_prefix/dolphins.mtx"
##"$nvg_data_prefix/chesapeake.mtx"
#"$nvg_data_prefix/lesmis.mtx"
#"$nvg_data_prefix/adjnoun.mtx"
#"$nvg_data_prefix/polbooks.mtx"
#"$nvg_data_prefix/football.mtx"
#"$nvg_data_prefix/celegansneural.mtx"
##"$nvg_data_prefix/jazz.mtx"
#"$nvg_data_prefix/netscience.mtx"
##"$nvg_data_prefix/email.mtx"
#"$nvg_data_prefix/power.mtx"
#"$nvg_data_prefix/hep-th.mtx"
#"$nvg_data_prefix/polblogs.mtx"
##"$nvg_data_prefix/PGPgiantcompo.mtx"
#"$nvg_data_prefix/cond-mat.mtx"
#"$nvg_data_prefix/as-22july06.mtx"
#"$nvg_data_prefix/cond-mat-2003.mtx"
#"$nvg_data_prefix/astro-ph.mtx"
#)
#declare -ia clusters=(2 3 5 7 11 17 19 23 29 31)
#for i in "${dataset_small[@]}"
#do
#  for j in "${clusters[@]}"
#  do
#     if [  $j -lt 10 ]
#     then
#        $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" $j $j --double --repeats 4
#     else
#        $nvg_bin_prefix/nvgraph_benchmark --modularity "$i" $j 7 --double --repeats 4
#     fi
#  done
#  echo
#done
#echo
