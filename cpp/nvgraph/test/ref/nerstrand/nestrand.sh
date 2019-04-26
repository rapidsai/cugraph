#!/bin/bash

nvg_data_prefix="/home/mnaumov/cuda_matrices/p4matrices/dimacs10"

declare -a dataset=(
"$nvg_data_prefix/preferentialAttachment.mtx"
"$nvg_data_prefix/caidaRouterLevel.mtx"
"$nvg_data_prefix/coAuthorsDBLP.mtx"
"$nvg_data_prefix/citationCiteseer.mtx"
"$nvg_data_prefix/coPapersDBLP.mtx"
"$nvg_data_prefix/coPapersCiteseer.mtx"
"/home/afender/modularity/as-Skitter.mtx"
"/home/afender/modularity/hollywood-2009.mtx"
)

for i in "${dataset[@]}"
do
   ./nerstrand_bench "$i" 7 
done
echo 

#run only best case according to Spreadsheet 1
./nerstrand_bench "$nvg_data_prefix/preferentialAttachment.mtx" 7
./nerstrand_bench "$nvg_data_prefix/caidaRouterLevel.mtx" 11
./nerstrand_bench "$nvg_data_prefix/coAuthorsDBLP.mtx" 7
./nerstrand_bench "$nvg_data_prefix/citationCiteseer.mtx" 17
./nerstrand_bench "$nvg_data_prefix/coPapersDBLP.mtx" 73
./nerstrand_bench "$nvg_data_prefix/coPapersCiteseer.mtx" 53
./nerstrand_bench "/home/afender/modularity/as-Skitter.mtx" 7
./nerstrand_bench "/home/afender/modularity/hollywood-2009.mtx" 11
