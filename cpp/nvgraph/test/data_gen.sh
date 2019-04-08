#!/bin/sh
#Usage sh data_gen size1 size2 ...
#Generate power law in-degree plus rmat graphs of size size1 ... sizeN
#Corresponding transposed and binary csr are generated as well

convert (){
edges=$1
#echo "Starting Sort on $edges..."
./generators/convertors/sort $edges
#echo "Done"

tmp="_s"
sedges=$edges$tmp
echo "Starting H on $sedges ..."
./generators/convertors/H $sedges
#echo "Done"

tmp="_mtx"
matrix=$sedges$tmp
#delete soted edges
rm $sedges

echo "Starting HTa on $matrix ..."
./generators/convertors/HTA $matrix

tmp="_T"
outp=$edges$tmp
outpp=$matrix$tmp
mv $outpp $outp
#delete H
rm $matrix

#echo "Starting binary conversion ..."
./generators/convertors/mtob $outp
#echo "Generated transposed coo and transposed csr bin"
}

echo "Building the tools ..."
make -C generators
make -C generators/convertors
#generate the graphs we need here
#loop over script arguments which represent graph sizes.
for var in "$@"
do
echo "Generate graphs of size $var"
vertices=$var
option="i"
./generators/plodg $vertices $option
./generators/rmatg $vertices $option
graph="plod_graph_"
format=".mtx"
path_to_data="local_test_data/"
name="$path_to_data$graph$vertices$format"
convert $name
graph="rmat_graph_"
name="$path_to_data$graph$vertices$format"
convert $name
done
