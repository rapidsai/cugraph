#!/bin/sh

edges="$1"
echo "Starting Sort on $edges..."
./sort $edges
echo "Done"

tmp="_s"
sedges=$edges$tmp
echo "Starting H on $sedges ..."
./H $sedges
echo "Done"

tmp="_mtx"
matrix=$sedges$tmp
#delete soted edges
rm $sedges

echo "Starting HTa on $matrix ..."
./HTA $matrix

tmp="_T"
outp=$edges$tmp
outpp=$matrix$tmp
mv $outpp $outp
#delete H
rm $matrix

echo "Starting binary conversion ..."
./mtob $outp
echo "Done"

