#!/bin/bash

echo Downloading ...
mkdir tmp
cd tmp
wget https://s3.us-east-2.amazonaws.com/gpuci/cugraph/test/datasets.tgz
wget https://s3.us-east-2.amazonaws.com/gpuci/cugraph/test/ref/pagerank.tgz
wget https://s3.us-east-2.amazonaws.com/gpuci/cugraph/test/ref/sssp.tgz
cd ..

mkdir test
mkdir test/ref

echo Decompressing ...
tar xvzf tmp/datasets.tar.gz -C test
tar xvzf tmp/pagerank.tar.gz -C test/ref
tar xvzf tmp/sssp.tar.gz -C test/ref

rm -rf tmp

export RAPIDS_DATASET_ROOT_DIR=$PWD
echo RAPIDS_DATASET_ROOT_DIR was set to $PWD
