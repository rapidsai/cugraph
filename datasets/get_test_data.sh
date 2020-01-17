#!/bin/bash

echo Downloading ...
mkdir tmp
cd tmp
wget --progress=dot:giga https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/datasets.tgz
wget --progress=dot:giga https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/ref/pagerank.tgz
wget --progress=dot:giga https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/ref/sssp.tgz
wget --progress=dot:giga https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_huge.tgz
wget --progress=dot:giga https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_large.tgz
wget --progress=dot:giga https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_small.tgz
cd ..

mkdir test
mkdir test/ref
mkdir benchmark

echo Decompressing ...
tar xvzf tmp/datasets.tgz -C test
tar xvzf tmp/pagerank.tgz -C test/ref
tar xvzf tmp/sssp.tgz -C test/ref
tar xvzf tmp/hibench_1_huge.tgz -C benchmark
tar xvzf tmp/hibench_1_large.tgz -C benchmark
tar xvzf tmp/hibench_1_small.tgz -C benchmark

rm -rf tmp

export RAPIDS_DATASET_ROOT_DIR=$PWD
echo RAPIDS_DATASET_ROOT_DIR was set to $PWD
