#!/bin/bash

# Update this to add/remove/change a dataset, using the following format:
#
#  comment about the dataset
#  dataset download URL
#  destination dir to untar to
#  blank line separator
DATASET_DATA="
# ~22s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/datasets.tgz
test

# ~14s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/ref/pagerank.tgz
test/ref

# ~1s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/ref/sssp.tgz
test/ref

# ~42s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_huge.tgz
benchmark

# ~15s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_large.tgz
benchmark

# ~1s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_small.tgz
benchmark
"

################################################################################
# Do not change the script below this line if only adding/updating a dataset
URLS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 3) print $0}'))  # extract 3rd fields to a bash array
DESTDIRS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 0) print $0}'))  # extract 4th fields to a bash array

echo Downloading ...
mkdir tmp
cd tmp
for url in ${URLS[*]}; do
   time wget --progress=dot:giga ${url}
done
cd ..

rm -rf test
rm -rf benchmark
mkdir -p test/ref
mkdir benchmark

echo Decompressing ...
for index in ${!DESTDIRS[*]}; do
    basename=$(basename ${URLS[$index]})
    tar xvzf tmp/${basename} -C ${DESTDIRS[$index]}
done

rm -rf tmp
