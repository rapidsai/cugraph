#!/bin/bash
set -e
set -o pipefail
NUMARGS=$#
ARGS=$*

# FIXME: consider using getopts for option parsing
# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Update this to add/remove/change a dataset, using the following format:
#
#  comment about the dataset
#  dataset download URL
#  destination dir to untar to
#  blank line separator
BASE_DATASET_DATA="
# ~22s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/datasets.tgz
test

# ~14s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/ref/pagerank.tgz
test/ref

# ~1s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/test/ref/sssp.tgz
test/ref

# ~15s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_large.tgz
benchmark

# ~1s download
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_small.tgz
benchmark
"

EXTENDED_DATASET_DATA="
# ~42s download - tests using this dataset are currently not run in test.sh with --quick
https://s3.us-east-2.amazonaws.com/rapidsai-data/cugraph/benchmark/hibench/hibench_1_huge.tgz
benchmark
"

# Select the datasets to install
if hasArg "--subset"; then
    DATASET_DATA="${BASE_DATASET_DATA}"
else
    DATASET_DATA="${BASE_DATASET_DATA} ${EXTENDED_DATASET_DATA}"
fi

################################################################################
# Do not change the script below this line if only adding/updating a dataset
URLS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 3) print $0}'))  # extract 3rd fields to a bash array
DESTDIRS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 0) print $0}'))  # extract 4th fields to a bash array

echo Downloading ...
rm -rf tmp
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

# Iterate over the arrays and untar the nth tarfile to the nth dest directory.
# The tarfile name is derived from the download url.
echo Decompressing ...
for index in ${!DESTDIRS[*]}; do
    tfname=$(basename ${URLS[$index]})
    tar xvzf tmp/${tfname} -C ${DESTDIRS[$index]}
done

rm -rf tmp
