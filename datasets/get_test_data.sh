#!/bin/bash
set -e
set -o pipefail

# Update this to add/remove/change a dataset, using the following format:
#
#  comment about the dataset
#  dataset download URL
#  destination dir to untar to
#  blank line separator
#
# FIXME: some test data needs to be extracted to "benchmarks", which is
# confusing now that there's dedicated datasets for benchmarks.
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

BENCHMARK_DATASET_DATA="
# ~90s download - these are used for benchmarks runs (code in <cugraph root>/benchmarks)
https://rapidsai-data.s3.us-east-2.amazonaws.com/cugraph/benchmark/benchmark_csv_data.tgz
csv
"
################################################################################
# Do not change the script below this line if only adding/updating a dataset

NUMARGS=$#
ARGS=$*
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "$0 [--subset | --benchmark]"
    exit 0
fi

# Select the datasets to install
if hasArg "--benchmark"; then
    DATASET_DATA="${BENCHMARK_DATASET_DATA}"
elif hasArg "--subset"; then
    DATASET_DATA="${BASE_DATASET_DATA}"
# Do not include benchmark datasets by default - too big
else
    DATASET_DATA="${BASE_DATASET_DATA} ${EXTENDED_DATASET_DATA}"
fi

URLS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 3) print $0}'))  # extract 3rd fields to a bash array
DESTDIRS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 0) print $0}'))  # extract 4th fields to a bash array

echo Downloading ...

# Download all tarfiles to a tmp dir
rm -rf tmp
mkdir tmp
cd tmp
for url in ${URLS[*]}; do
   time wget --progress=dot:giga ${url}
done
cd ..

# Setup the destination dirs, removing any existing ones first!
for index in ${!DESTDIRS[*]}; do
    rm -rf ${DESTDIRS[$index]}
done
for index in ${!DESTDIRS[*]}; do
    mkdir -p ${DESTDIRS[$index]}
done

# Iterate over the arrays and untar the nth tarfile to the nth dest directory.
# The tarfile name is derived from the download url.
echo Decompressing ...
for index in ${!DESTDIRS[*]}; do
    tfname=$(basename ${URLS[$index]})
    tar xvzf tmp/${tfname} -C ${DESTDIRS[$index]}
done

rm -rf tmp
