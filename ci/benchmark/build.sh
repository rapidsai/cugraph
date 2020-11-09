#!/usr/bin/env bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
##########################################
# cuGraph Benchmark test script for CI   #
##########################################

set -e
set -o pipefail
NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cleanup {
  gpuci_logger "Removing datasets and temp files"
  rm -rf $WORKSPACE/datasets/test
  rm -rf $WORKSPACE/datasets/benchmark
  rm -f testoutput.txt
}

# Set cleanup trap for Jenkins
if [ ! -z "$JENKINS_HOME" ] ; then
  gpuci_logger "Jenkins environment detected, setting cleanup trap"
  trap cleanup EXIT
fi

# Set path, build parallel level, and CUDA version
cd $WORKSPACE
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

# Set home
export HOME=$WORKSPACE

# Parse git describe
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Set Benchmark Vars
export DATASETS_DIR=${WORKSPACE}/datasets
export BENCHMARKS_DIR=${WORKSPACE}/benchmarks

##########################################
# Environment Setup                      #
##########################################

# TODO: Delete build section when artifacts are available

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Enter dependencies to be shown in ASV tooltips.
CUGRAPH_DEPS=(cudf rmm)
LIBCUGRAPH_DEPS=(cudf rmm)

gpuci_logger "Install required packages"
gpuci_conda_retry install -c nvidia -c rapidsai -c rapidsai-nightly -c conda-forge -c defaults \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "cudatoolkit=$CUDA_REL" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${MINOR_VERSION}" \
      "ucx-proc=*=gpu" \
      "rapids-build-env=${MINOR_VERSION}" \
      rapids-pytest-benchmark

gpuci_logger "Install the master version of dask and distributed"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

##########################################
# Build cuGraph                          #
##########################################

gpuci_logger "Build libcugraph"
$WORKSPACE/build.sh clean libcugraph cugraph

##########################################
# Run Benchmarks                         #
##########################################

gpuci_logger "Downloading Datasets for Benchmarks"
cd $DATASETS_DIR
bash ./get_test_data.sh --benchmark
ERRORCODE=$((ERRORCODE | $?))
# Exit if dataset download failed
if (( ${ERRORCODE} != 0 )); then
    exit ${ERRORCODE}
fi


# Concatenate dependency arrays, convert to JSON array,
# and remove duplicates.
X=("${CUGRAPH_DEPS[@]}" "${LIBCUGRAPH_DEPS[@]}")
DEPS=$(printf '%s\n' "${X[@]}" | jq -R . | jq -s 'unique')

# Build object with k/v pairs of "dependency:version"
DEP_VER_DICT=$(jq -n '{}')
for DEP in $(echo "${DEPS}" | jq -r '.[]'); do
  VER=$(conda list | grep "^${DEP}" | awk '{print $2"-"$3}')
  DEP_VER_DICT=$(echo "${DEP_VER_DICT}" | jq -c --arg DEP "${DEP}" --arg VER "${VER}" '. + { ($DEP): $VER }')
done

# Pass in an array of dependencies to get a dict of "dependency:version"
function getReqs() {
  local DEPS_ARR=("$@")
  local REQS="{}"
  for DEP in "${DEPS_ARR[@]}"; do
    VER=$(echo "${DEP_VER_DICT}" | jq -r --arg DEP "${DEP}" '.[$DEP]')
    REQS=$(echo "${REQS}" | jq -c --arg DEP "${DEP}" --arg VER "${VER}" '. + { ($DEP): $VER }')
  done

  echo "${REQS}"
}

REQS=$(getReqs "${CUGRAPH_DEPS[@]}")

BENCHMARK_META=$(jq -n \
  --arg NODE "${ASV_LABEL}" \
  --arg BRANCH "branch-${MINOR_VERSION}" \
  --argjson REQS "${REQS}" '
  {
    "machineName": $NODE,
    "commitBranch": $BRANCH,
    "requirements": $REQS
  }
')

echo "Benchmark meta:"
echo "${BENCHMARK_META}" | jq "."

gpuci_logger "Running Benchmarks"
cd $BENCHMARKS_DIR
set +e
time pytest -v -m "small and managedmem_on and poolallocator_on" \
    --benchmark-gpu-device=0 \
    --benchmark-gpu-max-rounds=3 \
    --benchmark-asv-output-dir="${S3_ASV_DIR}" \
    --benchmark-asv-metadata="${BENCHMARK_META}"



EXITCODE=$?

# The reqs below can be passed as requirements for
# C++ benchmarks in the future.
# REQS=$(getReqs "${LIBCUGRAPH_DEPS[@]}")

set -e
JOBEXITCODE=0
