#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##########################################
# cuGraph Benchmark test script for CI   #
##########################################

set -e
set -o pipefail
NUMARGS=$#
ARGS=$*

function logger {
  echo -e "\n>>>> $@\n"
}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cleanup {
  logger "Removing datasets and temp files..."
  rm -rf $WORKSPACE/datasets/test
  rm -rf $WORKSPACE/datasets/benchmark
  rm -f testoutput.txt
}

# Set cleanup trap for Jenkins
if [ ! -z "$JENKINS_HOME" ] ; then
  logger "Jenkins environment detected, setting cleanup trap..."
  trap cleanup EXIT
fi

# Set path, build parallel level, and CUDA version
cd $WORKSPACE
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}
export HOME=$WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Set Benchmark Vars
export DATASETS_DIR=${WORKSPACE}/datasets
export ASVRESULTS_DIR=${WORKSPACE}/ci/artifacts/asv/results
export BENCHMARKS_DIR=${WORKSPACE}/benchmarks

# Ensure ASV results directory exists

mkdir -p ${ASVRESULTS_DIR}

##########################################
# Environment Setup                      #
##########################################

# TODO: Delete build section when artifacts are available

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate rapids


# Enter dependencies to be shown in ASV tooltips.
# These dependencies will also be installed by conda.
# A dependency can exist in both arrays without causing issues.
CUGRAPH_DEPS=(cudf rmm)
LIBCUGRAPH_DEPS=(cudf rmm)

# Concatenate dependency arrays, convert to JSON array,
# and remove duplicates.
X=("${CUGRAPH_DEPS[@]}" "${LIBCUGRAPH_DEPS[@]}")
DEPS=$(printf '%s\n' "${X[@]}" | jq -R . | jq -s 'unique')

# Create install args for conda (i.e. "cudf=0.15" "rmm=0.15")
CONDA_INSTALL=
for DEP in $(echo "${DEPS}" | jq -r '.[]'); do
  CONDA_INSTALL+="${DEP}=${MINOR_VERSION}"
  CONDA_INSTALL+=" "
done

logger "conda install required packages"
conda install -c nvidia -c rapidsai -c rapidsai-nightly -c conda-forge -c defaults \
      ${CONDA_INSTALL} \
      "cudatoolkit=$CUDA_REL" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${MINOR_VERSION}" \
      "rapids-build-env=${MINOR_VERSION}" \
      rapids-pytest-benchmark

# Install the master version of dask and distributed
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps

logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

##########################################
# Build cuGraph                          #
##########################################

logger "Build libcugraph..."
$WORKSPACE/build.sh clean libcugraph cugraph

##########################################
# Run Benchmarks                         #
##########################################

logger "Downloading Datasets for Benchmarks..."
cd $DATASETS_DIR
bash ./get_test_data.sh --benchmark
ERRORCODE=$((ERRORCODE | $?))
# Exit if dataset download failed
if (( ${ERRORCODE} != 0 )); then
    exit ${ERRORCODE}
fi

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

logger "Running Benchmarks..."
cd $BENCHMARKS_DIR
set +e
time pytest -v -m "small and managedmem_on and poolallocator_on" \
    --benchmark-gpu-device=0 \
    --benchmark-gpu-max-rounds=3 \
    --benchmark-asv-metadata="machineName=${NODE_NAME}, commitBranch=branch-${MINOR_VERSION}, requirements=${REQS}" \
    --benchmark-asv-output-dir=${ASVRESULTS_DIR}



EXITCODE=$?

# libcugraph reqs
# REQS=$(getReqs "${LIBCUGRAPH_DEPS[@]}")

set -e
JOBEXITCODE=0
