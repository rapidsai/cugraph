#!/usr/bin/env bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
##########################################
# cuGraph GPU build & testscript for CI  #
##########################################
set -e           # abort the script on error, this will change for running tests (see below)
set -o pipefail  # piped commands propagate their error
NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path, build parallel level, and CUDA version
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}
export CONDA_ARTIFACT_PATH=${WORKSPACE}/ci/artifacts/cugraph/cpu/.conda-bld/

# Workaround to keep Jenkins builds working
# until we migrate fully to GitHub Actions
export RAPIDS_CUDA_VERSION="${CUDA}"
export SCCACHE_BUCKET=rapids-sccache
export SCCACHE_REGION=us-west-2
export SCCACHE_IDLE_TIMEOUT=32768

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

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
unset GIT_DESCRIBE_TAG

# ucx-py version
export UCX_PY_VERSION='0.30.*'

# Whether to keep `dask/label/dev` channel in the env. If INSTALL_DASK_MAIN=0,
# `dask/label/dev` channel is removed.
export INSTALL_DASK_MAIN=1

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids
export PATH=$(conda info --base)/envs/rapids/bin:$PATH


# Remove `dask/label/dev` channel if INSTALL_DASK_MAIN=0
if [[ "${INSTALL_DASK_MAIN}" == 0 ]]; then
  conda config --system --remove channels dask/label/dev
fi

gpuci_logger "Install dependencies"
# Assume libcudf and librmm will be installed via cudf and rmm respectively.
# This is done to prevent the following install scenario:
# libcudf = 22.04.00a220315, cudf = 22.04.00a220308
# where cudf 220308 was chosen possibly because it has fewer/different
# dependencies and the corresponding recipes have specified these combinations
# should work when sometimes they do not.
# FIXME: remove testing label when gpuCI has the ability to move the pyraft
# label from testing to main.
gpuci_mamba_retry install -c rapidsai-nightly/label/testing -y \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "libraft-headers=${MINOR_VERSION}" \
      "libraft-distance=${MINOR_VERSION}" \
      "pylibraft=${MINOR_VERSION}" \
      "raft-dask=${MINOR_VERSION}" \
      "cudatoolkit=$CUDA_REL" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${UCX_PY_VERSION}" \
      "ucx-proc=*=gpu" \
      "rapids-build-env=$MINOR_VERSION.*" \
      "rapids-notebook-env=$MINOR_VERSION.*" \
      "py" \
      rapids-pytest-benchmark

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_mamba_retry remove --force rapids-build-env rapids-notebook-env
# gpuci_mamba_retry install -y "your-pkg=1.0.0"

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD
################################################################################

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_logger "Build from source"
    $WORKSPACE/build.sh -v clean libcugraph pylibcugraph cugraph cugraph-service-server cugraph-service-client cugraph-pyg
else
    gpuci_logger "Installing libcugraph-tests"
    gpuci_mamba_retry install -c ${CONDA_ARTIFACT_PATH} libcugraph libcugraph_etl libcugraph-tests

    # TODO: Move boa install to gpuci/rapidsai
    gpuci_mamba_retry install boa

    gpuci_logger "Building and installing pylibcugraph and cugraph..."
    export CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"
    export VERSION_SUFFIX=""
    gpuci_logger "pylibcugraph"
    gpuci_conda_retry mambabuild conda/recipes/pylibcugraph --no-build-id --croot ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} --python=${PYTHON}
    gpuci_logger "cugraph"
    gpuci_conda_retry mambabuild conda/recipes/cugraph --no-build-id --croot ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} --python=${PYTHON}
    gpuci_logger "cugraph-service"
    gpuci_conda_retry mambabuild conda/recipes/cugraph-service --no-build-id --croot ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} --python=${PYTHON}

    gpuci_logger "Building and installing cugraph-pyg..."
    gpuci_conda_retry mambabuild conda/recipes/cugraph-pyg --no-build-id --croot ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} --python=${PYTHON}

    #gpuci_logger "Installing pylibcugraph, cugraph, cugraph-pyg and cugraph-service from build / artifact dirs"
    #gpuci_mamba_retry install -c ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} pylibcugraph cugraph cugraph-pyg cugraph-service-server cugraph-service-client
    gpuci_logger "Installing pylibcugraph from build / artifact dirs"
    gpuci_mamba_retry install -c ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} pylibcugraph
    gpuci_logger "Installing cugraph from build / artifact dirs"
    gpuci_mamba_retry install -c ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} cugraph
    gpuci_logger "Installing cugraph-pyg from build / artifact dirs"
    gpuci_mamba_retry install -c ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} cugraph-pyg
    gpuci_logger "Installing cugraph-service-server from build / artifact dirs"
    gpuci_mamba_retry install -c ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} cugraph-service-server
    gpuci_logger "Installing cugraph-service-client from build / artifact dirs"
    gpuci_mamba_retry install -c ${CONDA_BLD_DIR} -c ${CONDA_ARTIFACT_PATH} cugraph-service-client
fi

################################################################################
# Identify relevant testsets to run in CI based on the ChangeList
################################################################################

fnames=()
# Initialize all the run variables to true, as we want to run all the tests if the build_mode is NOT a pull-request
run_cpp_tests="true" run_python_tests="true" run_nb_tests="true"
if [ "$BUILD_MODE" == "pull-request" ]; then
    PR_ENDPOINT="https://api.github.com/repos/rapidsai/cugraph/pulls/${PR_ID}/files"
    fnames=(
      $(
      curl -s \
      -X GET \
      -H "Accept: application/vnd.github.v3+json" \
      -H "Authorization: token $GHTK" \
      "$PR_ENDPOINT" | \
      jq -r '.[].filename'
      )
    )
    # Initialize all the run variables to false, for pull-requests only, later, based on what's changed, these variables will be set to true
    run_cpp_tests="false" run_python_tests="false" run_nb_tests="false"
fi
# this will not do anything if the 'fnames' array is empty
for fname in "${fnames[@]}"
do
   if [[ "$fname" == *"cpp/"* && "$fname" != *"cpp/docs/"* && "$fname" != *"cpp/doxygen/"* ]]; then
      run_cpp_tests="true" run_python_tests="true" run_nb_tests="true"
   fi
   if [[ "$fname" == *"python/"* ]]; then
      run_python_tests="true" run_nb_tests="true"
   fi
   if [[ "$fname" == *"notebooks/"* ]]; then
      run_nb_tests="true"
   fi
done
################################################################################
# PRINT SUMMARY OF TESTS to BE EXECUTED
################################################################################
gpuci_logger "Summary of CI tests to be executed"
gpuci_logger "Run cpp tests=$run_cpp_tests"
gpuci_logger "Run python tests=$run_python_tests"
gpuci_logger "Run notebook tests=$run_nb_tests"

################################################################################
# TEST
################################################################################

# Switch to +e to allow failing commands to continue the script, which is needed
# so all testing commands run regardless of pass/fail. This requires the desired
# exit code to be managed using the ERR trap.
set +e           # allow script to continue on error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

EXITCODE=0

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    gpuci_logger "Check GPU usage"
    nvidia-smi

    # If this is a PR build, skip downloading large datasets and don't run the
    # slow-running tests that use them.
    # See: https://docs.rapids.ai/maintainers/gpuci/#environment-variables
    if [ "$BUILD_MODE" = "pull-request" ]; then
        TEST_MODE_FLAG="--quick"
    else
        TEST_MODE_FLAG=""
    fi

    NOTEBOOK_TEST_MODE="ci"

    gpuci_logger "Running cuGraph test.sh..."
    if [[ $run_cpp_tests == "true" ]]; then
        ${WORKSPACE}/ci/test.sh ${TEST_MODE_FLAG} --run-cpp-tests --run-python-tests | tee testoutput.txt
    elif [[ $run_python_tests == "true" ]]; then
        ${WORKSPACE}/ci/test.sh ${TEST_MODE_FLAG} --run-python-tests | tee testoutput.txt
    else
        ${WORKSPACE}/ci/test.sh ${TEST_MODE_FLAG} | tee testoutput.txt
    fi
    gpuci_logger "Ran cuGraph test.sh : return code was: $?, gpu/build.sh exit code is now: $EXITCODE"

    if [[ $run_nb_tests == "true" ]]; then
        gpuci_logger "Running cuGraph notebook test script..."
        ${WORKSPACE}/ci/gpu/test-notebooks.sh ${NOTEBOOK_TEST_MODE} 2>&1 | tee nbtest.log
        gpuci_logger "Ran cuGraph notebook test script : return code was: $?, gpu/build.sh exit code is now: $EXITCODE"
        python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log
    fi
fi

if [[ -n "${CODECOV_TOKEN}" && $run_python_tests == "true" ]]; then
    codecov -t $CODECOV_TOKEN
fi

gpuci_logger "gpu/build.sh returning value: $EXITCODE"
return ${EXITCODE}
