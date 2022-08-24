#!/bin/bash

set -euo pipefail

# TODO: Remove
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
source ci/check_env.sh

gpuci_logger "Check GPU usage"
nvidia-smi

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# Install libcugraph packages
gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  libcugraph libcugraph_etl libcugraph-tests

# Check if we need to run CPP tests
run_cpp_tests="false"
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

# this will not do anything if the 'fnames' array is empty
for fname in "${fnames[@]}"
do
   if [[ "$fname" == *"cpp/"* && "$fname" != *"cpp/docs/"* && "$fname" != *"cpp/doxygen/"* ]]; then
      run_cpp_tests="true"
   fi
done

gpuci_logger "Run cpp tests=$run_cpp_tests"

if [[ $run_cpp_tests == "false" ]]; then
  exit 0;
fi

set +e
set -E
trap "EXITCODE=1" ERR
EXITCODE=0

"${GITHUB_WORKSPLACE}/ci/test.sh" --quick --run-cpp-tests | tee testoutput.txt

exit "${EXITCODE}"
