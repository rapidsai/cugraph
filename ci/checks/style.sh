#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
########################
# cuGraph Style Tester #
########################

# Make failing commands visible when used in a pipeline and allow the script to
# continue on errors, but use ERRORCODE to still allow any failing command to be
# captured for returning a final status code. This allows all style check to
# take place to provide a more comprehensive list of style violations.
set -o pipefail
ERRORCODE=0
PATH=/conda/bin:$PATH
THISDIR=$(cd $(dirname $0);pwd)
CUGRAPH_ROOT=$(cd ${THISDIR}/../..;pwd)

# Activate common conda env
source activate gdf

# Run flake8 and get results/return code
FLAKE=`flake8 --config=python/.flake8 ${CUGRAPH_ROOT}/python`
ERRORCODE=$((ERRORCODE | $?))

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`python ${CUGRAPH_ROOT}/cpp/scripts/run-clang-format.py 2>&1`
CLANG_FORMAT_RETVAL=$?
ERRORCODE=$((ERRORCODE | ${CLANG_FORMAT_RETVAL}))

# Output results if failure otherwise show pass
if [ "$FLAKE" != "" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

if [ "$CLANG_FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: clang format check; begin output\n\n"
  echo -e "$CLANG_FORMAT"
  echo -e "\n\n>>>> FAILED: clang format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: clang format check\n\n"
fi

# Check for copyright headers in the files modified currently
#COPYRIGHT=`env PYTHONPATH=${CUGRAPH_ROOT}/ci/utils python ${THISDIR}/copyright.py cpp python benchmarks ci 2>&1`
COPYRIGHT=`env PYTHONPATH=${CUGRAPH_ROOT}/ci/utils python ${THISDIR}/copyright.py --git-modified-only 2>&1`
CR_RETVAL=$?
ERRORCODE=$((ERRORCODE | ${CR_RETVAL}))

# Output results if failure otherwise show pass
if [ "$CR_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
fi

exit ${ERRORCODE}
