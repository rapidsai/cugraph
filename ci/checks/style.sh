#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
########################
# cuGraph Style Tester #
########################

# Assume this script is run from the root of the cugraph repo

# Make failing commands visible when used in a pipeline and allow the script to
# continue on errors, but use ERRORCODE to still allow any failing command to be
# captured for returning a final status code. This allows all style check to
# take place to provide a more comprehensive list of style violations.
set -o pipefail
# CI does `set -e` then sources this file, so we override that so we can output
# the results from the various style checkers
set +e
ERRORCODE=0
PATH=/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run flake8 and get results/return code
FLAKE=`flake8 --config=python/.flake8 python`
ERRORCODE=$((ERRORCODE | $?))

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`python cpp/scripts/run-clang-format.py 2>&1`
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
#COPYRIGHT=`env PYTHONPATH=ci/utils python ci/checks/copyright.py cpp python benchmarks ci 2>&1`
COPYRIGHT=`env PYTHONPATH=ci/utils python ci/checks/copyright.py --git-modified-only 2>&1`
CR_RETVAL=$?
ERRORCODE=$((ERRORCODE | ${CR_RETVAL}))

if [ "$CR_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check; begin debug output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> PASSED: copyright check; end debug output\n\n"
fi

exit ${ERRORCODE}
