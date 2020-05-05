#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
########################
# cuGraph Style Tester #
########################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH

# Activate common conda env
source activate gdf

# Run flake8 and get results/return code
FLAKE=`flake8 python`
FLAKE_RETVAL=$?

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`python cpp/scripts/run-clang-format.py 2>&1`
CLANG_FORMAT_RETVAL=$?

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

RETVALS=($FLAKE_RETVAL $CLANG_FORMAT_RETVAL)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL

