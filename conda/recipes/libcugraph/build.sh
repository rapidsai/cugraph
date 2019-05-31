#!/usr/bin/env bash

# This assumes the script is executed from the root of the repo directory

# show environment
printenv
# Cleanup local git
if [ -d .git ]; then
    git clean -xdf
fi

./build.sh libcugraph -v
