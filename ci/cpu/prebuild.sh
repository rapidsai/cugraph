#!/usr/bin/env bash

export BUILD_ABI=1
export BUILD_CUGRAPH=1

if [[ "$PYTHON" == "3.6" ]]; then
    export BUILD_LIBCUGRAPH=1
else
    export BUILD_LIBCUGRAPH=0
fi
