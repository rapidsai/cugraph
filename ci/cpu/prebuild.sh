#!/usr/bin/env bash

export BUILD_ABI=1

if [[ "$CUDA" == "9.2" ]]; then
    export BUILD_CUGRAPH=1
else
    export BUILD_CUGRAPH=0
fi

if [[ "$PYTHON" == "3.6" ]]; then
    export BUILD_LIBCUGRAPH=1
else
    export BUILD_LIBCUGRAPH=0
fi
