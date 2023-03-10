# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is source'd from script-env.sh to add functions to the
# calling environment, hence no #!/bin/bash as the first line. This
# also assumes the variables used in this file have been defined
# elsewhere.

NUMARGS=$#
ARGS=$*
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function logger {
  echo -e ">>>> $@"
}

# Calling "setTee outfile" will cause all stdout and stderr of the
# current script to be output to "tee", which outputs to stdout and
# "outfile" simultaneously. This is useful by allowing a script to
# "tee" itself at any point without being called with tee.
_origFileDescriptorsSaved=0
function setTee {
    if [[ $_origFileDescriptorsSaved == 0 ]]; then
        # Save off the original file descr 1 and 2 as 3 and 4
        exec 3>&1 4>&2
        _origFileDescriptorsSaved=1
    fi
    teeFile=$1
    # Create a named pipe.
    pipeName=$(mktemp -u)
    mkfifo $pipeName
    # Close the currnet 1 and 2 and restore to original (3, 4) in the
    # event this function is called repeatedly.
    exec 1>&- 2>&-
    exec 1>&3 2>&4
    # Start a tee process reading from the named pipe. Redirect stdout
    # and stderr to the named pipe which goes to the tee process. The
    # named pipe "file" can be removed and the tee process stays alive
    # until the fd is closed.
    tee -a < $pipeName $teeFile &
    exec > $pipeName 2>&1
    rm $pipeName
}

# Call this to stop script output from going to "tee" after a prior
# call to setTee.
function unsetTee {
    if [[ $_origFileDescriptorsSaved == 1 ]]; then
        # Close the current fd 1 and 2 which should stop the tee
        # process, then restore 1 and 2 to original (saved as 3, 4).
        exec 1>&- 2>&-
        exec 1>&3 2>&4
    fi
}
