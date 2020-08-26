#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

MAGIC_OVERRIDE_CODE="
def my_run_line_magic(*args, **kwargs):
    g=globals()
    l={}
    for a in args:
        try:
            exec(str(a),g,l)
        except Exception as e:
            print('WARNING: %s\n   While executing this magic function code:\n%s\n   continuing...\n' % (e, a))
        else:
            g.update(l)

def my_run_cell_magic(*args, **kwargs):
    my_run_line_magic(*args, **kwargs)

get_ipython().run_line_magic=my_run_line_magic
get_ipython().run_cell_magic=my_run_cell_magic

"

NO_COLORS=--colors=NoColor
EXITCODE=0
NBTMPDIR=${WORKSPACE}/tmp
mkdir -p ${NBTMPDIR}

for nb in $*; do
    NBFILENAME=$1
    NBNAME=${NBFILENAME%.*}
    NBNAME=${NBNAME##*/}
    NBTESTSCRIPT=${NBTMPDIR}/${NBNAME}-test.py
    shift

    echo --------------------------------------------------------------------------------
    echo STARTING: ${NBNAME}
    echo --------------------------------------------------------------------------------
    jupyter nbconvert --to script ${NBFILENAME} --output ${NBTMPDIR}/${NBNAME}-test
    echo "${MAGIC_OVERRIDE_CODE}" > ${NBTMPDIR}/tmpfile
    cat ${NBTESTSCRIPT} >> ${NBTMPDIR}/tmpfile
    mv ${NBTMPDIR}/tmpfile ${NBTESTSCRIPT}

    echo "Running \"ipython ${NO_COLORS} ${NBTESTSCRIPT}\" on $(date)"
    echo
    time bash -c "ipython ${NO_COLORS} ${NBTESTSCRIPT}; EC=\$?; echo -------------------------------------------------------------------------------- ; echo DONE: ${NBNAME}; exit \$EC"
    NBEXITCODE=$?
    echo EXIT CODE: ${NBEXITCODE}
    echo
    EXITCODE=$((EXITCODE | ${NBEXITCODE}))
done

exit ${EXITCODE}
