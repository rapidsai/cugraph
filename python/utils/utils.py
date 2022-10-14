# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

import subprocess


def getRepoInfo():
    out = getCommandOutput("git remote -v")
    repo = out.split("\n")[-1].split()[1]
    branch = getCommandOutput("git rev-parse --abbrev-ref HEAD")
    return (repo, branch)


def getCommandOutput(cmd):
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout = result.stdout.decode().strip()
    if result.returncode == 0:
        return stdout

    stderr = result.stderr.decode().strip()
    raise RuntimeError(
        "Problem running '%s' (STDOUT: '%s' STDERR: '%s')" % (cmd, stdout, stderr)
    )


def getCommitInfo():
    commitHash = getCommandOutput("git rev-parse HEAD")
    commitTime = getCommandOutput("git log -n1 --pretty=%%ct %s" % commitHash)
    return (commitHash, str(int(commitTime) * 100))


def getCudaVer():
    # FIXME
    return "10.0"


def getGPUModel():
    # FIXME
    return "some GPU"
