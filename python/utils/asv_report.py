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

import platform

import psutil

from asvdb import BenchmarkInfo, BenchmarkResult, ASVDb
from utils import getCommitInfo, getRepoInfo


def cugraph_update_asv(
    asvDir,
    datasetName,
    algoRunResults,
    cudaVer="",
    pythonVer="",
    osType="",
    machineName="",
    repo="",
):
    """
    algoRunResults is a list of (algoName, exeTime) tuples
    """
    (commitHash, commitTime) = getCommitInfo()
    (actualRepo, branch) = getRepoInfo()
    repo = repo or actualRepo

    db = ASVDb(asvDir, repo, [branch])

    uname = platform.uname()

    prefixDict = dict(
        maxGpuUtil="gpuutil",
        maxGpuMemUsed="gpumem",
        exeTime="time",
    )
    unitsDict = dict(
        maxGpuUtil="percent",
        maxGpuMemUsed="bytes",
        exeTime="seconds",
    )

    bInfo = BenchmarkInfo(
        machineName=machineName or uname.machine,
        cudaVer=cudaVer or "unknown",
        osType=osType or "%s %s" % (uname.system, uname.release),
        pythonVer=pythonVer or platform.python_version(),
        commitHash=commitHash,
        commitTime=commitTime,
        gpuType="unknown",
        cpuType=uname.processor,
        arch=uname.machine,
        ram="%d" % psutil.virtual_memory().total,
    )

    validKeys = set(list(prefixDict.keys()) + list(unitsDict.keys()))

    for (funcName, metricsDict) in algoRunResults.items():
        for (metricName, val) in metricsDict.items():
            # If an invalid metricName is present (likely due to a benchmark
            # run error), skip
            if metricName in validKeys:
                bResult = BenchmarkResult(
                    funcName="%s_%s" % (funcName, prefixDict[metricName]),
                    argNameValuePairs=[("dataset", datasetName)],
                    result=val,
                )
                bResult.unit = unitsDict[metricName]
                db.addResult(bInfo, bResult)


if __name__ == "__main__":
    # Test ASVDb with some mock data (that just so happens to be very similar
    # to actual data)
    # FIXME: consider breaking this out to a proper test_whatever.py file!
    asvDir = "asv"

    datasetName = "dolphins.csv"
    algoRunResults = [
        ("loadDataFile", 3.2228727098554373),
        ("createGraph", 3.00713360495865345),
        ("pagerank", 3.00899268127977848),
        ("bfs", 3.004273353144526482),
        ("sssp", 3.004624705761671066),
        ("jaccard", 3.0025573652237653732),
        ("louvain", 3.32631026208400726),
        ("weakly_connected_components", 3.0034315641969442368),
        ("overlap", 3.002147899940609932),
        ("triangles", 3.2544921860098839),
        ("spectralBalancedCutClustering", 3.03329935669898987),
        ("spectralModularityMaximizationClustering", 3.011258183047175407),
        ("renumber", 3.001620553433895111),
        ("view_adj_list", 3.000927431508898735),
        ("degree", 3.0016251634806394577),
        ("degrees", None),
    ]
    cugraph_update_asv(
        asvDir, datasetName, algoRunResults, machineName="MN", pythonVer="3.6"
    )

    # Same arg values (the "datasetName" is still named "dolphins.csv"), but
    # different results - this should override just the results.
    algoRunResults = [(a, r + 1) for (a, r) in algoRunResults]
    cugraph_update_asv(
        asvDir, datasetName, algoRunResults, machineName="MN", pythonVer="3.6"
    )

    # New arg values (changed "datasetName" to "dolphins2.csv") - this should
    # create a new set or arg values and results.
    datasetName = "dolphins2.csv"
    cugraph_update_asv(
        asvDir, datasetName, algoRunResults, machineName="MN", pythonVer="3.6"
    )
