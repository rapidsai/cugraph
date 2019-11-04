import subprocess


def getRepoInfo():
    out = getCommandOutput("git remote -v")
    repo = out.split("\n")[-1].split()[1]
    branch = getCommandOutput("git rev-parse --abbrev-ref HEAD")
    return (repo, branch)


def getCommandOutput(cmd):
    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)
    stdout = result.stdout.decode().strip()
    if result.returncode == 0:
        return stdout

    stderr = result.stderr.decode().strip()
    raise RuntimeError("Problem running '%s' (STDOUT: '%s' STDERR: '%s')"
                       % (cmd, stdout, stderr))


def getCommitInfo():
    commitHash = getCommandOutput("git rev-parse HEAD")
    commitTime = getCommandOutput("git log -n1 --pretty=%%ct %s" % commitHash)
    return (commitHash, str(int(commitTime)*100))


def getCudaVer():
    # FIXME
    return "10.0"


def getGPUModel():
    # FIXME
    return "some GPU"
