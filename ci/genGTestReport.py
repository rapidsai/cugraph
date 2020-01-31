import xml.etree.ElementTree as ET
from glob import glob
from os import path

#### private helper functions
def _updateDictFromTestcases(xmlNode, testBinName, dataDict):
    """
    Recursively walk the tree at xmlNode and create a dictionary of testcase
    name (keys) to metadata (values) by finding all nodes tagged as "testcase".
    """
    retDict = {}
    if xmlNode.tag == "testcase":
        name = "%s:%s/%s" % (testBinName, xmlNode.attrib["classname"], xmlNode.attrib["name"])
        dataDict[name] = Testcase(name=name, status=xmlNode.attrib["status"], runtime=xmlNode.attrib["time"])
    else:
        for child in xmlNode:
            _updateDictFromTestcases(child, testBinName, dataDict)


def _getSortedResultsString(resultsList):
    """
    Given a flat list (unsorted) of Testcase objs, return a report string with
    test result info, 1 testcase per line, sorted by test runtime
    """
    retStringList = []
    sortedObjs = sorted(resultsList, key=lambda x: getattr(x, "runtime"), reverse=True)
    tcTuples = [(tc.name, tc.runtime) for tc in sortedObjs]
    # Format string by first determining the longest test name
    # TODO: is there a fancy pprint function to do this?
    maxWidth = max([len(t[0]) for t in tcTuples])
    for (tcName, tcRuntime) in tcTuples:
        whitespace = " " * ((maxWidth - len(tcName)) + 1)
        retStringList.append("%s%s: %f s" % (tcName, whitespace, tcRuntime))
    return "\n".join(retStringList)


#### public API functions
class Testcase:
    """
    Class containing Testcase run data
    """
    def __init__(self, name, status, runtime):
        self.name = name
        self.status = status
        self.runtime = float(runtime)


def getResultsDict(resultsDir):
    """
    Read each XML file in resultsDir and return a dictionary with the following:
    {<test binary name1> :
        {<test name> : <Testcase obj>,
         ...
        },
     <test binary name2> :
        {<test name> : <Testcase obj>,
         ...
        },
     ...
    }
    """
    retDict = {}
    for resultFile in glob(path.join(resultsDir, "*.xml")):
        testBinName = path.splitext(path.basename(resultFile))[0]
        retDict[testBinName] = {}
        _updateDictFromTestcases(ET.parse(resultFile).getroot(), testBinName, retDict[testBinName])

    return retDict


def getReportString(resultsDict):
    """
    Return the "default" report string - all test results sorted by runtime
    """
    retStringList = []
    allResults = [v for results in resultsDict.values() for v in results.values()]
    return _getSortedResultsString(allResults)


def getReportStringByBinary(resultsDict):
    """
    Return a report string where test results sorted by runtime are grouped by
    test binary
    """
    retStringList = []
    for testBinName, results in resultsDict.items():
        retStringList.append("%s:\n%s" %
                             (testBinName,
                              _getSortedResultsString(results.values())))
    return "\n\n".join(retStringList)


#### "main" if run as a script
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default=".",
                    help="directory containing XML test results"
                    )
    ap.add_argument("--show_by_test_binary",
                    default=False, action="store_true",
                    help="output results grouped by test binary in addition to all results combined"
                    )
    ap.add_argument("--dont_show_combined",
                    default=False, action="store_true",
                    help="do not output combined results list"
                    )
    args = ap.parse_args()

    if path.exists(args.results_dir):
        resultsDict = getResultsDict(args.results_dir)
        if args.show_by_test_binary:
            print(getReportStringByBinary(resultsDict))
        if not args.dont_show_combined:
            print(getReportString(resultsDict))
    else:
        print("%s does not exist!" % args.results_dir)
