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

# This script will print the gtest results sorted by runtime. This will print
# the results two ways: first by printing all tests sorted by runtime, then by
# printing all tests grouped by test binary with tests sorted by runtime within
# the group.
#
# To use this script, capture the test run output to a file then run this script
# with the file as the first arg, or just redirect test output to this script.

awk '/^Running GoogleTest .+$/ {
       testbinary = $3
     }
     /^\[       OK \].+$/ {
        testtime = substr($(NF-1),2)
        newtestdata = testbinary ":" substr($0,14)
        alltestdata = alltestdata newtestdata "\n"
        testdata[testbinary] = testdata[testbinary] newtestdata "\n"
        totaltime = totaltime + testtime
     }
     END {
        # Print all tests sorted by time
        system("echo \"" alltestdata "\" | sort -r -t\\( -nk2")
        print "\n================================================================================"
        # Print test binaries with tests sorted by time
        print "Tests grouped by test binary:"
        for (testbinary in testdata) {
           print testbinary
           system("echo \"" testdata[testbinary] "\" | sort -r -t\\( -nk2")
        }
        print "\n================================================================================"
        print totaltime " milliseconds = " totaltime/60000 " minutes"
     }
' $1
