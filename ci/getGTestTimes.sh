#!/bin/bash

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
