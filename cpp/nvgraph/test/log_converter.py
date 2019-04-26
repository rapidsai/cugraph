#!/usr/bin/python
from sys import argv
from subprocess import Popen, PIPE, STDOUT
from os import path, environ


def main():
    args = argv[1:]
    args[0] = path.join('./', args[0])
    print args
    environ["GTEST_PRINT_TIME"] = "0"
    popen = Popen(args, stdout=PIPE, stderr=STDOUT)
    stillParsing = True
    skip = []
    while not popen.poll():
        data = popen.stdout.readline().splitlines()
        if len(data) == 0:
            break
        data = data[0]
        try:
            STATUS = data[0:12]
            NAME = data[12:]
            if data.find('Global test environment tear-down') != -1:
                stillParsing = False
            if stillParsing:
                if STATUS == "[ RUN      ]":
                    print('&&&& RUNNING' + NAME)
                elif STATUS == "[       OK ]" and NAME.strip() not in skip:
                    print('&&&& PASSED ' + NAME)
                elif STATUS == "[  WAIVED  ]":
                    print('&&&& WAIVED ' + NAME)
                    skip.append(NAME.strip())
                elif STATUS == "[  FAILED  ]":
                    NAME = NAME.replace(', where', '\n where')
                    print('&&&& FAILED ' + NAME)
                else:
                    print(data)
            else:
                print(data)
        except IndexError:
            print(data)

    return popen.returncode

if __name__ == '__main__':
    main()
