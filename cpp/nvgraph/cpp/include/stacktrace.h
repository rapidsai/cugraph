/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//adapted from https://idlebox.net/2008/0901-stacktrace-demangled/ and licensed under WTFPL v2.0
#pragma once

#if defined(_WIN32) || defined (__ANDROID__) || defined(ANDROID) || defined (__QNX__) || defined (__QNXNTO__)
#else
 #include <execinfo.h>
 #include <dlfcn.h>
 #include <cxxabi.h>
 #include <unistd.h>
 #include <stdlib.h>
#endif

#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
namespace nvgraph {

/** Print a demangled stack backtrace of the caller function to FILE* out. */
static inline void printStackTrace(std::ostream &eout = std::cerr, unsigned int max_frames = 63)
{
#if defined(_WIN32) || defined (__ANDROID__) || defined(ANDROID) || defined (__QNX__) || defined (__QNXNTO__)
  //TODO add code for windows stack trace and android stack trace
#else
    std::stringstream out;

    // storage array for stack trace address data
    void* addrlist[max_frames+1];

    // retrieve current stack addresses
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

    if (addrlen == 0) {
        out << "  <empty, possibly corrupt>\n";
        return;
    }

    // resolve addresses into strings containing "filename(function+address)",
    // this array must be free()-ed
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    // allocate string which will be filled with the demangled function name
    size_t funcnamesize = 256;
    char* funcname = (char*)malloc(funcnamesize);

    // iterate over the returned symbol lines. skip the first, it is the
    // address of this function.
    for (int i = 1; i < addrlen; i++)
    {
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = symbollist[i]; *p; ++p)
        { 
            if (*p == '(')
                begin_name = p;   
            else if (*p == '+')
                begin_offset = p;
            else if (*p == ')' && begin_offset) {
                end_offset = p;
                break;
            }
        }

        if (begin_name && begin_offset && end_offset
            && begin_name < begin_offset)
        {
            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset = '\0';

            // mangled name is now in [begin_name, begin_offset) and caller
            // offset in [begin_offset, end_offset). now apply
            // __cxa_demangle():

            int status;
            char* ret = abi::__cxa_demangle(begin_name,
                                            funcname, &funcnamesize, &status);
            if (status == 0) {
                funcname = ret; // use possibly realloc()-ed string
                out << " " << symbollist[i] << " : " << funcname << "+" << begin_offset << "\n";
            }
            else {
                // demangling failed. Output function name as a C function with
                // no arguments.
                out << " " << symbollist[i] << " : " << begin_name << "()+" << begin_offset << "\n";
            }
        }
        else
        {
            // couldn't parse the line? print the whole line.
            out << " " << symbollist[i] << "\n";
        }
    }
    eout << out.str();
    //error_output(out.str().c_str(),out.str().size());
    free(funcname);
    free(symbollist);
    //printf("PID of failing process: %d\n",getpid());
    //while(1);
#endif
}

} //end namespace nvgraph

