#include <stdlib.h>   /* import labs() */
#include <math.h>

#include <iostream>
#include <string>

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <windows.h>
static double second (void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency (&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter (&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() / 1000.0;
    }
}

static long long getSystemMemory() 
{ 
    MEMORYSTATUSEX state; // Requires >= win2k 
    memset (&state, 0, sizeof(state)); 
    state.dwLength = sizeof(state); 
    if (0 == GlobalMemoryStatusEx(&state)) { 
        return 0; 
    } else {
        return (long long)state.ullTotalPhys; 
    }
} 
#elif defined(__linux) || defined(__powerpc64__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
static double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static long long getSystemMemory(void) 
{ 
    struct sysinfo s_info; 
    sysinfo (&s_info); 
    return (long long)s_info.totalram * (long long)s_info.mem_unit; 
} 
#elif defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/sysctl.h>
static double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static long long getSystemMemory(void) 
{ 
    int memmib[2] = { CTL_HW, HW_MEMSIZE };
    long long mem = (size_t)0;
    size_t memsz = sizeof(mem);

    /* NOTE: This may cap memory reported at 2GB */
    if (sysctl(memmib, 2, &mem, &memsz, NULL, 0) == -1) {
        return 0;
    } else {
        return mem;
    }
}
#elif defined(__QNX__)  
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
static double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static long long getSystemMemory(void) 
{ 
    return 0;
} 
#else
#error unsupported platform
#endif

std::string getFileName(const std::string& s) {

   char sep = '/';

#ifdef _WIN32
   sep = '\\';
#endif

   size_t i = s.rfind(sep, s.length());
   if (i != std::string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return("");
}
