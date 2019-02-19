/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef __UTILS_H__
#define __UTILS_H__

extern LINKAGE void *Malloc(size_t sz);
extern LINKAGE void Free(void **ptr);
extern LINKAGE void *Realloc(void *ptr, size_t sz);
extern LINKAGE FILE *Fopen(const char *path, const char *mode);
extern LINKAGE void Fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
extern LINKAGE void Fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
extern LINKAGE int Remove(const char *pathname);
extern LINKAGE off_t getFsize(const char *fpath);

#endif
