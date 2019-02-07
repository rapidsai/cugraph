/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
#ifndef __TMP_POOL_H__
#define	__TMP_POOL_H__

typedef struct tmp_pool_s tmp_pool_t;

#ifdef __cplusplus
extern "C" {
#endif
tmp_pool_t *tmp_create();
void *tmp_get(tmp_pool_t *tp, size_t size);
void tmp_release(tmp_pool_t *tp, void *ptr);
void *tmp_detach(tmp_pool_t *tp, void *ptr);
void tmp_remove(tmp_pool_t *tp, void *ptr);
void tmp_clearall(tmp_pool_t *tp);
void tmp_destroy(tmp_pool_t *tp);
void tmp_print(tmp_pool_t *tp);
void tmp_purge(tmp_pool_t *tp);
#ifdef __cplusplus
}
#endif

#endif
