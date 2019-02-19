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
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "tmp_pool.h"

#define CUDA_CHECK(call) {										\
				cudaError_t err = call;							\
				if( cudaSuccess != err) {						\
					fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
							__FILE__, __LINE__, cudaGetErrorString( err) ); \
					exit(EXIT_FAILURE);						\
				}									\
			    }

#define ALLOC_STEP	(1024)

typedef struct {

	void	*ptr;
	int	index;

} pref_t;

struct tmp_pool_s {

	void	**pool;
	size_t	*size;
	int	*used;
	int	n_alloc;
	int	n;
	size_t	totsz;

	// stack of pointers in use	
	pref_t	*out;
	int	nout_alloc;
	int	nout;
};

static void *Malloc(size_t sz) {

        void *ptr;

        if (!sz) {
                printf("Cannot allocate zero bytes...\n");
                exit(EXIT_FAILURE);
        }
        ptr = (void *)malloc(sz);
        if (!ptr) {
                fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
                exit(EXIT_FAILURE);
        }
        return ptr;
}

static void *Realloc(void *ptr, size_t sz) {

        void *lp;

        if (!sz) {
                printf("Re-allocating to zero bytes, are you sure you want this?\n");
        }
        lp = (void *)realloc(ptr, sz);
        if (!lp && sz) {
                fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
                exit(EXIT_FAILURE);
        }
        return lp;
}

static int bisect_left(const size_t *v, const int num, const size_t val) {

        if (0 == num) return -1;
#if 1
        int  min = 0;
        int  max = num-1;
        int  mid = max >> 1;

        while(min <= max) {

                if (v[mid] == val)      break;
                if (v[mid]  < val)      min = mid+1;
                else                    max = mid-1;
                mid = (max>>1)+(min>>1)+((min&max)&1);
        }  
        if (mid >= 0 && v[mid] == val) {
                while(mid) {
                        if (v[mid-1] == val) mid--;
                        else                 break;
                }
        } else mid++;

        return mid;
#else
        int i;
        for(i = 0; i < num; i++)
                if (v[i] >= val) break;

        return i;
#endif
}

tmp_pool_t *tmp_create() {

	tmp_pool_t *ptr = (tmp_pool_t *)Malloc(sizeof(tmp_pool_t));

	ptr->size = (size_t *)Malloc(ALLOC_STEP*sizeof(size_t));
	ptr->pool = (void **)Malloc(ALLOC_STEP*sizeof(void *));
	ptr->used = (int *)Malloc(ALLOC_STEP*sizeof(int));
	ptr->n_alloc = ALLOC_STEP;
	ptr->n = 0;
	ptr->totsz = 0;

	ptr->out = (pref_t *)Malloc(ALLOC_STEP*sizeof(pref_t));
	ptr->nout_alloc = ALLOC_STEP;
	ptr->nout = 0;

	return ptr;
}

static void expand(tmp_pool_t *tp) {

	if (tp->n == tp->n_alloc) {
		tp->size = (size_t *)Realloc(tp->size, (tp->n_alloc+ALLOC_STEP)*sizeof(size_t));
		tp->pool = (void **)Realloc(tp->pool, (tp->n_alloc+ALLOC_STEP)*sizeof(void *));
		tp->used = (int *)Realloc(tp->used, (tp->n_alloc+ALLOC_STEP)*sizeof(int));
		tp->n_alloc += ALLOC_STEP;
	}
	if (tp->nout == tp->nout_alloc) {
		tp->out = (pref_t *)Realloc(tp->out, (tp->nout_alloc+ALLOC_STEP)*sizeof(pref_t));
		tp->nout_alloc += ALLOC_STEP;
	}
}

void *tmp_get(tmp_pool_t *tp, size_t size) {

	int i, index;
	index = bisect_left(tp->size, tp->n, size);

	// initially tp->n==0 and bisect_left() returns -1
	if (index == -1) index = 0;

	for(i = index; i < tp->n; i++) {
		// as soon as we arrive to a buffer of 
		// size >= of 110% of size it's pointless to continue
		if (tp->size[i] > 11*size/10) {
			i = tp->n;
			break;
		}
		if (!tp->used[i]) break;
	}

	expand(tp);
	if (i < tp->n) {	// we have an allocated buffer to return
		index = i;
	} else {		// we need to allocate a new buffer
		int tries=0;
		void *__ptr=NULL;

		do {
			cudaError_t err = cudaMalloc(&__ptr, size);
			if (err == cudaSuccess) {
				//fprintf(stdout, "Memory allocated!\n");
				break;
			}
			cudaGetLastError();
			if (tries == 0) {
				/*
				fprintf(stderr,
					"%s:%d: not enough free mem, freeing unused buffer...\n",
					__FILE__, __LINE__);
				*/
				tmp_purge(tp);

				index = bisect_left(tp->size, tp->n, size);
				if (index == -1) index = 0;

				tries++;
			} else {
				fprintf(stdout, "Cuda error in file '%s' in line %i : %s.\n",
						__FILE__, __LINE__, cudaGetErrorString( err) );
				exit(EXIT_FAILURE);
			}
		} while(1);

		for(i = tp->n; i > index; i--) {
			tp->pool[i] = tp->pool[i-1];
			tp->size[i] = tp->size[i-1];
			tp->used[i] = tp->used[i-1];
		}
		tp->n++;

		tp->pool[index] = __ptr;
		tp->size[index] = size;
		tp->totsz += size;

		// adjust indices of "busy" buffer that have been moved
		for(i = 0; i < tp->nout; i++) {
			if (tp->out[i].index >= index)
				tp->out[i].index++;
		}
	}
	tp->used[index] = 1;

	tp->out[tp->nout].ptr = tp->pool[index];
	tp->out[tp->nout].index = index;
	tp->nout++;

	return tp->pool[index];
}

static inline int stack_find(tmp_pool_t *tp, void *ptr) {
	
	if (!tp) return -1;
	
	int i = tp->nout-1;
	while(i >= 0) {
		if (tp->out[i].ptr == ptr) break;
		i--;
	}
	return i;
}

void tmp_release(tmp_pool_t *tp, void *ptr) {

	if (!tp) return;
	
	int i = stack_find(tp, ptr);

	// if is on the stack iff
	// its "used" flag quals 1
	if (i == -1) return;

	int index = tp->out[i].index;
	tp->used[index] = 0;

	for(; i < tp->nout-1; i++)
		tp->out[i] = tp->out[i+1];
	tp->nout--;

	return;
}

void *tmp_detach(tmp_pool_t *tp, void *ptr) {

	if (!tp) return NULL;
	
	int i = stack_find(tp, ptr);
	if (i == -1) return NULL;

	int index = tp->out[i].index;

	int k;
	for(k = 0; k < i; k++) {
		if (tp->out[k].index > index)
			tp->out[k].index--;
	}
	for(; i < tp->nout-1; i++) {
		tp->out[i] = tp->out[i+1];
		if (tp->out[i].index > index)
			tp->out[i].index--;
	}
	tp->nout--;

	for(; index < tp->n-1; index++) {
		tp->pool[index] = tp->pool[index+1];
		tp->size[index] = tp->size[index+1];
		tp->used[index] = tp->used[index+1];
	}
	tp->n--;

	return ptr;
}

void tmp_remove(tmp_pool_t *tp, void *ptr) {

	if (!tp) return;

	ptr = tmp_detach(tp, ptr);
	if (ptr) CUDA_CHECK(cudaFree(ptr));

	return;
}

void tmp_purge(tmp_pool_t *tp) {

	if (!tp) return;

	int i;
	int j = 0;

	for(i = 0; i < tp->n; i++) {
		if (!tp->used[i]) {
			CUDA_CHECK(cudaFree(tp->pool[i]));
			tp->totsz -= tp->size[i];
		} else {
			tp->pool[j] = tp->pool[i];
			tp->used[j] = tp->used[i];
			tp->size[j] = tp->size[i];
			j++;
		}
	}
	tp->n = j;

	// rebuild tp->nout with all remaining, in use, buffers;
	for(i = 0; i < tp->n; i++) {
		tp->out[i].ptr = tp->pool[i];
		tp->out[i].index = i;
	}
	tp->nout = i;

	return;
}

void tmp_clearall(tmp_pool_t *tp) {

	if (!tp) return;

	int i;
	for(i = 0; i < tp->n; i++) {
		CUDA_CHECK(cudaFree(tp->pool[i]));
	}
	tp->n = 0;
	tp->nout = 0;

	return;
}

void tmp_destroy(tmp_pool_t *tp) {

	if (!tp) return;

	tmp_clearall(tp);

	free(tp->pool);
	free(tp->size);
	free(tp->used);
	free(tp->out);

	return;
}

void tmp_print(tmp_pool_t *tp) {

	if (!tp) return;

	int i;
	printf("Buffer pool:\n");
	for(i = 0; i < tp->n; i++) {
		printf("%d: ptr=%p size=%zu used=%d\n",
		       i, tp->pool[i], tp->size[i], tp->used[i]);
	}

	printf("Reserved buffers:\n");
	for(i = tp->nout-1; i >= 0; i--) {
		printf("%d: ptr=%p index=%d\n",
		       i, tp->out[i].ptr, tp->out[i].index);
	}
	printf("\n");
	return;
}

