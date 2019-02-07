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
#ifdef __cplusplus
#define __STDC_FORMAT_MACROS 1
#endif
#include <inttypes.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <mpi.h>
#include <vector_types.h>

#define NDEBUG
#include <assert.h>

#include "global.h"
#include "cuda_kernels.h"
#include "utils.h"

typedef struct {
	int	n, rem;
	LOCINT	*vals;
	LOCINT	*rmin;
	LOCINT	*rmax;
	int64_t	*sat; // satisfying value <= nloc*nproc
} split_t;

// block i (<nblk) samples between splitters:
//	blk_s[i] and blk_e[i]
// with n values:
// 	vals[SUM(j<i, blk_n)] ... vals[blk_n[i]-1]
typedef struct {
	LOCINT	*vals;
	int	n, n_alloc;

	int	nblk, nblk_alloc;
	int	*blk_s;
	int	*blk_e;
	int	*blk_n;
} probe_t;

typedef struct {
	int	n, n_alloc;
	LOCINT	*vals;
	int	*locs;
} pivot_t;

static int cmpll(const void *p1, const void *p2) {

        LOCINT l1 = *(LOCINT *)p1;
        LOCINT l2 = *(LOCINT *)p2;

        if (l1 < l2) return -1;
        if (l1 > l2) return  1;
        return 0;
}

// returns:
// 	-1 if num == 0
// 	else the smallest index i s.t. v[i] >= val, if all elements
// 	are smaller than val returns num
int bisect_left(const LOCINT *v, const int num, const LOCINT val) {

	if (0 == num) return -1;
#if 1
        int  min = 0;
        int  max = num-1;
        int  mid = max >> 1;

        while(min <= max) {

                if (v[mid] == val)      break;
                if (v[mid]  < val)      min = mid+1;
                else			max = mid-1;
                mid = (max>>1)+(min>>1)+((min&max)&1);
        }
	if (mid >= 0 && v[mid] == val) {
		while(mid) {
			if (v[mid-1] == val) mid--;
			else		     break;
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

static int bisect_left64(const int64_t *v, const int num, const int64_t val) {

	if (0 == num) return -1;
#if 1
        int  min = 0;
        int  max = num-1;
        int  mid = max >> 1;

        while(min <= max) {

                if (v[mid] == val)      break;
                if (v[mid]  < val)      min = mid+1;
                else			max = mid-1;
                mid = (max>>1)+(min>>1)+((min&max)&1);
        }
	if (mid >= 0 && v[mid] == val) {
		while(mid) {
			if (v[mid-1] == val) mid--;
			else		     break;
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

static void probe_add_block(int spl_si, int spl_ei, split_t *s, probe_t *p, int nsample) {

	int i;
	LOCINT rmin, rmax, step;

	//printf("ADDING block for splitters [%d, %d] with %d samples in the range [%"PRILOC", %"PRILOC"]\n", 
	//	spl_si, spl_ei, nsample, s->rmin[spl_si], s->rmax[spl_ei]);

	if (p->nblk == p->nblk_alloc) {
		p->blk_s = (int *)Realloc(p->blk_s, (p->nblk_alloc+100)*sizeof(*(p->blk_s)));
		p->blk_e = (int *)Realloc(p->blk_e, (p->nblk_alloc+100)*sizeof(*(p->blk_e)));
		p->blk_n = (int *)Realloc(p->blk_n, (p->nblk_alloc+100)*sizeof(*(p->blk_n)));
		p->nblk_alloc += 100;
	}
	p->blk_s[p->nblk] = spl_si;
	p->blk_e[p->nblk] = spl_ei;

	if (p->n+nsample >= p->n_alloc) {
		int newlen = p->n+nsample; //MAX(p->n+nsample, p->n_alloc+ntask-1);
		p->vals = (LOCINT *)Realloc(p->vals, newlen*sizeof(*(p->vals)));
		p->n_alloc = newlen;
	}

	rmin = s->rmin[spl_si];
	rmax = s->rmax[spl_ei];
	step = MAX(1, (rmax - rmin)/(nsample+1));

	// add distinct probe values in the range [rmin,rmax] as long as is possible
	for(i = 0; i < nsample; i++) {
		p->vals[p->n + i] = MIN(rmax, rmin + (i+1)*step);
		if (i && p->vals[p->n + i] == p->vals[p->n + i-1]) break;
	}

	p->blk_n[p->nblk] = i;//nsample;
	p->nblk++;

	p->n += i;//nsample;
	return;
}

static void get_probe(split_t *s, probe_t *p, int nguess) {

	int i, start=0;

	p->n = 0;	
	p->nblk = 0;
	for(i = 1; i < s->n; i++) {
		if (!s->sat[i] && s->sat[i-1])
			start = i;
		if (s->sat[i] && !s->sat[i-1])
			probe_add_block(start, i-1, s, p, (i-start)*nguess/s->rem/* + (p->nblk < nguess%s->rem)*/);
	}
	if (!s->sat[i-1])
		probe_add_block(start, i-1, s, p, (i-start)*nguess/s->rem);

	return;
}

static void print_probe(probe_t *p) {

	int i;
	LOCINT *pvals = p->vals;

	printf("Probe contains %d blocks:\n", p->nblk);
	for(i = 0; i < p->nblk; i++) {
		int j;
		
		printf("\tblock %d contains %d samples for splitters in (%d, %d):\n\t\t",
			i, p->blk_n[i], p->blk_s[i], p->blk_e[i]);

		for(j = 0; j < p->blk_n[i]; j++) {
			printf(" %"PRILOC"", *pvals++); 
		}
		printf("\n");
	}
	return;
}

static void print_hist(LOCINT *v, int n) {

	int i;
	
	printf("Gathered histogram:\n\t");
	for(i = 0; i < n; i++)
		printf(" %"PRILOC"", v[i]);

	printf("\n");
	return;
}

static inline int64_t split_idv(int isp, int nsp, int64_t nelem) {

	return (isp+1)*(nelem/nsp) + (isp < (nelem%nsp));
}

//#define DEBUG_PRINT
static void update_splitters(split_t *sp, probe_t *pb, int64_t *ht, int64_t ntot, LOCINT gmin, LOCINT gmax, double tolerance, int verbose) {

	int i, j;
	LOCINT *pbv = pb->vals;

	for(i = 0; i < pb->nblk; i++) {

		if (verbose) {
			printf("\tsplitters (%d,..,%d) covered with %d guesses in the range (%"PRILOC",..,%"PRILOC")\n",
				pb->blk_s[i], pb->blk_e[i], pb->blk_n[i], sp->rmin[pb->blk_s[i]], sp->rmax[pb->blk_e[i]]);
		}
		for(j = pb->blk_s[i]; j <= pb->blk_e[i]; j++) {
			assert(sp->sat[j] == 0);

			LOCINT m, M, rmin, rmax;
			int64_t ideal = split_idv(j, sp->n+1/*==nprocs*/, ntot);

			// the search range could be shrinked for subsequent searches in the same block
			M = bisect_left64(ht, pb->blk_n[i], ideal);
#if defined(DEBUG_PRINT)
			printf("\n\tbmaxlt([");
			int s;
			for(s = 0; s < pb->blk_n[i]; s++) printf(" %"PRId64",", ht[s]);
			printf("\b], %d, ideal=%"PRId64")=%"PRILOC"\n", pb->blk_n[i], ideal, M);
#endif
			if (0 == M) {
				m = M;
				rmin = j ? sp->vals[j-1] : gmin; //sp->vals[j-1] may be undefined
				rmax = pbv[M];
			} else {
				m = M-1;
				rmin = pbv[m];
				if (M == pb->blk_n[i]/*pb->n*/) {
					//rmax = (j+1 < sp->n) ? sp->vals[j+1] : gmax;
					// search for the value of a saturated splitter with index
					// greater than those in the current block
					int k;
					for(k = pb->blk_e[i]+1; k < sp->n && 0 == sp->sat[k]; k++);
					rmax = (k == sp->n) ? gmax : sp->vals[k];
					M = m;
				} else
					rmax = pbv[M];
			}
			sp->rmin[j] = MAX(sp->rmin[j], rmin);
			sp->rmax[j] = MIN(sp->rmax[j], rmax);

			LOCINT deltal = ABS(ht[m]-ideal);
			LOCINT deltar = ABS(ht[M]-ideal);
#if defined(DEBUG_PRINT)
			printf("\tm: %"PRILOC" M: %"PRILOC"\n", m, M);
			printf("\tnew range  for splitter %d: %"PRILOC", %"PRILOC"\n", j, sp->rmin[j], sp->rmax[j]);
			printf("\tdeltas: %"PRILOC", %"PRILOC"\n", deltal, deltar);
#endif
			// if m or M brings us under the tolerance OR we exhausted all the possible probe values
			if (MIN(deltal, deltar) < tolerance ||
			    (sp->rmax[pb->blk_e[i]]-sp->rmin[pb->blk_s[i]]) <= pb->n) {
				// ...pick the one with a smaller distance from the ideal value
				if (deltar < deltal) m = M;

				if (verbose) {
					printf("\t\tfound satisfying value for splitter %d: %"PRILOC"\n", j, pbv[m]);
				}
				sp->vals[j] = pbv[m];
				sp->sat[j] = ht[m];
				sp->rem--;
			}
		}
		ht += pb->blk_n[i];
		pbv += pb->blk_n[i];
	}
	return;
}

static void print_splitters(split_t *s) {

	int i;
	printf("\n\tcurrent splitters status (rmin, split, rmax):\n");
	for(i = 0; i < s->n; i++) {
		printf("\t\t%d: (%"PRILOC", ", i, s->rmin[i]);
		if (s->sat[i])	printf("%"PRILOC", ", s->vals[i]);
		else		printf("None, ");
		printf("%"PRILOC")\n", s->rmax[i]);
	}	
	return;
}

static inline void exchange_vals(LOCINT *sbuf, int *soff, int *snum,
                                 LOCINT *rbuf, int *roff, int *rnum,
                                 MPI_Request *request, MPI_Status *status) {
#if 1
        int i, p;
	int rank, ntask;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	// move the Irecv-s at the beginning of the computations...
	for(i = 1; i < ntask; i++) {
		p = (rank+i)%ntask;
		MPI_Irecv(rbuf + roff[p], rnum[p], LOCINT_MPI,
			  p, TAG(p), MPI_COMM_WORLD, request+i-1);
	}
        memcpy(rbuf+roff[rank], sbuf+soff[rank], snum[rank]*sizeof(*sbuf));
	MPI_Barrier(MPI_COMM_WORLD);
        for(i = 1; i < ntask; i++) {
                p = (rank+i)%ntask;
                MPI_Send(sbuf + soff[p], snum[p], LOCINT_MPI,
                         p, TAG(rank), MPI_COMM_WORLD);
        }
        MPI_Waitall(ntask-1, request, status);
#else
	MPI_Alltoallv(sbuf, snum, soff, LOCINT_MPI, rbuf, rnum, roff, LOCINT_MPI, MPI_COMM_WORLD);
#endif
        return;
}

// Sorts the n elements in u,v among the NP processor in MPI_COMM_WORLD
// assigning to each processor (n/NP) +- (perc)%. 
// (Note that in some cases it may assign slightly more elements)
void phsort(LOCINT **u, LOCINT **v, int64_t *n, double perc, int verbose) {

	int i;

	LOCINT *u_in = *u;
	LOCINT *v_in = *v;

	LOCINT nloc  = *n;

	split_t	spl;
	probe_t prb;
	pivot_t pvt;

	LOCINT lmin=LOCINT_MAX, lmax=LOCINT_MIN;
	LOCINT gmin=LOCINT_MAX, gmax=LOCINT_MIN;

	int rank, ntask;	
	MPI_Request *request=NULL;
	MPI_Status *status=NULL;

	double tolerance;

	double tmmax;
	double tsort2;
	double tsplit;
	double texch;

	int bbit=32, ebit=0;
	int64_t *hist=NULL, ntot;

	if (0 == nloc) return;
		
	spl.vals = NULL;
	spl.sat  = NULL;
	spl.rmin = NULL;
	spl.rmax = NULL;

	prb.vals = NULL;
	prb.blk_s = NULL;
	prb.blk_e = NULL;
	prb.blk_n = NULL;

	pvt.vals = NULL;
	pvt.locs = NULL;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);
	
	request = (MPI_Request *)Malloc(ntask*sizeof(*request));
	status = (MPI_Status *)Malloc(ntask*sizeof(*status));

	ntot = (int64_t)nloc;
	MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

	tolerance = (((double)ntot)/ntask)*perc;

	int64_t nfinal_max = MAX(nloc, split_idv(0, ntask, ntot)+2*tolerance);
	nfinal_max += nfinal_max/10; // add a 10%
	LOCINT *u_out = (LOCINT *)Malloc(nfinal_max*sizeof(*u_out));
	LOCINT *v_out = (LOCINT *)Malloc(nfinal_max*sizeof(*v_out));

	if (0 == rank && verbose) printf("Initializing cuda structures for parallel sort...");
	double tcinit = MPI_Wtime();
	copy_sort_samples(u_in, v_in, nloc, nfinal_max);
	tcinit = MPI_Wtime()-tcinit;
	if (0 == rank && verbose) printf("done in %lf secs\n\n", tcinit);

	if (0 == rank && verbose) printf("Selecting min/max...");
	tmmax = MPI_Wtime();
	limits_cuda(nloc, &lmin, &lmax, &bbit, &ebit);

	if (0 == rank && verbose) printf("done in %lf secs\n", MPI_Wtime()-tmmax);
	
	MPI_Reduce(&lmin, &gmin, 1, LOCINT_MPI, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&lmax, &gmax, 1, LOCINT_MPI, MPI_MAX, 0, MPI_COMM_WORLD);
	{
		int t1=bbit, t2=ebit;
		MPI_Allreduce(&t1, &bbit, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(&t2, &ebit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	}
	if (0 == rank && verbose) {
		printf("Min/max values (row indices): %"PRILOC" / %"PRILOC"\n", gmin, gmax);
		printf("Total number of elements (edges): %"PRId64"\n", ntot);
		printf("Elements (edges) per processor: %"PRId64" +- %.1lf\n\n", (ntot/ntask), tolerance);
	}

	spl.n = ntask-1;
	spl.rem = spl.n;
	// Ranks != 0 will use it in the final, shuffle phase
	if (ntask > 1) {
		spl.vals = (LOCINT *)Malloc(spl.n*sizeof(*spl.vals));
		spl.sat  = (int64_t *)Malloc(spl.n*sizeof(*spl.sat));
		if (0 == rank) {
			spl.rmin = (LOCINT *)Malloc(spl.n*sizeof(*spl.rmin));
			spl.rmax = (LOCINT *)Malloc(spl.n*sizeof(*spl.rmax));
			for(i = 0; i < spl.n; i++) {
				spl.rmin[i] = gmin;
				spl.rmax[i] = gmax;
			}
			memset(spl.sat, 0, spl.n*sizeof(*spl.sat));
		}

		prb.n = 0;
		prb.n_alloc = 100*spl.n;
		prb.vals = (LOCINT *)Malloc(prb.n_alloc*sizeof(*prb.vals));

		prb.nblk = 0;
		prb.nblk_alloc = 0;

		hist = (int64_t *)Malloc(prb.n_alloc*sizeof(*hist));

		pvt.n = 0;
		pvt.n_alloc = 100*spl.n;
		pvt.vals = (LOCINT *)Malloc(pvt.n_alloc*sizeof(*(pvt.vals)));
		pvt.locs = (int *)Malloc(pvt.n_alloc*sizeof(*(pvt.locs)));
	}

	if (0 == rank && verbose) printf("Sorting values...");
	double tsort1 = MPI_Wtime();
	sort_cuda(NULL, NULL, nloc, bbit, ebit);

	tsort1 = MPI_Wtime() - tsort1;
	if (0 == rank && verbose) printf("done in %lf secs\n\n", tsort1);

	int nit=0;
	int oldrem=INT_MAX;
	int nguess = spl.n;
	
	tsplit = MPI_Wtime();
	while(ntask > 1) { // do not iterate with only one task
		if (0 == rank) {
			if (spl.rem) {
				if (verbose) printf("Iteration %d:\n", nit);
				get_probe(&spl, &prb, nguess);
				//print_probe(&prb);
			} else
				prb.n = 0;
		}

		MPI_Bcast(&prb.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (!prb.n) break;
		if (prb.n >= prb.n_alloc) {
			prb.vals = (LOCINT *)Realloc(prb.vals, prb.n*sizeof(*(prb.vals)));
			hist = (int64_t *)Realloc(hist, prb.n*sizeof(*(hist)));
			prb.n_alloc = prb.n;
		}
		MPI_Bcast(prb.vals, prb.n, LOCINT_MPI, 0, MPI_COMM_WORLD);

		double htime = MPI_Wtime();
		get_hist_cuda(nloc, hist, prb.vals, prb.n);

		htime = MPI_Wtime() - htime;
		if (0 == rank && verbose) printf("\thistogram time: %lf\n", htime);

		MPI_Reduce(rank?hist:MPI_IN_PLACE, hist, prb.n, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		if (0 == rank) {
			//print_hist(hist, prb.n);
			update_splitters(&spl, &prb, hist, ntot, gmin, gmax, tolerance, verbose);
			if (verbose) print_splitters(&spl);
			if (spl.rem == oldrem || spl.rem == spl.n) {
				if (verbose)
					printf("\n\t0 new splitters satisfied, probe size doubled to %d\n", 2*nguess);
				nguess *= 2;
			} else oldrem = spl.rem;
			if (verbose) printf("\n");
		}
		nit++;
		//if (nit == 1) break;
	}
	tsplit = MPI_Wtime()-tsplit;

	if (0 == rank && verbose) {
		printf("Splitters found in %lf secs:\n", tsplit);
		for(i = 0; i < spl.n; i++) {
			printf("\t%12"PRILOC"\t%12"PRId64"\t%12"PRId64"\n",
				spl.vals[i], spl.sat[i], spl.sat[i]-(i?spl.sat[i-1]:0));
		}
		printf("\n");
	}

	if (0 == rank && verbose) printf("Exchanging values...");
	texch = MPI_Wtime();

	// probe length can only grow starting from spl.n, so there is enough space...
	MPI_Bcast(spl.vals, spl.n, LOCINT_MPI, 0, MPI_COMM_WORLD);
	MPI_Bcast(spl.sat, spl.n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

	int64_t nfinal;
	nfinal  = ((rank == ntask-1) ? ntot            : spl.sat[rank]) -
		  ((           rank) ? spl.sat[rank-1] :             0);
	//nfinal -= rank ? spl.sat[rank-1] : 0;


	int *snum = (int *)Malloc(ntask*sizeof(*snum));
	int *soff = (int *)Malloc(ntask*sizeof(*soff));

	getvals_cuda(u_in, v_in, nloc);
	// assumes vals[] sorted
	for(i = 0; i < ntask; i++) {
		soff[i] = (i) ? soff[i-1]+snum[i-1] : 0;

		int eoff;
		if (i < ntask-1) eoff = bisect_left(u_in, nloc, spl.vals[i]);
		else		 eoff = nloc;

		snum[i] = eoff - soff[i];
	}

	assert(soff[ntask-1]+snum[ntask-1] == nloc);
	MPI_Barrier(MPI_COMM_WORLD);		

	int *rnum = (int *)Malloc(ntask*sizeof(*rnum));
	int *roff = (int *)Malloc(ntask*sizeof(*roff));

	MPI_Alltoall(snum, 1, MPI_INT, rnum, 1, MPI_INT, MPI_COMM_WORLD);

	roff[0] = 0;
	for(i = 1; i < ntask; i++)
		roff[i] = roff[i-1] + rnum[i-1];

	assert(roff[ntask-1]+rnum[ntask-1] == nfinal);
	if (nfinal > nfinal_max) {
		fprintf(stderr, "[%d] error nfinal_max=%"PRId64" < nfinal=%"PRId64"\n", rank, nfinal_max, nfinal);
		exit(EXIT_FAILURE);
	}

	// try also with MPI_Alltoallv
	exchange_vals(u_in, soff, snum, u_out, roff, rnum, request, status);
	exchange_vals(v_in, soff, snum, v_out, roff, rnum, request, status);

	MPI_Barrier(MPI_COMM_WORLD);
	texch = MPI_Wtime() - texch;
	if (0 == rank && verbose) printf("done in %lf secs\n", texch);

	if (0 == rank && verbose) printf("Sorting final values...");
	tsort2 = MPI_Wtime();
	
	//sort_cuda(u_out, v_out, nfinal, bbit, ebit);
	final_sort_cuda(u_out, v_out, nfinal);

	tsort2 = MPI_Wtime() - tsort2;
	MPI_Barrier(MPI_COMM_WORLD);

	if (0 == rank && verbose)
		printf("done in %lf secs\n\n", tsort2);

	finalize_sort_cuda();

	if (u_in) free(u_in);
	if (v_in) free(v_in);
	*u = u_out;
	*v = v_out;
	*n = nfinal;

	if (hist) free(hist);

	if (spl.vals) free(spl.vals);
	if (spl.rmin) free(spl.rmin);
	if (spl.rmax) free(spl.rmax);
	if (spl.sat) free(spl.sat);

	if (prb.vals) free(prb.vals);
	if (prb.blk_s) free(prb.blk_s);
	if (prb.blk_e) free(prb.blk_e);
	if (prb.blk_n) free(prb.blk_n);

	if (pvt.vals) free(pvt.vals);
	if (pvt.locs) free(pvt.locs);

	if (request) free(request);
	if (status) free(status);

	return;
}
