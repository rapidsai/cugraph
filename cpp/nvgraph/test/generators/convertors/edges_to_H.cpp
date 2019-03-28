#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

struct edge {
  unsigned long int r;
  unsigned long int c;
};

void printUsageAndExit()
{
  printf("%s", "Fatal Error\n");
  printf("%s", "Usage: ./H edges.dat\n");
  printf("%s", "Input : Graph given as a sorted set of edges\n");
  printf("%s", "Output : Row sub-stochastic matrix in MatrixMarket format\n");
  exit(0);
}

int main (int argc, char *argv[])
{
	// Check args
  	if (argc != 2) printUsageAndExit();
  	
  	// Vars
  	unsigned long int n, nz, i = 0, current_r, nbr = 1;
  	int ok;
	double scal;
	char outp[128], cc;
	FILE *fpin = NULL, *fpout = NULL;
	edge e;
  	std::vector<struct edge> row;
  	// Get I/O names
  	// The output is filename.mtx
  	  while (argv[1][i] != '\0')
    {outp[i] = argv[1][i];i++;}
	  outp[i] = '_'; i++;
	  outp[i] = 'm';i++;outp[i] = 't';i++;outp[i] = 'x';i++;
	  outp[i]='\0';
  	
  	// Open files
	fpin = fopen(argv[1],"r");
	fpout = fopen(outp,"w");
  	if (!fpin || !fpout)
  	{
  		printf("%s", "Fatal Error : I/O fail\n");
  		exit(0);
  	}
  	
  	// Get n and nz
  	fscanf(fpin,"%lu",&n);
  	fscanf(fpin,"%lu",&n);
  	fscanf(fpin,"%lu",&nz);

	fprintf(fpout, "%s", "%%" );
	fprintf(fpout,"MatrixMarket matrix coordinate real general\n");
	fprintf(fpout,"%lu %lu %lu\n",n, n, nz);
	
	// Read the first edge
	ok = fscanf(fpin,"%lu",&e.r);
	if (ok)
	{
		fscanf(fpin,"%lu",&e.c);
		current_r = e.r;
		row.push_back(e);
	}
	else
	{
		printf("%s", "Fatal Error : Wrong data format\n");
  		exit(0);
	}
	
	//Loop
	for (i=0; i<nz-1; i++)
	{	
		fscanf(fpin,"%lu",&e.r);
		fscanf(fpin,"%lu",&e.c);
		if (current_r == e.r)
		{
			nbr++;
		}
		else
		{
			current_r = e.r;
			scal = 1.0/nbr;
			for (std::vector<struct edge>::iterator it = row.begin() ; it != row.end(); ++it)
				fprintf(fpout,"%lu %lu %.9lf\n",it->r, it->c, scal);
			row.clear();
			nbr = 1;
		}
		row.push_back(e);
	}
	// Last print
	scal = 1.0/nbr;
	for (std::vector<struct edge>::iterator it = row.begin() ; it != row.end(); ++it)
		fprintf(fpout,"%lu %lu %.9f\n",it->r, it->c, scal);

	return 0;
}

