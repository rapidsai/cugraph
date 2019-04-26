#include <stdio.h>
#include <stdlib.h>
#include <algorithm>    // std::sort
#include <vector>       // std::vector

struct edge {
  unsigned long int r;
  unsigned long int c;
};

void printUsageAndExit()
{
  printf("%s", "Fatal Error\n");
  printf("%s", "Usage: ./sort edges.dat\n");
  printf("%s", "Input : Graph in matrix market parttern format");
  printf("%s", "Output : Graph with sorted edges in matrix market parttern format\n");
  exit(0);
}

inline bool operator< (const edge& a, const edge& b){ if(a.r<b.r) return true; else return false; }

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
  std::vector<struct edge> edges;

  // Get I/O names
  // The output is filename.mtx
  while (argv[1][i] != '\0')
    {outp[i] = argv[1][i];i++;}
  outp[i] = '_'; i++;
  outp[i] = 's';i++;
  outp[i]='\0';
  	
  	// Open files
	fpin = fopen(argv[1],"r");
	fpout = fopen(outp,"w");
	if (!fpin || !fpout)
	{
		printf("%s", "Fatal Error : I/O fail\n");
		exit(0);
	}

	// Skip lines starting with "%""
	do
	{
		cc = fgetc(fpin); 
		if (cc == '%') fgets(outp,128,fpin);
	}
	while (cc == '%');
	fseek( fpin, -1, SEEK_CUR );

	// Get n and nz
	fscanf(fpin,"%lu",&n);
	//fscanf(fpin,"%lu",&n);
	fscanf(fpin,"%lu",&nz);
  	fprintf(fpout,"%lu %lu %lu\n",n, n, nz);
	// Read the first edge
	ok = fscanf(fpin,"%lu",&e.r);
	if (ok)
	{
		fscanf(fpin,"%lu",&e.c);
		edges.push_back(e);
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
		edges.push_back(e);
	}
  std::sort (edges.begin(), edges.end());
  for (std::vector<struct edge>::iterator it = edges.begin() ; it != edges.end(); ++it)
      fprintf(fpout,"%lu %lu\n",it->r, it->c);
	return 0;
}

