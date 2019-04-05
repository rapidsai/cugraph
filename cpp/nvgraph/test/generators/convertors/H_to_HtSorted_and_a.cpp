#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>    // std::sort
#include <vector>       // std::vector
// This code transpose a matrix H and compute the flag vector of empty rows a.
// We assume that H is row-substochastic, in MatrixMarket format and data are sorted by row id
// The output is filename_T.filetype, H is printed first then a is printed.

struct elt {
  long int r;
  long int c;
  double v;
};

void printUsageAndExit()
{
  printf("%s", "Fatal Error\n");
  printf("%s", "Usage: ./HTA H.mtx\n");
  printf("%s", "NOTE1: H is the row-substochastic matrix of a graph\n");
  printf("%s", "NOTE2: H is in MatrixMarket coordinate real general format\n");
  printf("%s", "NOTE3: Data are sorted by row id\n");
  printf("%s", "Output : H^t and the bookmark vector of empty rows\n");
  printf("%s", "***This output fits the input of AMGX PageRank***\n");
  exit(0);
}

inline bool operator< (const elt& a, const elt& b)
{ // ordered by row and then by colum inside a row
  return a.r<b.r || (a.r==b.r && a.c<b.c ) ;  
}

int main (int argc, char *argv[])
{
  // Check args
  if (argc == 1) printUsageAndExit();
  
  // Vars
  long int n, nz, start, i = 0 ,j, k, lastr;
  double v;
  char outp[128], cc;
  FILE *fpin = NULL, *fpout = NULL;
  elt e;
  std::vector<struct elt> A;
  std::vector<unsigned int> a;
  // Get I/O names
  // The output is filename_T
  while (argv[1][i] != '\0')
  {outp[i] = argv[1][i];i++;}
  outp[i] = '_'; i++;
  outp[i] = 'T';i++;
  outp[i]='\0';
  // Open files
  fpin = fopen(argv[1],"r");
  fpout = fopen(outp,"w");
  if (!fpin || !fpout)
  {
    printf("%s", "Fatal Error : I/O fail\n");
    exit(0);
  }
  
  // Skip lines starting with "%%""
  do
  {
    cc = fgetc(fpin); 
    if (cc == '%') fgets(outp,128,fpin);
  }
  while (cc == '%');
  fseek( fpin, -1, SEEK_CUR );

  // Get n and nz
  fscanf(fpin,"%ld",&n);
  fscanf(fpin,"%ld",&n);
  fscanf(fpin,"%ld",&nz);

  // Print format and size
  fprintf(fpout, "%s", "%%");
  fprintf(fpout,"MatrixMarket matrix coordinate real general\n");
  fprintf(fpout, "%s", "%%");
  fprintf(fpout,"AMGX rhs\n");
  fprintf(fpout,"%ld %ld %ld\n",n, n, nz);

  // Empty rows at the begining
  fscanf(fpin,"%ld",&e.c);
  fscanf(fpin,"%ld",&e.r);
  fscanf(fpin,"%lf",&e.v);
  A.push_back(e);

  for (j=0; j<static_cast<int>(e.c)-1; j++)
  {
    std::cout<<e.c<<' '<<e.r<<' '<<e.v<<'\n';
    a.push_back(1);
  }

    // Loop
  for (i=0; i< nz-1;i++)
  {
    lastr = e.c;
    fscanf(fpin,"%ld",&e.c);
    fscanf(fpin,"%ld",&e.r);
    fscanf(fpin,"%lf",&e.v);
    A.push_back(e);

    if (e.c > lastr)
    {
      if (e.c > lastr+1)
      {
        a.push_back(0); 
        //Successive empty rows 
        for (k=0; k<static_cast<int>(e.c)-lastr-1; k++)
          a.push_back(1);
      }
      else
        a.push_back(0);
    }
  }
  a.push_back(0);

  // Empty rows at the end
  for (k=a.size(); k<n; k++)
  {
    a.push_back(1);
  }

  std::sort (A.begin(), A.end());
  for (std::vector<struct elt>::iterator it = A.begin() ; it != A.end(); ++it)
    fprintf(fpout,"%ld %ld %.9f\n",it->r, it->c, it->v);

  for (std::vector<unsigned int>::iterator it = a.begin() ; it != a.end(); ++it)
    fprintf(fpout,"%u\n",*it);

  return 0;

}

