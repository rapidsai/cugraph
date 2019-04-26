#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     
#include <string.h>
#include <algorithm>    // std::sort etc.
#include <vector>       // std::vector
#include <iostream>     // std::cout

typedef int idx_t;
typedef double val_t;

void printUsageAndExit()
{
  printf("%s", "Usage:./mtob M.mtx\n");
  printf("%s", "NOTE1: M is square, in MatrixMarket coordinate real general format\n");
  printf("%s", "NOTE2: Data are sorted by row id\n");

  exit(0);
}

void print_csr(	std::vector<idx_t> &row_ptrs, std::vector<idx_t> &col_indices, std::vector<val_t> &val)
{
	for (std::vector<idx_t>::iterator it = row_ptrs.begin(); it != row_ptrs.end(); ++it)
    	std::cout << ' ' << *it;
  	std::cout << '\n';
  	for (std::vector<idx_t>::iterator it = col_indices.begin(); it != col_indices.end(); ++it)
    	std::cout << ' ' << *it;
  	std::cout << '\n';
  	for (std::vector<val_t>::iterator it = val.begin(); it != val.end(); ++it)
    	std::cout << ' ' << *it;
  	std::cout << '\n';
}

// Generates csr from matrix market format
void read_csr(	FILE *fpin,
				idx_t n, 
				idx_t nz,
				std::vector<idx_t> &row_weight,
				std::vector<idx_t> &row_ptrs,
				std::vector<idx_t> &col_indices,
				std::vector<val_t> &val)
{
	idx_t weight=0, i=0 ,j=0, k=0, lastr=0, r=0, c=0;
	double v;
	// Empty rows at the begining
	fscanf(fpin,"%d",&r);
	fscanf(fpin,"%d",&c);
	fscanf(fpin,"%lf",&v);
	row_ptrs.push_back(0);
	col_indices.push_back(c-1);
	val.push_back(v);
	weight++;

	for (j=0; j<r-1; j++)
	{
		row_ptrs.push_back(0);
		row_weight.push_back(0);
	}
	
  	// Loop
  	for (i=1; i< nz;i++)
	{
		lastr = r;
		fscanf(fpin,"%d",&r);
		fscanf(fpin,"%d",&c);
		fscanf(fpin,"%lf",&v);
		col_indices.push_back(c-1);
		val.push_back(v);
		
		if (lastr == r)
			weight++;
		else if (lastr < r)// new row
		{
			row_ptrs.push_back(row_ptrs.back()+weight);
			row_weight.push_back(weight);
			//Successive empty rows	
			for (k=row_weight.size(); k<r-1; k++)
			{
				row_ptrs.push_back(row_ptrs.back());
				row_weight.push_back(0);
			}
			weight = 1;
		}
		else
		{
			printf("%s", "Fatal Error : Data have to be sorted by row id\n");
  			exit(0);
		}
	}

	row_ptrs.push_back(row_ptrs.back()+weight);
	row_weight.push_back(weight);	

	// Empty rows at the end
	for (k=row_weight.size(); k<n; k++)
	{
		row_ptrs.push_back(row_ptrs.back());
		row_weight.push_back(0);
	}
}
void read_vector_mtx( 	FILE *fpin,
						idx_t n, 
						std::vector<val_t> &a)
{
	val_t v;
	for (idx_t i=0; i< n;i++)
	{
		fscanf(fpin,"%lf",&v);
		a.push_back(v);
	}
}
void write_csr_bin (char *argv[], 
					idx_t n, 
					idx_t nz, 
					std::vector<idx_t> &row_weight,
					std::vector<idx_t> &row_ptrs,
					std::vector<idx_t> &col_indices,
					std::vector<val_t> &val,
					std::vector<val_t> &a
					)
{
	idx_t i;
	char outp [128];
	// Generate output name
  	while (argv[1][i] != '\0')
  	{
  		outp[i] = argv[1][i];
  		i++;
  	}
  	outp[i] = '_';i++; 
  	outp[i] = 'b';i++;
  	outp[i] = 'i';i++;
  	outp[i] = 'n';i++;
 	outp[i]='\0';
 	FILE *fpout = NULL;
	
    fpout = fopen(outp,"w");
    if (!fpout)
  	{
  		printf("%s", "Fatal Error : I/O fail\n");
  		exit(0);
  	}
	const char header [] = "%%NVAMGBinary\n";
	const int system_header_size = 9;
	uint32_t system_flags [] = { 1, 1, 0, 0, 0, 1, 1, n, nz };
    fwrite(header, sizeof(char), strlen(header), fpout);
    fwrite(system_flags, sizeof(uint32_t), system_header_size, fpout);
    fwrite(&row_ptrs[0], sizeof(idx_t), row_ptrs.size(), fpout);
    fwrite(&col_indices[0], sizeof(idx_t), col_indices.size(), fpout);
    fwrite(&val[0], sizeof(val_t), val.size(), fpout);
    fwrite(&a[0], sizeof(val_t), a.size(), fpout);
    fclose(fpout);
}
int main (int argc, char **argv)
{
  	// Vars
  	idx_t i = 0;
  	idx_t n=0, m=0, nz=0, nparts=0, sym=0;
	char dum[128], cc;
	FILE *fpin = NULL;
	std::vector<idx_t> row_ptrs, col_indices, row_weight;
	std::vector<val_t> a , val;

	// Check args

  	if (argc != 2) printUsageAndExit();
  	
  	// Open file
	fpin = fopen(argv[1],"r");
  	if (!fpin)
  	{
  		printf("%s", "Fatal Error : I/O fail\n");
  		exit(0);
  	}
  	
  	// Skip lines starting with "%%""
  	do
  	{
  		cc = fgetc(fpin); 
  		if (cc == '%') fgets(dum,128,fpin);
  	}
  	while (cc == '%');
  	fseek( fpin, -1, SEEK_CUR );

  	// Get n and nz
  	fscanf(fpin,"%ld",&n);
  	fscanf(fpin,"%ld",&m);
  	fscanf(fpin,"%ld",&nz);
  	if (n != m)
  	{
  		printf("%s", "Fatal Error : The matrix is not square\n");
  		exit(0);
  	}

	//printf("Reading...\n");
    read_csr(fpin, n, nz, row_weight, row_ptrs, col_indices, val);  
    read_vector_mtx(fpin, n, a);  
  	
  	//printf("Writing...\n");
    write_csr_bin(argv, n, nz, row_weight, row_ptrs, col_indices, val,a);
  
    //printf("Success!\n");
	return 0;
}

