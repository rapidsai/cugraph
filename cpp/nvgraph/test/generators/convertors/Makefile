CC=g++
CFLAGS=-O3 -march=native -pipe -w
LDFLAGS=-lm

all: sort HTA H mtob

sort: sort_eges.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ 

HTA: H_to_HtSorted_and_a.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ 
 
H: edges_to_H.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ 

mtob: binary_converter.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ 

clean: 
	rm sort HTA mtob

