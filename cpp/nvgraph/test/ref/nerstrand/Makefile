CC=g++
CFLAGS=-O3 -fopenmp 
LDFLAGS=-I. -L. libnerstrand.a
EXEC=nerstrand_bench
SOURCES=nerstrand_driver.cpp mmio.cpp
OBJECTS=$(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mmio.o: mmio.cpp mmio.h
	$(CC) $(CFLAGS) -c $<

nerstand_driver.o: nerstand_driver.cpp mmio.h
	$(CC) $(CFLAGS) -c $<
clean:
	rm *.o