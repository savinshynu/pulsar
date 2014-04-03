CFLAGS = $(shell python-config --cflags) $(shell python -c "import numpy; print '-I' + numpy.get_include()") $(shell pkg-config --cflags fftw3)

LDFLAGS = $(shell python-config --ldflags) $(shell pkg-config --libs fftw3f)

_psr.so: psr.o
	$(CC) -o _psr.so psr.o -lm -shared -fopenmp $(LDFLAGS)
	
psr.o:
	$(CC) -c $(CFLAGS) -fPIC -o psr.o psr.c -funroll-loops -O3 -fopenmp
	
clean:
	rm -rf psr.o _psr.so
