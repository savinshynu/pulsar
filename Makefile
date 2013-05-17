CFLAGS = $(shell python-config --includes) $(shell python -c "import numpy; print '-I' + numpy.__path__[0] + '/core/include/numpy/'")

_psr.so: psr.o
	gcc $(CFLAGS) -o _psr.so psr.o -lm -shared -fopenmp -lfftw3
	
psr.o:
	gcc -c $(CFLAGS) -fPIC -o psr.o psr.c -funroll-loops -O3 -fopenmp
	
clean:
	rm -rf psr.o _psr.so
