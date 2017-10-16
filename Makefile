CFLAGS = $(shell python-config --cflags) \
         $(shell python -c "import numpy; print '-I' + numpy.get_include()") \
         $(shell pkg-config --cflags fftw3) \
         -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

LDFLAGS = $(shell python-config --ldflags) $(shell pkg-config --libs fftw3f)

.PHONY: all
all: _psr.so _helper.so

_psr.so: psr.o
	$(CC) -o _psr.so psr.o -lm -shared -fopenmp -march=native $(LDFLAGS)
	
psr.o: psr.c
	$(CC) -c $(CFLAGS) -fPIC -o psr.o psr.c -fopenmp -march=native 

	
_helper.so: helper.o
	$(CC) -o _helper.so helper.o $(LDFLAGS) -lm -shared -fopenmp -march=native 
	
helper.o: helper.c
	$(CC) -c $(CFLAGS) -fPIC -o helper.o helper.c -fopenmp -march=native 
	
clean:
	rm -rf psr.o _psr.so helper.o _helper.so
