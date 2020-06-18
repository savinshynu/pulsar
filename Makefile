CFLAGS = $(shell python-config --cflags) \
         $(shell python -c "from __future__ import print_function; import numpy; print('-I' + numpy.get_include())") \
         $(shell pkg-config --cflags fftw3) \
         -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

LDFLAGS = $(shell python-config --ldflags) $(shell pkg-config --libs fftw3f)

.PHONY: all
all: _psr.so _helper.so

_psr.so: psr.o utils.o fft.o kurtosis.o dedispersion.o reduce.o quantize.o
	$(CC) -o _psr.so psr.o utils.o fft.o kurtosis.o dedispersion.o reduce.o quantize.o -lm -shared -fopenmp -march=native $(LDFLAGS)
	
psr.o: psr.c
	$(CC) -c $(CFLAGS) -fPIC -o psr.o psr.c -fopenmp -march=native 
	
utils.o: utils.c
	$(CC) -c $(CFLAGS) -fPIC -o utils.o utils.c -fopenmp -march=native 
	
fft.o: fft.c
	$(CC) -c $(CFLAGS) -fPIC -o fft.o fft.c -fopenmp -march=native 
	
kurtosis.o: kurtosis.c
	$(CC) -c $(CFLAGS) -fPIC -o kurtosis.o kurtosis.c -fopenmp -march=native 
	
dedispersion.o: dedispersion.c
	$(CC) -c $(CFLAGS) -fPIC -o dedispersion.o dedispersion.c -fopenmp -march=native 
	
reduce.o: reduce.c
	$(CC) -c $(CFLAGS) -fPIC -o reduce.o reduce.c -fopenmp -march=native 
	
quantize.o: quantize.c
	$(CC) -c $(CFLAGS) -fPIC -o quantize.o quantize.c -fopenmp -march=native 
	
_helper.so: helper.o
	$(CC) -o _helper.so helper.o $(LDFLAGS) -lm -shared -fopenmp -march=native 
	
helper.o: helper.c
	$(CC) -c $(CFLAGS) -fPIC -o helper.o helper.c -fopenmp -march=native 
	
clean:
	rm -rf psr.o utils.o fft.o kurtosis.o dedispersion.o reduce.o quantize.o _psr.so helper.o _helper.so
