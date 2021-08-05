.PHONY: all
all: _psr.so _helper.so

_psr.so: psr.c utils.c fft.c kurtosis.c dedispersion.c reduce.c quantize.c setup.py
	python setup.py build
	mv build/lib*/*.so .
	rm -rf build
	
_helper.so: helper.c setup.py
	python setup.py build
	mv build/lib*/*.so .
	rm -rf build
	
clean:
	rm -rf _psr.so _helper.so
