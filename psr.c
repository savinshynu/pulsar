#include "Python.h"
#include <math.h>
#include <stdio.h>
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <pthread.h>

#ifdef _OPENMP
	#include <omp.h>
	
	// OpenMP scheduling method
	#ifndef OMP_SCHEDULER
	#define OMP_SCHEDULER dynamic
	#endif
#endif

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

// Dispersion constant in MHz^2 s / pc cm^-3
#define DCONST (double) 4.148808e3


/*
 Load in FFTW wisdom.  Based on the read_wisdom function in PRESTO.
*/

void read_wisdom(char *filename, PyObject *m) {
	int status = 0;
	FILE *wisdomfile;
	
	wisdomfile = fopen(filename, "r");
	if( wisdomfile != NULL ) {
		status = fftwf_import_wisdom_from_file(wisdomfile);
		PyModule_AddObject(m, "useWisdom", PyBool_FromLong(status));
		fclose(wisdomfile);
	} else {
		PyModule_AddObject(m, "useWisdom", PyBool_FromLong(status));
	}
}


/*
  Core binding function - based off the corresponding bifrost function
*/

#include <pthread.h>


/*
 setCore - Internal function to bind a thread to a particular core, or unbind
 it if core is set to -1.  Returns 0 is successful, non-zero otherwise.
*/         

int setCore(int core) {
#if defined __linux__ && __linux__
	int ncore;
	cpu_set_t cpuset;
	ncore = sysconf(_SC_NPROCESSORS_ONLN);
	
	// Basic validation
	if( core >= ncore || (core < 0 && core != -1) ) {
		return -100;
	}
	
	CPU_ZERO(&cpuset);
	if( core >= 0 ) {
		CPU_SET(core, &cpuset);
	} else {
		for(core=0; core<ncore; ++core) {
			CPU_SET(core, &cpuset);
		}
	}
	
	pthread_t tid = pthread_self();
	return pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
#else
	return -101;
#endif
}

/*
 getCore - Internal function to get the core binding of a thread.  Returns 
 0 is successful, non-zero otherwise.
*/

int getCore(int* core) {
#if defined __linux__ && __linux__
	int ret, c, ncore;
	cpu_set_t cpuset;
	ncore = sysconf(_SC_NPROCESSORS_ONLN);
	
	pthread_t tid = pthread_self();
	ret = pthread_getaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
	if( ret == 0 ) {
		if( CPU_COUNT(&cpuset) > 1 ) {
			*core = -1;
			return 0;
		} else {
			for(c=0; c<ncore; ++c ) {
				if(CPU_ISSET(c, &cpuset)) {
					*core = c;
					return 0;
				}
			}
		}
	}
	return ret;
#else
	return -101;
#endif
}

/* getCoreCount - Internal function to return the number of core available.  Returns >=1
   if successful, <1 otherwise.
*/

int getCoreCount(void) {
#if defined __linux__ && __linux__
	return sysconf(_SC_NPROCESSORS_ONLN);
#else
	return -101;
#endif
}

static PyObject *BindToCore(PyObject *self, PyObject *args, PyObject *kwds) {
	int ret, core = -1;
	
	if(!PyArg_ParseTuple(args, "i", &core)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	ret = setCore(core);
	
	if(ret == 0) {
		Py_RETURN_TRUE;
	} else if(ret == -100) {
		PyErr_Format(PyExc_ValueError, "Invalid core: %d", core);
		return NULL;
	} else if(ret == -101) {
		PyErr_Warn(PyExc_RuntimeWarning, "Changing of the thread core binding is not supported on this OS");
		Py_RETURN_FALSE;
	} else {
		PyErr_Format(PyExc_RuntimeError, "Cannot change core binding to %d", core);
		return NULL;
	}
}

PyDoc_STRVAR(BindToCore_doc, \
"Bind the current thread to the specified core.\n\
\n\
Input arguments are:\n\
 * core: scalar int core to bind to\n\
\n\
Outputs:\n\
 * True, if successful\n\
");


static PyObject *BindOpenMPToCores(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *cores, *core;
	int ret, nthread, t, tid, ncore, old_core, c;
	
	if(!PyArg_ParseTuple(args, "O", &cores)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	if(!PyList_Check(cores)) {
		PyErr_Format(PyExc_TypeError, "Invalid parameters");
		return NULL;
	}
	
	nthread = PyList_Size(cores);
	ncore = getCoreCount();
	if( ncore == -101 ) {
		PyErr_Warn(PyExc_RuntimeWarning, "Changing of the thread core binding is not supported on this OS");
		Py_RETURN_FALSE;
	}
	for(t=0; t<nthread; t++) {
		core = PyList_GetItem(cores, t);
		if(!PyInt_Check(core)) {
			PyErr_Format(PyExc_TypeError, "Invalid parameters");
			return NULL;
		}
		c = (int) PyInt_AsLong(core);
		if( c >= ncore || (c < 0 && c != -1 ) ) {
			PyErr_Format(PyExc_ValueError, "Invalid core for thread %d: %d", t+1, c);
			return NULL;
		}
	}
	
	ret = getCore(&old_core);
	if(ret == -101) {
		PyErr_Warn(PyExc_RuntimeWarning, "Changing of the thread core binding is not supported on this OS");
		Py_RETURN_FALSE;
	} else if(ret != 0) {
		PyErr_Format(PyExc_RuntimeError, "Cannot get core binding");
		return NULL;
	}
	
	ret = setCore(-1);
	if(ret != 0) {
		PyErr_Format(PyExc_RuntimeError, "Cannot unbind current thread");
		return NULL;
	}
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(tid, core, c)
		omp_set_num_threads(nthread);
		
		#pragma omp parallel for schedule(static, 1)
		for(t=0; t<nthread; ++t) {
			tid = omp_get_thread_num();
			core = PyList_GetItem(cores, tid);
			c = (int) PyInt_AsLong(core);
			ret |= setCore(c);
		}
	#else
		ret = -101;
	#endif
	if(ret == -101) {
		PyErr_Warn(PyExc_RuntimeWarning, "Changing of the thread core binding is not supported on this OS");
		Py_RETURN_FALSE;
	} else if(ret != 0) {
		PyErr_Format(PyExc_RuntimeError, "Cannot set all OpenMP thread core bindings");
		return NULL;
	}
	
	if(old_core != -1) {
		ret = setCore(old_core);
		if(ret != 0) {
			PyErr_Format(PyExc_RuntimeError, "Cannot re-bind current thread to %d", old_core);
			return NULL;
		}
	}
	
	Py_RETURN_TRUE;
}

PyDoc_STRVAR(BindOpenMPToCores_doc, \
"Bind OpenMP threads to the provided list of cores.\n\
\n\
Input arguments are:\n\
 * cores: list of int cores for OpenMP thread binding\n\
\n\
Outputs:\n\
  * True, if successful\n\
");


/*
  Complex magnitude squared functions
*/

double cabs2(double complex z) {
	return creal(z)*creal(z) + cimag(z)*cimag(z);
}

float cabs2f(float complex z) {
	return crealf(z)*crealf(z) + cimagf(z)*cimagf(z);
}


static PyObject *PulsarEngineRaw(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF=NULL;
	PyArrayObject *data, *dataF;
	int nChan = 64;
	
	long ij, i, j, k, nStand, nSamps, nFFT;
	
	static char *kwlist[] = {"signals", "LFFT", "signalsF", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO", kwlist, &signals, &nChan, &signalsF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 2, 2);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 2-D complex64");
		return NULL;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nSamps = (long) PyArray_DIM(data, 1);
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[3];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dims[2] = (npy_intp) (nSamps/nChan);
	if( signalsF != NULL ) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(signalsF, NPY_COMPLEX64, 3, 3);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output signalsF array to 3-D complex64");
			Py_XDECREF(data);
			return NULL;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of stands");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of channels");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 2) != dims[2]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of FFT windows");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(3, dims, NPY_COMPLEX64, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			Py_XDECREF(data);
			return NULL;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Create the FFTW plan
	float complex *inP, *in;
	inP = (float complex*) fftwf_malloc(sizeof(float complex) * nChan);
	fftwf_plan p;
	p = fftwf_plan_dft_1d(nChan, inP, inP, FFTW_FORWARD, FFTW_ESTIMATE);
	
	// FFT
	long secStart;
	float complex *a;
	float complex *b;
	a = (float complex *) PyArray_DATA(data);
	b = (float complex *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(in, secStart, i, j, k)
	#endif
	{
		in = (float complex*) fftwf_malloc(sizeof(float complex) * nChan);
		
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nFFT; ij++) {
			i = ij / nFFT;
			j = ij % nFFT;
			
			secStart = nSamps * i + nChan*j;
			
			for(k=0; k<nChan; k++) {
				in[k]  = *(a + secStart + k);
			}
			
			fftwf_execute_dft(p, in, in);
			
			for(k=0; k<nChan/2+nChan%2; k++) {
				*(b + nFFT*nChan*i + nFFT*(k + nChan/2) + j) = in[k] / sqrt(nChan);
			}
			for(k=nChan/2+nChan%2; k<nChan; k++) {
				*(b + nFFT*nChan*i + nFFT*(k - nChan/2 - nChan%2) + j) = in[k] / sqrt(nChan);
			}
		}
		
		fftwf_free(in);
	}
	fftwf_destroy_plan(p);
	fftwf_free(inP);
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(dataF);
	
	return signalsF;
}

PyDoc_STRVAR(PulsarEngineRaw_doc, \
"Perform a series of Fourier transforms on complex-valued data to get sub-\n\
integration data with linear polarization\n\
\n\
Input arguments are:\n\
 * signals: 2-D numpy.complex64 (stands by samples) array of data to FFT\n\
\n\
Input keywords are:\n\
 * LFFT: number of FFT channels to make (default=64)\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.uint8 (stands by channels) of FFT'd data\n\
");


static PyObject *PulsarEngineRawWindow(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *window, *signalsF=NULL;
	PyArrayObject *data=NULL, *win=NULL, *dataF=NULL;
	int nChan = 64;
	
	long ij, i, j, k, nStand, nSamps, nFFT;
	
	static char *kwlist[] = {"signals", "window", "LFFT", "signalsF", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iO", kwlist, &signals, &window, &nChan, &signalsF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		goto fail;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 2, 2);
	win  = (PyArrayObject *) PyArray_ContiguousFromObject(window,  NPY_DOUBLE,    1, 1);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 2-D complex64");
		goto fail;
	}
	if( win == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input window array to 1-D double");
		goto fail;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nSamps = (long) PyArray_DIM(data, 1);
	
	if( PyArray_DIM(win, 0) != nChan ) {
		PyErr_Format(PyExc_RuntimeError, "Window length does not match requested FFT length");
		goto fail;
	}
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[3];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dims[2] = (npy_intp) (nSamps/nChan);
	if( signalsF != NULL ) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(signalsF, NPY_COMPLEX64, 3, 3);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output signalsF array to 3-D complex64");
			Py_XDECREF(data);
			return NULL;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of stands");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of channels");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 2) != dims[2]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of FFT windows");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(3, dims, NPY_COMPLEX64, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			Py_XDECREF(data);
			return NULL;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Create the FFTW plan
	float complex *inP, *in;
	inP = (float complex*) fftwf_malloc(sizeof(float complex) * nChan);
	fftwf_plan p;
	p = fftwf_plan_dft_1d(nChan, inP, inP, FFTW_FORWARD, FFTW_ESTIMATE);
	
	// FFT
	long secStart;
	float complex *a;
	float complex *b;
	double *c;
	a = (float complex *) PyArray_DATA(data);
	b = (float complex *) PyArray_DATA(dataF);
	c = (double *) PyArray_DATA(win);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(in, secStart, i, j, k)
	#endif
	{
		in = (float complex*) fftwf_malloc(sizeof(float complex) * nChan);
		
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nFFT; ij++) {
			i = ij / nFFT;
			j = ij % nFFT;
			
			secStart = nSamps * i + nChan*j;
			
			for(k=0; k<nChan; k++) {
				in[k]  = *(a + secStart + k);
				in[k] *= *(c + k);
			}
			
			fftwf_execute_dft(p, in, in);
			
			for(k=0; k<nChan/2+nChan%2; k++) {
				*(b + nFFT*nChan*i + nFFT*(k + nChan/2) + j) = in[k] / sqrt(nChan);
			}
			for(k=nChan/2+nChan%2; k<nChan; k++) {
				*(b + nFFT*nChan*i + nFFT*(k - nChan/2 - nChan%2) + j) = in[k] / sqrt(nChan);
			}
		}
		
		fftwf_free(in);
	}
	fftwf_destroy_plan(p);
	fftwf_free(inP);
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(win);
	Py_XDECREF(dataF);
	
	return signalsF;
	
fail:
	Py_XDECREF(data);
	Py_XDECREF(win);
	Py_XDECREF(dataF);
	
	return NULL;
}

PyDoc_STRVAR(PulsarEngineRawWindow_doc, \
"Similar to PulsarEngineRaw but has an additional arugment of the window\n\
function to apply to the data prior to the FFT.\n\
\n\
Input arguments are:\n\
 * signals: 2-D numpy.complex64 (stands by samples) array of data to FFT\n\
 * window: 1-D numpy.float64 array of the window to use (must be the same\n\
   length at the FFT)\n\
\n\
Input keywords are:\n\
 * LFFT: number of FFT channels to make (default=64)\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.uint8 (stands by channels) of FFT'd data\n\
");


static PyObject *PhaseRotator(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *f1, *f2, *signalsF=NULL;
	PyArrayObject *data=NULL, *freq1=NULL, *freq2=NULL, *dataF=NULL;
	double delay;	

	long ij, i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", "freq1", "freq2", "delays", "signalsF", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOd|O", kwlist, &signals, &f1, &f2, &delay, &signalsF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		goto fail;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	freq1 = (PyArrayObject *) PyArray_ContiguousFromObject(f1, NPY_DOUBLE, 1, 1);
	freq2 = (PyArrayObject *) PyArray_ContiguousFromObject(f2, NPY_DOUBLE, 1, 1);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 3-D complex64");
		goto fail;
	}
	if( freq1 == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input f1 array to 1-D double");
		goto fail;
	}
	if( freq1 == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input f2 array to 1-D double");
		goto fail;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nChan  = (long) PyArray_DIM(data, 1);
	nFFT   = (long) PyArray_DIM(data, 2);
	
	// Validate
	if( PyArray_DIM(freq1, 0) != nChan ) {
		PyErr_Format(PyExc_ValueError, "Frequency array 1 has different dimensions than rawSpectra");
		goto fail;
	}
	if( PyArray_DIM(freq2, 0) != nChan ) {
		PyErr_Format(PyExc_ValueError, "Frequency array 2 has different dimensions than rawSpectra");
		goto fail;
	}	
	
	// Find out how large the output array needs to be and initialize it
	npy_intp dims[3];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dims[2] = (npy_intp) nFFT;
	if(signalsF != NULL) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(signalsF, NPY_COMPLEX64, 3, 3);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output signalsF array to 3-D complex64");
			Py_XDECREF(data);
			return NULL;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of stands");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of channels");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 2) != dims[2]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of FFT windows");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(3, dims, NPY_COMPLEX64, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			goto fail;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStart;
	double tempF;
	float complex *a, *d;
	double *b, *c;
	a = (float complex *) PyArray_DATA(data);
	b = (double *) PyArray_DATA(freq1);
	c = (double *) PyArray_DATA(freq2);
	d = (float complex *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempF)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nChan; ij++) {
			i = ij / nChan;
			j = ij % nChan;
			
			secStart = nSamps*i + nFFT*j;
			if( i/2 == 0 ) {
				tempF = *(b + j);
			} else {
				tempF = *(c + j);
			}
			
			for(k=0; k<nFFT; k++) {
				*(d + secStart + k) = *(a + secStart + k) * cexp(2*NPY_PI*_Complex_I*tempF*delay);
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(freq1);
	Py_XDECREF(freq2);
	Py_XDECREF(dataF);
	
	return signalsF;
	
fail:
	Py_XDECREF(data);
	Py_XDECREF(freq1);
	Py_XDECREF(freq2);
	Py_XDECREF(dataF);
	
	return NULL;
}

PyDoc_STRVAR(PhaseRotator_doc, \
"Given the output of PulsarEngineRaw, apply a sub-sample delay as a phase\n\
rotation to each channel\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
 * freq1: 1-D numpy.float64 array of frequencies for each channel for the\n\
   first two stands in signals\n\
 * freq2: 1-D numpy.float64 array of frequencies for each channel for the\n\
   second two stands in signals\n\
 * delay: delay in seconds to apply\n\
\n\
Outputs:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) of the\n\
   phase-rotated spectra data\n\
");


static PyObject *ComputeSKMask(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF;
	PyArrayObject *data, *dataF;
	double lower, upper;
	long ij, i, j, k, nStand, nSamps, nChan, nFFT;
	
	if(!PyArg_ParseTuple(args, "Odd", &signals, &lower, &upper)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 3-D complex64");
		return NULL;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nChan  = (long) PyArray_DIM(data, 1);
	nFFT   = (long) PyArray_DIM(data, 2);
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStart;
	float tempV, tempV2, temp2V;
	float complex *a;
	float *b;
	a = (float complex *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV, tempV2, temp2V)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nChan; ij++) {
			i = ij / nChan;
			j = ij % nChan;
			
			secStart = nSamps*i + nFFT*j;
			
			tempV2 = 0.0;
			temp2V = 0.0;
			for(k=0; k<nFFT; k++) {
				tempV  = cabs2f(*(a + secStart + k));
				temp2V += tempV*tempV;
				tempV2 += tempV;
			}
			
			tempV  = nFFT*temp2V / (tempV2*tempV2) - 1.0;
			tempV *= (nFFT + 1.0)/(nFFT - 1.0);
			
			if( tempV < lower || tempV > upper ) {
				*(b + nChan*i + j) = 0.0;
			} else {
				*(b + nChan*i + j) = 1.0;
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(dataF);
	
	return signalsF;
}

PyDoc_STRVAR(ComputeSKMask_doc, \
"Given the output of PulsarEngineRaw, calculate channel weights based on\n\
spectral kurtosis\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data\n\
 * lower: lower spectral kurtosis limit\n\
 * upper: upper spectral kurtosis limit\n\
\n\
Outputs:\n\
 * weight: 2-D numpy.float32 (stands by channels) of data weights\n\
");


static PyObject *ComputePseudoSKMask(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF;
	PyArrayObject *data, *dataF;
	double lower, upper;
	long ij, i, j, k, nStand, nSamps, nChan, nFFT, skN;
	
	if(!PyArg_ParseTuple(args, "Olldd", &signals, &nChan, &skN, &lower, &upper)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_DOUBLE, 2, 2);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 3-D complex64");
		return NULL;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nSamps  = (long) PyArray_DIM(data, 1);
	nFFT = nSamps / nChan;
	
	// Find out how large the output array needs to be and initialize it
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStart;
	double tempV, temp2V, tempV2;
	double *a;
	float *b;
	a = (double *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV, temp2V, tempV2)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nChan; ij++) {
			i = ij / nChan;
			j = ij % nChan;
			
			secStart = nSamps*i;
			
			temp2V = 0.0;
			tempV2 = 0.0;
			for(k=0; k<nFFT; k++) {
				tempV  = *(a + secStart + nChan*k + j);

				temp2V += tempV*tempV;
				tempV2 += tempV;
			}
			
			tempV  = nFFT*temp2V / (tempV2*tempV2) - 1.0;
			tempV *= (nFFT*skN + 1.0)/(nFFT - 1.0);
			
			if( tempV < lower || tempV > upper ) {
				*(b + nChan*i + j) = 0.0;
			} else {
				*(b + nChan*i + j) = 1.0;
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(dataF);
	
	return signalsF;
}

PyDoc_STRVAR(ComputePseudoSKMask_doc, \
"Given the output of DR spectrometer, calculate channel weights based on\n\
psuedo-spectral kurtosis\n\
\n\
Input arguments are:\n\
 * signals: 2-D numpy.complex64 (stands by channels/integrations) array\n\
   of data to from DR spectrometer\n\
 * LFFT: FFT length\n\
 * N: number of FFT windows per integration\n\
 * lower: lower spectral kurtosis limit\n\
 * upper: upper spectral kurtosis limit\n\
\n\
Outputs:\n\
 * weight: 2-D numpy.float32 (stands by channels) of data weights\n\
");



/*
  getCoherentSampleSize - Estimate the number of samples needed to 
  successfully apply coherent dedispersion to a data stream.
*/

long getCoherentSampleSize(double centralFreq, double sampleRate, double DM) {
	double delayBand;
	long samples;
	
	delayBand = DM*DCONST * (pow(1e6/(centralFreq-sampleRate/2.0), 2) - pow(1e6/(centralFreq+sampleRate/2.0), 2));
	delayBand *= sampleRate;
	samples = (long) ceil( log(delayBand)/log(2.0) );
	if( samples < 0 ) {
		samples = 0;
	}
	samples = (long) 1<<samples;
	samples *= 2;
	
	return samples;
}


/*
 getFFTChannelFreq - Compute the frequency of a given FFT channel.
*/

double getFFTChannelFreq(long i, long LFFT, double centralFreq, double sampleRate) {
	long N;
	double val;
	
	N = (LFFT-1) / 2 + 1;
	val = sampleRate / (double) LFFT;
	if( i < N ) {
		return i * val + centralFreq;
	} else { 
		return (i - N - LFFT/2) * val + centralFreq;
	}
}


/*
 chirpFunction - Chirp function for coherent dedispersion for a given set of 
 frequencies (in Hz).  Based on Equation (6) of "Pulsar Observations II -- 
 Coherent Dedispersion, Polarimetry, and Timing" By Stairs, I. H.
*/

void chirpFunction(long LFFT, double centralFreq, double sampleRate, double DM, float complex *chirp) {
	int i;
	double freqMHz;
	double fMHz0, fMHz1;
	
	// Compute the frequencies in MHz and find the average frequency
	if( LFFT % 2 == 0 ) {
		fMHz0 = (centralFreq - sampleRate / (double) LFFT / 2.0) / 1e6;
	} else {
		fMHz0 = centralFreq / 1e6;
	}
// 	fMHz0 = 0.0;
// 	for(i=0; i<LFFT; i++) {
// 		fMHz0 += getFreq(i, LFFT, centralFreq, sampleRate) / 1e6;
// 	}
// 	fMHz0 /= (double) LFFT;
	
	// Compute the chirp
	for(i=0; i<LFFT; i++) {
		freqMHz = getFFTChannelFreq(i, LFFT, centralFreq, sampleRate) / 1e6;
		fMHz1 = freqMHz - fMHz0;
		*(chirp + i) = cexp(-2.0*NPY_PI*_Complex_I*DCONST*1e6 * DM*fMHz1*fMHz1 / (fMHz0*fMHz0* freqMHz));
	}
}


static PyObject *MultiChannelCD(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *drxData, *spectraFreq1, *spectraFreq2, *prevData, *nextData, *drxDataF=NULL;
	PyArrayObject *data=NULL, *freq1=NULL, *freq2=NULL;
	PyArrayObject *pData=NULL, *nData=NULL, *dataF=NULL;
	double sRate, DM;
	
	long i, j, k, l, nStand, nChan, nFFT;
	
	static char *kwlist[] = {"rawSpectra", "freq1", "freq2", "sampleRate", "DM", "prevRawSpectra", "nextRawSpectra", "outRawSpectra", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOddOO|O", kwlist, &drxData, &spectraFreq1, &spectraFreq2, &sRate, &DM, &prevData, &nextData, &drxDataF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		goto fail;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(drxData, NPY_COMPLEX64, 3, 3);
	freq1 = (PyArrayObject *) PyArray_ContiguousFromObject(spectraFreq1, NPY_DOUBLE, 1, 1);
	freq2 = (PyArrayObject *) PyArray_ContiguousFromObject(spectraFreq2, NPY_DOUBLE, 1, 1);
	pData = (PyArrayObject *) PyArray_ContiguousFromObject(prevData, NPY_COMPLEX64, 3, 3);
	nData = (PyArrayObject *) PyArray_ContiguousFromObject(nextData, NPY_COMPLEX64, 3, 3);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input drxData array to 3-D complex64");
		goto fail;
	}
	if( freq1 == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input spectraFreq1 to 1-D double");
		goto fail;
	}
	if( freq2 == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input spectraFreq2 to 1-D double");
		goto fail;
	}
	if( pData == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input prevData array to 3-D complex64");
		goto fail;
	}
	if( nData == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input nextData array to 3-D complex64");
		goto fail;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nChan  = (long) PyArray_DIM(data, 1);
	nFFT   = (long) PyArray_DIM(data, 2);
	
	// Validate
	if( PyArray_DIM(freq1, 0) != nChan ) {
		PyErr_Format(PyExc_ValueError, "freq1 array has different dimensions than rawSpectra");
		goto fail;
	}
	if( PyArray_DIM(freq2, 0) != nChan ) {
		PyErr_Format(PyExc_ValueError, "freq2 array has different dimensions than rawSpectra");
		goto fail;
	}
	
	if( PyArray_DIM(data, 0) != PyArray_DIM(pData, 0) ) {
		PyErr_Format(PyExc_ValueError, "prevRawSpectra array has different stand dimension than rawSpectra");
		goto fail;
	}
	if( PyArray_DIM(data, 1) != PyArray_DIM(pData, 1) ) {
		PyErr_Format(PyExc_ValueError, "prevRawSpectra array has different channel dimension than rawSpectra");
		goto fail;
	}
	if( PyArray_DIM(data, 2) != PyArray_DIM(pData, 2) ) {
		PyErr_Format(PyExc_ValueError, "prevRawSpectra array has different FFT count dimension than rawSpectra");
		goto fail;
	}
	if( PyArray_DIM(data, 0) != PyArray_DIM(nData, 0) ) {
		PyErr_Format(PyExc_ValueError, "nextRawSpectra array has different stand dimension than rawSpectra");
		goto fail;
	}
	if( PyArray_DIM(data, 1) != PyArray_DIM(nData, 1) ) {
		PyErr_Format(PyExc_ValueError, "nextRawSpectra array has different channel dimension than rawSpectra");
		goto fail;
	}
	if( PyArray_DIM(data, 2) != PyArray_DIM(nData, 2) ) {
		PyErr_Format(PyExc_ValueError, "nextRawSpectra array has different FFT count dimension than rawSpectra");
		goto fail;
	}
	
	// Find out how large the output arrays needs to be and initialize then
	npy_intp dims[3];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dims[2] = (npy_intp) nFFT;
	if( drxDataF != NULL ) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(drxDataF, NPY_COMPLEX64, 3, 3);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output outRawSpectra array to 3-D complex64");
			goto fail;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "outRawSpectra has an unexpected number of stands");
			goto fail;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "outRawSpectra has an unexpected number of channels");
			goto fail;
		}
		if(PyArray_DIM(dataF, 2) != dims[2]) {
			PyErr_Format(PyExc_RuntimeError, "outRawSpectra has an unexpected number of FFT windows");
			goto fail;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(3, dims, NPY_COMPLEX64, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output data array");
			goto fail;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Get access to the frequency information
	double *cf1, *cf2;
	cf1 = (double *) PyArray_DATA(freq1);
	cf2 = (double *) PyArray_DATA(freq2);
	
	// Create the FFTW plans
	long N;
	float complex *inP;
	fftwf_plan *plansF, *plansB;
	plansF = (fftwf_plan *) malloc(nStand/2*nChan*sizeof(fftwf_plan));
	plansB = (fftwf_plan *) malloc(nStand/2*nChan*sizeof(fftwf_plan));
	for(j=0; j<nChan; j++) {
		for(i=0; i<nStand; i+=2) {
			//Compute the number of FFT channels to use
			if( i/2 == 0 ) {
				N = getCoherentSampleSize(*(cf1 + j), sRate, DM);
			} else {
				N = getCoherentSampleSize(*(cf2 + j), sRate, DM);
			}
			
			inP = (float complex *) fftwf_malloc(N*sizeof(float complex));
			*(plansF + nChan*i/2 + j) = fftwf_plan_dft_1d(N, inP, inP, FFTW_FORWARD, FFTW_ESTIMATE);
			*(plansB + nChan*i/2 + j) = fftwf_plan_dft_1d(N, inP, inP, FFTW_BACKWARD, FFTW_ESTIMATE);
			fftwf_free(inP);
		}
	}
	
	// Go!
	long nSets, start, stop, secStartX, secStartY;
	double cFreq;
	float complex *chirp;
	float complex *d0, *d1, *d2, *dF;
	
	float complex *inX, *inY;
	
	d0 = (float complex *) PyArray_DATA(pData);
	d1 = (float complex *) PyArray_DATA(data);
	d2 = (float complex *) PyArray_DATA(nData);
	dF = (float complex *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStartX, secStartY, i, j, k, l, N, nSets, start, stop, cFreq, chirp, inX, inY)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i+=2) {
				// Section start offset
				secStartX = nFFT*nChan*i     + nFFT*j;
				secStartY = nFFT*nChan*(i+1) + nFFT*j;
				
				// Get the correct center frequency to use
				if( i/2 == 0 ) {
					cFreq = *(cf1 + j);
				} else {
					cFreq = *(cf2 + j);
				}
				
				//Compute the number of FFT channels to use
				N = getCoherentSampleSize(cFreq, sRate, DM);
				
				// Compute the number of windows we need to use for CD
				nSets = nFFT / N;
				
				// Compute the chirp function
				chirp = (float complex *) malloc(N*sizeof(float complex));
				chirpFunction(N, cFreq, sRate, DM, chirp);
				
				// Create the FFTW array
				inX = (float complex *) fftwf_malloc(N*sizeof(float complex));
				inY = (float complex *) fftwf_malloc(N*sizeof(float complex));
				
				// Loop over the sets
				for(l=0; l<2*nSets+1; l++) {
					start = l*N/2 - N/4;
					stop = start + N;
					
					// Load in the data
					if( start < 0 ) {
						// "Previous" buffering
						for(k=0; k<-start; k++) {
							inX[k] = *(d0 + secStartX + k + nFFT + start);
							inY[k] = *(d0 + secStartY + k + nFFT + start);
						}
						
						// Current data
						for(k=-start; k<N; k++) {
							inX[k] = *(d1 + secStartX + k + start);
							inY[k] = *(d1 + secStartY + k + start);
						}
						
					} else if( stop > nFFT ) {
						// Current data
						if( start < nFFT ) {
							for(k=0; k<nFFT-start; k++) {
								inX[k] = *(d1 + secStartX + k + start);
								inY[k] = *(d1 + secStartY + k + start);
							}
						}
						
						// "Next" buffering
						for(k=nFFT-start; k<N; k++) {
							inX[k] = *(d2 + secStartX + k - nFFT + start);
							inY[k] = *(d2 + secStartY + k - nFFT + start);
						}
						
					} else {
						// Current data
						for(k=0; k<N; k++) {
							inX[k] = *(d1 + secStartX + k + start);
							inY[k] = *(d1 + secStartY + k + start);
						}
						
					}
					
					// Forward FFT
					fftwf_execute_dft(*(plansF + nChan*i/2 + j), inX, inX);
					fftwf_execute_dft(*(plansF + nChan*i/2 + j), inY, inY);
					
					// Chirp
					for(k=0; k<N; k++) {
						inX[k] *= *(chirp + k) / N;
						inY[k] *= *(chirp + k) / N;
					}
					
					// Backward FFT
					fftwf_execute_dft(*(plansB + nChan*i/2 + j), inX, inX);
					fftwf_execute_dft(*(plansB + nChan*i/2 + j), inY, inY);
					
					// Save
					start = l*N/2;
					stop = start + N/2;
					if( stop > nFFT ) {
						stop = nFFT;
					}
					
					for(k=start; k<stop; k++) {
						*(dF + secStartX + k) = inX[k - start + N/4];
						*(dF + secStartY + k) = inY[k - start + N/4];
					}
					
					if( stop == nFFT ) {
						break;
					}
					
				}
				
				// Cleanup
				free(chirp);
				fftwf_free(inX);
				fftwf_free(inY);
			}
		}
	}
	
	// Cleanup
	for(j=0; j<nChan; j++) {
		for(i=0; i<nStand; i+=2) {
			fftwf_destroy_plan(*(plansF + nChan*i/2 + j));
			fftwf_destroy_plan(*(plansB + nChan*i/2 + j));
		}
	}
	free(plansF);
	free(plansB);
	
	Py_END_ALLOW_THREADS
	
	drxDataF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(freq1);
	Py_XDECREF(freq2);
	Py_XDECREF(pData);
	Py_XDECREF(nData);
	Py_XDECREF(dataF);
	
	return drxDataF;
	
fail:
	Py_XDECREF(data);
	Py_XDECREF(freq1);
	Py_XDECREF(freq2);
	Py_XDECREF(pData);
	Py_XDECREF(nData);
	Py_XDECREF(dataF);
	
	return NULL;
}

PyDoc_STRVAR(MultiChannelCD_doc, \
"Given the output of one of the 'PulsarEngine' functions and information about\n\
the time and frequencies, apply coherent dedispersion to the data and return a\n\
two-element tuple giving the time and dedispersed data.\n\
\n\
Input arguments are:\n\
  * rawSpectra - 3-D numpy.complex64 (stands by channels by samples) of raw\n\
    spectra data generated by 'PulsarEngineRaw' or 'PulsarEngineWindow'\n\
  * freq1 - 1-D numpy.float64 (channels) array of frequencies for the first\n\
    set of two stands in Hz\n\
  * freq2 - 1-D numpy.float64 (channels) array of frequency for the second\n\
    set of two stands in Hz\n\
  * sampleRate - Channelized data sample rate in Hz\n\
  * DM - dispersion measure in pc cm^-3 to dedisperse at\n\
  * prevRawSpectra - 3-D numpy.complex64 (stands by channels by samples) of\n\
    the previously input data\n\
  * prevRawSpectra - 3-D numpy.complex64 (stands by channels by samples) of\n\
    the following input data\n\
\n\
Outputs:\n\
  * dedispRawSpectra - 3-D numpy.complex64 (stands by channels by samples) of\n\
    the coherently dedispersed spectra\n\
\n\
.. note::\n\
\tThere are a few things that look a little strange here.  First, the\n\
\tsample rate specified is that of the output spectra.  This is related\n\
\tto the input data range and channel count via <input rate> / <channel\n\
\tcount>.  Second, the time/prevTime/nextTime, etc. variables might be\n\
\ta little confusing.  Consider the following time/data flow in a file:\n\
\t  t0/d0, t1/d1, t2/d2, t3/d3, ...\n\
\tFor the first call to MultiDisp() the arguments are:\n\
\t  * t1/d1 are time/rawSpectra\n\
\t  * t0/d0 are prevTime/prevRawSpectra\n\
\t  * t2/d2 are nextTime/nextRawSpectra\n\
\tand for the subsequent call to MultiDisp() they are:\n\
\t  * t2/d2 are time/rawSpectra\n\
\t  * t1/d1 are prevTime/prevRawSpectra\n\
\t  * t3/d3 are nextTime/nextRawSpectra\n\
");


static PyObject *CombineToIntensity(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF=NULL;
	PyArrayObject *data, *dataF;
	
	long ij, i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", "signalsF", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &signals, &signalsF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 3-D complex64");
		return NULL;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nChan  = (long) PyArray_DIM(data, 1);
	nFFT   = (long) PyArray_DIM(data, 2);
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) (nStand/2);
	dims[1] = (npy_intp) nSamps;
	if( signalsF != NULL ) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(signalsF, NPY_FLOAT32, 2, 2);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output signalsF array to 2-D float32");
			Py_XDECREF(data);
			return NULL;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of stands");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of samples");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			Py_XDECREF(data);
			return NULL;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStartX, secStartY;
	float complex *a;
	float *b;
	a = (float complex *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStartX, secStartY, i, j, k)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand/2*nChan; ij++) {
			i = ij / nChan * 2;
			j = ij % nChan;
			
			secStartX = nSamps*i + nFFT*j;
			secStartY = nSamps*(i+1) + nFFT*j;
			
			for(k=0; k<nFFT; k++) {
				// I
				*(b + nSamps*(i/2) + nChan*k + j) = cabs2f(*(a + secStartX + k)) + \
				                                    cabs2f(*(a + secStartY + k));
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(dataF);
	
	return signalsF;
}

PyDoc_STRVAR(CombineToIntensity_doc, \
"Given the output of PulsarEngineRaw, calculate total intensity spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");


static PyObject *CombineToLinear(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF=NULL;
	PyArrayObject *data, *dataF;
	
	long ij, i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", "signalsF", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &signals, &signalsF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 3-D complex64");
		return NULL;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nChan  = (long) PyArray_DIM(data, 1);
	nFFT   = (long) PyArray_DIM(data, 2);
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	if( signalsF != NULL ) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(signalsF, NPY_FLOAT32, 2, 2);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output signalsF array to 2-D float32");
			Py_XDECREF(data);
			return NULL;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of stands");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of samples");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			Py_XDECREF(data);
			return NULL;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStart;
	float complex *a;
	float *b;
	a = (float complex *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStart, i, j, k)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nChan; ij++) {
			i = ij / nChan;
			j = ij % nChan;
			
			secStart = nSamps*i + nFFT*j;
			
			for(k=0; k<nFFT; k++) {
				*(b + nSamps*i + nChan*k + j) = cabs2f(*(a + secStart + k));
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(dataF);
	
	return signalsF;
}

PyDoc_STRVAR(CombineToLinear_doc, \
"Given the output of PulsarEngineRaw, calculate linear polarization spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");


static PyObject *CombineToCircular(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF=NULL;
	PyArrayObject *data, *dataF;
	
	long ij, i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", "signalsF", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &signals, &signalsF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 3-D complex64");
		return NULL;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nChan  = (long) PyArray_DIM(data, 1);
	nFFT   = (long) PyArray_DIM(data, 2);
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	if( signalsF != NULL ) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(signalsF, NPY_FLOAT32, 2, 2);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output signalsF array to 2-D float32");
			Py_XDECREF(data);
			return NULL;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of stands");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of samples");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			Py_XDECREF(data);
			return NULL;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStartX, secStartY;
	float complex *a;
	float *b;
	a = (float complex *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStartX, secStartY, i, j, k)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand/2*nChan; ij++) {
			i = ij / nChan * 2;
			j = ij % nChan;
			
			secStartX = nSamps*i + nFFT*j;
			secStartY = nSamps*(i+1) + nFFT*j;
			
			for(k=0; k<nFFT; k++) {
				// LL
				*(b + nSamps*(i+0) + nChan*k + j) = cabs2f( \
				                                           *(a + secStartX + k) + *(a + secStartY + k)*_Complex_I \
				                                         ) / 2.0;
				
				// RR
				*(b + nSamps*(i+1) + nChan*k + j) =  cabs2f( \
				                                            *(a + secStartX + k) - *(a + secStartY + k)*_Complex_I \
				                                          ) / 2.0;
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(dataF);
	
	return signalsF;
}

PyDoc_STRVAR(CombineToCircular_doc, \
"Given the output of PulsarEngineRaw, calculate circular polarization spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");


static PyObject *CombineToStokes(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF=NULL;
	PyArrayObject *data, *dataF;
	
	long ij, i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", "signalsF", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &signals, &signalsF)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input signals array to 3-D complex64");
		return NULL;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nChan  = (long) PyArray_DIM(data, 1);
	nFFT   = (long) PyArray_DIM(data, 2);
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) (4*(nStand/2));
	dims[1] = (npy_intp) nSamps;
	if( signalsF != NULL ) {
		dataF = (PyArrayObject *) PyArray_ContiguousFromObject(signalsF, NPY_FLOAT32, 2, 2);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output signalsF array to 2-D float32");
			Py_XDECREF(data);
			return NULL;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of stands");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "signalsF has an unexpected number of samples");
			Py_XDECREF(data);
			Py_XDECREF(dataF);
			return NULL;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			Py_XDECREF(data);
			return NULL;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStartX, secStartY;
	float complex *a;
	float *b;
	a = (float complex *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStartX, secStartY, i, j, k)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand/2*nChan; ij++) {
			i = ij / nChan * 2;
			j = ij % nChan;
			
			secStartX = nSamps*i + nFFT*j;
			secStartY = nSamps*(i+1) + nFFT*j;
			
			for(k=0; k<nFFT; k++) {
				// I
				*(b + nSamps*(4*(i/2)+0) + nChan*k + j) = cabs2(*(a + secStartX + k)) + \
				                                          cabs2(*(a + secStartY + k));
				
				// Q
				*(b + nSamps*(4*(i/2)+1) + nChan*k + j)  = cabs2(*(a + secStartX + k)) - \
				                                           cabs2(*(a + secStartY + k));
				
				// U
				*(b + nSamps*(4*(i/2)+2) + nChan*k + j) = 2.0 * \
				                                          creal(*(a + secStartX + k) * \
				                                          conj(*(a + secStartY + k)));
				
				// V
				*(b + nSamps*(4*(i/2)+3) + nChan*k + j) = -2.0 * \
				                                          cimag(*(a + secStartX + k) * \
				                                          conj(*(a + secStartY + k)));
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(dataF);

	return signalsF;
}

PyDoc_STRVAR(CombineToStokes_doc, \
"Given the output of PulsarEngineRaw, calculate Stokes parameter spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");


static PyObject *OptimizeDataLevels8Bit(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *spectra, *bzero=NULL, *bscale=NULL, *bdata=NULL, *output;
	PyArrayObject *data=NULL, *zeroF=NULL, *scaleF=NULL, *dataF=NULL;
	long ij, i, j, k, nStand, nSamps, nFFT;
	int nChan = 64;
	
	static char *kwlist[] = {"spectra", "nChan", "bzero", "bscale", "bdata", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Oi|OOO", kwlist, &spectra, &nChan, &bzero, &bscale, &bdata)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		goto fail;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(spectra, NPY_FLOAT32, 2, 2);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input spectra array to 2-D float32");
		goto fail;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nSamps = (long) PyArray_DIM(data, 1);
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	if( bzero != NULL ) {
		zeroF = (PyArrayObject*) PyArray_ContiguousFromObject(bzero, NPY_FLOAT32, 2, 2);
		if(zeroF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output bzero array to 2-D float32");
			goto fail;
		}
		if(PyArray_DIM(zeroF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "bzero has an unexpected number of stands");
			goto fail;
		}
		if(PyArray_DIM(zeroF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "bzero has an unexpected number of channels");
			goto fail;
		}
	} else {
		zeroF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(zeroF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			goto fail;
		}
	}
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	if( bscale != NULL ) {
		scaleF = (PyArrayObject*) PyArray_ContiguousFromObject(bzero, NPY_FLOAT32, 2, 2);
		if(scaleF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output bscale array to 2-D float32");
			goto fail;
		}
		if(PyArray_DIM(scaleF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "bscale has an unexpected number of stands");
			goto fail;
		}
		if(PyArray_DIM(scaleF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "bscale has an unexpected number of channels");
			goto fail;
		}
	} else {
		scaleF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(scaleF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			goto fail;
		}
	}
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	if( bdata != NULL ) {
		dataF = (PyArrayObject*) PyArray_ContiguousFromObject(bdata, NPY_UINT8, 2, 2);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output bdata array to 2-D uint8");
			goto fail;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "bdata has an unexpected number of stands");
			goto fail;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "bdata has an unexpected number of channels");
			goto fail;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_UINT8, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			goto fail;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStart;
	float tempMin, tempMax, tempV;
	float *a, *b, *c;
	unsigned char *d;
	a = (float *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(zeroF);
	c = (float *) PyArray_DATA(scaleF);
	d = (unsigned char *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV, tempMin, tempMax)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nChan; ij++) {
			i = ij / nChan;
			j = ij % nChan;
			
			secStart = nSamps*i + j;
			
			k = 0;
			tempMin = *(a + secStart + nChan*k);
			tempMax = *(a + secStart + nChan*k);
			
			for(k=1; k<nFFT; k++) {
				if( *(a + secStart + nChan*k) < tempMin ) {
					tempMin = *(a + secStart + nChan*k);
				} else if( *(a + secStart + nChan*k) > tempMax ) {
					tempMax = *(a + secStart + nChan*k);
				}
			}
			
			*(b + i*nChan + j) = tempMin;
			*(c + i*nChan + j) = (tempMax - tempMin) / 255.0;
			
			for(k=0; k<nFFT; k++) {
				tempV  = *(a + secStart + nChan*k) - tempMin;
				tempV /= *(c + i*nChan + j);
				tempV  = round(tempV);
				
				*(d + secStart + nChan*k) = (unsigned char) tempV;
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	output = Py_BuildValue("(OOO)", PyArray_Return(zeroF), PyArray_Return(scaleF), PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(zeroF);
	Py_XDECREF(scaleF);
	Py_XDECREF(dataF);
	
	return output;
	
fail:
	Py_XDECREF(data);
	Py_XDECREF(zeroF);
	Py_XDECREF(scaleF);
	Py_XDECREF(dataF);
	
	return NULL;
}

PyDoc_STRVAR(OptimizeDataLevels8Bit_doc, \
"Given the output of one of the 'Combine' functions, find the bzero and bscale\n\
values that yield the best representation of the data as unsigned characters\n\
and scale the data accordingly.\n\
\n\
Input arguments are:\n\
 * signals: 2-D numpy.float64 (stands by samples) array of combined data\n\
 * LFFT: number of channels per data stream\n\
\n\
Outputs:\n\
 * bzero: 2-D numpy.float32 (stands by channels) array of bzero values\n\
 * bscale: 2-D numpy.float32 (stand by channels) array of bscale values\n\
 * spectra: 2-D numpy.unit8 (stands by samples) array of spectra\n\
");


static PyObject *OptimizeDataLevels4Bit(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *spectra, *bzero=NULL, *bscale=NULL, *bdata=NULL, *output;
	PyArrayObject *data=NULL, *zeroF=NULL, *scaleF=NULL, *dataF=NULL;
	long ij, i, j, k, nStand, nSamps, nFFT;
	int nChan = 64;
	
	static char *kwlist[] = {"spectra", "nChan", "bzero", "bscale", "bdata", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Oi|OOO", kwlist, &spectra, &nChan, &bzero, &bscale, &bdata)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		goto fail;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(spectra, NPY_FLOAT32, 2, 2);
	if( data == NULL ) {
		PyErr_Format(PyExc_RuntimeError, "Cannot cast input spectra array to 2-D float32");
		goto fail;
	}
	
	// Get the properties of the data
	nStand = (long) PyArray_DIM(data, 0);
	nSamps = (long) PyArray_DIM(data, 1);
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	if( bzero != NULL ) {
		zeroF = (PyArrayObject*) PyArray_ContiguousFromObject(bzero, NPY_FLOAT32, 2, 2);
		if(zeroF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output bzero array to 2-D float32");
			goto fail;
		}
		if(PyArray_DIM(zeroF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "bzero has an unexpected number of stands");
			goto fail;
		}
		if(PyArray_DIM(zeroF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "bzero has an unexpected number of channels");
			goto fail;
		}
	} else {
		zeroF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(zeroF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			goto fail;
		}
	}
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	if( bscale != NULL ) {
		scaleF = (PyArrayObject*) PyArray_ContiguousFromObject(bzero, NPY_FLOAT32, 2, 2);
		if(scaleF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output bscale array to 2-D float32");
			goto fail;
		}
		if(PyArray_DIM(scaleF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "bscale has an unexpected number of stands");
			goto fail;
		}
		if(PyArray_DIM(scaleF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "bscale has an unexpected number of channels");
			goto fail;
		}
	} else {
		scaleF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
		if(scaleF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			goto fail;
		}
	}
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	if( bdata != NULL ) {
		dataF = (PyArrayObject*) PyArray_ContiguousFromObject(bdata, NPY_UINT8, 2, 2);
		if(dataF == NULL) {
			PyErr_Format(PyExc_RuntimeError, "Cannot cast output bdata array to 2-D uint8");
			goto fail;
		}
		if(PyArray_DIM(dataF, 0) != dims[0]) {
			PyErr_Format(PyExc_RuntimeError, "bdata has an unexpected number of stands");
			goto fail;
		}
		if(PyArray_DIM(dataF, 1) != dims[1]) {
			PyErr_Format(PyExc_RuntimeError, "bdata has an unexpected number of channels");
			goto fail;
		}
	} else {
		dataF = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_UINT8, 0);
		if(dataF == NULL) {
			PyErr_Format(PyExc_MemoryError, "Cannot create output array");
			goto fail;
		}
	}
	
	Py_BEGIN_ALLOW_THREADS
	
	// Go!
	long secStart;
	float tempMin, tempMax, tempV;
	float *a, *b, *c;
	unsigned char *d;
	a = (float *) PyArray_DATA(data);
	b = (float *) PyArray_DATA(zeroF);
	c = (float *) PyArray_DATA(scaleF);
	d = (unsigned char *) PyArray_DATA(dataF);
	
	#ifdef _OPENMP
		omp_set_dynamic(0);
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV, tempMin, tempMax)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(OMP_SCHEDULER)
		#endif
		for(ij=0; ij<nStand*nChan; ij++) {
			i = ij / nChan;
			j = ij % nChan;
			
			secStart = nSamps*i + j;
			
			k = 0;
			tempMin = *(a + secStart + nChan*k);
			tempMax = *(a + secStart + nChan*k);
			
			for(k=1; k<nFFT; k++) {
				if( *(a + secStart + nChan*k) < tempMin ) {
					tempMin = *(a + secStart + nChan*k);
				} else if( *(a + secStart + nChan*k) > tempMax ) {
					tempMax = *(a + secStart + nChan*k);
				}
			}
			
			*(b + i*nChan + j) = tempMin;
			*(c + i*nChan + j) = (tempMax - tempMin) / 15.0;
			
			for(k=0; k<nFFT; k++) {
				tempV  = *(a + secStart + nChan*k) - tempMin;
				tempV /= *(c + i*nChan + j);
				tempV  = round(tempV);
				
				*(d + secStart + nChan*k) = ((unsigned char) tempV ) & 0xF;
			}
		}
	}
	
	Py_END_ALLOW_THREADS
	
	output = Py_BuildValue("(OOO)", PyArray_Return(zeroF), PyArray_Return(scaleF), PyArray_Return(dataF));
	
	Py_XDECREF(data);
	Py_XDECREF(zeroF);
	Py_XDECREF(scaleF);
	Py_XDECREF(dataF);
	
	return output;
	
fail:
	Py_XDECREF(data);
	Py_XDECREF(zeroF);
	Py_XDECREF(scaleF);
	Py_XDECREF(dataF);
	
	return NULL;
}

PyDoc_STRVAR(OptimizeDataLevels4Bit_doc, \
"Given the output of one of the 'Combine' functions, find the bzero and bscale\n\
values that yield the best representation of the data as 4-bit unsigned integers\n\
and scale the data accordingly.\n\
\n\
Input arguments are:\n\
 * signals: 2-D numpy.float64 (stands by samples) array of combined data\n\
 * LFFT: number of channels per data stream\n\
\n\
Outputs:\n\
 * bzero: 2-D numpy.float32 (stands by channels) array of bzero values\n\
 * bscale: 2-D numpy.float32 (stand by channels) array of bscale values\n\
 * spectra: 2-D numpy.unit8 (stands by samples) array of spectra\n\
");


/*
  Module Setup - Function Definitions and Documentation
*/

static PyMethodDef SpecMethods[] = {
	{"BindToCore",             (PyCFunction) BindToCore,             METH_VARARGS,               BindToCore_doc             },
	{"BindOpenMPToCores",      (PyCFunction) BindOpenMPToCores,      METH_VARARGS,               BindOpenMPToCores_doc      }, 
	{"PulsarEngineRaw",        (PyCFunction) PulsarEngineRaw,        METH_VARARGS|METH_KEYWORDS, PulsarEngineRaw_doc        },
	{"PulsarEngineRawWindow",  (PyCFunction) PulsarEngineRawWindow,  METH_VARARGS|METH_KEYWORDS, PulsarEngineRawWindow_doc  },
	{"PhaseRotator",           (PyCFunction) PhaseRotator,           METH_VARARGS|METH_KEYWORDS, PhaseRotator_doc           },
	{"ComputeSKMask",          (PyCFunction) ComputeSKMask,          METH_VARARGS,               ComputeSKMask_doc          },
	{"ComputePseudoSKMask",    (PyCFunction) ComputePseudoSKMask,    METH_VARARGS,               ComputePseudoSKMask_doc    },
	{"MultiChannelCD",         (PyCFunction) MultiChannelCD,         METH_VARARGS|METH_KEYWORDS, MultiChannelCD_doc         },
	{"CombineToIntensity",     (PyCFunction) CombineToIntensity,     METH_VARARGS|METH_KEYWORDS, CombineToIntensity_doc     }, 
	{"CombineToLinear",        (PyCFunction) CombineToLinear,        METH_VARARGS|METH_KEYWORDS, CombineToLinear_doc        }, 
	{"CombineToCircular",      (PyCFunction) CombineToCircular,      METH_VARARGS|METH_KEYWORDS, CombineToCircular_doc      }, 
	{"CombineToStokes",        (PyCFunction) CombineToStokes,        METH_VARARGS|METH_KEYWORDS, CombineToStokes_doc        },
	{"OptimizeDataLevels8Bit", (PyCFunction) OptimizeDataLevels8Bit, METH_VARARGS|METH_KEYWORDS, OptimizeDataLevels8Bit_doc },
	{"OptimizeDataLevels4Bit", (PyCFunction) OptimizeDataLevels4Bit, METH_VARARGS|METH_KEYWORDS, OptimizeDataLevels4Bit_doc },
	{NULL,                     NULL,                                 0,                          NULL                       }
};

PyDoc_STRVAR(spec_doc, \
"Extension to take timeseries data and convert it to the frequency domain.\n\
\n\
The functions defined in this module are:\n\
  * PulsarEngineRaw - FFT function for computing a series of Fourier \n\
    transforms for a complex-valued (TBN and DRX) signal from a collection\n\
    of stands/beams all at once.\n\
  * PulsarEngineRawWindow - Similar to PulsarEngineRaw but also requires\n\
    a numpy.float64 array for a window to apply to the data\n\
  * PhaseRotator - Given the output of PulsarEngineRaw, apply a sub-sample\n\
    delay as a phase rotation.\n\
  * ComputeSKMask - Given the output of PulsarEngineRaw compute a mask for\n\
    using spectral kurtosis\n\
  * ComputePseudoSKMask - Similar to ComputeSKMask but for DR spectrometer data\n\
  * MultiChannelCD - Given the output of PulsarEngineRaw apply coherent \n\
    dedispersion to the data\n\
  * CombineToIntensity - Given the output of PulsarEngineRaw compute the total\n\
    intensity for both tunings\n\
  * CombineToLinear - Given the output of PulsarEngineRaw compute XX and YY\n\
    for both tunings\n\
  * CombineToCircular- Given the output of PulsarEngineRaw compute LL and RR\n\
    for both tunings\n\
  * CombineToStokes- Given the output of PulsarEngineRaw compute I, Q, U,\n\
    and V for both tunings\n\
  * OptimizeDataLevels - Given the output of the CombineTo* functions, find\n\
    optimal BZERO and BSCALE values for representing the data as unsigned bytes\n\
    (numpy.uint8)\n\
\n\
See the inidividual functions for more details.");


/*
  Module Setup - Initialization
*/

PyMODINIT_FUNC init_psr(void) {
	char filename[256];
	PyObject *m, *pModule, *pDataPath;

	// Module definitions and functions
	m = Py_InitModule3("_psr", SpecMethods, spec_doc);
	import_array();
	
	// Version and revision information
	PyModule_AddObject(m, "__version__", PyString_FromString("0.6"));
	PyModule_AddObject(m, "__revision__", PyString_FromString("$Rev$"));
	
	// LSL FFTW Wisdom
	pModule = PyImport_ImportModule("lsl.common.paths");
	if( pModule != NULL ) {
		pDataPath = PyObject_GetAttrString(pModule, "data");
		sprintf(filename, "%s/fftw_wisdom.txt", PyString_AsString(pDataPath));
		read_wisdom(filename, m);
	} else {
		PyErr_Warn(PyExc_RuntimeWarning, "Cannot load the LSL FFTWF wisdom");
	}
}
