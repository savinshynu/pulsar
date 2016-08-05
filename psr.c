#include "Python.h"
#include <math.h>
#include <stdio.h>
#ifdef _MKL
	#include "fftw3.h"
#else
	#include <fftw3.h>
#endif
#include <stdlib.h>
#include <complex.h>

#ifdef _OPENMP
	#include <omp.h>
	#ifdef _MKL
		#include "fftw3_mkl.h"
	#endif
#endif

#include "numpy/arrayobject.h"

#define PI 3.1415926535898
#define imaginary _Complex_I

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


static PyObject *PulsarEngineRaw(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *signals, *signalsF;
	PyArrayObject *data, *dataF;
	int nChan = 64;
	
	long i, j, k, nStand, nSamps, nFFT;
	
	static char *kwlist[] = {"signals", "LFFT", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &signals, &nChan)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 2, 2);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nSamps = (long) data->dimensions[1];
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[3];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dims[2] = (npy_intp) (nSamps/nChan);
	dataF = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_COMPLEX64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Create the FFTW plan
	fftwf_complex *inP, *in;
	inP = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nChan);
	fftwf_plan p;
	p = fftwf_plan_dft_1d(nChan, inP, inP, FFTW_FORWARD, FFTW_ESTIMATE);
	
	// FFT
	long secStart;
	float complex *a;
	float complex *b;
	a = (float complex *) data->data;
	b = (float complex *) dataF->data;
	
	#ifdef _OPENMP
		#ifdef _MKL
			fftw3_mkl.number_of_user_threads = omp_get_num_threads();
		#endif
		#pragma omp parallel default(shared) private(in, secStart, i, j, k)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nFFT; j++) {
			in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nChan);
			
			for(i=0; i<nStand; i++) {
				secStart = nSamps * i + nChan*j;
				
				for(k=0; k<nChan; k++) {
					in[k][0] = creal(*(a + secStart + k));
					in[k][1] = cimag(*(a + secStart + k));
				}
				
				fftwf_execute_dft(p, in, in);
				
				for(k=0; k<nChan; k++) {
					if( k < nChan/2 ) {
						*(b + nFFT*nChan*i + nFFT*(k + nChan/2) + j) = (in[k][0] + in[k][1]*imaginary) / sqrt(nChan);
					} else {
						*(b + nFFT*nChan*i + nFFT*(k - nChan/2) + j) = (in[k][0] + in[k][1]*imaginary) / sqrt(nChan);
					}
				}
			}
			fftwf_free(in);
		}
	}
	fftwf_destroy_plan(p);
	fftwf_free(inP);
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
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
	PyObject *signals, *window, *signalsF;
	PyArrayObject *data, *win, *dataF;
	int nChan = 64;
	
	long i, j, k, nStand, nSamps, nFFT;
	
	static char *kwlist[] = {"signals", "window", "LFFT", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist, &signals, &window, &nChan)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 2, 2);
	win  = (PyArrayObject *) PyArray_ContiguousFromObject(window,  NPY_FLOAT64,   1, 1);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nSamps = (long) data->dimensions[1];
	
	if( win->dimensions[0] != nChan ) {
		PyErr_Format(PyExc_RuntimeError, "Window length does not match requested FFT length");
		Py_XDECREF(data);
		Py_XDECREF(win);
		return NULL;
	}
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[3];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dims[2] = (npy_intp) (nSamps/nChan);
	dataF = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_COMPLEX64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		Py_XDECREF(win);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Create the FFTW plan
	fftwf_complex *inP, *in;
	inP = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nChan);
	fftwf_plan p;
	p = fftwf_plan_dft_1d(nChan, inP, inP, FFTW_FORWARD, FFTW_ESTIMATE);
	
	// FFT
	long secStart;
	float complex *a;
	float complex *b;
	double *c;
	a = (float complex *) data->data;
	b = (float complex *) dataF->data;
	c = (double *) win->data;
	
	#ifdef _OPENMP
		#ifdef _MKL
			fftw3_mkl.number_of_user_threads = omp_get_num_threads();
		#endif
		#pragma omp parallel default(shared) private(in, secStart, i, j, k)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nFFT; j++) {
			in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nChan);
			
			for(i=0; i<nStand; i++) {
				secStart = nSamps * i + nChan*j;
				
				for(k=0; k<nChan; k++) {
					in[k][0] = *(c + k) * creal(*(a + secStart + k));
					in[k][1] = *(c + k) * cimag(*(a + secStart + k));
				}
				
				fftwf_execute_dft(p, in, in);
				
				for(k=0; k<nChan; k++) {
					if( k < nChan/2 ) {
						*(b + nFFT*nChan*i + nFFT*(k + nChan/2) + j) = (in[k][0] + in[k][1]*imaginary) / sqrt(nChan);
					} else {
						*(b + nFFT*nChan*i + nFFT*(k - nChan/2) + j) = (in[k][0] + in[k][1]*imaginary) / sqrt(nChan);
					}
				}
			}
			fftwf_free(in);
		}
	}
	fftwf_destroy_plan(p);
	fftwf_free(inP);
	
	Py_XDECREF(data);
	Py_XDECREF(win);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	Py_XDECREF(dataF);

	return signalsF;
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
	PyObject *signals, *f1, *f2, *signalsF;
	PyArrayObject *data, *freq1, *freq2, *dataF;
	double delay;	

	long i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", "freq1", "freq2", "delay", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOd", kwlist, &signals, &f1, &f2, &delay)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	freq1 = (PyArrayObject *) PyArray_ContiguousFromObject(f1, NPY_FLOAT64, 1, 1);
	freq2 = (PyArrayObject *) PyArray_ContiguousFromObject(f2, NPY_FLOAT64, 1, 1);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nChan  = (long) data->dimensions[1];
	nFFT   = (long) data->dimensions[2];
	
	// Validate
	if( freq1->dimensions[0] != nChan ) {
		PyErr_Format(PyExc_ValueError, "Frequency array 1 has different dimensions than rawSpectra");
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		return NULL;
	}
	if( freq2->dimensions[0] != nChan ) {
		PyErr_Format(PyExc_ValueError, "Frequency array 2 has different dimensions than rawSpectra");
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		return NULL;
	}	
	
	// Find out how large the output array needs to be and initialize it
	npy_intp dims[3];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dims[2] = (npy_intp) nFFT;
	dataF = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_COMPLEX64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStart;
	double tempF;
	float complex *a, *d;
	double *b, *c;
	a = (float complex *) data->data;
	b = (double *) freq1->data;
	c = (double *) freq2->data;
	d = (float complex *) dataF->data;
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempF)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i+=1) {
				secStart = nSamps*i + nFFT*j;
				if( i/2 == 0 ) {
					tempF = *(b + j);
				} else {
					tempF = *(c + j);
				}
				
				for(k=0; k<nFFT; k++) {
					*(d + secStart + k) = *(a + secStart + k) * cexp(2*imaginary*PI*tempF*delay);
				}
			}
		}
	}
	
	Py_XDECREF(data);
	Py_XDECREF(freq1);
	Py_XDECREF(freq2);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	Py_XDECREF(dataF);

	return signalsF;
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
	long i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", "lower", "upper", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Odd", kwlist, &signals, &lower, &upper)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nChan  = (long) data->dimensions[1];
	nFFT   = (long) data->dimensions[2];
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStart;
	double tempV, tempV2, temp2V;
	float complex *a;
	float *b;
	a = (float complex *) data->data;
	b = (float *) dataF->data;
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV, tempV2, temp2V)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=1; j<nChan-1; j++) {
			for(i=0; i<nStand; i++) {
				secStart = nSamps*i + nFFT*j;
				
				tempV2 = 0.0;
				temp2V = 0.0;
				for(k=0; k<nFFT; k++) {
					tempV  = creal(*(a + secStart + k))*creal(*(a + secStart + k));
					tempV += cimag(*(a + secStart + k))*cimag(*(a + secStart + k));

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
	}
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
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
	long i, j, k, nStand, nSamps, nChan, nFFT, skN;
	
	static char *kwlist[] = {"signals", "LFFT", "N", "lower", "upper", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Olldd", kwlist, &signals, &nChan, &skN, &lower, &upper)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_FLOAT64, 2, 2);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nSamps  = (long) data->dimensions[1];
	nFFT = nSamps / nChan;
	
	// Find out how large the output array needs to be and initialize it
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStart;
	double tempV, temp2V, tempV2;
	double *a;
	float *b;
	a = (double *) data->data;
	b = (float *) dataF->data;
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV, temp2V, tempV2)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=1; j<nChan-1; j++) {
			for(i=0; i<nStand; i++) {
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
	}
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
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
 getFFTFreqs - Compute the frequencies for a collection of FFT channels.  Based on
 the numpy.fft.fftfreqs function.
*/

void getFFTFreqs(long LFFT, double centralFreq, double sampleRate, double *freq) {
	long i, N;
	double val;
	
	N = (LFFT-1) / 2 + 1;
	val = sampleRate / (double) LFFT;
	
	for(i=0; i<N; i++) {
		// Relative frequency
		*(freq + i) = i * val;
		
		// Add in the center
		*(freq + i) += centralFreq;
	}
	
	for(i=-(LFFT/2); i<0; i++) {
		// Relative frequency
		*(freq + i + (LFFT/2) + N) = i*val;
		
		// Add in the center
		*(freq + i + (LFFT/2) + N) += centralFreq;
	}
}


/*
 chirpFunction - Chirp function for coherent dedispersion for a given set of 
 frequencies (in Hz).  Based on Equation (6) of "Pulsar Observations II -- 
 Coherent Dedispersion, Polarimetry, and Timing" By Stairs, I. H.
*/

void chirpFunction(long LFFT, double *freq, double DM, float complex *chirp) {
	int i;
	double *freqMHz;
	double fMHz0, fMHz1;
	
	// Allocate the space for freqMHz, and fMHz1
	freqMHz = (double *) malloc(LFFT*sizeof(double));
	
	// Compute the frequencies in MHz and find the average frequency
	fMHz0 = 0.0;
	for(i=0; i<LFFT; i++) {
		*(freqMHz + i) = *(freq + i) / 1e6;
		fMHz0 += *(freqMHz+ i);
	}
	fMHz0 /= (double) LFFT;
	
	// Compute the chirp
	for(i=0; i<LFFT; i++) {
		fMHz1 = *(freqMHz + i) - fMHz0;
		*(chirp + i) = cexp(-2.0*imaginary*PI*DCONST*1e6 * DM*fMHz1*fMHz1 / (fMHz0*fMHz0* *(freqMHz + i)));
	}
	
	// Cleanup
	free(freqMHz);
}


static PyObject *MultiChannelCD(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *drxTime, *drxData, *spectraFreq1, *spectraFreq2, *prevTime, *prevData, *nextTime, *nextData, *drxDataF;
	PyArrayObject *time, *data, *freq1, *freq2, *pTime, *pData, *nTime, *nData, *timeF, *dataF;
	double sRate, DM;

	long i, j, k, l, nStand, nChan, nFFT;
	
	static char *kwlist[] = {"time", "rawSpectra", "freq1", "freq2", "sampleRate", "DM", "prevTime", "prevRawSpectra", "nextTime", "nextRawSpectra", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOddOOOO", kwlist, &drxTime, &drxData, &spectraFreq1, &spectraFreq2, &sRate, &DM, &prevTime, &prevData, &nextTime, &nextData)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	time = (PyArrayObject *) PyArray_ContiguousFromObject(drxTime, NPY_FLOAT64, 2, 2);
	data = (PyArrayObject *) PyArray_ContiguousFromObject(drxData, NPY_COMPLEX64, 3, 3);
	freq1 = (PyArrayObject *) PyArray_ContiguousFromObject(spectraFreq1, NPY_FLOAT64, 1, 1);
	freq2 = (PyArrayObject *) PyArray_ContiguousFromObject(spectraFreq2, NPY_FLOAT64, 1, 1);
	pTime = (PyArrayObject *) PyArray_ContiguousFromObject(prevTime, NPY_FLOAT64, 2, 2);
	pData = (PyArrayObject *) PyArray_ContiguousFromObject(prevData, NPY_COMPLEX64, 3, 3);
	nTime = (PyArrayObject *) PyArray_ContiguousFromObject(nextTime, NPY_FLOAT64, 2, 2);
	nData = (PyArrayObject *) PyArray_ContiguousFromObject(nextData, NPY_COMPLEX64, 3, 3);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nChan  = (long) data->dimensions[1];
	nFFT   = (long) data->dimensions[2];
	
	// Validate
	if( time->dimensions[0] != nStand ) {
		PyErr_Format(PyExc_ValueError, "time array has different stand dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( time->dimensions[1] != nFFT ) {
		PyErr_Format(PyExc_ValueError, "time array has different FFT count dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	
	if( freq1->dimensions[0] != nChan ) {
		PyErr_Format(PyExc_ValueError, "freq1 array has different dimensions than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( freq2->dimensions[0] != nChan ) {
		PyErr_Format(PyExc_ValueError, "freq2 array has different dimensions than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	
	if( time->dimensions[0] != pTime->dimensions[0] ) {
		PyErr_Format(PyExc_ValueError, "prevTime array has different stand dimension than time");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( time->dimensions[1] != pTime->dimensions[1] ) {
		PyErr_Format(PyExc_ValueError, "prevTime array has different FFT count dimension than time");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( time->dimensions[0] != nTime->dimensions[0] ) {
		PyErr_Format(PyExc_ValueError, "nextTime array has different stand dimension than time");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( time->dimensions[1] != nTime->dimensions[1] ) {
		PyErr_Format(PyExc_ValueError, "nextTime array has different FFT count dimension than time");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	
	if( data->dimensions[0] != pData->dimensions[0] ) {
		PyErr_Format(PyExc_ValueError, "prevRawSpectra array has different stand dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( data->dimensions[1] != pData->dimensions[1] ) {
		PyErr_Format(PyExc_ValueError, "prevRawSpectra array has different channel dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( data->dimensions[2] != pData->dimensions[2] ) {
		PyErr_Format(PyExc_ValueError, "prevRawSpectra array has different FFT count dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( data->dimensions[0] != nData->dimensions[0] ) {
		PyErr_Format(PyExc_ValueError, "nextRawSpectra array has different stand dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( data->dimensions[1] != nData->dimensions[1] ) {
		PyErr_Format(PyExc_ValueError, "nextRawSpectra array has different channel dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	if( data->dimensions[2] != nData->dimensions[2] ) {
		PyErr_Format(PyExc_ValueError, "nextRawSpectra array has different FFT count dimension than rawSpectra");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	
	// Find out how large the output arrays needs to be and initialize then
	npy_intp dimsT[2];
	dimsT[0] = (npy_intp) nStand;
	dimsT[1] = (npy_intp) nFFT;
	timeF = (PyArrayObject *) PyArray_SimpleNew(2, dimsT, NPY_FLOAT64);
	if(timeF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output time array");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		return NULL;
	}
	PyArray_FILLWBYTE(timeF, 0);
	
	npy_intp dimsD[3];
	dimsD[0] = (npy_intp) nStand;
	dimsD[1] = (npy_intp) nChan;
	dimsD[2] = (npy_intp) nFFT;
	dataF = (PyArrayObject*) PyArray_SimpleNew(3, dimsD, NPY_COMPLEX64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output data array");
		Py_XDECREF(time);
		Py_XDECREF(data);
		Py_XDECREF(freq1);
		Py_XDECREF(freq2);
		Py_XDECREF(pTime);
		Py_XDECREF(pData);
		Py_XDECREF(nTime);
		Py_XDECREF(nData);
		Py_XDECREF(timeF);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Get access to the frequency information
	double *cf1, *cf2;
	cf1 = (double *) freq1->data;
	cf2 = (double *) freq2->data;
	
	// Create the FFTW plans
	long N;
	fftwf_complex *inP;
	fftwf_plan *plansF, *plansB;
	plansF = (fftwf_plan *) malloc(nStand*nChan*sizeof(fftwf_plan));
	plansB = (fftwf_plan *) malloc(nStand*nChan*sizeof(fftwf_plan));
	for(j=0; j<nChan; j++) {
		for(i=0; i<nStand; i++) {
			//Compute the number of FFT channels to use
			if( i/2 == 0 ) {
				N = getCoherentSampleSize(*(cf1 + j), sRate, DM);
			} else {
				N = getCoherentSampleSize(*(cf2 + j), sRate, DM);
			}
			
			inP = (fftwf_complex *) fftwf_malloc(N*sizeof(fftwf_complex));
			*(plansF + nChan*i + j) = fftwf_plan_dft_1d(N, inP, inP, FFTW_FORWARD, FFTW_ESTIMATE);
			*(plansB + nChan*i + j) = fftwf_plan_dft_1d(N, inP, inP, FFTW_BACKWARD, FFTW_ESTIMATE);
			fftwf_free(inP);
		}
	}
	
	// Go!
	long nSets, start, stop, secStart;
	double cFreq, *rf, *inT;
	float complex tempF;
	float complex *chirp;
	double *t0, *t1, *t2, *tF;
	float complex *d0, *d1, *d2, *dF;
	
	fftwf_complex *in;
	
	t0 = (double *) pTime->data;
	d0 = (float complex *) pData->data;
	t1 = (double *) time->data;
	d1 = (float complex *) data->data;
	t2 = (double *) nTime->data;
	d2 = (float complex *) nData->data;
	tF = (double *) timeF->data;
	dF = (float complex *) dataF->data;
	
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStart, i, j, k, l, N, nSets, start, stop, cFreq, rf, chirp, in, inT, tempF)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i++) {
				// Section start offset
				secStart = nFFT*nChan*i + nFFT*j;
				
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
				
				// Compute the relative frequencies for this channel
				rf = (double *) malloc(N*sizeof(double));
				getFFTFreqs(N, cFreq, sRate, rf);
				
				// Compute the chirp function
				chirp = (float complex *) malloc(N*sizeof(float complex));
				chirpFunction(N, rf, DM, chirp);
				
				// Create the time array
				inT = (double *) malloc(N*sizeof(double));
				
				// Create the FFTW array
				in = (fftwf_complex *) fftwf_malloc(N*sizeof(fftwf_complex));
				
				// Loop over the sets
				for(l=0; l<2*nSets+1; l++) {
					start = l*N/2 - N/4;
					stop = start + N;
					
					// Load in the data
					if( start < 0 ) {
						// "Previous" buffering
						for(k=0; k<-start; k++) {
							*(inT + k) = *(t0 + nFFT*i + k + nFFT + start);
							in[k][0] = creal(*(d0 + secStart + k + nFFT + start));
							in[k][1] = cimag(*(d0 + secStart + k + nFFT + start));
						}
						
						// Current data
						for(k=-start; k<N; k++) {
							*(inT + k) = *(t1 + nFFT*i + k + start);
							in[k][0] = creal(*(d1 + secStart + k + start));
							in[k][1] = cimag(*(d1 + secStart + k + start));
						}
						
					} else if( stop > nFFT ) {
						// Current data
						if( start < nFFT ) {
							for(k=0; k<nFFT-start; k++) {
								*(inT + k) = *(t1 + nFFT*i + k + start);
								in[k][0] = creal(*(d1 + secStart + k + start));
								in[k][1] = cimag(*(d1 + secStart + k + start));
							}
						}
						
						// "Next" buffering
						for(k=nFFT-start; k<N; k++) {
							*(inT + k) = *(t2 + nFFT*i + k - nFFT + start);
							in[k][0] = creal(*(d2 + secStart + k - nFFT + start));
							in[k][1] = cimag(*(d2 + secStart + k - nFFT + start));
						}
						
					} else {
						// Current data
						for(k=0; k<N; k++) {
							*(inT + k) = *(t1 + nFFT*i + k + start);
							in[k][0] = creal(*(d1 + secStart + k + start));
							in[k][1] = cimag(*(d1 + secStart + k + start));
						}
						
					}
					
					// Forward FFT
					fftwf_execute_dft(*(plansF + nChan*i + j), in, in);
					
					// Chirp
					for(k=0; k<N; k++) {
						tempF = in[k][0] + imaginary*in[k][1];
						tempF /= N;
						tempF *= *(chirp + k);
						in[k][0] = creal(tempF);
						in[k][1] = cimag(tempF);
					}
					
					// Backward FFT
					fftwf_execute_dft(*(plansB + nChan*i + j), in, in);
					
					// Save
					start = l*N/2;
					stop = start + N/2;
					if( stop > nFFT ) {
						stop = nFFT;
					}
					
					for(k=start; k<stop; k++) {
						dimsT[1] = (npy_intp) k;
						dimsD[2] = (npy_intp) k;
						if( j == 0 ) {
							// We only need to do this once for each stand since timeF is 2-D (stands by FFT windows)
							*(tF + nFFT*i + k) =  *(inT + k - start + N/4);
						}
						*(dF + secStart + k) = in[k - start + N/4][0] + imaginary*in[k - start + N/4][1];
					}
					
					if( stop == nFFT ) {
						break;
					}
					
				}
				
				// Cleanup
				fftwf_free(in);
				free(inT);
				free(chirp);
				free(rf);
			}
		}
	}
	
	// Cleanup
	for(j=0; j<nChan; j++) {
		for(i=0; i<nStand; i++) {
			fftwf_destroy_plan(*(plansF + nChan*i + j));
			fftwf_destroy_plan(*(plansB + nChan*i + j));
		}
	}
	free(plansF);
	free(plansB);
	
	Py_XDECREF(time);
	Py_XDECREF(data);
	Py_XDECREF(freq1);
	Py_XDECREF(freq2);
	Py_XDECREF(pTime);
	Py_XDECREF(pData);
	Py_XDECREF(nTime);
	Py_XDECREF(nData);
	
	drxDataF = Py_BuildValue("(OO)", PyArray_Return(timeF), PyArray_Return(dataF));
	Py_XDECREF(timeF);
	Py_XDECREF(dataF);

	return drxDataF;
}

PyDoc_STRVAR(MultiChannelCD_doc, \
"Given the output of one of the 'PulsarEngine' functions and information about\n\
the time and frequencies, apply coherent dedispersion to the data and return a\n\
two-element tuple giving the time and dedispersed data.\n\
\n\
Input arguments are:\n\
  * time - 2-D numpy.float64 (stands by samples) array of times for the input\n\
    data\n\
  * rawSpectra - 3-D numpy.complex64 (stands by channels by samples) of raw\n\
    spectra data generated by 'PulsarEngineRaw' or 'PulsarEngineWindow'\n\
  * freq1 - 1-D numpy.float64 (channels) array of frequencies for the first\n\
    set of two stands in Hz\n\
  * freq2 - 1-D numpy.float64 (channels) array of frequency for the second\n\
    set of two stands in Hz\n\
  * sampleRate - Channelized data sample rate in Hz\n\
  * DM - dispersion measure in pc cm^-3 to dedisperse at\n\
  * prevTime - 2-D numpy.float64 (stands by samples) array of times for the\n\
    previously input data\n\
  * prevRawSpectra - 3-D numpy.complex64 (stands by channels by samples) of\n\
    the previously input data\n\
  * nextTime - 2-D numpy.float64 (stands by samples) array of times for the\n\
    following input data\n\
  * prevRawSpectra - 3-D numpy.complex64 (stands by channels by samples) of\n\
    the following input data\n\
\n\
Outputs:\n\
  * dedispTime - 2-D numpy.float64 (stands by samples) array of times for\n\
    the coherently dedispersed data\n\
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
	PyObject *signals, *signalsF;
	PyArrayObject *data, *dataF;
	
	long i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &signals)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nChan  = (long) data->dimensions[1];
	nFFT   = (long) data->dimensions[2];
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) (nStand/2);
	dims[1] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStartX, secStartY;
	double tempV;
	float complex *a;
	double *b;
	a = (float complex *) data->data;
	b = (double *) dataF->data;
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStartX, secStartY, i, j, k, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i+=2) {
				secStartX = nSamps*i + nFFT*j;
				secStartY = nSamps*(i+1) + nFFT*j;
				
				for(k=0; k<nFFT; k++) {
					// I
					tempV  = creal(*(a + secStartX + k))*creal(*(a + secStartX + k));
					tempV += cimag(*(a + secStartX + k))*cimag(*(a + secStartX + k));
					tempV += creal(*(a + secStartY + k))*creal(*(a + secStartY + k));
					tempV += cimag(*(a + secStartY + k))*cimag(*(a + secStartY + k));
					*(b + nSamps*(i/2) + nChan*k + j) = tempV;
				}
			}
		}
	}
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
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
	PyObject *signals, *signalsF;
	PyArrayObject *data, *dataF;
	
	long i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &signals)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nChan  = (long) data->dimensions[1];
	nFFT   = (long) data->dimensions[2];
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStart;
	double tempV;
	float complex *a;
	double *b;
	a = (float complex *) data->data;
	b = (double *) dataF->data;
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i++) {
				secStart = nSamps*i + nFFT*j;
				
				for(k=0; k<nFFT; k++) {
					tempV = cabs(*(a + secStart + k));
					tempV *= tempV;
					*(b + nSamps*i + nChan*k + j) = tempV;
				}
			}
		}
	}
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
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
	PyObject *signals, *signalsF;
	PyArrayObject *data, *dataF;
	
	long i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &signals)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nChan  = (long) data->dimensions[1];
	nFFT   = (long) data->dimensions[2];
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStartX, secStartY;
	double tempV;
	double complex tempC;
	float complex *a;
	double *b;
	a = (float complex *) data->data;
	b = (double *) dataF->data;
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStartX, secStartY, i, j, k, tempC, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i+=2) {
				secStartX = nSamps*i + nFFT*j;
				secStartY = nSamps*(i+1) + nFFT*j;
				
				for(k=0; k<nFFT; k++) {
					// LL
					tempC = *(a + secStartX + k) + *(a + secStartY + k)*imaginary;
					tempC /= sqrt(2.0);
					
					tempV  = creal(tempC)*creal(tempC);
					tempV += cimag(tempC)*cimag(tempC);
					*(b + nSamps*(i+0) + nChan*k + j) = tempV;
					
					// RR
					tempC = *(a + secStartX + k) - *(a + secStartY + k)*imaginary;
					tempC /= sqrt(2.0);
					
					tempV  = creal(tempC)*creal(tempC);
					tempV += cimag(tempC)*cimag(tempC);
					*(b + nSamps*(i+1) + nChan*k + j) = tempV;
				}
			}
		}
	}
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
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
	PyObject *signals, *signalsF;
	PyArrayObject *data, *dataF;
	
	long i, j, k, nStand, nSamps, nChan, nFFT;
	
	static char *kwlist[] = {"signals", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &signals)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(signals, NPY_COMPLEX64, 3, 3);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nChan  = (long) data->dimensions[1];
	nFFT   = (long) data->dimensions[2];
	
	// Find out how large the output array needs to be and initialize it
	nSamps = nChan*nFFT;
	npy_intp dims[2];
	dims[0] = (npy_intp) (4*(nStand/2));
	dims[1] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStartX, secStartY;
	double tempV;
	float complex *a;
	double *b;
	a = (float complex *) data->data;
	b = (double *) dataF->data;
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStartX, secStartY, i, j, k, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i+=2) {
				secStartX = nSamps*i + nFFT*j;
				secStartY = nSamps*(i+1) + nFFT*j;
				
				for(k=0; k<nFFT; k++) {
					// I
					tempV  = creal(*(a + secStartX + k))*creal(*(a + secStartX + k));
					tempV += cimag(*(a + secStartX + k))*cimag(*(a + secStartX + k));
					tempV += creal(*(a + secStartY + k))*creal(*(a + secStartY + k));
					tempV += cimag(*(a + secStartY + k))*cimag(*(a + secStartY + k));
					*(b + nSamps*(4*(i/2)+0) + nChan*k + j) = tempV;
					
					// Q
					tempV  = creal(*(a + secStartX + k))*creal(*(a + secStartX + k));
					tempV += cimag(*(a + secStartX + k))*cimag(*(a + secStartX + k));
					tempV -= creal(*(a + secStartY + k))*creal(*(a + secStartY + k));
					tempV -= cimag(*(a + secStartY + k))*cimag(*(a + secStartY + k));
					*(b + nSamps*(4*(i/2)+1) + nChan*k + j) = tempV;
					
					// U
					tempV = 2.0*creal(*(a + secStartX + k) * conj(*(a + secStartY + k)));
					*(b + nSamps*(4*(i/2)+2) + nChan*k + j) = tempV;
					
					// V
					tempV = -2.0*cimag(*(a + secStartX + k) * conj(*(a + secStartY + k)));
					*(b + nSamps*(4*(i/2)+3) + nChan*k + j) = tempV;
				}
			}
		}
	}
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
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
	PyObject *spectra, *output;
	PyArrayObject *data, *zeroF, *scaleF, *dataF;
	long i, j, k, nStand, nSamps, nFFT;
	int nChan = 64;
	
	static char *kwlist[] = {"signals", "LFFT", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &spectra, &nChan)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(spectra, NPY_FLOAT64, 2, 2);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nSamps = (long) data->dimensions[1];
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	zeroF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	if(zeroF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(zeroF, 0);
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	scaleF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	if(scaleF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(scaleF, 0);
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_UINT8);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStart;
	double *tempMin, *tempMax, tempV;
	double *a;
	float *b, *c;
	unsigned char *d;
	a = (double *) data->data;
	b = (float *) zeroF->data;
	c = (float *) scaleF->data;
	d = (unsigned char *) dataF->data;
	
	tempMin = (double *) malloc(nStand*nChan*sizeof(double));
	tempMax = (double *) malloc(nStand*nChan*sizeof(double));
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i++) {
				secStart = nSamps*i + j;
				
				k = 0;
				*(tempMin + i*nChan + j) = *(a + secStart + nChan*k);
				*(tempMax + i*nChan + j) = *(a + secStart + nChan*k);
				
				for(k=1; k<nFFT; k++) {
					if( *(a + secStart + nChan*k) < *(tempMin + i*nChan + j) ) {
						*(tempMin + i*nChan + j) = *(a + secStart + nChan*k);
					} else {
						if( *(a + secStart + nChan*k) > *(tempMax + i*nChan + j) ) {
							*(tempMax + i*nChan + j) = *(a + secStart + nChan*k);
						}
					}
				}
				
				*(b + i*nChan + j) = (float) *(tempMin + i*nChan + j);
				*(c + i*nChan + j) = (float) ((*(tempMax + i*nChan + j) - *(tempMin + i*nChan + j)) / 255.0);
				
				for(k=0; k<nFFT; k++) {
					tempV  = *(a + secStart + nChan*k) - *(b + i*nChan + j);
					tempV /= *(c + i*nChan + j);
					tempV  = round(tempV);
					
					*(d + secStart + nChan*k) = (unsigned char) tempV;
				}
			}
		}
	}
	
	free(tempMin);
	free(tempMax);
	
	Py_XDECREF(data);
	
	output = Py_BuildValue("(OOO)", PyArray_Return(zeroF), PyArray_Return(scaleF), PyArray_Return(dataF));
	Py_XDECREF(zeroF);
	Py_XDECREF(scaleF);
	Py_XDECREF(dataF);
	
	return output;
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
	PyObject *spectra, *output;
	PyArrayObject *data, *zeroF, *scaleF, *dataF;
	long i, j, k, nStand, nSamps, nFFT;
	int nChan = 64;
	
	static char *kwlist[] = {"signals", "LFFT", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &spectra, &nChan)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(spectra, NPY_FLOAT64, 2, 2);
	
	// Get the properties of the data
	nStand = (long) data->dimensions[0];
	nSamps = (long) data->dimensions[1];
	
	// Find out how large the output array needs to be and initialize it
	nFFT = nSamps / nChan;
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	zeroF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	if(zeroF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(zeroF, 0);
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nChan;
	scaleF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	if(scaleF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(scaleF, 0);
	
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_UINT8);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Go!
	long secStart;
	double *tempMin, *tempMax, tempV;
	double *a;
	float *b, *c;
	unsigned char *d;
	a = (double *) data->data;
	b = (float *) zeroF->data;
	c = (float *) scaleF->data;
	d = (unsigned char *) dataF->data;
	
	tempMin = (double *) malloc(nStand*nChan*sizeof(double));
	tempMax = (double *) malloc(nStand*nChan*sizeof(double));
	
	#ifdef _OPENMP
		#pragma omp parallel default(shared) private(secStart, i, j, k, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(dynamic)
		#endif
		for(j=0; j<nChan; j++) {
			for(i=0; i<nStand; i++) {
				secStart = nSamps*i + j;
				
				k = 0;
				*(tempMin + i*nChan + j) = *(a + secStart + nChan*k);
				*(tempMax + i*nChan + j) = *(a + secStart + nChan*k);
				
				for(k=1; k<nFFT; k++) {
					if( *(a + secStart + nChan*k) < *(tempMin + i*nChan + j) ) {
						*(tempMin + i*nChan + j) = *(a + secStart + nChan*k);
					} else {
						if( *(a + secStart + nChan*k) > *(tempMax + i*nChan + j) ) {
							*(tempMax + i*nChan + j) = *(a + secStart + nChan*k);
						}
					}
				}
				
				*(b + i*nChan + j) = (float) *(tempMin + i*nChan + j);
				*(c + i*nChan + j) = (float) ((*(tempMax + i*nChan + j) - *(tempMin + i*nChan + j)) / 15.0);
				
				for(k=0; k<nFFT; k++) {
					tempV  = *(a + secStart + nChan*k) - *(b + i*nChan + j);
					tempV /= *(c + i*nChan + j);
					tempV  = round(tempV);
					
					*(d + secStart + nChan*k) = ((unsigned char) tempV ) & 0xF;
				}
			}
		}
	}
	
	free(tempMin);
	free(tempMax);
	
	Py_XDECREF(data);
	
	output = Py_BuildValue("(OOO)", PyArray_Return(zeroF), PyArray_Return(scaleF), PyArray_Return(dataF));
	Py_XDECREF(zeroF);
	Py_XDECREF(scaleF);
	Py_XDECREF(dataF);
	
	return output;
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
	{"PulsarEngineRaw",        (PyCFunction) PulsarEngineRaw,        METH_VARARGS|METH_KEYWORDS, PulsarEngineRaw_doc        },
	{"PulsarEngineRawWindow",  (PyCFunction) PulsarEngineRawWindow,  METH_VARARGS|METH_KEYWORDS, PulsarEngineRawWindow_doc  },
	{"PhaseRotator",           (PyCFunction) PhaseRotator,           METH_VARARGS,               PhaseRotator_doc           },
	{"ComputeSKMask",          (PyCFunction) ComputeSKMask,          METH_VARARGS,               ComputeSKMask_doc          },
	{"ComputePseudoSKMask",    (PyCFunction) ComputePseudoSKMask,    METH_VARARGS,               ComputePseudoSKMask_doc    },
	{"MultiChannelCD",         (PyCFunction) MultiChannelCD,         METH_VARARGS,               MultiChannelCD_doc         },
	{"CombineToIntensity",     (PyCFunction) CombineToIntensity,     METH_VARARGS,               CombineToIntensity_doc     }, 
	{"CombineToLinear",        (PyCFunction) CombineToLinear,        METH_VARARGS,               CombineToLinear_doc        }, 
	{"CombineToCircular",      (PyCFunction) CombineToCircular,      METH_VARARGS,               CombineToCircular_doc      }, 
	{"CombineToStokes",        (PyCFunction) CombineToStokes,        METH_VARARGS,               CombineToStokes_doc        },
	{"OptimizeDataLevels8Bit", (PyCFunction) OptimizeDataLevels8Bit, METH_VARARGS,               OptimizeDataLevels8Bit_doc },
	{"OptimizeDataLevels4Bit", (PyCFunction) OptimizeDataLevels4Bit, METH_VARARGS,               OptimizeDataLevels4Bit_doc },
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
	PyModule_AddObject(m, "__version__", PyString_FromString("0.5"));
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
