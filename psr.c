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


static PyObject *PulsarEngine(PyObject *self, PyObject *args, PyObject *kwds) {
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
	npy_intp dims[2];
	dims[0] = (npy_intp) nStand;
	dims[1] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_UINT8);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Create the FFTW plan
	fftw_complex *inP, *in;
	inP = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nChan);
	fftw_plan p;
	p = fftw_plan_dft_1d(nChan, inP, inP, FFTW_FORWARD, FFTW_MEASURE);

	// Integer delay, FFT, and fractional delay
	long secStart;
	float complex *a;
	unsigned char *b;
	double tempV;
	a = (float complex *) data->data;
	b = (unsigned char *) dataF->data;
	
	#ifdef _OPENMP
		#ifdef _MKL
			fftw3_mkl.number_of_user_threads = omp_get_num_threads();
		#endif
		#pragma omp parallel default(shared) private(in, secStart, i, j, k, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(static)
		#endif
		for(i=0; i<nStand; i++) {
			in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nChan);
			
			for(j=0; j<nFFT; j++) {
				secStart = nSamps * i + nChan*j;
				
				for(k=0; k<nChan; k++) {
					in[k][0] = creal(*(a + secStart + k));
					in[k][1] = cimag(*(a + secStart + k));
				}
				
				fftw_execute_dft(p, in, in);
				
				for(k=0; k<nChan; k++) {
					tempV = in[k][0]*in[k][0] + in[k][1]*in[k][1];
					tempV /= nChan;
					if( tempV > 255) {
						tempV = 255;
					}
					
					if( k < nChan/2 ) {
						*(b + nSamps*i + nChan*j + k + nChan/2) = (unsigned char) tempV;
					} else {
						*(b + nSamps*i + nChan*j + k - nChan/2) = (unsigned char) tempV;
					}
				}
			}
			fftw_free(in);
		}
	}
	fftw_destroy_plan(p);
	fftw_free(inP);
	
	Py_XDECREF(data);
	
	signalsF = Py_BuildValue("O", PyArray_Return(dataF));
	Py_XDECREF(dataF);

	return signalsF;
}

PyDoc_STRVAR(PulsarEngine_doc, \
"Perform a series of Fourier transforms on complex-valued data to get sub-\n\
integration data\n\
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


static PyObject *SumPolsWithOverflow(PyObject *self, PyObject *args, PyObject *kwds) {
	PyObject *spectra, *spectraF;
	PyArrayObject *data, *dataF;
	long i, index0, index1, nStands, nSamps;
	
	static char *kwlist[] = {"spectra", "index0", "index1", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Oii", kwlist, &spectra, &index0, &index1)) {
		PyErr_Format(PyExc_RuntimeError, "Invalid parameters");
		return NULL;
	}
	
	// Bring the data into C and make it usable
	data = (PyArrayObject *) PyArray_ContiguousFromObject(spectra, NPY_UINT8, 2, 2);
	
	// Get the properties of the data
	nStands = (long) data->dimensions[0];
	nSamps = (long) data->dimensions[1];
	
	// Validation
	if( index0 < 0 || index0 >= nStands ) {
		PyErr_Format(PyExc_ValueError, "index0 is out-of-range");
		Py_XDECREF(data);
		return NULL;
	}
	if( index1 < 0 || index1 >= nStands ) {
		PyErr_Format(PyExc_ValueError, "index1 is out-of-range");
		Py_XDECREF(data);
		return NULL;
	}
	if( index0 == index1 ) {
		PyErr_Format(PyExc_ValueError, "index1 cannot equal index0");
		Py_XDECREF(data);
		return NULL;
	}
	
	// Find out how large the output array needs to be and initialize it
	npy_intp dims[1];
	dims[0] = (npy_intp) nSamps;
	dataF = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_UINT8);
	if(dataF == NULL) {
		PyErr_Format(PyExc_MemoryError, "Cannot create output array");
		Py_XDECREF(data);
		return NULL;
	}
	PyArray_FILLWBYTE(dataF, 0);
	
	// Integer delay, FFT, and fractional delay
	unsigned char *a, *b;
	unsigned short int tempV;
	a = (unsigned char *) data->data;
	b = (unsigned char *) dataF->data;
	
	#ifdef _OPENMP
		#ifdef _MKL
			fftw3_mkl.number_of_user_threads = omp_get_num_threads();
		#endif
		#pragma omp parallel default(shared) private(i, tempV)
	#endif
	{
		#ifdef _OPENMP
			#pragma omp for schedule(static)
		#endif
		for(i=0; i<nSamps; i++) {
			tempV = (unsigned short) *(a + index0*nSamps + i) + (unsigned short) *(a + index1*nSamps + i);
			if( tempV > 255 ) {
				*(b + i) = 255;
			} else {
				*(b + i) = (unsigned char) tempV;
			}
		}
	}
	
	Py_XDECREF(data);
	
	spectraF = Py_BuildValue("O", PyArray_Return(dataF));
	Py_XDECREF(dataF);

	return spectraF;
}

PyDoc_STRVAR(SumPolsWithOverflow_doc, \
"Given a 2-D numpy.unit8 array created by PulsarEngine() and two indicies\n\
along the zeroth axis that specify the two polarization, sum the\n\
polarizations.\n\
\n\
Input arguments are:\n\
 * signals: 2-D numpy.uint8 (stands by samples) array of sub-integration data\n\
 * index0: integer index of the first polarization\n\
 * index1: integer index of the second polarization\n\
\n\
Input keywords are:\n\
 None\n\
\n\
Outputs:\n\
 * sub-integration: 1-D numpy.uint8 (stands by channels) of the summed data\n\
");


/*
  Module Setup - Function Definitions and Documentation
*/

static PyMethodDef SpecMethods[] = {
	{"PulsarEngine",        (PyCFunction) PulsarEngine,        METH_VARARGS|METH_KEYWORDS, PulsarEngine_doc}, 
	{"SumPolsWithOverflow", (PyCFunction) SumPolsWithOverflow, METH_VARARGS,               SumPolsWithOverflow_doc},
	{NULL,      NULL,    0,                          NULL      }
};

PyDoc_STRVAR(spec_doc, \
"Extension to take timeseries data and convert it to the frequency domain.\n\
\n\
The functions defined in this module are:\n\
  * PulsarEngine - FFT and integrate function for computing a series of\n\
    Fourier transforms for a complex-valued (TBN and DRX) signal from a \n\
    collection of stands/beams all at once.\n\
  * SumPolsWithOverflow - Sum two polarization stored in an unsigned\n\
    byte array being mindful of overflows.\n\
\n\
See the inidividual functions for more details.");


/*
  Module Setup - Initialization
*/

PyMODINIT_FUNC init_psr(void) {
	PyObject *m;

	// Module definitions and functions
	m = Py_InitModule3("_psr", SpecMethods, spec_doc);
	import_array();
	
	// Version and revision information
	PyModule_AddObject(m, "__version__", PyString_FromString("0.1"));
	PyModule_AddObject(m, "__revision__", PyString_FromString("$Rev$"));
	
}
