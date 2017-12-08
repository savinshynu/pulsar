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

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL psr_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "psr.h"


PyObject *CombineToIntensity(PyObject *self, PyObject *args, PyObject *kwds) {
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

char CombineToIntensity_doc[] = PyDoc_STR(\
"Given the output of PulsarEngineRaw, calculate total intensity spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");


PyObject *CombineToLinear(PyObject *self, PyObject *args, PyObject *kwds) {
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

char CombineToLinear_doc[] = PyDoc_STR(\
"Given the output of PulsarEngineRaw, calculate linear polarization spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");


PyObject *CombineToCircular(PyObject *self, PyObject *args, PyObject *kwds) {
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

char CombineToCircular_doc[] = PyDoc_STR(\
"Given the output of PulsarEngineRaw, calculate circular polarization spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");


PyObject *CombineToStokes(PyObject *self, PyObject *args, PyObject *kwds) {
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

char CombineToStokes_doc[] = PyDoc_STR(\
"Given the output of PulsarEngineRaw, calculate Stokes parameter spectra\n\
\n\
Input arguments are:\n\
 * signals: 3-D numpy.complex64 (stands by channels by integrations) array\n\
   of data to FFT\n\
\n\
Outputs:\n\
 * sub-integration: 2-D numpy.float64 (stands by channels) of spectra data\n\
");
