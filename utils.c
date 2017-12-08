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


/*
  Core binding function - based off the corresponding bifrost function
*/


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

PyObject *BindToCore(PyObject *self, PyObject *args, PyObject *kwds) {
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

char BindToCore_doc[] = PyDoc_STR(\
"Bind the current thread to the specified core.\n\
\n\
Input arguments are:\n\
 * core: scalar int core to bind to\n\
\n\
Outputs:\n\
 * True, if successful\n\
");


PyObject *BindOpenMPToCores(PyObject *self, PyObject *args, PyObject *kwds) {
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

char BindOpenMPToCores_doc[] = PyDoc_STR(\
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
