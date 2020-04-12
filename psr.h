#ifndef __PSR_H
#define __PSR_H

#include <complex.h>

// Python3 support
#if PY_MAJOR_VERSION >= 3
    #define PyInt_AsLong PyLong_AsLong
    #define PyInt_FromLong PyLong_FromLong
#endif

// Dispersion constant in MHz^2 s / pc cm^-3
#define DCONST (double) 4.148808e3

/* 
 Support Functions
*/

// fft.c
void read_wisdom(char*, PyObject*);
// utils.c
extern PyObject *BindToCore(PyObject*, PyObject*, PyObject*);
extern char BindToCore_doc[];
extern PyObject *BindOpenMPToCores(PyObject*, PyObject*, PyObject*);
extern char BindOpenMPToCores_doc[];
double cabs2(double complex);
float cabs2f(float complex);


/*
 FFT Functions
*/

// fft.c
extern PyObject *PulsarEngineRaw(PyObject*, PyObject*, PyObject*);
extern char PulsarEngineRaw_doc[];
extern PyObject *PulsarEngineRawWindow(PyObject*, PyObject*, PyObject*);
extern char PulsarEngineRawWindow_doc[];
extern PyObject *PhaseRotator(PyObject*, PyObject*, PyObject*);
extern char PhaseRotator_doc[];

/*
 Spectral Kurtosis (RFI Flagging) Functions
*/

// kurtosis.c
extern PyObject *ComputeSKMask(PyObject*, PyObject*, PyObject*);
extern char ComputeSKMask_doc[];
extern PyObject *ComputePseudoSKMask(PyObject*, PyObject*, PyObject*);
extern char ComputePseudoSKMask_doc[];


/*
 Coherent Dedispersion
*/

// dedispersion.c
extern PyObject *MultiChannelCD(PyObject*, PyObject*, PyObject*);
extern char MultiChannelCD_doc[];


/*
 Reduction Functions
*/

// reduce.c
extern PyObject *CombineToIntensity(PyObject*, PyObject*, PyObject*);
extern char CombineToIntensity_doc[];
extern PyObject *CombineToLinear(PyObject*, PyObject*, PyObject*);
extern char CombineToLinear_doc[];
extern PyObject *CombineToCircular(PyObject*, PyObject*, PyObject*);
extern char CombineToCircular_doc[];
extern PyObject *CombineToStokes(PyObject*, PyObject*, PyObject*);
extern char CombineToStokes_doc[];


/*
 Quantizing Functions
*/

// quantize.c
extern PyObject *OptimizeDataLevels8Bit(PyObject*, PyObject*, PyObject*);
extern char OptimizeDataLevels8Bit_doc[];
extern PyObject *OptimizeDataLevels4Bit(PyObject*, PyObject*, PyObject*);
extern char OptimizeDataLevels4Bit_doc[];

#endif	// __PSR_H