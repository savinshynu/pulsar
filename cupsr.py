import cupy as cp
import numpy as np

__version__ = "0.1"
__all__ = ['PulsarEngineRaw', 'PhaseRotator',
           'ComputeSKMask',
           'CombineToIntensity', 'CombineToLinear',
           'OptimizeDataLevels8Bit', 'OptimizeDataLevels4Bit']


def PulsarEngineRaw(signals, LFFT=64, signalsF=None, asnumpy=False):
    nstand, nsamp = signals.shape
    nchan = LFFT
    nwin = nsamp // nchan
    assert(nsamp % nchan == 0)
    if not isinstance(signals, cp.ndarray):
        signals = cp.asarray(signals)
        
    signals.shape = (nstand, nwin, nchan)
    signalsF = cp.fft.fft(signals, n=nchan, axis=2, norm='ortho')
    signalsF = cp.fft.fftshift(signalsF, axes=2)
    signalsF = signalsF.transpose(0,2,1)
    signalsF = signalsF.copy()
    
    if asnumpy:
        signalsF = cp.asnumpy(signalsF)
    return signalsF


_PROT = cp.RawKernel(r"""
#define M_PI 3.1415926535897931e+0

extern "C"
__global__ void prot(const float2 *input,
                     const double *freq1,
                     const double *freq2,
                     const double delay,
                     int nstand,
                     int nchan,
                     int nwin,
                     float2 *output) {
  int s = blockIdx.x;
  int c = blockIdx.y;
  int w = blockIdx.z*512 + threadIdx.x;
  if( w < nwin ) {
    float2 temp;
    double2 rot, out;
    double f;
    
    temp = *(input + s*nchan*nwin + c*nwin + w);
    if( s / 2 == 0 ) {
      f = *(freq1 + c);
    } else {
      f = *(freq2 + c);
    }
    
    rot.x = cos(2*M_PI*f*delay);
    rot.y = sin(2*M_PI*f*delay);
    
    out.x = temp.x*rot.x - temp.y*rot.y;
    out.y = temp.x*rot.y + temp.y*rot.x;
    
    *(output + s*nchan*nwin + c*nwin + w) = make_float2(out.x, out.y);
  }
}
""", 'prot')


def PhaseRotator(signals, freq1, freq2, delays, signalsF=None, asnumpy=False):
    nstand, nchan, nwin = signals.shape
    assert(nchan == freq1.size)
    assert(nchan == freq2.size)
    if not isinstance(signals, cp.ndarray):
        signals = cp.asarray(signals)
    if not isinstance(freq1, cp.ndarray):
        freq1 = cp.asarray(freq1)
    if not isinstance(freq2, cp.ndarray):
        freq2 = cp.asarray(freq2)
        
    if signalsF is None:
        signalsF = cp.empty(shape=signals.shape, dtype=signals.dtype)
    else:
        if not isinstance(signalsF, cp.ndarray):
            signalsF = cp.asarray(signalsF)
            
    _PROT((nstand,nchan,int(np.ceil(nwin/512))), (min([512, nwin]),),
          (signals, freq1, freq2, cp.float64(delays),
           cp.int32(nstand), cp.int32(nchan), cp.int32(nwin),
           signalsF))
    
    if asnumpy:
        signalsF = cp.asnumpy(signalsF)
    return signalsF


_SKMASK = cp.RawKernel(r"""
extern "C"
__global__ void skmask(const float2 *input,
                       double lower,
                       double upper,
                       int nstand,
                       int nchan,
                       int nwin,
                       float *mask) {
  int s = blockIdx.x;
  int c = blockIdx.y*512 + threadIdx.x;
  if( c < nchan ) {
    int w;
    float2 temp;
    float tempV, tempV2, temp2V;
    
    tempV2 = temp2V = 0.0;
    for(w=0; w<nwin; w++) {
      temp = *(input + s*nchan*nwin + c*nwin + w);
      tempV = temp.x*temp.x + temp.y*temp.y;
      temp2V += tempV*tempV;
      tempV2 += tempV;
    }
    
    tempV  = nwin*temp2V / (tempV2*tempV2) - 1.0;
    tempV *= (nwin + 1.0)/(nwin - 1.0);
    
    if( tempV < lower || tempV > upper ) {
        *(mask + s*nchan + c) = 0.0;
    } else {
        *(mask + s*nchan + c) = 1.0;
    }
  }
}
""", 'skmask')


def ComputeSKMask(signals, lower, upper, asnumpy=False):
    nstand, nchan, nwin = signals.shape
    if not isinstance(signals, cp.ndarray):
        signals = cp.asarray(signals)
        
    mask = cp.empty((nstand,nchan), dtype=np.float32)
    
    _SKMASK((nstand,int(np.ceil(nchan/512))), (min([512, nchan]),),
            (signals, cp.double(lower), cp.double(upper),
             cp.int32(nstand), cp.int32(nchan), cp.int32(nwin),
             mask))
    
    if asnumpy:
        mask = cp.asnumpy(mask)
    return mask


_INTEN = cp.RawKernel(r"""
extern "C"
__global__ void inten(const float2 *input,
                      int nstand,
                      int nchan,
                      int nwin,
                      float *output) {
  int s = blockIdx.x;
  int c = blockIdx.y;
  int w = blockIdx.z*512 + threadIdx.x;
  if( w < nwin ) {
    float2 tempX, tempY;
    
    tempX = *(input + (2*s + 0)*nchan*nwin + c*nwin + w);
    tempY = *(input + (2*s + 1)*nchan*nwin + c*nwin + w);
    
    *(output + s*nwin*nchan + w*nchan + c) = tempX.x*tempX.x + tempX.y*tempX.y \
                                           + tempY.x*tempY.x + tempY.y*tempY.y;
  }
}
""", 'inten')


def CombineToIntensity(signals, signalsF=None, asnumpy=False):
    nstand, nchan, nwin = signals.shape
    if not isinstance(signals, cp.ndarray):
        signals = cp.asarray(signals)
        
    if signalsF is None:
        signalsF = cp.empty(shape=(nstand//2, nchan, nwin), dtype=np.float32)
    else:
        if not isinstance(signalsF, cp.ndarray):
            signalsF = cp.asarray(signalsF)
            
    _INTEN((nstand//2,nchan,int(np.ceil(nwin/512))), (min([512, nwin]),),
           (signals,
            cp.int32(nstand//2), cp.int32(nchan), cp.int32(nwin),
            signalsF))
    
    signalsF.shape = (nstand//2, nchan*nwin)
    if asnumpy:
        signalsF = cp.asnumpy(signalsF)
    return signalsF


_LINEAR = cp.RawKernel(r"""
extern "C"
__global__ void linear(const float2 *input,
                       int nstand,
                       int nchan,
                       int nwin,
                       float *output) {
  int s = blockIdx.x;
  int c = blockIdx.y;
  int w = blockIdx.z*512 + threadIdx.x;
  if( w < nwin ) {
    float2 temp;
    
    temp = *(input + s*nchan*nwin + c*nwin + w);
    
    *(output + s*nwin*nchan + w*nchan + c) = temp.x*temp.x + temp.y*temp.y;
  }
}
""", 'linear')


def CombineToLinear(signals, signalsF=None, asnumpy=False):
    nstand, nchan, nwin = signals.shape
    if not isinstance(signals, cp.ndarray):
        signals = cp.asarray(signals)
        
    if signalsF is None:
        signalsF = cp.empty(shape=signals.shape, dtype=np.float32)
    else:
        if not isinstance(signalsF, cp.ndarray):
            signalsF = cp.asarray(signalsF)
            
    _LINEAR((nstand,nchan,int(np.ceil(nwin/512))), (min([512, nwin]),),
            (signals,
             cp.int32(nstand), cp.int32(nchan), cp.int32(nwin),
             signalsF))
    
    signalsF.shape = (nstand, nchan*nwin)
    if asnumpy:
        signalsF = cp.asnumpy(signalsF)
    return signalsF


_OPT8 = cp.RawKernel(r"""
extern "C"
__global__ void opt8(const float *input,
                     const float *zero,
                     const float *scale,
                     int nstand,
                     int nchan,
                     int nwin,
                     unsigned char *output) {
  int s = blockIdx.x;
  int c = blockIdx.y;
  int w = blockIdx.z*512 + threadIdx.x;
  if( w < nwin ) {
    float temp, z, a;
    temp = *(input + s*nwin*nchan + w*nchan + c);
    z = *(zero + s*nchan + c);
    a = *(scale + s*nchan + c);
    
    temp = (temp - z) / a;
    temp = round(temp);
    
    *(output + s*nwin*nchan + w*nchan + c) = (unsigned char) temp;
  }
}
""", 'opt8')


def OptimizeDataLevels8Bit(signals, LFFT, bzero=None, bscale=None, spectra=None, asnumpy=False):
    nstand, nsamp = signals.shape
    nchan = LFFT
    nwin = nsamp // nchan
    assert(nsamp % nchan == 0)
    if not isinstance(signals, cp.ndarray):
        signals = cp.asarray(signals)
        
    if bzero is None:
        bzero = cp.empty(shape=(nstand,nchan), dtype=np.float32)
    else:
        if not isinstance(bzero, cp.ndarray):
            bzero = cp.asarray(bzero)
    if bscale is None:
        bscale = cp.empty(shape=(nstand,nchan), dtype=np.float32)
    else:
        if not isinstance(bscale, cp.ndarray):
            bscale = cp.asarray(bscale)
    if spectra is None:
        spectra = cp.empty(shape=(nstand,nsamp), dtype=np.uint8)
    else:
        if not isinstance(spectra, cp.ndarray):
            spectra = cp.asarray(spectra)
            
    signals.shape = (nstand,nwin,nchan)
    lower = cp.min(signals, axis=1)
    upper = cp.max(signals, axis=1)
    bzero = lower
    bscale = (upper - lower) / 255.0
    
    _OPT8((nstand,nchan,int(np.ceil(nwin/512))), (min([512, nwin]),),
          (signals, bzero, bscale,
           cp.int32(nstand), cp.int32(nchan), cp.int32(nwin),
           spectra))
           
    if asnumpy:
        bzero = cp.asnumpy(bzero)
        bscale = cp.asnumpy(bscale)
        spectra = cp.asnumpy(spectra)
    return bzero, bscale, spectra


_OPT4 = cp.RawKernel(r"""
extern "C"
__global__ void opt4(const float *input,
                     const float *zero,
                     const float *scale,
                     int nstand,
                     int nchan,
                     int nwin,
                     unsigned char *output) {
  int s = blockIdx.x;
  int c = blockIdx.y;
  int w = blockIdx.z*512 + threadIdx.x;
  if( w < nwin ) {
    float temp, z, a;
    temp = *(input + s*nwin*nchan + w*nchan + c);
    z = *(zero + s*nchan + c);
    a = *(scale + s*nchan + c);
    
    temp = (temp - z) / a;
    temp = round(temp);
    
    *(output + s*nwin*nchan + w*nchan + c) = ((unsigned char) temp) & 0xF;
  }
}
""", 'opt4')


def OptimizeDataLevels4Bit(signals, LFFT, bzero=None, bscale=None, spectra=None, asnumpy=False):
    nstand, nsamp = signals.shape
    nchan = LFFT
    nwin = nsamp // nchan
    assert(nsamp % nchan == 0)
    if not isinstance(signals, cp.ndarray):
        signals = cp.asarray(signals)
        
    if bzero is None:
        bzero = cp.empty(shape=(nstand,nchan), dtype=np.float32)
    else:
        if not isinstance(bzero, cp.ndarray):
            bzero = cp.asarray(bzero)
    if bscale is None:
        bscale = cp.empty(shape=(nstand,nchan), dtype=np.float32)
    else:
        if not isinstance(bscale, cp.ndarray):
            bscale = cp.asarray(bscale)
    if spectra is None:
        spectra = cp.empty(shape=(nstand,nsamp), dtype=np.uint8)
    else:
        if not isinstance(spectra, cp.ndarray):
            spectra = cp.asarray(spectra)
            
    signals.shape = (nstand,nwin,nchan)
    lower = cp.min(signals, axis=1)
    upper = cp.max(signals, axis=1)
    bzero = lower
    bscale = (upper - lower) / 15.0
    
    _OPT4((nstand,nchan,int(np.ceil(nwin/512))), (min([512, nwin]),),
          (signals, bzero, bscale,
           cp.int32(nstand), cp.int32(nchan), cp.int32(nwin),
           spectra))
           
    if asnumpy:
        bzero = cp.asnumpy(bzero)
        bscale = cp.asnumpy(bscale)
        spectra = cp.asnumpy(spectra)
    return bzero, bscale, spectra
