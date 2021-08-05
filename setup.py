import os
import tempfile
import subprocess
from setuptools import setup, Extension, find_packages
from distutils import log
from distutils.command.install import install
try:
    # Attempt to use Cython for building extensions, if available
    from Cython.Distutils.build_ext import build_ext
    # Additionally, assert that the compiler module will load
    # also. Ref #1229.
    __import__('Cython.Compiler.Main')
except ImportError:
    from distutils.command.build_ext import build_ext


def get_openmp():
    """Try to compile/link an example program to check for OpenMP support.
    
    Based on:
    1) http://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
    2) https://github.com/lpsinger/healpy/blob/6c3aae58b5f3281e260ef7adce17b1ffc68016f0/setup.py
    """
    
    import shutil
    from distutils import sysconfig
    from distutils import ccompiler
    compiler = ccompiler.new_compiler()
    sysconfig.get_config_vars()
    sysconfig.customize_compiler(compiler)
    cc = compiler.compiler
    
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    
    fh = open('test.c', 'w')
    fh.write(r"""#include <omp.h>
#include <stdio.h>
int main(void) {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
return 0;
}
""")
    fh.close()
    
    ccmd = []
    ccmd.extend( cc )
    ccmd.extend( ['-fopenmp', 'test.c', '-o test'] )
    if os.path.basename(cc[0]).find('gcc') != -1:
        ccmd.append( '-lgomp' )
    elif os.path.basename(cc[0]).find('clang') != -1:
        ccmd.extend( ['-L/opt/local/lib/libomp', '-lomp'] )
    try:
        output = subprocess.check_call(ccmd)
        outCFLAGS = ['-fopenmp',]
        outLIBS = []
        if os.path.basename(cc[0]).find('gcc') != -1:
            outLIBS.append( '-lgomp' )
        elif os.path.basename(cc[0]).find('clang') != -1:
            outLIBS.extend( ['-L/opt/local/lib/libomp', '-lomp'] )
            
    except subprocess.CalledProcessError:
        print("WARNING:  OpenMP does not appear to be supported by %s, disabling" % cc[0])
        outCFLAGS = []
        outLIBS = []
        
    finally:
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
        
    return outCFLAGS, outLIBS


def get_fftw():
    """Use pkg-config (if installed) to figure out the C flags and linker flags
    needed to compile a C program with single precision FFTW3.  If FFTW3 cannot 
    be found via pkg-config, some 'sane' values are returned."""
    
    try:
        subprocess.check_call(['pkg-config', 'fftw3f', '--exists'])
        
        p = subprocess.Popen(['pkg-config', 'fftw3f', '--modversion'], stdout=subprocess.PIPE)
        outVersion = p.communicate()[0].rstrip().split()
        
        p = subprocess.Popen(['pkg-config', 'fftw3f', '--cflags'], stdout=subprocess.PIPE)
        outCFLAGS = p.communicate()[0].rstrip().split()
        try:
            outCFLAGS = [str(v, 'utf-8') for v in outCFLAGS]
        except TypeError:
            pass
        
        p = subprocess.Popen(['pkg-config', 'fftw3f', '--libs'], stdout=subprocess.PIPE)
        outLIBS = p.communicate()[0].rstrip().split()
        try:
            outLIBS = [str(v, 'utf-8') for v in outLIBS]
        except TypeError:
            pass
            
        if len(outVersion) > 0:
            print("Found FFTW3, version %s" % outVersion[0])
            
    except (OSError, subprocess.CalledProcessError):
        print("WARNING:  single precision FFTW3 cannot be found, using defaults")
        outCFLAGS = []
        outLIBS = ['-lfftw3f', '-lm']
        
    return outCFLAGS, outLIBS


class dummy_install(install):
    def finalize_options(self, *args, **kwargs):
        raise RuntimeError("This is a dummy package that cannot be installed")


openmpFlags, openmpLibs = get_openmp()
fftwFlags, fftwLibs = get_fftw()


coreExtraFlags = openmpFlags
coreExtraFlags.extend(fftwFlags)
coreExtraFlags.append('-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION')
coreExtraLibs = openmpLibs
coreExtraLibs.extend(fftwLibs)


ExtensionModules = [Extension('_psr', ['psr.c', 'utils.c', 'fft.c', 'kurtosis.c', 'dedispersion.c', 'reduce.c', 'quantize.c'],
                              include_dirs=[numpy.get_include()], libraries=['m'],
                              extra_compile_args=coreExtraFlags, extra_link_args=coreExtraLibs),
                    Extension('_helper', ['helper.c',],
                              include_dirs=[numpy.get_include()], libraries=['m'],
                              extra_compile_args=coreExtraFlags, extra_link_args=coreExtraLibs)]


setup(
    cmdclass = {'install': dummy_install}, 
    name = 'dummy_package',
    version = '0.0',
    description = 'This is a dummy package to help build the pulsar extensions',
    ext_modules = ExtensionModules
)
