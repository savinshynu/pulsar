# The LWA Pulsar Stack

Pulsar data reduction software (and versions) currently avaliable on the LWA Users Computing Facility ([LWA Memo #193](http://www.phys.unm.edu/~lwa/memos/memo/lwa0193d.pdf)).  This is in addition to the [LWA Software Library](https://fornax.phys.unm.edu/lwa/trac/) and the [pulsar extension](https://github.com/lwa-project/pulsar/).

## TEMPO
http://tempo.sourceforge.net/
```
git clone git://git.code.sf.net/p/tempo/tempo
git checkout 6bab1083350eca24745eafed79a55156bdd1e7d5
./prepare
./configure
make
sudo make install
```

## PRESTO
https://www.cv.nrao.edu/~sransom/presto/
```
git clone https://github.com/scottransom/presto.git
git checkout e90b8148f813c151f588f5f94b81b606964b03a8
cd src
make
```
with one small change to get around some segfaults in `prepdata`:
```
diff --git a/src/backend_common.c b/src/backend_common.c
index d38c3ba..f9f173b 100644
--- a/src/backend_common.c
+++ b/src/backend_common.c
@@ -532,8 +532,8 @@ int read_psrdata(float *fdata, int numspect, struct spectra_info *s,
             numsubints = numspect / s->spectra_per_subint;
         if (obsmask->numchan)
             mask = 1;
-        rawdata1 = gen_fvect(numsubints * s->spectra_per_subint * s->num_channels);
-        rawdata2 = gen_fvect(numsubints * s->spectra_per_subint * s->num_channels);
+        rawdata1 = gen_fvect((long) numsubints * s->spectra_per_subint * s->num_channels);
+        rawdata2 = gen_fvect((long) numsubints * s->spectra_per_subint * s->num_channels);
         allocd = 1;
         duration = numsubints * s->time_per_subint;
         currentdata = rawdata1;
```

## psrfits_utils
```
git clone https://github.com/lwa-project/psrfits_utils.git
./prepare
./configure
make
sudo make install
```

## EPSIC
```
git clone https://github.com/straten/epsic.git
git checkout 5315cc634f6539ea0a34e403e492472b97e0f086
cd src/
./bootstrap
./configure
make
sudo make install
```

## PSRCAT
https://www.atnf.csiro.au/research/pulsar/psrcat/
```
wget https://www.atnf.csiro.au/research/pulsar/psrcat/downloads/psrcat_pkg.tar.gz
tar xzvf psrcat_pkg.tar.gz
mv psrcat_tar psrcat
./makeit
sudo cp psrcat /usr/local/bin/
```

## PSRCHIVE
http://psrchive.sourceforge.net/
```
git clone git://git.code.sf.net/p/psrchive/code
git checkout ca12b4a279f3d4adcca223508116d9d270df8cc6
./bootstrap
./configure --enable-shared --disable-tempo2 --with-psrcat=/usr/local/psrcat/
make
sudo make install
```

## DSPSR
http://dspsr.sourceforge.net/
```
git clone git://git.code.sf.net/p/dspsr/code
git checkout c277eba1e05ffa5e03310b13c2a0f0477758cf4f
./bootstrap
./configure --enable-shared --with-cuda-dir=/usr/local/cuda --with-cuda-include-dir=/usr/local/cuda/include --with-cuda-lib-dir=/usr/local/cuda/lib64
make
sudo make install
```
