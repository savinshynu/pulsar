# The LWA Pulsar Stack



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
./bootstap
./configure --enable-shared --with-cuda-dir=/usr/local/cuda --with-cuda-include-dir=/usr/local/cuda/include --with-cuda-lib-dir=/usr/local/cuda/lib64
make
sudo make install
```
