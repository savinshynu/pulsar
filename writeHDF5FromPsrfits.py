#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a PSRFITS file, create a HDF5 file in the standard LWA1 format.
"""

import os
import re
import sys
import h5py
import numpy
import ephem
import ctypes
import getopt
import pyfits
from datetime import datetime

import data as hdfData

import lsl.astro as astro
import lsl.common.progress as progress


_fnRE = re.compile('.*_b(?P<beam>[1-4])(t(?P<tuning>[12]))?_.*\.fits')


def usage(exitCode=None):
    print """writeHDF5FromPsrfits.py - Read in a PSRFITS file and create an HDF5 
file in the standard LWA1 format.

Usage: writeFromHDF5FromPsrfits.py [OPTIONS] file

Options:
-h, --help                  Display this help information
-s, --skip                  Time in seconds to skip into the file, in 
                            seconds (default = 0)
-d, --duration              Amount of time to save in seconds (default = 10)
-o, --output                Output file basename
"""

    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseOptions(args):
    config = {}
    # Command line flags - default values
    config['output'] = None
    config['skip'] = 0.0
    config['duration'] = 10.0
    config['args'] = []
    
    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "ho:s:d:", ["help", "output=", "skip=", "duration="])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage(exitCode=2)
        
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-s', '--skip'):
            config['skip'] = float(value)
        elif opt in ('-d', '--duration'):
            config['duration'] = float(value)
        elif opt in ('-o', '--output'):
            config['output'] = value
        else:
            assert False
            
    # Add in arguments
    config['args'] = args
    
    # Return configuration
    return config


def main(args):
    # Parse command line options
    config = parseOptions(args)
    filenames = config['args'][:2]
    
    # Variable that will be associated with the HDF5 instance
    f = None
    
    # Open the file and load in basic information about the observation's goal
    for c,filename in enumerate(filenames):
        ## Ready the PSRFITS file
        hdulist = pyfits.open(filename, memmap=True)
        
        ## Try to find out the beam/tuning
        mtch = _fnRE.search(filename)
        
        if mtch is None:
            beam = 0
            tuning = 1
        else:
            beam = int(mtch.group('beam'))
            try:
                tuning = int(mtch.group('tuning'))
            except:
                tuning = 1
                
        ## File specifics
        sourceName = hdulist[0].header['SRC_NAME']
        ra = hdulist[0].header['RA']
        ra = ra.split(':', 2)
        ra = sum([float(v)/60**i for i,v in enumerate(ra)])*15.0
        dec = hdulist[0].header['DEC']
        decSign = -1.0 if dec.find('-') != -1 else 1.0
        dec = dec.replace('-', '')
        dec = dec.split(':', 2)
        dec = decSign*sum([float(v)/60**i for i,v in enumerate(dec)])
        epoch = float(hdulist[0].header['EQUINOX'])
        
        tStart = astro.utcjd_to_unix(hdulist[0].header['STT_IMJD'] + (hdulist[0].header['STT_SMJD'] + hdulist[0].header['STT_OFFS'])/86400.0 + astro.MJD_OFFSET)
        cFreq = hdulist[0].header['OBSFREQ']*1e6	# MHz -> Hz
        srate = hdulist[0].header['OBSBW']*1e6		# MHz -> Hz
        LFFT = hdulist[1].header['NCHAN']
        tInt = hdulist[1].header['TBIN']
        nSubs = hdulist[1].header['NSBLK']
        tSubs = nSubs*tInt
        nPol = hdulist[1].header['NPOL']
        if nPol == 1:
            data_products = ['I',]
        elif nPol == 2:
            if hdulist[0].header['FD_POLN'] == 'CIRC':
                data_products = ['LL', 'RR']
            else:
                data_products = ['XX', 'YY']
        else:
            data_products = ['I', 'Q', 'U', 'V']
        nChunks = len(hdulist[1].data)
        
        ## File cross-validation to make sure that everything is in order
        try:
            validationPass = True
            for keyword in ('sourceName', 'ra', 'dec', 'epoch', 'tStart', 'srate', 'LFFT', 'tInt', 'tSubs', 'nPol', 'nChunks'):
                keywordOld = keyword+"Old"
                valid = eval("%s == %s" % (keyword, keywordOld))
                if not valid:
                    print "ERROR:  Detail '%s' of %s does not match that of the first file" % (keyword, os.path.basename(filename))
                    print "ERROR:  Aborting"
                    validationPass = False
                    
            if not validationPass:
                continue
                
        except NameError, e:
            sourceNameOld = sourceName
            raOld = ra
            decOld = dec
            epochOld = epoch
            tStartOld = tStart
            srateOld = srate
            LFFTOld = LFFT
            tIntOld = tInt
            tSubsOld = tSubs
            nPolOld = nPol
            nChunksOld = nChunks
            
        ## Convert the skip and duration values to subblocks
        skip = int(round(config['skip'] / tSubs))
        dur  = int(round(config['duration'] / tSubs))
        dur  = dur if dur else 1
        config['skip'] = skip * tSubs
        config['duration'] = dur * tSubs
        
        ## Report
        print "Filename: %s (%i of %i)" % (filename, c+1, len(filenames))
        print "Date of First Frame: %s" % datetime.utcfromtimestamp(tStart)
        print "Beam: %i" % beam
        print "Tuning: %i" % tuning
        print "Sample Rate: %i Hz" % srate
        print "Tuning Frequency: %.3f Hz" % cFreq
        print "---"
        print "Target: %s" % sourceName
        print "RA: %.3f hours" % (ra/15.0,)
        print "Dec: %.3f degrees" % dec
        print "Data Products: %s" % ','.join(data_products)
        print "Integration Time: %.3f ms" % (tInt*1e3,)
        print "Sub-integrations: %i (%.3f s)" % (nChunks, nChunks*tSubs)
        print "---"
        print "Offset: %.3f s (%i subints.)" % (config['skip'], skip)
        print "Duration: %.3f s (%i subints.)" % (config['duration'], dur)
        print "Transform Length: %i" % LFFT
        
        ## Prepare the HDF5 file
        if f is None:
            ### Create the HDF5 file
            outname = os.path.split(filename)[1]
            outname = os.path.splitext(outname)[0]
            if len(filenames) == 2:
                outname = outname.replace('t1', '')
                outname = outname.replace('t2', '')
            outname = '%s.hdf5' % outname
            
            if os.path.exists(outname):
                yn = raw_input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
                if yn not in ('n', 'N'):
                    os.unlink(outname)
                else:
                    raise RuntimeError("Output file '%s' already exists" % outname)
                    
            ### Populate the groups
            f = hdfData.createNewFile(outname)
            hdfData.fillMinimum(f, 1, beam, srate)
            for t in (1, 2):
                hdfData.createDataSets(f, 1, t, numpy.arange(LFFT, dtype=numpy.float64), dur*nSubs, data_products)
            f.attrs['FileGenerator'] = 'writeHDF5FromPsrfits.py'
            f.attrs['InputData'] = os.path.basename(filename)
            
            ds = {}
            ds['obs1'] = hdfData.getObservationSet(f, 1)
            ds['obs1-time'] = ds['obs1'].create_dataset('time', (dur*nSubs,), 'f8')
            for t in (1, 2):
                ds['obs1-freq%i' % (t,)] = hdfData.get_data_set(f, 1, t, 'freq')
                for p in data_products:
                    ds["obs1-%s%i" % (p, t)] = hdfData.get_data_set(f, 1, t, p)
                    
            ### Add in mask information
            for t in (1, 2):
                tuningInfo = ds["obs1"].get("Tuning%i" % t, None)
                maskInfo = tuningInfo.create_group("Mask")
                for p in data_products:
                    maskInfo.create_dataset(p, ds["obs1-%s%i" % (p, t)].shape, 'bool')
                    ds["obs1-mask-%s%i" % (p, t)] = maskInfo.get(p, None)
                    
            ### Target metadata
            ds['obs1'].attrs['ObservationName'] = sourceName
            ds['obs1'].attrs['TargetName'] = sourceName
            ds['obs1'].attrs['RA'] = ra/15.0
            ds['obs1'].attrs['RA_Units'] = 'hours'
            ds['obs1'].attrs['Dec'] = dec
            ds['obs1'].attrs['Dec_Units'] = 'degrees'
            ds['obs1'].attrs['Epoch'] = epoch
            ds['obs1'].attrs['TrackingMode'] = hdulist[0].header['TRK_MODE']
            
            ### Observation metadata
            ds['obs1'].attrs['tInt'] = tInt
            ds['obs1'].attrs['tInt_Units'] = 's'
            ds['obs1'].attrs['LFFT'] = LFFT
            ds['obs1'].attrs['nchan'] = LFFT
            
            ### Create the progress bar so that we can keep up with the conversion.
            try:
                pbar = progress.ProgressBarPlus(max=len(filenames)*dur)
            except AttributeError:
                pbar = progress.ProgressBar(max=len(filenames)*dur)
                
        ## Frequency information
        freq = hdulist[1].data[0][12]*1e6		# MHz -> Hz
        ds['obs1'].attrs['RBW'] = freq[1]-freq[0]
        ds['obs1'].attrs['RBW_Units'] = 'Hz'
        ds['obs1-freq%i' % (tuning,)][:] = freq
        
        ## Read in the data and apply what ever scaling is needed
        for i in xrange(skip, skip+dur):
            ### Access the correct subintegration
            subint = hdulist[1].data[i]
            
            ### Pull out various bits that we need, including:
            ###  * the start time of the subint. - tOff
            ###  * the weight mask, converted to binary - msk
            ###  * the scale and offset values - bscl and bzero
            ###  * the actual data - data
            tOff = subint[1] - subint[0]/2
            msk = numpy.where(subint[13] >= 0.5, False, True)
            bzero = subint[14]
            bscl = subint[15]
            bzero.shape = (LFFT,nPol)
            bscl.shape = (LFFT,nPol)
            bzero = bzero.T
            bscl = bscl.T
            data = subint[16]
            data.shape = (nSubs,LFFT,nPol)
            data = data.T
            
            ### Apply the scaling/offset to the data and save the results 
            ### to the HDF5 file
            for j in xrange(nSubs):
                k = (i-skip)*nSubs + j
                t = subint[1] + tInt*(j-nSubs/2)
                d = data[:,:,j]*bscl + bzero
                
                if c == 0:
                    ds['obs1-time'][k] = tStart + t
                for l,p in enumerate(data_products):
                    ds['obs1-%s%i' % (p,tuning)][k,:] = d[l,:]
                    ds['obs1-mask-%s%i' % (p,tuning)][k,:] = msk
                    
            ### Update the progress bar and remaining time estimate
            pbar.inc()
            sys.stdout.write('%s\r' % (pbar.show()))
            sys.stdout.flush()
            
        ## Close out the current PSRFITS file
        hdulist.close()
        
        sys.stdout.write(pbar.show()+'\n')
        sys.stdout.flush()
        
    # Done
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    
