#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy
import getopt
import pyfits

import lsl.common.progress as progress
from lsl.statistics import robust, kurtosis


def usage(exitCode=None):
	print """updatePsrfitsMask.py - Read in a PSRFITS file and update the mask to exclude
frequencies and/or time windows using a frequency mask and (pseudo) spectral 
kurtosis.

Usage: updatePsrfitsMask.py [OPTIONS] file

Options:
-h, --help                Display this help information
-s, --sk-sigma            (p)SK masking limit in sigma (default = 4)
-f, --frequencies         Comma seperated list of frequency to mask in MHz
-d, --duration            (p)SK update interval (default = 10 s)
-r, --replace             Replace the current weight mask rather than 
                          augment it (default = augment)
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseOptions(args):
	config = {}
	# Command line flags - default values
	config['skSigma'] = 4.0
	config['duration'] = 10.0
	config['frequencies'] = []
	config['replace'] = False
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hs:d:f:r", ["help", "sk-sigma=", "duration=", "frequencies=", "replace"])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-s', '--sk-sigma'):
			config['skSigma'] = float(value)
		elif opt in ('-d', '--duration'):
			config['duration'] = float(value)
		elif opt in ('-f', '--frequencies'):
			values = value.split(',')
			for v in values:
				if v.find('-') == -1:
					config['frequencies'].append( float(v) )
				else:
					v1, v2 = [float(vs) for vs in v.split('-', 1)]
					v = v1
					while v <= v2:
						config['frequencies'].append( v )
						v += 0.1
					config['frequencies'].append( v2 )
					
		elif opt in ('-r', '--replace'):
			config['replace'] = True
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Return configuration
	return config


def main(args):
	config = parseOptions(args)
	
	filenames = config['args']
	
	for filename in filenames:
		print "Working on '%s'..." % os.path.basename(filename)
		
		# Open the PRSFITS file
		hdulist = pyfits.open(filename, mode='update', memmap=True)
		
		# Figure out the integration time per sub-integration so we know how 
		# many sections to work with at a time
		nPol = hdulist[1].header['NPOL']
		nSubs = hdulist[1].header['NSBLK']
		tInt = hdulist[1].data[0][0]
		nSubsChunk = int( numpy.ceil( config['duration']/tInt ) )
		print "  Polarizations: %i" % nPol
		print "  Sub-integration time: %.3f ms" % (tInt/nSubs*1000.0,)
		print "  Sub-integrations per block: %i" % nSubs
		print "  Block integration time: %.3f ms" % (tInt*1000.0,)
		print "  Working in chunks of %i blocks (%.3f s)" % (nSubsChunk, nSubsChunk*tInt)
		
		# Figure out the SK parameters to use
		srate = hdulist[0].header['OBSBW']*1e6
		LFFT = hdulist[1].data[0][12].size
		skM = nSubsChunk*nSubs
		skN = srate / LFFT * (tInt / nSubs)
		if nPol == 1:
			skN *= 2
		skLimits = kurtosis.getLimits(config['skSigma'], skM, N=1.0*skN)
		print "  (p)SK M: %i" % (nSubsChunk*nSubs,)
		print "  (p)SK N: %i" % skN
		print "  (p)SK Limits: %.4f <= valid <= %.4f" % skLimits
		
		# Figure out what to mask for the specified frequencies and report
		toMask = []
		freq = hdulist[1].data[0][12]
		for f in config['frequencies']:
			metric = numpy.abs( freq - f )
			toMaskCurrent = numpy.where( metric <= 0.05 )[0]
			toMask.extend( list(toMaskCurrent) )
		if len(toMask) > 0:
			toMask = list(set(toMask))
			toMask.sort()
			print "  Masking Channels:"
			for c in toMask:
				print "    %i -> %.3f MHz" % (c, freq[c])
				
		# Setup the progress bar
		try:
			pbar = progress.ProgressBarPlus(max=len(hdulist[1].data)/nSubsChunk, span=58)
		except AttributeError:
			pbar = progress.ProgressBar(max=len(hdulist[1].data)/nSubsChunk, span=58)
			
		# Go!
		flagged = 0
		processed = 0
		sk = numpy.zeros((nPol, LFFT)) - 99.99
		for i in xrange(0, (len(hdulist[1].data)/nSubsChunk)*nSubsChunk, nSubsChunk):
			## Load in the current block of data
			blockData = []
			blockMask = None
			for j in xrange(i, i+nSubsChunk):
				### Access the correct subintegration
				subint = hdulist[1].data[j]
				
				### Pull out various bits that we need, including:
				###  * the weight mask
				###  * the scale and offset values - bscl and bzero
				###  * the actual data - data
				msk = subint[13]
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
				### to blockData
				for k in xrange(nSubs):
					d = data[:,:,k]*bscl + bzero
					d.shape += (1,)
					blockData.append( d )
					
				### Save a master mask
				try:
					blockMask *= msk
				except TypeError:
					blockMask = msk
					
			blockData = numpy.concatenate(blockData, axis=2)
			
			## Compute the S-K statistics
			for p in xrange(nPol):
				for l in xrange(LFFT):
					sk[p,l] = kurtosis.spectralPower(blockData[p,l,:], N=1.0*skN)
					
			## Compute the new mask - both SK and the frequency flagging
			newMask = numpy.where( (sk < skLimits[0]) | (sk > skLimits[1]), 0.0, 1.0 )
			newMask = numpy.where( newMask.mean(axis=0) <= 0.5, 0.0, 1.0 )
			for c in toMask:
				newMask[c] *= 0.0
				
			if config['replace']:
				## Replace the existing mask
				blockMask = newMask
			else:
				## Update the existing mask
				blockMask *= newMask
				
			## Update file
			for j in xrange(i, i+nSubsChunk):
				hdulist[1].data[j][13] = blockMask
				
			## Update the counters
			processed += LFFT
			flagged += (1.0-blockMask).sum()
			
			## Update the progress bar and remaining time estimate
			pbar.inc()
			sys.stdout.write('  %5.1f%% %s\r' % (100.0*(1.0-blockMask).sum()/LFFT, pbar.show()))
			sys.stdout.flush()
			
		# Update the progress bar with the total time used
		sys.stdout.write('  %5.1f%% %s\n' % (100.0*flagged/processed, pbar.show()))
		sys.stdout.flush()
		
		# Done
		hdulist.close()


if __name__ == "__main__":
	main(sys.argv[1:])
	