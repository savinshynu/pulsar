#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a DRX file, create one of more PSRFITS file(s).

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import numpy
import ephem
import ctypes
import getopt

import psrfits_utils.psrfits_utils as pfu

import lsl.reader.drx as drx
import lsl.reader.errors as errors
import lsl.astro as astro
import lsl.common.progress as progress

from _psr import PulsarEngine, SumPolsWithOverflow


def usage(exitCode=None):
	print """writePrsfits2.py - Read in DRX files and create one or more PSRFITS file(s).

Usage: writePsrfits2.py [OPTIONS] file

Options:
-h, --help                  Display this help information
-o, --output                Output file basename
-c, --nchan                 Set FFT length (default = 4096)
-n, --no-summing            Do not sum polarizations
-i, --circularize           Convert data to RR/LL
-s, --source                Source name
-r, --ra                    Right Ascension (HH:MM:SS.SS, J2000)
-d, --dec                   Declination (sDD:MM:SS.S, J2000)

Note:  If a source name is provided and the RA or declination is not, the script
       will attempt to determine these values.
"""

	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseOptions(args):
	config = {}
	# Command line flags - default values
	config['output'] = None
	config['args'] = []
	config['nchan'] = 4096
	config['sumPols'] = True
	config['circularize'] = False
	config['source'] = None
	config['ra'] = None
	config['dec'] = None
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hc:nsi:o:r:d:", ["help", "nchan=", "no-summing", "circularize", "source=", "output=", "ra=", "dec="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-c', '--nchan'):
			config['nchan'] = int(value)
		elif opt in ('-n', '--no-summing'):
			config['sumPols'] = False
		elif opt in ('-i', '--circularize'):
			config['circularize'] = True
		elif opt in ('-s', '--source'):
			config['source'] = value
		elif opt in ('-r', '--ra'):
			config['ra'] = value
		elif opt in ('-d', '--dec'):
			config['dec'] = value
		elif opt in ('-o', '--output'):
			config['output'] = value
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Return configuration
	return config


def resolveTarget(name):
	import urllib
	
	try:
		result = urllib.urlopen('http://www3.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/NameResolver/find?target=%s' % urllib.quote_plus(name))
		
		line = result.readlines()
		target = (line[0].replace('\n', '').split('='))[1]
		service = (line[1].replace('\n', '').split('='))[1]
		service = service.replace('(', ' @ ')
		coordsys = (line[2].replace('\n', '').split('='))[1]
		ra = (line[3].replace('\n', '').split('='))[1]
		dec = (line[4].replace('\n', '').split('='))[1]
		
		temp = astro.deg_to_hms(float(ra))
		raS = "%i:%02i:%05.2f" % (temp.hours, temp.minutes, temp.seconds)
		temp = astro.deg_to_dms(float(dec))
		decS = "%+i:%02i:%04.1f" % ((-1.0 if temp.neg else 1.0)*temp.degrees, temp.minutes, temp.seconds)
		serviceS = service[0:-2]
		
	except (IOError, ValueError):
		raS = "---"
		decS = "---"
		serviceS = "Error"
		
	return raS, decS, serviceS


def main(args):
	# Parse command line options
	config = parseOptions(args)
	
	# Find out where the source is if needed
	if config['source'] is not None:
		if config['ra'] is None or config['dec'] is None:
			tempRA, tempDec, tempService = resolveTarget('PSR '+config['source'])
			print "%s resolved to %s, %s using '%s'" % (config['source'], tempRA, tempDec, tempService)
			out = raw_input('=> Accept? [Y/n] ')
			if out == 'n' or out == 'N':
				sys.exit()
			else:
				config['ra'] = tempRA
				config['dec'] = tempDec
				
	else:
		config['source'] = "None"
		
	if config['ra'] is None:
		config['ra'] = "00:00:00.00"
	if config['dec'] is None:
		config['dec'] = "+00:00:00.0"
		
	# FFT length
	LFFT = config['nchan']
	
	fh = open(config['args'][0], "rb")
	nFramesFile = os.path.getsize(config['args'][0]) / drx.FrameSize
	
	# Find the good data (non-zero decimation)
	while True:
		try:
			junkFrame = drx.readFrame(fh)
			srate = junkFrame.getSampleRate()
			break
		except ZeroDivisionError:
			pass
	fh.seek(-drx.FrameSize, 1)
	
	# Line up the time tags for the various tunings/polarizations
	timeTags = []
	for i in xrange(16):
		junkFrame = drx.readFrame(fh)
		timeTags.append(junkFrame.data.timeTag)
	fh.seek(-16*drx.FrameSize, 1)
	
	i = 0
	while (timeTags[i+0] != timeTags[i+1]) or (timeTags[i+0] != timeTags[i+2]) or (timeTags[i+0] != timeTags[i+3]):
		i += 1
		fh.seek(drx.FrameSize, 1)
		
	# Load in basic information about the data
	junkFrame = drx.readFrame(fh)
	fh.seek(-drx.FrameSize, 1)
	## What's in the data?
	srate = junkFrame.getSampleRate()
	beams = drx.getBeamCount(fh)
	tunepols = drx.getFramesPerObs(fh)
	tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
	
	## Date
	beginDate = ephem.Date(astro.unix_to_utcjd(junkFrame.getTime()) - astro.DJD_OFFSET)
	beginTime = beginDate.datetime()
	mjd = astro.jd_to_mjd(astro.unix_to_utcjd(junkFrame.getTime()))
	mjd_day = int(mjd)
	mjd_sec = (mjd-mjd_day)*86400
	if config['output'] is None:
		config['output'] = "drx_%05d_%05d" % (mjd_day, int(mjd_sec))
		
	## Tuning frequencies
	for i in xrange(4):
		junkFrame = drx.readFrame(fh)
		beam,tune,pol = junkFrame.parseID()
		if tune == 1:
			centralFreq1 = junkFrame.getCentralFreq()
		else:
			centralFreq2 = junkFrame.getCentralFreq()
	fh.seek(-4*drx.FrameSize, 1)
	
	# File summary
	print "Input Filename: %s" % config['args'][0]
	print "Date of First Frame: %s (MJD=%f)" % (str(beginDate),mjd)
	print "Beams: %i" % beams
	print "Tune/Pols: %i %i %i %i" % tunepols
	print "Sample Rate: %i Hz" % srate
	print "Sample Time: %f s" % (LFFT/srate,)
	print "Frames: %i (%.3f s)" % (nFramesFile, 4096.0*nFramesFile / srate / tunepol)
	print "---"
	
	# Create the output PSRFITS file(s)
	pfu_out = []
	nsblk = 4096
	if config['sumPols']:
		pols = [-1,]
	else:
		pols = [0, 1]
		
	for t in xrange(1, 2+1):
		for p in pols:
			## Basic structure and bounds
			pfo = pfu.psrfits()
			if p == -1:
				pfo.basefilename = "%s_b%it%i" % (config['output'], beam, t)
			else:
				pfo.basefilename = "%s_b%it%ip%i" % (config['output'], beam, t, p)
			pfo.filenum = 0
			pfo.tot_rows = pfo.N = pfo.T = pfo.status = pfo.multifile = 0
			pfo.rows_per_file = 8192
			
			## Frequency, bandwidth, and channels
			if t == 1:
				pfo.hdr.fctr=centralFreq1/1e6
			else:
				pfo.hdr.fctr=centralFreq2/1e6
			pfo.hdr.BW = srate/1e6
			pfo.hdr.nchan = LFFT
			pfo.hdr.df = srate/1e6/LFFT
			pfo.hdr.dt = LFFT / srate
			
			## Metadata about the observation/observatory/pulsar
			pfo.hdr.observer = "writePsrfits2.py"
			pfo.hdr.source = config['source']
			pfo.hdr.fd_hand = 1
			pfo.hdr.nbits = 8
			pfo.hdr.nsblk = nsblk
			pfo.hdr.ds_freq_fact = 1
			pfo.hdr.ds_time_fact = 1
			pfo.hdr.npol = 1
			pfo.hdr.summed_polns = 1 if config['sumPols'] else 0
			pfo.hdr.obs_mode = "SEARCH"
			pfo.hdr.telescope = "LWA"
			pfo.hdr.frontend = "LWA"
			pfo.hdr.backend = "DRX"
			pfo.hdr.project_id = "Pulsar"
			pfo.hdr.ra_str = config['ra']
			pfo.hdr.dec_str = config['dec']
			pfo.hdr.poln_type = "LIN" if not config['circularize'] else "CIRC"
			pfo.hdr.date_obs = str(beginTime.strftime("%Y-%m-%dT%H:%M:%S"))     
			pfo.hdr.MJD_epoch = pfu.get_ld(mjd)
			
			## Setup the subintegration structure
			pfo.sub.bytes_per_subint = pfo.hdr.nchan*pfo.hdr.nsblk*pfo.hdr.nbits/8
			pfo.sub.dat_freqs = pfu.malloc_floatp(pfo.hdr.nchan*4)		# 4-bytes per float @ LFFT channels
			pfo.sub.dat_weights = pfu.malloc_floatp(pfo.hdr.nchan*4)	# 4-bytes per float @ LFFT channels
			pfo.sub.dat_offsets = pfu.malloc_floatp(pfo.hdr.nchan*4)	# 4-bytes per float @ LFFT channels
			pfo.sub.dat_scales = pfu.malloc_floatp(pfo.hdr.nchan*4)	# 4-bytes per float @ LFFT channels
			pfo.sub.rawdata = pfu.malloc_ucharp(pfo.hdr.nchan*pfo.hdr.nsblk)	# 1-byte per unsigned char @ (LFFT channels x nsblk sub-integrations) samples
			
			## Create and save it for later use
			pfu.psrfits_create(pfo)
			pfu_out.append(pfo)
			
	freqBaseMHz = numpy.fft.fftshift( numpy.fft.fftfreq(LFFT, d=1/srate) ) / 1e6
	for i in xrange(len(pfu_out)):
		# Define the frequencies available in the file (in MHz)
		pfu.convert2_float_array(pfu_out[i].sub.dat_freqs, freqBaseMHz + pfu_out[i].hdr.fctr, LFFT)
		
		# Define which part of the spectra are good (1) or bad (0).  All channels
		# are good except for the two outermost.
		pfu.convert2_float_array(pfu_out[i].sub.dat_weights, numpy.ones(LFFT),  LFFT)
		pfu.set_float_value(pfu_out[i].sub.dat_weights, 0,      0)
		pfu.set_float_value(pfu_out[i].sub.dat_weights, LFFT-1, 0)
		
		# Define the data scaling (default is a scale of one and an offset of zero)
		pfu.convert2_float_array(pfu_out[i].sub.dat_offsets, numpy.zeros(LFFT), LFFT)
		pfu.convert2_float_array(pfu_out[i].sub.dat_scales,  numpy.ones(LFFT),  LFFT)
		
	# Speed things along, the data need to be processed in units of 'nsblk'.  
	# Find out how many frames per tuning/polarization that corresponds to.
	chunkSize = nsblk*LFFT/4096
	
	# Create the progress bar so that we can keep up with the conversion.
	try:
		pbar = progress.ProgressBarPlus(max=nFramesFile/(4*chunkSize))
	except AttributeError:
		pbar = progress.ProgressBar(max=nFramesFile/(4*chunkSize))
		
	# Go!
	done = False
	while True:
		## Read in the data
		data = numpy.zeros((4, 4096*chunkSize), dtype=numpy.complex64)
		count = [0 for i in xrange(data.shape[0])]
		
		for i in xrange(4*chunkSize):
			try:
				frame = drx.readFrame(fh)
			except errors.eofError, errors.syncError:
				done = True
				break
				
			beam,tune,pol = frame.parseID()
			aStand = 2*(tune-1) + pol
			
			data[aStand, count[aStand]*4096:(count[aStand]+1)*4096] = frame.data.iq
			count[aStand] += 1
			
		## Are we done yet?
		if done:
			break
			
		## Circularize, if needed
		if config['circularize']:
			l1 = (data[0,:] + j*data[1,:]) / numpy.sqrt(2)
			r1 = (data[0,:] - j*data[1,:]) / numpy.sqrt(2)
			l2 = (data[2,:] + j*data[3,:]) / numpy.sqrt(2)
			r2 = (data[2,:] - j*data[3,:]) / numpy.sqrt(2)
			
			data[0,:] = l1
			data[1,:] = r1
			data[2,:] = l2
			data[3,:] = r2
			
		## FFT, detect power, and combine the polarizations
		data = PulsarEngine(data, LFFT)
		
		if config['sumPols']:
			data1 = SumPolsWithOverflow(data, 0, 1)
			data2 = SumPolsWithOverflow(data, 2, 3)
		else:
			data1 = data[0,:]
			data2 = data[1,:]
			data3 = data[2,:]
			data4 = data[3,:]
			
		## Write the spectra to the PSRFITS files
		if config['sumPols']:
			for j,spec in zip(range(2), (data1, data2)):
				pfu_out[j].sub.offs = (pfu_out[j].tot_rows)*pfu_out[j].hdr.nsblk*pfu_out[j].hdr.dt
				ptr, junk = spec.__array_interface__['data']
				ctypes.memmove(int(pfu_out[j].sub.rawdata), ptr, pfu_out[j].hdr.nchan*pfu_out[j].hdr.nsblk)
				pfu.psrfits_write_subint(pfu_out[j])
				
		else:
			for j,spec in zip(range(4), (data1, data2, data3, data4)):
				pfu_out[j].sub.offs = (pfu_out[j].tot_rows)*pfu_out[j].hdr.nsblk*pfu_out[j].hdr.dt
				ptr, junk = spec.__array_interface__['data']
				ctypes.memmove(int(pfu_out[j].sub.rawdata), ptr, pfu_out[j].hdr.nchan*pfu_out[j].hdr.nsblk)
				pfu.psrfits_write_subint(pfu_out[j])
				
		## Update the progress bar and remaining time estimate
		pbar.inc()
		sys.stdout.write('%s\r' % pbar.show())
		sys.stdout.flush()
		
	# Update the progress bar with the total time used
	sys.stdout.write('%s\n' % pbar.show())
	sys.stdout.flush()


if __name__ == "__main__":
	main(sys.argv[1:])
	
