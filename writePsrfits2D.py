#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a DRX file, create one of more PSRFITS file(s).

$Rev: 1317 $
$LastChangedBy: jdowell $
$LastChangedDate: 2013-06-11 14:04:15 -0600 (Tue, 11 Jun 2013) $
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
from lsl.common.dp import fS
from lsl.statistics import kurtosis
from lsl.misc.dedispersion import getCoherentSampleSize

from _psr import *


def usage(exitCode=None):
	print """writePrsfits2D.py - Read in DRX files and create one or more PSRFITS file(s).

Usage: writePsrfits2D.py [OPTIONS] DM file

Options:
-h, --help                  Display this help information
-o, --output                Output file basename
-c, --nchan                 Set FFT length (default = 512)
-p, --no-sk-flagging        Disable on-the-fly SK flagging of RFI
-n, --no-summing            Do not sum polarizations
-i, --circularize           Convert data to RR/LL
-k, --stokes                Convert data to full Stokes
-s, --source                Source name
-r, --ra                    Right Ascension (HH:MM:SS.SS, J2000)
-d, --dec                   Declination (sDD:MM:SS.S, J2000)
-4, --4bit-data             Save the spectra in 4-bit mode (default = 8-bit)
-w, --disable-window        Disable time-domain data windowing

Note:  If a source name is provided and the RA or declination is not, the script
       will attempt to determine these values.
       
Note:  Setting -i/--circularize or -k/--stokes disables polarization summing
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
	config['nchan'] = 512
	config['useSK'] = True
	config['sumPols'] = True
	config['circularize'] = False
	config['stokes'] = False
	config['source'] = None
	config['ra'] = None
	config['dec'] = None
	config['dataBits'] = 8
	config['enableWindow'] = True
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hc:pniks:o:r:d:4w", ["help", "nchan=", "no-sk", "no-summing", "circularize", "stokes", "source=", "output=", "ra=", "dec=", "4bit-mode", "disable-window"])
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
		elif opt in ('-p', '--no-sk-flagging'):
			config['useSK'] = False
		elif opt in ('-n', '--no-summing'):
			config['sumPols'] = False
		elif opt in ('-i', '--circularize'):
			config['sumPols'] = False
			config['circularize'] = True
		elif opt in ('-k', '--stokes'):
			config['stokes'] = True
			config['sumPols'] = False
			config['circularize'] = False
		elif opt in ('-s', '--source'):
			config['source'] = value
		elif opt in ('-r', '--ra'):
			config['ra'] = value
		elif opt in ('-d', '--dec'):
			config['dec'] = value
		elif opt in ('-o', '--output'):
			config['output'] = value
		elif opt in ('-4', '--4bit-mode'):
			config['dataBits'] = 4
		elif opt in ('-w', '--disable-window'):
			config['enableWindow'] = False
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
	
	DM = float(config['args'][0])
	fh = open(config['args'][1], "rb")
	nFramesFile = os.path.getsize(config['args'][1]) / drx.FrameSize
	
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
		config['output'] = "drx_%05d_%s" % (mjd_day, config['source'].replace(' ', ''))
		
	## Tuning frequencies and initial time tags
	ttStep = int(fS / srate * 4096)
	ttFlow = [0, 0, 0, 0]
	for i in xrange(4):
		junkFrame = drx.readFrame(fh)
		beam,tune,pol = junkFrame.parseID()
		aStand = 2*(tune-1) + pol
		ttFlow[aStand] = junkFrame.data.timeTag - ttStep
		
		if tune == 1:
			centralFreq1 = junkFrame.getCentralFreq()
		else:
			centralFreq2 = junkFrame.getCentralFreq()
	fh.seek(-4*drx.FrameSize, 1)
	
	## Coherent Dedispersion Setup
	timesPerFrame = numpy.arange(4096, dtype=numpy.float64)/srate
	spectraFreq1 = numpy.fft.fftshift( numpy.fft.fftfreq(LFFT, d=1.0/srate) ) + centralFreq1
	spectraFreq2 = numpy.fft.fftshift( numpy.fft.fftfreq(LFFT, d=1.0/srate) ) + centralFreq2
	
	# File summary
	print "Input Filename: %s" % config['args'][1]
	print "Date of First Frame: %s (MJD=%f)" % (str(beginDate),mjd)
	print "Beams: %i" % beams
	print "Tune/Pols: %i %i %i %i" % tunepols
	print "Tunings: %.1f Hz, %.1f Hz" % (centralFreq1, centralFreq2)
	print "Sample Rate: %i Hz" % srate
	print "Sample Time: %f s" % (LFFT/srate,)
	print "Frames: %i (%.3f s)" % (nFramesFile, 4096.0*nFramesFile / srate / tunepol)
	print "---"
	print "Using FFTW Wisdom? %s" % useWisdom
	print "DM: %.4f pc / cm^3" % DM
	print "Samples Needed: %i, %i to %i, %i" % (getCoherentSampleSize(centralFreq1-srate/2, 1.0*srate/LFFT, DM), getCoherentSampleSize(centralFreq2-srate/2, 1.0*srate/LFFT, DM), getCoherentSampleSize(centralFreq1+srate/2, 1.0*srate/LFFT, DM), getCoherentSampleSize(centralFreq2+srate/2, 1.0*srate/LFFT, DM))
	
	# Create the output PSRFITS file(s)
	pfu_out = []
	nsblk = 4096
	if config['sumPols']:
		polNames = 'I'
		nPols = 1
		reduceEngine = CombineToIntensity
	elif config['stokes']:
		polNames = 'IQUV'
		nPols = 4
		reduceEngine = CombineToStokes
	elif config['circularize']:
		polNames = 'LLRR'
		nPols = 2
		reduceEngine = CombineToCircular
	else:
		polNames = 'XXYY'
		nPols = 2
		reduceEngine = CombineToLinear
		
	if config['dataBits'] == 4:
		OptimizeDataLevels = OptimizeDataLevels4Bit
	else:
		OptimizeDataLevels = OptimizeDataLevels8Bit
		
	# Parameter validation
	if getCoherentSampleSize(centralFreq1-srate/2, 1.0*srate/LFFT, DM) > nsblk:
		raise RuntimeError("Too few samples for coherent dedispersion.  Considering increasing the number of channels.")
	elif getCoherentSampleSize(centralFreq2-srate/2, 1.0*srate/LFFT, DM)> nsblk:
		raise RuntimeError("Too few samples for coherent dedispersion.  Considering increasing the number of channels.")
		
	# Adjust the time for the padding used for coherent dedispersion
	print "MJD shifted by %.3f ms to account for padding" %  (nsblk*LFFT/srate*1000.0,)
	beginDate = ephem.Date(astro.unix_to_utcjd(junkFrame.getTime() + nsblk*LFFT/srate) - astro.DJD_OFFSET)
	beginTime = beginDate.datetime()
	mjd = astro.jd_to_mjd(astro.unix_to_utcjd(junkFrame.getTime() + nsblk*LFFT/srate))
	
	for t in xrange(1, 2+1):
		## Basic structure and bounds
		pfo = pfu.psrfits()
		pfo.basefilename = "%s_b%it%i" % (config['output'], beam, t)
		pfo.filenum = 0
		pfo.tot_rows = pfo.N = pfo.T = pfo.status = pfo.multifile = 0
		pfo.rows_per_file = 32768
		
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
		pfo.hdr.observer = "writePsrfits2D.py"
		pfo.hdr.source = config['source']
		pfo.hdr.fd_hand = 1
		pfo.hdr.nbits = config['dataBits']
		pfo.hdr.nsblk = nsblk
		pfo.hdr.ds_freq_fact = 1
		pfo.hdr.ds_time_fact = 1
		pfo.hdr.npol = nPols
		pfo.hdr.summed_polns = 1 if config['sumPols'] else 0
		pfo.hdr.obs_mode = "SEARCH"
		pfo.hdr.telescope = "LWA"
		pfo.hdr.frontend = "LWA"
		pfo.hdr.backend = "DRX"
		pfo.hdr.project_id = "Pulsar"
		pfo.hdr.ra_str = config['ra']
		pfo.hdr.dec_str = config['dec']
		pfo.hdr.poln_type = "LIN" if not config['circularize'] else "CIRC"
		pfo.hdr.poln_order = polNames
		pfo.hdr.date_obs = str(beginTime.strftime("%Y-%m-%dT%H:%M:%S"))     
		pfo.hdr.MJD_epoch = pfu.get_ld(mjd)
		
		## Coherent dedispersion information
		pfo.hdr.chan_dm = DM
		
		## Setup the subintegration structure
		pfo.sub.tsubint = pfo.hdr.dt*pfo.hdr.nsblk
		pfo.sub.bytes_per_subint = pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk*pfo.hdr.nbits/8
		pfo.sub.dat_freqs   = pfu.malloc_doublep(pfo.hdr.nchan*8)				# 8-bytes per double @ LFFT channels
		pfo.sub.dat_weights = pfu.malloc_floatp(pfo.hdr.nchan*4)				# 4-bytes per float @ LFFT channels
		pfo.sub.dat_offsets = pfu.malloc_floatp(pfo.hdr.nchan*pfo.hdr.npol*4)		# 4-bytes per float @ LFFT channels per pol.
		pfo.sub.dat_scales  = pfu.malloc_floatp(pfo.hdr.nchan*pfo.hdr.npol*4)		# 4-bytes per float @ LFFT channels per pol.
		if config['dataBits'] == 4:
			pfo.sub.data = pfu.malloc_ucharp(pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk)	# 1-byte per unsigned char @ (LFFT channels x pols. x nsblk sub-integrations) samples
			pfo.sub.rawdata = pfu.malloc_ucharp(pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk/2)	# 4-bits per nibble @ (LFFT channels x pols. x nsblk sub-integrations) samples
		else:
			pfo.sub.rawdata = pfu.malloc_ucharp(pfo.hdr.nchan*pfo.hdr.npol*pfo.hdr.nsblk)	# 1-byte per unsigned char @ (LFFT channels x pols. x nsblk sub-integrations) samples
			
		## Create and save it for later use
		pfu.psrfits_create(pfo)
		pfu_out.append(pfo)
		
	freqBaseMHz = numpy.fft.fftshift( numpy.fft.fftfreq(LFFT, d=1.0/srate) ) / 1e6
	for i in xrange(len(pfu_out)):
		# Define the frequencies available in the file (in MHz)
		pfu.convert2_double_array(pfu_out[i].sub.dat_freqs, freqBaseMHz + pfu_out[i].hdr.fctr, LFFT)
		
		# Define which part of the spectra are good (1) or bad (0).  All channels
		# are good except for the two outermost.
		pfu.convert2_float_array(pfu_out[i].sub.dat_weights, numpy.ones(LFFT),  LFFT)
		pfu.set_float_value(pfu_out[i].sub.dat_weights, 0,      0)
		pfu.set_float_value(pfu_out[i].sub.dat_weights, LFFT-1, 0)
		
		# Define the data scaling (default is a scale of one and an offset of zero)
		pfu.convert2_float_array(pfu_out[i].sub.dat_offsets, numpy.zeros(LFFT*nPols), LFFT*nPols)
		pfu.convert2_float_array(pfu_out[i].sub.dat_scales,  numpy.ones(LFFT*nPols),  LFFT*nPols)
		
	# Speed things along, the data need to be processed in units of 'nsblk'.  
	# Find out how many frames per tuning/polarization that corresponds to.
	chunkSize = nsblk*LFFT/4096
	
	# Evaluate the window function, if needed
	if config['enableWindow']:
		window = numpy.hamming(LFFT)
		fftEngine = lambda x: PulsarEngineRawWindow(x, window, LFFT=LFFT)
	else:
		fftEngine = lambda x: PulsarEngineRaw(x, LFFT=LFFT)
		
	# Calculate the SK limites for weighting
	if config['useSK']:
		skLimits = kurtosis.getLimits(4.0, 1.0*nsblk)
		
		GenerateMask = lambda x: ComputeSKMask(x, skLimits[0], skLimits[1])
	else:
		def GenerateMask(x):
			flag = numpy.ones((4, LFFT), dtype=numpy.float32)
			flag[:,0] = 0.0
			flag[:,-1] = 0.0
			return flag
			
	# Create the progress bar so that we can keep up with the conversion.
	try:
		pbar = progress.ProgressBarPlus(max=nFramesFile/(4*chunkSize)-2, span=55)
	except AttributeError:
		pbar = progress.ProgressBar(max=nFramesFile/(4*chunkSize)-2, span=55)
		
	# Go!
	done = False
	siCount = 0
	
	## Read in the first data block
	dataPrev = numpy.zeros((4, 4096*chunkSize), dtype=numpy.complex64)
	drxtPrev = numpy.zeros(dataPrev.shape, dtype=numpy.float64)
	countPrev = [0 for i in xrange(dataPrev.shape[0])]
	
	for i in xrange(4*chunkSize):
		try:
			frame = drx.readFrame(fh)
		except errors.eofError, errors.syncError:
			done = True
			break
			
		beam,tune,pol = frame.parseID()
		aStand = 2*(tune-1) + pol
		
		if ttFlow[aStand] + ttStep != frame.data.timeTag:
			print 'Warning: Time tag error in subint. %i; %.3f > %.3f + %.3f' % (siCount, frame.data.timeTag/fS, ttFlow[aStand]/fS, ttStep/fS)
		ttFlow[aStand] = frame.data.timeTag
		
		dataPrev[aStand, countPrev[aStand]*4096:(countPrev[aStand]+1)*4096] = frame.data.iq
		drxtPrev[aStand, countPrev[aStand]*4096:(countPrev[aStand]+1)*4096] = frame.getTime() + timesPerFrame
		countPrev[aStand] += 1
	siCount += 1
	
	## Read in the data
	data = numpy.zeros((4, 4096*chunkSize), dtype=numpy.complex64)
	drxt = numpy.zeros(data.shape, dtype=numpy.float64)
	count = [0 for i in xrange(data.shape[0])]
	
	for i in xrange(4*chunkSize):
		try:
			frame = drx.readFrame(fh)
		except errors.eofError, errors.syncError:
			done = True
			break
			
		beam,tune,pol = frame.parseID()
		aStand = 2*(tune-1) + pol
		
		if ttFlow[aStand] + ttStep != frame.data.timeTag:
			print 'Warning: Time tag error in subint. %i; %.3f > %.3f + %.3f' % (siCount, frame.data.timeTag/fS, ttFlow[aStand]/fS, ttStep/fS)
		ttFlow[aStand] = frame.data.timeTag
		
		data[aStand, count[aStand]*4096:(count[aStand]+1)*4096] = frame.data.iq
		drxt[aStand, count[aStand]*4096:(count[aStand]+1)*4096] = frame.getTime() + timesPerFrame
		count[aStand] += 1
	siCount += 1
	
	## FFT
	rawSpectraPrev = fftEngine(dataPrev)
	rawSpectra = fftEngine(data)
	
	dataNext = numpy.zeros((4, 4096*chunkSize), dtype=numpy.complex64)
	drxtNext = numpy.zeros(dataNext.shape, dtype=numpy.float64)
	while True:
		## Read in the data next data block
		countNext = [0 for i in xrange(dataNext.shape[0])]
		
		for i in xrange(4*chunkSize):
			try:
				frame = drx.readFrame(fh)
			except errors.eofError, errors.syncError:
				done = True
				break
				
			beam,tune,pol = frame.parseID()
			aStand = 2*(tune-1) + pol
			
			if ttFlow[aStand] + ttStep != frame.data.timeTag:
				print 'Warning: Time tag error in subint. %i; %.3f > %.3f + %.3f' % (siCount, frame.data.timeTag/fS, ttFlow[aStand]/fS, ttStep/fS)
			ttFlow[aStand] = frame.data.timeTag
			
			dataNext[aStand, countNext[aStand]*4096:(countNext[aStand]+1)*4096] = frame.data.iq
			drxtNext[aStand, countNext[aStand]*4096:(countNext[aStand]+1)*4096] = frame.getTime() + timesPerFrame
			countNext[aStand] += 1
		
		siCount += 1
		
		## Are we done yet?
		if done:
			break
			
		## FFT
		rawSpectraNext = fftEngine(dataNext)
		
		## S-K flagging
		flag = GenerateMask(rawSpectra)
		weight1 = numpy.where( flag[:2,:].sum(axis=0) == 0, 0, 1 ).astype(numpy.float32)
		weight2 = numpy.where( flag[2:,:].sum(axis=0) == 0, 0, 1 ).astype(numpy.float32)
		ff1 = 1.0*(LFFT - weight1.sum()) / LFFT
		ff2 = 1.0*(LFFT - weight2.sum()) / LFFT
		
		## Dedisperse
		drxtOut, rawSpectraDedispersed = MultiChannelCD(drxt[:,::LFFT], rawSpectra, spectraFreq1, spectraFreq2, 1.0*srate/LFFT, DM, 
												drxtPrev[:,::LFFT], rawSpectraPrev, drxtNext[:,::LFFT], rawSpectraNext)
												
		## Update the state variables used to get the CD process continuous
		rawSpectraPrev = rawSpectra
		drxtPrev = drxt
		
		rawSpectra = rawSpectraNext
		drxt = drxtNext
		
		## Detect power
		data = reduceEngine(rawSpectraDedispersed)
		
		## Optimal data scaling
		bzero, bscale, bdata = OptimizeDataLevels(data, LFFT)
		
		## Polarization mangling
		bzero1 = bzero[:nPols,:].T.ravel()
		bzero2 = bzero[nPols:,:].T.ravel()
		bscale1 = bscale[:nPols,:].T.ravel()
		bscale2 = bscale[nPols:,:].T.ravel()
		bdata1 = bdata[:nPols,:].T.ravel()
		bdata2 = bdata[nPols:,:].T.ravel()
		
		## Write the spectra to the PSRFITS files
		for j,sp,bz,bs,wt in zip(range(2), (bdata1, bdata2), (bzero1, bzero2), (bscale1, bscale2), (weight1, weight2)):
			## Time
			pfu_out[j].sub.offs = (pfu_out[j].tot_rows)*pfu_out[j].hdr.nsblk*pfu_out[j].hdr.dt+pfu_out[j].hdr.nsblk*pfu_out[j].hdr.dt/2.0
			
			## Data
			ptr, junk = sp.__array_interface__['data']
			if config['dataBits'] == 4:
				ctypes.memmove(int(pfu_out[j].sub.data), ptr, pfu_out[j].hdr.nchan*nPols*pfu_out[j].hdr.nsblk)
			else:
				ctypes.memmove(int(pfu_out[j].sub.rawdata), ptr, pfu_out[j].hdr.nchan*nPols*pfu_out[j].hdr.nsblk)
				
			## Zero point
			ptr, junk = bz.__array_interface__['data']
			ctypes.memmove(int(pfu_out[j].sub.dat_offsets), ptr, pfu_out[j].hdr.nchan*nPols*4)
			
			## Scale factor
			ptr, junk = bs.__array_interface__['data']
			ctypes.memmove(int(pfu_out[j].sub.dat_scales), ptr, pfu_out[j].hdr.nchan*nPols*4)
			
			## SK
			ptr, junk = wt.__array_interface__['data']
			ctypes.memmove(int(pfu_out[j].sub.dat_weights), ptr, pfu_out[j].hdr.nchan*4)
			
			## Save
			pfu.psrfits_write_subint(pfu_out[j])
			
		## Update the progress bar and remaining time estimate
		pbar.inc()
		sys.stdout.write('%5.1f%% %5.1f%% %s\r' % (ff1*100, ff2*100, pbar.show()))
		sys.stdout.flush()
		
	# Update the progress bar with the total time used
	sys.stdout.write('              %s\n' % pbar.show())
	sys.stdout.flush()


if __name__ == "__main__":
	main(sys.argv[1:])
	
