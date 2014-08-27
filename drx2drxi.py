#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import ephem
import struct
import getopt
from datetime import datetime

from lsl import astro
from lsl.reader import drx
from lsl.common import progress
from lsl.common.dp import fS


def usage(exitCode=None):
	print """drx2drxi.py - Convert a DRX file into two polarization-interleaved DRX 
files, one for each tuning.

Usage: drx2drxi.py [OPTIONS] file

Options:
-h, --help             Display this help information
-c, --count            Number of seconds to keep
-o, --offset           Number of seconds to skip before splitting
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['offset'] = 0
	config['count'] = 0
	config['date'] = False
	
	# Read in and process the command line flags
	try:
		opts, arg = getopt.getopt(args, "hc:o:", ["help", "count=", "offset="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-c', '--count'):
			config['count'] = float(value)
		elif opt in ('-o', '--offset'):
			config['offset'] = float(value)
		else:
			assert False
			
	# Add in arguments
	config['args'] = arg
	
	# Return configuration
	return config


def main(args):
	config = parseConfig(args)
	
	# Open the file
	fh = open(config['args'][0], "rb")
	sizeB = os.path.getsize(config['args'][0])
	nFramesFile = sizeB / drx.FrameSize
	
	# Find the good data (non-zero decimation)
	while True:
		try:
			junkFrame = drx.readFrame(fh)
			srate = junkFrame.getSampleRate()
			t0 = junkFrame.getTime()
			break
		except ZeroDivisionError:
			pass
	fh.seek(-drx.FrameSize, 1)
	
	# Load in basic information about the data
	junkFrame = drx.readFrame(fh)
	fh.seek(-drx.FrameSize, 1)
	## What's in the data?
	srate = junkFrame.getSampleRate()
	beams = drx.getBeamCount(fh)
	tunepols = drx.getFramesPerObs(fh)
	tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
	beampols = tunepol
	
	# Offset in frames for beampols beam/tuning/pol. sets
	offset = int(round(config['offset'] * srate / 4096 * beampols))
	offset = int(1.0 * offset / beampols) * beampols
	fh.seek(offset*drx.FrameSize, 1)
	
	# Iterate on the offsets until we reach the right point in the file.  This
	# is needed to deal with files that start with only one tuning and/or a 
	# different sample rate.  
	while True:
		## Figure out where in the file we are and what the current tuning/sample 
		## rate is
		junkFrame = drx.readFrame(fh)
		srate = junkFrame.getSampleRate()
		t1 = junkFrame.getTime()
		tunepols = drx.getFramesPerObs(fh)
		tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
		beampols = tunepol
		fh.seek(-drx.FrameSize, 1)
		
		## See how far off the current frame is from the target
		tDiff = t1 - (t0 + config['offset'])
		
		## Half that to come up with a new seek parameter
		tCorr = -tDiff / 2.0
		cOffset = int(tCorr * srate / 4096 * beampols)
		cOffset = int(1.0 * cOffset / beampols) * beampols
		offset += cOffset
		
		## If the offset is zero, we are done.  Otherwise, apply the offset
		## and check the location in the file again/
		if cOffset is 0:
			break
		fh.seek(cOffset*drx.FrameSize, 1)
		
	# Update the offset actually used
	config['offset'] = t1 - t0
	
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
	beampols = tunepol
	
	## Date
	beginDate = ephem.Date(astro.unix_to_utcjd(junkFrame.getTime()) - astro.DJD_OFFSET)
	beginTime = beginDate.datetime()
	mjd = astro.jd_to_mjd(astro.unix_to_utcjd(junkFrame.getTime()))
	mjd_day = int(mjd)
	mjd_sec = (mjd-mjd_day)*86400
	
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
	
	# File summary
	print "Input Filename: %s" % config['args'][0]
	print "Date of First Frame: %s (MJD=%f)" % (str(beginDate),mjd)
	print "Beams: %i" % beams
	print "Tune/Pols: %i %i %i %i" % tunepols
	print "Tunings: %.1f Hz, %.1f Hz" % (centralFreq1, centralFreq2)
	print "Sample Rate: %i Hz" % srate
	print "Frames: %i (%.3f s)" % (nFramesFile, 4096.0*nFramesFile / srate / tunepol)
	
	if config['count'] > 0:
		nCaptures = int(config['count'] * srate / 4096)
	else:
		nCaptures = nFramesFile/beampols
		config['count'] = nCaptures * 4096 / srate
	nSkip = int(config['offset'] * srate / 4096 )
	
	print "Seconds to Skip:  %.2f (%i captures)" % (config['offset'], nSkip)
	print "Seconds to Split: %.2f (%i captures)" % (config['count'], nCaptures)
	
	outname = os.path.basename(config['args'][0])
	outname = os.path.splitext(outname)[0]
	print "Writing %.2f s to file '%s_b%it[12].dat'" % (nCaptures*4096/srate, outname, beam)
	
	# Ready the output files - one for each tune/pol
	fhOut = []
	fhOut.append( open("%s_b%it1.dat" % (outname, beam), 'wb') )
	fhOut.append( open("%s_b%it2.dat" % (outname, beam), 'wb') )
	
	try:
		pb = progress.ProgressBarPlus(max=nCaptures)
	except AttributeError:
		pb = progress.ProgressBar(max=nCaptures)
		
	newFrame = bytearray([0 for i in xrange(32+4096*2)])
	for c in xrange(int(nCaptures)):
		## Create a place to save the frame pairs
		pairs = [[None,None], 
			    [None, None]]
			    
		## Load in the frame pairs
		for i in xrange(tunepol):
			cFrame = fh.read(drx.FrameSize)
			id = struct.unpack('>B', cFrame[4])[0]
			tuning = (id>>3)&7
			pol    = (id>>7)&1
			
			pairs[tuning-1][pol] = cFrame
			
		for tuning in (1, 2):
			### ID manipulation
			idX = struct.unpack('>B', pairs[tuning-1][0][4])[0]
			#idY = struct.unpack('>B', pairs[tuning-1][1][4])[0]
			id = (0<<7) | (1<<6) | (idX&(7<<3)) | (idX&7)
			
			### Time tag manipulation to remove the T_NOM offset
			tNomX, timetagX = struct.unpack('>HQ', pairs[tuning-1][0][14:24])
			#tNomY, timetagX = struct.unpack('>HQ', pairs[tuning-1][1][14:24])
			tNom = tNomX - tNomX
			timetag = timetagX - tNomX
			
			### Build the new frame
			newFrame[0:32] = pairs[tuning-1][0][0:32]
			newFrame[32:8224:2] = pairs[tuning-1][0][32:]
			newFrame[33:8224:2] = pairs[tuning-1][1][32:]
			
			### Update the quatities that have changed
			newFrame[4] = struct.pack('>B', id)
			newFrame[14:24] = struct.pack('>HQ', tNom, timetag)
			
			### Save
			fhOut[tuning-1].write(newFrame)
			
		pb.inc(amount=1)
		if c != 0 and c % 100 == 0:
			sys.stdout.write(pb.show()+'\r')
			sys.stdout.flush()
			
	sys.stdout.write(pb.show()+'\n')
	sys.stdout.flush()
	for f in fhOut:
		f.close()
		
	fh.close()


if __name__ == "__main__":
	main(sys.argv[1:])
