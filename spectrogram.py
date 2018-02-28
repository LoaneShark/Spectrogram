# -*- coding: utf-8 -*-
# Spectrogram analysis of .wav files
# Santiago Loane
# PHY 256: Final Project
# Presenting 4/26/17

import numpy as np
import numpy.fft as fft
import scipy
import wave as wv
import wavio
import matplotlib.pyplot as plt

# NOTE: Only guaranteed to work for uncompressed WAV files, with int data points
# remember to install wavio package, only nonstandard dependancy
# (this can be done by typing <pip install wavio> into a command terminal with python installed)

# The program enclosed in this file does the following:
# - Reads in a .wav file, as specified by the file titles found in the main() function
# - Extracts the similarly specified segments of time as frames of audio data
# - Plots the waveform of the aforementioned series of audio frames
# - Computes the spectrogram of the above segments of time
# - Plots the spectrogram of the above segments of time

# main function, edit things here to change which sound files/sections are used
# works for stereo wav files (stereo will only plot right channel) 
#   (could work with mono files, but only tested on stereo)
# not guaranteed to work for 24-bit wav files (but it might)
def main():

	songs = ['Beethovens_5th','Loud_Pipes','YYZ','Dolphins','Equation']
	times = [[0,10],[0,10],[0,10],[0,10],[325,340]]
	for song, Trange in zip(songs,times):
		start = Trange[0]
		end = Trange[1]
		data, params, timerange = read_song(song, T1=start, T2=end)
		plot_song(data, song, params)
		analysis(data, params, song, timerange)

# Given file title, reads in a .wav file and extracts data
# title must be file title, e.g. for "test.wav" title would be "test"
# returns data as 2d array and tuple of important parameters
def read_song(title, T1=0, T2='all'):

	w = wv.open('%s.wav' %title,'r')
	
	channels = w.getnchannels() 	# 1 for mono, 2 for stereo
	samplew  = w.getsampwidth() 	# sample width (in bytes)
	samplef  = w.getframerate() 	# sampling frequency
	nframes  = w.getnframes() 		# number of audio frames
	comptype = w.getcomptype() 		# compression type (unused)
	compname = w.getcompname() 		# compression name (unused)
	sampleln = 1.0/float(samplef) 			# amount of time in s per sample
	params = (channels,samplew,samplef,nframes) 	# tuple of important parameters
	

	# reads all the frames of audio, as a string of bytes
	songstring = w.readframes(nframes)
	w.close() 

	#print(type(songstring))
	#print(len(songstring))
	#print(comptype)
	#print(compname)
	#print(channels)
	#print(samplew)
	#print(samplef)
	#print(nframes*1.0/(samplew*channels))
	#print(len(songstring)*1.0/(samplew*1))

	# wavio package can convert a .wav file data (bytestrings) to 
	# an array (size = [nframes,nchannels]) of ints
	# this makes it much easier to work with, and I can feed the numerical
	# data into a fft. 
	data_array = wavio._wav2array(channels,samplew,songstring)
	# isolate left and right channels (unused)
	# left = data_array[:,0]
	# right = data_array[:,1]

	if (T2 == 'all'):
		return data_array[int(T1*samplef):], params, (T1,nframes*samplef)
	else:
		return data_array[int(np.ceil(T1*samplef)):int(T2*samplef)], params, (T1,T2)



# write song to output, using wave and wavio packages
# data: 2d array of songdata
# title: string to title .wav file ("title.wav")
# params: tuple of parameters (channels, samplewidth, samplefrequency)
def write_song(data,title,params):
	nframes = len(data)
	
	# write song to output!
	beethovenw = wv.open('%s.wav' %title,'w')
	# set parameters to be the same as that of original song
	# (no mapping of sample rates, changing channels, etc.)

	beethovenw.setnchannels(channels)	# 1 for mono, 2 for stereo
	beethovenw.setsampwidth(samplew)  	# sample width (in bytes)
	beethovenw.setframerate(samplef)  	# sampling frequency
	beethovenw.setnframes(nframes)  	# number of audio frames
	
	beethovenw.writeframes(wavio._array2wav(data,samplew))
	beethovenw.close()

# plots the .wav file over T (in seconds) time
def plot_song(data, title, params):
	channels, samplew, samplef, nframes = params
	numframes = len(data)

	tvals = []
	for f in range(0, numframes):
		tvals.append(float(f)/samplef)

	if (channels == 2):
		plt.plot(tvals,data[:numframes,1],'k',label='left')
		plt.plot(tvals,data[:numframes,0],'b',label='right')
	else:
		plt.plot(tvals,data[:numframes],'g',label='mono')

	plt.xlabel("t (sec)")
	plt.title(title)
	plt.legend(loc='best')
	plt.show()

# perform fourier analysis on data
def analysis(data, params, title, times):
	channels, samplew, samplef, nframes = params
	dt = 1.0/float(samplef)

	transform_left,transform_right,frequencies = sftf(data,dt)

	plot_frequencies(frequencies, transform_right, params, dt, times, title)

# perform Short-Time Fourier Transform on stereo data
# data: the 2D array containing the data of the audio file
# dt: length of audio frames/timestep size (in s)
# size: width of window for use in STFT (in frames)
# overlap: number of windows to overlap (avoids artifacting)
def sftf(data, dt, size=2048, overlap=4):
	channels = len(data[0])
	stepsize = size/overlap
	window = scipy.hanning(size+1)[:-1]
	# compile array where each "item" (array) is the real FFT performed on the window-modified data, for each "bin" of data 
	result_l = np.array([np.fft.rfft(window*data[i:i+size,0]) for i in range(0,len(data)-size, stepsize)])
	if (channels == 2):
		result_r = np.array([np.fft.rfft(window*data[i:i+size,1]) for i in range(0,len(data)-size, stepsize)])
	else:
		result_r = result_l
	freqs_l = np.fft.rfftfreq((window*data[i:i+size,0]).size, dt)
	return result_l, result_r, freqs_l

# Plots the spectrogram of the data
# frequencies: Frequencies to be plotted. Current implementation ignores this.
# analysis: Result of STFT (2D complex array)
# params: parameters of song
# dt: size of timesteps (in seconds)
# times: tuple with start and end times (in s)
# title: string representing title of song (to be displayed on spectrogram)
def plot_frequencies(frequencies, analysis, params, dt, times, title):
	channels, samplew, samplef, nframes = params
	T1, T2 = times
	intensities = analysis.T
	T = nframes/samplef
	timesteps = np.arange(T1,T2,dt)

	# Spectrogram = |STFT|^2
	C = np.absolute(intensities)**2
	# normalize array for conversion to dB
	# (don't want 0 values or log will throw error)
	minval = np.min(C[np.nonzero(C)])
	C[C == 0] = minval
	# convert intensity to decibels
	C_norm = (1.0/np.max(C)) * C
	C_dB = 10*np.log10(C_norm)

	# plot specrogram
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.xlabel('Timesteps (bins) [dt = %.3e s]*4096 from T = [%d,%d] (s)' %(dt,T1,T2-1))
	plt.ylabel('Frequency (daHz) [%.1e-22050]' %minval)
	plt.semilogy(1)
	ax.set_yscale('log')
	# use image generator rather than plotting point by point as it is much faster
	# (and also looks better)
	im = ax.imshow(C_dB, origin='lower',cmap='plasma')
	if (channels == 2):
		channel = 'right'
	else:
		channel = 'mono'
	plt.title('Spectrogram of %s.wav (%s audio, dB scale)' %(title,channel))
	plt.show()

# debug functions that can be called instead of main() (not in use)
# (only plot Equation or Dolphin calls, respectively)
def equation():
	data, params, timerange = read_song('Equation',T1=325,T2=340)
	analysis(data, params, 'Equation - Aphex Twin', timerange)
def dolphin():
	data, params, timerange = read_song('Dolphins',T1=0,T2=10)
	plot_song(data, 'Dolphins', params)
	analysis(data, params, 'Dolphin Chirps', timerange)

if __name__ == "__main__":
	main()
	#equation()
	#dolphin()
