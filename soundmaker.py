# -*- coding: utf-8 -*-
# Neural Net to compose music
# Santiago Loane
# PHY 256: Final Project
# Due 4/10/17

import numpy as np
import numpy.fft as fft
import scipy
import wave as wv
import wavio
import struct
import matplotlib.pyplot as plt

# gonna wanna compute fourier transform for sections of song
# (every frame? probably too many//too short of a timescale)
# look into Gabor transform
# can this be used to then write music afterwards?

# NOTE: Only works for uncompressed WAV files, with int data points

# TODO: Test & Fix Spectrograms (compare to scipy's built-in function?)
# 		Plot both L and R transforms (or combine them)
# 		Play song & show tracker with spectrogram when displayed
#		Fix aspect ratio issues in display
# 		Maybe make display better with subplots and stuff
# 		Finish presentation 

def main():

	songs = ['Beethovens_5th','Loud_Pipes','YYZ','Equation']
	times = [[0,10],[0,10],[0,10],[325,340]]
	#songs = ['Beethovens_5th']
	for song, Trange in zip(songs,times):
		start = Trange[0]
		end = Trange[1]
		data, params, timerange = read_song(song, T1=start, T2=end)
		#print("array length: %s" %len(data))
		plot_song(data, song, params)
		analysis(data, params, song, timerange)

def equation():
	data, params, timerange = read_song('Equation',T1=325,T2=340)
	analysis(data, params, 'Equation - Aphex Twin', timerange)

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
	left = data_array[:,0]
	right = data_array[:,1]

	if (T2 == 'all'):
		return data_array[int(T1*samplef):], params
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
	beethovenw.setnframes(nframes)  		# number of audio frames
	
	beethovenw.writeframes(wavio._array2wav(data,samplew))
	beethovenw.close()

# plots the .wav file over T (in seconds) time
def plot_song(data, title, params):
	channels, samplew, samplef, nframes = params
	numframes = len(data)

	tvals = []
	for f in range(0, numframes):
		tvals.append(float(f)/samplef)
	plt.plot(tvals,data[:numframes,1],'k',label='left')
	plt.plot(tvals,data[:numframes,0],'b',label='right')
	plt.xlabel("t (sec)")
	plt.title(title)
	plt.legend(loc='best')
	plt.show()

# perform fourier analysis on data
def analysis(data, params, title, times):
	channels, samplew, samplef, nframes = params
	dt = 1.0/float(samplef)

	transform_left,transform_right,frequencies = sftf(data,dt)
	#spectrogram_l = np.absolute(transform_left)**2
	#spectrogram_r = np.absolute(transform_right)**2

	plot_frequencies(frequencies, transform_right, params, dt, times, title)
	#frequencies = np.fft.fftfreq(data.size,dt)
	#i = np.argsort(frequencies)

	#plt.plot(frequencies[i],transform_left[i])
	#plt.show()

# perform Short-Time Fourier Transform on stereo data
def sftf(data, dt, size=4096, overlap=4):
	stepsize = size/overlap
	window = scipy.hanning(size+1)[:-1]
	result_l = np.array([np.fft.rfft(window*data[i:i+size,0]) for i in range(0,len(data)-size, stepsize)])
	result_r = np.array([np.fft.rfft(window*data[i:i+size,1]) for i in range(0,len(data)-size, stepsize)])
	freqs_l = np.fft.rfftfreq((window*data[i:i+size,0]).size, dt)
	return result_l, result_r, freqs_l

# Plots the spectrogram of the data
# frequencies: Frequencies to be plotted. Current implementation ignores this.
# analysis: Result of STFT (2D complex array)
# params: parameters of song
# dt: size of timesteps (in seconds)
# title: string representing title of song (to be displayed on spectrogram)
def plot_frequencies(frequencies, analysis, params, dt, times, title):
	channels, samplew, samplef, nframes = params
	T1, T2 = times
	intensities = analysis.T
	T = nframes/samplef
	timesteps = np.arange(T1,T2,dt)


	#C = intensities.astype(np.int64)
	#X = timesteps.astype(np.int64)
	#Y = frequencies.astype(np.int64)

	# Spectrogram = |STFT|^2
	C = np.absolute(intensities)**2
	# normalize array for conversion to dB
	# (don't want 0 values or log will throw error)
	minval = np.min(C[np.nonzero(C)])
	C[C == 0] = minval
	# convert intensity to decibels
	C_norm = (1.0/np.max(C)) * C
	C_dB = 10*np.log10(C_norm)
	# Not used in current implementation
	X = timesteps
	Y = frequencies

	#print("Title: %s" %title)
	#print("X, Y, C dtypes: %s" %[arr.dtype for arr in [X,Y,C]])
	#print("X, Y ranges: %s" %[arr.max() for arr in [X,Y]])
	#print("X.shape: %s\nY.shape: %s\nC.shape: %s" %(X.shape,Y.shape,C.shape))

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.xlabel('Timesteps [dt = %.3e s] from T = [%d,%d] (s)' %(dt,T1,T2-1))
	plt.ylabel('Frequency (Hz)')
	plt.semilogy(10)
	#ax.set_xticks(np.arange(0,len(X),int(1/dt)))
	#ax.set_xticklabels(np.arange(T1,T2+1,1), rotation=45)
	ax.set_yscale('log')
	#plt.ylim(0,1500)
	#plt.yticks(np.arange(0,1500,150),np.arange(0,1500,150))
	im = ax.imshow(C_dB, origin='lower',cmap='plasma')
	plt.title('Spectrogram of %s.wav' %title)
	#plt.legend(loc='best')
	plt.show()

if __name__ == "__main__":
	main()
	#equation()
