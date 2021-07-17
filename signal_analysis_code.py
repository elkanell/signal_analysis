#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:25:22 2019

@author: ioanniskatsioulas
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import copy
import ctypes

import struct
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal

from array import array

from datetime import datetime



from matplotlib import interactive
interactive(True)

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

#%%
def ion_current(time, r_a = 0.1, r_c = 15, voltage = 2000, pressure = 1, mobility_0 = 2.e-6):
    """ J.Derre derivation of the formula
        distance [cm]
        pressure [bar]
        time [usec]
        voltage [V]
    """
    rho = (r_c * r_a) / (r_c - r_a)
    alpha = ( 2.e-6 ) * voltage * rho / pressure # constant that depends on the gas and the anode radius

    # return rho * ( 1 / r_a - 1 / (r_a**3 + 3 * alpha * time)**(1/3) )
    return alpha * rho * (r_a**3 + 3 * alpha * time)**(-4/3)

def induced_current(t, d, z, ve, vi):
	#Units used cm, us, fC
	q = 1.6E-4 #fC
	
	tel_max = d-z / ve
	ti_max = z/vi
	
#	print("Electron signal duration {0:.3f} us".format(tel_max))
#	print("Electron signal duration {0:.3f} us".format(ti_max))

	if t < tel_max:
		return -q*t*(ve+vi)/d
	elif d-z / ve < t < ti_max:
		return -q*(d-z+vi*t)/d
	else: 
		return -q

#charge = induced_current(10, 0.5, 0.5, 4.5, 1.55e-3)
#print(charge)

#time = np.arange(0, n, dt)
#	charge_t = np.array([])
#
#for t in time:
#	charge_t = np.append(charge_t, [induced_current(t, 0.5, 0.5, 4.5, 1.55e-3)])



#-----------------------------------------------------------------------------#
#Create the pulse to be denoised
#-----------------------------------------------------------------------------#

#units us, MHz 

# create signal
srate = 1e+6 # MHz
duration = 4e+3 #usec
dt = 1e+6 / srate # in us
time  = np.arange(0, duration, dt) #0 to 1000 us (1 ms)
n     = len(time)
p     = 15 # poles for random interpolation

# produce simulated pulse without noise
pulse_max_duration = 50 # us
indexes = np.random.permutation(np.arange(int(n/2),int(n/2) + pulse_max_duration/dt))

n_el = 10 #number of electrons
gain = 100 #average gain
fano = 0.25 # a value close to argon
sigmaF = np.sqrt(fano*gain)# sigma avec Fano sigma = sqrt(fano*mean)
gains = np.round(np.random.normal(gain, sigmaF, n_el))
indexes = indexes[:n_el]

#charge_t = np.array([])

#for t in time:
#	charge_t = np.append(charge_t, [induced_current(t, 0.5, 0.5, 4.5, 1.55e-3)])
raw_pulse = np.zeros(n)

for num, index in enumerate(indexes):
	print(num, index)
	zeros_part = np.zeros(int(index))
	time_temp = np.arange(0, dt*(n-index), dt)
	ic = gains[num] * ion_current(time_temp)
	pulse_temp = np.concatenate( (zeros_part, ic), axis=0)
	raw_pulse = raw_pulse + pulse_temp



plt.figure(1)
plt.plot(time, raw_pulse)

len_preamp_response = int(n/2)
preamp_fall_time = 125
preamp_response = np.exp(- dt * np.arange( len_preamp_response ) / preamp_fall_time)

raw_pulse_temp = np.concatenate( (np.zeros(1000), raw_pulse), axis=0 )
pulse =  scipy.signal.fftconvolve(raw_pulse_temp, preamp_response, "same")
pulse = np.delete(pulse, range(4000, 5000), axis=0)

plt.figure(2)
plt.plot(time, raw_pulse)
plt.plot(time, pulse)

# start playing with fake noises
# noise level, measured in standard deviations
noiseamp = 20

# amplitude modulator and noise level
#ampl   = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p)*30)
ampl   = np.zeros(n) + np.linspace(50, -50, n) - 10000
noise  = noiseamp * np.random.randn(n)
signal = pulse + ampl + noise

# add a spike
# proportion of time points to replace with spikes
#propnoise = .05
#
## find noise points
#noisepnts = np.random.permutation(n)
#noisepnts = noisepnts[0:int(n*propnoise)]
noisepnts = [int(n/4), int(n/2)]

# generate signal and replace points with noise
signal[noisepnts] += 200+np.random.rand(len(noisepnts))*100

plt.figure(3)
#plt.plot(time, pulse)
plt.plot(time, signal, c = 'k')
#plt.plot(time, filtsig, label = 'filtered')

#plt.legend()
plt.xlabel('Time (usec.)')
plt.ylabel('Amplitude')
#plt.title('Running-mean filter with a k={0:.3f} us filter'. format(windowsize))
#plt.show()

signal = scipy.signal.detrend(signal)

baseline = np.mean(signal[int(0.1*n):int(0.3*n)])

signal -= baseline

#remove median
k = 20 # actual window is k*2+1

# lower and upper bounds
lowbnd = np.max((1,int(n/2)-k))
uppbnd = np.min((int(n/2)+k,n))
# compute median of surrounding points    
signal[int(n/2)] = np.median(signal[lowbnd:uppbnd])

plt.figure(4)
#plt.plot(time, pulse)
plt.plot(time, signal, c = 'k')
#plt.plot(time, filtsig, label = 'filtered')

#plt.legend()
plt.xlabel('Time (usec.)')
plt.ylabel('Amplitude')

# implement the running mean filter
k = 20 # filter window is actually k*2+1

#signal_conv = np.concatenate( (signal[:k], signal, signal[-k:]) ,axis=0)
signal_conv = np.concatenate( (signal[k:0:-1], signal , signal[-1:-1-k:-1]) ,axis=0)

n_new = len(signal_conv)
# initialize filtered signal vector
filtsig = np.zeros(n_new)

for i in range(k+1, n_new-k-1):
    # each point is the average of k surrounding points
    filtsig[i] = np.mean(signal_conv[i-k:i+k])
#filtsig = np.smooth(signal, window_len=k, window='flat')
filtsig = filtsig[k:-k]
# compute window size in us
windowsize = (k*2+1) * dt


# plot the noisy and filtered signals
plt.figure(5)
plt.plot(time, signal, c = 'k', label = 'orig')
plt.plot(time, filtsig, label = 'filtered')

plt.legend()
plt.xlabel('Time (usec.)')
plt.ylabel('Amplitude')
#plt.title('Running-mean filter with a k={0:.3f} us filter'. format(windowsize))

plt.show()



# %%

"""Read pulse output from samba and process them.

Author:
	Ioannis Katsioulas (ioannis.katsioulas@gmail.com)

Usage:
	python3 pulse_prosessing.py


Notes:
	1) The pulse is stored in binary form as an array of short int (C++)
	2)
"""

# %matplotlib osx


def get_event_pulse(file, length):
	"""Read the chunk of binary data that is the pulse and return an array.

	Args:
	    file: The file object that is being processed
	    length: the length of the short int array
       """
	shortSize = ctypes.sizeof(ctypes.c_short)  #short int size in bytes
	chunkSize = length*shortSize  # total amounts of bytes to read
	dataBinary = file.read(chunkSize) # read the chunk size data

	return struct.unpack('h'*length, dataBinary)

def get_baseline_and_noise(pulse, baseline_calulation_start,
	baseline_calulation_end):
	# Must include detrend option detrend =  True
	baseline_part = pulse[baseline_calulation_start:baseline_calulation_end]

	baseline = np.mean(baseline_part)

	baseline_part_detrend = scipy.signal.detrend(baseline_part)

	noise = np.std(baseline_part_detrend)

	return baseline, noise


def ecdf(data, length):
	"""Compute ECDF for a one-dimensional array of measurements.
	Args:
		data: The data array
		length: The length of the array
	Return:
		data_sorted: The sorted array
		ecdf: The ecdf of the distribution in the array
	"""
	data_sorted = np.sort(data)
	ecdf = np.arange(1, length + 1) / float(length)
	return data_sorted, ecdf
# ============================================================================ #
# Deconvolution functions
# ============================================================================ #

def ion_current(time, r_a = 0.1, r_c = 15, voltage = 2000, pressure = 1, mobility_0 = 2.e-6):
    """ J.Derre derivation of the formula
        distance [cm]
        pressure [bar]
        time [usec]
        voltage [V]

		mobility_0 is at 1 bar
    """
    rho = (r_c * r_a) / (r_c - r_a)
    alpha = mobility_0 * voltage * rho / pressure # constant that depends on the gas and the anode radius

    # return rho * ( 1 / r_a - 1 / (r_a**3 + 3 * alpha * time)**(1/3) )
    return alpha * rho * (r_a**3 + 3 * alpha * time)**(-4/3)


def deconvolve_preamp_response(pulse, preamp_fall_time, length, pretrigger):
	# Deconvolution of the pulse
	len_preamp_response = length - int(pretrigger)

	preamp_response = np.exp(- dt * np.arange( len_preamp_response ) / preamp_fall_time)

	# preamp_response = np.array(preamp_response)
	# threshold_over_zero = 0.01 # a value over zero to make sure that Deconvolution doesn't fail due to numeric errors
	# preamp_response = preamp_response[preamp_response>threshold_over_zero]

	pulse_padded = np.zeros( length + len(preamp_response) - 1 )
	pulse_padded[:length] = pulse

	# pulse_padded = np.zeros(2*length-1-pretrigger)
	# pulse_padded[:length] = pulse

	signal, residual = scipy.signal.deconvolve(pulse_padded, preamp_response)

	return  signal, residual


def deconvolve_ion_response(signal, length, pretrigger):

   ion_resp = ion_current(dt*np.arange(pretrigger), r_a = 0.05, r_c = 15, voltage = 1625, pressure = 1, mobility_0 = 2.e-6)

   pulse_padded = np.zeros( length + len(ion_resp) - 1 )
   pulse_padded[:length] = signal

   electron_signal, residual = scipy.signal.deconvolve(pulse_padded, ion_resp)

   return electron_signal, residual

# ============================================================================ #
# Butterworth filter functions
# ============================================================================ #

def array_cdf(array, length, dt):

	array_cdf = np.zeros(length)

	for i in range(length-1):
	  array_cdf[i+1] = np.trapz(array[:i+1], dx = dt)

	return array_cdf

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

# ============================================================================ #
# Running mean filter
# ============================================================================ #
def running_mean_filer(signal, k = 50):
	"""
	Should add option for edges and if statement for k > something

	"""
	n = len(signal)

	# initialize filtered signal vector
	filtsig = np.zeros(n)

	# implement the running mean filter
	for i in range(k+1, n-k-1):
	    # each point is the average of k surrounding points
	    filtsig[i] = np.mean(signal[i-k:i+k])

	return filtsig

def deconvolve_preamp_response_FFT(signal, preamp_fall_time, length, pretrigger):
	# Deconvolution of the pulse
	len_preamp_response = length - int(pretrigger)

	preamp_response = np.exp(- dt * np.arange( len_preamp_response ) / preamp_fall_time)
	## deconvolution implemented in the frequency domain
	# convolution sizes
	nSign = len(signal)
#	nKern = len(preamp_response)
#	print("Length nKern = %d" % nKern)
#	nConv = nSign + nKern - 1
#	print("Length nConv = %d" % nConv)

#	half_kern = int( np.floor(nKern/2) )
#	print("Length half_kern = %d" % half_kern)

	# spectra of signal and kernel
	signalF = scipy.fftpack.fft(signal, nSign)
	preamp_responseF = scipy.fftpack.fft(preamp_response, nSign)

	# element-wise multiply
	sigXkern = signalF / preamp_responseF

	# inverse FFT to get back to the time domain
	deconv_resFFT = np.real( scipy.fftpack.ifft( sigXkern ) )

	# cut off edges
#	deconv_resFFT = deconv_resFFT[half_kern+1:-half_kern]

	return deconv_resFFT

# ============================================================================ #
# Samba File info
# ============================================================================ #

#filename = "tj10n002.root"
#
#direc = "/Users/ioanniskatsioulas/Google Drive/WorkDir/Analysis/Repository/LabData/FastNeutrons/tj10n002/"
#
#os.chdir(direc)
#
#rootfile = ROOT.TFile(filename, "read")
#
#treeSamba = rootfile.Get("SambaData")
## filename = "tf21s006_pulses_samba.bin"
#
#nevents = treeSamba.GetEntries()
#
#print('*---------------------------------------------------------------*\n')
#print('Info\n')
#
#print("File: %s" % filename)
#
#print('Number of Events %d' % nevents)
#
#print('*---------------------------------------------------------------*\n')
#
#treeSamba.GetEntry(0)

# ============================================================================ #
# Constants
# ============================================================================ #
length = int(n)
pretrigger = int(n/2)
preamp_fall_time = 125 # usec preamplifier fall time constant, *** This should be an argv input ***

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# %%
# ============================================================================ #
# Output Tree info
# ============================================================================ #
#outfile = ROOT.TFile("pros_"+filename, "recreate")

#treePros = ROOT.TTree( 'tpros', 'ProcessedData' )

# initialise tree variables in memory
time = array('f', [ 0 ])
baseline  = array( 'f', [ 0 ] )
noise  = array( 'f', [ 0 ] ) # ton xrono tha ton kanw enan
pulse_height = array( 'f', [ 0 ] )
risetime = array('f', [ 0 ])
width = array('f', [ 0 ])
integral = array('f', [ 0 ])
signal_noise = array('f', [ 0 ])
signal_integral = array('f', [ 0 ])
signal_integral_true = array('f', [ 0 ])
signal_risetime = array('f', [ 0 ])
signal_width = array('f', [ 0 ])
dodgy = array('i', [ 0 ])

# Branches defintion
#treePros.Branch("time", time, "time/F")
#treePros.Branch("baseline", baseline, "baseline/F")
#treePros.Branch("noise", noise, "noise/F")
#treePros.Branch("amplitude", pulse_height, "amplitude/F")
#treePros.Branch("risetime", risetime, "risetime/F")
#treePros.Branch("width", width, "width/F")
#treePros.Branch("integral", integral, "integral/F")
#treePros.Branch("signal_noise", signal_noise, "signal_noise/F")
#treePros.Branch("signal_integral", signal_integral, "signal_integral/F")
#treePros.Branch("signal_integral_true", signal_integral_true, "signal_integral_true/F")
#treePros.Branch("signal_risetime", signal_risetime, "signal_risetime/F")
#treePros.Branch("signal_width", signal_width, "signal_width/F")
#treePros.Branch("dodgy", dodgy, "dodgy/I")
# Prepei na min eisai malakas kai swzeis to xrono !!!
# ============================================================================ #
# Get the pulse of an event
# ============================================================================ #
verbose = True


startTime = datetime.now()

#for i in [67, 53 ,80, 89, 90, 114]:


#	treeSamba.GetEntry(i)

#	print ("Event {0} / Processed {1:3.2f}% of file \r".format(i, 100. * float(i) / nevents), end='', flush = True)

time[0] = 0

dodgy[0] = 0

pulse =  copy.deepcopy(filtsig)

# Calculate preliminary baseline to calculate constant
# term of the pulse from the pretrace
# Region to calculate noise and baseline
baseline_calulation_start = int(pretrigger*0.2)
baseline_calulation_end = int(pretrigger*0.8)

baseline[0], noise[0] = get_baseline_and_noise(pulse, baseline_calulation_start, baseline_calulation_end)


print('# -----------------------------------------------------------------#')
print(f'Event number: {i:d}')
print(f'Baseline: {baseline[0]:.2f}')

# pulse after baseline restoration

pulse = pulse - baseline

pulse_pre_treatment = pulse

# Filter with running mean
pulse_filtered = running_mean_filer(pulse, 11)

pulse = pulse_filtered


#    print(f'Noise: {noise[0]:.2f}')
pulse_height_position = np.argmax(pulse)
pulse_height[0] = pulse[pulse_height_position]

# signal = deconvolve_preamp_response_FFT(pulse, preamp_fall_time, length, pretrigger)
signal, residual = deconvolve_preamp_response(pulse, preamp_fall_time, length, pretrigger)


first_deriv_pulse = np.gradient(pulse, dt)


# ======================================================================== #
# The max derivative way for pulse height
# ======================================================================== #
#max_deriv_position = np.argmax(first_deriv_pulse)
#pulse_height_position = np.argmax(first_deriv_pulse[max_deriv_position:] < 0) + max_deriv_position
#pulse_height[0] = pulse[pulse_height_position]

#        if pulse_height_position - 200 <  pretrigger:
#            dodgy[0] = 1
#            print("Found dodgy3 event %d" % i)
#            continue

    #    if pulse_height[0] == 0:
    #        pulse_height[0] = 1

print(f'Pulse height: {pulse_height[0]:.2f}')

time_over_90_perc = np.where(pulse>0.9*pulse_height[0] )[0][0] # point when we go over 90% of the pulse
time_over_10_perc = np.where(pulse>0.1*pulse_height[0] )[0][0] # point when we go over 10% of the pulse



time_over_50_perc_rise = np.where(pulse[:pulse_height_position]>0.5*pulse_height[0] )[0][0] # point when we go over 90% of the pulse
time_over_50_perc_fall = pulse_height_position + np.where(pulse[pulse_height_position:]<0.5*pulse_height[0])[0][0] # point when we go over 10% of the pulse

time_at_90 = np.interp(0.9, [pulse[time_over_90_perc-1]/pulse_height[0], pulse[time_over_90_perc]/pulse_height[0]], [time_over_90_perc-1, time_over_90_perc])
time_at_10 = np.interp(0.1, [pulse[time_over_10_perc-1]/pulse_height[0], pulse[time_over_10_perc]/pulse_height[0]], [time_over_10_perc-1, time_over_10_perc])

time_at_50_rise = np.interp(0.5, [pulse[time_over_50_perc_rise-1]/pulse_height[0], pulse[time_over_50_perc_rise]/pulse_height[0]], [time_over_50_perc_rise-1, time_over_50_perc_rise])
time_at_50_fall = np.interp(0.5, [pulse[time_over_50_perc_fall-1]/pulse_height[0], pulse[time_over_50_perc_fall]/pulse_height[0]], [time_over_50_perc_fall-1, time_over_50_perc_fall])

risetime[0] = (time_at_90 - time_at_10)*dt
width[0] = (time_at_50_fall - time_at_50_rise)*dt
integral[0] = np.trapz(pulse, dx = dt)

print(f'Rise time: {risetime[0]:.2f} usec')
print(f'Width: {width[0]:.2f} usec')
print(f'Integral: {integral[0]:.2f} ADU*usec')

# ======================================================================== #
# Calcualte the parameters of a deconvoluted pulse
# ======================================================================== #



# electron_signal, residual_el = deconvolve_ion_response(signal, length, pretrigger)
# testing zero padding for safety
# signal[:baseline_calulation_start] = 0

signal_noise[0] = np.std(signal[baseline_calulation_start:baseline_calulation_end])

trigger_amplitude = 5 * signal_noise # trigger criterion


print(f'Signal noise: {signal_noise[0]:.2f}')
print(f'Trigger amplitude: {trigger_amplitude[0]:.2f} ADU')

#calculate the ecdf of the pulse points
# signal_sorted, signal_ecdf = ecdf(signal, length) # Calculate the ecdf to use for timing calculations

signal_integral[0] = np.trapz(signal, dx = dt) # Measure of the charge deposited

# Calculate pulse points percentage over an amplitude
# amplitude_over_90 = signal_sorted[np.where(signal_ecdf > 0.9 )[0][0]]
# amplitude_over_95 = signal_sorted[np.where(signal_ecdf > 0.95)[0][0]]
# amplitude_over_10 = signal_sorted[np.where(signal_ecdf > 0.1 )[0][0]]
# amplitude_over_50 = signal_sorted[np.where(signal_ecdf > 0.5 )[0][0]]
# amplitude_over_99 = signal_sorted[np.where(signal_ecdf > 0.99)[0][0]]
# amplitude_over_01 = signal_sorted[np.where(signal_ecdf > 0.01)[0][0]]

# ======================================================================== #
# Calculate the cdf of the signal and its timing parameters
# ======================================================================== #

# Signal integration
signal_cdf = array_cdf(signal, length, dt)
#plt.plot(signal_cdf)

# electron_signal_cdf = array_cdf(electron_signal, length, dt)

# where the true maximum should be found if there is undershoot
# the choice for order pretrigger is so that is not hardcoded but I need to check if it's the optimal
max_in_signal_cdf = scipy.signal.argrelmax(signal_cdf, order = pretrigger)[0][0]
signal_integral_true[0] = signal_cdf[max_in_signal_cdf] #correct for undershoot
signal_cdf_norm = signal_cdf/signal_integral_true[0]

print(f'Signal integral: {signal_integral_true[0]:.2f} ADU')
#print(f'Ballistic deficit {100. *pulse_height/signal_integral_true:.2f}%')

signal_time_over_90_perc = np.where(signal_cdf_norm>0.9)[0][0] # point when we go over 90% of the pulse
signal_time_over_10_perc = np.where(signal_cdf_norm>0.1)[0][0] # point when we go over 10% of the pulse

signal_time_over_50_perc = np.where(signal_cdf_norm>0.5)[0][0] # point when we go over 90% of the pulse

signal_time_at_90 = np.interp(0.9, [signal_cdf_norm[signal_time_over_90_perc-1], signal_cdf_norm[signal_time_over_90_perc]], [signal_time_over_90_perc-1, signal_time_over_90_perc]) #times dt to make it in usec
signal_time_at_10 = np.interp(0.1, [signal_cdf_norm[signal_time_over_10_perc-1], signal_cdf_norm[signal_time_over_10_perc]], [signal_time_over_10_perc-1, signal_time_over_10_perc]) #times dt to make it in usec

signal_time_at_50 = np.interp(0.5, [signal_cdf_norm[signal_time_over_50_perc-1], signal_cdf_norm[signal_time_over_50_perc]], [signal_time_over_50_perc-1, signal_time_over_50_perc])
signal_time_at_100 = max_in_signal_cdf * dt

signal_risetime[0] = (signal_time_at_90 - signal_time_at_10)*dt
signal_width[0] = (signal_time_at_100 - signal_time_at_10)*dt
		    	# integral = np.trapz(pulse, dx = dt)

print(f'Signal rise time: {signal_risetime[0]:.2f} usec')
print(f'Signal width: {signal_width[0]:.2f} usec')

first_deriv_signal = np.gradient(signal, dt)
#
# first_deriv_signal = scipy.signal.savgol_filter(first_deriv_signal, 33, 2)
# thres = 6 * np.std(first_deriv_signal[baseline_calulation_start:baseline_calulation_end])
# first_deriv_signal = abs(first_deriv_signal)

# Fill the Tree with the calculated varialbles
#		treePros.Fill()

	


# =============================================================================#
# Saving event data to tree
# =============================================================================#
print('# -----------------------------------------------------------------#')
print("		Processing time: {0} ".format(datetime.now() - startTime) )
print('# -----------------------------------------------------------------#')

#treePros.Print() # to see if everything went ok
#
#outfile.Write()
#outfile.Close()

# =============================================================================#
#    peaks, properties = scipy.signal.find_peaks(first_deriv_signal, height = thres)
#    print(f'Number of peaks: {len(peaks)}')

#	outfile.cd()

    # =============================================================================#
	# Pulse plots
	# =============================================================================#
plt.close(0)
plt.figure(100)
plt.plot(pulse, 'b--')
plt.plot(50*first_deriv_pulse, 'r--')
plt.xlabel(r"Time ($\mu$s)")
plt.ylabel("Amplitude")

plt.figure(0)
plt.plot(pulse, 'b--')
plt.plot(signal, 'r--')
##	plt.plot(electron_signal, 'g--')
#
#
plt.figure(1)
plt.plot(pulse_pre_treatment, 'k--')
plt.plot(pulse_filtered, 'r--')
#
plt.figure(2)
plt.plot(raw_pulse, 'r-', lw = 3)
plt.plot(signal, 'k--', lw = 3)
plt.xlabel(r"Time ($\mu$s)")
plt.axvline(x = pulse_height_position , label = "Pulse height position", color = 'r', lw = 2)
plt.title('The deconvolved signal')
plt.legend()
#
#    plt.figure(3)
#    plt.plot(signal, 'k--', lw = 3)
#    plt.plot(residual, 'r--', lw = 3)
#    plt.axvline(x = pulse_height_position , label = "Pulse height position", color = 'r', lw = 2)
#    plt.legend()
#
plt.figure(4)
#    plt.plot(signal_cdf / np.max(signal_cdf), 'rd-')
plt.plot(signal_cdf , 'rd-')
#	plt.plot(electron_signal_cdf / np.max(electron_signal_cdf), 'bd-')
plt.axvline(x = signal_time_at_10, label = "Rise time interval", color = 'k')
plt.axvline(x = signal_time_at_90, color = 'k')
plt.axvline(x = max_in_signal_cdf, color = 'k', label = "Width interval", linestyle = '--')
plt.xlabel(r"Time ($\mu$s)")
plt.ylabel("Amplitude")
plt.legend()
#
#    plt.figure(5)
#    plt.plot(signal_sorted, signal_ecdf, 'r+', lw = 3)
#    plt.axvline(x = amplitude_over_10, label = "10%", color = 'g', lw = 2)
#    plt.axvline(x = amplitude_over_90, label = "90%", color = 'b', lw = 2)
#    plt.axvline(x = amplitude_over_95, label = "95%", color = 'm', lw = 4)
#    plt.axvline(x = amplitude_over_99, label = "99%", color = 'r', lw = 2)
#    plt.axvline(x = amplitude_over_01, label = "1%",  color = 'y', lw = 2)
#    plt.xlabel("Amplitude")
#    plt.ylabel("Percentile")
#    plt.legend()
#
plt.figure(6)
plt.plot(signal_cdf_norm, 'r+', label = "signal_cdf (norm)", lw = 3)
plt.plot(signal/np.max(signal), 'k.-', label = "signal (norm)", lw = 3)
plt.plot(pulse/np.max(pulse), linestyle = 'none', marker = '+', c = 'b', label = 'pulse (norm)')
plt.xlabel(r"Time ($\mu$s)")
plt.legend()
# =============================================================================#
# Figure displaying the basic pulse parameters
# =============================================================================#
plt.figure(7)

plt.plot(pulse, linestyle = 'none', marker = '+', c = 'b', label = 'pulse')

plt.plot(first_deriv_pulse*10, linestyle = 'none', marker = '+', c = 'r', label = 'pulse derivative')

plt.axvline(x = pretrigger, label='Pretrigger point = {}'.format(pretrigger), c=colors[3])
plt.axvline(x = time_at_10, label = "Rise time interval", color = 'k')
plt.axvline(x = time_at_90, color = 'k')
plt.hlines(pulse_height[0]*0.5, time_at_50_rise, time_at_50_fall, color = 'm', label = "width interval")
plt.hlines(pulse[baseline_calulation_start], baseline_calulation_start, baseline_calulation_end, color = 'g', label = "baseline", lw = 3)
plt.xlabel(r"Time ($\mu$s)")
	# baseline_part_detrend_vis = np.zeros(baseline_calulation_start+len(baseline_part_detrend))
	# baseline_part_detrend_vis[:baseline_calulation_start] = pulse[:baseline_calulation_start]
	# baseline_part_detrend_vis[baseline_calulation_start:] = baseline_part_detrend
	#
	# plt.plot(baseline_part_detrend_vis, 'c>', lw = 4)
plt.legend()
plt.draw()
plt.show()
	# =============================================================================#
	# Up to here I plot now
    # =============================================================================#


	# b, a = signal.butter(4, 100, 'low', analog=True)
	# w, h = signal.freqs(b, a)
	# plt.plot(w, 20 * np.log10(abs(h)))
	# plt.xscale('log')
	# plt.title('Butterworth filter frequency response')
	# plt.xlabel('Frequency [radians / second]')
	# plt.ylabel('Amplitude [dB]')
	# plt.margins(0, 0.1)
	# plt.grid(which='both', axis='both')
	# plt.axvline(100, color='green') # cutoff frequency
	# plt.show()

	# =============================================================================#
	# Filtering
	# =============================================================================#
	# N  = 3    # Filter order
	# fs = 0.5*1e6/dt # Hz
	# Wn = 0.25 #40e3/fs # Cutoff frequency
	# B, A = scipy.signal.butter(N, Wn, btype='low', output='ba')
	# smooth_data = scipy.signal.lfilter(B, A, pulse)
	#
	# Wn = 0.005 # Cutoff frequency
	# B1, A1 = scipy.signal.butter(N, Wn, btype='high', output='ba')
	# smooth_data1 = scipy.signal.filtfilt(B1, A1, smooth_data)
	#
	# plt.figure(25)
	# plt.plot(smooth_data)
	# plt.plot(smooth_data1)
	# plt.plot(pulse)
# Sample rate and desired cutoff frequencies (in Hz).

# fs = 0.5*1e6/dt # Hz
# # lowcut = 500.0
# lowcut = 3E+4
# highcut = 6E+4
#
# # Plot the frequency response for a few different orders.
# plt.figure(31)
# # plt.clf()
# for order in [3, 6, 9]:
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     w, h = freqz(b, a, worN=2000)
# plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#
# plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain')
# plt.grid(True)
# plt.legend(loc='best')



	# plt.figure(32)
	# plt.plot(filtered_pulse, 'k--', label = f'Filtered pulse at {cutoff_freq}')
	# # plt.plot(smooth_data1, 'r--')
	# plt.plot(pulse, 'r--', label = 'Raw pulse')
	# plt.ylabel('Pulse height [Hz]')
	# plt.xlabel('Point')
	# plt.legend()
	#
	# filt_base, filt_noise = get_baseline_and_noise(filtered_pulse, baseline_calulation_start,
	# 	baseline_calulation_end)
	# print(f'Filtered Signal noise: {filt_noise:.2f}')
	# =============================================================================#
	# =============================================================================#


	# =============================================================================#
	# FFT
	# =============================================================================#
	#
# plt.figure(32)
# plt.clf()
# plt.plot(t, x, label='Noisy signal')
# #
# y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
# plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
# plt.xlabel('time (seconds)')
# plt.hlines([-a, a], 0, T, linestyles='--')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc='upper left')

	# Frequency vs Time Power weighted
#	plt.figure(13)
#	f, t, Sxx = scipy.signal.spectrogram(pulse, window= 'boxcar', fs = (1e6)/dt, scaling = 'spectrum', nperseg = 220)
#	plt.pcolormesh(t, f, Sxx)
#	plt.ylabel('Frequency [Hz]')
#	plt.xlabel('Time [sec]')
#	plt.show()
#
#	plt.figure(12)
#	f,  Pxx_den  = scipy.signal.periodogram(pulse[600:],  fs = (1e6)/dt, scaling = 'spectrum')
#	f1, Pxx_den1 = scipy.signal.periodogram(pulse,        fs = (1e6)/dt, scaling = 'spectrum')
#	f2, Pxx_den2 = scipy.signal.periodogram(pulse[0:200], fs = (1e6)/dt, scaling = 'spectrum')
#	plt.semilogy(f,  Pxx_den ,  label = '600:', color='b')
#	plt.semilogy(f1, Pxx_den1,  label = 'full', color='r')
#	plt.semilogy(f2, Pxx_den2,  label = '0:200', color='k')

	# plt.loglog(f, Pxx_den , label = '600:')
	# plt.loglog(f1, Pxx_den1, label = 'full')
	# plt.loglog(f2, Pxx_den2,  label = '0:200')
	# plt.ylim([1e-10, 10])
#	plt.ylim([1e-4, 1e7])
#	plt.xlabel('Frequency [Hz]')
	# plt.ylabel('PSD [V**2/Hz]')
#	plt.ylabel('Linear spectrum [V RMS]')
#	plt.legend()
	# =============================================================================#
	# =============================================================================#

	# plt.figure(10)
	# plt.plot(first_deriv_signal)
	# plt.plot(peaks, first_deriv_signal[peaks], "x")
	# plt.plot(np.zeros_like(first_deriv_signal), "--", color="gray")
	# plt.hlines(thres, 0, length, color = 'm', label = "Threshold")
	# plt.show()
	# ion_shape = np.array([ion_pulse_shape(alpha = 0.005, rho = 10, r_int = 0.1, time = t*dt ) for t in range(length)])
	#
	# plt.figure(10)
	# plt.plot(ion_shape)
	# # plt.figure(2)
	# # plt.plot(preamp_response, 'b-')
	# plt.show()



	# plt.show()



#

	# =============================================================================#
	# =============================================================================#


	# =============================================================================#
	# =============================================================================#
	# plt.figure(7)
	# plt.plot(first_deriv_signal*5, 'r--', label = "derivative ( x 5)")
	# plt.plot(signal, 'k.-', label = "signal")
	# plt.legend()
	# =============================================================================#
	# =============================================================================#

	# plt.figure(8)
	# plt.title("First derivative of the raw pulse")
	# plt.plot(first_deriv_pulse*10, 'r--', label = "derivative ( x 10)")
	# # plt.plot(signal, 'k.-', label = "signal")
	# plt.legend()
	# =============================================================================#
	# =============================================================================#


	# plt.plot(pulse)

#    plt.show()
#
#    in_key = input('Press key to go to next pulse (0 to exit):')
#    if(in_key == '0'):
#        break




#TFile *f = new TFile("hs.root","update");
#TTree *T = (TTree*)f->Get("ntuple");
#
#float px,py;
#float pt;
#
#TBranch *bpt = T->Branch("pt",&pt,"pt/F");
#T->SetBranchAddress("px",&px);
#T->SetBranchAddress("py",&py);
#
#Long64_t nentries = T->GetEntries();
#for (Long64_t i=0;i<nentries;i++)
#{
#     T->GetEntry(i);
#     pt = TMath::Sqrt(px*px+py*py);
#     bpt->Fill();
#}
#T->Print();
#T->Write();
#delete f;

	# # Figure to display the Cumulative pulse distribution
	# plt.figure(2)
	# plt.plot(pulse_sorted, ecdf)
	# plt.xlabel("Pulse Height (ADU)")
	# plt.ylabel("ECDF")
	# plt.show()


# %%