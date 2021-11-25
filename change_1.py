#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:25:22 2019

@author: ioanniskatsioulas
"""

#%%
import matplotlib
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

import pandas as pd

from matplotlib import interactive
interactive(True)

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

#%%

# definition of the functions: ion current and induced current
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
	
	if t < tel_max:
		return -q*t*(ve+vi)/d
	elif d-z / ve < t < ti_max:
		return -q*(d-z+vi*t)/d
	else: 
		return -q


# %%
srate = 1e+6 # MHz
duration = 4e+3 #usec
dt = 1e+6 / srate # in us
time  = np.arange(0, duration, dt) #0 to 1000 us (1 ms)
n     = len(time)
p     = 15 # poles for random interpolation


# produce simulated pulse without noise
pulse_max_duration = 50 # us
#indexes = np.random.permutation(np.arange(int(n/2),int(n/2) + pulse_max_duration/dt))

n_el = 2 #number of electrons
gain = 100 #average gain
fano = 0.25 # a value close to argon
sigmaF = np.sqrt(fano*gain)# sigma avec Fano sigma = sqrt(fano*mean)
gains = np.round(np.random.normal(gain, sigmaF, n_el))
#indexes = indexes[:n_el]
a = 1000 #the distance of the two pulses
 
# create two pulses for the two electrons in a distance a  
s1 = np.zeros(n)
s2 = np.zeros(n)
raw_pulse = np.zeros(n)

for num in range (0, 2):
	index = 2000
	zeros_part = np.zeros(int(index))
	time_temp = np.arange(0, dt*(n-index), dt)

	if num == 0:
		ic1 = gains[num] * ion_current(time_temp)
		pulse_temp1 = np.concatenate( (zeros_part, ic1), axis=0) 
		s1 = s1 + pulse_temp1 
	if num == 1:
		index = index + a
		zeros_part = np.zeros(int(index))
		time_temp = np.arange(0, dt*(n-index), dt)
		ic2 = gains[num] * ion_current(time_temp) 
		pulse_temp2 = np.concatenate( (zeros_part, ic2), axis=0) 
		s2 = s2 + pulse_temp2 

# the two pulses in one array
raw_pulse = s1 + s2

plt.figure(1)
plt.plot(time, raw_pulse)
plt.title('The two pulses in a distance a')

# %%
# preamplifier response
len_preamp_response = int(n/2)
preamp_fall_time = 125
preamp_response = np.exp(- dt * np.arange( len_preamp_response ) / preamp_fall_time)


# convoluted pulse with the preamplifier response 
# electronic signal
raw_pulse_temp_1 = np.concatenate( (np.zeros(1000), s1), axis=0 )
pulse_1 =  scipy.signal.fftconvolve(raw_pulse_temp_1, preamp_response, "same")
pulse_1 = np.delete(pulse_1, range(4000, 5000), axis=0)

raw_pulse_temp_2 = np.concatenate( (np.zeros(1000), s2), axis=0 )
pulse_2 =  scipy.signal.fftconvolve(raw_pulse_temp_2, preamp_response, "same")
pulse_2 = np.delete(pulse_2, range(4000, 5000), axis=0)

#plt.figure(2)
#plt.plot(time, s1, s2)
#plt.plot(time, pulse_1, pulse_2)

raw_pulse_temp = np.concatenate( (np.zeros(1000), raw_pulse), axis=0 )
pulse =  scipy.signal.fftconvolve(raw_pulse_temp, preamp_response, "same")
pulse = np.delete(pulse, range(4000, 5000), axis=0)

plt.figure(2)
#plt.plot(time, raw_pulse)
plt.plot(time, pulse)
plt.title('The electronic signal of the two electrons')

length = len(pulse)
pulse_padded = np.zeros( length + len(preamp_response) - 1 )
pulse_padded[:length] = pulse

deconv, _ = scipy.signal.deconvolve(pulse_padded, preamp_response)

plt.figure(3)
plt.plot(time, deconv)
plt.title('The two pulses in a distance a after deconvolution')

#white noise
noiseamp = 2

# add a white noise to the signal

#ampl = np.zeros(n) + np.linspace(50, -50, n) - 10000
noise = noiseamp * np.random.randn(n)
signal_1 = pulse_1 + noise #+ ampl 
signal_2 = pulse_2 + noise #+ ampl 

signal = pulse + noise

#noisepnts = [int(n/4), int(n/2)]
#signal_1[noisepnts] += 200 + np.random.randn(len(noisepnts)) * 100
#signal_2[noisepnts] += 200 + np.random.randn(len(noisepnts)) * 100

plt.figure(4)
plt.plot(time, signal)
plt.title('The electronic signal with noise')

#plt.figure(31)
#plt.plot(time, signal_1)

#plt.figure(32)
#plt.plot(time, signal_2, c = 'y')

#preamp_response = np.array(preamp_response)

# %%
# deconvoluted signal with noise and preamplifier response
length = len(signal)
signal_padded = np.zeros( length + len(preamp_response) - 1 )
signal_padded[:length] = signal

deconv, _ = scipy.signal.deconvolve(signal_padded, preamp_response)


plt.figure(5)
plt.plot(time, deconv)
plt.title('The pulse of the two electrons with noise')

# %% 
#ion deconvolution
ion_resp = ion_current(dt*np.arange(len_preamp_response), r_a = 0.1, r_c = 15, voltage = 2000, pressure = 1, mobility_0 = 2.e-6)

length = len(raw_pulse)
pulse_padded = np.zeros(length + len(ion_resp) - 1)
pulse_padded[:length] = raw_pulse

electron_signal, residual = scipy.signal.deconvolve(pulse_padded, ion_resp)

plt.figure(6)
plt.plot(time, electron_signal)
plt.title('The deconvolution of the ion response and the raw pulse')

#heights for the normalization
raw_pulse_height = array( 'f', [0])
electron_signal_height = array( 'f', [0])

electron_signal_height_position = np.argmax(electron_signal)
electron_signal_height[0] = electron_signal[electron_signal_height_position] 

raw_pulse_height_position = np.argmax(raw_pulse)
raw_pulse_height[0] = raw_pulse[raw_pulse_height_position] 


#normalized electron signal and raw pulse
electron_signal_norm = electron_signal / (electron_signal_height[0])
raw_pulse_norm = raw_pulse / (raw_pulse_height[0])

plt.figure(7)
plt.plot(time, electron_signal_norm, label = "electron signal")
plt.plot(time, raw_pulse_norm, label = "raw pulse")
plt.title('The electron signal and the raw pulse (normalized)')
plt.legend()

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

	pulse_padded = np.zeros( length + len(preamp_response) - 1 )
	pulse_padded[:length] = pulse

	signal, residual = scipy.signal.deconvolve(pulse_padded, preamp_response)

	return  signal, residual


def deconvolve_ion_response(signal, length, pretrigger):

   ion_resp = ion_current(dt*np.arange(pretrigger), r_a = 0.05, r_c = 15, voltage = 1625, pressure = 1, mobility_0 = 2.e-6)

   pulse_padded = np.zeros( length + len(ion_resp) - 1 )
   pulse_padded[:length] = signal

   electron_signal, residual = scipy.signal.deconvolve(pulse_padded, ion_resp)

   return electron_signal, residual

# ============================================================================ #
# Constants
# ============================================================================ #
length = int(n)
pretrigger = int(n/2)
preamp_fall_time = 125 # usec preamplifier fall time constant, *** This should be an argv input ***

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# ============================================================================ #
# Output Tree info
# ============================================================================ #
#outfile = ROOT.TFile("pros_"+filename, "recreate")

#treePros = ROOT.TTree( 'tpros', 'ProcessedData' )

# initialise tree variables in memory
time = array('f', [ 0 ])
pulse_height_1 = array( 'f', [ 0 ] )
risetime_1 = array('f', [ 0 ])
width_1 = array('f', [ 0 ])
integral_1 = array('f', [ 0 ])
pulse_height_2 = array( 'f', [ 0 ] )
risetime_2 = array('f', [ 0 ])
width_2 = array('f', [ 0 ])
integral_2 = array('f', [ 0 ])

signal_integral = array('f', [ 0 ])
signal_integral_true = array('f', [ 0 ])
signal_risetime = array('f', [ 0 ])
signal_width = array('f', [ 0 ])
dodgy = array('i', [ 0 ])

# ============================================================================ #
# Get the pulse of an event
# ============================================================================ #
verbose = True

startTime = datetime.now()

time[0] = 0

dodgy[0] = 0

pulse_height_position_1 = np.argmax(pulse_1)
pulse_height_1[0] = pulse_1[pulse_height_position_1]

pulse_height_position_2 = np.argmax(pulse_2)
pulse_height_2[0] = pulse_2[pulse_height_position_2]

# signal = deconvolve_preamp_response_FFT(pulse, preamp_fall_time, length, pretrigger)
signal_1, residual_1 = deconvolve_preamp_response(pulse_1, preamp_fall_time, length, pretrigger)
signal_2, residual_2 = deconvolve_preamp_response(pulse_2, preamp_fall_time, length, pretrigger)


first_deriv_pulse_1 = np.gradient(pulse_1, dt)
first_deriv_pulse_2 = np.gradient(pulse_2, dt)

# ======================================================================== #
# The max derivative way for pulse height
# ======================================================================== #


time_over_90_perc_1 = np.where(pulse_1>0.9*pulse_height_1[0] )[0][0] # point when we go over 90% of the pulse
time_over_10_perc_1 = np.where(pulse_1>0.1*pulse_height_1[0] )[0][0] # point when we go over 10% of the pulse

time_over_50_perc_rise_1 = np.where(pulse_1[:pulse_height_position_1]>0.5*pulse_height_1[0] )[0][0] # point when we go over 90% of the pulse
time_over_50_perc_fall_1 = pulse_height_position_1 + np.where(pulse_1[pulse_height_position_1:]<0.5*pulse_height_1[0])[0][0] # point when we go over 10% of the pulse

time_at_90_1 = np.interp(0.9, [pulse_1[time_over_90_perc_1-1]/pulse_height_1[0], pulse_1[time_over_90_perc_1]/pulse_height_1[0]], [time_over_90_perc_1-1, time_over_90_perc_1])
time_at_10_1 = np.interp(0.1, [pulse_1[time_over_10_perc_1-1]/pulse_height_1[0], pulse_1[time_over_10_perc_1]/pulse_height_1[0]], [time_over_10_perc_1-1, time_over_10_perc_1])

time_at_50_rise_1 = np.interp(0.5, [pulse_1[time_over_50_perc_rise_1-1]/pulse_height_1[0], pulse_1[time_over_50_perc_rise_1]/pulse_height_1[0]], [time_over_50_perc_rise_1-1, time_over_50_perc_rise_1])
time_at_50_fall_1 = np.interp(0.5, [pulse_1[time_over_50_perc_fall_1-1]/pulse_height_1[0], pulse_1[time_over_50_perc_fall_1]/pulse_height_1[0]], [time_over_50_perc_fall_1-1, time_over_50_perc_fall_1])



time_over_90_perc_2 = np.where(pulse_2>0.9*pulse_height_2[0] )[0][0] # point when we go over 90% of the pulse
time_over_10_perc_2 = np.where(pulse_2>0.1*pulse_height_2[0] )[0][0] # point when we go over 10% of the pulse

time_over_50_perc_rise_2 = np.where(pulse_2[:pulse_height_position_2]>0.5*pulse_height_2[0] )[0][0] # point when we go over 90% of the pulse
time_over_50_perc_fall_2 = pulse_height_position_2 + np.where(pulse_2[pulse_height_position_2:]<0.5*pulse_height_2[0])[0][0] # point when we go over 10% of the pulse

time_at_90_2 = np.interp(0.9, [pulse_2[time_over_90_perc_2-1]/pulse_height_2[0], pulse_2[time_over_90_perc_2]/pulse_height_2[0]], [time_over_90_perc_2-1, time_over_90_perc_2])
time_at_10_2 = np.interp(0.1, [pulse_2[time_over_10_perc_2-1]/pulse_height_2[0], pulse_2[time_over_10_perc_2]/pulse_height_2[0]], [time_over_10_perc_2-1, time_over_10_perc_2])

time_at_50_rise_2 = np.interp(0.5, [pulse_1[time_over_50_perc_rise_2-1]/pulse_height_2[0], pulse_2[time_over_50_perc_rise_2]/pulse_height_2[0]], [time_over_50_perc_rise_2-1, time_over_50_perc_rise_2])
time_at_50_fall_2 = np.interp(0.5, [pulse_1[time_over_50_perc_fall_2-1]/pulse_height_2[0], pulse_2[time_over_50_perc_fall_2]/pulse_height_2[0]], [time_over_50_perc_fall_2-1, time_over_50_perc_fall_2])



risetime_1[0] = (time_at_90_1 - time_at_10_1)*dt
width_1[0] = (time_at_50_fall_1 - time_at_50_rise_1)*dt
integral_1[0] = np.trapz(pulse_1, dx = dt)


risetime_2[0] = (time_at_90_2 - time_at_10_2)*dt
width_2[0] = (time_at_50_fall_2 - time_at_50_rise_2)*dt
integral_2[0] = np.trapz(pulse_2, dx = dt)


df = pd.DataFrame(
     {"pulse": [1, 2], "height": [round(pulse_height_1[0], 2), round(pulse_height_2[0], 2)], "risetime [usec]": [round(risetime_1[0], 2), round(risetime_2[0], 2)], "width [usec]": [round(width_1[0], 2), round(width_2[0], 2)], "integral [ADU*usec]": [round(integral_1[0], 2), round(integral_2[0], 2)]}
)
print(df)

# %%
