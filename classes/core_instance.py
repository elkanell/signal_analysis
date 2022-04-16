import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import copy
import ctypes

import struct
import numpy as np
import matplotlib.pyplot as plt

import scipy.fftpack

from array import array

from datetime import datetime

import pandas as pd

from matplotlib import interactive

SRATE = 1e+6 # MHz
DURATION = 4e+3 #usec
DT = 1e+6 / SRATE # in us
# produce simulated pulse without noise
PULSE_MAX_DURATION = 50 # us
#indexes = np.random.permutation(np.arange(int(n/2),int(n/2) + pulse_max_duration/dt))
N_EL = 2 #number of electrons
GAIN = 100 #average gain
FANO = 0.25 # a value close to argon
P = 15 # poles for random interpolation

class CoreInstance(object):   
    #Define properties 
    time  = 0
    n = 0
    sigmaF = 0 
    gains = 0
    distance = 0
    len_preamp_response = 0
    preamp_fall_time = 0
    preamp_response = 0

    
    def __init__(self, distance, **kwargs):
        # Initialize properties' values 
        self.time  = np.arange(0, DURATION, DT) #0 to 1000 us (1 ms)
        self.n = len(self.time)
        self.sigmaF = np.sqrt(FANO*GAIN)# sigma avec Fano sigma = sqrt(fano*mean)
        self.gains = np.round(np.random.normal(GAIN, self.sigmaF, N_EL))
        self.distance = distance
        self.len_preamp_response = int(self.n/2)
        self.preamp_fall_time = 125
        self.preamp_response = np.exp(- DT * np.arange( self.len_preamp_response ) / self.preamp_fall_time)

        return
