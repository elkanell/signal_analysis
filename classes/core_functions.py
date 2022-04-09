
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
duration = 4e+3 #usec
DT = 1e+6 / SRATE # in us
# produce simulated pulse without noise
pulse_max_duration = 50 # us
#indexes = np.random.permutation(np.arange(int(n/2),int(n/2) + pulse_max_duration/dt))
n_el = 2 #number of electrons
gain = 100 #average gain
fano = 0.25 # a value close to argon
p = 15 # poles for random interpolation

class CoreFuncs(object):
    
    time  = np.arange(0, duration, DT) #0 to 1000 us (1 ms)
    n = len(time)
    sigmaF = np.sqrt(fano*gain)# sigma avec Fano sigma = sqrt(fano*mean)
    gains = np.round(np.random.normal(gain, sigmaF, n_el))

    def __init__(self):
        return

    @classmethod
    def show(cls):
        print("[{}] The message is: {}".format(cls,"message"))
        return

    @classmethod
    def test(cls):
        print("Test message")
        return
    
    
    # definition of the functions: ion current and induced current
    @staticmethod
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

    @staticmethod
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


    @classmethod
    def createPulses(self, distance, num=2): 
        #indexes = indexes[:n_el]
        a = distance #the distance of the two pulses
        
        # create two pulses for the two electrons in a distance a  
        s1 = np.zeros(self.n)
        s2 = np.zeros(self.n)
        raw_pulse = np.zeros(self.n)

        for num in range (0, 2):
            index = 2000
            zeros_part = np.zeros(int(index))
            time_temp = np.arange(0, DT*(self.n-index), DT)

            if num == 0:
                ic1 = self.gains[num] * CoreFuncs.ion_current(time_temp)
                pulse_temp1 = np.concatenate( (zeros_part, ic1), axis=0) 
                s1 = s1 + pulse_temp1 
            if num == 1:
                index = index + a
                zeros_part = np.zeros(int(index))
                time_temp = np.arange(0, DT*(self.n-index), DT)
                ic2 = self.gains[num] * CoreFuncs.ion_current(time_temp) 
                pulse_temp2 = np.concatenate( (zeros_part, ic2), axis=0) 
                s2 = s2 + pulse_temp2 

        # # the two pulses in one array
        raw_pulse = s1 + s2
        return self.time,raw_pulse
        