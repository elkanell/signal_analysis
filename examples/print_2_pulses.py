
from classes import core_functions as coreFuncs

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
interactive(True)


plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16


class ExampleTwoPulses(object):

    figureNum = 1

    @staticmethod
    def printExample(core, distance):
        time,raw_pulse = core.createPulses(distance)
        plt.figure(ExampleTwoPulses.figureNum)
        ExampleTwoPulses.figureNum = ExampleTwoPulses.figureNum +1
        plt.plot(time,raw_pulse)
        plt.title("The two pulses in a distance {}".format(distance))
        return

    @staticmethod
    def printExample2(core, raw_pulse):
        len_preamp_response = int(core.n/2)
        preamp_fall_time = 125
        preamp_response = np.exp(- coreFuncs.DT * np.arange( len_preamp_response ) / preamp_fall_time)


        raw_pulse_temp = np.concatenate( (np.zeros(1000), raw_pulse), axis=0 )
        pulse =  scipy.signal.fftconvolve(raw_pulse_temp, preamp_response, "same")
        pulse = np.delete(pulse, range(4000, 5000), axis=0)
        
        plt.figure(ExampleTwoPulses.figureNum)
        ExampleTwoPulses.figureNum = ExampleTwoPulses.figureNum +1
        #plt.plot(time, raw_pulse)
        plt.plot(core.time, pulse)
        plt.title('The electronic signal of the two electrons')

