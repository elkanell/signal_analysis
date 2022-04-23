
from classes import signal_analysis_functions as base
from classes import core_instance as instance

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

core = base.CoreFuncs


class Plotter(object):

    # definition of a function that plots a signal in time domain 

    figureNumOne = 1

    @staticmethod
    def plot(coreInstance, signal, title):       
        
        plt.figure(Plotter.figureNumOne)
        Plotter.figureNumOne = Plotter.figureNumOne + 1
        plt.plot(coreInstance.time, signal)
        plt.title(title)
        return


    # definition of a function that plots two signals together in time domain

    figureNumTwo = 200

    @staticmethod
    def plotTwoSignals(coreInstance, signal1, label1, signal2, label2, title):       
        
        plt.figure(Plotter.figureNumTwo)
        Plotter.figureNumTwo = Plotter.figureNumTwo + 1
        plt.plot(coreInstance.time, signal1, label = label1)
        plt.plot(coreInstance.time, signal2, label = label2)
        plt.title(title)
        plt.legend()
        return


    # definition of a function that plots a signal in frequency domain 

    figureNumThree = 400

    @staticmethod
    def plotFourier(coreInstance, signal, title):       
        
        plt.figure(Plotter.figureNumThree)
        Plotter.figureNumThree = Plotter.figureNumThree + 1
        the_instance = instance.CoreInstance()
        transformed, frequencies = core.FourierTransform(the_instance, signal) 
        plt.plot(frequencies, np.abs(transformed))
        plt.title(title)      
        return