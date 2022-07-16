
from classes import signal_analysis_functions as base
from classes import core_instance as instance

import matplotlib.pyplot as plt
import numpy as np
from os import path


plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

core = base.CoreFuncs
outpath = "/root/signal_analysis/plotters/signal_analysis_plots"

class Plotter(object):

    # definition of a function that plots a signal in time domain 

    figureNumOne = 0

    @staticmethod
    def plot(coreInstance, signal, title):       
        
        plt.figure(Plotter.figureNumOne)
        Plotter.figureNumOne = Plotter.figureNumOne + 1
        plt.plot(coreInstance.time, signal)
        plt.title(title)
        plt.savefig(path.join(outpath,"plot_{0}.png".format(Plotter.figureNumOne)))
        return


    # definition of a function that plots two signals together in time domain

    figureNumTwo = 10000

    @staticmethod
    def plotTwoSignals(coreInstance, signal1, label1, signal2, label2, title):       
        
        plt.figure(Plotter.figureNumTwo)
        Plotter.figureNumTwo = Plotter.figureNumTwo + 1
        plt.plot(coreInstance.time, signal1, label = label1)
        plt.plot(coreInstance.time, signal2, label = label2)
        plt.title(title)
        plt.legend()
        plt.savefig(path.join(outpath,"plot_{0}.png".format(Plotter.figureNumTwo)))
        return


    # definition of a function that plots a signal in frequency domain 

    figureNumThree = 20000

    @staticmethod
    def plotFourier(coreInstance, signal, frequencies, title):       
        
        plt.figure(Plotter.figureNumThree)
        Plotter.figureNumThree = Plotter.figureNumThree + 1
        the_instance = instance.CoreInstance()
        #transformed, frequencies = core.FourierTransform(the_instance, signal) 
        plt.plot(frequencies[1:len(signal)//2], 2.0/len(signal) * np.abs(signal[1:len(signal)//2]))
        plt.title(title) 
        plt.savefig(path.join(outpath,"plot_{0}.png".format(Plotter.figureNumThree)))   
        return

    figureNumFour = 30000

    @staticmethod
    def plotTwoFourier(coreInstance, signal1, frequencies1, label1, signal2, frequencies2, label2, title):       
        
        plt.figure(Plotter.figureNumFour)
        Plotter.figureNumFour = Plotter.figureNumFour + 1
        plt.plot(frequencies1[1:len(signal1)//2], 2.0/len(signal1) * np.abs(signal1[1:len(signal1)//2]), label = label1)
        plt.plot(frequencies2[1:len(signal2)//2], 2.0/len(signal2) * np.abs(signal2[1:len(signal2)//2]), label = label2)
        plt.title(title)
        plt.legend()
        plt.xlim(0, 60000)
        plt.savefig(path.join(outpath,"plot_{0}.png".format(Plotter.figureNumFour)))
        return
