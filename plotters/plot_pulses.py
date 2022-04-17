
from classes import core_functions as coreFuncs
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16


class Plotter(object):

    figureNum = 1

    @staticmethod
    def plot(coreInstance, signal, title):       
        
        plt.figure(Plotter.figureNum)
        Plotter.figureNum = Plotter.figureNum +1
        plt.plot(coreInstance.time, signal)
        plt.title(title)
        return

    @staticmethod
    def plotTwoSignals(coreInstance, signal1, label1, signal2, label2, title):       
        
        plt.figure(Plotter.figureNum)
        plt.plot(coreInstance.time, signal1, label = label1)
        plt.plot(coreInstance.time, signal2, label = label2)
        plt.title(title)
        plt.legend()
        return