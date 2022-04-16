
from classes import core_functions as coreFuncs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


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
