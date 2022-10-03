# %%
from plotters import plot_pulses as plotter
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.stats import norm

from classes import signal_analysis_functions as base
from classes import core_instance as instance

import numpy as np
import pandas as pd
import openpyxl

# Create a new instance of coreInstance object
the_instance = instance.CoreInstance()
core = base.CoreFuncs
radial_distance_values = [30,20,10] #the radial distances that we will use for the electron counting

# We use a gas: Ne and CH4 
# r_a = 0.315 cm, r_c = 30 cm, voltage = 2520 V
# The mobility for the ions of CH4 in a gas of Ne is of the order of 7.5 cm2 V-1 sec-1


# We take the raw pulse of two electrons in a given time distance a = 2 μs
raw_pulse = core.createTwoPulses(the_instance, distance = 2, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
plotter.Plotter.plot(
    the_instance, 
    raw_pulse,
    'The two pulses in a given distance a = 2 μs'
)
plt.xlim(1950, 2050)

# We take the convolution of the raw pulse with the preamplifier response
# We take the electronic signal of the two electrons
pulse = core.createElectronicSignal(the_instance, raw_pulse)
plotter.Plotter.plot(
    the_instance, 
    pulse,
    'The electronic signal of the two electrons'
)


# We add a white noise to the electronic signal 
signalWithNoise = core.createElectronicSignalWithNoise(the_instance, pulse)
plotter.Plotter.plot(
    the_instance, 
    signalWithNoise,
    'The electronic signal with noise'
)


# We take the deconvolution of the electronic signal with noise and the preamplifier response
# We take the initial raw pulse of the two electrons with noise
deconv =  core.deconvolutedSingalWithNoise(the_instance, signalWithNoise)
plotter.Plotter.plot(
    the_instance, 
    deconv,
    'The pulse of the two electrons with noise'
)


# We take the deconvolution of the raw pulse with the ion response
# We take delta-functions
electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, raw_pulse, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
plotter.Plotter.plot(
    the_instance, 
    electron_signal,
    'The deconvolution of the ion response and the raw pulse'
)


# We normalize the raw pulse and the delta-functions
# We plot them together 
raw_pulse_norm = core.normalizedSignal(raw_pulse)
electron_signal_norm = core.normalizedSignal(electron_signal)

plotter.Plotter.plotTwoSignals(
    the_instance, 
    electron_signal_norm,
    'deconvolution',
    raw_pulse_norm,
    'raw pulse',
    'The deconvolution and the raw pulse (normalized)'
)


# We take the Discrete Fourier Transform of the electronic signal
transform, frequencies = core.FourierTransform(the_instance, pulse)
plotter.Plotter.plotFourier(
    the_instance,
    transform,
    frequencies,
    'Fourier Transform of the electronic signal'
)


# We take the Discrete Fourier Transform of the electronic signal with noise
transformWithNoise, frequenciesNoise = core.FourierTransform(the_instance, signalWithNoise)
plotter.Plotter.plotFourier(
    the_instance,
    transformWithNoise,
    frequenciesNoise,
    'Fourier Transform of the electronic signal with noise'
)

plotter.Plotter.plotTwoFourier(
    the_instance, 
    transform,
    frequencies,
    'Fourier',
    transformWithNoise,
    frequenciesNoise,
    'Fourier with noise',
    'The Fourier Transforms'
)


# %%
drift_time = []

rows, cols = (4000, 1)
height_array = []
arr = []
time_col = 0

# save the data to an excel
for k in range(rows):
    col = []
    for j in range(cols):
        if j == 0:
            col.append(time_col)
            time_col = time_col + 1
            arr.append(col)
 
# We have two electrons that begin from different radial distances
for rd in radial_distance_values:
    one_el = 0
    two_el = 0
 
    for i in range (10):  #10
    # We take the raw pulse of two electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance

        raw_pulses, a_dist = core.createTwoPulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        plotter.Plotter.plot(
            the_instance,
            raw_pulses,
            'The pulses of the two electrons (for ' +str(rd) +' cm)'
        )
        plt.xlim(1950, 2100)

        drift_time.append(a_dist)

        # We check how many electrons we can see
        if a_dist >= (2000+2):
            two_el += 1
        else:
            one_el += 1

        rows, cols = (4000, 1)
        height_array = []

        for k in range(rows):
            col = []
            for j in range(cols):
                if j == 0:
                    col.append(raw_pulses[k])
                    height_array.append(col)
   
        arr = np.column_stack((arr, height_array))

        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createNoiseForElectronicSignal(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        plotter.Plotter.plot(
            the_instance, 
            deconv,
            'The raw pulse with noise'
        )


    for i in range (10):  #890
    # We take the raw pulse of two electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance

        raw_pulses, a_dist = core.createTwoPulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        drift_time.append(a_dist)

        # We check how many electrons we can see
        if a_dist >= (2000+2):
            two_el += 1
        else:
            one_el += 1

        rows, cols = (4000, 1)
        height_array = []

        for k in range(rows):
            col = []
            for j in range(cols):
                if j == 0:
                    col.append(raw_pulses[k])
                    height_array.append(col)
   
        arr = np.column_stack((arr, height_array))

        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createNoiseForElectronicSignal(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
    
    for i in range (10):  #100
    # We take the raw pulse of two electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance

        raw_pulses, a_dist = core.createTwoPulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        drift_time.append(a_dist)

        # We check how many electrons we can see
        if a_dist >= (2000+2):
            two_el += 1
        else:
            one_el += 1

        rows, cols = (4000, 1)
        height_array = []

        for k in range(rows):
            col = []
            for j in range(cols):
                if j == 0:
                    col.append(raw_pulses[k])
                    height_array.append(col)
   
        arr = np.column_stack((arr, height_array))

        plt.figure(40000)
        plt.plot(the_instance.time, raw_pulses, 'k', alpha=0.2)
        plt.xlim(1900, 2200)

        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createNoiseForElectronicSignal(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)

    print('Distance:', rd)
    print('Number of one electron:', one_el)
    print('Number of two electron:', two_el)

    num_of_electrons = ['1', '2']
    results = [one_el, two_el]
    plt.figure(plt.gcf().number+1)
    plt.bar(num_of_electrons, results, color = '#4CAF50', width = 0.2)

    #set title and x, y - axes labels
    plt.title('For radial distance = '+str(rd) +' cm')
    plt.xlabel('Number of electrons')
    plt.ylabel('Appearances')
    
    #show plot to user
    plt.show()



# %%
#We have three electrons that begin from different radial distances
for rd in radial_distance_values:
# the plots for 10 raw pulses of three electrons
    # print('radial_distance '+ rd)
    one_el = 0
    two_el = 0
    three_el = 0

    for i in range (10):  #10
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance

        raw_pulses, a_dist, b_dist = core.createThreePulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        plotter.Plotter.plot(
            the_instance,
            raw_pulses,
            'The pulses of the three electrons (for ' +str(rd) +' cm)'
        )
        plt.xlim(1950, 2100)

        drift_time.append(a_dist)
        drift_time.append(b_dist)

        if a_dist == b_dist: 
            if a_dist== 2000:
                one_el += 1
            elif a_dist >= (2000+2):
                two_el += 1
        else:
            if (a_dist >= (2000+2) and b_dist >= (2000+2)):
                if (abs(a_dist-b_dist) >= 2):
                    three_el += 1
                else:
                    two_el += 1
            elif ((a_dist >= (2000+2) and b_dist < (2000+2)) or (a_dist < (2000+2) and b_dist >= (2000+2))):
                if (abs(a_dist-b_dist) >=2):
                    two_el +=1
                else:
                    one_el +=1
            else:
                one_el += 1

        rows, cols = (4000, 1)
        height_array = []

        for k in range(rows):
            col = []
            for j in range(cols):
                if j == 0:
                    col.append(raw_pulses[k])
                height_array.append(col)
        #print("height array:", height_array)

        arr = np.column_stack((arr, height_array))

        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createNoiseForElectronicSignal(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        plotter.Plotter.plot(
            the_instance, 
            deconv,
            'The raw pulse with noise'
        )

    for i in range (10):  #890
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance

        raw_pulses, a_dist, b_dist = core.createThreePulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        
        drift_time.append(a_dist)
        drift_time.append(b_dist)

        if a_dist == b_dist: 
            if a_dist== 2000:
                one_el += 1
            elif a_dist >= (2000+2):
                two_el += 1
        else:
            if (a_dist >= (2000+2) and b_dist >= (2000+2)):
                if (abs(a_dist-b_dist) >= 2):
                    three_el += 1
                else:
                    two_el += 1
            elif ((a_dist >= (2000+2) and b_dist < (2000+2)) or (a_dist < (2000+2) and b_dist >= (2000+2))):
                if (abs(a_dist-b_dist) >=2):
                    two_el +=1
                else:
                    one_el +=1
            else:
                one_el += 1

        rows, cols = (4000, 1)
        height_array = []

        for k in range(rows):
            col = []
            for j in range(cols):
                if j == 0:
                    col.append(raw_pulses[k])
                height_array.append(col)
        #print("height array:", height_array)

        arr = np.column_stack((arr, height_array))

        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createNoiseForElectronicSignal(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)

    for i in range (10):  #100
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance

        raw_pulses, a_dist, b_dist = core.createThreePulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        
        drift_time.append(a_dist)
        drift_time.append(b_dist)

        if a_dist == b_dist: 
            if a_dist== 2000:
                one_el += 1
            elif a_dist >= (2000+2):
                two_el += 1
        else:
            if (a_dist >= (2000+2) and b_dist >= (2000+2)):
                if (abs(a_dist-b_dist) >= 2):
                    three_el += 1
                else:
                    two_el += 1
            elif ((a_dist >= (2000+2) and b_dist < (2000+2)) or (a_dist < (2000+2) and b_dist >= (2000+2))):
                if (abs(a_dist-b_dist) >=2):
                    two_el +=1
                else:
                    one_el +=1
            else:
                one_el += 1

        rows, cols = (4000, 1)
        height_array = []

        for k in range(rows):
            col = []
            for j in range(cols):
                if j == 0:
                    col.append(raw_pulses[k])
                height_array.append(col)
        #print("height array:", height_array)

        arr = np.column_stack((arr, height_array))

        plt.figure(5000)
        plt.plot(the_instance.time, raw_pulses, 'k', alpha=0.2)
        plt.xlim(1900, 2200)
    
        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createNoiseForElectronicSignal(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        
    print('Distance:', rd)
    print('Number of one electron:', one_el)
    print('Number of two electrons:', two_el)
    print('Number of three electrons:', three_el)

    num_of_electrons = ['1', '2', '3']
    results = [one_el, two_el, three_el]
    plt.figure(plt.gcf().number+1)
    plt.bar(num_of_electrons, results, color = 'red', width = 0.2)

    #set title and x, y - axes labels
    plt.title('For radial distance = '+str(rd) +' cm')
    plt.xlabel('Number of electrons')
    plt.ylabel('Appearances')
    
    #show plot to user
    plt.show()

#print("final arr:", arr)

df = pd.DataFrame(data = arr)
df.to_excel('pandas_to_excel.xlsx', sheet_name='signal analysis data test')
    

# %%
import scipy
from scipy.stats import norm

 
plt.figure(40001)
# the histogram of the data
_, bins, _ = plt.hist(drift_time, 20, density=1, alpha=0.5)
mu, sigma = scipy.stats.norm.fit.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line)
plt.title("Histogram of drift time")
plt.show()

print(mu, sigma)

# %%



