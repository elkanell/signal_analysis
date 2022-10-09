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

import scipy

# Create a new instance of coreInstance object
the_instance = instance.CoreInstance()
core = base.CoreFuncs
radial_distance_values = [30,15,5] #the radial distances that we will use for the electron counting

# We use a gas: Ne and CH4 
# r_a = 0.315 cm, r_c = 30 cm, voltage = 2520 V
# The mobility for the ions of CH4 in a gas of Ne is of the order of 7.5 cm2 V-1 sec-1


# We take the raw pulse of two electrons in a given time distance a = 2 μs
raw_pulse = core.createTwoPulses(the_instance, distance = 2, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
plotter.Plotter.plot(
    the_instance, 
    raw_pulse,
    'Time [μs]',
    'Amplitude', 
    color = 'blue'
)
plt.xlim(1950, 2050)

# We take the convolution of the raw pulse with the preamplifier response
# We take the electronic signal of the two electrons
pulse = core.createElectronicSignal(the_instance, raw_pulse)
plotter.Plotter.plot(
    the_instance, 
    pulse,
    'Time [μs]',
    'Amplitude',
    color = 'purple'
)

# We add a white noise to the electronic signal 
signalWithNoise = core.createElectronicSignalWithNoise(the_instance, pulse)
plotter.Plotter.plot(
    the_instance, 
    signalWithNoise,
    'Time [μs]',
    'Amplitude',
    color = 'darkcyan'
)

# We take the deconvolution of the electronic signal with noise and the preamplifier response
# We take the initial raw pulse of the two electrons with noise
deconv =  core.deconvolutedSingalWithNoise(the_instance, signalWithNoise)
plotter.Plotter.plot(
    the_instance, 
    deconv,
    'Time [μs]',
    'Amplitude',
    color = 'green'
)
plt.xlim(1950, 2050)

# We take the deconvolution of the deconvolved signal with noise with the ion response
# We take delta-functions
electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, deconv, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
plotter.Plotter.plot(
    the_instance, 
    electron_signal,
    'Time [μs]',
    'Amplitude',
    color = 'red'
)
plt.xlim(1950, 2050)

indexes_max_above_noise = np.where(electron_signal > 550)

print(indexes_max_above_noise)
values_max = electron_signal[indexes_max_above_noise]
print(values_max)

number_of_max = len(values_max)
print(number_of_max)


tdiff = np.diff(indexes_max_above_noise, axis = 1)[0]
print(tdiff)

two_noise = 0
for differ in range(len(tdiff)):
    if tdiff[differ] >=2 and tdiff[differ] < 100:
        two_noise += 1


"""# We normalize the raw pulse and the delta-functions
# We plot them together 
raw_pulse_norm = core.normalizedSignal(raw_pulse)
electron_signal_norm = core.normalizedSignal(electron_signal)

plotter.Plotter.plotTwoSignals(
    the_instance, 
    electron_signal_norm,
    'deconvolution',
    raw_pulse_norm,
    'raw pulse',
    'Time [μs]',
    'Normalized Amplitude'
)


# We take the Discrete Fourier Transform of the electronic signal
transform, frequencies = core.FourierTransform(the_instance, pulse)
plotter.Plotter.plotFourier(
    the_instance,
    transform,
    frequencies,
    'Frequency [Hz]',
    'Amplitude'
)


# We take the Discrete Fourier Transform of the electronic signal with noise
transformWithNoise, frequenciesNoise = core.FourierTransform(the_instance, signalWithNoise)
plotter.Plotter.plotFourier(
    the_instance,
    transformWithNoise,
    frequenciesNoise,
    'Frequency [Hz]',
    'Amplitude'
)

plotter.Plotter.plotTwoFourier(
    the_instance, 
    transform,
    frequencies,
    'Fourier',
    transformWithNoise,
    frequenciesNoise,
    'Fourier with noise',
    'Frequency [Hz]',
    'Amplitude'
)
"""

# %%

# For two electrons:
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

    one_noise = 0
    two_noise = 0

    print('Distance:', rd, 'cm')
 
    for i in range (10):  #10
    # We take the raw pulse of two electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance
        el = 1

        raw_pulses, a_dist, r = core.createTwoPulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        plotter.Plotter.plot(
            the_instance,
            raw_pulses,
            'Time [μs]',
            'Amplitude',
            color = 'blue'
        )
        plt.xlim(1900, 2100)

        drift_time.append(int(r[0]))
        drift_time.append(int(r[1]))
         
        ### CHECK ###

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

        electronicSignalWithNoise = core.createElectronicSignalWithNoise(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        plotter.Plotter.plot(
            the_instance, 
            deconv,
            'Time [μs]',
            'Amplitude',
            color = 'green'
        )

        electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, deconv, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
        plotter.Plotter.plot(
            the_instance, 
            electron_signal,
            'Time [μs]',
            'Amplitude',
            color = 'red'
        )
        plt.xlim(1900, 2100)

        indexes_max_above_noise = np.where(electron_signal > 550)
        
        values_max = electron_signal[indexes_max_above_noise]
        

        number_of_max = len(values_max)
        

        tdiff = np.diff(indexes_max_above_noise, axis = 1)[0]
        

        for differ in range(len(tdiff)):
            if tdiff[differ] >=2 and tdiff[differ] < 100:
                el += 1
            
        if el == 1:
            one_noise += 1
        elif el == 2:
            two_noise += 1

    for i in range (890):  #890
    # We take the raw pulse of two electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance
        el = 1

        raw_pulses, a_dist, r = core.createTwoPulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        drift_time.append(int(r[0]))
        drift_time.append(int(r[1]))

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

        electronicSignalWithNoise = core.createElectronicSignalWithNoise(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        
        electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, deconv, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))

        indexes_max_above_noise = np.where(electron_signal > 550)
        
        values_max = electron_signal[indexes_max_above_noise]
        
        number_of_max = len(values_max)
    
        tdiff = np.diff(indexes_max_above_noise, axis = 1)[0]

        for differ in range(len(tdiff)):
            if tdiff[differ] >=2 and tdiff[differ] < 100:
                el += 1
            
        if el == 1:
            one_noise += 1
        elif el == 2:
            two_noise += 1


    for i in range (100):  #100
    # We take the raw pulse of two electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance
        el = 1

        raw_pulses, a_dist, r = core.createTwoPulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        drift_time.append(int(r[0]))
        drift_time.append(int(r[1]))

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
        plt.xlabel('Time [μs]')
        plt.ylabel('Amplitude')
        plt.plot(the_instance.time, raw_pulses, 'k', alpha=0.2)
        plt.xlim(1850, 2150)

        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createElectronicSignalWithNoise(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        
        electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, deconv, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))

        indexes_max_above_noise = np.where(electron_signal > 550)
        
        values_max = electron_signal[indexes_max_above_noise]
        
        number_of_max = len(values_max)
        
        tdiff = np.diff(indexes_max_above_noise, axis = 1)[0]
        
        for differ in range(len(tdiff)):
            if tdiff[differ] >=2 and tdiff[differ] < 100:
                el += 1
            
        if el == 1:
            one_noise += 1
        elif el == 2:
            two_noise += 1


    print('Number of one electron:', one_el)
    print('Number of two electrons:', two_el)

    num_of_electrons = ['1', '2']
    results = [one_el, two_el]
    plt.figure(plt.gcf().number+1)
    plt.bar(num_of_electrons, results, color = 'mediumvioletred', width = 0.2)

    #set title and x, y - axes labels
    plt.xlabel('Number of electrons')
    plt.ylabel('Frequency')
    
    #show plot to user
    plt.show()

    print('Number of one electron with noise:', one_noise)
    print('Number of two electrons with noise:', two_noise)

    num_of_electrons = ['1', '2']
    results = [one_noise, two_noise]
    plt.figure(plt.gcf().number+1)
    plt.bar(num_of_electrons, results, color = 'rebeccapurple', width = 0.2)

    #set title and x, y - axes labels
    plt.xlabel('Number of electrons')
    plt.ylabel('Frequency')
    
    #show plot to user
    plt.show()

    #print(drift_time)

    plt.figure(plt.gcf().number+1)
    # best fit of data
    (mu, sigma) = norm.fit(drift_time)

    # the histogram of the data
    n, bins, patches = plt.hist(drift_time, 20, density=True, stacked=True, facecolor='orchid', alpha=0.75)

    # add a 'best fit' line
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.show()

# %%
#We have three electrons that begin from different radial distances
for rd in radial_distance_values:
# the plots for 10 raw pulses of three electrons
    # print('radial_distance '+ rd)
    one_el = 0
    two_el = 0
    three_el = 0

    one_noise = 0
    two_noise = 0
    three_noise = 0

    print('Distance:', rd)

    for i in range (10):  #10
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance
        el = 1

        raw_pulses, a_dist, b_dist, r = core.createThreePulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        plotter.Plotter.plot(
            the_instance,
            raw_pulses,
            'Time [μs]',
            'Amplitude',
            color = 'blue'
        )
        plt.xlim(1900, 2100)

        drift_time.append(int(r[0]))
        drift_time.append(int(r[1]))
        drift_time.append(int(r[2]))


        ### CHECK ###

        # We check how many electrons we can see
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

        electronicSignalWithNoise = core.createElectronicSignalWithNoise(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        plotter.Plotter.plot(
            the_instance, 
            deconv,
            'Time [μs]',
            'Amplitude',
            color = 'green'
        )

        electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, deconv, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
        plotter.Plotter.plot(
            the_instance, 
            electron_signal,
            'Time [μs]',
            'Amplitude',
            color = 'red'
        )
        plt.xlim(1900, 2100)

        indexes_max_above_noise = np.where(electron_signal > 550)

        values_max = electron_signal[indexes_max_above_noise]
        
        number_of_max = len(values_max)
        
        tdiff = np.diff(indexes_max_above_noise, axis = 1)[0]
        
        for differ in range(len(tdiff)):
            if tdiff[differ] >=2 and tdiff[differ] < 100:
                el += 1

        if el == 1:
            one_noise += 1
        elif el == 2:
            two_noise += 1
        elif el == 3:
            three_noise += 1


    for i in range (10):  #890
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance
        el = 1

        raw_pulses, a_dist, b_dist, r = core.createThreePulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        
        drift_time.append(int(r[0]))
        drift_time.append(int(r[1]))
        drift_time.append(int(r[2]))

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

        electronicSignalWithNoise = core.createElectronicSignalWithNoise(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        
        electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, deconv, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
        
        indexes_max_above_noise = np.where(electron_signal > 550)
        
        values_max = electron_signal[indexes_max_above_noise]
        
        number_of_max = len(values_max)
        
        tdiff = np.diff(indexes_max_above_noise, axis = 1)[0]
           
        for differ in range(len(tdiff)):
            if tdiff[differ] >=2 and tdiff[differ] < 100:
                el += 1

        if el == 1:
            one_noise += 1
        elif el == 2:
            two_noise += 1
        elif el == 3:
            three_noise += 1

    for i in range (10):  #100
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance
        el = 1

        raw_pulses, a_dist, b_dist, r = core.createThreePulsesRandom(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6), radial_distance = rd)
        
        drift_time.append(int(r[0]))
        drift_time.append(int(r[1]))
        drift_time.append(int(r[2]))

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
        plt.xlabel('Time [μs]')
        plt.ylabel('Amplitude')
        plt.plot(the_instance.time, raw_pulses, 'k', alpha=0.2)
        plt.xlim(1850, 2150)
    
        electronicSignal = core.createElectronicSignal(the_instance, raw_pulses)

        electronicSignalWithNoise = core.createElectronicSignalWithNoise(the_instance, electronicSignal)

        deconv = core.deconvolutedSingalWithNoise(the_instance, electronicSignalWithNoise)
        
        electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, deconv, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 7.5*10**(-6))
        
        indexes_max_above_noise = np.where(electron_signal > 550)
        
        values_max = electron_signal[indexes_max_above_noise]
        
        number_of_max = len(values_max)
        
        tdiff = np.diff(indexes_max_above_noise, axis = 1)[0]
           
        for differ in range(len(tdiff)):
            if tdiff[differ] >=2 and tdiff[differ] < 100:
                el += 1

        if el == 1:
            one_noise += 1
        elif el == 2:
            two_noise += 1
        elif el == 3:
            three_noise += 1

    print('Number of one electron:', one_el)
    print('Number of two electrons:', two_el)
    print('Number of three electrons:', three_el)

    num_of_electrons = ['1', '2', '3']
    results = [one_el, two_el, three_el]
    plt.figure(plt.gcf().number+1)
    plt.bar(num_of_electrons, results, color = 'mediumvioletred', width = 0.2)

    #set title and x, y - axes labels
    plt.xlabel('Number of electrons')
    plt.ylabel('Frequency')
    
    #show plot to user
    plt.show()

    print('Number of one electron with noise:', one_noise)
    print('Number of two electrons with noise:', two_noise)
    print('Number of three electrons with noise:', three_noise)

    num_of_electrons = ['1', '2', '3']
    results = [one_noise, two_noise, three_noise]
    plt.figure(plt.gcf().number+1)
    plt.bar(num_of_electrons, results, color = 'rebeccapurple', width = 0.2)

    #set title and x, y - axes labels
    plt.xlabel('Number of electrons')
    plt.ylabel('Frequency')
    
    #show plot to user
    plt.show()

    plt.figure(plt.gcf().number+1)
    # best fit of data
    (mu, sigma) = norm.fit(drift_time)

    # the histogram of the data
    n, bins, patches = plt.hist(drift_time, 20, density=True, stacked=True, facecolor='orchid', alpha=0.75)

    # add a 'best fit' line
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.show()

#print("final arr:", arr)

df = pd.DataFrame(data = arr)
df.to_excel('pandas_to_excel.xlsx', sheet_name='signal analysis data test')

# %%

shmeia = np.array([-0.43, 4.26, -4.26, 2.13, -4.69, -0.86, -7.23, 1.28, -8.51, -2.13, -8.09, -2.98, -4.26, -7.23, 0.43, -4.26, 1.70, -5.11, 3.40, 6.81, 2.98])
std = np.std(shmeia)
print(std)
    

# %%
def plotGaussianDistribution(mean, std_dev, graph_color, graph_label):
    x_min = 0.0
    x_max = mean * 2

    x = np.linspace(x_min, x_max, 1000)
    y = scipy.stats.norm.pdf(x, mean, std_dev)

    plt.plot(x, y, label=graph_label)
    # plt.fill_between(x, y, color=graph_color, alpha='0.5')

    plt.title('Gaussian Distribution')
    plt.ylim(0, 0.04)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.savefig("gaussian_distribution.png")
    plt.show()

    


# Execute only if run as a script 
mean = 100
std_dev = 20
graph_color = "black"
graph_label = " "
plotGaussianDistribution(mean, std_dev, graph_color, graph_label) 

# %%

one_el = np.array([1000])
two_el = np.array([0])

one_el_mean = np.mean(one_el)
two_el_mean = np.mean(two_el)

one_el_std = np.std(one_el)
two_el_std = np.std(two_el)

electrons = ['1', '2']
x_pos =np.arange(len(electrons))
CTEs = [one_el_mean, two_el_mean]
error = [one_el_std, two_el_std]

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha = 0.5, ecolor = 'black', capsize = 10)
ax.set_xsticks(x_pos)

plt.tight_layout()
plt.show()

# %%