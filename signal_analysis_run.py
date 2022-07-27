# %%
from plotters import plot_pulses as plotter

from classes import signal_analysis_functions as base
from classes import core_instance as instance

import numpy as np
import pandas as pd
import openpyxl

# Create a new instance of coreInstance object
the_instance = instance.CoreInstance()
core = base.CoreFuncs

# We use a gas: Ne and CH4 
# r_a = 0.63 cm, r_c = 60 cm, voltage = 2520 V
# The mobility for the ions of CH4 in a gas of Ne is of the order of 2 cm2 V-1 sec-1


# We take the raw pulse of two electrons in a given time distance a = 100 μs
raw_pulse = core.createTwoPulses(the_instance, distance = 10, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 2*10**(-6))
plotter.Plotter.plot(
    the_instance, 
    raw_pulse,
    'The two pulses in a given distance a = 10 μs'
)


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
electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, raw_pulse, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 2*10**(-6))
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

rows, cols = (4000, 1)
arr = []
time_col = 0

for i in range(rows):
    col = []
    for j in range(cols):
        if j == 0:
            col.append(time_col)
            time_col = time_col + 1
        arr.append(col)
print("array:", arr)

for i in range (10):
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance

    raw_pulses = core.createThreePulses(the_instance, r_a = 0.315, r_c = 30, voltage = 2520, pressure = 3.1, mobility_0 = 2*10**(-6))
    plotter.Plotter.plot(
        the_instance,
        raw_pulses,
        'The pulses of the three electrons'
    )

    rows, cols = (4000, 1)
    height_array = []

    for i in range(rows):
        col = []
        for j in range(cols):
            if j == 0:
                col.append(raw_pulses[i])
            height_array.append(col)
    print("height array:", height_array)

    arr = np.column_stack((arr, height_array))

    rawPulseWithNoise = core.createNoiseForRawPulse(the_instance, raw_pulses)
    plotter.Plotter.plot(
        the_instance, 
        rawPulseWithNoise,
        'The raw pulse with noise'
    )


print("final arr:", arr)

df = pd.DataFrame(data = arr)
df.to_excel('pandas_to_excel.xlsx', sheet_name='signal analysis data')


# %%



