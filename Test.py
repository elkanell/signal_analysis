 # %%

from plotters import plot_pulses as plotter

from classes import core_functions as base
from classes import core_instance as instance

distance = 1000
# Create a new instance of coreInstance object
instance1 = instance.CoreInstance(distance)
core = base.CoreFuncs


raw_pulse = core.createPulses(instance1)
plotter.Plotter.plot(
    instance1, 
    raw_pulse,
    "The two pulses in a distance {}".format(instance1.distance)
)



pulse = core.createOnePulse(instance1)
plotter.Plotter.plot(
    instance1, 
    pulse,
    'The electronic signal of the two electrons'
)

singalWithNoise = core.createOnePulseWithNoise(instance1)
plotter.Plotter.plot(
    instance1, 
    singalWithNoise,
    'The electronic signal with noise'
)

decov =  core.deconvoluteSingalWithNoise(instance1, singalWithNoise)
plotter.Plotter.plot(
    instance1, 
    decov,
    'The pulse of the two electrons with noise'
)

electron_signal, residual =  core.deconcoluteSignal(instance1, raw_pulse)
plotter.Plotter.plot(
    instance1, 
    electron_signal,
    'The pulse of the two electrons with noise'
)

raw_pulse_norm = core.normalizeSignal(raw_pulse)
electron_signal_norm = core.normalizeSignal(electron_signal)

plotter.Plotter.plotTwoSignals(
    instance1, 
    electron_signal_norm,
    'deconvolution',
    raw_pulse_norm,
    'raw pulse',
    'The deconvolution and the raw pulse (normalized)'
)


# %%

from scipy.fft import fft, fftfreq
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt
import numpy as np

# signal_length = len(pulse)
sample_rate = 1e+6
# dt = 1e+6/sample_rate

# print(signal_length)
# df = 1/signal_length

# plt.figure(37)
plt.plot(instance1.time, singalWithNoise)

f = fft(singalWithNoise)


# n_t = len(instance1.time)

# freqs = df*scipy.arange(-(n_t - 1)/2, (n_t - 1)/2+1, dtype = 'd')
# print(len(freqs))
# n_freq = len(freqs)

N = 4e+3
xf = fftfreq(int(N), 1 / sample_rate)
plt.figure(446)

plt.plot(xf, np.abs(f))

# plt.xlim([-0.1, 0.1])



# %%
