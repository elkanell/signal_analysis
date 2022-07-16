# %%
from plotters import plot_pulses as plotter

from classes import signal_analysis_functions as base
from classes import core_instance as instance

# Create a new instance of coreInstance object
the_instance = instance.CoreInstance()
core = base.CoreFuncs

# We use a gas: Ne and CH4 
# r_a = 0.63 cm, r_c = 60 cm, voltage = 2520 V
# The mobility for the ions of CH4 in a gas of Ne is of the order of 2 cm2 V-1 sec-1


# We take the raw pulse of two electrons in a given time distance a = 100 μs
raw_pulse = core.createTwoPulses(the_instance, distance = 10, r_a = 0.63, r_c = 60, voltage = 2520, pressure = 3.1, mobility_0 = 2*10**(-6))
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
electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, raw_pulse, r_a = 0.63, r_c = 60, voltage = 2520, pressure = 3.1, mobility_0 = 2*10**(-6))
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

for i in range (10):
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution with sigma that depends on the radial distance
    raw_pulses = core.ThreePulsesRadialDistance(the_instance, r_a = 0.63, r_c = 60, voltage = 2520, pressure = 3.1, mobility_0 = 2*10**(-6))
    plotter.Plotter.plot(
        the_instance,
        raw_pulses,
        'The pulses of the three electrons'
    )



# mexri edw, ta parakatw mallon de tha xreiastoun



"""# %%
# We use a gas: Ar and CH4 
# r_a = 0.1 cm, r_c = 70 cm, voltage = 2000 V
# The mobility for the ions of CH4 in a gas of Ar is 1.87 cm2 V-1 sec-1


# We take the raw pulse of two electrons in a given time distance a = 100 μs
raw_pulse = core.createTwoPulses(the_instance, distance = 100, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 1.87*10**(-6))
plotter.Plotter.plot(
    the_instance, 
    raw_pulse,
    'The two pulses in a given distance a = 100 μs'
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
electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, raw_pulse, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 1.87*10**(-6))
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
for i in range (1):
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution 
    raw_pulses = core.createThreePulses(the_instance, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 1.87*10**(-6), diffusion_coefficient = 0.04 * 10**(-6))
    plotter.Plotter.plot(
    the_instance,
    raw_pulses,
    'The pulses of the three electrons'
    )

    # We take the convolution of the raw pulse with the preamplifier response
    # We take the electronic signal of the three electrons
    pulses = core.createElectronicSignal(the_instance, raw_pulses)
    plotter.Plotter.plot(
        the_instance, 
        pulses,
        'The electronic signal of the three electrons'
    )


    # We add a white noise to the electronic signal 
    singalWithNoise = core.createElectronicSignalWithNoise(the_instance, pulses)
    plotter.Plotter.plot(
        the_instance, 
        singalWithNoise,
        'The electronic signal with noise'
    )


    # We take the deconvolution of the electronic signal with noise and the preamplifier response
    # We take the initial raw pulse of the three electrons with noise
    deconv =  core.deconvolutedSingalWithNoise(the_instance, singalWithNoise)
    plotter.Plotter.plot(
        the_instance, 
        deconv,
        'The pulse of the three electrons with noise'
    )


    # We take the deconvolution of the raw pulse with the ion response
    # We take delta-functions
    electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, raw_pulses, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 1.87*10**(-6))
    plotter.Plotter.plot(
        the_instance, 
        electron_signal,
        'The deconvolution of the ion response and the raw pulse'
    )


    # We normalize the raw pulse and the delta-functions
    # We plot them together 
    raw_pulse_norm = core.normalizedSignal(raw_pulses)
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
    transform, frequencies = core.FourierTransform(the_instance, pulses)
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
# We use a gas: Isobutane and He
# r_a = 0.1 cm, r_c = 70 cm, voltage = 2000 V
# The mobility for the ions of He in a gas of IsoC4H10 is    


# We take the raw pulse of two electrons in a given time distance a = 100 μs
raw_pulse = core.createTwoPulses(the_instance, distance = 100, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 0.55*10**(-6))
plotter.Plotter.plot(
    the_instance, 
    raw_pulse,
    'The two pulses in a given distance a = 100 μs'
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
singalWithNoise = core.createElectronicSignalWithNoise(the_instance, pulse)
plotter.Plotter.plot(
    the_instance, 
    singalWithNoise,
    'The electronic signal with noise'
)


# We take the deconvolution of the electronic signal with noise and the preamplifier response
# We take the initial raw pulse of the two electrons with noise
deconv =  core.deconvolutedSingalWithNoise(the_instance, singalWithNoise)
plotter.Plotter.plot(
    the_instance, 
    deconv,
    'The pulse of the two electrons with noise'
)


# We take the deconvolution of the raw pulse with the ion response
# We take delta-functions
electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, raw_pulse, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 0.55*10**(-6))
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

for i in range (1):
    # We take the raw pulse of three electrons in a random time distance between them 
    # We take the random distance from a normal distribution 
    raw_pulses = core.createThreePulses(the_instance, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 0.55*10**(-6), diffusion_coefficient=1)
    plotter.Plotter.plot(
    the_instance,
    raw_pulses,
    'The pulses of the three electrons'
    )


    # We take the convolution of the raw pulse with the preamplifier response
    # We take the electronic signal of the three electrons
    pulses = core.createElectronicSignal(the_instance, raw_pulses)
    plotter.Plotter.plot(
        the_instance, 
        pulses,
        'The electronic signal of the three electrons'
    )


    # We add a white noise to the electronic signal 
    singalWithNoise = core.createElectronicSignalWithNoise(the_instance, pulses)
    plotter.Plotter.plot(
        the_instance, 
        singalWithNoise,
        'The electronic signal with noise'
    )


    # We take the deconvolution of the electronic signal with noise and the preamplifier response
    # We take the initial raw pulse of the three electrons with noise
    deconv =  core.deconvolutedSingalWithNoise(the_instance, singalWithNoise)
    plotter.Plotter.plot(
        the_instance, 
        deconv,
        'The pulse of the three electrons with noise'
    )


    # We take the deconvolution of the raw pulse with the ion response
    # We take delta-functions
    electron_signal, residual =  core.deconvolutionWithIonResponse(the_instance, raw_pulses, r_a = 0.1, r_c = 70, voltage = 2000, pressure = 1, mobility_0 = 0.55*10**(-6))
    plotter.Plotter.plot(
        the_instance, 
        electron_signal,
        'The deconvolution of the ion response and the raw pulse'
    )


    # We normalize the raw pulse and the delta-functions
    # We plot them together 
    raw_pulse_norm = core.normalizedSignal(raw_pulses)
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
    transform, frequencies = core.FourierTransform(the_instance, pulses)
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

# %%"""
