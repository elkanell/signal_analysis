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




# %%
