 # %%

from examples import print_2_pulses as example1

from classes import core_functions as core

distance = 1000
core = core.CoreFuncs

example1.ExampleTwoPulses.printExample(core, distance)
example1.ExampleTwoPulses.printExample(core, distance)


t,raw_pulse = core.createPulses(distance)
example1.ExampleTwoPulses.printExample2(core, raw_pulse)


# %%
