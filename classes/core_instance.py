import numpy as np

SRATE = 1e+6 # MHz
DURATION = 4e+3 # in μs
DT = 1e+6 / SRATE # in μs

PULSE_MAX_DURATION = 50 # in μs

N_EL = 2 # the initial number of the electrons
GAIN = 100 # average gain
FANO = 0.25 # a value close to argon

class CoreInstance(object):   
    #Define properties 
    time  = 0
    n = 0
    sigmaF = 0 
    gains = 0
    #distance = 0
    len_preamp_response = 0
    preamp_fall_time = 0
    preamp_response = 0
    
    def __init__(self, **kwargs):
        # Initialize properties' values 
        self.time  = np.arange(0, DURATION, DT) # 0 to 4000 μs with step 1 μs 
        self.n = len(self.time)
        self.sigmaF = np.sqrt(FANO*GAIN) # sigma avec Fano sigma = sqrt(fano*mean)
        self.gains = np.round(np.random.normal(GAIN, self.sigmaF, N_EL))
        # self.distance = distance
        self.len_preamp_response = int(self.n/2)
        self.preamp_fall_time = 125
        self.preamp_response = np.exp(- DT * np.arange( self.len_preamp_response ) / self.preamp_fall_time)

        return
