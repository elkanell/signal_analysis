
import scipy.signal
import numpy as np

import scipy.fftpack
from classes import core_instance as InstanceRef
from array import array


class CoreFuncs(object):

    def __init__(self, **kwargs):
        return
    
    # definition of the functions: ion current and induced current
    @staticmethod
    def ion_current(time, r_a = 0.1, r_c = 15, voltage = 2000, pressure = 1, mobility_0 = 2.e-6):
        """ J.Derre derivation of the formula
            distance [cm]
            pressure [bar]
            time [usec]
            voltage [V]
        """
        rho = (r_c * r_a) / (r_c - r_a)
        alpha = ( 2.e-6 ) * voltage * rho / pressure # constant that depends on the gas and the anode radius
        
        # return rho * ( 1 / r_a - 1 / (r_a**3 + 3 * alpha * time)**(1/3) )
        return alpha * rho * (r_a**3 + 3 * alpha * time)**(-4/3)

    @staticmethod
    def induced_current(t, d, z, ve, vi):
        #Units used cm, us, fC
        q = 1.6E-4 #fC
        
        tel_max = d-z / ve
        ti_max = z/vi
        
        if t < tel_max:
            return -q*t*(ve+vi)/d
        elif d-z / ve < t < ti_max:
            return -q*(d-z+vi*t)/d
        else: 
            return -q
    

    @staticmethod
    def createPulses(instance, num=2):
        
        #indexes = indexes[:n_el]
        # the distance of the two pulses
        # create two pulses for the two electrons in a distance a  
        a = instance.distance
        s1 = np.zeros(instance.n)
        s2 = np.zeros(instance.n)
        raw_pulse = np.zeros(instance.n)

        for num in range (0, 2):
            index = 2000
            zeros_part = np.zeros(int(index))
            time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)

            if num == 0:
                ic1 = instance.gains[num] * CoreFuncs.ion_current(time_temp)
                pulse_temp1 = np.concatenate( (zeros_part, ic1), axis=0) 
                s1 = s1 + pulse_temp1 
            if num == 1:
                index = index + a
                zeros_part = np.zeros(int(index))
                time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)
                ic2 = instance.gains[num] * CoreFuncs.ion_current(time_temp) 
                pulse_temp2 = np.concatenate( (zeros_part, ic2), axis=0) 
                s2 = s2 + pulse_temp2 

        # # the two pulses in one array
        raw_pulse = s1 + s2
        return raw_pulse 
        
    @staticmethod
    def createOnePulse(coreInstance):
        raw_pulse = CoreFuncs.createPulses(coreInstance)       
        raw_pulse_temp = np.concatenate( (np.zeros(1000), raw_pulse), axis=0 )
        pulse =  scipy.signal.fftconvolve(raw_pulse_temp, coreInstance.preamp_response, "same")
        pulse = np.delete(pulse, range(4000, 5000), axis=0)
        return pulse

    @staticmethod
    def createOnePulseWithNoise(coreInstance):
        #white noise
        noiseamp = 2

        # add a white noise to the signal

        noise = noiseamp * np.random.randn(coreInstance.n)
        pulse = CoreFuncs.createOnePulse(coreInstance)
        signal = pulse + noise
        return signal

    @staticmethod
    def deconvoluteSingalWithNoise(coreInstance, signal):
        length = len(signal)
        signal_padded = np.zeros( length + len(coreInstance.preamp_response) - 1 )
        signal_padded[:length] = signal

        deconv, _ = scipy.signal.deconvolve(signal_padded, coreInstance.preamp_response)
        return deconv


    @staticmethod
    def deconcoluteSignal(coreInstance, raw_pulse):
        ion_resp = CoreFuncs.ion_current(InstanceRef.DT*np.arange(coreInstance.len_preamp_response), r_a = 0.1, r_c = 15, voltage = 2000, pressure = 1, mobility_0 = 2.e-6)

        length = len(raw_pulse)
        pulse_padded = np.zeros(length + len(ion_resp) - 1)
        pulse_padded[:length] = raw_pulse

        electron_signal, residual = scipy.signal.deconvolve(pulse_padded, ion_resp)
        return electron_signal, residual


    @staticmethod
    def normalizeSignal(signal):
        
        #heights for the normalization
        
        signal_height = array( 'f', [0])
        signal_height_position = np.argmax(signal)
        signal_height[0] = signal[signal_height_position] 
        signal_norm = signal / (signal_height[0])

        return signal_norm

        # raw_pulse_height = array( 'f', [0])
        # raw_pulse_height_position = np.argmax(raw_pulse)
        # raw_pulse_height[0] = raw_pulse[raw_pulse_height_position] 
        # raw_pulse_norm = raw_pulse / (raw_pulse_height[0])
        #normalized electron signal and raw pulse
        
        
       
