
import scipy.signal
import numpy as np

import scipy.fftpack

from scipy.fft import fft, fftfreq
from classes import core_instance as InstanceRef
from array import array


class CoreFuncs(object):

    def __init__(self, **kwargs):
        return
    
    # definition of the functions: ion current and induced current
    @staticmethod
    def ion_current(time, r_a, r_c, voltage, pressure, mobility_0):
        """ J.Derre derivation of the formula
            distance [cm]
            pressure [bar]
            time [μs]
            voltage [V]
        """
        rho = (r_c * r_a) / (r_c - r_a)
        alpha = ( mobility_0 ) * voltage * rho / pressure # constant that depends on the gas and the anode radius
        
        # return rho * ( 1 / r_a - 1 / (r_a**3 + 3 * alpha * time)**(1/3) )
        return alpha * rho * (r_a**3 + 3 * alpha * time)**(-4/3)

    @staticmethod
    def induced_current(t, d, z, ve, vi):
        # Units used cm, μs, fC
        q = 1.6E-4 # fC
        
        tel_max = d-z / ve
        ti_max = z/vi
        
        if t < tel_max:
            return -q*t*(ve+vi)/d
        elif d-z / ve < t < ti_max:
            return -q*(d-z+vi*t)/d
        else: 
            return -q
    

    # definition of a function that gives the pulses of two electrons in a given distance a 
    @staticmethod
    def createTwoPulses(instance, distance, r_a, r_c, voltage, pressure, mobility_0, num = 2):
        a = distance
        s1 = np.zeros(instance.n)
        s2 = np.zeros(instance.n)
        raw_pulse = np.zeros(instance.n)

        for num in range (0, 2):
            index = 2000
            zeros_part = np.zeros(int(index))
            time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)

            if num == 0:
                ic1 = instance.gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0)
                pulse_temp1 = np.concatenate( (zeros_part, ic1), axis=0) 
                s1 = s1 + pulse_temp1 
            if num == 1:
                index = index + a
                zeros_part = np.zeros(int(index))
                time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)
                ic2 = instance.gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0) 
                pulse_temp2 = np.concatenate( (zeros_part, ic2), axis=0) 
                s2 = s2 + pulse_temp2 

        # the two pulses in one array
        raw_pulse = s1 + s2
        return raw_pulse 


    #definition of a function that gives the pulses of three electrons in random distances between them 
    @staticmethod
    def createThreePulses(instance, r_a, r_c, voltage, pressure, mobility_0, num=3): 
     
        n_el = 3 # number of the electrons

        sigmaF = np.sqrt(InstanceRef.FANO*InstanceRef.GAIN) # sigma avec Fano sigma = sqrt(fano*mean)
        gains = np.round(np.random.normal(InstanceRef.GAIN, sigmaF, n_el))
        
        #for i in range (10):
        s = np.zeros(instance.n)
        s1 = np.zeros(instance.n)
        raw_pulse = np.zeros(instance.n)

        index = 2000
        zeros_part = np.zeros(int(index))
        time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)

        r = np.random.normal(2000.0, 100.0, 3) # the distance of the pulses
        print(r)

        num = 0

        while num < 3:
            index = r[num]
            zeros_part = np.zeros(int(index))
            time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)
            ic = gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0)
            pulse_temp = np.concatenate( (zeros_part, ic), axis=0)
            s = s + pulse_temp
            num = num + 1
            raw_pulse = raw_pulse + s

        return raw_pulse 


    # definition of a function that gives the convolution of the pulses with the preamplifier response    
    @staticmethod
    def createElectronicSignal(coreInstance, raw_pulse):
        raw_pulse_temp = np.concatenate( (np.zeros(1000), raw_pulse), axis=0 )
        pulse =  scipy.signal.fftconvolve(raw_pulse_temp, coreInstance.preamp_response, "same")
        pulse = np.delete(pulse, range(4000, 5000), axis=0)
        return pulse


    # definition of a function that adds a white noise to the electronic signal
    @staticmethod
    def createElectronicSignalWithNoise(coreInstance, pulse):
        #white noise
        noiseamp = 2

        # add a white noise to the signal

        noise = noiseamp * np.random.randn(coreInstance.n)
        signal = pulse + noise
        return signal


    # definition of a function that gives the deconvolution of the electronic signal with noise 
    # and the preamplifier response
    @staticmethod
    def deconvolutedSingalWithNoise(coreInstance, signal):
        length = len(signal)
        signal_padded = np.zeros( length + len(coreInstance.preamp_response) - 1 )
        signal_padded[:length] = signal

        deconv, _ = scipy.signal.deconvolve(signal_padded, coreInstance.preamp_response)
        return deconv


    # definition of a function that gives the deconvolution of the raw pulse with the ion response
    @staticmethod
    def deconvolutionWithIonResponse(coreInstance, raw_pulse, r_a, r_c, voltage, pressure, mobility_0):
        ion_resp = CoreFuncs.ion_current(InstanceRef.DT*np.arange(coreInstance.len_preamp_response), r_a, r_c, voltage, pressure, mobility_0)

        length = len(raw_pulse)
        pulse_padded = np.zeros(length + len(ion_resp) - 1)
        pulse_padded[:length] = raw_pulse

        electron_signal, residual = scipy.signal.deconvolve(pulse_padded, ion_resp)
        return electron_signal, residual


    # definition of a function that gives the normalized signal 
    @staticmethod
    def normalizedSignal(signal):
        
        #heights for the normalization
        
        signal_height = array( 'f', [0])
        signal_height_position = np.argmax(signal)
        signal_height[0] = signal[signal_height_position] 
        signal_norm = signal / (signal_height[0])

        return signal_norm


    # definition of a function that computes the discrete Fourier Transform 
    @staticmethod
    def FourierTransform(coreInstance, signal):
        N = len(signal)
        d = 1.0 / InstanceRef.SRATE

        yf = fft(signal)
        xf = fftfreq(N, d)

        return yf, xf



        
        
    
