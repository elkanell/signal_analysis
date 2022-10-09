
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


    # definition of a function that gives the pulses of two electrons in random distance (gaussian with the radial distance)
    @staticmethod
    def createTwoPulsesRandom(instance, r_a, r_c, voltage, pressure, mobility_0, radial_distance, num=2):
        n_el = 2 # number of the electrons

        #sigmaF = np.sqrt(InstanceRef.THETA*InstanceRef.GAIN) # sigma avec Fano sigma = sqrt(fano*mean)
        gains = np.round(np.random.exponential(InstanceRef.GAIN, n_el))
        
        #for i in range (10):
        s = np.zeros(instance.n)
        s1 = np.zeros(instance.n)
        s2 = np.zeros(instance.n)
        raw_pulse = np.zeros(instance.n)

        index = 2000
        zeros_part = np.zeros(int(index))
        time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)

        time_of_arrival = 2000
        sigma = ((radial_distance / r_c) ** 3) * 20 # diffusion time

        mean_diffusion_time = 100

        r = np.random.normal(mean_diffusion_time, sigma, 2) # arrival time

        r = np.sort([(r[0]), (r[1])])

        a = r[1] - r[0]
        a_dist = int(time_of_arrival + a)
        #print(a_dist)

        num = 0

        while num < 2:
            if num == 0:
                ic1 = gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0)
                pulse_temp1 = np.concatenate( (zeros_part, ic1), axis=0) 
                s1 = s1 + pulse_temp1 
                num = num + 1
            if num == 1:
                index = a_dist
                zeros_part = np.zeros(int(index))
                time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)
                ic2 = gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0) 
                pulse_temp2 = np.concatenate( (zeros_part, ic2), axis=0) 
                s2 = s2 + pulse_temp2 
                num = num + 1

        raw_pulse = np.add(s1, s2)

        return raw_pulse, a_dist, r
        


    ### CHECK ###

    # definition of a function that gives the pulses of three electrons in random distance (gaussian with the radial distance)
    @staticmethod
    def createThreePulsesRandom(instance, r_a, r_c, voltage, pressure, mobility_0, radial_distance, num=3):
        n_el = 3 # number of the electrons

        #sigmaF = np.sqrt(InstanceRef.THETA*InstanceRef.GAIN) # sigma avec Fano sigma = sqrt(fano*mean)
        gains = np.round(np.random.exponential(InstanceRef.GAIN, n_el))
        
        #for i in range (10):
        s = np.zeros(instance.n)
        s1 = np.zeros(instance.n)
        s2 = np.zeros(instance.n)
        s3 = np.zeros(instance.n)
        raw_pulse = np.zeros(instance.n)

        index = 2000
        zeros_part = np.zeros(int(index))
        time_temp = np.arange(0, InstanceRef.DT*(instance.n-index), InstanceRef.DT)

        time_of_arrival = 2000
        sigma = ((radial_distance / r_c) ** 3) * 20 # diffusion time

        mean_diffusion_time = 100

        r = np.random.normal(mean_diffusion_time, sigma, 3) # arrival time

        r = np.sort([(r[0]), (r[1]), (r[2])])

        a = r[1] - r[0]
        a_dist = int(time_of_arrival + a)
        b = r[2] - r[1]
        b_dist = int(a_dist + b)
        #print(a_dist)
        #print(b_dist)

        num = 0

        while num < 3:
            if num == 0:
                ic1 = gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0)
                pulse_temp1 = np.concatenate( (zeros_part, ic1), axis=0) 
                s1 = s1 + pulse_temp1 
                num = num + 1
            if num == 1:
                index1 = a_dist
                zeros_part = np.zeros(int(index1))
                time_temp = np.arange(0, InstanceRef.DT*(instance.n-index1), InstanceRef.DT)
                ic2 = gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0) 
                pulse_temp2 = np.concatenate( (zeros_part, ic2), axis=0) 
                s2 = s2 + pulse_temp2 
                num = num + 1
            if num == 2:
                index2 = b_dist
                zeros_part = np.zeros(int(index2))
                time_temp = np.arange(0, InstanceRef.DT*(instance.n-index2), InstanceRef.DT)
                ic3 = gains[num] * CoreFuncs.ion_current(time_temp, r_a, r_c, voltage, pressure, mobility_0) 
                pulse_temp3 = np.concatenate( (zeros_part, ic3), axis=0) 
                s3 = s3 + pulse_temp3 
                num = num + 1
        
        # raw_pulse = np.add(s1, s2, s3)

        arr = np.array([s1, 
          s2,
          s3])
        raw_pulse = np.add(0, arr.sum(axis=0))

        return raw_pulse, a_dist, b_dist, r
        



    # definition of a function that gives the convolution of the pulses with the preamplifier response    
    @staticmethod
    def createElectronicSignal(coreInstance, raw_pulse):
        raw_pulse_temp = np.concatenate( (np.zeros(1000), raw_pulse), axis=0 )
        pulse =  scipy.signal.fftconvolve(raw_pulse_temp, coreInstance.preamp_response, "same")
        pulse = np.delete(pulse, range(4000, 5000), axis=0)
        return pulse



    ### CHECK ###

    # definition of a function that adds a white noise to the electronic signal
    @staticmethod
    def createElectronicSignalWithNoise(coreInstance, pulse):
        #white noise
        #noiseamp = 2

        # add a white noise to the signal

        #noise = noiseamp * np.random.randn(coreInstance.n)
        noise = np.random.normal(0, 4.3, coreInstance.n)
        signal = pulse + noise
        return signal

    # definition of a function that adds a noise to the raw pulse
    @staticmethod
    def createNoiseForElectronicSignal(coreInstance, pulse):
        #white noise
        noiseamp = 2
        
        #noise
        noise = noiseamp * np.random.randn(coreInstance.n) + np.sin(2*np.pi*(10**-4)*coreInstance.n) + np.sin(2*np.pi*(5*10**-3)*coreInstance.n)
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




    ### CHECK ###

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
        xf = fftfreq(N, d)[:N//2]
        
        return yf, xf



        
        
    
