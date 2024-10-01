# -*- coding: utf-8 -*-


import numpy as np
from scipy.signal import butter, lfilter,chirp,iirpeak,freqs

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def narrow_bandpass(lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    f0=(high+low)/2

    
    b, a = iirpeak(f0, 10, fs=fs)
    return b, a


def narrow_bandpass_filter(data, lowcut, highcut, fs,):
    b, a = narrow_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # w, h = freqs(b, a, worN=6000)
    return b,a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b,a= butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_highpass(highcut, fs, order=4):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='high')
    return b, a

def butter_highpass_filter(data, highcut, fs, order=4):
    b, a = butter_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
