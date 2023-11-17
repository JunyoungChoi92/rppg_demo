import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import scipy.signal
from scipy.signal import welch, periodogram
from scipy.interpolate import UnivariateSpline
from ..rppg.processors.filtering import filter_signal
from .predict_respr import calc_resp_rate2
import math 
def bpm_from_inds(inds, ts):
    """Calculate heart rate (in beat/min) from indices and time vector

    Args:
        inds (`1d array-like`): indices of heart beats
        ts (`1d array-like`): time vector corresponding to indices

    Returns:
        float: heart rate in beats per minute (bpm)
    """

    if len(inds) < 2:
        return np.nan

    return 60. / np.mean(np.diff(ts[inds]))


def get_sampling_rate(ts):
    """Calculate sampling rate from time vector
    """
    return 1. / np.mean(np.diff(ts))


def from_peaks(vs, ts, mindist=0.35):
    """Calculate heart rate by finding peaks in the given signal

    Args:
        vs (`1d array-like`): pulse wave signal
        ts (`1d array-like`): time vector corresponding to pulse signal
        mindist (float): minimum distance between peaks (in seconds)

    Returns:
        float: heart rate in beats per minute (bpm)
    """

    if len(ts) != len(vs) or len(ts) < 2:
        return np.nan
    f = 1. / np.mean(np.diff(ts))
    peaks, _ = scipy.signal.find_peaks(vs, distance=int(f*mindist))

    return bpm_from_inds(peaks, ts)

def get_rr_interval(vs, ts, mindist=0.35):
    if len(ts) != len(vs) or len(ts) < 2:
        return np.nan
    f = get_sampling_rate(ts)
    peaks, _ = scipy.signal.find_peaks(vs, distance=int(f*mindist))
    intervals = np.diff([ts[p] for p in peaks])

    return intervals

def calculate_rr_intervals(vs, ts, mindist=0.35):
    # Calculate the minimum number of samples between peaks based on time
    dt = ts[1] - ts[0]  # Assuming uniform time steps
    min_samples = int(mindist / dt)
    
    # Detect peaks in the pulse wave signal
    peaks, _ = scipy.signal.find_peaks(vs, distance=min_samples)
    
    # Extract the time stamps of the detected peaks
    peak_times = ts[peaks]
    
    # Calculate the RR intervals
    rr_intervals = np.diff(peak_times)
    
    return rr_intervals
def normalize_list(nums):
    min_val = min(nums)
    max_val = max(nums)

    return [(num - min_val) / (max_val - min_val) for num in nums]

def calc_resp_rate(vs, ts, mindist=0.35, calc_method="fft", sampling_rate=1000):
    """this function will calculate respiratory rate(number of breaths per minute). the logic is to upsample the detected rr_intervals
    using cubic slpine interplation, apply threshold at min 6 ~ max 24 breaths per min,
    at the end extract respiration rate from the signal. the calc_method can be fft, welch and peridogram. the spline interploation,
    welch and periodogram are available in scipy and numpy libraries.
    """
    rr_intervals = get_rr_interval(vs, ts, mindist)
    independant_var = rr_intervals
    dependant_var = np.linspace(0, len(rr_intervals), len(rr_intervals))

    # cubic spline interpolation
    spline = UnivariateSpline(dependant_var, independant_var, k=3)
    new_data = np.linspace(0, len(rr_intervals), int(np.sum(rr_intervals)))
    breathing_signal = spline(new_data)
    
    filtered_breath_signal = filter_signal(data=breathing_signal, cutoff=[0.1, 0.4], filtertype="bandpass", sample_rate=sampling_rate)

    if calc_method == "fft":
        len_data = len(filtered_breath_signal)
        freq = np.fft.fftfreq(len_data, d= 1/sampling_rate)
        freq = freq[range(int(len_data/2))]
        
        psd_var = np.fft.fft(filtered_breath_signal)/len_data
        psd_var = psd_var[range(int(len_data/2))]
        psd = np.power(np.abs(psd_var), 2)

    elif calc_method == "welch":
        freq, psd = welch(filtered_breath_signal, fs=sampling_rate, nperseg=len(filtered_breath_signal))
        
    elif calc_method == "periodogram":
        freq, psd = periodogram(filtered_breath_signal, fs=sampling_rate, window="hamming", scaling="spectrum")
    else:
        raise ValueError("calc_method should be one of fft, welch and periodogram")

    freq = np.multiply(normalize_list(freq), 18) + 6

    print("freq: ", freq)
    print("psd: ", psd)

    ts_measure = freq[np.argmax(psd)]
    
    return ts_measure

def from_fft(vs, ts):
    """Calculate heart rate as most dominant frequency in pulse signal

    Args:
        vs (`1d array-like`): pulse wave signal
        ts (`1d array-like`): time vector corresponding to pulse signal

    Returns:
        float: heart rate in beats per minute (bpm)
    """

    f = get_sampling_rate(ts)
    vf = np.fft.fft(vs)
    xf = np.linspace(0.0, f/2., len(vs)//2)
    return 60 * xf[np.argmax(np.abs(vf[:len(vf)//2]))]


class HRCalculator(QObject):
    new_hr = pyqtSignal(float)

    def __init__(self, parent=None, update_interval=30, winsize=300,
                 filt_fun=None, hr_fun=None):
        QObject.__init__(self, parent)

        self._counter = 0
        self.update_interval = update_interval
        self.winsize = winsize
        self.filt_fun = filt_fun
        self.hr_fun = from_peaks
        if hr_fun is not None and callable(hr_fun):
            self.hr_fun = hr_fun

    def update(self, rppg):
        self._counter += 1
        if self._counter >= self.update_interval:
            self._counter = 0
            ts = rppg.get_ts(self.winsize)
            vs = next(rppg.get_vs(self.winsize))
            if self.filt_fun is not None and callable(self.filt_fun):
                vs = self.filt_fun(vs)
            self.new_hr.emit(self.hr_fun(vs, ts))

class RRCalculator(QObject):
    new_respr = pyqtSignal(float)

    def __init__(self, parent=None, update_interval=30, winsize=300,
                 filt_fun=None, rr_fun=None):
        QObject.__init__(self, parent)

        self._counter = 0
        self.update_interval = update_interval
        self.winsize = winsize
        self.filt_fun = filt_fun
        self.rr_fun = calc_resp_rate2
        if rr_fun is not None and callable(rr_fun):
            self.rr_fun = rr_fun
        
    def update(self, rppg):
        self._counter += 1
        if self._counter >= self.update_interval:
            self._counter = 0
            ts = rppg.get_ts(self.winsize)
            vs = next(rppg.get_vs(self.winsize))

            if self.filt_fun is not None and callable(self.filt_fun):
                vs = self.filt_fun(vs)
                
            self.new_respr.emit(self.rr_fun(vs, ts, sampling_rate=1000, calc_method="hilbert"))