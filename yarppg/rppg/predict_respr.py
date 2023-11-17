from scipy.interpolate import interp1d
from scipy.fftpack import fft,fftfreq
from scipy.signal import hann, hilbert, butter,sosfiltfilt

import numpy as np

def calc_resp_rate2(vs, ts, mindist=0.35, calc_method="fft", sampling_rate=100):
    f = interp1d(ts, vs, kind='cubic')
    ts_new = np.linspace(ts[0], ts[-1], len(ts)*10)  # Upsampling by 10 times
    vs_new = f(ts_new)
    window = hann(len(vs_new))

    vs_new = vs_new * window
    low_freq = 6 / 60  
    high_freq = 24 / 60  

    if calc_method == "fft":
        yf = fft(vs_new)
        xf = fftfreq(len(vs_new), ts_new[1] - ts_new[0])

        mask = (xf > low_freq) & (xf < high_freq)
        dominant_freq = xf[mask][np.argmax(np.abs(yf[mask]))]

    elif calc_method == "hilbert":
        sos = butter(2, [low_freq, high_freq], btype='band', fs=sampling_rate, output='sos')

        filtered_vs = sosfiltfilt(sos, vs_new)
        analytic_signal = hilbert(filtered_vs)

        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        instantaneous_frequency = np.diff(instantaneous_phase)
        instantaneous_frequency /= (2.0 * np.pi)

        dominant_freq = np.median(instantaneous_frequency) * 1000

    else:
        raise ValueError("Invalid calculation method provided!")

    return dominant_freq * 60  

