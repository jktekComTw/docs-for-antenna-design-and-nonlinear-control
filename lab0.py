# Create a complete Python script as a .py file

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# === Parameters ===
fs = 100e6  # Sampling frequency: 100 MHz
duration = 500e-6  # Duration: 0.5 ms
t = np.arange(0, duration, 1/fs)

# === Frequencies ===
f_carrier = 2e5  # Carrier frequency: 5 GHz
f_mod = 4e3      # Modulation frequency: 4 kHz

# === Signal generation ===
modulating = np.cos(2 * np.pi * f_mod * t)
carrier = np.cos(2 * np.pi * f_carrier * t)
# am_signal = (1 + 0.9 * modulating) * carrier  # AM signal
am_signal = (modulating) * carrier
# === Time-domain plot (one period of modulating signal) ===
samples_per_period = int(5e5)
t_ns = t[:samples_per_period] * 1e10
# t_ns = t[:samples_per_period]
am_wave = am_signal[:samples_per_period]
envlope=modulating[:samples_per_period]
plt.figure(figsize=(12, 4))
plt.plot(t_ns, am_wave)
plt.plot(t_ns, envlope)
plt.title("Time Domain: AM Wave (5 GHz carrier × 4 kHz modulating signal)")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# # === Frequency-domain plot ===
# n = len(am_signal)
# freqs = fftfreq(n, 1/fs)
# spectrum = np.abs(fft(am_signal)) / n

# plt.figure(figsize=(10, 4))
# plt.plot(freqs / 1e6, 20*np.log10(spectrum + 1e-12))
# plt.title("Frequency Domain: AM Spectrum")
# plt.xlabel("Frequency (MHz)")
# plt.ylabel("Magnitude (dB)")
# plt.xlim(4900, 5100)  # Focus on 5 GHz area
# plt.grid(True)
# plt.tight_layout()
# plt.show()



