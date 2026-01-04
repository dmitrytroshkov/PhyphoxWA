import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks, welch

st.set_page_config(page_title="Phyphox Walk Analyzer", layout="wide")
st.title("📱 Phyphox GPS + kiihtyvyys – analyysi ja visualisointi")

# =========================
# Load data
# =========================
acc = pd.read_csv("Accelerometer.csv")
gps = pd.read_csv("Location.csv")

# =========================
# Columns (Phyphox format)
# =========================
t_acc = acc["Time (s)"].values
ax = acc["X (m/s^2)"].values
ay = acc["Y (m/s^2)"].values
az = acc["Z (m/s^2)"].values

t_gps = gps["Time (s)"].values
lat = gps["Latitude (°)"].values
lon = gps["Longitude (°)"].values

# Normalize time
t_acc -= t_acc[0]
t_gps -= t_gps[0]

# =========================
# Sampling rate
# =========================
fs = 1 / np.median(np.diff(t_acc))
st.write(f"Accelerometer sampling rate: **{fs:.1f} Hz**")

# =========================
# Choose best axis
# =========================
variances = {
    "X": np.var(ax),
    "Y": np.var(ay),
    "Z": np.var(az),
}
axis = max(variances, key=variances.get)
acc_raw = {"X": ax, "Y": ay, "Z": az}[axis]

st.write(f"Selected acceleration component: **{axis}**")

# =========================
# Bandpass filter
# =========================
lowcut = 0.7
highcut = 3.0
b, a = butter(4, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype="band")
acc_filt = filtfilt(b, a, acc_raw)

# =========================
# Step count (peaks)
# =========================
peaks, _ = find_peaks(acc_filt, distance=fs/2, prominence=0.5*np.std(acc_filt))
steps_peaks = len(peaks)

# =========================
# Step count (Fourier / PSD)
# =========================
f, Pxx = welch(acc_raw, fs=fs)
mask = (f > 0.7) & (f < 3.0)
step_freq = f[mask][np.argmax(Pxx[mask])]
duration = t_acc[-1] - t_acc[0]
steps_fft = int(step_freq * duration)

# =========================
# GPS distance & speed
# =========================
R = 6371000
lat_r = np.radians(lat)
lon_r = np.radians(lon)

dlat = np.diff(lat_r)
dlon = np.diff(lon_r)

a = np.sin(dlat/2)**2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
distance = np.sum(R * c)

gps_time = t_gps[-1] - t_gps[0]
avg_speed = distance / gps_time

step_length = distance / steps_peaks

# =========================
# RESULTS
# =========================
st.subheader("Tulokset")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Askelmäärä (suodatettu)", steps_peaks)
c2.metric("Askelmäärä (Fourier)", steps_fft)
c3.metric("Kuljettu matka", f"{distance/1000:.3f} km")
c4.metric("Keskinopeus", f"{avg_speed*3.6:.2f} km/h")
c5.metric("Askelpituus", f"{step_length:.2f} m")

# =========================
# PLOTS
# =========================
st.subheader("Kuvaajat")

fig1 = plt.figure()
plt.plot(t_acc, acc_filt)
plt.plot(t_acc[peaks], acc_filt[peaks], "ro", markersize=3)
plt.xlabel("Aika (s)")
plt.ylabel("Suodatettu kiihtyvyys")
st.pyplot(fig1)

fig2 = plt.figure()
plt.semilogy(f, Pxx)
plt.xlabel("Taajuus (Hz)")
plt.ylabel("PSD")
st.pyplot(fig2)

st.subheader("Reitti kartalla")

st.map(pd.DataFrame({"lat": lat, "lon": lon}))
