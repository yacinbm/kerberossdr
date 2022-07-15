from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
import scipy.signal
import math

def compute_intersection(radius_1, radius_2, distance):
    result = None
    # Check if intersection exist
    x = (radius_1**2 - radius_2**2 + distance**2)/(2*distance)
    if x <= radius_1:
        y = np.sqrt(radius_1**2 - x**2)
        result = (x,y)
    return result

block_size = 131072/4
Fs = 2.056e6/4
fmcw_slope = (1.40e9/1.4e-3)/2
c = 2.998e8
binWidth = (Fs/block_size)
num_channels = 2
samples = np.load("samples.npy", mmap_mode="r")
num_frames = samples.shape[1]//block_size
frames = np.array(np.array_split(samples, num_frames,1))
Fmax = 100e3

buffer_size = 6
buffer = np.array([[3]*buffer_size]*num_channels, dtype=np.float64)
distances = np.array([3]*num_channels)
pos_distances =np.array([3]*num_channels)
neg_distances = np.array([3]*num_channels)
maha = [0]*num_channels
inter_element_dist = 0.45

#Transition Matrix
A = np.array([1.0]*num_channels)
#Observation Matrix
C = np.array([1.0]*num_channels)
#Process Noise Covariance
Rww = np.array([1]*num_channels)
#Measurement Noise Covariance
Rvv = np.array([2.5]*num_channels)
#state vector
x = np.array([[0.0]*num_channels]*num_frames, dtype=np.float64)
#Covariance Matrix
P = np.array([0.7]*num_channels)
#Weighted MD vector
MDw = np.array([0.0]*num_channels)
#Initial Covariance Value
I = np.array([1.0]*num_channels)

coord_test = []
for i in np.arange(0,num_frames):
    window = np.hamming(block_size//decimationFactor + 1)
    fft = abs(abs(np.fft.fft(frames[i,4:6,:]*window))-abs(np.fft.fft(frames[i-1,4:6,:]*window)))
    fft = np.fft.fftshift(fft)
    pos_fft = fft[:, block_size//2:]
    neg_fft = np.flip(fft[:, 0:block_size//2])
    fft = (pos_fft + neg_fft)/2
    pos_peaks = np.argmax(pos_fft,1)
    neg_peaks = np.argmax(neg_fft,1)
    peaks = np.argmax(fft,1)
    freq = np.fft.fftfreq(block_size, 1/Fs)
    freq = freq[0:block_size//2]
    dist = c*(freq/fmcw_slope)/2
    buffer = np.roll(buffer, 1)
    distance = ((dist[pos_peaks] + dist[neg_peaks])/2)
    buffer[:, 0] = distance
    avg = np.sum(buffer, axis=1)/buffer_size
    maha = (np.sqrt((distance-avg)**2))
    distances = np.vstack([distances, distance])
    pos_distances = np.vstack([pos_distances, dist[pos_peaks]])
    neg_distances = np.vstack([neg_distances, dist[neg_peaks]])

    # Kalmann Filtering
    """
    Prediction
    """
    x[i] =  A*x[i-1,:]
    P = (A**2)*P + Rww
    
    """
    Innovation
    """
    outliers = maha>0.6
    distance[outliers] = np.sign(distance[outliers]-x[i,outliers])*0.3 + x[i,outliers]
    e = distance - C*x[i,:]
    Ree = (C**2) * P + Rvv
    #Weighted MD
    MDw = 1/(1+(np.exp(-maha) + 0))
    
    #New Measurement Noise Covariance
    Rvv= 8*MDw
    #Kalman gain
    K = P * C/Ree
    """
    Update
    """
    x[i] = x[i] + (K**2) * e
    P = (I - K*C) * P

    # Intersections
    for indx in list(combinations(range(num_channels), 2)):
        coord = compute_intersection(x[i][indx[0]], x[i][indx[0]], (indx[1]-indx[0])*inter_element_dist)
    coord_test.append(coord)

    time = np.linspace(0, (num_frames*block_size/2)/Fs, num_frames)
    #plt.plot(time, np.array(coord_test)[:])
    ind_max = int(100e3//binWidth)
    for ch in range(pos_fft.shape[0]):
        plt.plot(freq[:ind_max], pos_fft[ch,:ind_max])
    plt.pause(0.1)
    plt.clf()


time = np.linspace(0, (num_frames*block_size/2)/Fs, num_frames)
#plt.plot(time, np.array(coord_test)[:])
plt.plot(time, distances[:,0])
#plt.plot(time, pos_distances)
#plt.plot(time, neg_distances)
#plt.plot(time, x)



plt.ylabel("Distance (m)")
plt.xlabel("Temps (s)")
plt.legend(("Avg", "Positive", "Negative", "Filtered Avg"))
plt.show()