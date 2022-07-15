from ast import expr_context
from audioop import avg
import logging
import math
from turtle import distance
import numpy as np
import time
from scipy.signal import find_peaks
from scipy.constants import speed_of_light
from PyQt5.QtCore import QThread, pyqtSignal
from itertools import combinations
from npy_append_array import NpyAppendArray
from sympy import combsimp

# Import Kerberos receiver
from _receiver.hydra_receiver import ReceiverRTLSDR

# Import the pyArgus module
from pyargus import directionEstimation as de

def compute_intersection(radius_1, radius_2, distance):
    result = None
    # Check if intersection exists
    x = (radius_1**2 - radius_2**2 + distance**2)/(2*distance)
    diff = radius_1**2 - x**2
    if (diff > 0):
        y = np.sqrt(radius_1**2 - x**2)
        result = (x,y)
    return result

class Tracker(QThread):
    signal_scan_done = pyqtSignal()
    signal_sync_ready = pyqtSignal()
    signal_period = pyqtSignal(float)
    signal_spectrum_ready = pyqtSignal()
    signal_aoa_ready = pyqtSignal()
    signal_distance_ready = pyqtSignal()
    signal_backgrnd_ready = pyqtSignal()

    def __init__(self, receiver:ReceiverRTLSDR) -> None:
        #TODO: 
        # Determine the optimal (or at least the role of) spectrum_sample_size, xcorr_sample_size & aoa_sample_size
        
        # Class initialization
        super(QThread, self).__init__()
        self.receiver = receiver
        # Scan variables
        self.scan_start_f = 24e6
        self.scan_stop_f = 240e6
        self.scan_step = 0  # Initialize before start
        # Control flags
        self.running = False
        self.en_noise_meas = False
        self.en_scan = False
        self.en_sync = False
        self.en_sample_offset_sync = False
        self.en_calib_iq = False
        self.en_aoa_FB_avg = False
        self.en_estimate_aoa = False
        self.en_estimate_distance = False
        self.en_save_samples = False
        self.en_capture_background = False
        # Cross Correlation variab27071les
        self.xcorr_sample_size = 2**18
        self.xcorr = np.ones((self.receiver.channel_number-1,self.xcorr_sample_size*2), dtype=np.complex64)        
        # Distance estimation variables
        self.freqOffset = 9000 # Offset due to the length of the cables, for both channels
        self.chirp_span = 1.4e9
        self.chirp_time = (1.4e-3)/2 # Triangle sweep, so the span is done twice per chirp
        self.chirp_slope = 1.1551e12 # Measured in a test @3.8m
        self.minFreq = 1e3 # Under 1khz, it is just noise between the antennas
        self.num_antennas = 2
        self.distances = np.array([[0]*100]*self.num_antennas, dtype=np.float64)
        self.decimationFactor = 1
        self.maha_sample_size = 6 # Mahalanobis distance buffer size
        self.maha_buffer = np.array([[0]*self.maha_sample_size]*self.num_antennas, dtype=np.float64)
        self.A = np.array([1.0]*self.num_antennas, dtype=np.float64)       #Transition Matrix
        self.C = np.array([1.0]*self.num_antennas, dtype=np.float64)       #Observation Matrix
        self.Rww = np.array([1.0]*self.num_antennas, dtype=np.float64)     #Process Noise Covariance
        self.Rvv = np.array([2.5]*self.num_antennas, dtype=np.float64)     #Measurement Noise Covariance
        self.MDw = np.array([0.0]*self.num_antennas, dtype=np.float64)     #Weighted MD vector
        self.P = np.array([0.7]*self.num_antennas, dtype=np.float64)       #Initial Covariance Value
        self.x = np.array([0.0]*self.num_antennas, dtype=np.float64)       #Initialization of the vector state
        self.I = np.array([1.0]*self.num_antennas, dtype=np.float64)       # Identity
        # Direction of Arrival (aoa) variables
        self.antenna_distance = 0.05 # Antennas distance in m
        self.aoa_sample_size = 2**15 
        
        # Result Vectors
        self.target_coordonates = np.array([0]*self.num_antennas)
        self.spectrum_sample_size = (self.receiver.block_size//2)//self.decimationFactor
        self.spectrum = np.zeros((self.receiver.channel_number+1,self.spectrum_sample_size), dtype=np.float32)
        self.noise_lvl = np.empty(self.receiver.channel_number, dtype=np.float32)
        self.noise_var= np.empty(self.receiver.channel_number, dtype=np.float32)
        self.delay_log= np.array([[0]]*(self.receiver.channel_number-1))
        self.phase_log= np.array([[0]]*(self.receiver.channel_number-1))
        self.peak_freq = np.array([],dtype=np.float32)
        self.peak_pwr = np.array([],dtype=np.float32)
        self.spectrum_log = np.array([[]]*(self.receiver.channel_number+1), dtype=np.float32) # idx 0 are the frequencies and the rest are the results
        self.aoa_MUSIC_res = np.ones(181)
        self.aoa_theta = np.arange(0,181,1)
        self.background_spectrum = np.zeros((self.receiver.channel_number+1, self.spectrum_sample_size), dtype=np.float32)

        self.spectrum_bcpk = np.array([[0]*self.spectrum_sample_size]*(self.receiver.channel_number+1), dtype=np.float32)

    def run(self):
        # Lock the receiver
        if not self.receiver.mutex.tryLock(timeout=0):
            logging.warning("Tracker: Could not lock the receiver, returning...")
            return

        self.running = True
        while self.running:
            start_time = time.time()
            
            # Acquire Background (always on for now)
            if self.en_capture_background:
                # Save last spectrum as background
                self.background_spectrum = np.copy(self.spectrum)
                self.en_capture_background = False

            self.receiver.download_iq_samples()

            # Compute FFT
            self.spectrum[0, :] = np.fft.fftfreq(self.spectrum_sample_size, 1/(self.receiver.fs/self.decimationFactor))
            window = np.hanning(self.spectrum_sample_size)
            self.spectrum[1:,:] = np.abs(np.fft.fft(self.receiver.iq_samples[:,::self.decimationFactor]*window)) # Compute FFT
            self.spectrum = np.fft.fftshift(self.spectrum, 1) # Center the spectrum around 0
        
            # Compute Distance
            self.foreground_spectrum = abs(abs(self.spectrum[1:self.num_antennas+1,:]) - abs(self.background_spectrum[1:self.num_antennas+1,:]))
            
            # Limit the spectrum around the area of interest (over 1khz delta)
            binWidth = (self.receiver.fs/self.decimationFactor) / self.spectrum_sample_size
            minIndx = int(self.minFreq // binWidth)
            self.pos_fft = self.foreground_spectrum[:, self.spectrum_sample_size//2:]
            self.neg_fft = np.flip(self.foreground_spectrum[:, :self.spectrum_sample_size//2], axis=1)
            positive_peaks = np.argmax(self.pos_fft[:,  minIndx:], 1) # Get peaks around the carrier, should be symmetric. Used to correct carrier error.
            negative_peaks = np.argmax(self.neg_fft[:,  minIndx:], 1)
            dist = (self.spectrum[0,self.spectrum_sample_size//2:] - self.freqOffset) * speed_of_light/(self.chirp_slope*2)
            avg_distance = (dist[positive_peaks] + dist[negative_peaks])/2
            # Mahalanobis Distance 
            self.maha_buffer = np.roll(self.maha_buffer, 1)
            self.maha_buffer[:, 0] = avg_distance
            maha_distance = np.sqrt((avg_distance - np.sum(self.maha_buffer, 1)/self.maha_sample_size)**2)
            self.foreground_spectrum = self.pos_fft + self.neg_fft
            self.signal_spectrum_ready.emit()
            # Kalman Filtering
            """
            Prediction
            """
            self.x = self.x * self.A
            self.P = (self.A**2)*self.P + self.Rww

            """
            Innovation
            """
            outliers = maha_distance>0.6 # Detect outliers
            avg_distance[outliers] = np.sign(avg_distance[outliers] - self.x[outliers])*0.3 + self.x[outliers]
            e = avg_distance - self.C*self.x
            Ree = (self.C**2) * self.P + self.Rvv
            # Weighted Maha
            self.MDw = 1/(1+np.exp(-maha_distance))
            #New Measurement Noise Covariance
            self.Rvv= 8*self.MDw
            #Kalman gain
            K = self.P * self.C / Ree
            """
            Update
            """
            self.x = self.x + K**2 * e
            self.P = (1-K*self.C) * self.P

            self.distances = np.roll(self.distances, 1)
            self.distances[:,0] = self.x[0:self.num_antennas]

            # Correlate the circles
            self.target_coordonates = np.array([0]*self.num_antennas, np.float64)
            combs = list(combinations(range(self.num_antennas), 2))
            for indx in combs:
                intersection = compute_intersection(self.x[indx[0]], self.x[indx[1]], (indx[1]-indx[0])*self.antenna_distance)
                if intersection is not None:
                    self.target_coordonates += np.array(intersection)/len(combs)
            self.signal_distance_ready.emit()

            # Characterize noise
            if self.receiver.is_noise_source_on and self.en_noise_meas:
                self.noise_var = np.var(self.receiver.iq_samples,1)
                self.noise_lvl = np.amax(self.spectrum[1:,:],1)
                self.en_noise_meas = False
            
            # IQ calibration request
            if self.en_calib_iq:
                # IQ correction
                for m in range(self.receiver.channel_number):
                    self.receiver.iq_corrections[m] *= np.size(self.receiver.iq_samples[0, :])/(np.dot(self.receiver.iq_samples[m, :],self.receiver.iq_samples[0, :].conj()))
                c = np.sqrt(np.sum(np.abs(self.receiver.iq_corrections)**2))
                self.receiver.iq_corrections = np.divide(self.receiver.iq_corrections, c)
                logging.debug(f"Corrections: {self.receiver.iq_corrections}")
                self.en_calib_iq = False

            # Send sync to receiver
            if self.en_sample_offset_sync:
                self.receiver.set_sample_offsets(self.delay_log[:,-1])
                self.en_sample_offset_sync = False
                self.signal_sync_ready.emit()

            # Spectrum scan to find peaks
            if self.en_scan:
                self.spectrum_log = np.concatenate((self.spectrum_log,self.spectrum),1)
                if self.center_f >= self.scan_stop_f:
                    self.en_scan = False
                    logging.info(f"Scan duration: {time.time()-scan_start_time}")
                    self.signal_scan_done.emit()
                
                if self.center_f == self.scan_start_f:
                        scan_start_time = time.time()
                        self.peak_freq = np.array([],dtype=np.float32)
                        self.peak_pwr = np.array([],dtype=np.float32)
                
                # Find the peaks
                #TODO: Find optimal distance
                peak_idx, param = find_peaks(self.spectrum[1,:], height=self.noise_lvl[0], distance=2000)
                if np.any(peak_idx):
                    self.peak_freq = np.concatenate((self.peak_freq,self.spectrum[0,peak_idx]*1e6+self.center_f))
                    self.peak_pwr = np.concatenate((self.peak_pwr,param["peak_heights"]))
                
                # compute next center freq
                next_freq = self.center_f + self.scan_step
                
                if next_freq < self.scan_stop_f:
                    self.center_f = next_freq
                else:
                    self.center_f = self.scan_stop_f

                logging.debug(f"Next scan freq: {self.center_f}")

                # Update receiver
                self.receiver.reconfigure_tuner(self.center_f, self.receiver.fs*self.receiver.decimation_ratio, self.receiver.receiver_gain)  

            if self.en_save_samples:
                fileName = "samples.npy"
                with NpyAppendArray(fileName) as npaa:
                    npaa.append(self.spectrum)


            self.signal_period.emit(time.time() - start_time)
        # Release the receiver
        self.receiver.mutex.unlock()

        np.save("samples", self.spectrum_bcpk)

    def estimate_aoa(self):
        logging.info("Tracker: Estimating aoa")

        iq_samples = self.receiver.iq_samples[:, 0:self.aoa_sample_size]
        # Calculating spatial correlation matrix
        R = de.corr_matrix_estimate(iq_samples.T, imp="fast")

        if self.en_aoa_FB_avg:
            R=de.forward_backward_avg(R)

        M = np.size(iq_samples, 0)

        # Generate antenna coordonates
        wave_length = speed_of_light / self.receiver.center_f
        self.aoa_inter_elem_space = self.antenna_distance/wave_length
        logging.debug(f"Inter elem: {self.aoa_inter_elem_space}")
        self.aoa_theta =  np.linspace(-45,45,46)#np.linspace(-90,90,181)
        x = np.zeros(M)
        y = np.arange(M) * self.aoa_inter_elem_space            
        scanning_vectors = de.gen_scanning_vectors(M, x, y, self.aoa_theta)

        # Calculate aoa
        self.aoa_MUSIC_res = de.DOA_MUSIC(R, scanning_vectors, signal_dimension=1)
    
    
    def stop(self):
        self.running = False

