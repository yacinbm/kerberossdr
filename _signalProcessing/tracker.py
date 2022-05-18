import logging
import numpy as np
import time
from scipy.signal import find_peaks
from PyQt5.QtCore import QThread, pyqtSignal

# Import Kerberos receiver
from _receiver.hydra_receiver import ReceiverRTLSDR

# Import the pyArgus module
from pyargus import directionEstimation as de


class Tracker(QThread):
    signal_scan_done = pyqtSignal()
    signal_sync_ready = pyqtSignal()
    signal_period = pyqtSignal(float)
    signal_spectrum_ready = pyqtSignal()

    def __init__(self, receiver:ReceiverRTLSDR) -> None:
        #TODO: 
        # Determine the optimal (or at least the role of) spectrum_sample_size, xcorr_sample_size & DOA_sample_size
          
        # Class initialization
        super(QThread, self).__init__()
        self.receiver = receiver
        # Scan variables
        self.scan_start_f = 24e6
        self.scan_stop_f = 240e6
        self.scan_step = 0  # Initialize before start
        # Control flags
        self.tracker_running = False
        self.en_noise_meas = False
        self.en_scan = False
        self.en_sync = False
        self.en_sample_offset_sync = False
        self.en_calib_iq = False
        self.en_DOA_FB_avg = False
        # Spectrum variables
        self.spectrum_sample_size = 2**14
        self.spectrum = np.ones((self.receiver.channel_number+1,self.spectrum_sample_size), dtype=np.float32) # idx 0 are the frequencies and the rest are the results
        # Cross Correlation variables
        self.xcorr_sample_size = 2**18
        self.xcorr = np.ones((self.receiver.channel_number-1,self.xcorr_sample_size*2), dtype=np.complex64)        
        # Direction of Arrival (DOA) variables
        self.DOA_inter_elem_space = 0.5 # d_antennas/lambda
        self.DOA_sample_size = 2**15 
        # Result Vectors
        self.noise_lvl = np.empty(self.receiver.channel_number, dtype=np.float32)
        self.noise_var= np.empty(self.receiver.channel_number, dtype=np.float32)
        self.delay_log= np.array([[0]]*(self.receiver.channel_number-1))
        self.phase_log= np.array([[0]]*(self.receiver.channel_number-1))
        self.peak_freq = np.array([],dtype=np.float32)
        self.peak_pwr = np.array([],dtype=np.float32)
        self.spectrum_log = np.array([[]]*(self.receiver.channel_number+1), dtype=np.float32) # idx 0 are the frequencies and the rest are the results
        self.DOA_MUSIC_res = np.ones(181)
        self.DOA_theta = np.arange(0,181,1)

    def run(self):
        # Lock the receiver
        if not self.receiver.mutex.tryLock(timeout=0):
            logging.warning("Tracker: Could not lock the receiver, returning...")
            return

        self.tracker_running = True
        while self.tracker_running:
            start_time = time.time()

            self.receiver.download_iq_samples()
            self.xcorr_sample_size = self.receiver.iq_samples[0,:].size
            self.xcorr = np.ones((self.receiver.channel_number-1,self.xcorr_sample_size*2), dtype=np.complex64) 
            
            # Compute FFT
            self.spectrum[0, :] = np.fft.fftshift(np.fft.fftfreq(self.spectrum_sample_size, 1/self.receiver.fs)) + self.receiver.center_f
            for m in range(self.receiver.channel_number):
                self.spectrum[m+1,:] = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(self.receiver.iq_samples[m, 0:self.spectrum_sample_size]))))
            
            self.signal_spectrum_ready.emit()

             # Characterize noise
            if self.receiver.is_noise_source_on and self.en_noise_meas:
                self.noise_var = np.var(self.receiver.iq_samples,1)
                self.noise_lvl = np.amax(self.spectrum[1:,:],1)
                logging.debug(f"Noise variance: {self.noise_var}")
                logging.debug(f"Noise level: {self.noise_lvl}")
                self.en_noise_meas = False
            
            # IQ calibration request
            if self.en_calib_iq:
                # IQ correction
                for m in range(self.receiver.channel_number):
                    self.receiver.iq_corrections[m] *= np.size(self.receiver.iq_samples[0, :])/(np.dot(self.receiver.iq_samples[m, :],self.receiver.iq_samples[0, :].conj()))
                c = np.sqrt(np.sum(np.abs(self.receiver.iq_corrections)**2))
                self.receiver.iq_corrections = np.divide(self.receiver.iq_corrections, c)
                logging.info("Corrections: ",self.receiver.iq_corrections)
                self.en_calib_iq = False

            # Synchronization
            if self.en_sync:
                self.en_sync = False
                sync_time = time.time()
                logging.info("Synching...")
                self.sample_delay()
                logging.debug(f"Sync done in: {time.time()-sync_time}")
                self.signal_sync_ready.emit()

            # Send sync to receiver
            if self.en_sample_offset_sync:
                self.receiver.set_sample_offsets(self.delay_log[:,-1])
                self.en_sample_offset_sync = False
            
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

            self.signal_period.emit(time.time() - start_time)
        # Release the receiver
        self.receiver.mutex.unlock()
    
    def sample_delay(self):
        N = self.xcorr_sample_size
        iq_samples = self.receiver.iq_samples[:, 0:N]
       
        delays = np.array([[0],[0],[0]])
        phases = np.array([[0],[0],[0]])
        # Channel matching
        np_zeros = np.zeros(N, dtype=np.complex64)
        x_padd = np.concatenate([iq_samples[0, :], np_zeros])
        x_fft = np.fft.fft(x_padd)
        for m in np.arange(1, self.receiver.channel_number):
            y_padd = np.concatenate([np_zeros, iq_samples[m, :]])
            y_fft = np.fft.fft(y_padd)
            self.xcorr[m-1] = np.fft.ifft(x_fft.conj() * y_fft)
            delay = np.argmax(np.abs(self.xcorr[m-1])) - N
            phase = np.rad2deg(np.angle(self.xcorr[m-1, N]))
            
            logging.debug(f"Sample delay: {str(delay)}")
            delays[m-1,0] = delay
            phases[m-1,0] = phase

        self.delay_log = np.concatenate((self.delay_log, delays),axis=1)
        self.phase_log = np.concatenate((self.phase_log, phases),axis=1)

    def estimate_DOA(self):
        logging.info("Tracker: Estimating DOA")

        iq_samples = self.receiver.iq_samples[:, 0:self.DOA_sample_size]
        # Calculating spatial correlation matrix
        R = de.corr_matrix_estimate(iq_samples.T, imp="fast")

        if self.en_DOA_FB_avg:
            R=de.forward_backward_avg(R)

        M = np.size(iq_samples, 0)

        # Generate antenna coordonates
        self.DOA_theta =  np.linspace(-90,90,181)
        x = np.zeros(M)
        y = np.arange(M) * self.DOA_inter_elem_space            
        scanning_vectors = de.gen_scanning_vectors(M, x, y, self.DOA_theta)

        # Calculate DOA
        self.DOA_MUSIC_res = de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
    
    
    def stop(self):
        self.tracker_running = False

