# KerberosSDR Signal Processor
#
# Copyright (C) 2018-2019  Carl Laufer, Tamás Pető
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# -*
# - coding: utf-8 -*-

import logging
import time
# Math support
import numpy as np

# Hydra receiver to interface with RTL
from _receiver.hydra_receiver import ReceiverRTLSDR

# Signal processing support
from scipy import fft,ifft
from scipy import signal
from scipy.signal import correlate

# Plot support
#import matplotlib.pyplot as plt

# GUI support
from PyQt5.QtCore import QThread, pyqtSignal

# Import the pyArgus module
from pyargus import directionEstimation as de


class SignalProcessor(QThread):
    # GUI declarations
    signal_spectrum_ready = pyqtSignal()
    signal_sync_ready = pyqtSignal()
    signal_DOA_ready = pyqtSignal()
    signal_overdrive = pyqtSignal(int)
    signal_period    = pyqtSignal(float)
    signal_PR_ready = pyqtSignal()
    signal_scan_ready = pyqtSignal()

    def __init__(self, parent=None, module_receiver:ReceiverRTLSDR=None):
        super(SignalProcessor, self).__init__(parent)

        self.module_receiver = module_receiver
        self.en_spectrum = True
        self.en_sync = True
        self.en_sample_offset_sync = False
        self.en_record = False
        self.en_calib_iq = False
        self.en_calib_DOA_90 = False
        self.en_DOA_estimation = False
        self.en_PR_processing = False
        self.en_PR_autodet = False
        self.en_noise_var = False
        
        # Scanning option
        self.en_scan = True
        self.start_freq = 24e6
        self.stop_freq = 30e6#250e6
        self.noise_threshold = 0
        self.detected_freq = None

        # DOA processing options
        self.en_DOA_Bartlett = False
        self.en_DOA_Capon = False
        self.en_DOA_MEM = False
        self.en_DOA_MUSIC = False
        self.en_DOA_FB_avg = False
        self.DOA_inter_elem_space = 0.5 # lambda/
        self.DOA_ant_alignment = "ULA" # Antenna spacial arrangement in line: "ULA", in circle: "UCA"
        
        # Passive Radar processing parameters
        self.ref_ch_id = 0
        self.surv_ch_id = 1
        self.en_td_filtering = False
        self.td_filter_dimension = 1        
        self.max_Doppler = 500  # [Hz]
        self.windowing_mode = 0
        self.max_range = 128  # [range cell]
        self.cfar_win_params = [10,10,4,4] # [Est. win length, Est. win width, Guard win length, Guard win width]
        self.cfar_threshold = 13
        self.RD_matrix = np.ones((10,10))
        self.hit_matrix = np.ones((10,10))
        self.RD_matrix_last = np.ones((10,10))
        self.RD_matrix_last_2 = np.ones((10,10))
        self.RD_matrix_last_3 = np.ones((10,10))
        
        self.center_freq = 0  # TODO: Initialize this [Hz]
        self.fs = 1.024 * 10**6  # Decimated sampling frequncy - Update from GUI
        self.channel_number = 4
        
        # Processing parameters        
        self.test = None
        self.spectrum_sample_size = 2**20
        self.DOA_sample_size = 2**15 # Connect to GUI value??
        self.xcorr_sample_size = 2**18 #2**18
        self.spectrum = np.ones((self.channel_number+1,self.spectrum_sample_size), dtype=np.float32) # idx 0 are the frequencies and the rest are the results
        self.xcorr = np.ones((self.channel_number-1,self.xcorr_sample_size*2), dtype=np.complex64)        
        self.phasor_win = 2**10 # Phasor plot window
        self.phasors = np.ones((self.channel_number-1, self.phasor_win), dtype=np.complex64)
        self.run_processing = False
        
        # Result vectors
        self.noise_var= np.empty(self.channel_number, dtype=float)
        self.delay_log= np.array([[0],[0],[0]])
        self.phase_log= np.array([[0],[0],[0]])
        self.DOA_Bartlett_res = np.ones(181)
        self.DOA_Capon_res = np.ones(181)
        self.DOA_MEM_res = np.ones(181)
        self.DOA_MUSIC_res = np.ones(181)
        self.DOA_theta = np.arange(0,181,1)
        
        # Auto resync params
        self.lastTime = 0
        self.runningSync = 0
        self.timed_sync = False
        self.noise_checked = False
        self.resync_time = -1
    
    def run(self):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        #    Redefinition of QThread run function
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.run_processing = True
        
        while self.run_processing:
            start_time = time.time()

            # Download samples
            self.module_receiver.download_iq_samples()
            
            self.spectrum_sample_size = self.module_receiver.iq_samples[0,:].size
            self.DOA_sample_size = self.module_receiver.iq_samples[0,:].size
            self.xcorr_sample_size = self.module_receiver.iq_samples[0,:].size
            self.xcorr = np.ones((self.channel_number-1,self.xcorr_sample_size*2), dtype=np.complex64) 

            # Check overdrive
            if self.module_receiver.overdrive_detect_flag:
                self.signal_overdrive.emit(1)
                
            else:
                self.signal_overdrive.emit(0)
            
            # Display spectrum
            if self.en_spectrum:
                self.spectrum[0, :] = np.fft.fftshift(np.fft.fftfreq(self.spectrum_sample_size, 1/self.fs))

                for m in range(self.channel_number):
                    self.spectrum[m+1,:] = np.fft.fft(self.module_receiver.iq_samples[m, :])
                self.signal_spectrum_ready.emit()
            
            # Synchronization
            if self.en_sync or self.timed_sync:
                sync_time = time.time()
                logging.info("Sync graph enabled")
                self.sample_delay()
                logging.debug(f"Sync time: {time.time()-sync_time}")
                self.signal_sync_ready.emit()
            
            if self.module_receiver.is_noise_source_on and self.en_noise_var:
                self.noise_var = np.var(self.module_receiver.iq_samples,1)
                logging.debug(f"Noise variance: {tuple(self.noise_var)}")

            # Sample offset compensation request
            if self.en_sample_offset_sync:
                self.module_receiver.set_sample_offsets(self.delay_log[:,-1])
                self.en_sample_offset_sync = False
            
            # IQ calibration request
            if self.en_calib_iq:
                # IQ correction
                for m in range(self.channel_number):
                    self.module_receiver.iq_corrections[m] *= np.size(self.module_receiver.iq_samples[0, :])/(np.dot(self.module_receiver.iq_samples[m, :],self.module_receiver.iq_samples[0, :].conj()))
                c = np.sqrt(np.sum(np.abs(self.module_receiver.iq_corrections)**2))
                self.module_receiver.iq_corrections = np.divide(self.module_receiver.iq_corrections, c)
                logging.info("Corrections: ",self.module_receiver.iq_corrections)
                self.en_calib_iq = False
            
            if self.en_calib_DOA_90:
                #TODO: Experimental only for UCA, implement this properly!
                # This calibration is currently done for 0 deg not 90 
                x = self.DOA_inter_elem_space * np.cos(2*np.pi/4 * np.arange(4))
                y = self.DOA_inter_elem_space * np.sin(-2*np.pi/4 * np.arange(4)) # For this specific array only
                ref_vector = de.gen_scanning_vectors(4, x, y, np.zeros(1))[:, 0]                
                N= np.size(self.module_receiver.iq_samples[0, :])
                for m in range(self.channel_number):
                    self.module_receiver.iq_corrections[m] *= ref_vector[m]*N/(np.dot(self.module_receiver.iq_samples[m, :],self.module_receiver.iq_samples[0, :].conj()))                
                logging.info(f"Corrections: {self.module_receiver.iq_corrections}")
                self.en_calib_DOA_90 = False
                
            # Direction of Arrival estimation
            if self.en_DOA_estimation:
                # Get FFT for squelch
                self.spectrum[1,:] = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(self.module_receiver.iq_samples[0, 0:self.spectrum_sample_size]))))

                self.estimate_DOA()
                self.signal_DOA_ready.emit()
            
            # Passive Radar processing
            if self.en_PR_processing:
                self.PR_processing()
                self.signal_PR_ready.emit()
            
            # Record IQ samples
            if self.en_record:
                np.save('hydra_samples.npy', self.module_receiver.iq_samples)   
                

            # Code to maintain sync
            '''if self.timed_sync and not self.en_sync:
                if not self.noise_checked:
                    self.module_receiver.switch_noise_source(0)
                self.timed_sync = False
                self.en_sample_offset_sync=True
                self.runningSync = 0

            resync_on = True
            if(self.resync_time < 10):
                resync_on = False

            if(((start_time - self.lastTime) > self.resync_time) and not self.en_sync and resync_on):
                self.lastTime = start_time
                self.module_receiver.switch_noise_source(1)
                time.sleep(0.1)
                self.runningSync = 1
                self.timed_sync = True'''

            stop_time = time.time()
            self.signal_period.emit(stop_time - start_time)


    def sample_delay(self):
        logging.info("Entered sample delay func")
        N = self.xcorr_sample_size
        iq_samples = self.module_receiver.iq_samples[:, 0:N]
       
        delays = np.array([[0],[0],[0]])
        phases = np.array([[0],[0],[0]])
        # Channel matching
        np_zeros = np.zeros(N, dtype=np.complex64)
        x_padd = np.concatenate([iq_samples[0, :], np_zeros])
        x_fft = np.fft.fft(x_padd)
        for m in np.arange(1, self.channel_number):
            y_padd = np.concatenate([np_zeros, iq_samples[m, :]])
            y_fft = np.fft.fft(y_padd)
            self.xcorr[m-1] = np.fft.ifft(x_fft.conj() * y_fft)
            delay = np.argmax(np.abs(self.xcorr[m-1])) - N
            phase = np.rad2deg(np.angle(self.xcorr[m-1, N]))
            
            logging.info(f"Delay: {str(delay)}")
            delays[m-1,0] = delay
            phases[m-1,0] = phase

        self.delay_log = np.concatenate((self.delay_log, delays),axis=1)
        self.phase_log = np.concatenate((self.phase_log, phases),axis=1)
    
    def delete_sync_history(self):
        self.delay_log= np.array([[0],[0],[0]])
        self.phase_log= np.array([[0],[0],[0]])
    

    def estimate_DOA(self):
        logging.info("Python DSP: Estimating DOA")

        iq_samples = self.module_receiver.iq_samples[:, 0:self.DOA_sample_size]
        # Calculating spatial correlation matrix
        R = de.corr_matrix_estimate(iq_samples.T, imp="fast")

        if self.en_DOA_FB_avg:
            R=de.forward_backward_avg(R)

        M = np.size(iq_samples, 0)

        # Antennas in a cirle
        if self.DOA_ant_alignment == "UCA":
            self.DOA_theta =  np.linspace(0,360,361)
            x = self.DOA_inter_elem_space * np.cos(2*np.pi/M * np.arange(M))
            y = self.DOA_inter_elem_space * np.sin(-2*np.pi/M * np.arange(M)) # For this specific array only
            scanning_vectors = de.gen_scanning_vectors(M, x, y, self.DOA_theta)

        # Antennas on a line
        elif self.DOA_ant_alignment == "ULA":
            self.DOA_theta =  np.linspace(-90,90,181)
            x = np.zeros(M)
            y = np.arange(M) * self.DOA_inter_elem_space            
            scanning_vectors = de.gen_scanning_vectors(M, x, y, self.DOA_theta)

        # DOA estimation
        if self.en_DOA_Bartlett:
            self.DOA_Bartlett_res = de.DOA_Bartlett(R, scanning_vectors)
        if self.en_DOA_Capon:
            self.DOA_Capon_res = de.DOA_Capon(R, scanning_vectors)
        if self.en_DOA_MEM:
            self.DOA_MEM_res = de.DOA_MEM(R, scanning_vectors,  column_select = 0)
        if self.en_DOA_MUSIC:
            self.DOA_MUSIC_res = de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
    
    def stop(self):
        self.run_processing = False