# KerberosSDR Receiver

# Copyright (C) 2018-2019  Carl Laufer, Tamás Pető
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

# -*- coding: utf-8 -*-

import logging
import numpy as np
import sys
import select
import time
from struct import pack
from scipy import signal
from PyQt5.QtCore import QMutex

class ReceiverRTLSDR():
    """
        RTL SDR based receiver controller module



        Description:
         ------------
           Implements the functions to handle multiple RTL-SDR receivers 

        Main functions:
         ------------------

        Notes:
         ------------
         
 
        Features:
         ------------

        Project: Hydra

        Authors: Tamás Pető

        License: No license

        Changelog :
            - Ver 1.0000 : Initial version (2018 04 23)       
            - Ver 2.0000 : (2018 05 22) 

    """

    # GUI Signal definitions
    def __init__(self):
            
            logging.info("Python rec: Starting Python RTL-SDR receiver")
            # Thread control
            self.mutex = QMutex()

            # Receiver control parameters            
            self.gc_fifo_name = "_receiver/C/gate_control_fifo"
            self.sync_fifo_name = "_receiver/C/sync_control_fifo"
            self.rec_control_fifo_name = "_receiver/C/rec_control_fifo"
            
            self.gate_trigger_byte = pack('B',1)          
            self.gate_close_byte = pack('B',2)
            self.gate_flush_byte = pack('B',3)
            self.sync_close_byte = pack('B',2)
            self.rec_control_close_byte = pack('B',2) 
            self.sync_delay_byte = 'd'.encode('ascii')
            self.reconfig_tuner_byte = 'r'.encode('ascii')
            self.noise_source_on_byte = 'n'.encode('ascii')
            self.noise_source_off_byte = 'f'.encode('ascii')
            self.gc_fifo_descriptor = open(self.gc_fifo_name, 'w+b', buffering=0)
            self.sync_fifo_descriptor = open(self.sync_fifo_name, 'w+b', buffering=0)
            self.rec_control_fifo_descriptor = open(self.rec_control_fifo_name, 'w+b', buffering=0)
            
            self.is_noise_source_on = False
            
            # Data acquisition parameters
            self.receiver_gain = [0,0,0,0]
            self.center_f = 24e6
            self.channel_number = 4
            self.block_size = 0
            self.overdrive_detect_flag = False

            # IQ preprocessing parameters
            self.en_dc_compensation = False
            self.fs = 1.024 * 10**6  # Sampling frequency
            self.iq_corrections = np.array([1,1,1,1], dtype=np.complex64)  # Used for phase and amplitude correction
            self.fir_size = 0
            self.fir_bw = 1  # Normalized to sampling frequency 
            self.fir_filter_coeffs = np.empty(0)
            self.decimation_ratio = 1               
            
            
    def set_sample_offsets(self, sample_offsets):
        logging.info("Python rec: Setting sample offset")
        delays = [0] + (sample_offsets.tolist())
        self.sync_fifo_descriptor.write(self.sync_delay_byte)
        self.sync_fifo_descriptor.write(pack("i"*4,*delays))
    
    def reconfigure_tuner(self, center_freq, sample_rate, gain):
       logging.info(f"Python rec: Setting receiver center frequency to:{center_freq}")
       logging.info(f"Python rec: Setting receiver sample rate to:{sample_rate}")
       logging.info(f"Python rec: Setting receiver gain to:{gain}")
       self.center_f = center_freq
       self.receiver_gain = gain
       self.fs = sample_rate
       # Send new config to the control FIFO
       self.rec_control_fifo_descriptor.write(self.reconfig_tuner_byte)    
       self.rec_control_fifo_descriptor.write(pack("I", int(center_freq)))
       self.rec_control_fifo_descriptor.write(pack("I", int(sample_rate)))
       self.rec_control_fifo_descriptor.write(pack("i", int(gain[0])))
       self.rec_control_fifo_descriptor.write(pack("i", int(gain[1])))
       self.rec_control_fifo_descriptor.write(pack("i", int(gain[2])))
       self.rec_control_fifo_descriptor.write(pack("i", int(gain[3])))
       time.sleep(0.05) # Wait for Rx to stabilize
       # Flush the input stream to remove any pending captures
       self.gc_fifo_descriptor.write(self.gate_flush_byte)
    
    def switch_noise_source(self, state):
        if state:
            logging.info("Python rec: Turning on noise source")
            self.rec_control_fifo_descriptor.write(self.noise_source_on_byte)
        else:
            logging.info("Python rec: Turning off noise source")
            self.rec_control_fifo_descriptor.write(self.noise_source_off_byte)

        self.is_noise_source_on = state
            
    def set_fir_coeffs(self, fir_size, bw):
        """
            Set FIR filter coefficients
            
            TODO: Implement FIR in C and send coeffs there
        """
        
        # Data preprocessing parameters
        if fir_size >0 :
            cut_off = bw/(self.fs / self.decimation_ratio)
            self.fir_filter_coeffs = signal.firwin(fir_size, cut_off, window="hann")
        self.fir_size = fir_size
        
    def download_iq_samples(self):
            self.iq_samples = np.zeros((self.channel_number, self.block_size//2), dtype=np.complex64)
            self.gc_fifo_descriptor.write(self.gate_trigger_byte)
            logging.debug("Python rec: Trigger written")
            read_size = self.block_size * self.channel_number
            byte_array_read = sys.stdin.buffer.read(read_size)
            overdrive_margin = 0.95
            self.overdrive_detect_flag = False

            try:
                byte_data_np = np.frombuffer(byte_array_read, dtype='uint8', count=read_size)
            except Exception as e:
                return

            self.iq_samples.real = byte_data_np[0:self.channel_number*self.block_size:2].reshape(self.channel_number, self.block_size//2)
            self.iq_samples.imag = byte_data_np[1:self.channel_number*self.block_size:2].reshape(self.channel_number ,self.block_size//2)
            self.iq_samples /= (255 / 2)
            self.iq_samples -= (1 + 1j) 
            self.iq_preprocessing()

            logging.debug("IQ sample read ready")
       
    def iq_preprocessing(self):
        # Decimation (downsampling)
        if self.decimation_ratio > 1:
           iq_samples_dec = np.zeros((self.channel_number, round(self.block_size//2/self.decimation_ratio)), dtype=np.complex64)
           for m in range(self.channel_number):
               iq_samples_dec[m, :] = self.iq_samples[m, 0::self.decimation_ratio]
           self.iq_samples = iq_samples_dec

        # FIR filtering
        if self.fir_size > 0:
            for m in range(self.channel_number):
                self.iq_samples[m, :] = np.convolve(self.fir_filter_coeffs, self.iq_samples[m, :], mode="same")

        # Remove DC content (Force on for now)
        if self.en_dc_compensation or True:
            for m in np.arange(0, self.channel_number):
               self.iq_samples[m,:]-= np.average( self.iq_samples[m,:])
           
        # IQ correction
        for m in np.arange(0, self.channel_number):
            self.iq_samples[m, :] *= self.iq_corrections[m]
    
        
    def close(self):
        self.gc_fifo_descriptor.write(self.gate_close_byte)
        self.sync_fifo_descriptor.write(self.sync_close_byte)
        self.rec_control_fifo_descriptor.write(self.rec_control_close_byte)
        time.sleep(1)
        self.gc_fifo_descriptor.close()
        self.sync_fifo_descriptor.close()
        logging.info("Python rec: FIFOs are closed")

        
        


