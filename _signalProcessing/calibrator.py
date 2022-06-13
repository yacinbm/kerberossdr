import logging
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
from _receiver.hydra_receiver import ReceiverRTLSDR


"""
    The Calibrator class is used to correct the non-linearity
    of the VCO's output frequency in relation to its input frequency.

    It is necessary to have such a calibration since the VCO is running
    in an open loop while generating a FMCW chirp.

    The Calibration process works as follows:
    1)  Wait for a peak to appear at the start of the spectrum
    2)  Capture the frequency of the mian peak
    3)  If the peak is within the margin of the edge of the spectrum,
        update the center frequency of the receiver to keep the peak
        in the spectrum's range. 
    4)  Repeat 2-4 until the whole spectrum has been covered

    The Calibrator controls the ReceiverRTLSDR object and should not 
    run at the same time as the tracker, since both objects may try
    to change the receiver's settings.
"""
class Calibrator(QThread):
    signal_calibration_done = pyqtSignal()
    signal_calibration_progress = pyqtSignal(int)
    signal_spectrum_ready = pyqtSignal()
    
    def __init__(self, receiver: ReceiverRTLSDR) -> None:
        super(QThread, self).__init__()
        self.receiver = receiver

        # Calibration variables 
        self.start_f = 119e6
        self.stop_f = 240e6
        self.margin = 0.3e6 # Margin for retuning center freq
        self.locked = False # Used to track the first peak
        self.calibration_done = False
        self.peak_thrld = 35 # Peak detection threshold in dBm
        
        # Spectrum variables
        # TODO: Verifiy sample sample size (FFT bin)
        self.spectrum_sample_size = 2**14
        self.spectrum = np.ones((2, self.spectrum_sample_size), dtype=np.float64) # First index contains frequencies, second values

    def run(self):
        # Lock the receiver
        if not self.receiver.mutex.tryLock(timeout=0):
            logging.warning("Calibrator: Could not lock the receiver, returning...")
            return

        # Initialize the receiver
        center_f = self.start_f + self.receiver.fs/2
        self.receiver.reconfigure_tuner(center_f, self.receiver.fs, self.receiver.receiver_gain)

        # Reset control variables
        self.calibration_done = False
        self.locked = False
        # Reset result vector
        self.calibration_data = np.array([[],[]],dtype=(np.float64,np.float64)) # First index contains time, second VCO frequency
        while not self.calibration_done:
            # Download samples
            self.receiver.download_iq_samples()

            # Only do calibration on the first channel
            self.spectrum[0,:] = np.fft.fftshift(np.fft.fftfreq(self.spectrum_sample_size, 1/self.receiver.fs)) + self.receiver.center_f# Compute frequencies
            self.spectrum[1,:] = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(self.receiver.iq_samples[1, 0:self.spectrum_sample_size])))) # Compute values
            self.signal_spectrum_ready.emit()

            peak_idx = np.argmax(self.spectrum[1,:])
            peak_f = self.spectrum[0,peak_idx]
            peak_pwr = self.spectrum[1,peak_idx]

            # Wait for the starting peak
            if peak_pwr >= self.peak_thrld:
                if not self.locked:
                    # Found the first peak
                    self.locked = True
                    start_time = time.time()
            else: 
                continue
            logging.debug(f"Calibrator: Peak detected at {peak_f}")
            # Save data point
            self.calibration_data = np.append(self.calibration_data, [[time.time()-start_time], [peak_f]], axis=1)

            if peak_f >= self.stop_f:
                # Calibration done
                self.signal_calibration_done.emit()
                self.signal_calibration_progress.emit(100)
                self.calibration_done = True

            # Check recentering
            spectrum_edge = self.spectrum[0,-1]
            if peak_f >= (spectrum_edge - self.margin):
                center_f = int(spectrum_edge - 1.5*self.margin + self.receiver.fs/2)

                logging.debug(f"Calibrator: Reconfiguring center to {center_f}")
                self.receiver.reconfigure_tuner(center_f, self.receiver.fs, self.receiver.receiver_gain)
                # Update progress 
                progress = (center_f - self.start_f)*100//(self.stop_f - self.start_f)
                logging.debug(f"Calibration: Progress {int(progress)}%")
                self.signal_calibration_progress.emit(int(progress))

                    
        
        # Release the receiver
        self.receiver.mutex.unlock()

    def stop(self):
        self.calibration_done = True

