from _receiver.hydra_receiver import ReceiverRTLSDR
from _signalProcessing.hydra_signal_processor import SignalProcessor
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QTabWidget
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from pyqtgraph import GraphicsWindow
from scipy.signal import find_peaks
import numpy as np
import time
import logging
from sys import argv


rec = ReceiverRTLSDR()

class Scanner(QThread):
    signal_scan_done = pyqtSignal()
    signal_sync_ready = pyqtSignal()
    signal_period = pyqtSignal(float)
    signal_spectrum_ready = pyqtSignal()

    def __init__(self, receiver:ReceiverRTLSDR) -> None:
        # Class initialization
        super().__init__()
        self.receiver = receiver
        # Scan variables
        self.scan_start_f = 24e6
        self.scan_stop_f = 240e6
        self.scan_step = 0  # Initialize before start
        self.center_f = 0   # Initialize before start
        # Control flags
        self.en_noise_meas = False
        self.en_scan = False
        self.en_sync = False
        self.en_sample_offset_sync = False
        self.en_calib_iq = False
        # Spectrum variables
        self.spectrum_sample_size = 2**14
        self.spectrum = np.ones((self.receiver.channel_number+1,self.spectrum_sample_size), dtype=np.float32) # idx 0 are the frequencies and the rest are the results
        # Cross Correlation variables
        self.xcorr_sample_size = 2**18
        self.xcorr = np.ones((self.receiver.channel_number-1,self.xcorr_sample_size*2), dtype=np.complex64)        
        # Result Vectors
        self.noise_lvl = np.empty(self.receiver.channel_number, dtype=np.float32)
        self.noise_var= np.empty(self.receiver.channel_number, dtype=np.float32)
        self.delay_log= np.array([[0]]*(self.receiver.channel_number-1))
        self.phase_log= np.array([[0]]*(self.receiver.channel_number-1))
        self.peak_freq = np.array([],dtype=np.float32)
        self.peak_pwr = np.array([],dtype=np.float32)
        self.spectrum_log = np.array([[]]*(self.receiver.channel_number+1), dtype=np.float32) # idx 0 are the frequencies and the rest are the results

    def run(self):
        while True:
            start_time = time.time()

            self.receiver.download_iq_samples()
            self.xcorr_sample_size = self.receiver.iq_samples[0,:].size
            self.xcorr = np.ones((self.receiver.channel_number-1,self.xcorr_sample_size*2), dtype=np.complex64) 
            
            # Compute FFT
            self.spectrum[0, :] = np.fft.fftshift(np.fft.fftfreq(self.spectrum_sample_size, 1/self.receiver.fs)) + self.center_f
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
    
    def sample_delay(self):
        logging.info("Entered sample delay func")
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
            
            logging.info(f"Delay: {str(delay)}")
            delays[m-1,0] = delay
            phases[m-1,0] = phase

        self.delay_log = np.concatenate((self.delay_log, delays),axis=1)
        self.phase_log = np.concatenate((self.phase_log, phases),axis=1)


class MainWindow(QMainWindow):
    def spectrum_plot(self):
        # Gather data
        frequencies = self.scanner.spectrum[0,:]
        y1 = self.scanner.spectrum[1,:]
        y2 = self.scanner.spectrum[2,:]
        y3 = self.scanner.spectrum[3,:]
        y4 = self.scanner.spectrum[4,:]

        # Update figure
        self.spectrum_ch1_curve.setData(frequencies, y1)
        self.spectrum_ch2_curve.setData(frequencies, y2)
        self.spectrum_ch3_curve.setData(frequencies, y3)
        self.spectrum_ch4_curve.setData(frequencies, y4)

        logging.debug(f"span: {frequencies[0],frequencies[-1]}")

    def period_time_update(self, update_period):
        logging.info(f"Signal Period: {update_period}")
    
    def scan_finished(self):
        # Save spectrum 
        np.save('full_spectrum.npy', self.scanner.spectrum_log)
        logging.info(f"Peaks (Hz,dBm):{(self.scanner.peak_freq,self.scanner.peak_pwr)}")

    def sync_finished(self):
        self.receiver.switch_noise_source(0)

    def __init__(self, *args, **kwargs):
        # GUI INIT ---->
        super(MainWindow, self).__init__(*args, **kwargs)
        # QT5 Settings
        self.setWindowTitle("MyApp")
        buffer_size = int(argv[1])

        # Tabs
        self.tabWidget = QTabWidget(self)

        # Spectrum display
        self.tab_spectrum = QMainWindow()
        self.gridLayout_spectrum = QGridLayout(self.tab_spectrum)
        self.tabWidget.addTab(self.tab_spectrum, "")

        self.win_spectrum = GraphicsWindow(title="Quad Channel Spectrum")
        self.plotWidget_spectrum_ch1 = self.win_spectrum.addPlot(title="Channel 1")
        self.plotWidget_spectrum_ch2 = self.win_spectrum.addPlot(title="Channel 2")
        self.win_spectrum.nextRow()
        self.plotWidget_spectrum_ch3 = self.win_spectrum.addPlot(title="Channel 3")
        self.plotWidget_spectrum_ch4 = self.win_spectrum.addPlot(title="Channel 4")

        self.gridLayout_spectrum.addWidget(self.win_spectrum,1,1,1,1)
        
        x = np.arange(1000)
        y = np.random.normal(size=(4,1000))

        self.spectrum_ch1_curve = self.plotWidget_spectrum_ch1.plot(x, y[0], clear=True, pen=(255, 199, 15))
        self.spectrum_ch2_curve = self.plotWidget_spectrum_ch2.plot(x, y[1], clear=True, pen='r')
        self.spectrum_ch3_curve = self.plotWidget_spectrum_ch3.plot(x, y[2], clear=True, pen='g')
        self.spectrum_ch4_curve = self.plotWidget_spectrum_ch4.plot(x, y[3], clear=True, pen=(9, 237, 237))


        self.plotWidget_spectrum_ch1.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch1.setLabel("left", "Amplitude [dBm]")
        self.plotWidget_spectrum_ch2.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch2.setLabel("left", "Amplitude [dBm]")
        self.plotWidget_spectrum_ch3.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch3.setLabel("left", "Amplitude [dBm]")
        self.plotWidget_spectrum_ch4.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch4.setLabel("left", "Amplitude [dBm]")

        # <---- GUI INIT 

        # Worker thread object declaration
        self.receiver = ReceiverRTLSDR()
        self.scanner = Scanner(receiver=self.receiver)

        # Signal connection
        self.scanner.signal_period.connect(self.period_time_update)
        self.scanner.signal_scan_done.connect(self.scan_finished)
        self.scanner.signal_sync_ready.connect(self.sync_finished)
        self.scanner.signal_spectrum_ready.connect(self.spectrum_plot)

        # Scanner initialization
        self.scanner.scan_start_f = 100e6
        self.scanner.scan_stop_f = 200e6
        self.scanner.receiver.fs = 2.56e6
        self.scanner.scan_step = 2559840
        self.scanner.center_f = self.scanner.scan_start_f
        sample_rate = self.scanner.receiver.fs * self.scanner.receiver.decimation_ratio
        self.scanner.receiver.block_size = buffer_size*1024
        self.scanner.receiver.switch_noise_source(0)
        self.scanner.receiver.reconfigure_tuner(self.scanner.center_f, sample_rate, self.scanner.receiver.receiver_gain)
        # Control flag
        self.scanner.en_scan = True
        self.scanner.en_noise_meas = True
        self.scanner.en_sync = True
        self.scanner.en_sample_offset_sync = True
        self.en_calib_iq = True
        
        logging.info("Starting scanner")
        self.scanner.start()

logging.basicConfig(filename="main.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG)

app = QApplication(argv)
win = MainWindow()
win.show()
app.exec()