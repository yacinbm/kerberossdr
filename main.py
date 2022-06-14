from _receiver.hydra_receiver import ReceiverRTLSDR as kerberosSDR
from _signalProcessing.hydra_signal_processor import SignalProcessor
import numpy as np
from time import sleep
from sys import argv
import logging

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QTabWidget
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from pyqtgraph import GraphicsWindow

class Worker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        """Long-running task."""
        for i in range(5):
            sleep(1)
            self.progress.emit(i + 1)
        self.finished.emit()

class MainWindow(QMainWindow):
    def scan_print(self):
        logging.info(f"Number of peaks: {self.signal_processor.detected_freq.size}")
        pass
    
    def power_level_update(self):
        pass

    def period_time_update(self, update_period):
        # TODO: Display it also
        logging.info(f"Signal Period: {update_period}")
        pass

    def spectrum_plot(self):
        # Gather data
        frequencies = self.signal_processor.spectrum[0,:]
        y1 = self.signal_processor.spectrum[1,:]
        y2 = self.signal_processor.spectrum[2,:]
        y3 = self.signal_processor.spectrum[3,:]
        y4 = self.signal_processor.spectrum[4,:]

        # Update figure
        self.spectrum_ch1_curve.setData(frequencies, y1)
        self.spectrum_ch2_curve.setData(frequencies, y2)
        self.spectrum_ch3_curve.setData(frequencies, y3)
        self.spectrum_ch4_curve.setData(frequencies, y4)

    def sync_callback(self):
        self.signal_processor.en_sync = False
        self.signal_processor.module_receiver.switch_noise_source(0) # Turn off noise source
        logging.info("sync Done")

    def DOA_plot(self):
        pass

    def RD_plot(self):
        pass

    def close(self):
        # Finish
        self.signal_processor.stop()
        self.signal_processor.module_receiver.close()

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # QT5 Settings
        self.setWindowTitle("MyApp")
        buffer_size = int(argv[1])

        # Tabs
        self.tabWidget = QTabWidget(self)

        # Worker thread object declaration
        self.receiver = kerberosSDR(debug=False)
        self.signal_processor = SignalProcessor(module_receiver=self.receiver)

        # Signal callback connection
        self.signal_processor.signal_overdrive.connect(self.power_level_update)
        self.signal_processor.signal_period.connect(self.period_time_update)
        self.signal_processor.signal_spectrum_ready.connect(self.spectrum_plot)
        self.signal_processor.signal_sync_ready.connect(self.sync_callback)
        self.signal_processor.signal_DOA_ready.connect(self.DOA_plot)
        self.signal_processor.signal_PR_ready.connect(self.RD_plot)
        self.signal_processor.signal_scan_ready.connect(self.scan_print)
        
        
        # Receiver settings
        self.signal_processor.center_freq = 155e6 # Test frequency 1.5GHz
        self.signal_processor.module_receiver.fs = 2.56e6
        self.sample_rate = self.signal_processor.module_receiver.fs* self.receiver.decimation_ratio

        # Receiver Setup
        self.signal_processor.module_receiver.block_size = buffer_size*1024
        self.signal_processor.module_receiver.reconfigure_tuner(self.signal_processor.center_freq, self.sample_rate, self.signal_processor.module_receiver.receiver_gain)

        # Plot variables
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

        # Initialize signal processor
        self.signal_processor.module_receiver.switch_noise_source(1) # Turn on noise to sync
        logging.info("syncing")
        self.signal_processor.en_sample_offset_sync = True # Ask to send offset to RTLSDR
        self.signal_processor.en_calib_iq = True # Ask for IQ Correction
        self.signal_processor.en_sync = True # Ask to compute sample offset
        self.signal_processor.en_spectrum = True # Analyze the frequency
        self.signal_processor.en_scan = False
        self.signal_processor.en_noise_var = True
        # Start processing thread
        logging.info("starting thread")
        self.signal_processor.start()
        
        

logging.basicConfig(filename="main.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG)

app = QApplication(argv)
win = MainWindow()
win.show()
app.exec()