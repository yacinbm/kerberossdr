from cmath import exp
from re import L
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout
import pyqtgraph as pg
import numpy as np
import logging
from sys import argv
import datetime

from _GUI.sdr_people_tracker_main_window import Ui_MainWindow
from _receiver.hydra_receiver import ReceiverRTLSDR
from _signalProcessing.tracker import Tracker
from _signalProcessing.calibrator import Calibrator

# DEBUG LOGGING
logging.basicConfig(filename="main.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        # Parse arguments
        self.buffer_size = int(argv[1]) * 1024
        # GUI INIT ---->
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)
        
        # Calibration display
        self.cal_display = QHBoxLayout(self.graphCal)
        self.plotWidget_cal_data = pg.plot()
        self.plotWidget_cal_spectrum = pg.plot()
        
        self.cal_display.addWidget(self.plotWidget_cal_data)
        self.cal_display.addWidget(self.plotWidget_cal_spectrum)

        x = np.arange(1000)
        y = np.random.normal(size=1000)

        self.cal_data_curve = self.plotWidget_cal_data.plot(x, y, clear=True, pen=(255,199,15))
        self.cal_spectrum_curve = self.plotWidget_cal_spectrum.plot(x, y, pen="r")
        self.horizontal_line = pg.InfiniteLine(pos=0, angle=0)
        self.plotWidget_cal_spectrum.addItem(self.horizontal_line)

        self.plotWidget_cal_data.setTitle("Calibration Data")
        self.plotWidget_cal_data.setLabel("bottom", "Time [s]")
        self.plotWidget_cal_data.setLabel("left", "Frequency [MHz]")
        self.plotWidget_cal_spectrum.setTitle("Calibration Spectum")
        self.plotWidget_cal_spectrum.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_cal_spectrum.setLabel("left", "Amplitude [dBm]")

        # Spectrum display (disabled for now)
        """
        self.plotWidget_spectrum_ch1 = pg.plot()
        self.plotWidget_spectrum_ch2 = pg.plot()
        self.plotWidget_spectrum_ch3 = pg.plot()
        self.plotWidget_spectrum_ch4 = pg.plot()

        
        self.win_spectrum.addWidget(self.plotWidget_spectrum_ch1, 0, 0)
        self.win_spectrum.addWidget(self.plotWidget_spectrum_ch2, 0, 1)
        self.win_spectrum.addWidget(self.plotWidget_spectrum_ch3, 1, 0)
        self.win_spectrum.addWidget(self.plotWidget_spectrum_ch4, 1, 1)
        
        
        x = np.arange(1000)
        y = np.random.normal(size=(4,1000))

        self.spectrum_ch1_curve = self.plotWidget_spectrum_ch1.plot(x, y[0], clear=True, pen=(255, 199, 15))
        self.spectrum_ch2_curve = self.plotWidget_spectrum_ch2.plot(x, y[1], clear=True, pen='r')
        self.spectrum_ch3_curve = self.plotWidget_spectrum_ch3.plot(x, y[2], clear=True, pen='g')
        self.spectrum_ch4_curve = self.plotWidget_spectrum_ch4.plot(x, y[3], clear=True, pen=(9, 237, 237))

        self.plotWidget_spectrum_ch1.setTitle("Channel 1")
        self.plotWidget_spectrum_ch1.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch1.setLabel("left", "Amplitude [dBm]")
        self.plotWidget_spectrum_ch2.setTitle("Channel 2")
        self.plotWidget_spectrum_ch2.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch2.setLabel("left", "Amplitude [dBm]")
        self.plotWidget_spectrum_ch3.setTitle("Channel 3")
        self.plotWidget_spectrum_ch3.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch3.setLabel("left", "Amplitude [dBm]")
        self.plotWidget_spectrum_ch4.setTitle("Channel 4")
        self.plotWidget_spectrum_ch4.setLabel("bottom", "Frequency [MHz]")
        self.plotWidget_spectrum_ch4.setLabel("left", "Amplitude [dBm]")
        """

        # <---- GUI INIT 

        # Worker thread object declaration
        self.receiver = ReceiverRTLSDR()
        self.receiver.block_size = self.buffer_size
        self.receiver.fs = 2.5e6
        self.calibrator = Calibrator(receiver=self.receiver)
        self.tracker = Tracker(receiver=self.receiver)
        
        # Signal connection
        # Calibrator
        self.btnStartCal.clicked.connect(self.cal_btn_clicked)
        self.lineStartFreq.textChanged.connect(self.cal_edit_start_freq)
        self.lineStopFreq.textChanged.connect(self.cal_edit_stop_freq)
        self.lineDetectThrhld.textChanged.connect(self.cal_edit_detect_thrshld)
        self.calibrator.signal_calibration_done.connect(self.cal_done)
        self.calibrator.signal_calibration_progress.connect(self.cal_prog_updt)
        self.calibrator.signal_spectrum_ready.connect(self.cal_plot)

        # Tracker
        self.tracker.signal_period.connect(self.period_time_update)
        self.tracker.signal_sync_ready.connect(self.sync_finished)
        self.tracker.signal_spectrum_ready.connect(self.spectrum_plot)

        # Tracker initialization
        self.tracker.scan_step = 2559840
        self.tracker.center_f = self.tracker.scan_start_f
        
        # Control flag
        self.tracker.en_scan = True
        self.tracker.en_noise_meas = True
        self.tracker.en_sync = True
        self.tracker.en_sample_offset_sync = True
        self.en_calib_iq = True
        self.calibrator_running = False
        

    # Callback function declaration
    def cal_edit_start_freq(self, value):
        try:
            self.calibrator.start_f = float(value) * 1e6
        except Exception as e:
            logging.error("Calibrator: Invalid start frequency")
        
        

    def cal_edit_stop_freq(self, value):
        try:
            self.calibrator.stop_f = float(value) * 1e6
        except Exception as e:
            logging.error("Calibrator: Invalid stop frequency")
        
    def cal_edit_detect_thrshld(self, value):
        try: 
            self.calibrator.peak_thrld = float(value)
        except Exception as e:
            logging.error("Calibrator: Invalid detection threshold")
        self.horizontal_line.setPos(self.calibrator.peak_thrld)

    def enable_fields(self):
        self.lineDetectThrhld.setDisabled(False)
        self.lineStartFreq.setDisabled(False)
        self.lineStopFreq.setDisabled(False)

    def disable_fields(self):
        self.lineDetectThrhld.setDisabled(True)
        self.lineStartFreq.setDisabled(True)
        self.lineStopFreq.setDisabled(True)

    def cal_btn_clicked(self):
        if not self.calibrator_running:
            self.calibrator_running = True
            self.progBarCal.setValue(0)
            self.btnStartCal.setText("Cancel")
            self.disable_fields()
            self.calibrator.start()
            logging.info("Calibrator started")
        else:
            self.calibrator_running = False
            self.progBarCal.setValue(0)
            self.btnStartCal.setText("Start Calibration")
            self.enable_fields()
            self.calibrator.stop()
            logging.info("Calibrator aborted")

    def cal_done(self):
        self.enable_fields()
        self.btnStartCal.setText("Calibration Done!")
        dateTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        fileName = f"vco_cal_data_{dateTime}.csv"
        logging.info(f"Calibration done, saving calibration data to {fileName}")
        np.savetxt(fileName,self.calibrator.calibration_data,delimiter=",")
    
    def cal_prog_updt(self, value):
        self.progBarCal.setValue(value)

    def cal_plot(self):
        logging.debug("Plotting cal data")
        # Calibration
        self.cal_spectrum_curve.setData(self.calibrator.spectrum[0,:], self.calibrator.spectrum[1,:])
        # Spectrum
        self.cal_data_curve.setData(self.calibrator.calibration_data[0,:], self.calibrator.calibration_data[1,:])

    def spectrum_plot(self):
        # Gather data
        frequencies = self.tracker.spectrum[0,:]
        y1 = self.tracker.spectrum[1,:]
        y2 = self.tracker.spectrum[2,:]
        y3 = self.tracker.spectrum[3,:]
        y4 = self.tracker.spectrum[4,:]

        # Update figure
        self.spectrum_ch1_curve.setData(frequencies, y1)
        self.spectrum_ch2_curve.setData(frequencies, y2)
        self.spectrum_ch3_curve.setData(frequencies, y3)
        self.spectrum_ch4_curve.setData(frequencies, y4)

        logging.debug(f"span: {frequencies[0],frequencies[-1]}")

    def period_time_update(self, update_period):
        logging.info(f"Signal Period: {update_period}")
    
    def save_calibration(self):
        # Save spectrum 
        np.save('full_spectrum.npy', self.tracker.spectrum_log)
        logging.info(f"Peaks (Hz,dBm):{(self.tracker.peak_freq,self.tracker.peak_pwr)}")

    def sync_finished(self):
        self.receiver.switch_noise_source(0)

logging.basicConfig(filename="main.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG)

app = QApplication(argv)
win = MainWindow()
win.show()
app.exec()