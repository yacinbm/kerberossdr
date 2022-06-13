from cmath import exp
from re import L
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QGraphicsScene, QGraphicsView
import pyqtgraph as pg
from scipy.constants import speed_of_light
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
        y = np.zeros(1000)

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
        self.tracker_meas_display = QHBoxLayout(self.graphMeasure)
        self.tracker_pos_scene = QGraphicsScene()
        self.graphPosition.setScene(self.tracker_pos_scene)
        # Measurements plot
        self.plotWidget_tracker_aoa = pg.plot()
        self.plotWidget_tracker_dist = pg.plot()
        self.tracker_meas_display.addWidget(self.plotWidget_tracker_aoa)
        self.tracker_meas_display.addWidget(self.plotWidget_tracker_dist)


        self.tracker_aoa_curve = self.plotWidget_tracker_aoa.plot(x,y, clear=True, pen=(255,199,15))
        #self.tracker_dist_curve = self.plotWidget_tracker_dist.plot(x,y,clear=True, pen="r")
        self.tracker_dist_hist = pg.ImageItem()
        cm = pg.colormap.get("CET-L17")
        cm.reverse()
        self.tracker_dist_hist.setColorMap(cm)
        self.tracker_dist_hist.setLevels([-50,50])
        self.plotWidget_tracker_dist.addItem(self.tracker_dist_hist)
        #self.tracker_dist_curve_2 = self.plotWidget_tracker_dist.plot(x,y, pen="g")
        #self.tracker_dist_curve_3 = self.plotWidget_tracker_dist.plot(x,y, pen="b")
        #self.tracker_dist_curve_4 = self.plotWidget_tracker_dist.plot(x,y, pen="y")
        
        
        self.plotWidget_tracker_aoa.setTitle("Detected Angle of Arrival")
        self.plotWidget_tracker_aoa.setLabel("left", "Power [dBm]")
        self.plotWidget_tracker_aoa.setLabel("right", "Angle of arrival [deg]")
        self.plotWidget_tracker_dist.setTitle("Detected Distance")
        self.plotWidget_tracker_dist.setLabel("left", "Distance")

        # Tracker plot
        self.plotWidget_tracker_position = pg.plot()
        self.tracker_pos_scene.addWidget(self.plotWidget_tracker_position)

        # <---- GUI INIT 

        # Worker thread object declaration
        self.receiver = ReceiverRTLSDR()
        self.receiver.block_size = self.buffer_size
        self.receiver.fs = 1.024e6
        self.receiver.center_f = 1700.04e6
        self.calibrator = Calibrator(receiver=self.receiver)
        self.tracker = Tracker(receiver=self.receiver)
        
        # Signal connection
        # Tabs manager
        self.tabWidget.currentChanged.connect(self.tab_changed)

        # === Calibrator ===
        self.btnStartCal.clicked.connect(self.cal_btn_clicked)
        self.lineStartFreq.textChanged.connect(self.cal_edit_start_freq)
        self.lineStopFreq.textChanged.connect(self.cal_edit_stop_freq)
        self.lineDetectThrhld.textChanged.connect(self.cal_edit_detect_thrshld)
        self.calibrator.signal_calibration_done.connect(self.cal_done)
        self.calibrator.signal_calibration_progress.connect(self.cal_prog_updt)
        self.calibrator.signal_spectrum_ready.connect(self.cal_plot)

        # === Tracker ===
        self.spinInterAntDist.valueChanged.connect(self.update_antenna_dist)
        self.spinCenterFreq.valueChanged.connect(self.update_center_freq)
        self.spinFilterBw.valueChanged.connect(self.update_filter_bw)
        self.spinFirTapSize.valueChanged.connect(self.update_fir_tap_size)
        self.btnStartTrack.clicked.connect(self.det_btn_clicked)
        self.tracker.signal_spectrum_ready.connect(self.det_dist_plot)
        self.tracker.signal_period.connect(self.det_period_time_update)
        self.tracker.signal_sync_ready.connect(self.det_sync_done)
        self.tracker.signal_aoa_ready.connect(self.det_aoa_plot)

        # Tracker initialization
        self.tracker.scan_step = 2559840
        self.tracker.center_f = self.tracker.scan_start_f
        
        # Control flag
        self.tracker.en_scan = False
        self.tracker.en_noise_meas = True
        self.tracker.en_sync = True
        self.tracker.en_sample_offset_sync = True
        self.tracker.en_calib_iq = True
        

    # Callback function declaration
    # Tab manager 
    def tab_changed(self, idx):
        # Stop all processes
        self.calibrator.stop()
        self.tracker.stop()
        
    # === CALIBRATION TAB ===
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

    def cal_enable_fields(self):
        self.lineDetectThrhld.setDisabled(False)
        self.lineStartFreq.setDisabled(False)
        self.lineStopFreq.setDisabled(False)

    def cal_disable_fields(self):
        self.lineDetectThrhld.setDisabled(True)
        self.lineStartFreq.setDisabled(True)
        self.lineStopFreq.setDisabled(True)

    def cal_btn_clicked(self):
        if not self.calibrator.isRunning():
            self.progBarCal.setValue(0)
            self.btnStartCal.setText("Cancel")
            self.cal_disable_fields()
            self.calibrator.start()
            logging.info("Calibrator started")
        else:
            self.progBarCal.setValue(0)
            self.btnStartCal.setText("Start Calibration")
            self.cal_enable_fields()
            self.calibrator.stop()
            logging.info("Calibrator aborted")

    def cal_done(self):
        self.cal_enable_fields()
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

    # === DETECTION TAB ===
    def update_antenna_dist(self, value):
        self.tracker.antenna_distance = value/100

    def update_center_freq(self, value):
        self.tracker.en_estimate_aoa = False
        self.receiver.center_f = value*1e6
        self.receiver.reconfigure_tuner(self.receiver.center_f, self.receiver.fs, self.receiver.receiver_gain)
        self.receiver.switch_noise_source(1)
        self.tracker.en_noise_meas = True
        self.tracker.en_sync = True
        self.tracker.en_sample_offset_sync = True
        self.tracker.en_calib_iq = True
        # TODO: Clear histogram

    def update_filter_bw(self, value):
        self.receiver.fir_bw = value
        self.receiver.set_fir_coeffs(self.receiver.fir_size, self.receiver.fir_bw)

    def update_fir_tap_size(self, value):
        self.receiver.fir_size = value
        self.receiver.set_fir_coeffs(self.receiver.fir_size, self.receiver.fir_bw)

    def det_btn_clicked(self):
        if not self.tracker.isRunning():
            self.tracker.receiver.reconfigure_tuner(self.tracker.receiver.center_f, self.tracker.receiver.fs, self.tracker.receiver.receiver_gain)
            self.tracker.receiver.decimation_ratio = 1
            self.tracker.receiver.set_fir_coeffs(self.spinFirTapSize.value(), self.spinFilterBw.value()*1e3)
            self.tracker.receiver.switch_noise_source(1)
            self.btnStartTrack.setText("Stop Tracking")
            self.tracker.start()
            logging.info("Tracker started")
        else:
            self.btnStartTrack.setText("Start Tracking")
            self.tracker.stop()
            logging.info("Tracker aborted")

    def det_aoa_plot(self):
        thetas = self.tracker.aoa_theta
        result = self.tracker.aoa_MUSIC_res
        result = np.divide(np.abs(result), np.max(np.abs(result))) # Normalize around maximum
        result = 10*np.log10(result)
        self.labelAOA.setText(f"{int(thetas[np.argmax(result)])}")
        self.tracker_aoa_curve.setData(thetas, result)  
    
    def det_dist_plot(self):
        self.tracker.distance_res = np.roll(self.tracker.distance_res,-1,0)
        self.tracker.distance_res[-1:] = self.tracker.spectrum[1]
        self.tracker_dist_hist.setImage(self.tracker.distance_res)

    def det_period_time_update(self, update_period):
        logging.info(f"Signal Period: {update_period}")
    
    def det_sync_done(self):
        self.tracker.receiver.switch_noise_source(0)
        self.tracker.en_estimate_aoa = True

logging.basicConfig(filename="main.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG)

app = QApplication(argv)
win = MainWindow()
win.show()
app.exec()