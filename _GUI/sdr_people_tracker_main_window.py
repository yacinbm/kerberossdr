# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './SDR_People_Tracker.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1083, 738)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tabCal = QtWidgets.QWidget()
        self.tabCal.setObjectName("tabCal")
        self.formLayout = QtWidgets.QFormLayout(self.tabCal)
        self.formLayout.setObjectName("formLayout")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.lineStartFreq = QtWidgets.QLineEdit(self.tabCal)
        self.lineStartFreq.setObjectName("lineStartFreq")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineStartFreq)
        self.labelStopFreq = QtWidgets.QLabel(self.tabCal)
        self.labelStopFreq.setObjectName("labelStopFreq")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.labelStopFreq)
        self.lineStopFreq = QtWidgets.QLineEdit(self.tabCal)
        self.lineStopFreq.setObjectName("lineStopFreq")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineStopFreq)
        self.labelStartFreq = QtWidgets.QLabel(self.tabCal)
        self.labelStartFreq.setObjectName("labelStartFreq")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelStartFreq)
        self.labelDetectionThreshold = QtWidgets.QLabel(self.tabCal)
        self.labelDetectionThreshold.setObjectName("labelDetectionThreshold")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.labelDetectionThreshold)
        self.lineDetectThrhld = QtWidgets.QLineEdit(self.tabCal)
        self.lineDetectThrhld.setObjectName("lineDetectThrhld")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineDetectThrhld)
        self.verticalLayout_5.addLayout(self.formLayout_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        self.btnStartCal = QtWidgets.QPushButton(self.tabCal)
        self.btnStartCal.setAutoExclusive(False)
        self.btnStartCal.setObjectName("btnStartCal")
        self.verticalLayout_5.addWidget(self.btnStartCal)
        self.progBarCal = QtWidgets.QProgressBar(self.tabCal)
        self.progBarCal.setProperty("value", 0)
        self.progBarCal.setObjectName("progBarCal")
        self.verticalLayout_5.addWidget(self.progBarCal)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.verticalLayout_5)
        self.graphCal = QtWidgets.QGraphicsView(self.tabCal)
        self.graphCal.setObjectName("graphCal")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.graphCal)
        self.tabWidget.addTab(self.tabCal, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.formLayout_3 = QtWidgets.QFormLayout(self.tab_2)
        self.formLayout_3.setObjectName("formLayout_3")
        self.layoutGraphs = QtWidgets.QVBoxLayout()
        self.layoutGraphs.setObjectName("layoutGraphs")
        self.layoutMeasure = QtWidgets.QHBoxLayout()
        self.layoutMeasure.setObjectName("layoutMeasure")
        self.graphMeasure = QtWidgets.QGraphicsView(self.tab_2)
        self.graphMeasure.setObjectName("graphMeasure")
        self.layoutMeasure.addWidget(self.graphMeasure)
        self.layoutGraphs.addLayout(self.layoutMeasure)
        self.graphPosition = QtWidgets.QGraphicsView(self.tab_2)
        self.graphPosition.setObjectName("graphPosition")
        self.layoutGraphs.addWidget(self.graphPosition)
        self.formLayout_3.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.layoutGraphs)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setObjectName("label_3")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.spinCenterFreq = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.spinCenterFreq.setMinimum(24.0)
        self.spinCenterFreq.setMaximum(1750.0)
        self.spinCenterFreq.setProperty("value", 100.0)
        self.spinCenterFreq.setObjectName("spinCenterFreq")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinCenterFreq)
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setObjectName("label")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.spinFirTapSize = QtWidgets.QSpinBox(self.tab_2)
        self.spinFirTapSize.setMaximum(1000)
        self.spinFirTapSize.setObjectName("spinFirTapSize")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinFirTapSize)
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setObjectName("label_2")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.spinFilterBw = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.spinFilterBw.setMaximum(200.0)
        self.spinFilterBw.setProperty("value", 30.0)
        self.spinFilterBw.setObjectName("spinFilterBw")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.spinFilterBw)
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setObjectName("label_4")
        self.formLayout_4.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.spinInterAntDist = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.spinInterAntDist.setMaximum(100.0)
        self.spinInterAntDist.setObjectName("spinInterAntDist")
        self.formLayout_4.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.spinInterAntDist)
        self.labelAOA = QtWidgets.QLabel(self.tab_2)
        self.labelAOA.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelAOA.setObjectName("labelAOA")
        self.formLayout_4.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.labelAOA)
        self.btnStartTrack = QtWidgets.QPushButton(self.tab_2)
        self.btnStartTrack.setObjectName("btnStartTrack")
        self.formLayout_4.setWidget(6, QtWidgets.QFormLayout.SpanningRole, self.btnStartTrack)
        self.btnCapBkgrnd = QtWidgets.QPushButton(self.tab_2)
        self.btnCapBkgrnd.setObjectName("btnCapBkgrnd")
        self.formLayout_4.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.btnCapBkgrnd)
        self.formLayout_3.setLayout(0, QtWidgets.QFormLayout.LabelRole, self.formLayout_4)
        self.tabWidget.addTab(self.tab_2, "")
        self.horizontalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1083, 22))
        self.menubar.setAutoFillBackground(False)
        self.menubar.setObjectName("menubar")
        self.menu_File = QtWidgets.QMenu(self.menubar)
        self.menu_File.setObjectName("menu_File")
        self.menu_Save = QtWidgets.QMenu(self.menu_File)
        self.menu_Save.setObjectName("menu_Save")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionCalibration_Data = QtWidgets.QAction(MainWindow)
        self.actionCalibration_Data.setObjectName("actionCalibration_Data")
        self.menu_Save.addAction(self.actionCalibration_Data)
        self.menu_File.addAction(self.menu_Save.menuAction())
        self.menu_File.addSeparator()
        self.menubar.addAction(self.menu_File.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelStopFreq.setText(_translate("MainWindow", "Stop Freq (MHz)"))
        self.labelStartFreq.setText(_translate("MainWindow", "Start Freq (MHz)"))
        self.labelDetectionThreshold.setText(_translate("MainWindow", "Detection Threshold (dbM)"))
        self.btnStartCal.setText(_translate("MainWindow", "Start Calibration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabCal), _translate("MainWindow", "Calibration"))
        self.label_3.setText(_translate("MainWindow", "Center Frequency [MHz]"))
        self.label.setText(_translate("MainWindow", "FIR Tap Size"))
        self.label_2.setText(_translate("MainWindow", "Filter Bandwith [kHz]"))
        self.label_4.setText(_translate("MainWindow", "Inter Element Distance [cm]"))
        self.labelAOA.setText(_translate("MainWindow", "0"))
        self.btnStartTrack.setText(_translate("MainWindow", "Start Tracking"))
        self.btnCapBkgrnd.setText(_translate("MainWindow", "Capture Background"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Distance Estimation"))
        self.menu_File.setTitle(_translate("MainWindow", "&File"))
        self.menu_Save.setTitle(_translate("MainWindow", "&Save"))
        self.actionCalibration_Data.setText(_translate("MainWindow", "Calibration Data"))
