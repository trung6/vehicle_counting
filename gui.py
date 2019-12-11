# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/trung/Desktop/HHTQĐ/vehicle_counting/vehicle_counting.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Ui_MainWindow(Object):
    self.MainWindow

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(650, 454)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(20, 130, 113, 25))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("1")

        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(20, 190, 113, 25))# (top, left, width, heigt) of rectangle representing button
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setText("2")

        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(20, 60, 113, 25))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setText("10/12/2019")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 290, 111, 25))
        self.pushButton.setObjectName("pushButton")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 100, 111, 21))
        self.label.setObjectName("label")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 170, 111, 21))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 30, 111, 21))
        self.label_3.setObjectName("label_3")

        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(260, 10, 256, 51))
        self.textBrowser.setObjectName("textBrowser")
        
        # self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        # self.imageLbl.setGeometry(QtCore.QRect(180, 90, 451, 271))
        # self.imageLbl.setText("")
        # self.imageLbl.setObjectName("imageLbl")
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 650, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.show_chart_on_main_window)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Enter"))
        self.label.setText(_translate("MainWindow", "From time "))
        self.label_2.setText(_translate("MainWindow", "To time"))
        self.label_3.setText(_translate("MainWindow", "Date"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:100%; background-color:transparent;\"><span style=\" background-color:transparent;\">Report the number of vehicles among regions</span></p></body></html>"))

    def enter(self):
        """
        Args: 
        Returns:
        - cam_ids: a list of str representing names of cameras
        - nums_vehicles: a list of int representing the numbers of vehicles that each camera counted in 1 hour.
        """
        start_time = self.lineEdit.text()
        # end_time = self.lineEdit_2.text()
        date = self.lineEdit_3.text()

        cam_ids, nums_vehicles = Controller().get_data(date, int(start_time))
        
        # fileName = self.controller_obj.get_chart(value1, value2)
        # if fileName != None:
        #     pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
        #     pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        #     self.imageLbl.setPixmap(pixmap) # Set the pixmap onto the label
        #     self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center
        # else:
        #     print("There is no [{0}, {1}] in database.".format(value1, value2))
        
        return cam_ids, nums_vehicles

    def show_chart_on_main_window(self):
        canvas = Canvas(parent = self, width=8, height=4)
        canvas.move(0,0)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        title = "  "
        top = 400
        left = 400
        width = 900
        height = 500

        self.setWindowTitle(title)
        self.setGeometry(top, left, width, height)


class Canvas(FigureCanvas):
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        cam_ids, nums_vehicles = parent.enter()
        # print(cam_ids, nums_vehicles)
        self.plot(nums_vehicles, cam_ids)
        print("DONE")

    def plot(self, nums, labels):
        """
        To plot bar chart
        Args:
        - labels: a list of strings representing names of  columns in bar chart
        - nums: a list of int representing quatities w.r.t columns.
        """
        ax = self.figure.add_subplot(111)
        y_pos = np.arange(len(labels))

        ax.bar(y_pos, nums, align='center', alpha=0.5)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel('the number of vehicles')
        # plt.title('Programming language usage')



class Controller:
    def __init__(self):
        super(Controller, self).__init__()

    @staticmethod
    def get_data(date, start_time):
        """
        Args:
        - date : a string in format '%d/%m/%y' representing interested date.
        - start_time: a int scalar representing interested hour in date.
        """
        df = pd.read_csv('traffic_measurement.csv', delimiter=',', encoding="utf-8-sig")
        interest_df = df.loc[lambda df: (df.date == date) & \
            (df.start_time == start_time), ['cam_id', 'num_vehicles']]
        return interest_df['cam_id'].tolist(), interest_df['num_vehicles'].tolist()
 
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec_()
    # sys.exit(app.exec_())

