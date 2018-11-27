from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

class DispThread(QtCore.QThread):
    def __init__(self, parent, t, *args, **kwargs):
        QtCore.QThread.__init__(self, parent)

        self.t = t
        self.H = None
        self.H_RAW = None
        self.hr_est = None

        self.win_sig = Signal("Filtered pulse signal")
        self.win_sig.show()
        self.win_sig.resize(1200, 300)
        self.win_sig.raise_()

        self.win_sig2 = Signal("RAW pulse signal")
        self.win_sig2.show()
        self.win_sig2.resize(1200, 300)
        self.win_sig2.raise_()

        self.win_HR = HeartRateLCD()
        self.win_HR.show()

        self.plot_data = list()

    def start_plotting_thread(self, H, H_RAW, hr_est, on_finish=None):
        """ Start plotting """
        self.H = H
        self.H_RAW = H_RAW
        self.hr_est = hr_est

        self.start()

    def run(self):
        """ Run as a thread """
        self.win_HR.disp(self.hr_est)
        self.win_sig.setData(self.t, self.H)
        self.win_sig2.setData(self.t, self.H_RAW)


class Signal(pg.GraphicsWindow):

    def __init__(self, titel, parent=None):
        pg.GraphicsWindow.__init__(self, parent=parent)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.plotItem = self.addPlot(titel=titel)

        self.plotDataItem = self.plotItem.plot([], pen=1, symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)

    def setData(self, x, y):
        self.plotDataItem.setData(x, y)


class HeartRateLCD(QtGui.QMainWindow):
 
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.initUI()
 
    def initUI(self):

        self.lcd = QtGui.QLCDNumber(self)
        self.lcd.display(0)
 
        self.setCentralWidget(self.lcd)
 
#---------Window settings --------------------------------
         
        self.setGeometry(300, 300, 250, 100)
        self.setWindowTitle("Heart Rate")
 
#-------- Slots ------------------------------------------
 
    def disp(self, HR):
        self.lcd.display(HR)
         