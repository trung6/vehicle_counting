from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np



class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        title = "Matplotlib Embeding In PyQt5"
        top = 400
        left = 400
        width = 900
        height = 500

        self.setWindowTitle(title)
        self.setGeometry(top, left, width, height)

        self.MyUI()


    def MyUI(self):

        canvas = Canvas(self, width=8, height=4)
        canvas.move(0,0)

        button = QPushButton("Click Me", self)
        button.move(100, 450)

        button2 = QPushButton("Click Me Two", self)
        button2.move(250, 450)


class Canvas(FigureCanvas):
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.plot()


    def plot(self, nums = [1,2,3,4], labels=[1,2,3,4]):
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




app = QApplication(sys.argv)
window = Window()
window.show()
app.exec()