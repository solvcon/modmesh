import matplotlib.pyplot as plt
import numpy as np


from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget
from .. import core
from .. import view as mh_view
from .. import apputil
from PUI.PySide6 import *
from PySide6 import Qt

class PlotArea(FigureCanvas):
    def __init__(self):
        self.figure = Figure()
        super().__init__(self.figure)
        self._plot()

    def _plot(self):
        #  plt.style.use('_mpl-gallery')
        # make data
        x = np.linspace(0, 10, 100)
        y = 4 + 2 * np.sin(2 * x)
        # plot
        ax = self.figure.subplots()
        # Need to set tight layout before plot the
        self.figure.tight_layout(w_pad=0.0)
        ax.plot(x, y, linewidth=2.0)
        ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        ylim=(0, 8), yticks=np.arange(1, 8))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("test plot")

        self.figure.canvas.draw()


class SolverConfig(PUIView):
    class TableAdapter:
        def __init__(self, data):
            self._tbl_content = data
            self._col_header = ["Variable", "Value", "Description"]

        def data(self, row, col):
            return self._tbl_content.value[row][col]

        def setData(self, row, col, value):
            self._tbl_content.value[row][col] = value
            self._tbl_content.emit()

        def columnHeader(self, col):
            return self._col_header[col]

        # Delete row header
        rowHeader = None

        def rowCount(self):
            return len(self._tbl_content.value)

        def columnCount(self):
            return len(self._tbl_content.value[0])

        def editable(self, row, col):
            return True

    def setup(self):
        self.state = State()
        self.state.data = [
                ["Gamma", 1.4, "The ratio of the specific heats."],
                ["Pressure left", 1.0, "The pressure of left hand side."],
                ["Density left", 1.0, "The density of left hand side."],
                ["Pressure right", 0.1, "The pressure of right hand side."],
                ["Density right", 0.125, "The density of right hand side."],
                ]

    def content(self):
        Table(self.TableAdapter(self.state("data")))

class Euler1DApp(PuiInQt):
    def content(self):
        with MenuBar():
            with Menu("File"):
                MenuAction("Save").trigger(self.save_file_cb)
        with Splitter():
            with VBox():
                with VBox().layout(weight=4):
                    Label("Solver")
                    with ComboBox():
                        ComboBoxItem("CESE")
                    Label("Configuration")
                    with Scroll():
                        SolverConfig()
                    Button("Set")
                    #  Spacer()
                with VBox().layout(weight=1):
                    Spacer()
                    Button("Start")
                    Button("Stop")
                    Button("Step")

            with VBox():
                QtInPui(PlotArea()).layout(width=700)


    def save_file_cb(self):
        print("Save file mockup")

def load_app():
    app = Euler1DApp(Window())
    app.redraw()
