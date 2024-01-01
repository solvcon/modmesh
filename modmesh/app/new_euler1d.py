import matplotlib
import matplotlib.pyplot
import numpy as np

from dataclasses import dataclass
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
#  from PySide6.QtWidgets import QWidget
from PUI.PySide6 import *
from ..onedim import euler1d
#  from PySide6 import Qt

@dataclass
class QuantityLine:
    ana: matplotlib.lines.Line2D = None
    num: matplotlib.lines.Line2D = None
    axis: matplotlib.pyplot.axis = None
    name: str = ""
    unit: str = ""

    def bind_axis(self, axis):
        self.axis = axis

    def update(self, adata, ndata):
        self.ana.set_ydata(adata)
        self.num.set_ydata(ndata)
        self.axis.relim()
        self.axis.autoscale_view()
        self.ana.figure.canvas.draw()
        self.num.figure.canvas.draw()


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

    def __call__(self, key):
        for ele in self.state("data").value:
            if key == ele[0]:
                return ele[1]
        return None

    def setup(self):
        self.state = State()
        self.state.data = [
                ["gamma", 1.4, "The ratio of the specific heats."],
                ["p_left", 1.0, "The pressure of left hand side."],
                ["rho_left", 1.0, "The density of left hand side."],
                ["p_right", 0.1, "The pressure of right hand side."],
                ["rho_right", 0.125, "The density of right hand side."],
                ["xmin", -10, "The most left point of x axis."],
                ["xmax", 10, "The most right point of x axis."],
                ["ncoord", 201, "Number of grid point."],
                ["time_increment", 0.05, "The density of right hand side."],
                ]

    def content(self):
        Table(self.TableAdapter(self.state("data")))


class Euler1DApp(PuiInQt):
    def _init_solver(self, gamma=1.4, pressure_left=1.0, density_left=1.0,
                     pressure_right=0.1, density_right=0.125, xmin=-10,
                     xmax=10, ncoord=201, time_increment=0.05):
        self.st = euler1d.ShockTube()
        self.st.build_constant(gamma, pressure_left, density_left,
                               pressure_right, density_right)
        self.st.build_numerical(xmin, xmax, ncoord, time_increment)
        self.st.build_field(t=0)

    def setup(self):
        self._init_solver(gamma=1.4, pressure_left=1.0, density_left=1.0,
                          pressure_right=0.1, density_right=0.125, xmin=-10,
                          xmax=10, ncoord=201, time_increment=0.05)

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
                        self.config = SolverConfig()
                    Button("Set").click(self.set_solver_config)
                with VBox().layout(weight=1):
                    Spacer()
                    Button("Start")
                    Button("Stop")
                    Button("Step")

            with VBox():
                QtInPui(PlotArea()).layout(width=700)

    def save_file_cb(self):
        print("Save file mockup")

    def set_solver_config(self):
        print("Set solver config")
        self._init_solver(gamma=self.config("gamma"),
                          pressure_left=self.config("p_left"),
                          density_left=self.config("rho_left"),
                          pressure_right=self.config("p_right"),
                          density_right=self.config("rho_right"),
                          xmin=self.config("xmin"),
                          xmax=self.config("xmax"),
                          ncoord=self.config("ncoord"),
                          time_increment=self.config("time_increment"))

def load_app():
    app = Euler1DApp(Window())
    app.redraw()
