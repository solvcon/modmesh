import sys
import matplotlib
import matplotlib.pyplot
import numpy as np
import modmesh as mm

from dataclasses import dataclass
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QDockWidget
from PUI.state import State
from PUI.PySide6.base import PuiInQt, QtInPui
from PUI.PySide6.button import Button
from PUI.PySide6.layout import VBox, Spacer
from PUI.PySide6.menu import Menu, MenuAction, MenuBar
from PUI.PySide6.scroll import Scroll
from PUI.PySide6.window import Window
from PUI.PySide6.combobox import ComboBox, ComboBoxItem
from PUI.PySide6.label import Label
from PUI.PySide6.table import Table
from ..onedim import euler1d
from .. import view


@dataclass
class QuantityLine:
    ana: matplotlib.lines.Line2D = None
    num: matplotlib.lines.Line2D = None
    axis: matplotlib.pyplot.axis = None
    name: str = ""
    unit: str = ""

    def update(self, xdata, adata, ndata):
        self.ana.set_data(xdata, adata)
        self.num.set_data(xdata, ndata)
        self.axis.relim()
        self.axis.autoscale_view()
        self.ana.figure.canvas.draw()
        self.num.figure.canvas.draw()


class SolverConfig():
    def __init__(self):
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
                ["timer_interval", 10, "Qt timer interval"],
                ["max_steps", 50, "Maximum step"],
                ["profiling", False, "Turn on / off solver profiling"],
                ]
        self._tbl_content = self.state("data")
        self._col_header = ["Variable", "Value", "Description"]

    def data(self, row, col):
        return self._tbl_content.value[row][col]

    def setData(self, row, col, value):
        self._tbl_content.value[row][col] = value
        self._tbl_content.emit()

    def columnHeader(self, col):
        return self._col_header[col]

    def editable(self, row, col):
        if col == 1:
            return True
        return False

    # Delete row header
    rowHeader = None

    def rowCount(self):
        return len(self._tbl_content.value)

    def columnCount(self):
        return len(self._tbl_content.value[0])

    def get_var(self, key):
        for ele in self.state("data").value:
            if key == ele[0]:
                return ele[1]
        return None


class Euler1DApp(PuiInQt):
    def __init__(self, parent):
        super().__init__(parent)
        self.config = SolverConfig()
        self.data_lines = {}
        self.density = QuantityLine(name="density",
                                    unit=r"$\mathrm{kg}/\mathrm{m}^3$")
        self.data_lines[self.density.name] = [self.density, True]
        self.velocity = QuantityLine(name="velocity",
                                     unit=r"$\mathrm{m}/\mathrm{s}$")
        self.data_lines[self.velocity.name] = [self.velocity, True]
        self.pressure = QuantityLine(name="pressure", unit=r"$\mathrm{Pa}$")
        self.data_lines[self.pressure.name] = [self.pressure, True]
        self.temperature = QuantityLine(name="temperature",
                                        unit=r"$\mathrm{K}$")
        self.data_lines[self.temperature.name] = [self.temperature, False]
        self.internal_energy = QuantityLine(name="internal_energy",
                                            unit=r"$\mathrm{J}/\mathrm{kg}$")
        self.data_lines[self.internal_energy.name] = [self.internal_energy,
                                                      False]
        self.entropy = QuantityLine(name="entropy",
                                    unit=r"$\mathrm{J}/\mathrm{K}$")
        self.data_lines[self.entropy.name] = [self.entropy, False]
        self.use_grid_layout = False
        self.checkbox_select_num = 3
        self.plot_holder = State()

    def init_solver(self, gamma=1.4, pressure_left=1.0, density_left=1.0,
                    pressure_right=0.1, density_right=0.125, xmin=-10,
                    xmax=10, ncoord=201, time_increment=0.05):
        self.st = euler1d.ShockTube()
        self.st.build_constant(gamma, pressure_left, density_left,
                               pressure_right, density_right)
        self.st.build_numerical(xmin, xmax, ncoord, time_increment)
        self.st.build_field(t=0)

    def set_solver_config(self):
        self.init_solver(gamma=self.config.get_var("gamma"),
                         pressure_left=self.config.get_var("p_left"),
                         density_left=self.config.get_var("rho_left"),
                         pressure_right=self.config.get_var("p_right"),
                         density_right=self.config.get_var("rho_right"),
                         xmin=self.config.get_var("xmin"),
                         xmax=self.config.get_var("xmax"),
                         ncoord=self.config.get_var("ncoord"),
                         time_increment=self.config.get_var("time_increment"))
        self.current_step = 0
        self.interval = self.config.get_var("timer_interval")
        self.max_steps = self.config.get_var("max_steps")
        self.profiling = self.config.get_var("profiling")

    def setup_timer(self):
        """
        :return: nothing
        """
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_timeout)

    def build_grid_figure(self):
        x = self.st.svr.coord[::2]
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots(3, 2)
        fig.tight_layout()

        for i, (data, color) in enumerate((
                (self.density, 'r'),
                (self.velocity, 'g'),
                (self.pressure, 'b'),
                (self.temperature, 'c'),
                (self.internal_energy, 'k'),
                (self.entropy, 'm')
        )):
            axis = ax[i // 2][i % 2]
            axis.autoscale(enable=True, axis='y', tight=False)
            data.axis = axis
            data.ana, = axis.plot(x, np.zeros_like(x),
                                  f'{color}-',
                                  label=f'{data.name}_ana')
            data.num, = axis.plot(x, np.zeros_like(x),
                                  f'{color}x',
                                  label=f'{data.name}_num')
            axis.set_ylabel(f'{data.name} ({data.unit})')

            axis.set_xlabel("distance (m)")
            axis.legend()
            axis.grid()

        self.update_lines()
        return canvas

    def build_single_figure(self):
        x = self.st.svr.coord[::2]
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots()
        fig.tight_layout()
        ax.autoscale(enable=True, axis='y', tight=False)

        # Matplotlib need to plot y axis on the left hand side first
        # then the reset of axis can be plotted on right hand side
        main_axis_plotted = False

        # Record how many lines had been selected to plot
        select_num = 0

        for data, color in (
            (self.density, 'r'),
            (self.velocity, 'g'),
            (self.pressure, 'b'),
            (self.temperature, 'c'),
            (self.internal_energy, 'k'),
            (self.entropy, 'm')
        ):
            # Plot multiple data line with same X axis, it need to plot a
            # data line on main axis first
            if self.data_lines[data.name][1]:
                if not main_axis_plotted:
                    data.axis = ax
                    data.ana, = ax.plot(x, np.zeros_like(x),
                                        f'{color}-',
                                        label=f'{data.name}_ana')
                    data.num, = ax.plot(x, np.zeros_like(x),
                                        f'{color}x',
                                        label=f'{data.name}_num')
                    ax.set_ylabel(f'{data.name} ({data.unit})')
                    main_axis_plotted = True
                else:
                    ax_new = ax.twinx()
                    data.axis = ax_new
                    data.ana, = ax_new.plot(x, np.zeros_like(x),
                                            f'{color}-',
                                            label=f'{data.name}_ana')
                    data.num, = ax_new.plot(x, np.zeros_like(x),
                                            f'{color}x',
                                            label=f'{data.name}_num')
                    ax_new.spines.right.set_position(("axes",
                                                      (1 + (select_num - 1)
                                                       * 0.2)))
                    ax_new.set_ylabel(f'{data.name} ({data.unit})')
                    ax_new.yaxis.set_major_formatter((FormatStrFormatter
                                                      ('%.2f')))
                select_num += 1

        ax.set_xlabel("distance (m)")
        ax.grid()

        # These parameters are the results obtained by my tuning on GUI
        fig.subplots_adjust(left=0.1,
                            right=0.97 - (self.checkbox_select_num - 1) * 0.1,
                            bottom=0.093, top=0.976,
                            wspace=0.2, hspace=0.2)

        self.update_lines()

        return canvas

    def march_alpha2(self, steps=1):
        if self.max_steps and self.current_step > self.max_steps:
            self.stop()
            return

        self.st.svr.march_alpha2(steps=steps)
        self.current_step += steps
        time_current = self.current_step * self.st.svr.time_increment
        self.st.build_field(t=time_current)
        cfl = self.st.svr.cfl
        self.log(f"CFL: min {cfl.min()} max {cfl.max()}")
        self.update_lines()
        if self.profiling:
            self.log(mm.time_registry.report())

    def step(self, steps=1):
        self.march_alpha2(steps=steps)

    def start(self):
        """
        This callback function don't care button's checked state,
        therefore the checked state is not used in this function.

        :param checked: button is checked or not
        :return: nothing
        """
        self.timer.start(self.interval)

    def set(self):
        self.set_solver_config()
        self.setup_timer()
        if self.use_grid_layout:
            self.plot_holder.plot = self.build_grid_figure()
        else:
            self.plot_holder.plot = self.build_single_figure()

    def stop(self):
        """
        The stop button callback for stopping Qt timer.
        :return: nothing
        """
        self.timer.stop()

    def single_layout(self):
        self.use_grid_layout = False
        self.plot_holder.plot = self.build_single_figure()

    def grid_layout(self):
        self.use_grid_layout = True
        self.plot_holder.plot = self.build_grid_figure()

    def save_file(self):
        """
        This callback function don't care button's checked state,
        therefore the checked state is not used in this function.

        :param checked: button is checked or not
        :return: nothing
        """
        fig = QPixmap(self.plot_holder.plot.size())
        self.plot_holder.plot.render(fig)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(None, "Save file", "",
                                                  "All Files (*)",
                                                  options=options)

        if fileName != "":
            fig.save(fileName, "JPG", 100)

    @Slot()
    def timer_timeout(self):
        self.step()

    @staticmethod
    def log(msg):
        sys.stdout.write(msg)
        sys.stdout.write('\n')
        view.mgr.pycon.writeToHistory(msg)
        view.mgr.pycon.writeToHistory('\n')

    def update_lines(self):
        if self.use_grid_layout:
            self.density.update(xdata=self.st.svr.coord[::2],
                                adata=self.st.density_field,
                                ndata=self.st.svr.density[::2])
            self.pressure.update(xdata=self.st.svr.coord[::2],
                                 adata=self.st.pressure_field,
                                 ndata=self.st.svr.pressure[::2])
            self.velocity.update(xdata=self.st.svr.coord[::2],
                                 adata=self.st.velocity_field,
                                 ndata=self.st.svr.velocity[::2])
            self.temperature.update(xdata=self.st.svr.coord[::2],
                                    adata=self.st.temperature_field,
                                    ndata=self.st.svr.temperature[::2])
            self.internal_energy.update(xdata=self.st.svr.coord[::2],
                                        adata=(self.st.internal_energy_field),
                                        ndata=(self.st.svr.
                                               internal_energy[::2]))
            self.entropy.update(xdata=self.st.svr.coord[::2],
                                adata=self.st.entropy_field,
                                ndata=self.st.svr.entropy[::2])
        else:
            for name, data_line in self.data_lines.items():
                if data_line[1]:
                    eval(f'(data_line[0].update(xdata=self.st.svr.coord[::2]'
                         f', adata=self.st.{name}_field, ndata=self.st.svr.'
                         f'{name}[::2]))')

    def setup(self):
        self.set_solver_config()
        self.setup_timer()
        self.plot_holder.plot = self.build_single_figure()

    def content(self):
        with MenuBar():
            with Menu("File"):
                MenuAction("Save").trigger(self.save_file)
            with Menu("Layout"):
                MenuAction("Single").trigger(self.single_layout)
                MenuAction("Grid").trigger(self.grid_layout)
        with VBox():
            QtInPui(self.plot_holder.plot)


class ConfigWindow(PuiInQt):
    def __init__(self, parent, controlled):
        super().__init__(parent)
        self.controlled = controlled

    def setup(self):
        self.config = self.controlled.config

    def content(self):
        with VBox():
            with VBox().layout(weight=4):
                Label("Solver")
                with ComboBox():
                    ComboBoxItem("Euler1D-CESE")
                Label("Configuration")
                with Scroll():
                    Table(self.config)
                Button("Set").click(self.controlled.set)
            with VBox().layout(weight=1):
                Spacer()
                Button("Start").click(self.controlled.start)
                Button("Stop").click(self.controlled.stop)
                Button("Step").click(self.controlled.step)


def load_app():
    app = Euler1DApp(Window())
    config_window = ConfigWindow(Window(), app)

    config_widget = QDockWidget("config")
    config_widget.setWidget(config_window.ui.ui)
    view.mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea, config_widget)

    config_window.redraw()

    _subwin = view.mgr.addSubWindow(app.ui.ui)
    _subwin.showMaximized()
    _subwin.show()

    app.redraw()
